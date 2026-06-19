// Native C++ core of the DATAHACKS2026 backtester (Phase 2 accelerator).
//
// Reproduces the pure-Python engine (engine.py + execution.py + portfolio.py + market_manager.py)
// and the WorldModelStrategy / NaiveLagStrategy decision logic from pnl_backtest.py, callback-free:
// Python marshals the scoped timeline into a sparse CSR over the present (tick, market) cells once,
// C++ runs the whole tick loop, and the result is handed back. The pure-Python engine stays the
// canonical reference; this path is opt-in and parity-checked.
//
// Sparse contract: the timeline is dense per *active* market (every active tick carries a forward-
// filled book), but markets are short-lived, so the dense (T, M) grid is ~99% empty and its ladder
// dimension (depth up to hundreds) is infeasible at headline scale. Instead each tick's present cells
// are listed contiguously (CSR via tick_offset) and each cell refreshes that market's "current" book;
// the per-market loops read that current state. Ask ladders are truncated at cumulative MAX_SHARES_PER
// _TOKEN, which is exact because each strategy issues at most one buy per market per tick, so no order
// (and no same-tick depletion) ever walks beyond that depth.
//
// Strategy coverage (mirrors pnl_backtest exactly): fixed or quarter-Kelly sizing, optional CUSUM
// gate, the tte / 500-share position gates, the predictability gate in both its live-spot-window form
// and its per-market precomputed-ER form, the two-sided book-depth band, settlement-probability
// trading with optional temperature calibration, and edge mode (trade from predicted book-relative
// EV). Per-run signals (gate ER, gate direction, edge EV) arrive as cell-aligned arrays; book depth is
// timeline-only and baked into the marshalling. The run/strategy configuration is one py::dict.
// region includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <vector>
// endregion
namespace py = pybind11;

// Token encoding shared with the Python wrapper: 0 = YES, 1 = NO.
constexpr int TOKEN_YES = 0;
constexpr int TOKEN_NO = 1;
constexpr double MAX_SHARES_PER_TOKEN = 500.0;
constexpr int MAX_BOOK_STALENESS_S = 5;
// Shares below this count as a non-fill, matching the Python walk-the-book's 1e-9 guard.
constexpr double FILL_EPSILON = 1e-9;
// Sizing kinds shared with the wrapper (params["sizing"]).
constexpr int SIZING_FIXED = 0;
constexpr int SIZING_KELLY = 1;
// Predictability-gate modes: 0 = live spot-window efficiency ratio, 1 = per-market precomputed ER.
constexpr int GATE_LIVE_WINDOW = 0;
constexpr int GATE_PER_MARKET = 1;


struct Position {
    double yes_shares = 0.0;
    double no_shares = 0.0;
    double cost_basis = 0.0;
};


struct PendingOrder {
    int market = 0;
    int token = 0;
    double size = 0.0;
    long long execute_tick = 0;
};


struct EngineFill {
    int market = 0;
    int token = 0;
    double size = 0.0;
    double avg_price = 0.0;
};

// CusumGate (Lopez de Prado event sampling): accumulate signed returns, fire +1/-1 on a threshold
// crossing and reset that arm. Mirrors pnl_backtest.CusumGate exactly.
struct CusumGate {
    double threshold = 0.0;
    double pos = 0.0;
    double neg = 0.0;
    int update(double spot_return) {
        pos = std::max(0.0, pos + spot_return);
        neg = std::min(0.0, neg + spot_return);
        if (pos >= threshold) { pos = 0.0; return 1; }
        if (neg <= -threshold) { neg = 0.0; return -1; }
        return 0;
    }
};

// Quarter-Kelly share size for one binary-contract trade, mirroring pnl_backtest.kelly_shares: the
// edge-over-breakeven fraction, a non-positive edge sizing to zero, the bet capped at cap_fraction.
double kelly_shares(double model_prob, double book_prob, int token, double bankroll, double price,
                    double kelly_fraction, double cap_fraction) {
    const double edge_fraction = (token == TOKEN_YES)
        ? (model_prob - book_prob) / std::max(1.0 - book_prob, 1e-6)
        : (book_prob - model_prob) / std::max(book_prob, 1e-6);
    if (edge_fraction <= 0.0) {
        return 0.0;
    }
    const double bet_fraction = std::min(kelly_fraction * edge_fraction, cap_fraction);
    return (bet_fraction * bankroll) / std::max(price, 1e-6);
}

// Temperature-scaled settlement probability, mirroring the WorldModelStrategy calibration branch:
// sigmoid(logit(clip(p)) / T). T > 1 softens an over-confident head toward 0.5.
double calibrate_probability(double model_prob, double temperature) {
    const double clipped = std::min(std::max(model_prob, 1e-6), 1.0 - 1e-6);
    const double logit = std::log(clipped / (1.0 - clipped));
    return 1.0 / (1.0 + std::exp(-logit / temperature));
}

// Kaufman efficiency ratio over the live spot window: |net move| / total path length, in [0, 1].
// Mirrors predictability.efficiency_ratio (0 when fewer than two points or a degenerate path).
double window_efficiency_ratio(const std::deque<double>& window) {
    if (window.size() < 2) {
        return 0.0;
    }
    double total_path = 0.0;
    for (std::size_t i = 1; i < window.size(); ++i) {
        total_path += std::abs(window[i] - window[i - 1]);
    }
    if (total_path <= 0.0) {
        return 0.0;
    }
    return std::abs(window.back() - window.front()) / total_path;
}


py::dict run_backtest(
    // Per-tick arrays, shape (T,).
    py::array_t<long long> ts,
    py::array_t<double> chainlink_btc,
    // Per-market arrays, shape (M,).
    py::array_t<long long> market_start,
    py::array_t<long long> market_end,
    py::array_t<int> market_outcome,  // 1 = YES, 0 = NO, -1 = unsettled.
    // CSR over the present (tick, market) cells: tick t owns cells [tick_offset(t), tick_offset(t+1)).
    py::array_t<long long> tick_offset,  // shape (T + 1,).
    py::array_t<int> cell_market,        // shape (P,); market index of each cell.
    py::array_t<double> cell_prob,       // shape (P,); NaN where no probability exists for this cell.
    py::array_t<long long> cell_book_ts, // shape (P,); the forward-filled book's source ts (for staleness).
    py::array_t<double> cell_yes_mid,    // shape (P,); YES book implied probability.
    py::array_t<double> cell_yes_price,  // shape (P,); mark-to-market YES price.
    py::array_t<double> cell_no_price,   // shape (P,); mark-to-market NO price.
    // Ragged YES/NO ask ladders for walk-the-book, truncated at cumulative MAX_SHARES_PER_TOKEN.
    py::array_t<long long> cell_yes_off, // shape (P + 1,); cell c's YES levels are [off(c), off(c+1)).
    py::array_t<double> yes_ask_px,
    py::array_t<double> yes_ask_sz,
    py::array_t<long long> cell_no_off,  // shape (P + 1,); cell c's NO levels are [off(c), off(c+1)).
    py::array_t<double> no_ask_px,
    py::array_t<double> no_ask_sz,
    py::array_t<double> cell_yes_depth,  // shape (P,) when the depth band is active, else (0,).
    // Per-run cell-aligned signals; each is shape (P,) when its gate is active, else (0,).
    py::array_t<double> cell_gate_er,    // per-market predictability ER.
    py::array_t<double> cell_gate_dir,   // per-market naive trend direction (sign).
    py::array_t<double> cell_ev_yes,     // edge-mode YES EV (NaN where absent).
    py::array_t<double> cell_ev_no,      // edge-mode NO EV (NaN where absent).
    py::dict config
) {
    const auto ts_ = ts.unchecked<1>();
    const auto cl_ = chainlink_btc.unchecked<1>();
    const auto m_start_ = market_start.unchecked<1>();
    const auto m_end_ = market_end.unchecked<1>();
    const auto m_out_ = market_outcome.unchecked<1>();
    const auto toff_ = tick_offset.unchecked<1>();
    const auto c_mkt_ = cell_market.unchecked<1>();
    const auto c_prob_ = cell_prob.unchecked<1>();
    const auto c_bts_ = cell_book_ts.unchecked<1>();
    const auto c_ymid_ = cell_yes_mid.unchecked<1>();
    const auto c_yprice_ = cell_yes_price.unchecked<1>();
    const auto c_nprice_ = cell_no_price.unchecked<1>();
    const auto yoff_ = cell_yes_off.unchecked<1>();
    const auto ypx_ = yes_ask_px.unchecked<1>();
    const auto ysz_ = yes_ask_sz.unchecked<1>();
    const auto noff_ = cell_no_off.unchecked<1>();
    const auto npx_ = no_ask_px.unchecked<1>();
    const auto nsz_ = no_ask_sz.unchecked<1>();
    const auto depth_ = cell_yes_depth.unchecked<1>();
    const auto ger_ = cell_gate_er.unchecked<1>();
    const auto gdir_ = cell_gate_dir.unchecked<1>();
    const auto evy_ = cell_ev_yes.unchecked<1>();
    const auto evn_ = cell_ev_no.unchecked<1>();
    // Run / strategy configuration.
    const int strategy_kind = config["strategy_kind"].cast<int>();
    const double edge_threshold = config["edge_threshold"].cast<double>();
    const double order_size = config["order_size"].cast<double>();
    const double max_tte_frac = config["max_tte_frac"].cast<double>();
    const double min_tte_frac = config["min_tte_frac"].cast<double>();
    const bool use_cusum = config["use_cusum"].cast<bool>();
    const double cusum_threshold = config["cusum_threshold"].cast<double>();
    const double starting_cash = config["starting_cash"].cast<double>();
    const long long snapshot_interval = config["snapshot_interval"].cast<long long>();
    const int sizing_kind = config["sizing_kind"].cast<int>();
    const double kelly_fraction = config["kelly_fraction"].cast<double>();
    const double kelly_cap = config["kelly_cap"].cast<double>();
    const double bankroll = config["bankroll"].cast<double>();
    const double calibration_temperature = config["calibration_temperature"].cast<double>();
    const bool use_predictability_gate = config["use_predictability_gate"].cast<bool>();
    const int gate_mode = config["gate_mode"].cast<int>();
    const double predictability_threshold = config["predictability_threshold"].cast<double>();
    const int predictability_window = config["predictability_window"].cast<int>();
    const double min_book_depth = config["min_book_depth"].cast<double>();
    const double max_book_depth = config["max_book_depth"].cast<double>();
    const bool use_edge = config["use_edge"].cast<bool>();
    // Which cell-aligned signals are live (and therefore safe to index).
    const bool use_depth = (min_book_depth > 0.0 || max_book_depth > 0.0);
    const bool use_per_market_er = (use_predictability_gate && gate_mode == GATE_PER_MARKET);
    const bool use_gate_dir = (use_per_market_er && strategy_kind != 0);
    const bool use_live_window = (use_predictability_gate && gate_mode == GATE_LIVE_WINDOW);
    const long long total_ticks = ts_.shape(0);
    const int num_markets = static_cast<int>(m_start_.shape(0));
    // Engine state.
    double cash = starting_cash;
    std::vector<Position> positions(num_markets);
    std::vector<int> status(num_markets, 0);  // 0 = upcoming, 1 = active, 2 = settled.
    std::vector<PendingOrder> pending;
    // Per-market current (forward-filled) state, refreshed from each tick's CSR cells. Once a market's
    // first cell arrives cur_present stays 1 for its active life (the timeline forward-fills a cell every
    // active tick); after expiry the status guard, not present, excludes it — matching the dense engine.
    std::vector<signed char> cur_present(num_markets, 0);
    std::vector<long long> cur_book_ts(num_markets, 0);
    std::vector<double> cur_yes_mid(num_markets, 0.0);
    std::vector<double> cur_yes_price(num_markets, 0.0);
    std::vector<double> cur_no_price(num_markets, 0.0);
    std::vector<double> cur_prob(num_markets, std::nan(""));
    std::vector<long long> cur_yes_off(num_markets, 0);
    std::vector<long long> cur_yes_end(num_markets, 0);
    std::vector<long long> cur_no_off(num_markets, 0);
    std::vector<long long> cur_no_end(num_markets, 0);
    std::vector<double> cur_depth(num_markets, 0.0);
    std::vector<double> cur_gate_er(num_markets, 0.0);
    std::vector<double> cur_gate_dir(num_markets, 1.0);
    std::vector<double> cur_ev_yes(num_markets, std::nan(""));
    std::vector<double> cur_ev_no(num_markets, std::nan(""));
    int total_trades = 0;
    std::vector<EngineFill> all_fills;
    // Markets that settle within the timeline, matching the Python engine's recorded all_settlements:
    // per_trade_pnls scores only fills whose market actually settled, so an unsettled market's fills
    // (still active at the last tick) must be excluded from the realized per-trade list.
    std::vector<int> settled_markets;
    std::vector<double> snapshot_values;
    CusumGate cusum{cusum_threshold, 0.0, 0.0};
    std::deque<double> spot_window;
    bool have_prev_spot = false;
    double prev_spot = 0.0;
    for (long long t = 0; t < total_ticks; ++t) {
        const long long now = ts_(t);
        // 0. Refresh each present market's current book / signals from this tick's CSR cells.
        for (long long c = toff_(t); c < toff_(t + 1); ++c) {
            const int m = c_mkt_(c);
            cur_present[m] = 1;
            cur_book_ts[m] = c_bts_(c);
            cur_yes_mid[m] = c_ymid_(c);
            cur_yes_price[m] = c_yprice_(c);
            cur_no_price[m] = c_nprice_(c);
            cur_prob[m] = c_prob_(c);
            cur_yes_off[m] = yoff_(c);
            cur_yes_end[m] = yoff_(c + 1);
            cur_no_off[m] = noff_(c);
            cur_no_end[m] = noff_(c + 1);
            if (use_depth) {
                cur_depth[m] = depth_(c);
            }
            if (use_per_market_er) {
                cur_gate_er[m] = ger_(c);
            }
            if (use_gate_dir) {
                cur_gate_dir[m] = gdir_(c);
            }
            if (use_edge) {
                cur_ev_yes[m] = evy_(c);
                cur_ev_no[m] = evn_(c);
            }
        }
        // 1. Advance market lifecycle and settle anything that expires this tick (market_manager.update
        //    + portfolio.apply_settlement): UPCOMING -> ACTIVE before expiry, -> SETTLED at/after it.
        for (int m = 0; m < num_markets; ++m) {
            bool settle_now = false;
            if (status[m] == 0 && now >= m_start_(m)) {
                if (now < m_end_(m)) {
                    status[m] = 1;
                } else {
                    status[m] = 2;
                    settle_now = true;
                }
            }
            if (status[m] == 1 && now >= m_end_(m)) {
                status[m] = 2;
                settle_now = true;
            }
            if (settle_now && m_out_(m) >= 0) {
                Position& position = positions[m];
                const double payout = (m_out_(m) == 1) ? position.yes_shares : position.no_shares;
                cash += payout;
                position.yes_shares = 0.0;
                position.no_shares = 0.0;
                position.cost_basis = 0.0;
                settled_markets.push_back(m);
            }
        }
        // 2. Execute pending orders due at or before this tick (T+1 latency), walking the current book.
        std::unordered_map<int, std::vector<double>> consumed_by_key;
        std::vector<PendingOrder> still_pending;
        still_pending.reserve(pending.size());
        for (const PendingOrder& order : pending) {
            if (order.execute_tick > now) {
                still_pending.push_back(order);
                continue;
            }
            const int m = order.market;
            // Reject when the market is no longer active or the book is stale, matching execution.py.
            if (status[m] != 1 || cur_present[m] == 0) {
                continue;
            }
            if (now - cur_book_ts[m] > MAX_BOOK_STALENESS_S) {
                continue;
            }
            const long long level_begin = (order.token == TOKEN_YES) ? cur_yes_off[m] : cur_no_off[m];
            const long long level_end = (order.token == TOKEN_YES) ? cur_yes_end[m] : cur_no_end[m];
            const int n_levels = static_cast<int>(level_end - level_begin);
            if (n_levels <= 0) {
                continue;
            }
            const int key = m * 2 + order.token;
            std::vector<double>& consumed = consumed_by_key[key];
            if (static_cast<int>(consumed.size()) < n_levels) {
                consumed.resize(n_levels, 0.0);
            }
            double remaining = order.size;
            double total_cost = 0.0;
            double total_filled = 0.0;
            for (int level = 0; level < n_levels; ++level) {
                const long long index = level_begin + level;
                const double price = (order.token == TOKEN_YES) ? ypx_(index) : npx_(index);
                const double level_size = (order.token == TOKEN_YES) ? ysz_(index) : nsz_(index);
                const double available = std::max(0.0, level_size - consumed[level]);
                if (available <= 0.0) {
                    continue;
                }
                const double fill_at_level = std::min(remaining, available);
                total_cost += fill_at_level * price;
                total_filled += fill_at_level;
                remaining -= fill_at_level;
                consumed[level] += fill_at_level;
                if (remaining <= FILL_EPSILON) {
                    break;
                }
            }
            if (total_filled <= FILL_EPSILON) {
                continue;
            }
            const double avg_price = total_cost / total_filled;
            // Apply the fill to the portfolio (buys only; the strategies never sell).
            Position& position = positions[m];
            cash -= total_cost;
            if (order.token == TOKEN_YES) {
                position.yes_shares += total_filled;
            } else {
                position.no_shares += total_filled;
            }
            position.cost_basis += total_cost;
            total_trades += 1;
            all_fills.push_back(EngineFill{m, order.token, total_filled, avg_price});
        }
        pending.swap(still_pending);
        // 3. Mark to market over the active markets, matching portfolio.mark_to_market.
        double total_value = cash;
        for (int m = 0; m < num_markets; ++m) {
            const Position& position = positions[m];
            if (position.yes_shares <= 0.0 && position.no_shares <= 0.0) {
                continue;
            }
            if (status[m] == 1) {
                total_value += position.yes_shares * cur_yes_price[m] + position.no_shares * cur_no_price[m];
            } else {
                total_value += position.cost_basis;
            }
        }
        // 4. Strategy decision for this tick (callback-free; mirrors WorldModelStrategy / NaiveLagStrategy).
        const double spot = cl_(t);
        int cusum_event = 0;
        if (have_prev_spot && prev_spot > 0.0 && spot > 0.0) {
            cusum_event = cusum.update((spot - prev_spot) / prev_spot);
        }
        have_prev_spot = true;
        prev_spot = spot;
        // The live spot window is updated every tick (even on a CUSUM-blocked tick), matching Python.
        if (use_live_window && spot > 0.0) {
            spot_window.push_back(spot);
            if (static_cast<int>(spot_window.size()) > predictability_window) {
                spot_window.pop_front();
            }
        }
        bool blocked = false;
        int live_direction = 0;
        if (strategy_kind == 0) {
            if (use_cusum && cusum_event == 0) {
                blocked = true;
            }
            if (!blocked && use_live_window && spot_window.size() >= 2 &&
                window_efficiency_ratio(spot_window) < predictability_threshold) {
                blocked = true;
            }
        } else if (use_live_window) {
            if (spot_window.size() < 2 || window_efficiency_ratio(spot_window) < predictability_threshold) {
                blocked = true;
            } else {
                live_direction = (spot_window.back() >= spot_window.front()) ? 1 : -1;
            }
        } else if (!use_predictability_gate) {
            if (cusum_event == 0) {
                blocked = true;
            } else {
                live_direction = cusum_event;
            }
        }
        std::vector<PendingOrder> new_orders;
        if (!blocked) {
            for (int m = 0; m < num_markets; ++m) {
                if (status[m] != 1 || cur_present[m] == 0) {
                    continue;
                }
                if (use_per_market_er && cur_gate_er[m] < predictability_threshold) {
                    continue;
                }
                if (use_depth) {
                    if (cur_depth[m] < min_book_depth) {
                        continue;
                    }
                    if (max_book_depth > 0.0 && cur_depth[m] > max_book_depth) {
                        continue;
                    }
                }
                const double duration = static_cast<double>(m_end_(m) - m_start_(m));
                const double time_remaining_s = std::max(0.0, static_cast<double>(m_end_(m) - now));
                const double time_remaining_frac = (duration > 0.0) ? time_remaining_s / duration : 0.0;
                const double book_prob = cur_yes_mid[m];
                // best_ask = first (cheapest) ask level, or 0 when the truncated ask ladder is empty.
                const double yes_best_ask = (cur_yes_end[m] > cur_yes_off[m]) ? ypx_(cur_yes_off[m]) : 0.0;
                const double no_best_ask = (cur_no_end[m] > cur_no_off[m]) ? npx_(cur_no_off[m]) : 0.0;
                Position& position = positions[m];
                if (strategy_kind == 0) {
                    if (time_remaining_frac > max_tte_frac || time_remaining_frac < min_tte_frac) {
                        continue;
                    }
                    if (use_edge) {
                        const double ev_yes = cur_ev_yes[m];
                        const double ev_no = cur_ev_no[m];
                        if (std::isnan(ev_yes) || std::isnan(ev_no)) {
                            continue;
                        }
                        if (ev_yes >= edge_threshold && ev_yes > ev_no && yes_best_ask > 0.0) {
                            const double size = std::min(order_size, MAX_SHARES_PER_TOKEN - position.yes_shares);
                            if (size > 0.0) {
                                new_orders.push_back(PendingOrder{m, TOKEN_YES, size, now + 1});
                            }
                        } else if (ev_no >= edge_threshold && ev_no > ev_yes && no_best_ask > 0.0) {
                            const double size = std::min(order_size, MAX_SHARES_PER_TOKEN - position.no_shares);
                            if (size > 0.0) {
                                new_orders.push_back(PendingOrder{m, TOKEN_NO, size, now + 1});
                            }
                        }
                        continue;
                    }
                    double prob = cur_prob[m];
                    if (std::isnan(prob)) {
                        continue;
                    }
                    if (calibration_temperature != 1.0) {
                        prob = calibrate_probability(prob, calibration_temperature);
                    }
                    if (book_prob <= 0.0 || book_prob >= 1.0) {
                        continue;
                    }
                    const double edge = prob - book_prob;
                    if (std::abs(edge) < edge_threshold) {
                        continue;
                    }
                    if (edge > 0.0 && yes_best_ask > 0.0) {
                        const double raw = (sizing_kind == SIZING_KELLY)
                            ? kelly_shares(prob, book_prob, TOKEN_YES, bankroll, yes_best_ask, kelly_fraction, kelly_cap)
                            : order_size;
                        const double size = std::min(raw, MAX_SHARES_PER_TOKEN - position.yes_shares);
                        if (size > 0.0) {
                            new_orders.push_back(PendingOrder{m, TOKEN_YES, size, now + 1});
                        }
                    } else if (edge < 0.0 && no_best_ask > 0.0) {
                        const double raw = (sizing_kind == SIZING_KELLY)
                            ? kelly_shares(prob, book_prob, TOKEN_NO, bankroll, no_best_ask, kelly_fraction, kelly_cap)
                            : order_size;
                        const double size = std::min(raw, MAX_SHARES_PER_TOKEN - position.no_shares);
                        if (size > 0.0) {
                            new_orders.push_back(PendingOrder{m, TOKEN_NO, size, now + 1});
                        }
                    }
                } else {
                    if (time_remaining_frac > max_tte_frac) {
                        continue;
                    }
                    if (book_prob <= 0.0 || book_prob >= 1.0) {
                        continue;
                    }
                    const int direction = use_per_market_er ? ((cur_gate_dir[m] >= 0.0) ? 1 : -1) : live_direction;
                    if (direction > 0 && book_prob < 0.5 && yes_best_ask > 0.0 &&
                        position.yes_shares + order_size <= MAX_SHARES_PER_TOKEN) {
                        new_orders.push_back(PendingOrder{m, TOKEN_YES, order_size, now + 1});
                    } else if (direction < 0 && book_prob > 0.5 && no_best_ask > 0.0 &&
                        position.no_shares + order_size <= MAX_SHARES_PER_TOKEN) {
                        new_orders.push_back(PendingOrder{m, TOKEN_NO, order_size, now + 1});
                    }
                }
            }
        }
        // 5. Validate + queue the new orders for T+1, pre-reserving cash exactly as execution.queue_orders.
        double available_cash = cash;
        for (const PendingOrder& order : new_orders) {
            if (status[order.market] != 1 || order.size <= 0.0) {
                continue;
            }
            Position& position = positions[order.market];
            const double held = (order.token == TOKEN_YES) ? position.yes_shares : position.no_shares;
            if (held + order.size > MAX_SHARES_PER_TOKEN) {
                continue;
            }
            const double cost = order.size * 1.0;  // Market order: reserve at the 1.0 conservative price.
            if (cost > available_cash) {
                continue;
            }
            available_cash -= cost;
            pending.push_back(order);
        }
        // 6. Record the portfolio snapshot on the configured cadence.
        if (snapshot_interval > 0 && now % snapshot_interval == 0) {
            snapshot_values.push_back(total_value);
        }
    }
    // Final mark-to-market over the last tick's current books, matching the Python engine's final value.
    double final_value = cash;
    for (int m = 0; m < num_markets; ++m) {
        const Position& position = positions[m];
        if (position.yes_shares <= 0.0 && position.no_shares <= 0.0) {
            continue;
        }
        if (status[m] == 1) {
            final_value += position.yes_shares * cur_yes_price[m] + position.no_shares * cur_no_price[m];
        } else {
            final_value += position.cost_basis;
        }
    }
    py::list fills_out;
    for (const EngineFill& fill : all_fills) {
        fills_out.append(py::make_tuple(fill.market, fill.token, fill.size, fill.avg_price));
    }
    py::dict result;
    result["total_pnl"] = final_value - starting_cash;
    result["total_trades"] = total_trades;
    result["fills"] = fills_out;
    result["settled_markets"] = py::cast(settled_markets);
    result["snapshot_values"] = py::cast(snapshot_values);
    return result;
}


PYBIND11_MODULE(_engine, module) {
    module.doc() = "Native C++ core of the DATAHACKS2026 backtester (parity with the Python engine).";
    module.def("run_backtest", &run_backtest, "Run one backtest over a sparse CSR scoped timeline.");
}
