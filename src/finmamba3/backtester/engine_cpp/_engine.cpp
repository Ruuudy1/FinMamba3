// Native C++ core of the DATAHACKS2026 backtester (Phase 2 accelerator).
//
// Reproduces the pure-Python engine (engine.py + execution.py + portfolio.py + market_manager.py)
// and the settlement-mode WorldModelStrategy / NaiveLagStrategy decision logic from pnl_backtest.py,
// callback-free: Python marshals the scoped timeline + precomputed per-(market, tick) model
// probabilities into dense arrays once, C++ runs the whole tick loop, and the result is handed back.
// The pure-Python engine stays the canonical reference; this path is opt-in and parity-checked.
//
// v1 scope (the default run_pnl_backtest path): fixed-size sizing, optional CUSUM gate, the tte and
// 500-share position gates, walk-the-book execution with depletion + T+1 latency + 5s staleness, and
// settlement payouts. The Python wrapper fail-fast rejects any option not yet ported here (Kelly
// sizing, calibration temperature, the predictability gate, the book-depth band, edge mode) so the
// native path can never silently diverge from the reference.
// region includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <unordered_map>
#include <vector>
// endregion
namespace py = pybind11;

// Token encoding shared with the Python wrapper: 0 = YES, 1 = NO.
constexpr int TOKEN_YES = 0;
constexpr int TOKEN_NO = 1;
constexpr double MAX_SHARES_PER_TOKEN = 500.0;
constexpr int MAX_BOOK_STALENESS_S = 5;


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


py::dict run_backtest(
    // Per-tick arrays, shape (T,).
    py::array_t<long long> ts,
    py::array_t<double> chainlink_btc,
    // Per-market arrays, shape (M,).
    py::array_t<long long> market_start,
    py::array_t<long long> market_end,
    py::array_t<int> market_outcome,  // 1 = YES, 0 = NO, -1 = unsettled.
    // Dense per-(tick, market) book state, shape (T, M).
    py::array_t<signed char> present,
    py::array_t<long long> book_ts,
    py::array_t<double> yes_mid,
    py::array_t<double> yes_best_ask,
    py::array_t<double> no_best_ask,
    py::array_t<double> yes_price,
    py::array_t<double> no_price,
    py::array_t<double> model_prob,  // NaN where no probability exists for (tick, market).
    // Dense per-(tick, market) ask ladders for walk-the-book, shape (T, M, K).
    py::array_t<double> yes_ask_px,
    py::array_t<double> yes_ask_sz,
    py::array_t<int> yes_ask_n,
    py::array_t<double> no_ask_px,
    py::array_t<double> no_ask_sz,
    py::array_t<int> no_ask_n,
    // Strategy + run configuration.
    int strategy_kind,  // 0 = WorldModelStrategy, 1 = NaiveLagStrategy.
    double edge_threshold,
    double order_size,
    double max_tte_frac,
    double min_tte_frac,
    bool use_cusum,
    double cusum_threshold,
    double starting_cash,
    long long snapshot_interval
) {
    auto ts_ = ts.unchecked<1>();
    auto cl_ = chainlink_btc.unchecked<1>();
    auto m_start_ = market_start.unchecked<1>();
    auto m_end_ = market_end.unchecked<1>();
    auto m_out_ = market_outcome.unchecked<1>();
    auto present_ = present.unchecked<2>();
    auto book_ts_ = book_ts.unchecked<2>();
    auto y_mid_ = yes_mid.unchecked<2>();
    auto y_ask_ = yes_best_ask.unchecked<2>();
    auto n_ask_ = no_best_ask.unchecked<2>();
    auto y_price_ = yes_price.unchecked<2>();
    auto n_price_ = no_price.unchecked<2>();
    auto prob_ = model_prob.unchecked<2>();
    auto y_lpx_ = yes_ask_px.unchecked<3>();
    auto y_lsz_ = yes_ask_sz.unchecked<3>();
    auto y_ln_ = yes_ask_n.unchecked<2>();
    auto n_lpx_ = no_ask_px.unchecked<3>();
    auto n_lsz_ = no_ask_sz.unchecked<3>();
    auto n_ln_ = no_ask_n.unchecked<2>();
    const long long total_ticks = ts_.shape(0);
    const int num_markets = static_cast<int>(m_start_.shape(0));
    // Engine state.
    double cash = starting_cash;
    std::vector<Position> positions(num_markets);
    std::vector<int> status(num_markets, 0);  // 0 = upcoming, 1 = active, 2 = settled.
    std::vector<PendingOrder> pending;
    int total_trades = 0;
    std::vector<EngineFill> all_fills;
    std::vector<double> snapshot_values;
    CusumGate cusum{cusum_threshold, 0.0, 0.0};
    bool have_prev_spot = false;
    double prev_spot = 0.0;
    for (long long t = 0; t < total_ticks; ++t) {
        const long long now = ts_(t);
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
            }
        }
        // 2. Execute pending orders due at or before this tick (T+1 latency), walking the recorded book.
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
            if (status[m] != 1 || present_(t, m) == 0) {
                continue;
            }
            if (now - book_ts_(t, m) > MAX_BOOK_STALENESS_S) {
                continue;
            }
            const int n_levels = (order.token == TOKEN_YES) ? y_ln_(t, m) : n_ln_(t, m);
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
                const double price = (order.token == TOKEN_YES) ? y_lpx_(t, m, level) : n_lpx_(t, m, level);
                const double level_size = (order.token == TOKEN_YES) ? y_lsz_(t, m, level) : n_lsz_(t, m, level);
                const double available = std::max(0.0, level_size - consumed[level]);
                if (available <= 0.0) {
                    continue;
                }
                const double fill_at_level = std::min(remaining, available);
                total_cost += fill_at_level * price;
                total_filled += fill_at_level;
                remaining -= fill_at_level;
                consumed[level] += fill_at_level;
                if (remaining <= 1e-9) {
                    break;
                }
            }
            if (total_filled <= 1e-9) {
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
                total_value += position.yes_shares * y_price_(t, m) + position.no_shares * n_price_(t, m);
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
        std::vector<PendingOrder> new_orders;
        const bool cusum_blocks = (strategy_kind == 0) ? (use_cusum && cusum_event == 0)
                                                       : (cusum_event == 0);
        if (!cusum_blocks) {
            for (int m = 0; m < num_markets; ++m) {
                if (status[m] != 1 || present_(t, m) == 0) {
                    continue;
                }
                const double duration = static_cast<double>(m_end_(m) - m_start_(m));
                const double time_remaining_s = std::max(0.0, static_cast<double>(m_end_(m) - now));
                const double time_remaining_frac = (duration > 0.0) ? time_remaining_s / duration : 0.0;
                const double book_prob = y_mid_(t, m);
                Position& position = positions[m];
                if (strategy_kind == 0) {
                    if (time_remaining_frac > max_tte_frac || time_remaining_frac < min_tte_frac) {
                        continue;
                    }
                    const double prob = prob_(t, m);
                    if (std::isnan(prob)) {
                        continue;
                    }
                    if (book_prob <= 0.0 || book_prob >= 1.0) {
                        continue;
                    }
                    const double edge = prob - book_prob;
                    if (std::abs(edge) < edge_threshold) {
                        continue;
                    }
                    if (edge > 0.0 && y_ask_(t, m) > 0.0) {
                        const double size = std::min(order_size, MAX_SHARES_PER_TOKEN - position.yes_shares);
                        if (size > 0.0) {
                            new_orders.push_back(PendingOrder{m, TOKEN_YES, size, now + 1});
                        }
                    } else if (edge < 0.0 && n_ask_(t, m) > 0.0) {
                        const double size = std::min(order_size, MAX_SHARES_PER_TOKEN - position.no_shares);
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
                    const int direction = cusum_event;
                    if (direction > 0 && book_prob < 0.5 && y_ask_(t, m) > 0.0 &&
                        position.yes_shares + order_size <= MAX_SHARES_PER_TOKEN) {
                        new_orders.push_back(PendingOrder{m, TOKEN_YES, order_size, now + 1});
                    } else if (direction < 0 && book_prob > 0.5 && n_ask_(t, m) > 0.0 &&
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
    // Final mark-to-market over the last tick's active books, matching the Python engine's final value.
    double final_value = cash;
    const long long last = total_ticks - 1;
    if (last >= 0) {
        for (int m = 0; m < num_markets; ++m) {
            const Position& position = positions[m];
            if (position.yes_shares <= 0.0 && position.no_shares <= 0.0) {
                continue;
            }
            if (status[m] == 1) {
                final_value += position.yes_shares * y_price_(last, m) + position.no_shares * n_price_(last, m);
            } else {
                final_value += position.cost_basis;
            }
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
    result["snapshot_values"] = py::cast(snapshot_values);
    return result;
}


PYBIND11_MODULE(_engine, module) {
    module.doc() = "Native C++ core of the DATAHACKS2026 backtester (parity with the Python engine).";
    module.def("run_backtest", &run_backtest, "Run one backtest over a dense scoped timeline.");
}
