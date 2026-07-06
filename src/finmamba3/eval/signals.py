"""Model-to-signal inference for the PnL backtest.

The frozen world model's per-tick settlement-YES probability and book-relative EV series,
plus the causal predictability (efficiency-ratio) and trend-direction gate series read by
the strategy and its naive baseline.
"""
# region imports
from __future__ import annotations
import numpy as np
import torch
from finmamba3.backtester.data_loader import _asset_from_slug
# endregion


def world_model_yes_prob_series(
    wm, seq, device: torch.device, window_len: int = 64, chunk: int = 512,
    sample_mode: str = "random_sample",
) -> dict:
    """Per-tick settlement YES probability for one market, keyed by tick second.

    For each tick with at least window_len preceding context, the settlement head reads the causal
    window ending at that tick and the sigmoid of its last-position logit is the spot-conditioned YES
    probability. Reuses the exact encode -> sequence -> settlement_head path the forecasting evaluator
    scores, so train and serve see one forward. Windows are batched in chunks to bound GPU memory.
    sample_mode='probs' takes the deterministic latent (no random sample), making the PnL reproducible.
    """
    flat = seq.to_flat()
    total_ticks = flat.shape[0]
    prob_by_ts = {}
    if total_ticks < window_len:
        return prob_by_ts
    starts = np.arange(0, total_ticks - window_len + 1)
    for chunk_start in range(0, len(starts), chunk):
        batch_starts = starts[chunk_start : chunk_start + chunk]
        windows = np.stack([flat[s : s + window_len] for s in batch_starts], axis=0)
        obs = torch.from_numpy(windows).float().to(device)
        action = torch.zeros((obs.shape[0], window_len), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
            embedding = wm.encoder(obs)
            post_logits = wm.dist_head.forward_post(embedding)
            sample = wm.straight_through_gradient(post_logits, sample_mode=sample_mode)
            flattened_sample = wm.flatten_sample(sample)
            if wm.model in ("Transformer", "TransformerModern"):
                from finmamba3.models.attention import get_subsequent_mask_with_batch_length
                mask = get_subsequent_mask_with_batch_length(window_len, flattened_sample.device)
                dist_feat = wm.sequence_model(flattened_sample, action, mask)
            else:
                dist_feat = wm.sequence_model(flattened_sample, action)
            dist_feat = wm.condition_dist_feat(dist_feat)
            settle_logits = wm.settlement_head(dist_feat).float()
        last_prob = torch.sigmoid(settle_logits[:, -1]).cpu().numpy()
        for i, start in enumerate(batch_starts):
            prob_by_ts[int(seq.ts_sec[start + window_len - 1])] = float(last_prob[i])
    return prob_by_ts


def world_model_edge_ev_series(
    wm, seq, device: torch.device, window_len: int = 64, chunk: int = 512,
    sample_mode: str = "random_sample",
) -> dict:
    """Per-tick (ev_yes_hat, ev_no_hat) for one market, keyed by tick second.

    Mirrors world_model_yes_prob_series() exactly except the edge head replaces the settlement head.
    Returns an empty dict when the market is too short or the model has no edge head.
    """
    if not wm.use_edge_head or wm.edge_head is None:
        return {}
    flat = seq.to_flat()
    total_ticks = flat.shape[0]
    ev_by_ts = {}
    if total_ticks < window_len:
        return ev_by_ts
    starts = np.arange(0, total_ticks - window_len + 1)
    for chunk_start in range(0, len(starts), chunk):
        batch_starts = starts[chunk_start : chunk_start + chunk]
        windows = np.stack([flat[s : s + window_len] for s in batch_starts], axis=0)
        obs = torch.from_numpy(windows).float().to(device)
        action = torch.zeros((obs.shape[0], window_len), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
            embedding = wm.encoder(obs)
            post_logits = wm.dist_head.forward_post(embedding)
            sample = wm.straight_through_gradient(post_logits, sample_mode=sample_mode)
            flattened_sample = wm.flatten_sample(sample)
            if wm.model in ("Transformer", "TransformerModern"):
                from finmamba3.models.attention import get_subsequent_mask_with_batch_length
                mask_t = get_subsequent_mask_with_batch_length(window_len, flattened_sample.device)
                dist_feat = wm.sequence_model(flattened_sample, action, mask_t)
            else:
                dist_feat = wm.sequence_model(flattened_sample, action)
            dist_feat = wm.condition_dist_feat(dist_feat)
            ev_hat = wm.edge_head(dist_feat).float()
        last_ev = ev_hat[:, -1].cpu().numpy()
        for i, start in enumerate(batch_starts):
            ts = int(seq.ts_sec[start + window_len - 1])
            ev_by_ts[ts] = (float(last_ev[i, 0]), float(last_ev[i, 1]))
    return ev_by_ts


def _causal_er_series(spot: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing efficiency ratio of a dense spot series at every index, shape (n,) in [0, 1].

    The net move over the trailing window divided by its total path length; computed from a prefix sum
    of absolute increments so each index is O(1). This is the asset-level predictability the gate reads,
    precomputed so the strategy can gate any asset's markets (the engine's MarketState carries only BTC).
    """
    spot = np.asarray(spot, dtype=np.float64)
    num = spot.shape[0]
    abs_increment = np.abs(np.diff(spot, prepend=spot[:1]))
    cumulative_path = np.concatenate([[0.0], np.cumsum(abs_increment)])
    er = np.zeros(num, dtype=np.float64)
    for t in range(num):
        lo = max(0, t - window + 1)
        path = cumulative_path[t + 1] - cumulative_path[lo]
        er[t] = abs(spot[t] - spot[lo]) / (path + 1e-8)
    return er


def _causal_net_sign_series(spot: np.ndarray, window: int) -> np.ndarray:
    """Sign of the trailing-window net move at every index (+1 up, -1 down), shape (n,), causal.

    The direction a no-model trend-follower would take; precomputed per asset so the naive baseline is
    asset-correct (a SOL market follows SOL's trend, not BTC's) for the cross-asset anti-artifact check.
    """
    spot = np.asarray(spot, dtype=np.float64)
    num = spot.shape[0]
    sign = np.ones(num, dtype=np.float64)
    for t in range(num):
        lo = max(0, t - window + 1)
        if spot[t] < spot[lo]:
            sign[t] = -1.0
    return sign


def _gate_signals_by_slug_ts(bt, prob_by_slug_ts: dict, window: int) -> tuple:
    """Causal gate ER and trend-direction at each (slug, tick), using the slug's own asset spot.

    Builds each asset's causal ER and net-sign series from its Chainlink spot once, then looks up the
    right asset's signals at each market tick. This makes both the gate (a SOL market gates on SOL's
    trend) and the naive direction asset-correct, so the strategy and its baseline generalize across
    assets without touching the BTC-only engine. Returns (gate_er_by_slug_ts, gate_dir_by_slug_ts).
    """
    num = len(bt.timeline)
    ts_array = np.fromiter((int(tick.ts_sec) for tick in bt.timeline), dtype=np.int64, count=num)
    btc = np.fromiter((float(tick.chainlink_btc) for tick in bt.timeline), dtype=np.float64, count=num)
    eth = np.fromiter((float(tick.chainlink_eth) for tick in bt.timeline), dtype=np.float64, count=num)
    sol = np.fromiter((float(tick.chainlink_sol) for tick in bt.timeline), dtype=np.float64, count=num)
    er_by_asset = {
        "BTC": _causal_er_series(btc, window),
        "ETH": _causal_er_series(eth, window),
        "SOL": _causal_er_series(sol, window),
    }
    sign_by_asset = {
        "BTC": _causal_net_sign_series(btc, window),
        "ETH": _causal_net_sign_series(eth, window),
        "SOL": _causal_net_sign_series(sol, window),
    }
    gate_er_by_slug_ts = {}
    gate_dir_by_slug_ts = {}
    for slug, ts in prob_by_slug_ts.keys():
        asset = _asset_from_slug(slug)
        index = min(max(int(np.searchsorted(ts_array, ts)), 0), num - 1)
        gate_er_by_slug_ts[(slug, ts)] = float(er_by_asset[asset][index])
        gate_dir_by_slug_ts[(slug, ts)] = float(sign_by_asset[asset][index])
    return gate_er_by_slug_ts, gate_dir_by_slug_ts
