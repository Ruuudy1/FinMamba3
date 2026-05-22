"""Run a trained FinDrama world model as a competition backtester strategy.

Implements the BaseStrategy interface from lob.backtester.strategy (the same
contract the DATAHACKS competition harness uses). Each tick the active market's
order book is appended to a short rolling window of TickData and pushed through
the exact training feature pipeline (extract_features + normalization), so there
is no train/serve feature skew. The world-model direction head then picks a
bullish, bearish, or flat action, mirroring the offline GreedyDirectionPolicy.
"""
# region imports
from __future__ import annotations
from collections import deque
import numpy as np
import torch
from envs.lob_features import apply_normalization, extract_features, load_normalization
from lob.backtester.data_loader import TickData
from lob.backtester.strategy import BaseStrategy, MarketState, Order, Side, StoredBook, Token
# endregion


class FinDramaCompetitionStrategy(BaseStrategy):
    """Wraps a pretrained world model as a competition BaseStrategy.

    The world model is injected (already constructed and loaded) so the feature
    path can be unit-tested without a GPU. Use build_competition_strategy to
    construct one from a checkpoint.
    """

    def __init__(
        self,
        world_model,
        stats,
        mid_index: int,
        threshold: float = 0.005,
        order_size: float = 10.0,
        window: int = 64,
        device: str = "cuda",
        include_binary_features: bool = True,
    ) -> None:
        if world_model.direction_head is None:
            raise ValueError(
                "FinDramaCompetitionStrategy needs a world model trained with the "
                "direction head enabled (Models.WorldModel.Direction.Enabled=true)."
            )
        self.world_model = world_model
        self.stats = stats
        self.mid_index = int(mid_index)
        self.threshold = float(threshold)
        self.order_size = float(order_size)
        self.window = int(window)
        self.device = device
        self.include_binary_features = bool(include_binary_features)
        self.tick_window_by_slug: dict[str, deque] = {}
    def _feature_vector(self, slug: str, view, timestamp: int) -> np.ndarray:
        window = self.tick_window_by_slug.get(slug)
        if window is None:
            window = deque(maxlen=self.window)
            self.tick_window_by_slug[slug] = window
        tick = TickData(ts_sec=int(timestamp))
        tick.order_books[slug] = StoredBook(yes_book=view.yes_book, no_book=view.no_book, book_ts=int(timestamp))
        tick.book_timestamps[slug] = int(timestamp)
        window.append(tick)
        seq = extract_features(list(window), slug, include_binary_features=self.include_binary_features)
        seq = apply_normalization(seq, self.stats)
        return seq.to_flat()[-1]
    @torch.no_grad()
    def on_tick(self, state: MarketState) -> list[Order]:
        # Target the soonest-resolving market that has a two-sided YES book.
        active = [
            (view.end_ts, slug)
            for slug, view in state.markets.items()
            if view.yes_book.bids and view.yes_book.asks
        ]
        if not active:
            return []
        active.sort()
        slug = active[0][1]
        view = state.markets[slug]
        flat = self._feature_vector(slug, view, state.timestamp)
        x = torch.from_numpy(flat.astype(np.float32)).to(self.device).reshape(1, 1, -1)
        latent = self.world_model.encode_obs(x)
        dist_feat = self.world_model.sequence_model(
            latent[:, -1:],
            torch.zeros((1, 1), dtype=torch.long, device=self.device),
        )
        direction = int(self.world_model.direction_head(dist_feat).argmax(dim=-1).item())
        # A bullish call buys YES; a bearish call buys NO (the competition-correct way to
        # express downside, since selling YES with no inventory is rejected).
        if direction == 2:
            return [Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=self.order_size, limit_price=None)]
        if direction == 0:
            return [Order(market_slug=slug, token=Token.NO, side=Side.BUY, size=self.order_size, limit_price=None)]
        return []


def build_competition_strategy(
    checkpoint_path: str,
    config_path: str,
    norm_path: str,
    device: str = "cuda",
    threshold: float = 0.005,
    order_size: float = 10.0,
):
    """Construct a strategy from a Phase-A checkpoint, config, and normalization stats."""
    import yaml
    from config_utils import DotDict
    from sub_models.world_models import WorldModel
    with open(config_path) as f:
        config = DotDict(yaml.safe_load(f))
    world_model = WorldModel(action_dim=1, config=config, device=torch.device(device)).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    world_model.load_state_dict(state.get("world_model", state))
    world_model.eval()
    stats = load_normalization(norm_path)
    encoder_cfg = config.Models.WorldModel.Encoder
    mid_index = int(encoder_cfg.K) * int(encoder_cfg.FeatureDimLevel)
    return FinDramaCompetitionStrategy(
        world_model=world_model,
        stats=stats,
        mid_index=mid_index,
        threshold=threshold,
        order_size=order_size,
        device=device,
        include_binary_features=bool(encoder_cfg.BinaryMarketFeatures),
    )
