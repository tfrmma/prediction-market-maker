"""
Combines N per-outcome MarketState snapshots into one basket, and
flags the sum(bid)>1 / sum(ask)<1 arb the per-market arb_gap can't see
across outcomes. Not wired into BookRegistry yet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

from src.data.unified_book import MarketState

logger = structlog.get_logger(__name__)


@dataclass
class CategoricalMarketState:
    """Snapshot of an entire N-outcome basket at a point in time."""
    event_id: str
    ts: float

    outcomes: Dict[str, MarketState]   # market_id -> latest state, one per outcome

    sum_p_bid: float    # bids sum > 1: mint 1 of each outcome for $1, dump on the bids
    sum_p_ask: float    # asks sum < 1: buy 1 of each outcome, redeem the winner for $1
    sum_p_mid: float     # informational only, mid isn't tradable

    @property
    def arb_gap_mint_and_dump(self) -> float:
        return self.sum_p_bid - 1.0

    @property
    def arb_gap_buy_basket(self) -> float:
        return 1.0 - self.sum_p_ask

    def is_valid(self, max_staleness_spread_s: float = 2.0) -> bool:
        """Sanity gate before pricing off this basket. Doesn't gate on
        arb_gap_*, async outcomes drifting off sum=1 for a moment is
        routine, not bad data, that's a pricing signal not a validity check."""
        if not self.outcomes:
            return False
        for state in self.outcomes.values():
            if not state.is_valid():
                return False

        timestamps = [s.ts for s in self.outcomes.values()]
        if max(timestamps) - min(timestamps) > max_staleness_spread_s:
            return False   # one leg stale relative to its siblings

        return True


class CategoricalBookAggregator:
    """One instance per event. Feed it MarketState updates for its outcome
    markets, get a CategoricalMarketState back once every outcome has
    reported at least once."""

    ARB_LOG_THRESHOLD = 0.01   # log if either arb_gap exceeds this, purely diagnostic

    def __init__(self, event_id: str, outcome_ids: List[str]):
        if len(outcome_ids) < 2:
            raise ValueError(f"{event_id}: need at least 2 outcomes, got {outcome_ids}")
        self._event_id = event_id
        self._outcome_ids = list(outcome_ids)
        self._latest: Dict[str, MarketState] = {}
        self._log = logger.bind(component="categorical_book", event_id=event_id)

    @property
    def ready(self) -> bool:
        return all(oid in self._latest for oid in self._outcome_ids)

    def update(self, market_id: str, state: MarketState) -> Optional[CategoricalMarketState]:
        """Record a snapshot for one outcome, returns the basket once every
        outcome has reported at least once, None while warming up. Can
        return a basket whose is_valid() is False on purpose, callers
        decide what to do with it."""
        if market_id not in self._outcome_ids:
            return None   # not one of ours, caller routed this wrong

        self._latest[market_id] = state
        if not self.ready:
            return None

        sum_bid = sum(s.p_bid for s in self._latest.values())
        sum_ask = sum(s.p_ask for s in self._latest.values())
        sum_mid = sum(s.p_mid for s in self._latest.values())

        basket = CategoricalMarketState(
            event_id=self._event_id,
            ts=time.monotonic(),
            outcomes=dict(self._latest),
            sum_p_bid=sum_bid,
            sum_p_ask=sum_ask,
            sum_p_mid=sum_mid,
        )

        if (basket.arb_gap_mint_and_dump > self.ARB_LOG_THRESHOLD
                or basket.arb_gap_buy_basket > self.ARB_LOG_THRESHOLD):
            self._log.warning(
                "categorical_arb_detected",
                mint_and_dump=round(basket.arb_gap_mint_and_dump, 4),
                buy_basket=round(basket.arb_gap_buy_basket, 4),
                sum_p_bid=round(sum_bid, 4),
                sum_p_ask=round(sum_ask, 4),
            )

        return basket
