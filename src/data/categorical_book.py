"""
Aggregates N per-outcome MarketState snapshots (one per leg of an
EventGroupConfig) into a single consistent basket view.

Each outcome already comes out of UnifiedBook as an ordinary binary
MarketState, p_mid there just means "probability this specific outcome
happens" instead of "probability YES". The only new thing here is
combining N of those and computing the same style of arb_gap signal
unified_book.py already computes for YES/NO, just across outcomes
instead of across the two legs of one market:

  sum(bid_i) > 1  -> mint one of each outcome for $1, dump into every
                      bid, guaranteed profit (needs a CTF-style split
                      mechanism at the venue, true for Polymarket
                      neg-risk markets).
  sum(ask_i) < 1  -> buy one of each outcome at the ask, redeem
                      whichever wins for $1, guaranteed profit.

Not wired into BookRegistry yet. That happens once the strategy loop
knows how to consume a basket instead of a single market at a time.
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

    sum_p_bid: float    # see module docstring for what each of these means
    sum_p_ask: float
    sum_p_mid: float     # informational only, mid isn't tradable

    @property
    def arb_gap_mint_and_dump(self) -> float:
        return self.sum_p_bid - 1.0

    @property
    def arb_gap_buy_basket(self) -> float:
        return 1.0 - self.sum_p_ask

    def is_valid(self, max_staleness_spread_s: float = 2.0) -> bool:
        """
        Sanity gate before this basket is safe to price off of.

        Deliberately doesn't gate on the arb_gap properties above, a
        few outcomes updating asynchronously means sum(bid_i) drifting
        off 1 for a moment is routine, not bad data (that's a pricing
        signal, see arb_gap_* above, not a validity check).
        """
        if not self.outcomes:
            return False
        for state in self.outcomes.values():
            if not state.is_valid():
                return False

        timestamps = [s.ts for s in self.outcomes.values()]
        if max(timestamps) - min(timestamps) > max_staleness_spread_s:
            # one leg is stale relative to the others, netting them right
            # now would price fresh outcomes against a stale one
            return False

        return True


class CategoricalBookAggregator:
    """
    One instance per event. Feed it MarketState updates for any of its
    outcome markets as they arrive off the normal per-market feeds, get
    a fresh CategoricalMarketState back once every outcome has reported
    at least once.
    """

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
        """
        Record a new snapshot for one outcome.

        Returns the basket snapshot once every outcome has reported at
        least once, None while still warming up. Note this can return a
        basket whose is_valid() is False, that's on purpose, callers
        decide what to do with a bad snapshot rather than never seeing it.
        """
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
