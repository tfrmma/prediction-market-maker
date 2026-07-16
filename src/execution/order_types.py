"""
Order bookkeeping types shared across venues. Used to live inline in
order_manager.py, pulled out here once Kalshi needed the same
ManagedOrder/FlickeringFilter machinery instead of a copy-paste.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict

import structlog

logger = structlog.get_logger(__name__)


def round_to_tick(price: float, tick_size: float) -> float:
    """Snap a price to the nearest valid tick. An order priced off-tick
    gets rejected outright, better to round here than find out from a
    400 mid-quoting-cycle."""
    if tick_size <= 0:
        return price
    ticks = round(price / tick_size)
    # guard against float noise turning e.g. 0.5700000000000001 into a
    # value that still doesn't compare equal to a clean multiple
    return round(ticks * tick_size, 10)


class OrderStatus(str, Enum):
    PENDING      = "pending"      # sent, awaiting ack
    OPEN         = "open"         # on book
    PARTIAL_FILL = "partial_fill"
    FILLED       = "filled"
    CANCELLED    = "cancelled"
    REJECTED     = "rejected"
    FAILED       = "failed"       # network/API error


class OrderSideStr(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


@dataclass
class ManagedOrder:
    order_id: str
    market_id: str
    token_id: str          # Polymarket token id, or the Kalshi ticker
    side: OrderSideStr
    price: float            # quote price (probability)
    size: float              # contracts
    placed_ts: float         # time.monotonic()
    status: OrderStatus  = OrderStatus.PENDING
    filled_size: float   = 0.0
    last_update_ts: float = field(default_factory=time.monotonic)


class FlickeringFilter:
    """
    Detects rapid cancel-replace cycles (used to manipulate queue
    position or signal intent) and freezes quoting on that side for a
    bit. Trigger: N cancels on one side within window_ms.
    """

    def __init__(
        self,
        window_ms: int = 500,
        cancel_threshold: int = 3,
        freeze_ms: int = 5_000,
    ):
        self._window_ms = window_ms
        self._threshold = cancel_threshold
        self._freeze_ms = freeze_ms
        self._cancels: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: {"BUY": deque(), "SELL": deque()}
        )
        self._frozen: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"BUY": 0.0, "SELL": 0.0}
        )

    def record_cancel(self, market_id: str, side: str, ts: float) -> None:
        q = self._cancels[market_id][side]
        q.append(ts)
        cutoff = ts - self._window_ms / 1000.0
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= self._threshold:
            freeze_until = ts + self._freeze_ms / 1000.0
            self._frozen[market_id][side] = freeze_until
            logger.warning(
                "flickering_detected",
                market_id=market_id,
                side=side,
                cancel_count=len(q),
                freeze_s=self._freeze_ms / 1000,
            )

    def is_frozen(self, market_id: str, side: str) -> bool:
        return time.monotonic() < self._frozen[market_id][side]

    def unfreeze(self, market_id: str, side: str) -> None:
        self._frozen[market_id][side] = 0.0
