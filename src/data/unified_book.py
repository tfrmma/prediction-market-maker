"""
src/data/unified_book.py
─────────────────────────
Core State Engine: Unified YES/NO synthetic probability matrix.

KEY DESIGN:
  Polymarket has two separate token books: YES and NO.
  The no-arbitrage constraint:

      Bid_YES  ≤  1 - Ask_NO   (otherwise: buy YES + buy NO < $1 → free money)
      Ask_YES  ≥  1 - Bid_NO

  This module:
    1. Maintains L2 books for YES and NO sides independently
    2. Folds them into a SINGLE synthetic probability density matrix
    3. Computes:
       - Synthetic mid:   P_mid  = (Bid_YES_mid + (1 - Ask_NO_mid)) / 2
       - Implied spread:  Σ      = Ask_YES - Bid_YES  (and 1-side equivalent)
       - Order Flow Imbalance (OFI): depth-weighted buy/sell pressure
       - CVD (Cumulative Volume Delta): aggressive taker direction
       - Arbitrage signal:  arb_gap = Bid_YES + Bid_NO - 1.0  (>0 = arb exists)
    4. Emits a MarketState snapshot on every update (consumed by pricing layer)

Handles both Polymarket (two-token) and Kalshi (single YES-side book).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.data.polymarket_feed import (
    PolyBookSnapshot,
    PolyPriceDelta,
    PolyTrade,
    PriceLevel,
)
from src.data.kalshi_feed import (
    KalshiBookSnapshot,
    KalshiBookDelta,
    KalshiTrade,
    KalshiLevel,
    KalshiTicker,
)

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Output types
# ──────────────────────────────────────────────

class BookSource(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI     = "kalshi"


@dataclass(slots=True)
class DepthLevel:
    price: float   # probability [0, 1]
    size: float    # USD notional equivalent


@dataclass
class MarketState:
    """
    Single consistent snapshot of the full market microstructure.
    Emitted by UnifiedBook after every update.
    Consumed by: pricing layer, risk engine, hedging module.
    """
    market_id: str         # condition_id (Poly) or ticker (Kalshi)
    source: BookSource
    ts: float              # monotonic

    # ── Synthetic probability ──────────────
    p_mid: float           # best estimate of true probability [0, 1]
    p_bid: float           # best bid (YES side)  — highest willing buyer
    p_ask: float           # best ask (YES side)  — lowest willing seller
    spread: float          # p_ask - p_bid
    
    # From NO side (only Polymarket)
    p_bid_no: Optional[float] = None   # best bid on NO token
    p_ask_no: Optional[float] = None   # best ask on NO token
    
    # Arb signal (Poly only): > 0 means risk-free profit exists
    arb_gap: float = 0.0   # Bid_YES + Bid_NO - 1  (should be ≤ 0 in efficient market)

    # ── Order book depth ──────────────────
    bids: List[DepthLevel] = field(default_factory=list)   # top-5 YES bids
    asks: List[DepthLevel] = field(default_factory=list)   # top-5 YES asks

    # ── Flow metrics ──────────────────────
    ofi: float = 0.0          # Order Flow Imbalance (positive = buy pressure)
    cvd: float = 0.0          # Cumulative Volume Delta (rolling window)
    
    # ── Volatility proxy ──────────────────
    realized_vol_1m: float = 0.0    # σ of mid-price changes over 1-min window
    
    # ── Liquidity metrics ─────────────────
    bid_depth_usd: float = 0.0     # total bid liquidity in top 5 levels
    ask_depth_usd: float = 0.0     # total ask liquidity in top 5 levels
    imbalance: float = 0.0         # (bid_depth - ask_depth) / (bid_depth + ask_depth)

    # ── Time to resolution ────────────────
    resolution_ts: int = 0         # unix timestamp
    time_to_resolution_s: float = 0.0

    # ── Exchange timestamps ───────────────
    book_ts_ms: int = 0
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[float] = None

    def is_valid(self) -> bool:
        """Basic sanity checks before passing to pricing layer."""
        if not (0.0 < self.p_bid < self.p_ask < 1.0):
            return False
        if self.spread > 0.30:  # >30 cent spread → illiquid, skip
            return False
        if self.arb_gap > 0.005:   # >0.5c arb → data inconsistency
            return False
        return True


# ──────────────────────────────────────────────
# L2 Book: mutable in-memory representation
# ──────────────────────────────────────────────

class L2Book:
    """
    Thread-safe-ish (single-threaded asyncio) L2 order book.
    Prices stored as floats in [0, 1].
    """

    MAX_DEPTH = 20   # levels to retain

    def __init__(self, label: str):
        self._label = label
        # price → size
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._ts_ms: int = 0

    def apply_snapshot(
        self,
        bids: List[PriceLevel | KalshiLevel],
        asks: List[PriceLevel | KalshiLevel],
        ts_ms: int,
    ) -> None:
        self._bids = {lvl.price: lvl.size for lvl in bids if lvl.size > 0}
        self._asks = {lvl.price: lvl.size for lvl in asks if lvl.size > 0}
        self._ts_ms = ts_ms
        self._trim()

    def apply_delta(
        self,
        side: str,    # "BUY"/"bid"/"buy" → bids
        price: float,
        size: float,  # 0 = remove
        ts_ms: int,
    ) -> None:
        book = self._bids if side.upper() in ("BUY", "BID") else self._asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size
        self._ts_ms = max(self._ts_ms, ts_ms)
        self._trim()

    def best_bid(self) -> Optional[Tuple[float, float]]:
        if not self._bids:
            return None
        p = max(self._bids)
        return (p, self._bids[p])

    def best_ask(self) -> Optional[Tuple[float, float]]:
        if not self._asks:
            return None
        p = min(self._asks)
        return (p, self._asks[p])

    def top_levels(self, n: int = 5) -> Tuple[List[DepthLevel], List[DepthLevel]]:
        bids = sorted(self._bids.items(), reverse=True)[:n]
        asks = sorted(self._asks.items())[:n]
        return (
            [DepthLevel(p, s) for p, s in bids],
            [DepthLevel(p, s) for p, s in asks],
        )

    def depth_usd(self, n: int = 5) -> Tuple[float, float]:
        bids, asks = self.top_levels(n)
        # In binary markets, 1 contract = $1 notional at payout
        # USD value at mid-point: approximate each level as size * price
        bid_usd = sum(lvl.price * lvl.size for lvl in bids)
        ask_usd = sum((1 - lvl.price) * lvl.size for lvl in asks)
        return bid_usd, ask_usd

    def _trim(self) -> None:
        """Keep only top MAX_DEPTH levels."""
        if len(self._bids) > self.MAX_DEPTH:
            sorted_bids = sorted(self._bids, reverse=True)
            for p in sorted_bids[self.MAX_DEPTH:]:
                del self._bids[p]
        if len(self._asks) > self.MAX_DEPTH:
            sorted_asks = sorted(self._asks)
            for p in sorted_asks[self.MAX_DEPTH:]:
                del self._asks[p]

    @property
    def ts_ms(self) -> int:
        return self._ts_ms


# ──────────────────────────────────────────────
# OFI + CVD rolling state
# ──────────────────────────────────────────────

@dataclass
class OFIState:
    """
    Order Flow Imbalance: tracks changes in top-of-book depth.

    OFI = Σ [bid_depth_change - ask_depth_change]
    where changes triggered by aggressive orders carry sign.
    Ref: Cont, Kukanov, Stoikov (2014)
    """
    prev_bid: float = 0.0
    prev_ask: float = 0.0

    def update(self, new_bid: float, new_ask: float) -> float:
        delta_bid = new_bid - self.prev_bid
        delta_ask = new_ask - self.prev_ask
        self.prev_bid = new_bid
        self.prev_ask = new_ask
        # Buy pressure → bid depth increases OR ask depth decreases
        return delta_bid - delta_ask


class CVDAccumulator:
    """
    Cumulative Volume Delta over a rolling time window.
    Taker buys increment; taker sells decrement.
    """
    WINDOW_S: float = 300.0   # 5-min rolling

    def __init__(self):
        # deque of (timestamp, signed_volume)
        self._trades: Deque[Tuple[float, float]] = deque()

    def add_trade(self, ts: float, price: float, size: float, taker_side: str) -> None:
        sign = 1.0 if taker_side.upper() in ("BUY", "YES") else -1.0
        self._trades.append((ts, sign * size * price))  # $-weighted
        self._evict(ts)

    def value(self) -> float:
        return sum(v for _, v in self._trades)

    def _evict(self, now: float) -> None:
        cutoff = now - self.WINDOW_S
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()


class MidHistory:
    """Tracks recent mid-prices for realized volatility."""
    WINDOW_S: float = 60.0

    def __init__(self):
        self._mids: Deque[Tuple[float, float]] = deque()

    def add(self, ts: float, mid: float) -> None:
        self._mids.append((ts, mid))
        self._evict(ts)

    def realized_vol(self) -> float:
        if len(self._mids) < 2:
            return 0.0
        prices = np.array([m for _, m in self._mids])
        returns = np.diff(prices)
        return float(np.std(returns)) if len(returns) > 0 else 0.0

    def _evict(self, now: float) -> None:
        cutoff = now - self.WINDOW_S
        while self._mids and self._mids[0][0] < cutoff:
            self._mids.popleft()


# ──────────────────────────────────────────────
# Unified Book (main class)
# ──────────────────────────────────────────────

class UnifiedBook:
    """
    Manages the combined YES/NO probability surface for one market.
    
    Call process(event) with any feed event.
    After each update, returns a MarketState or None if not yet ready.
    """

    def __init__(
        self,
        market_id: str,
        source: BookSource,
        resolution_ts: int = 0,
    ):
        self._market_id = market_id
        self._source    = source
        self._res_ts    = resolution_ts

        # Books
        self._yes_book  = L2Book(f"{market_id}:YES")
        self._no_book   = L2Book(f"{market_id}:NO")   # Poly only

        # Flow state
        self._ofi  = OFIState()
        self._cvd  = CVDAccumulator()
        self._mids = MidHistory()

        # Flags
        self._yes_initialized = False
        self._no_initialized  = False   # Kalshi: always True (single book)

        if source == BookSource.KALSHI:
            self._no_initialized = True   # NO is implicit via complement

        self._log = logger.bind(market_id=market_id, source=source.value)

    # ── Public entry point ────────────────────

    def process(self, event) -> Optional[MarketState]:
        """
        Dispatch any feed event. Returns updated MarketState or None.
        """
        if isinstance(event, PolyBookSnapshot):
            return self._handle_poly_snapshot(event)
        elif isinstance(event, PolyPriceDelta):
            return self._handle_poly_delta(event)
        elif isinstance(event, PolyTrade):
            return self._handle_poly_trade(event)
        elif isinstance(event, KalshiBookSnapshot):
            return self._handle_kalshi_snapshot(event)
        elif isinstance(event, KalshiBookDelta):
            return self._handle_kalshi_delta(event)
        elif isinstance(event, (KalshiTrade, KalshiTicker)):
            return self._handle_kalshi_ticker_or_trade(event)
        return None

    # ── Polymarket handlers ───────────────────

    def _handle_poly_snapshot(self, ev: PolyBookSnapshot) -> Optional[MarketState]:
        book = self._yes_book if ev.is_yes_token else self._no_book
        book.apply_snapshot(ev.bids, ev.asks, ev.timestamp_ms)

        if ev.is_yes_token:
            self._yes_initialized = True
        else:
            self._no_initialized = True

        return self._build_state(ev.recv_ts, ev.timestamp_ms)

    def _handle_poly_delta(self, ev: PolyPriceDelta) -> Optional[MarketState]:
        book = self._yes_book if ev.is_yes_token else self._no_book
        book.apply_delta(
            side="BUY" if ev.side in ("BUY", "bid") else "SELL",
            price=ev.price,
            size=ev.size,
            ts_ms=ev.timestamp_ms,
        )
        return self._build_state(ev.recv_ts, ev.timestamp_ms)

    def _handle_poly_trade(self, ev: PolyTrade) -> Optional[MarketState]:
        side = "YES" if ev.is_yes_token else "NO"
        # Taker buys YES = bullish signal
        taker_direction = ev.side   # already "BUY" or "SELL"
        effective_side  = "BUY" if (
            (ev.is_yes_token and taker_direction == "BUY") or
            (not ev.is_yes_token and taker_direction == "SELL")
        ) else "SELL"

        self._cvd.add_trade(
            ts=time.monotonic(),
            price=ev.price,
            size=ev.size,
            taker_side=effective_side,
        )
        return self._build_state(ev.recv_ts, ev.timestamp_ms)

    # ── Kalshi handlers ───────────────────────

    def _handle_kalshi_snapshot(self, ev: KalshiBookSnapshot) -> Optional[MarketState]:
        # Kalshi provides YES side; we store NO as complement
        # Convert KalshiLevel → PriceLevel-compatible for L2Book
        bids = [PriceLevel(price=lvl.price, size=float(lvl.size)) for lvl in ev.yes_bids]
        asks = [PriceLevel(price=lvl.price, size=float(lvl.size)) for lvl in ev.yes_asks]
        self._yes_book.apply_snapshot(bids, asks, ev.timestamp_ms)
        self._yes_initialized = True
        return self._build_state(ev.recv_ts, ev.timestamp_ms)

    def _handle_kalshi_delta(self, ev: KalshiBookDelta) -> Optional[MarketState]:
        # Kalshi delta only quotes YES side
        # "yes" bid delta → BUY side update
        raw_side = "BUY" if ev.side == "yes" and ev.action in ("add",) else "SELL"
        size = float(ev.delta) if ev.action == "add" else 0.0
        self._yes_book.apply_delta(raw_side, ev.price, size, ev.timestamp_ms)
        return self._build_state(ev.recv_ts, ev.timestamp_ms)

    def _handle_kalshi_ticker_or_trade(self, ev) -> Optional[MarketState]:
        if isinstance(ev, KalshiTrade):
            self._cvd.add_trade(
                ts=time.monotonic(),
                price=ev.price,
                size=float(ev.size),
                taker_side="YES" if ev.taker_side == "yes" else "NO",
            )
        return self._build_state(ev.recv_ts, getattr(ev, "timestamp_ms", 0))

    # ── State construction ────────────────────

    def _build_state(self, recv_ts: float, book_ts_ms: int) -> Optional[MarketState]:
        if not (self._yes_initialized and self._no_initialized):
            return None  # not ready yet

        yes_bid_lvl = self._yes_book.best_bid()
        yes_ask_lvl = self._yes_book.best_ask()

        if yes_bid_lvl is None or yes_ask_lvl is None:
            return None  # empty book

        p_bid_yes = yes_bid_lvl[0]
        p_ask_yes = yes_ask_lvl[0]

        # NO-side complement (Polymarket)
        arb_gap  = 0.0
        p_bid_no = None
        p_ask_no = None

        if self._source == BookSource.POLYMARKET:
            no_bid_lvl = self._no_book.best_bid()
            no_ask_lvl = self._no_book.best_ask()

            if no_bid_lvl and no_ask_lvl:
                p_bid_no = no_bid_lvl[0]
                p_ask_no = no_ask_lvl[0]

                # Synthetic mid: average YES-mid and (1 - NO-mid)
                yes_mid_raw = (p_bid_yes + p_ask_yes) / 2
                no_mid_raw  = 1.0 - (p_bid_no + p_ask_no) / 2
                p_mid = (yes_mid_raw + no_mid_raw) / 2

                # Arb gap: if buyers of YES AND NO can lock profit
                arb_gap = p_bid_yes + p_bid_no - 1.0
                if arb_gap > 0.005:
                    self._log.warning(
                        "arb_detected",
                        arb_gap=round(arb_gap, 4),
                        p_bid_yes=p_bid_yes,
                        p_bid_no=p_bid_no,
                    )
            else:
                p_mid = (p_bid_yes + p_ask_yes) / 2
        else:
            p_mid = (p_bid_yes + p_ask_yes) / 2

        # OFI
        bid_d, ask_d = self._yes_book.depth_usd(n=5)
        ofi = self._ofi.update(bid_d, ask_d)

        # Realized vol
        self._mids.add(recv_ts, p_mid)
        vol = self._mids.realized_vol()

        # Top levels
        bids_top, asks_top = self._yes_book.top_levels(5)

        # Imbalance
        imb = 0.0
        total_depth = bid_d + ask_d
        if total_depth > 0:
            imb = (bid_d - ask_d) / total_depth

        # Time to resolution
        t_to_res = max(0.0, self._res_ts - time.time()) if self._res_ts else 0.0

        return MarketState(
            market_id=self._market_id,
            source=self._source,
            ts=recv_ts,
            p_mid=p_mid,
            p_bid=p_bid_yes,
            p_ask=p_ask_yes,
            spread=p_ask_yes - p_bid_yes,
            p_bid_no=p_bid_no,
            p_ask_no=p_ask_no,
            arb_gap=arb_gap,
            bids=bids_top,
            asks=asks_top,
            ofi=ofi,
            cvd=self._cvd.value(),
            realized_vol_1m=vol,
            bid_depth_usd=bid_d,
            ask_depth_usd=ask_d,
            imbalance=imb,
            resolution_ts=self._res_ts,
            time_to_resolution_s=t_to_res,
            book_ts_ms=book_ts_ms,
        )


# ──────────────────────────────────────────────
# Registry: multiple markets managed together
# ──────────────────────────────────────────────

class BookRegistry:
    """
    Manages one UnifiedBook per market.
    Acts as a single dispatch point for all raw feed events.
    """

    def __init__(self, state_queue: "asyncio.Queue[MarketState]"):
        self._books: Dict[str, UnifiedBook] = {}
        self._state_queue = state_queue

    def register(
        self,
        market_id: str,
        source: BookSource,
        resolution_ts: int = 0,
    ) -> None:
        self._books[market_id] = UnifiedBook(market_id, source, resolution_ts)

    async def process(self, event) -> None:
        """
        Route event to the correct UnifiedBook and emit state if updated.
        """
        mid = self._resolve_market_id(event)
        if mid is None or mid not in self._books:
            return

        state = self._books[mid].process(event)
        if state is not None:
            await self._state_queue.put(state)

    def _resolve_market_id(self, event) -> Optional[str]:
        """Extract market identifier from any event type."""
        if hasattr(event, "condition_id"):
            return event.condition_id
        if hasattr(event, "market_ticker"):
            return event.market_ticker
        return None
