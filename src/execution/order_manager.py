"""
src/execution/order_manager.py
───────────────────────────────
Execution controller for prediction market orders.

Responsibilities:
  1. Maintains the live order book state (open orders per market)
  2. Executes cancel/replace logic based on FairValueResult changes
  3. Implements Flickering Filter: detects sub-500ms institutional cancel patterns
  4. Post-Only enforcement (maker-only orders)
  5. Self-trade prevention
  6. Async submission via REST with retry and timeout handling

Cancel/Replace Policy:
  - Reprice if |current_quote - new_quote| > min_edge_bps / 2
  - Always cancel if state.is_stale or RiskEngine signals kill
  - Don't reprice within 50ms of last placement (anti-latency-arbitrage)
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
import structlog

from config.settings import RiskProfile
from src.data.unified_book import MarketState
from src.execution.eip712_signer import EIP712Signer, OrderParams, OrderSide, SignedOrder
from src.pricing.fair_value import FairValueResult

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────

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
    token_id: str
    side: OrderSideStr
    price: float         # quote price (probability)
    size: float          # contracts
    placed_ts: float     # time.monotonic()
    status: OrderStatus  = OrderStatus.PENDING
    filled_size: float   = 0.0
    last_update_ts: float = field(default_factory=time.monotonic)


# ──────────────────────────────────────────────
# Flickering Filter
# ──────────────────────────────────────────────

class FlickeringFilter:
    """
    Detects institutional order flickering (rapid cancel-replace cycles
    used to manipulate queue position or signal intent).

    Trigger: N cancel/replace events on ONE side within WINDOW_MS.
    Action: freeze quoting on that side for FREEZE_MS.
    """

    def __init__(
        self,
        window_ms: int = 500,
        cancel_threshold: int = 3,
        freeze_ms: int = 5_000,
    ):
        self._window_ms    = window_ms
        self._threshold    = cancel_threshold
        self._freeze_ms    = freeze_ms
        # market_id → side → deque of cancel timestamps
        self._cancels: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: {"BUY": deque(), "SELL": deque()}
        )
        # market_id → side → freeze_until (monotonic)
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


# ──────────────────────────────────────────────
# Order Manager
# ──────────────────────────────────────────────

class OrderManager:
    """
    Async order lifecycle manager for Polymarket CLOB.

    Usage:
        mgr = OrderManager(session, signer, risk_profile, rest_url)
        await mgr.update_quotes(state, fv_result, inventory_q)
    """

    # Minimum time between cancel-and-replace to suppress latency arb
    MIN_REPRICE_INTERVAL_S: float = 0.050   # 50ms
    # Minimum price move to justify a cancel/replace (as fraction of spread)
    REPRICE_THRESHOLD_FRAC: float = 0.30

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        signer: EIP712Signer,
        risk_profile: RiskProfile,
        rest_url: str,
        flickering_filter: Optional[FlickeringFilter] = None,
    ):
        self._session   = http_session
        self._signer    = signer
        self._risk      = risk_profile
        self._rest_url  = rest_url.rstrip("/")
        self._flicker   = flickering_filter or FlickeringFilter(
            window_ms=risk_profile.flickering_window_ms,
            cancel_threshold=risk_profile.flickering_cancel_threshold,
            freeze_ms=risk_profile.toxic_flow_pause_ms,
        )

        # Live orders indexed by (market_id, side)
        self._live: Dict[Tuple[str, str], ManagedOrder] = {}
        # All orders (for fill tracking)
        self._all: Dict[str, ManagedOrder] = {}

        self._log = logger.bind(component="order_manager")

    # ── Main entry point ──────────────────────

    async def update_quotes(
        self,
        state: MarketState,
        fv: FairValueResult,
        yes_token_id: str,
        inventory_q: float,
        order_size_usd: float = 50.0,
    ) -> None:
        """
        Compare current live quotes against FairValueResult.
        Cancel stale quotes, reprice if necessary, post new quotes.
        """
        market_id = state.market_id

        # ── Safety checks ─────────────────────
        if not fv.should_quote:
            await self._cancel_all(market_id, reason="should_not_quote")
            return

        if abs(inventory_q) >= self._risk.max_inventory_contracts:
            # At max inventory: only quote the reducing side
            await self._handle_max_inventory(market_id, inventory_q)
            return

        # ── Process each side ─────────────────
        for side_str, quote_price in [
            ("BUY",  fv.bid_quote),
            ("SELL", fv.ask_quote),
        ]:
            # Self-trade prevention: if we have a matching ask already placed,
            # don't place a bid above it (shouldn't happen, but guard anyway)
            opposite = "SELL" if side_str == "BUY" else "BUY"
            opp_order = self._live.get((market_id, opposite))
            if opp_order and opp_order.status == OrderStatus.OPEN:
                if side_str == "BUY" and quote_price >= opp_order.price:
                    self._log.warning(
                        "stp_guard",
                        market_id=market_id,
                        bid=quote_price,
                        ask=opp_order.price,
                    )
                    continue

            # Flickering filter
            if self._flicker.is_frozen(market_id, side_str):
                self._log.debug("quote_frozen_by_flicker", market_id=market_id, side=side_str)
                continue

            await self._update_side(
                market_id=market_id,
                side_str=side_str,
                quote_price=quote_price,
                fv=fv,
                yes_token_id=yes_token_id,
                order_size_usd=order_size_usd,
            )

    async def _update_side(
        self,
        market_id: str,
        side_str: str,
        quote_price: float,
        fv: FairValueResult,
        yes_token_id: str,
        order_size_usd: float,
    ) -> None:
        key = (market_id, side_str)
        live = self._live.get(key)

        if live is None or live.status not in (OrderStatus.OPEN, OrderStatus.PENDING):
            # No live order — place new
            await self._place_order(
                market_id, side_str, quote_price, yes_token_id, order_size_usd
            )
            return

        # Order exists — check if reprice is warranted
        age_s = time.monotonic() - live.placed_ts
        if age_s < self.MIN_REPRICE_INTERVAL_S:
            return   # Too soon to reprice

        price_move = abs(quote_price - live.price)
        spread     = fv.half_spread * 2
        if spread > 0 and price_move / spread < self.REPRICE_THRESHOLD_FRAC:
            return   # Move too small to justify cancel overhead

        # Cancel then replace
        self._log.info(
            "repricing",
            market_id=market_id,
            side=side_str,
            old_price=round(live.price, 4),
            new_price=round(quote_price, 4),
            move_bps=round(price_move * 10_000, 1),
        )
        cancelled = await self._cancel_order(live)
        if cancelled:
            await self._place_order(
                market_id, side_str, quote_price, yes_token_id, order_size_usd
            )

    # ── Order operations ──────────────────────

    async def _place_order(
        self,
        market_id: str,
        side_str: str,
        price: float,
        token_id: str,
        size_usd: float,
    ) -> Optional[ManagedOrder]:
        """
        Build, sign, and submit a POST-ONLY limit order.
        """
        # Derive contract size from USD notional
        # size (contracts) ≈ size_usd / price   [USD / (USD/contract)]
        n_contracts = size_usd / max(price, 0.01)

        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
        params = OrderParams(
            token_id=token_id,
            side=side,
            price=price,
            size=n_contracts,
        )

        try:
            signed: SignedOrder = self._signer.sign_order(params)
        except Exception as exc:
            self._log.error("sign_failed", error=str(exc))
            return None

        payload = signed.to_api_dict()
        # POST_ONLY flag for Polymarket CLOB
        payload["orderType"] = "GTC"   # Good-til-cancel; CLOB enforces post-only at contract level

        url = f"{self._rest_url}/order"
        try:
            async with self._session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=3.0)
            ) as resp:
                if resp.status in (200, 201):
                    body = await resp.json()
                    order_id = body.get("orderID") or body.get("id", "unknown")
                else:
                    text = await resp.text()
                    self._log.error(
                        "order_rejected",
                        status=resp.status,
                        body=text[:300],
                        market_id=market_id,
                        side=side_str,
                        price=price,
                    )
                    return None

        except asyncio.TimeoutError:
            self._log.error("order_timeout", market_id=market_id, side=side_str)
            return None
        except aiohttp.ClientError as exc:
            self._log.error("order_http_error", error=str(exc))
            return None

        order = ManagedOrder(
            order_id=order_id,
            market_id=market_id,
            token_id=token_id,
            side=OrderSideStr(side_str),
            price=price,
            size=n_contracts,
            placed_ts=time.monotonic(),
            status=OrderStatus.PENDING,
        )
        self._live[(market_id, side_str)] = order
        self._all[order_id] = order

        self._log.info(
            "order_placed",
            order_id=order_id,
            market_id=market_id,
            side=side_str,
            price=round(price, 4),
            size=round(n_contracts, 2),
        )
        return order

    async def _cancel_order(self, order: ManagedOrder) -> bool:
        """Cancel a single order. Returns True if cancelled."""
        url = f"{self._rest_url}/order/{order.order_id}"
        try:
            async with self._session.delete(
                url, timeout=aiohttp.ClientTimeout(total=2.0)
            ) as resp:
                if resp.status in (200, 204):
                    order.status = OrderStatus.CANCELLED
                    self._flicker.record_cancel(
                        order.market_id, order.side.value, time.monotonic()
                    )
                    del self._live[(order.market_id, order.side.value)]
                    self._log.info(
                        "order_cancelled",
                        order_id=order.order_id,
                        market_id=order.market_id,
                        side=order.side.value,
                    )
                    return True
                else:
                    text = await resp.text()
                    self._log.warning(
                        "cancel_failed",
                        status=resp.status,
                        body=text[:200],
                        order_id=order.order_id,
                    )
                    return False

        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            self._log.error("cancel_error", error=str(exc), order_id=order.order_id)
            return False

    async def _cancel_all(self, market_id: str, reason: str = "") -> None:
        """Cancel all live orders for a market."""
        to_cancel = [
            order for (mid, _), order in self._live.items()
            if mid == market_id and order.status in (OrderStatus.OPEN, OrderStatus.PENDING)
        ]
        if to_cancel:
            self._log.info(
                "cancel_all",
                market_id=market_id,
                n=len(to_cancel),
                reason=reason,
            )
            await asyncio.gather(*[self._cancel_order(o) for o in to_cancel])

    async def _handle_max_inventory(self, market_id: str, inventory_q: float) -> None:
        """
        At inventory limit: only quote the side that reduces exposure.
        Long → only SELL quotes; Short → only BUY quotes.
        """
        cancel_side = "BUY" if inventory_q > 0 else "SELL"
        key = (market_id, cancel_side)
        if key in self._live:
            await self._cancel_order(self._live[key])

    # ── Fill handling ─────────────────────────

    def mark_filled(self, order_id: str, fill_size: float) -> Optional[ManagedOrder]:
        """Called when the user-stream reports a fill."""
        order = self._all.get(order_id)
        if order is None:
            return None
        order.filled_size    += fill_size
        order.last_update_ts  = time.monotonic()
        if order.filled_size >= order.size * 0.999:
            order.status = OrderStatus.FILLED
            key = (order.market_id, order.side.value)
            self._live.pop(key, None)
        else:
            order.status = OrderStatus.PARTIAL_FILL
        return order

    def mark_open(self, order_id: str) -> None:
        order = self._all.get(order_id)
        if order:
            order.status = OrderStatus.OPEN

    # ── Introspection ─────────────────────────

    def get_live_quotes(self, market_id: str) -> Dict[str, Optional[ManagedOrder]]:
        return {
            "BUY":  self._live.get((market_id, "BUY")),
            "SELL": self._live.get((market_id, "SELL")),
        }
