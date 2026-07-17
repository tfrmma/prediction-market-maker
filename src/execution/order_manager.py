"""
Execution controller. Tracks open orders per market, decides when to
cancel/replace based on FairValueResult changes, filters out flickering
(sub-500ms cancel spam), enforces post-only, self-trade prevention.

Reprice if |current_quote - new_quote| > min_edge_bps / 2. Always cancel
on state.is_stale or a kill signal from RiskEngine. Won't reprice within
50ms of the last placement, mostly to avoid chasing our own tail.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
import structlog

from config.settings import RiskProfile
from src.data.unified_book import MarketState
from src.execution.eip712_signer import EIP712Signer, OrderParams, OrderSide, SignedOrder
from src.execution.order_types import FlickeringFilter, ManagedOrder, OrderSideStr, OrderStatus, round_to_tick
from src.execution.polymarket_auth import PolyL2Auth
from src.pricing.fair_value import FairValueResult

logger = structlog.get_logger(__name__)


# Order Manager
class OrderManager:
    """Owns the order lifecycle for the Polymarket CLOB, cancel/replace,
    flickering suppression, fill bookkeeping."""

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
        l2_auth: Optional[PolyL2Auth] = None,
    ):
        self._session   = http_session
        self._signer    = signer
        self._risk      = risk_profile
        self._rest_url  = rest_url.rstrip("/")
        self._l2_auth   = l2_auth   # None => requests go out unauthenticated (will 401 for real trading)
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

    # Main entry point
    async def update_quotes(
        self,
        state: MarketState,
        fv: FairValueResult,
        yes_token_id: str,
        inventory_q: float,
        order_size_usd: float = 50.0,
        neg_risk: bool = False,
        tick_size: float = 0.01,
    ) -> None:
        """
        Compare current live quotes against FairValueResult.
        Cancel stale quotes, reprice if necessary, post new quotes.
        """
        market_id = state.market_id

        # Safety checks
        if not fv.should_quote:
            await self._cancel_all(market_id, reason="should_not_quote")
            return

        if abs(inventory_q) >= self._risk.max_inventory_contracts:
            # At max inventory: only quote the reducing side
            await self._handle_max_inventory(market_id, inventory_q)
            return

        # Process each side
        for side_str, quote_price in [
            ("BUY",  fv.bid_quote),
            ("SELL", fv.ask_quote),
        ]:
            # Self-trade prevention: don't let our own bid cross our own
            # ask, in either direction, shouldn't happen given how fv
            # is computed but the cost of checking is one comparison
            opposite = "SELL" if side_str == "BUY" else "BUY"
            opp_order = self._live.get((market_id, opposite))
            if opp_order and opp_order.status == OrderStatus.OPEN:
                crosses = (
                    (side_str == "BUY"  and quote_price >= opp_order.price) or
                    (side_str == "SELL" and quote_price <= opp_order.price)
                )
                if crosses:
                    self._log.warning(
                        "stp_guard",
                        market_id=market_id,
                        side=side_str,
                        quote_price=quote_price,
                        opposite_price=opp_order.price,
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
                neg_risk=neg_risk,
                tick_size=tick_size,
            )

    async def _update_side(
        self,
        market_id: str,
        side_str: str,
        quote_price: float,
        fv: FairValueResult,
        yes_token_id: str,
        order_size_usd: float,
        neg_risk: bool = False,
        tick_size: float = 0.01,
    ) -> None:
        key = (market_id, side_str)
        live = self._live.get(key)

        if live is None or live.status not in (OrderStatus.OPEN, OrderStatus.PENDING):
            # No live order , place new
            await self._place_order(
                market_id, side_str, quote_price, yes_token_id, order_size_usd,
                neg_risk=neg_risk, tick_size=tick_size,
            )
            return

        # Order exists , check if reprice is warranted
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
                market_id, side_str, quote_price, yes_token_id, order_size_usd,
                neg_risk=neg_risk, tick_size=tick_size,
            )

    # Order operations
    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        """L2 HMAC auth headers, or {} if no credentials configured (dev/backtest)."""
        if self._l2_auth is None:
            return {}
        return self._l2_auth.headers(method, path, body)

    async def _place_order(
        self,
        market_id: str,
        side_str: str,
        price: float,
        token_id: str,
        size_usd: float,
        neg_risk: bool = False,
        tick_size: float = 0.01,
    ) -> Optional[ManagedOrder]:
        """Build, sign, and submit a post-only limit order."""
        price = round_to_tick(price, tick_size)

        # Derive contract size from USD notional
        # size (contracts) ~= size_usd / price   [USD / (USD/contract)]
        n_contracts = size_usd / max(price, tick_size)

        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
        params = OrderParams(
            token_id=token_id,
            side=side,
            price=price,
            size=n_contracts,
            neg_risk=neg_risk,
        )

        try:
            signed: SignedOrder = self._signer.sign_order(params)
        except Exception as exc:
            self._log.error("sign_failed", error=str(exc))
            return None

        payload = signed.to_api_dict()
        payload["orderType"] = "GTC"
        # postOnly is a real wire-body flag, not something the exchange
        # infers, rejects instead of executing if the order would cross
        # and take liquidity. Without this a "maker" order can silently
        # execute as taker and eat the taker fee.
        payload["postOnly"] = True

        url = f"{self._rest_url}/order"
        body_str = json.dumps(payload, separators=(",", ":"))
        headers = self._auth_headers("POST", "/order", body_str)
        try:
            async with self._session.post(
                url, data=body_str,
                headers={**headers, "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                if resp.status in (200, 201):
                    body = await resp.json()
                    order_id = body.get("orderID") or body.get("id", "unknown")
                    status, filled_size = self._resolve_placed_status(body, n_contracts)
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
            status=status,
            filled_size=filled_size,
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
            status=status.value,
        )
        return order

    @staticmethod
    def _resolve_placed_status(body: dict, n_contracts: float) -> Tuple[OrderStatus, float]:
        """
        The POST /order response carries a `status` field: "live" (resting,
        our order made it onto the book), "matched" (filled at placement),
        or "delayed" (matching engine hasn't decided yet, e.g. RFQ flow).
        This used to be ignored entirely and every order sat at PENDING
        forever regardless of what actually happened.

        "matched" doesn't come with a documented partial-fill breakdown
        we could find, so we treat it as fully filled and log loudly,
        a post-only order matching at all is already the unexpected case,
        better to flag it than guess at the fill size.
        """
        status = body.get("status", "")
        if status == "live":
            return OrderStatus.OPEN, 0.0
        if status == "matched":
            logger.warning("postonly_order_matched_at_placement", body=str(body)[:200])
            return OrderStatus.FILLED, n_contracts
        if status == "delayed":
            return OrderStatus.PENDING, 0.0
        logger.warning("unrecognized_order_status", status=status)
        return OrderStatus.PENDING, 0.0

    async def _cancel_order(self, order: ManagedOrder) -> bool:
        """Cancel a single order.

        Order id goes in the DELETE body, not the path, since the L2
        HMAC signature is computed over the body. Path-param DELETE
        401s with "Invalid api key" even with correct headers, cost me
        an afternoon figuring that out.
        """
        url = f"{self._rest_url}/order"
        body_str = json.dumps({"orderID": order.order_id}, separators=(",", ":"))
        headers = self._auth_headers("DELETE", "/order", body_str)
        try:
            async with self._session.delete(
                url, data=body_str,
                headers={**headers, "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=2.0),
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

    # Fill handling
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

    # Introspection
    def get_live_quotes(self, market_id: str) -> Dict[str, Optional[ManagedOrder]]:
        return {
            "BUY":  self._live.get((market_id, "BUY")),
            "SELL": self._live.get((market_id, "SELL")),
        }
