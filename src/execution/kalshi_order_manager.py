"""
Order execution for Kalshi. Mirrors the cancel/replace and flickering
logic in order_manager.py (Polymarket), but Kalshi's wire format is
different enough that it isn't worth forcing into the same class:
fixed-point dollar strings instead of raw EVM integers, RSA-PSS request
signing instead of EIP-712, and a "bid"/"ask" side instead of BUY/SELL
token amounts.

Endpoints (V2, verified against docs.kalshi.com):
  POST   /portfolio/events/orders        create
  DELETE /portfolio/orders/{order_id}    cancel (still on the old path,
                                          the V2 migration only touched
                                          order creation)

We always trade on the YES leg. `side: "bid"` buys YES, `side: "ask"`
sells YES (selling YES is economically the same as buying NO at
1-price, but the API quotes everything in YES terms so we never have
to build a NO-side order ourselves).
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import structlog

from config.settings import RiskProfile
from src.data.unified_book import MarketState
from src.execution.kalshi_auth import KalshiRsaSigner
from src.execution.order_types import FlickeringFilter, ManagedOrder, OrderSideStr, OrderStatus
from src.pricing.fair_value import FairValueResult

logger = structlog.get_logger(__name__)


class KalshiOrderManager:
    MIN_REPRICE_INTERVAL_S: float = 0.050
    REPRICE_THRESHOLD_FRAC: float = 0.30

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        api_key_id: str,
        rsa_signer: KalshiRsaSigner,
        risk_profile: RiskProfile,
        rest_url: str,
        flickering_filter: Optional[FlickeringFilter] = None,
    ):
        self._session   = http_session
        self._api_key   = api_key_id
        self._rsa       = rsa_signer
        self._risk      = risk_profile
        self._rest_url  = rest_url.rstrip("/")
        self._flicker   = flickering_filter or FlickeringFilter(
            window_ms=risk_profile.flickering_window_ms,
            cancel_threshold=risk_profile.flickering_cancel_threshold,
            freeze_ms=risk_profile.toxic_flow_pause_ms,
        )

        self._live: Dict[Tuple[str, str], ManagedOrder] = {}
        self._all: Dict[str, ManagedOrder] = {}

        self._log = logger.bind(component="kalshi_order_manager")

    async def update_quotes(
        self,
        state: MarketState,
        fv: FairValueResult,
        ticker: str,
        inventory_q: float,
        order_size_usd: float = 50.0,
    ) -> None:
        market_id = state.market_id

        if not fv.should_quote:
            await self._cancel_all(market_id, reason="should_not_quote")
            return

        if abs(inventory_q) >= self._risk.max_inventory_contracts:
            await self._handle_max_inventory(market_id, ticker, inventory_q)
            return

        for side_str, quote_price in [
            ("BUY",  fv.bid_quote),
            ("SELL", fv.ask_quote),
        ]:
            opposite = "SELL" if side_str == "BUY" else "BUY"
            opp_order = self._live.get((market_id, opposite))
            if opp_order and opp_order.status == OrderStatus.OPEN:
                if side_str == "BUY" and quote_price >= opp_order.price:
                    self._log.warning("stp_guard", market_id=market_id, bid=quote_price, ask=opp_order.price)
                    continue

            if self._flicker.is_frozen(market_id, side_str):
                self._log.debug("quote_frozen_by_flicker", market_id=market_id, side=side_str)
                continue

            await self._update_side(market_id, side_str, quote_price, fv, ticker, order_size_usd)

    async def _update_side(
        self,
        market_id: str,
        side_str: str,
        quote_price: float,
        fv: FairValueResult,
        ticker: str,
        order_size_usd: float,
    ) -> None:
        key = (market_id, side_str)
        live = self._live.get(key)

        if live is None:
            await self._place_order(market_id, side_str, quote_price, ticker, order_size_usd)
            return

        age_s = time.monotonic() - live.placed_ts
        if age_s < self.MIN_REPRICE_INTERVAL_S:
            return

        price_move = abs(quote_price - live.price)
        spread = fv.half_spread * 2
        if spread > 0 and price_move / spread < self.REPRICE_THRESHOLD_FRAC:
            return

        self._log.info(
            "repricing",
            market_id=market_id,
            side=side_str,
            old_price=round(live.price, 4),
            new_price=round(quote_price, 4),
        )
        cancelled = await self._cancel_order(live)
        if cancelled:
            await self._place_order(market_id, side_str, quote_price, ticker, order_size_usd)

    # Order operations
    def _auth_headers(self, method: str, url: str) -> dict:
        path = urlparse(url).path
        ts_ms = int(time.time() * 1000)
        return self._rsa.headers(self._api_key, ts_ms, method, path)

    async def _place_order(
        self,
        market_id: str,
        side_str: str,
        price: float,
        ticker: str,
        size_usd: float,
    ) -> Optional[ManagedOrder]:
        n_contracts = round(size_usd / max(price, 0.01), 2)
        if n_contracts <= 0:
            return None

        wire_side = "bid" if side_str == "BUY" else "ask"
        payload = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": wire_side,
            "count": f"{n_contracts:.2f}",
            "price": f"{price:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": True,
        }

        url = f"{self._rest_url}/portfolio/events/orders"
        body_str = json.dumps(payload, separators=(",", ":"))
        headers = self._auth_headers("POST", url)
        try:
            async with self._session.post(
                url, data=body_str,
                headers={**headers, "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                if resp.status == 201:
                    body = await resp.json()
                    order_id = body.get("order_id", "unknown")
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
            token_id=ticker,
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
            size=n_contracts,
        )
        return order

    async def _cancel_order(self, order: ManagedOrder) -> bool:
        url = f"{self._rest_url}/portfolio/orders/{order.order_id}"
        headers = self._auth_headers("DELETE", url)
        try:
            async with self._session.delete(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                if resp.status in (200, 204):
                    order.status = OrderStatus.CANCELLED
                    self._flicker.record_cancel(order.market_id, order.side.value, time.monotonic())
                    del self._live[(order.market_id, order.side.value)]
                    self._log.info("order_cancelled", order_id=order.order_id, market_id=order.market_id)
                    return True
                else:
                    text = await resp.text()
                    self._log.warning("cancel_failed", status=resp.status, body=text[:200], order_id=order.order_id)
                    return False
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            self._log.error("cancel_error", error=str(exc), order_id=order.order_id)
            return False

    async def _cancel_all(self, market_id: str, reason: str = "") -> None:
        to_cancel = [
            order for (mid, _), order in self._live.items()
            if mid == market_id and order.status in (OrderStatus.OPEN, OrderStatus.PENDING)
        ]
        if to_cancel:
            self._log.info("cancel_all", market_id=market_id, n=len(to_cancel), reason=reason)
            await asyncio.gather(*[self._cancel_order(o) for o in to_cancel])

    async def _handle_max_inventory(self, market_id: str, ticker: str, inventory_q: float) -> None:
        cancel_side = "BUY" if inventory_q > 0 else "SELL"
        key = (market_id, cancel_side)
        if key in self._live:
            await self._cancel_order(self._live[key])

    # Fill handling
    def mark_filled(self, order_id: str, fill_size: float) -> Optional[ManagedOrder]:
        order = self._all.get(order_id)
        if order is None:
            return None
        order.filled_size += fill_size
        order.last_update_ts = time.monotonic()
        if order.filled_size >= order.size * 0.999:
            order.status = OrderStatus.FILLED
            self._live.pop((order.market_id, order.side.value), None)
        else:
            order.status = OrderStatus.PARTIAL_FILL
        return order

    def get_live_quotes(self, market_id: str) -> Dict[str, Optional[ManagedOrder]]:
        return {
            "BUY":  self._live.get((market_id, "BUY")),
            "SELL": self._live.get((market_id, "SELL")),
        }
