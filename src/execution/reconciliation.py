"""
Startup reconciliation. If the process crashed or got redeployed with
resting orders still live on either exchange, we can't just start
quoting from a blank InventoryManager, that's flying with a fuel gauge
stuck at zero.

Strategy is deliberately simple: pull real positions from the venue and
seed InventoryManager with them, then cancel every resting order we
find rather than trying to adopt it back into ManagedOrder state. We
can't recover queue position or partial-fill history for an order that
existed before this process did, and guessing is worse than just
flattening the book and re-quoting fresh on the next strategy tick.

Polymarket positions come from the public Data API (no auth), open
orders from the authenticated CLOB /data/orders read. Kalshi positions
and orders both come from the authenticated portfolio endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
import structlog

from src.execution.kalshi_auth import KalshiRsaSigner
from src.execution.polymarket_auth import PolyL2Auth

logger = structlog.get_logger(__name__)


@dataclass
class ReconciledPosition:
    market_id: str
    net_qty: float     # signed, +long YES
    avg_entry: float


class StartupReconciler:
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._log = logger.bind(component="reconciler")

    # Polymarket
    async def reconcile_polymarket(
        self,
        wallet_address: str,
        data_api_url: str,
        clob_rest_url: str,
        l2_auth: Optional[PolyL2Auth],
        condition_to_mid: Dict[str, str],
    ) -> List[ReconciledPosition]:
        positions = await self._fetch_poly_positions(wallet_address, data_api_url, condition_to_mid)
        if l2_auth is not None:
            await self._cancel_poly_open_orders(clob_rest_url, l2_auth)
        return positions

    async def fetch_poly_balance(self, clob_rest_url: str, l2_auth: PolyL2Auth) -> float:
        """pUSD collateral balance via GET /balance-allowance. Raw units
        are 6-decimal, same as the order amounts we sign."""
        url = f"{clob_rest_url.rstrip('/')}/balance-allowance"
        path = urlparse(url).path
        headers = l2_auth.headers("GET", path)
        try:
            async with self._session.get(
                url, params={"asset_type": "COLLATERAL"}, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error("poly_balance_fetch_failed", status=resp.status)
                    return 0.0
                data = await resp.json()
        except Exception as exc:
            self._log.error("poly_balance_error", error=str(exc))
            return 0.0

        try:
            return float(data.get("balance", 0)) / 1_000_000
        except (TypeError, ValueError):
            self._log.error("poly_balance_unexpected_shape", body=str(data)[:200])
            return 0.0

    async def _fetch_poly_positions(
        self,
        wallet_address: str,
        data_api_url: str,
        condition_to_mid: Dict[str, str],
    ) -> List[ReconciledPosition]:
        url = f"{data_api_url.rstrip('/')}/positions"
        try:
            async with self._session.get(
                url, params={"user": wallet_address, "sizeThreshold": "0"},
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error("poly_positions_fetch_failed", status=resp.status)
                    return []
                raw = await resp.json()
        except Exception as exc:
            self._log.error("poly_positions_error", error=str(exc))
            return []

        # a market can show up as two rows (YES leg, NO leg) if the
        # account holds both, net them into one YES-equivalent qty
        by_market: Dict[str, ReconciledPosition] = {}
        for row in raw:
            condition_id = row.get("conditionId")
            market_id = condition_to_mid.get(condition_id)
            if market_id is None:
                continue

            outcome = str(row.get("outcome", "")).strip().lower()
            size = float(row.get("size", 0))
            avg_price = float(row.get("avgPrice", 0))
            if size == 0:
                continue

            signed_qty  = size if outcome == "yes" else -size
            yes_entry   = avg_price if outcome == "yes" else (1.0 - avg_price)

            existing = by_market.get(market_id)
            if existing is None:
                by_market[market_id] = ReconciledPosition(market_id, signed_qty, yes_entry)
            else:
                # holding both legs at once is unusual (leftover from a
                # merge/split), just net the quantity and keep whichever
                # leg is bigger as the cost basis, good enough for a
                # startup estimate, on_fill will true it up from there
                self._log.warning("poly_position_both_legs", market_id=market_id)
                new_qty = existing.net_qty + signed_qty
                existing.avg_entry = yes_entry if abs(signed_qty) > abs(existing.net_qty) else existing.avg_entry
                existing.net_qty = new_qty

        return list(by_market.values())

    async def _cancel_poly_open_orders(self, clob_rest_url: str, l2_auth: PolyL2Auth) -> None:
        list_url = f"{clob_rest_url.rstrip('/')}/data/orders"
        path = urlparse(list_url).path
        headers = l2_auth.headers("GET", path)
        try:
            async with self._session.get(
                list_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error("poly_open_orders_fetch_failed", status=resp.status)
                    return
                orders = await resp.json()
        except Exception as exc:
            self._log.error("poly_open_orders_error", error=str(exc))
            return

        for order in orders:
            order_id = order.get("id") or order.get("orderID")
            if not order_id:
                continue
            await self._cancel_poly_order(clob_rest_url, l2_auth, order_id)

    async def _cancel_poly_order(self, clob_rest_url: str, l2_auth: PolyL2Auth, order_id: str) -> None:
        import json
        url = f"{clob_rest_url.rstrip('/')}/order"
        body = json.dumps({"orderID": order_id}, separators=(",", ":"))
        headers = l2_auth.headers("DELETE", "/order", body)
        try:
            async with self._session.delete(
                url, data=body,
                headers={**headers, "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                if resp.status == 200:
                    self._log.info("stale_order_cancelled", order_id=order_id, venue="polymarket")
                else:
                    self._log.warning("stale_order_cancel_failed", order_id=order_id, status=resp.status)
        except Exception as exc:
            self._log.error("stale_order_cancel_error", order_id=order_id, error=str(exc))

    # Kalshi
    async def reconcile_kalshi(
        self,
        rest_url: str,
        api_key_id: str,
        rsa_signer: KalshiRsaSigner,
        ticker_to_mid: Dict[str, str],
    ) -> List[ReconciledPosition]:
        positions = await self._fetch_kalshi_positions(rest_url, api_key_id, rsa_signer, ticker_to_mid)
        await self._cancel_kalshi_open_orders(rest_url, api_key_id, rsa_signer)
        return positions

    async def fetch_kalshi_balance(
        self, rest_url: str, api_key_id: str, rsa_signer: KalshiRsaSigner,
    ) -> float:
        """USD balance via GET /portfolio/balance. Field name is another
        one that's shifted with the fixed-point migration, parsed
        defensively same as everywhere else Kalshi touches this repo."""
        url = f"{rest_url.rstrip('/')}/portfolio/balance"
        headers = self._kalshi_headers(api_key_id, rsa_signer, "GET", url)
        try:
            async with self._session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error("kalshi_balance_fetch_failed", status=resp.status)
                    return 0.0
                data = await resp.json()
        except Exception as exc:
            self._log.error("kalshi_balance_error", error=str(exc))
            return 0.0

        if "balance_dollars" in data:
            return float(data["balance_dollars"])
        if "balance" in data:
            return float(data["balance"]) / 100.0   # legacy cents
        self._log.error("kalshi_balance_unexpected_shape", body=str(data)[:200])
        return 0.0

    def _kalshi_headers(self, api_key_id: str, rsa_signer: KalshiRsaSigner, method: str, url: str) -> dict:
        import time as _time
        path = urlparse(url).path
        ts_ms = int(_time.time() * 1000)
        return rsa_signer.headers(api_key_id, ts_ms, method, path)

    async def _fetch_kalshi_positions(
        self,
        rest_url: str,
        api_key_id: str,
        rsa_signer: KalshiRsaSigner,
        ticker_to_mid: Dict[str, str],
    ) -> List[ReconciledPosition]:
        url = f"{rest_url.rstrip('/')}/portfolio/positions"
        headers = self._kalshi_headers(api_key_id, rsa_signer, "GET", url)
        try:
            async with self._session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error("kalshi_positions_fetch_failed", status=resp.status)
                    return []
                data = await resp.json()
        except Exception as exc:
            self._log.error("kalshi_positions_error", error=str(exc))
            return []

        out = []
        # field names here are best-effort, Kalshi's fixed-point migration
        # has been renaming position fields all year, confirm against a
        # live account before trusting this in prod
        for row in data.get("market_positions", []):
            ticker = row.get("ticker")
            market_id = ticker_to_mid.get(ticker)
            if market_id is None:
                continue
            qty = row.get("position_fp", row.get("position"))
            cost = row.get("market_exposure_dollars", row.get("market_exposure"))
            if qty is None:
                continue
            qty = float(qty)
            if qty == 0:
                continue
            avg_entry = abs(float(cost) / qty) if cost is not None and qty != 0 else 0.0
            out.append(ReconciledPosition(market_id, qty, avg_entry))
        return out

    async def _cancel_kalshi_open_orders(
        self, rest_url: str, api_key_id: str, rsa_signer: KalshiRsaSigner,
    ) -> None:
        list_url = f"{rest_url.rstrip('/')}/portfolio/orders"
        headers = self._kalshi_headers(api_key_id, rsa_signer, "GET", list_url)
        try:
            async with self._session.get(
                list_url, headers=headers, params={"status": "resting"},
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error("kalshi_open_orders_fetch_failed", status=resp.status)
                    return
                data = await resp.json()
        except Exception as exc:
            self._log.error("kalshi_open_orders_error", error=str(exc))
            return

        for order in data.get("orders", []):
            order_id = order.get("order_id")
            if not order_id:
                continue
            cancel_url = f"{rest_url.rstrip('/')}/portfolio/orders/{order_id}"
            cancel_headers = self._kalshi_headers(api_key_id, rsa_signer, "DELETE", cancel_url)
            try:
                async with self._session.delete(
                    cancel_url, headers=cancel_headers,
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    if resp.status in (200, 204):
                        self._log.info("stale_order_cancelled", order_id=order_id, venue="kalshi")
                    else:
                        self._log.warning("stale_order_cancel_failed", order_id=order_id, status=resp.status)
            except Exception as exc:
                self._log.error("stale_order_cancel_error", order_id=order_id, error=str(exc))
