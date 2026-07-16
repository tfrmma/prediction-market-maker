"""
Mid price + realized vol poller for Hyperliquid perps.

A proper websocket feed would be lower latency, but the hedge engine
only recomputes once per quoting cycle anyway (see main.py's
STRATEGY_LOOP_INTERVAL_S), not on every tick, so polling POST /info
{"type": "allMids"} once a second is plenty and a lot less to maintain.
This replaces the hardcoded S_perp/sigma_perp constants that used to
sit in main.py, which meant the hedge was sizing itself off numbers
that had nothing to do with the actual market.

TODO: switch to the websocket allMids subscription if hedge latency
ever actually becomes the bottleneck, right now it isn't.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class HyperliquidPriceFeed:
    POLL_INTERVAL_S = 1.0
    VOL_WINDOW_S = 3600.0   # 1h realized vol, annualized from log returns

    def __init__(self, info_url: str, coins: List[str]):
        self._url = info_url.rstrip("/") + "/info"
        self._coins = coins
        self._mids: Dict[str, float] = {}
        self._history: Dict[str, Deque[Tuple[float, float]]] = {c: deque() for c in coins}
        self._asset_index: Dict[str, int] = {}
        self._sz_decimals: Dict[str, int] = {}
        self._log = logger.bind(component="hl_price_feed")
        self._session: Optional[aiohttp.ClientSession] = None

    def get_mid(self, coin: str) -> Optional[float]:
        return self._mids.get(coin)

    def get_asset_index(self, coin: str) -> Optional[int]:
        return self._asset_index.get(coin)

    def get_sz_decimals(self, coin: str) -> int:
        return self._sz_decimals.get(coin, 3)   # 3 is a reasonable perp default

    def get_realized_vol(self, coin: str) -> Optional[float]:
        """Annualized realized vol from log returns, needs >=10 samples in the window."""
        hist = self._history.get(coin)
        if hist is None or len(hist) < 10:
            return None

        prices = [p for _, p in hist]
        rets = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]
        if len(rets) < 5:
            return None

        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        periods_per_year = (365.25 * 24 * 3600) / self.POLL_INTERVAL_S
        return math.sqrt(var * periods_per_year)

    async def fetch_meta(self) -> None:
        """Pull the perp universe once at startup: coin -> asset index, szDecimals.
        Needed because order placement wants a numeric asset index, not a name."""
        session = self._session or aiohttp.ClientSession()
        try:
            async with session.post(
                self._url, json={"type": "meta"},
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                data = await resp.json()
        finally:
            if self._session is None:
                await session.close()

        universe = data.get("universe", [])
        for idx, entry in enumerate(universe):
            name = entry.get("name")
            if name in self._coins:
                self._asset_index[name] = idx
                self._sz_decimals[name] = int(entry.get("szDecimals", 3))

        missing = set(self._coins) - set(self._asset_index)
        if missing:
            self._log.warning("hl_coins_not_in_universe", missing=list(missing))

    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        try:
            await self.fetch_meta()
            while True:
                await self._poll_once()
                await asyncio.sleep(self.POLL_INTERVAL_S)
        finally:
            await self._session.close()

    async def _poll_once(self) -> None:
        try:
            async with self._session.post(
                self._url, json={"type": "allMids"},
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                data = await resp.json()
        except Exception as exc:
            self._log.warning("hl_poll_failed", error=str(exc))
            return

        now = time.monotonic()
        for coin in self._coins:
            raw = data.get(coin)
            if raw is None:
                continue
            try:
                price = float(raw)
            except (TypeError, ValueError):
                continue

            self._mids[coin] = price
            hist = self._history[coin]
            hist.append((now, price))
            cutoff = now - self.VOL_WINDOW_S
            while hist and hist[0][0] < cutoff:
                hist.popleft()
