"""
src/data/polymarket_feed.py
────────────────────────────
Async Polymarket CLOB WebSocket connector.

Polymarket CLOB specifics:
  - Market = (condition_id) → two token IDs: YES token, NO token
  - Separate order books per token. We subscribe to both.
  - WS channels: "market" (L2 book) + "user" (own orders/fills)
  - Message types: book (snapshot), price_change (delta), last_trade_price
  - Auth header: HMAC-SHA256 of (timestamp + method + path + body) via API key

This connector normalises both books into a unified YES/NO structure
consumed by the UnifiedBook engine in unified_book.py.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

import structlog

from src.data.base_feed import BaseFeed, RawMessage

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────

@dataclass(slots=True)
class PriceLevel:
    price: float
    size: float


@dataclass
class PolyBookSnapshot:
    """
    Full L2 snapshot for a single Polymarket token (YES or NO).
    Prices are in cents [0, 100] → divide by 100 for probabilities.
    """
    condition_id: str
    token_id: str
    is_yes_token: bool
    bids: List[PriceLevel]   # sorted descending
    asks: List[PriceLevel]   # sorted ascending
    timestamp_ms: int
    recv_ts: float           # wall-clock monotonic at reception


@dataclass
class PolyPriceDelta:
    """Incremental update to a single level."""
    condition_id: str
    token_id: str
    is_yes_token: bool
    side: str            # "BUY" | "SELL"
    price: float
    size: float          # 0.0 = level removed
    timestamp_ms: int
    recv_ts: float


@dataclass
class PolyTrade:
    condition_id: str
    token_id: str
    is_yes_token: bool
    price: float
    size: float
    side: str           # aggressor side
    trade_id: str
    timestamp_ms: int
    recv_ts: float


# ──────────────────────────────────────────────
# Market Metadata Cache
# ──────────────────────────────────────────────

@dataclass
class PolyMarket:
    condition_id: str
    yes_token_id: str
    no_token_id:  str
    question:     str = ""
    end_date_iso: str = ""


# ──────────────────────────────────────────────
# Feed
# ──────────────────────────────────────────────

class PolymarketFeed(BaseFeed):
    """
    Subscribes to YES and NO token books for each tracked condition_id.
    Emits:  PolyBookSnapshot | PolyPriceDelta | PolyTrade
    """

    # Polymarket CLOB WS: no sequence numbers at the WS layer,
    # but book snapshots carry a hash we can checksum.
    STALE_FEED_TIMEOUT_S = 15.0

    def __init__(
        self,
        ws_url: str,
        markets: List[PolyMarket],
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        out_queue,
        health_queue=None,
    ):
        super().__init__(ws_url, "polymarket", out_queue, health_queue)
        self._markets = {m.condition_id: m for m in markets}
        self._api_key        = api_key
        self._api_secret     = api_secret.encode()
        self._api_passphrase = api_passphrase

        # Build reverse map: token_id → (condition_id, is_yes)
        self._token_map: Dict[str, Tuple[str, bool]] = {}
        for m in markets:
            self._token_map[m.yes_token_id] = (m.condition_id, True)
            self._token_map[m.no_token_id]  = (m.condition_id, False)

        self._log = logger.bind(venue="polymarket", n_markets=len(markets))

    # ── BaseFeed interface ────────────────────

    async def _build_subscribe_msgs(self) -> list:
        """
        Polymarket WS auth + subscribe to all tracked token books.
        Auth schema: L1 header via HMAC-SHA256.
        """
        ts     = str(int(time.time() * 1000))
        nonce  = ""
        sig    = self._sign(ts, "GET", "/ws/market", nonce)

        auth_msg = {
            "type": "auth",
            "apiKey": self._api_key,
            "secret": sig,
            "passphrase": self._api_passphrase,
            "timestamp": ts,
        }

        # One subscription per token (YES + NO per market)
        token_ids = list(self._token_map.keys())
        sub_msg = {
            "type": "subscribe",
            "channels": ["market"],
            "assets_ids": token_ids,
        }

        return [json.dumps(auth_msg), json.dumps(sub_msg)]

    async def _parse_message(self, raw: RawMessage) -> AsyncIterator:
        try:
            msg = json.loads(raw.payload)
        except json.JSONDecodeError:
            self._log.warning("bad_json", snippet=raw.payload[:120])
            return

        msg_type = msg.get("event_type") or msg.get("type")

        if msg_type == "book":
            for ev in self._parse_book_snapshot(msg, raw.recv_ts):
                yield ev

        elif msg_type == "price_change":
            for ev in self._parse_price_change(msg, raw.recv_ts):
                yield ev

        elif msg_type == "last_trade_price":
            for ev in self._parse_trade(msg, raw.recv_ts):
                yield ev

        elif msg_type in ("subscribed", "auth", "pong"):
            pass  # control messages

        else:
            self._log.debug("unknown_msg_type", msg_type=msg_type)

    # ── Parsers ───────────────────────────────

    def _parse_book_snapshot(self, msg: dict, recv_ts: float):
        asset_id = msg.get("asset_id")
        if asset_id not in self._token_map:
            return

        condition_id, is_yes = self._token_map[asset_id]
        ts_ms = int(msg.get("timestamp", 0) or 0)

        def parse_levels(raw_levels: list) -> List[PriceLevel]:
            levels = []
            for lvl in (raw_levels or []):
                try:
                    p = float(lvl["price"]) / 100.0   # cents → probability
                    s = float(lvl["size"])
                    if s > 0:
                        levels.append(PriceLevel(price=p, size=s))
                except (KeyError, ValueError, TypeError):
                    continue
            return levels

        bids = sorted(
            parse_levels(msg.get("bids", [])),
            key=lambda x: -x.price,
        )
        asks = sorted(
            parse_levels(msg.get("asks", [])),
            key=lambda x: x.price,
        )

        # Sanity: crossed book is a data error
        if bids and asks and bids[0].price >= asks[0].price:
            self._log.error(
                "crossed_book",
                condition_id=condition_id,
                is_yes=is_yes,
                best_bid=bids[0].price,
                best_ask=asks[0].price,
            )
            # Emit anyway; UnifiedBook will handle/discard
        
        yield PolyBookSnapshot(
            condition_id=condition_id,
            token_id=asset_id,
            is_yes_token=is_yes,
            bids=bids,
            asks=asks,
            timestamp_ms=ts_ms,
            recv_ts=recv_ts,
        )

    def _parse_price_change(self, msg: dict, recv_ts: float):
        changes = msg.get("changes", [])
        asset_id = msg.get("asset_id")
        if asset_id not in self._token_map:
            return

        condition_id, is_yes = self._token_map[asset_id]
        ts_ms = int(msg.get("timestamp", 0) or 0)

        for ch in changes:
            try:
                side  = ch["side"].upper()   # "BUY" | "SELL"
                price = float(ch["price"]) / 100.0
                size  = float(ch["size"])    # 0 = remove level
            except (KeyError, ValueError, TypeError):
                continue

            yield PolyPriceDelta(
                condition_id=condition_id,
                token_id=asset_id,
                is_yes_token=is_yes,
                side=side,
                price=price,
                size=size,
                timestamp_ms=ts_ms,
                recv_ts=recv_ts,
            )

    def _parse_trade(self, msg: dict, recv_ts: float):
        asset_id = msg.get("asset_id")
        if asset_id not in self._token_map:
            return

        condition_id, is_yes = self._token_map[asset_id]

        try:
            yield PolyTrade(
                condition_id=condition_id,
                token_id=asset_id,
                is_yes_token=is_yes,
                price=float(msg["price"]) / 100.0,
                size=float(msg.get("size", 0)),
                side=msg.get("side", "").upper(),
                trade_id=str(msg.get("id", "")),
                timestamp_ms=int(msg.get("timestamp", 0)),
                recv_ts=recv_ts,
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log.warning("bad_trade", error=str(exc))

    # ── Auth ─────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str) -> str:
        """
        Polymarket CLOB HMAC-SHA256 signature.
        message = timestamp + method + path + body
        """
        message = timestamp + method + path + body
        sig = hmac.new(self._api_secret, message.encode(), hashlib.sha256).hexdigest()
        return sig
