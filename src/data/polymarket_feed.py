"""
Polymarket CLOB websocket connector (V2 protocol, post 2026-04-28 migration).

Each market has two token IDs (YES/NO), each with its own order book, we
subscribe to both. Host is wss://ws-subscriptions-clob.polymarket.com/ws/{market|user}.

The "market" channel is public, no auth, don't bother sending one. There's
no {"type": "auth", ...} message, it doesn't exist, this tripped us up
during the V2 migration when we copied the old auth flow over by habit.
"user" channel auth (needed for our own fills) rides inline in the
subscribe payload, and it's the raw api secret, not an HMAC.

Messages we care about: book (snapshot), price_change (delta, nested
under price_changes), last_trade_price, tick_size_change. We also send a
literal "PING" text frame every ~10s on top of protocol pings since some
proxies in front of the CLOB only honor application-level pings.

Both books get normalized into the shared YES/NO structure consumed by
UnifiedBook.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple

import structlog

from src.data.base_feed import BaseFeed, RawMessage

logger = structlog.get_logger(__name__)


# Domain types
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


# Market Metadata Cache
@dataclass
class PolyMarket:
    condition_id: str
    yes_token_id: str
    no_token_id:  str
    question:     str = ""
    end_date_iso: str = ""
    neg_risk:     bool = False


# Feed
class PolymarketFeed(BaseFeed):
    """
    Subscribes to YES and NO token books for each tracked condition_id
    on the public "market" channel (no auth needed).

    Emits:  PolyBookSnapshot | PolyPriceDelta | PolyTrade
    """

    STALE_FEED_TIMEOUT_S = 15.0
    # Polymarket expects a client-side "PING" text frame roughly every
    # 10s in addition to protocol-level WS pings (some proxies in the
    # path only recognise the application-level frame).
    APP_PING_INTERVAL_S = 10.0

    def __init__(
        self,
        ws_url: str,
        markets: List[PolyMarket],
        out_queue,
        health_queue=None,
        # Only needed if/when subscribing to the "user" channel for fills.
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
    ):
        super().__init__(ws_url, "polymarket", out_queue, health_queue)
        self._markets = {m.condition_id: m for m in markets}
        self._api_key        = api_key
        self._api_secret     = api_secret
        self._api_passphrase = api_passphrase

        # Build reverse map: token_id → (condition_id, is_yes)
        self._token_map: Dict[str, Tuple[str, bool]] = {}
        for m in markets:
            self._token_map[m.yes_token_id] = (m.condition_id, True)
            self._token_map[m.no_token_id]  = (m.condition_id, False)

        self._log = logger.bind(venue="polymarket", n_markets=len(markets))
        self._ping_task: Optional[asyncio.Task] = None

    # BaseFeed interface
    async def _build_subscribe_msgs(self) -> list:
        """
        Public "market" channel subscribe , no auth. One message covering
        every YES/NO token we track.
        """
        token_ids = list(self._token_map.keys())
        sub_msg = {
            "assets_ids": token_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
        return [json.dumps(sub_msg)]

    async def _connect_and_consume(self) -> None:
        """Wrap base implementation to also run the app-level PING loop."""
        self._ping_task = asyncio.ensure_future(self._app_ping_loop())
        try:
            await super()._connect_and_consume()
        finally:
            if self._ping_task:
                self._ping_task.cancel()

    async def _app_ping_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(self.APP_PING_INTERVAL_S)
            if self._ws is not None:
                try:
                    await self._ws.send("PING")
                except Exception:
                    return  # connection is going down; outer loop will reconnect

    async def _parse_message(self, raw: RawMessage) -> AsyncIterator:
        if raw.payload in (b"PONG", b"PING"):
            return
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

        elif msg_type == "tick_size_change":
            self._log.info(
                "tick_size_change",
                asset_id=msg.get("asset_id"),
                old=msg.get("old_tick_size"),
                new=msg.get("new_tick_size"),
            )

        elif msg_type in ("subscribed",):
            pass  # control message

        else:
            self._log.debug("unknown_msg_type", msg_type=msg_type)

    # Parsers
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
                    p = float(lvl["price"])
                    # Polymarket book levels are already probability-scaled
                    # strings like "0.42", not cents , normalise defensively
                    # in case a venue update reverts to cents-style values.
                    if p > 1.0:
                        p = p / 100.0
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

        if bids and asks and bids[0].price >= asks[0].price:
            self._log.error(
                "crossed_book",
                condition_id=condition_id,
                is_yes=is_yes,
                best_bid=bids[0].price,
                best_ask=asks[0].price,
            )

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
        """
        Real payload nests changes under `price_changes`, one entry per
        (asset_id, side, price) update , NOT a flat top-level `changes`
        list against a single asset_id as the old V1-era assumption had it.
        """
        changes = msg.get("price_changes", msg.get("changes", []))
        ts_ms = int(msg.get("timestamp", 0) or 0)

        for ch in changes:
            asset_id = ch.get("asset_id", msg.get("asset_id"))
            if asset_id not in self._token_map:
                continue
            condition_id, is_yes = self._token_map[asset_id]
            try:
                side  = ch["side"].upper()
                price = float(ch["price"])
                if price > 1.0:
                    price = price / 100.0
                size  = float(ch["size"])
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
            price = float(msg["price"])
            if price > 1.0:
                price = price / 100.0
            yield PolyTrade(
                condition_id=condition_id,
                token_id=asset_id,
                is_yes_token=is_yes,
                price=price,
                size=float(msg.get("size", 0)),
                side=msg.get("side", "").upper(),
                trade_id=str(msg.get("id", "")),
                timestamp_ms=int(msg.get("timestamp", 0)),
                recv_ts=recv_ts,
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log.warning("bad_trade", error=str(exc))


# User Feed: own order lifecycle + fills
@dataclass
class PolyOwnFill:
    """A fill on one of OUR own resting orders."""
    condition_id: str
    token_id: str
    order_id: str
    price: float
    size: float           # size of THIS fill (incremental, not cumulative)
    side: str              # "BUY" | "SELL"
    trade_id: str
    timestamp_ms: int
    recv_ts: float


@dataclass
class PolyOwnOrderUpdate:
    """Lifecycle event for one of our own orders (placed/cancelled/etc)."""
    order_id: str
    condition_id: str
    token_id: str
    status: str            # "LIVE" | "MATCHED" | "CANCELLED" | ...
    timestamp_ms: int
    recv_ts: float


class PolymarketUserFeed(BaseFeed):
    """
    Subscribes to the authenticated "user" channel, keyed by condition_id
    (NOT token_id , unlike the market channel). Requires L2 credentials.

    Auth is carried inline in the subscribe payload as the RAW api secret
    (not an HMAC signature):
        {"type": "user", "markets": [condition_id, ...],
         "auth": {"apiKey", "secret", "passphrase"}}

    Emits: PolyOwnFill | PolyOwnOrderUpdate
    """

    STALE_FEED_TIMEOUT_S = 30.0  # user channel can be quiet for long stretches

    def __init__(
        self,
        ws_url: str,
        condition_ids: List[str],
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        out_queue,
        health_queue=None,
    ):
        super().__init__(ws_url, "polymarket_user", out_queue, health_queue)
        self._condition_ids   = condition_ids
        self._api_key         = api_key
        self._api_secret      = api_secret
        self._api_passphrase  = api_passphrase
        self._log = logger.bind(venue="polymarket_user", n_markets=len(condition_ids))

    async def _build_subscribe_msgs(self) -> list:
        sub_msg = {
            "type": "user",
            "markets": self._condition_ids,
            "auth": {
                "apiKey": self._api_key,
                "secret": self._api_secret,       # raw secret, NOT HMAC-signed
                "passphrase": self._api_passphrase,
            },
        }
        return [json.dumps(sub_msg)]

    async def _parse_message(self, raw: RawMessage) -> AsyncIterator:
        if raw.payload in (b"PONG", b"PING"):
            return
        try:
            msg = json.loads(raw.payload)
        except json.JSONDecodeError:
            self._log.warning("bad_json", snippet=raw.payload[:120])
            return

        event_type = msg.get("event_type") or msg.get("type")

        if event_type == "trade":
            for ev in self._parse_own_fill(msg, raw.recv_ts):
                yield ev
        elif event_type == "order":
            for ev in self._parse_own_order(msg, raw.recv_ts):
                yield ev
        elif event_type in ("subscribed",):
            pass
        else:
            self._log.debug("unknown_user_event", event_type=event_type)

    def _parse_own_fill(self, msg: dict, recv_ts: float):
        try:
            price = float(msg["price"])
            if price > 1.0:
                price = price / 100.0
            yield PolyOwnFill(
                condition_id=msg.get("market", ""),
                token_id=str(msg.get("asset_id", "")),
                order_id=str(msg.get("order_id") or msg.get("id", "")),
                price=price,
                size=float(msg.get("size", 0)),
                side=str(msg.get("side", "")).upper(),
                trade_id=str(msg.get("trade_id") or msg.get("id", "")),
                timestamp_ms=int(msg.get("timestamp", 0) or 0),
                recv_ts=recv_ts,
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log.warning("bad_own_fill", error=str(exc))

    def _parse_own_order(self, msg: dict, recv_ts: float):
        try:
            yield PolyOwnOrderUpdate(
                order_id=str(msg.get("id") or msg.get("order_id", "")),
                condition_id=msg.get("market", ""),
                token_id=str(msg.get("asset_id", "")),
                status=str(msg.get("status", "")).upper(),
                timestamp_ms=int(msg.get("timestamp", 0) or 0),
                recv_ts=recv_ts,
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log.warning("bad_own_order", error=str(exc))
