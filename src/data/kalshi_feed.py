"""
Kalshi API v2 websocket connector.

Big gotcha: the orderbook is bids-only on both legs. Kalshi never sends
an ask array, full stop. A YES ask is just 1 - best NO bid and vice
versa. We only store raw bid data here per leg, the complement math
lives in UnifiedBook next to the equivalent Polymarket YES/NO logic.

Auth is RSA-PSS(SHA-256) over (timestamp_ms + method + path), sent as
KALSHI-ACCESS-* headers on the websocket handshake itself. There's no
in-band auth message like some other venues use.

Prices come in as fixed-point dollar strings ("0.4200"), not cents, after
Kalshi's 2026 fixed-point migration. Field names have moved around more
than once this year (ticker_v2 retirement, the fixed-point switch, the
new get_snapshot resync action) so treat anything below as best-effort
and diff it against a live capture before trusting it in prod.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import structlog

from src.data.base_feed import BaseFeed, RawMessage

logger = structlog.get_logger(__name__)


# Domain types
@dataclass(slots=True)
class KalshiLevel:
    price: float   # probability [0, 1]
    size: float    # contracts (fixed-point, can be fractional)


@dataclass
class KalshiBookSnapshot:
    """
    Raw bids-only snapshot for BOTH legs of one Kalshi market.
    There is no ask data here by design , see module docstring.
    """
    market_ticker: str
    yes_bids: List[KalshiLevel]   # sorted desc (best first)
    no_bids:  List[KalshiLevel]   # sorted desc (best first)
    seq: int
    timestamp_ms: int
    recv_ts: float


@dataclass
class KalshiBookDelta:
    market_ticker: str
    side: str         # "yes" | "no" , which BID book this delta applies to
    price: float      # [0, 1]
    delta: float      # +/- contracts (fixed-point)
    seq: int
    timestamp_ms: int
    recv_ts: float


@dataclass
class KalshiTrade:
    market_ticker: str
    yes_price: float
    size: float
    taker_side: str   # "yes" | "no"
    trade_id: str
    timestamp_ms: int
    recv_ts: float


@dataclass
class KalshiTicker:
    market_ticker: str
    yes_bid: float
    yes_ask: float          # as reported by ticker channel (may be derived server-side)
    last_price: float
    volume_24h: int
    open_interest: int
    timestamp_ms: int
    recv_ts: float


# Feed
class KalshiFeed(BaseFeed):
    """
    Kalshi WS feed.  Emits:
      KalshiBookSnapshot | KalshiBookDelta | KalshiTrade | KalshiTicker
    """

    STALE_FEED_TIMEOUT_S = 12.0
    WS_AUTH_PATH = "/trade-api/ws/v2"

    def __init__(
        self,
        ws_url: str,
        tickers: List[str],              # e.g. ["BTC-23DEC-T100K"]
        api_key_id: str,
        private_key_pem: str,            # RSA PKCS8 PEM string
        out_queue,
        health_queue=None,
    ):
        super().__init__(ws_url, "kalshi", out_queue, health_queue)
        self._tickers        = tickers
        self._api_key_id     = api_key_id
        self._private_key    = self._load_rsa_key(private_key_pem)
        self._cmd_id         = 0
        self._seq_by_market: Dict[str, int] = {}
        self._log = logger.bind(venue="kalshi", n_markets=len(tickers))

    # BaseFeed interface
    def _extra_ws_headers(self) -> dict:
        """
        Kalshi authenticates the WS *handshake* via signed HTTP headers,
        not via an in-band message. Timestamp must be fresh, so this is
        computed on every (re)connect.
        """
        ts_ms = int(time.time() * 1000)
        sig = self._sign_rsa(ts_ms, "GET", self.WS_AUTH_PATH)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        }

    async def _build_subscribe_msgs(self) -> list:
        """
        Kalshi WS v2 subscribe. Auth already happened at the handshake
        (see _extra_ws_headers); no auth command is sent here.
        """
        sub = {
            "id": self._next_cmd_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "ticker", "trade"],
                "market_tickers": self._tickers,
            },
        }
        return [json.dumps(sub)]

    async def _parse_message(self, raw: RawMessage) -> AsyncIterator:
        try:
            msg = json.loads(raw.payload)
        except json.JSONDecodeError:
            self._log.warning("bad_json", snippet=raw.payload[:120])
            return

        msg_type = msg.get("type", "")
        data     = msg.get("msg", {})

        if msg_type == "orderbook_snapshot":
            for ev in self._parse_book_snapshot(data, raw.recv_ts):
                yield ev

        elif msg_type == "orderbook_delta":
            for ev in self._parse_book_delta(data, raw.recv_ts):
                yield ev

        elif msg_type == "trade":
            for ev in self._parse_trade(data, raw.recv_ts):
                yield ev

        elif msg_type == "ticker":
            for ev in self._parse_ticker(data, raw.recv_ts):
                yield ev

        elif msg_type in ("subscribed", "ok"):
            pass

        elif msg_type == "error":
            self._log.error("kalshi_error", detail=data)
            code = data.get("code")
            if code == 9:  # "Authentication required" , headers rejected/expired
                raise RuntimeError("Kalshi WS auth rejected , check signed headers")

        else:
            self._log.debug("unknown_type", msg_type=msg_type)

    def _extract_sequence(self, payload: bytes) -> Optional[int]:
        try:
            msg = json.loads(payload)
            seq = msg.get("seq")
            return int(seq) if seq is not None else None
        except Exception:
            return None

    def _extract_exchange_ts(self, payload: bytes) -> Optional[int]:
        try:
            msg = json.loads(payload)
            m = msg.get("msg") or {}
            # Kalshi has been migrating fields towards *_ts_ms; fall back
            # to legacy "ts" (ISO string or epoch) if present.
            for key in ("ts_ms", "created_ts_ms"):
                if key in m and isinstance(m[key], (int, float)):
                    return int(m[key])
            ts = m.get("ts")
            if isinstance(ts, (int, float)):
                return int(ts)
            return None
        except Exception:
            return None

    # Parsers
    @staticmethod
    def _parse_fp_levels(raw: list) -> List[KalshiLevel]:
        """
        Parse a Kalshi fixed-point [price_dollars_str, count_fp_str] level
        array. Bids only , arrays are one-sided by construction.
        """
        out = []
        for entry in (raw or []):
            try:
                price = float(entry[0])
                size  = float(entry[1])
                if size > 0:
                    out.append(KalshiLevel(price=price, size=size))
            except (IndexError, ValueError, TypeError):
                continue
        return out

    def _parse_book_snapshot(self, data: dict, recv_ts: float):
        ticker = data.get("market_ticker")
        if ticker not in self._tickers:
            return

        seq   = int(data.get("seq", 0))
        ts_ms = int(data.get("ts_ms", data.get("ts", 0)) or 0)
        self._seq_by_market[ticker] = seq

        # Field names have shifted with Kalshi's fixed-point migration;
        # accept both the *_dollars_fp names and older bare yes/no.
        # TODO: confirm yes_dollars_fp/no_dollars_fp against a live capture,
        # docs.kalshi.com has changed this schema twice already this year
        yes_raw = data.get("yes_dollars_fp") or data.get("yes") or []
        no_raw  = data.get("no_dollars_fp")  or data.get("no")  or []

        yes_bids = sorted(self._parse_fp_levels(yes_raw), key=lambda x: -x.price)
        no_bids  = sorted(self._parse_fp_levels(no_raw),  key=lambda x: -x.price)

        yield KalshiBookSnapshot(
            market_ticker=ticker,
            yes_bids=yes_bids,
            no_bids=no_bids,
            seq=seq,
            timestamp_ms=ts_ms,
            recv_ts=recv_ts,
        )

    def _parse_book_delta(self, data: dict, recv_ts: float):
        ticker = data.get("market_ticker")
        if ticker not in self._tickers:
            return

        seq   = int(data.get("seq", 0))
        ts_ms = int(data.get("ts_ms", data.get("ts", 0)) or 0)

        # Validate sequence continuity per-market
        last_seq = self._seq_by_market.get(ticker)
        if last_seq is not None and seq != last_seq + 1:
            self._log.error(
                "kalshi_seq_gap",
                ticker=ticker,
                expected=last_seq + 1,
                got=seq,
            )
            raise RuntimeError(f"Kalshi seq gap on {ticker}: {last_seq} → {seq}")
        self._seq_by_market[ticker] = seq

        side = str(data.get("side", "")).lower()
        if side not in ("yes", "no"):
            return

        try:
            price = float(data.get("price_dollars", data.get("price", 0)))
            if "price_dollars" not in data and "price" in data:
                # Legacy cents-integer fallback
                price = float(data["price"]) / 100.0
            delta = float(data.get("delta", 0))
        except (ValueError, TypeError):
            return

        yield KalshiBookDelta(
            market_ticker=ticker,
            side=side,
            price=price,
            delta=delta,
            seq=seq,
            timestamp_ms=ts_ms,
            recv_ts=recv_ts,
        )

    def _parse_trade(self, data: dict, recv_ts: float):
        ticker = data.get("market_ticker")
        if ticker not in self._tickers:
            return
        try:
            yes_price = data.get("yes_price_dollars")
            if yes_price is None:
                yes_price = float(data.get("yes_price", 0)) / 100.0
            else:
                yes_price = float(yes_price)
            yield KalshiTrade(
                market_ticker=ticker,
                yes_price=yes_price,
                size=float(data.get("count", 0)),
                taker_side=str(data.get("taker_side", "")).lower(),
                trade_id=str(data.get("trade_id", "")),
                timestamp_ms=int(data.get("ts_ms", data.get("ts", 0)) or 0),
                recv_ts=recv_ts,
            )
        except (KeyError, ValueError) as exc:
            self._log.warning("bad_trade", error=str(exc))

    def _parse_ticker(self, data: dict, recv_ts: float):
        ticker = data.get("market_ticker")
        if ticker not in self._tickers:
            return
        try:
            def _dollars(key_dollars: str, key_cents: str) -> float:
                if key_dollars in data:
                    return float(data[key_dollars])
                return float(data.get(key_cents, 0)) / 100.0

            yield KalshiTicker(
                market_ticker=ticker,
                yes_bid=_dollars("yes_bid_dollars", "yes_bid"),
                yes_ask=_dollars("yes_ask_dollars", "yes_ask"),
                last_price=_dollars("last_price_dollars", "last_price"),
                volume_24h=int(data.get("volume_24h", 0) or 0),
                open_interest=int(data.get("open_interest", 0) or 0),
                timestamp_ms=int(data.get("ts_ms", data.get("ts", 0)) or 0),
                recv_ts=recv_ts,
            )
        except (KeyError, ValueError) as exc:
            self._log.warning("bad_ticker", error=str(exc))

    # Auth
    @staticmethod
    def _load_rsa_key(pem: str):
        """Load RSA private key from PEM string."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        return serialization.load_pem_private_key(
            pem.encode(),
            password=None,
            backend=default_backend(),
        )

    def _sign_rsa(self, timestamp_ms: int, method: str, path: str) -> str:
        """
        Kalshi v2 auth: RSA-PSS SHA-256 over (timestamp_ms + method + path).
        salt_length = digest size (32 bytes), per docs.kalshi.com's own
        quick-start example , NOT PSS.MAX_LENGTH, which some third-party
        guides use inconsistently and can fail signature verification.
        Returns base64-encoded signature.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        msg = f"{timestamp_ms}{method}{path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def _next_cmd_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id
