"""
Base websocket feed: reconnect with backoff, detect sequence gaps and
force a resync, track feed lag, push health metrics to a queue.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


# Types
class FeedStatus(str, Enum):
    CONNECTING   = "connecting"
    CONNECTED    = "connected"
    RESYNCING    = "resyncing"
    DISCONNECTED = "disconnected"
    FAILED       = "failed"


@dataclass
class FeedHealth:
    venue: str
    status: FeedStatus
    last_msg_ts: float      = 0.0   # wall clock of last received message
    feed_lag_ms: float      = 0.0   # exchange_ts - wall_clock (negative = we're ahead)
    reconnect_count: int    = 0
    sequence_gaps: int      = 0
    messages_total: int     = 0
    bytes_total: int        = 0
    ts: float               = field(default_factory=time.monotonic)


@dataclass
class RawMessage:
    """Envelope around raw WS bytes with reception metadata."""
    venue: str
    payload: bytes
    recv_ts: float     # time.monotonic() at reception
    recv_wall: float   # time.time() for exchange-ts comparison


# Base Feed
class BaseFeed(ABC):
    """
    Async WebSocket feed abstraction.

    Subclasses implement:
      _build_subscribe_msg()  → bytes/str sent on connect
      _parse_message()        → yields parsed domain objects (or None to skip)
      _extract_sequence()     → optional sequence number from raw msg for gap detection
      _extract_exchange_ts()  → optional exchange timestamp (ms) for lag tracking
    """

    # Reconnection config
    RECONNECT_BASE_DELAY_S: float = 0.5
    RECONNECT_MAX_DELAY_S: float  = 30.0
    RECONNECT_JITTER: float       = 0.1
    HEARTBEAT_INTERVAL_S: float   = 20.0
    STALE_FEED_TIMEOUT_S: float   = 10.0   # seconds without msg → reconnect

    def __init__(
        self,
        url: str,
        venue: str,
        out_queue: asyncio.Queue,
        health_queue: Optional[asyncio.Queue] = None,
    ):
        self._url = url
        self._venue = venue
        self._out_queue = out_queue
        self._health_queue = health_queue
        self._health = FeedHealth(venue=venue, status=FeedStatus.DISCONNECTED)
        self._shutdown = asyncio.Event()
        self._last_seq: Optional[int] = None
        self._ws = None
        self._log = logger.bind(venue=venue)

    # Public API
    async def run(self) -> None:
        """
        Main loop.  Call this as a Task.
        Reconnects indefinitely until shutdown() is called.
        """
        delay = self.RECONNECT_BASE_DELAY_S
        while not self._shutdown.is_set():
            try:
                await self._connect_and_consume()
                delay = self.RECONNECT_BASE_DELAY_S   # reset on clean exit
            except Exception as exc:
                self._health.reconnect_count += 1
                self._health.status = FeedStatus.DISCONNECTED
                self._log.warning(
                    "feed_disconnected",
                    error=str(exc),
                    reconnect_in=delay,
                    total_reconnects=self._health.reconnect_count,
                )
                await self._emit_health()
                if not self._shutdown.is_set():
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.RECONNECT_MAX_DELAY_S)

    def shutdown(self) -> None:
        self._shutdown.set()

    # Abstract interface
    @abstractmethod
    async def _build_subscribe_msgs(self) -> list[str | bytes]:
        """Return list of subscribe payloads to send after connection."""
        ...

    def _extra_ws_headers(self) -> dict:
        """Override for venues that auth at the handshake (e.g. Kalshi's signed headers)."""
        return {}

    @abstractmethod
    async def _parse_message(self, raw: RawMessage) -> AsyncIterator:
        """
        Parse raw message; yield domain objects for downstream.
        Yield nothing to discard.
        """
        ...

    def _extract_sequence(self, payload: bytes) -> Optional[int]:
        """Override to enable gap detection."""
        return None

    def _extract_exchange_ts(self, payload: bytes) -> Optional[int]:
        """Override to return exchange timestamp in ms."""
        return None

    # Internal
    async def _connect_and_consume(self) -> None:
        import websockets  # local import to avoid circular

        self._health.status = FeedStatus.CONNECTING
        await self._emit_health()

        headers = self._extra_ws_headers()

        # websockets >=13 renamed extra_headers -> additional_headers.
        # TODO: just pin the version in pyproject and drop this shim
        connect_kwargs: dict = dict(
            ping_interval=self.HEARTBEAT_INTERVAL_S,
            ping_timeout=10,
            max_size=2**23,   # 8MB frame limit
        )
        if headers:
            import inspect
            sig = inspect.signature(websockets.connect)
            header_kw = "additional_headers" if "additional_headers" in sig.parameters else "extra_headers"
            connect_kwargs[header_kw] = headers

        async with websockets.connect(self._url, **connect_kwargs) as ws:
            self._ws = ws
            self._health.status = FeedStatus.CONNECTED
            self._log.info("feed_connected", url=self._url)

            # Send subscribe messages
            for msg in await self._build_subscribe_msgs():
                await ws.send(msg)

            # Concurrent: consumer + stale-feed watchdog
            await asyncio.gather(
                self._consume_loop(ws),
                self._watchdog(),
            )

    async def _consume_loop(self, ws) -> None:
        async for raw_bytes in ws:
            if self._shutdown.is_set():
                return

            recv_ts   = time.monotonic()
            recv_wall = time.time()
            payload   = raw_bytes if isinstance(raw_bytes, bytes) else raw_bytes.encode()

            raw = RawMessage(
                venue=self._venue,
                payload=payload,
                recv_ts=recv_ts,
                recv_wall=recv_wall,
            )

            # Sequence gap detection
            seq = self._extract_sequence(payload)
            if seq is not None and self._last_seq is not None:
                expected = self._last_seq + 1
                if seq > expected:
                    gap = seq - expected
                    self._health.sequence_gaps += gap
                    self._log.error(
                        "sequence_gap",
                        expected=expected,
                        received=seq,
                        gap=gap,
                    )
                    # Force resync (raise so outer loop reconnects)
                    raise RuntimeError(f"Sequence gap detected: {gap} messages lost")
            if seq is not None:
                self._last_seq = seq

            # Feed lag tracking
            exch_ts = self._extract_exchange_ts(payload)
            if exch_ts is not None:
                self._health.feed_lag_ms = (recv_wall * 1000) - exch_ts

            # Update health metrics
            self._health.last_msg_ts  = recv_ts
            self._health.messages_total += 1
            self._health.bytes_total  += len(payload)

            # Dispatch parsed events downstream
            async for event in self._parse_message(raw):
                await self._out_queue.put(event)

    async def _watchdog(self) -> None:
        """Raise if no messages received within STALE_FEED_TIMEOUT_S."""
        while not self._shutdown.is_set():
            await asyncio.sleep(1.0)
            if self._health.last_msg_ts > 0:
                age = time.monotonic() - self._health.last_msg_ts
                if age > self.STALE_FEED_TIMEOUT_S:
                    raise RuntimeError(
                        f"Feed stale: no message in {age:.1f}s"
                    )

    async def _emit_health(self) -> None:
        self._health.ts = time.monotonic()
        if self._health_queue is not None:
            await self._health_queue.put(self._health)
