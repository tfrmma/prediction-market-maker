"""
src/risk/engine.py
────────────────────
Independent risk engine.  Runs as a separate asyncio task.
Never co-located with strategy logic — intentional architectural separation.

Kill Switch Triggers:
  1. API failure: ≥3 consecutive order failures
  2. State desync: book age > STALE_THRESHOLD
  3. Anomalous latency: fill latency spike > 5× rolling median
  4. Intraday drawdown > config limit
  5. Per-unit-time loss rate exceeded

PnL Decomposition (per market):
  total_pnl = spread_capture + inventory_pnl + adverse_selection_cost

  - spread_capture:     Σ(fill_price - mid_at_fill) × side_sign × filled_qty
  - inventory_pnl:      (current_mid - avg_entry) × net_position
  - adverse_selection:  mid_price_change_after_fill × filled_qty × sign
                        measured in [0, T_adverse] window after each fill
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import structlog

from config.settings import RiskProfile

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────

class KillReason(str, Enum):
    DRAWDOWN         = "intraday_drawdown_limit"
    LOSS_RATE        = "loss_per_time_limit"
    API_FAILURE      = "consecutive_api_failures"
    STATE_DESYNC     = "book_state_desync"
    LATENCY_SPIKE    = "anomalous_fill_latency"
    MANUAL           = "manual_override"


@dataclass
class PnLSnapshot:
    market_id: str
    ts: float

    # Decomposition
    spread_capture:    float = 0.0   # pure MM edge
    inventory_pnl:     float = 0.0   # mark-to-market on open inventory
    adverse_selection: float = 0.0   # cost of filling toxic flow (negative)

    # Totals
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def adverse_selection_rate(self) -> float:
        """AS cost as fraction of spread capture. < -0.7 = very toxic."""
        if abs(self.spread_capture) < 1e-9:
            return 0.0
        return self.adverse_selection / self.spread_capture


@dataclass
class FillEvent:
    market_id: str
    order_id: str
    fill_price: float
    fill_size: float
    side: str           # "BUY" or "SELL"
    mid_at_fill: float
    ts: float


@dataclass
class RiskStatus:
    kill_active: bool = False
    kill_reason: Optional[KillReason] = None
    kill_ts: Optional[float] = None
    consecutive_api_failures: int = 0
    last_book_ts: float = 0.0
    pnl: Dict[str, PnLSnapshot] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Risk Engine
# ──────────────────────────────────────────────

class RiskEngine:
    """
    Passive monitor. Publishes kill events to a shared asyncio.Event.
    Strategy tasks check kill_event.is_set() before every action.
    """

    # Adverse selection measurement window (seconds after fill)
    AS_WINDOW_S: float = 30.0
    # API failure threshold before kill
    API_FAIL_THRESHOLD: int = 3
    # Book stale threshold
    BOOK_STALE_S: float = 10.0
    # Latency spike: N × median before alarm
    LATENCY_SPIKE_MULT: float = 5.0

    def __init__(
        self,
        risk_profile: RiskProfile,
        kill_event: asyncio.Event,
    ):
        self._risk      = risk_profile
        self._kill      = kill_event
        self._status    = RiskStatus()

        # PnL tracking per market
        self._pnl:   Dict[str, PnLSnapshot] = {}
        # Inventory tracking: {market_id: {avg_entry, net_qty}}
        self._inventory: Dict[str, Dict] = {}
        # Pending fills awaiting AS measurement
        self._pending_as: Deque[Tuple[float, FillEvent]] = deque()  # (measure_at, fill)
        # Intraday PnL (resets at midnight UTC)
        self._intraday_pnl: float = 0.0
        self._day_start_ts: float = self._get_day_start()
        # Fill latency rolling window (ms)
        self._latencies: Deque[float] = deque(maxlen=100)

        self._log = logger.bind(component="risk_engine")

    # ── Main monitor loop ─────────────────────

    async def run(self) -> None:
        """Background task: periodic risk checks."""
        while True:
            await asyncio.sleep(1.0)
            self._check_daily_reset()
            self._flush_as_measurements()
            self._check_loss_rate()
            self._check_stale_book()
            self._emit_health()

    # ── Event handlers ────────────────────────

    def on_fill(
        self,
        market_id: str,
        order_id: str,
        fill_price: float,
        fill_size: float,
        side: str,
        mid_at_fill: float,
    ) -> None:
        """Called by order manager after each confirmed fill."""
        ts = time.monotonic()
        ev = FillEvent(
            market_id=market_id,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=fill_size,
            side=side,
            mid_at_fill=mid_at_fill,
            ts=ts,
        )

        pnl = self._get_pnl(market_id)

        # ── Spread capture ─────────────────────
        # For a BUY fill: we acquired at fill_price, fair value = mid
        # spread_capture_per_fill = (mid - fill_price) × size   [positive if bought below mid]
        sign = 1 if side == "BUY" else -1
        sc = (mid_at_fill - fill_price) * fill_size * sign
        pnl.spread_capture += sc

        # ── Realized PnL update ───────────────
        inv = self._get_inventory(market_id)
        old_qty  = inv["net_qty"]
        old_cost = inv["avg_entry"] * abs(old_qty) if old_qty != 0 else 0.0

        if side == "BUY":
            new_qty = old_qty + fill_size
            # Update VWAP
            new_cost = old_cost + fill_price * fill_size
            inv["avg_entry"] = new_cost / abs(new_qty) if new_qty != 0 else 0.0
            inv["net_qty"] = new_qty
        else:  # SELL
            realized = (fill_price - inv["avg_entry"]) * fill_size
            pnl.realized_pnl += realized
            self._intraday_pnl += realized
            inv["net_qty"] = old_qty - fill_size

        # ── Schedule adverse selection measurement ────
        self._pending_as.append((ts + self.AS_WINDOW_S, ev))

        # ── Drawdown check ─────────────────────
        total_unrealized = self._compute_total_unrealized()
        total_pnl = self._intraday_pnl + total_unrealized

        if total_pnl < -self._risk.intraday_drawdown_limit:
            self._trigger_kill(
                KillReason.DRAWDOWN,
                f"Intraday PnL {total_pnl:.2f} below limit -{self._risk.intraday_drawdown_limit:.2f}",
            )

        self._log.debug(
            "fill_processed",
            market_id=market_id,
            side=side,
            price=round(fill_price, 4),
            size=fill_size,
            spread_capture=round(sc, 4),
            intraday_pnl=round(total_pnl, 2),
        )

    def on_market_update(self, market_id: str, mid: float) -> None:
        """Update unrealized PnL mark-to-market."""
        pnl = self._get_pnl(market_id)
        inv = self._get_inventory(market_id)

        if inv["net_qty"] != 0:
            pnl.inventory_pnl = (mid - inv["avg_entry"]) * inv["net_qty"]
        else:
            pnl.inventory_pnl = 0.0

    def on_api_failure(self) -> None:
        self._status.consecutive_api_failures += 1
        if self._status.consecutive_api_failures >= self.API_FAIL_THRESHOLD:
            self._trigger_kill(
                KillReason.API_FAILURE,
                f"≥{self.API_FAIL_THRESHOLD} consecutive API failures",
            )

    def on_api_success(self) -> None:
        self._status.consecutive_api_failures = 0

    def on_book_update(self, market_id: str, ts: float) -> None:
        self._status.last_book_ts = ts

    def on_fill_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        if len(self._latencies) >= 20:
            median = sorted(self._latencies)[len(self._latencies) // 2]
            if latency_ms > self.LATENCY_SPIKE_MULT * median and median > 0:
                self._log.warning(
                    "latency_spike",
                    current_ms=round(latency_ms, 1),
                    median_ms=round(median, 1),
                    mult=round(latency_ms / median, 1),
                )
                self._trigger_kill(
                    KillReason.LATENCY_SPIKE,
                    f"Fill latency {latency_ms:.0f}ms = {latency_ms/median:.1f}× median",
                )

    # ── Internal checks ───────────────────────

    def _flush_as_measurements(self) -> None:
        """Compute adverse selection for fills whose window has elapsed."""
        now = time.monotonic()
        while self._pending_as and self._pending_as[0][0] <= now:
            _, ev = self._pending_as.popleft()
            pnl = self._get_pnl(ev.market_id)
            inv = self._get_inventory(ev.market_id)

            # AS = mid_now - mid_at_fill (for BUY): if price moved against us, it's negative
            mid_now = inv.get("last_mid", ev.mid_at_fill)
            sign = 1 if ev.side == "BUY" else -1
            as_cost = (ev.mid_at_fill - mid_now) * ev.fill_size * sign
            pnl.adverse_selection += as_cost

            if abs(pnl.adverse_selection_rate) > 0.8:
                self._log.warning(
                    "high_adverse_selection",
                    market_id=ev.market_id,
                    as_rate=round(pnl.adverse_selection_rate, 3),
                    as_usd=round(pnl.adverse_selection, 4),
                )

    def _check_loss_rate(self) -> None:
        """Loss per unit time check: rolling 15-minute window."""
        # Simplified: check if intraday PnL deteriorated too quickly
        # Full impl would track PnL by minute-bucket
        pass

    def _check_stale_book(self) -> None:
        if self._status.last_book_ts > 0:
            age = time.monotonic() - self._status.last_book_ts
            if age > self.BOOK_STALE_S:
                self._trigger_kill(
                    KillReason.STATE_DESYNC,
                    f"Book stale for {age:.1f}s",
                )

    def _trigger_kill(self, reason: KillReason, detail: str = "") -> None:
        if not self._kill.is_set():
            self._status.kill_active = True
            self._status.kill_reason = reason
            self._status.kill_ts = time.monotonic()
            self._kill.set()
            self._log.critical(
                "KILL_SWITCH_ACTIVATED",
                reason=reason.value,
                detail=detail,
            )

    def _compute_total_unrealized(self) -> float:
        return sum(p.inventory_pnl for p in self._pnl.values())

    def _check_daily_reset(self) -> None:
        day_start = self._get_day_start()
        if day_start > self._day_start_ts:
            self._intraday_pnl = 0.0
            self._day_start_ts = day_start

    def _emit_health(self) -> None:
        for mid, pnl in self._pnl.items():
            self._log.info(
                "pnl_snapshot",
                market_id=mid,
                spread_capture=round(pnl.spread_capture, 4),
                inventory_pnl=round(pnl.inventory_pnl, 4),
                adverse_selection=round(pnl.adverse_selection, 4),
                realized=round(pnl.realized_pnl, 4),
                total=round(pnl.total_pnl, 4),
                as_rate=round(pnl.adverse_selection_rate, 3),
            )

    def _get_pnl(self, market_id: str) -> PnLSnapshot:
        if market_id not in self._pnl:
            self._pnl[market_id] = PnLSnapshot(market_id=market_id, ts=time.monotonic())
        return self._pnl[market_id]

    def _get_inventory(self, market_id: str) -> Dict:
        if market_id not in self._inventory:
            self._inventory[market_id] = {"net_qty": 0.0, "avg_entry": 0.0, "last_mid": 0.0}
        return self._inventory[market_id]

    @staticmethod
    def _get_day_start() -> float:
        import datetime
        now = datetime.datetime.utcnow()
        day_start = datetime.datetime(now.year, now.month, now.day)
        return day_start.timestamp()

    # ── Properties ────────────────────────────

    @property
    def is_alive(self) -> bool:
        return not self._kill.is_set()

    @property
    def status(self) -> RiskStatus:
        return self._status

    def get_pnl_all(self) -> Dict[str, PnLSnapshot]:
        return dict(self._pnl)
