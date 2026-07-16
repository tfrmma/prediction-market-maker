"""
Collateral and multi-market exposure tracking.

Tracks USDC/USD collateral across venues, computes per-market exposure,
enforces concentration limits, and keeps VWAP cost basis for PnL.

Polymarket: BUY locks USDC (makerAmount) immediately, SELL locks the
outcome tokens you already hold. On resolution, USDC comes back at $1
(YES) or $0 (NO).

Kalshi is margin-based: margin = price * size on a BUY, (1-price) * size
on a SELL.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from config.settings import RiskProfile

logger = structlog.get_logger(__name__)


# Types
@dataclass
class Position:
    """Live position in one prediction market."""
    market_id: str
    venue: str                   # "polymarket" | "kalshi"

    # Inventory (contracts, signed: +long YES, -short YES)
    net_qty: float = 0.0
    long_qty: float = 0.0
    short_qty: float = 0.0

    # Cost basis (VWAP)
    avg_entry_long: float = 0.0
    avg_entry_short: float = 0.0

    # Collateral locked (USD)
    collateral_locked: float = 0.0

    # Mark-to-market
    current_mid: float = 0.0
    last_update_ts: float = field(default_factory=time.monotonic)

    @property
    def unrealized_pnl(self) -> float:
        """
        Long PnL:  (mid - avg_entry) × long_qty
        Short PnL: (avg_entry - mid) × short_qty
        """
        long_pnl  = (self.current_mid - self.avg_entry_long) * self.long_qty
        short_pnl = (self.avg_entry_short - self.current_mid) * self.short_qty
        return long_pnl + short_pnl

    @property
    def gross_exposure(self) -> float:
        """Total USD exposure (long + short collateral at current mid)."""
        return (self.long_qty * self.current_mid +
                self.short_qty * (1.0 - self.current_mid))

    @property
    def net_delta_usd(self) -> float:
        """Signed USD delta: positive = net long YES."""
        return self.net_qty * self.current_mid


@dataclass
class CollateralAccount:
    """Tracks free vs locked collateral per venue."""
    venue: str
    currency: str             # "USDC" | "USD"
    total_balance: float      = 0.0
    locked: float             = 0.0   # in open orders/positions
    pending_settlement: float = 0.0   # filled but not yet settled

    @property
    def free(self) -> float:
        return max(0.0, self.total_balance - self.locked - self.pending_settlement)

    @property
    def utilization(self) -> float:
        if self.total_balance <= 0:
            return 1.0
        return (self.locked + self.pending_settlement) / self.total_balance


@dataclass
class ExposureReport:
    """Full portfolio snapshot."""
    ts: float
    positions: Dict[str, Position]
    accounts: Dict[str, CollateralAccount]
    total_net_delta_usd: float
    total_gross_exposure_usd: float
    total_unrealized_pnl: float
    concentration: Dict[str, float]     # market_id → % of gross
    rebalance_signals: List[str]        # human-readable rebalance actions


# Fill Processor
class FillProcessor:
    """
    VWAP-correct position update from fills.
    Handles:
      - Partial fills
      - Position flips (long → short)
      - Realized PnL extraction on closing fills
    """

    @staticmethod
    def apply_fill(
        pos: Position,
        fill_side: str,    # "BUY" | "SELL"
        fill_price: float,
        fill_qty: float,
    ) -> float:
        """
        Update position in-place.
        Returns realized PnL from this fill (0 if opening).
        """
        realized = 0.0

        if fill_side == "BUY":
            if pos.net_qty >= 0:
                # Opening or adding to long
                total_cost = pos.avg_entry_long * pos.long_qty + fill_price * fill_qty
                pos.long_qty += fill_qty
                pos.avg_entry_long = total_cost / pos.long_qty if pos.long_qty > 0 else 0.0
                pos.net_qty += fill_qty

            else:
                # Closing short
                closing_qty = min(fill_qty, pos.short_qty)
                realized = (pos.avg_entry_short - fill_price) * closing_qty
                pos.short_qty -= closing_qty
                pos.net_qty += closing_qty

                # If fill_qty > short_qty, remainder opens long
                remainder = fill_qty - closing_qty
                if remainder > 0:
                    pos.long_qty = remainder
                    pos.avg_entry_long = fill_price
                    pos.net_qty += remainder

        else:  # SELL
            if pos.net_qty <= 0:
                # Opening or adding to short
                total_cost = pos.avg_entry_short * pos.short_qty + fill_price * fill_qty
                pos.short_qty += fill_qty
                pos.avg_entry_short = total_cost / pos.short_qty if pos.short_qty > 0 else 0.0
                pos.net_qty -= fill_qty

            else:
                # Closing long
                closing_qty = min(fill_qty, pos.long_qty)
                realized = (fill_price - pos.avg_entry_long) * closing_qty
                pos.long_qty -= closing_qty
                pos.net_qty -= closing_qty

                remainder = fill_qty - closing_qty
                if remainder > 0:
                    pos.short_qty = remainder
                    pos.avg_entry_short = fill_price
                    pos.net_qty -= remainder

        pos.last_update_ts = time.monotonic()
        return realized


# Inventory Manager
class InventoryManager:
    """
    Central inventory and collateral state manager.

    Thread-safety: single-threaded asyncio. No locking needed.
    """

    # Rebalance signal thresholds
    CONCENTRATION_WARN: float  = 0.30   # warn if one market > 30% of gross
    CONCENTRATION_LIMIT: float = 0.50   # hard limit at 50%
    UTILIZATION_WARN: float    = 0.80   # warn if collateral utilization > 80%

    def __init__(self, risk_profile: RiskProfile):
        self._risk     = risk_profile
        self._positions: Dict[str, Position] = {}
        self._accounts:  Dict[str, CollateralAccount] = {}
        self._realized_pnl: Dict[str, float] = {}
        self._fill_proc = FillProcessor()
        self._log = logger.bind(component="inventory")

    # Setup
    def register_market(self, market_id: str, venue: str) -> None:
        if market_id not in self._positions:
            self._positions[market_id] = Position(market_id=market_id, venue=venue)
            self._realized_pnl[market_id] = 0.0

    def register_account(
        self,
        venue: str,
        currency: str,
        balance: float,
    ) -> None:
        self._accounts[venue] = CollateralAccount(
            venue=venue,
            currency=currency,
            total_balance=balance,
        )

    def seed_position(self, market_id: str, net_qty: float, avg_entry: float) -> None:
        """
        Overwrite a freshly-registered (zeroed) position with what the
        venue actually reports we're holding. Only meant to be called
        once at startup, right after register_market and before any
        fills come in, this is not a general-purpose position setter.
        """
        pos = self._positions.get(market_id)
        if pos is None:
            self._log.warning("seed_position_unknown_market", market_id=market_id)
            return
        if net_qty >= 0:
            pos.net_qty = net_qty
            pos.long_qty = net_qty
            pos.avg_entry_long = avg_entry
        else:
            pos.net_qty = net_qty
            pos.short_qty = -net_qty
            pos.avg_entry_short = avg_entry
        self._log.info("position_seeded", market_id=market_id, net_qty=net_qty, avg_entry=avg_entry)

    # State updates
    def on_fill(
        self,
        market_id: str,
        fill_side: str,
        fill_price: float,
        fill_qty: float,
        collateral_used: float,
    ) -> float:
        """
        Process a fill. Returns realized PnL.
        Updates collateral account (lock/unlock).
        """
        pos = self._get_position(market_id)
        realized = self._fill_proc.apply_fill(pos, fill_side, fill_price, fill_qty)
        self._realized_pnl[market_id] = self._realized_pnl.get(market_id, 0.0) + realized

        # Update collateral
        venue = pos.venue
        acct = self._accounts.get(venue)
        if acct:
            if fill_side == "BUY":
                acct.locked += collateral_used
            else:
                # Release collateral proportional to close
                # Simplified: lock outcome tokens instead
                acct.locked = max(0.0, acct.locked - collateral_used)

        self._log.info(
            "fill_applied",
            market_id=market_id,
            side=fill_side,
            price=round(fill_price, 4),
            qty=fill_qty,
            realized_pnl=round(realized, 4),
            net_qty=round(pos.net_qty, 2),
        )

        # Post-fill risk check
        self._check_concentration()

        return realized

    def on_order_placed(self, venue: str, collateral_amount: float) -> bool:
        """
        Reserve collateral for a pending order.
        Returns False if insufficient free collateral.
        """
        acct = self._accounts.get(venue)
        if acct is None:
            return True   # No tracking → allow

        if acct.free < collateral_amount:
            self._log.warning(
                "insufficient_collateral",
                venue=venue,
                required=round(collateral_amount, 2),
                available=round(acct.free, 2),
            )
            return False

        acct.locked += collateral_amount
        return True

    def on_order_cancelled(self, venue: str, collateral_amount: float) -> None:
        """Release reserved collateral on cancel."""
        acct = self._accounts.get(venue)
        if acct:
            acct.locked = max(0.0, acct.locked - collateral_amount)

    def update_mid(self, market_id: str, mid: float) -> None:
        pos = self._get_position(market_id)
        pos.current_mid = mid

    def on_resolution(
        self,
        market_id: str,
        resolved_yes: bool,
    ) -> float:
        """
        Market resolved. Compute final settlement PnL.
        Returns total PnL from position.
        """
        pos = self._get_position(market_id)
        resolution_price = 1.0 if resolved_yes else 0.0

        # Final realized PnL
        final_long_pnl  = (resolution_price - pos.avg_entry_long) * pos.long_qty
        final_short_pnl = (pos.avg_entry_short - resolution_price) * pos.short_qty
        final_pnl = final_long_pnl + final_short_pnl

        self._realized_pnl[market_id] = self._realized_pnl.get(market_id, 0.0) + final_pnl

        # Release all collateral
        acct = self._accounts.get(pos.venue)
        if acct:
            acct.locked = max(0.0, acct.locked - pos.collateral_locked)
            acct.total_balance += final_pnl   # settle to balance

        self._log.info(
            "market_resolved",
            market_id=market_id,
            resolved_yes=resolved_yes,
            final_pnl=round(final_pnl, 4),
            long_qty=pos.long_qty,
            short_qty=pos.short_qty,
        )

        # Zero out position
        pos.net_qty = pos.long_qty = pos.short_qty = 0.0
        pos.collateral_locked = 0.0

        return final_pnl

    # Query
    def get_net_qty(self, market_id: str) -> float:
        return self._positions.get(market_id, Position("", "")).net_qty

    def get_position(self, market_id: str) -> Optional[Position]:
        return self._positions.get(market_id)

    def get_free_collateral(self, venue: str) -> float:
        acct = self._accounts.get(venue)
        return acct.free if acct else float("inf")

    def generate_report(self) -> ExposureReport:
        """Full portfolio snapshot."""
        total_net_delta   = sum(p.net_delta_usd for p in self._positions.values())
        total_gross       = sum(p.gross_exposure for p in self._positions.values())
        total_unrealized  = sum(p.unrealized_pnl for p in self._positions.values())

        concentration = {}
        if total_gross > 0:
            for mid, pos in self._positions.items():
                concentration[mid] = pos.gross_exposure / total_gross

        signals = self._generate_rebalance_signals(concentration)

        return ExposureReport(
            ts=time.monotonic(),
            positions=dict(self._positions),
            accounts=dict(self._accounts),
            total_net_delta_usd=total_net_delta,
            total_gross_exposure_usd=total_gross,
            total_unrealized_pnl=total_unrealized,
            concentration=concentration,
            rebalance_signals=signals,
        )

    # Internal
    def _get_position(self, market_id: str) -> Position:
        if market_id not in self._positions:
            self._positions[market_id] = Position(market_id=market_id, venue="unknown")
        return self._positions[market_id]

    def _check_concentration(self) -> None:
        """Emit warnings if any single market exceeds concentration limits."""
        total_gross = sum(p.gross_exposure for p in self._positions.values())
        if total_gross <= 0:
            return

        for mid, pos in self._positions.items():
            conc = pos.gross_exposure / total_gross
            if conc > self.CONCENTRATION_LIMIT:
                self._log.warning(
                    "concentration_limit_breach",
                    market_id=mid,
                    concentration=round(conc, 3),
                    limit=self.CONCENTRATION_LIMIT,
                )
            elif conc > self.CONCENTRATION_WARN:
                self._log.info(
                    "concentration_warn",
                    market_id=mid,
                    concentration=round(conc, 3),
                )

    def _generate_rebalance_signals(
        self,
        concentration: Dict[str, float],
    ) -> List[str]:
        signals = []

        for mid, conc in concentration.items():
            if conc > self.CONCENTRATION_LIMIT:
                signals.append(
                    f"REDUCE {mid}: concentration {conc:.1%} > limit {self.CONCENTRATION_LIMIT:.1%}"
                )

        for venue, acct in self._accounts.items():
            if acct.utilization > self.UTILIZATION_WARN:
                signals.append(
                    f"COLLATERAL_LOW {venue}: utilization {acct.utilization:.1%}"
                )

        net_delta = sum(p.net_delta_usd for p in self._positions.values())
        if abs(net_delta) > self._risk.max_net_delta_usd:
            signals.append(
                f"DELTA_LIMIT: net delta ${net_delta:.0f} > limit ${self._risk.max_net_delta_usd:.0f}"
            )

        return signals
