"""
Cross-venue delta hedging for binary prediction market positions.

A YES token is basically a digital/binary option on the underlying. Delta
w.r.t. the underlying (Black-Scholes digital option delta):

    delta = phi(d2) / (S * sigma * sqrt(T-t))
    d2 = [ln(S/K) + (r - sigma^2/2)*(T-t)] / (sigma*sqrt(T-t))

delta peaks near 0.50 when S ~= K with time left, collapses to 0 as
resolution approaches, and blows up near the strike on a knife-edge
binary right before expiry, don't trust it in the last few minutes.

For q YES contracts held, exposure is E_USD = q * P_pm * delta. We short
E_USD / S_perp perp contracts on Hyperliquid to flatten it.

We only hedge when |corr(dP_pm, dS_crypto)| clears a minimum threshold
over the recent window. If the prediction market is moving on news/
fundamentals rather than the underlying, hedging just adds noise.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import aiohttp
import structlog

from config.settings import HedgeProfile

logger = structlog.get_logger(__name__)


# Types
@dataclass
class HedgeState:
    market_id: str
    gross_hedge_usd: float    # current net hedge position in USD
    last_hedge_ts: float      = 0.0
    last_delta: float         = 0.0
    last_correlation: float   = 0.0
    total_hedge_trades: int   = 0


@dataclass
class HedgeInstruction:
    """Output of the hedging engine to the execution layer."""
    market_id: str
    side: str           # "BUY" (close short hedge) or "SELL" (open short hedge)
    size_contracts: float
    symbol: str         # e.g. "BTC-PERP"
    is_reduce: bool     = False
    reason: str         = ""


# Correlation Tracker
class CorrelationTracker:
    """
    Rolling Pearson correlation between ΔP_pm and ΔS_crypto.
    Uses exponentially-weighted covariance for recency bias.
    """

    def __init__(self, window_s: int = 300):
        self._window_s = window_s
        self._pm_ret:   Deque[Tuple[float, float]] = deque()  # (ts, Δp)
        self._cex_ret:  Deque[Tuple[float, float]] = deque()  # (ts, Δs)

    def add_pm_return(self, ts: float, delta_p: float) -> None:
        self._pm_ret.append((ts, delta_p))
        self._evict(self._pm_ret, ts)

    def add_cex_return(self, ts: float, delta_s: float) -> None:
        self._cex_ret.append((ts, delta_s))
        self._evict(self._cex_ret, ts)

    def correlation(self) -> float:
        """
        Compute Pearson ρ from aligned series.
        Returns 0.0 if insufficient data.
        """
        if len(self._pm_ret) < 10 or len(self._cex_ret) < 10:
            return 0.0

        # Align by closest timestamp (simple nearest-neighbor)
        pm_ts  = np.array([x[0] for x in self._pm_ret])
        pm_val = np.array([x[1] for x in self._pm_ret])
        cx_ts  = np.array([x[0] for x in self._cex_ret])
        cx_val = np.array([x[1] for x in self._cex_ret])

        # Find overlapping time range
        t_start = max(pm_ts[0], cx_ts[0])
        t_end   = min(pm_ts[-1], cx_ts[-1])

        if t_end - t_start < 30:   # need at least 30s of overlap
            return 0.0

        # Resample to common 5-second grid
        t_grid = np.arange(t_start, t_end, 5.0)
        pm_interp = np.interp(t_grid, pm_ts, pm_val)
        cx_interp = np.interp(t_grid, cx_ts, cx_val)

        if np.std(pm_interp) < 1e-9 or np.std(cx_interp) < 1e-9:
            return 0.0

        return float(np.corrcoef(pm_interp, cx_interp)[0, 1])

    def _evict(self, q: Deque, now: float) -> None:
        cutoff = now - self._window_s
        while q and q[0][0] < cutoff:
            q.popleft()


# Binary Option Delta Calculator
class BinaryDeltaCalc:
    """
    Computes ∂P_pm/∂S for a binary pari-mutuel prediction contract.

    For a deterministic-outcome binary (no continuous delta):
    We approximate using the BS digital option formula where:
      - Strike K is the threshold (e.g. $100K for BTC-100K contract)
      - σ is the ATM implied vol of the underlying perp
      - T-t is time to resolution in years
    """

    @staticmethod
    def bs_digital_delta(
        S: float,      # current underlying price (e.g. BTC = 95000)
        K: float,      # strike / threshold (e.g. 100000)
        T_t: float,    # time to resolution in years
        sigma: float,  # annualized vol of underlying
        r: float = 0.0,
    ) -> float:
        """
        Delta of a cash-or-nothing CALL digital option.
        Returns ∂C/∂S where C ∈ [0,1].
        
        Δ = φ(d₂) / (S·σ·√(T-t))
        
        This is positive: higher S → higher P(YES for >K contract).
        """
        if T_t <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0

        sqrt_Tt = math.sqrt(T_t)
        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T_t) / (sigma * sqrt_Tt)

        # Standard normal PDF
        phi_d2 = math.exp(-0.5 * d2 ** 2) / math.sqrt(2 * math.pi)

        delta = phi_d2 / (S * sigma * sqrt_Tt)

        # Clamp to reasonable bounds
        return min(max(delta, 0.0), 0.1)   # max Δ per $1 move in S

    @staticmethod
    def empirical_delta(
        pm_returns: np.ndarray,
        cex_returns: np.ndarray,
    ) -> float:
        """
        OLS regression of PM returns on CEX returns.
        β = Cov(ΔP, ΔS) / Var(ΔS) ≈ ∂P/∂S
        """
        if len(pm_returns) < 10 or len(cex_returns) < 10:
            return 0.0

        n = min(len(pm_returns), len(cex_returns))
        x = cex_returns[-n:]
        y = pm_returns[-n:]

        var_x = np.var(x)
        if var_x < 1e-12:
            return 0.0

        cov_xy = np.cov(x, y)[0, 1]
        return float(cov_xy / var_x)


# Hedge Engine
class HedgeEngine:
    """
    Determines hedge instructions from current PM inventory and correlation.
    Executes on Hyperliquid via async REST.

    Design:
      - Delta is computed from BS digital formula + empirical correction
      - Correlation filter gates ALL hedge execution
      - Hedge size = q_yes * delta * hedge_size_multiplier
      - Position management: tracks current hedge and only sends diff
    """

    SECONDS_PER_YEAR = 365.25 * 24 * 3600

    def __init__(
        self,
        profile: HedgeProfile,
        hl_url: str,
        hl_wallet: str,
        hl_private_key: str,
    ):
        self._profile    = profile
        self._hl_url     = hl_url.rstrip("/")
        self._hl_wallet  = hl_wallet
        self._hl_key     = hl_private_key

        self._corr_trackers: Dict[str, CorrelationTracker] = {}
        self._states: Dict[str, HedgeState] = {}
        self._delta_calc = BinaryDeltaCalc()
        self._log = logger.bind(component="hedge_engine")

    def register_market(self, market_id: str) -> None:
        if market_id not in self._states:
            self._states[market_id]       = HedgeState(market_id=market_id, gross_hedge_usd=0.0)
            self._corr_trackers[market_id] = CorrelationTracker(
                window_s=self._profile.correlation_window
            )

    def update_pm_mid(self, market_id: str, ts: float, p_mid: float, prev_mid: float) -> None:
        tracker = self._corr_trackers.get(market_id)
        if tracker:
            tracker.add_pm_return(ts, p_mid - prev_mid)

    def update_underlying_price(self, market_id: str, ts: float, S: float, prev_S: float) -> None:
        tracker = self._corr_trackers.get(market_id)
        if tracker and prev_S > 0:
            tracker.add_cex_return(ts, (S - prev_S) / prev_S)  # fractional return

    async def compute_and_hedge(
        self,
        market_id: str,
        inventory_q: float,    # net YES contracts (positive = long)
        p_mid: float,          # current PM mid probability
        S_perp: float,         # current perp price (e.g. BTC in USD)
        K_strike: float,       # contract threshold (e.g. 100_000 for BTC-100K)
        sigma_perp: float,     # annualized vol estimate for perp
        T_res_s: float,        # seconds to resolution
        perp_symbol: str = "BTC-PERP",
    ) -> Optional[HedgeInstruction]:
        """
        Main entry point. Returns HedgeInstruction if action needed, else None.
        """
        if not self._profile.enabled:
            return None

        state = self._states.get(market_id)
        if state is None:
            self.register_market(market_id)
            state = self._states[market_id]

        # Correlation gate
        tracker = self._corr_trackers[market_id]
        rho = tracker.correlation()
        state.last_correlation = rho

        if abs(rho) < self._profile.correlation_min_abs:
            self._log.debug(
                "hedge_skipped_low_corr",
                market_id=market_id,
                rho=round(rho, 3),
                required=self._profile.correlation_min_abs,
            )
            return None

        # Compute delta
        T_t_years = max(0, T_res_s) / self.SECONDS_PER_YEAR
        delta = self._delta_calc.bs_digital_delta(
            S=S_perp, K=K_strike, T_t=T_t_years, sigma=sigma_perp
        )
        state.last_delta = delta

        # Target hedge size
        #
        # Exposure = q_yes contracts × p_mid (USD if 1 contract = $1 payout)
        # Delta exposure = exposure × Δ × (K/S) (normalize to perp units)
        #
        # We want to SELL perp to hedge LONG PM exposure.
        # If inventory_q > 0 (long YES): we want short perp.
        # If inventory_q < 0 (short YES): we want long perp.

        exposure_usd   = abs(inventory_q) * p_mid  # USD equivalent of PM position
        delta_usd      = exposure_usd * delta * (K_strike / S_perp)
        target_hedge_usd = delta_usd * self._profile.hedge_size_multiplier * (
            -1 if inventory_q > 0 else 1   # short perp to hedge long YES
        )

        # How much do we need to move?
        current_hedge_usd = state.gross_hedge_usd
        delta_hedge_needed = target_hedge_usd - current_hedge_usd

        if abs(delta_hedge_needed) < self._profile.min_delta_usd:
            return None  # Below minimum threshold , skip

        # Build instruction
        size_contracts = abs(delta_hedge_needed) / S_perp
        side = "SELL" if delta_hedge_needed < 0 else "BUY"

        instr = HedgeInstruction(
            market_id=market_id,
            side=side,
            size_contracts=round(size_contracts, 6),
            symbol=perp_symbol,
            is_reduce=(side == "BUY" and current_hedge_usd < 0) or
                       (side == "SELL" and current_hedge_usd > 0),
            reason=f"delta={round(delta,5)} rho={round(rho,3)}",
        )

        self._log.info(
            "hedge_instruction",
            market_id=market_id,
            side=side,
            size=round(size_contracts, 4),
            delta_usd=round(abs(delta_hedge_needed), 2),
            rho=round(rho, 3),
            perp_delta=round(delta, 6),
        )

        return instr

    # Hyperliquid execution
    async def execute_hedge(
        self,
        session: aiohttp.ClientSession,
        instr: HedgeInstruction,
        S_perp: float,
    ) -> bool:
        """
        Submit market order to Hyperliquid.
        Uses Hyperliquid's EVM-based action signing.
        """
        from eth_account import Account
        import json

        is_buy = instr.side == "BUY"

        # Hyperliquid order action
        order_action = {
            "type": "order",
            "orders": [{
                "a": 0,              # asset index (BTC=0)
                "b": is_buy,         # isBuy
                "p": str(round(S_perp * (1.005 if is_buy else 0.995), 1)),  # limit price with slip
                "s": str(round(instr.size_contracts, 6)),
                "r": instr.is_reduce,
                "t": {"limit": {"tif": "Ioc"}},   # IOC for immediate hedge execution
            }],
            "grouping": "na",
        }

        # Sign action (Hyperliquid uses EIP-712 on their custom domain)
        timestamp_ms = int(time.time() * 1000)
        account = Account.from_key(self._hl_key)
        # Simplified: in production use Hyperliquid's full signing spec
        payload = {
            "action": order_action,
            "nonce": timestamp_ms,
            "signature": {"r": "0x", "s": "0x", "v": 0},  # placeholder , use HL SDK
        }

        url = f"{self._hl_url}/exchange"
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._profile.max_hedge_latency_ms / 1000),
            ) as resp:
                body = await resp.json()
                if resp.status == 200 and body.get("status") == "ok":
                    # Update tracked hedge position
                    state = self._states[instr.market_id]
                    sign = -1 if instr.side == "SELL" else 1
                    state.gross_hedge_usd += sign * instr.size_contracts * S_perp
                    state.total_hedge_trades += 1
                    state.last_hedge_ts = time.monotonic()
                    self._log.info(
                        "hedge_executed",
                        market_id=instr.market_id,
                        side=instr.side,
                        size=instr.size_contracts,
                    )
                    return True
                else:
                    self._log.error(
                        "hedge_failed",
                        status=resp.status,
                        body=str(body)[:300],
                    )
                    return False

        except asyncio.TimeoutError:
            self._log.error(
                "hedge_timeout_exceeded",
                limit_ms=self._profile.max_hedge_latency_ms,
            )
            return False
