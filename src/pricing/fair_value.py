"""
src/pricing/fair_value.py
──────────────────────────
Pricing engine for binary prediction markets.

THEORETICAL FOUNDATION:

1. AVELLANEDA-STOIKOV ADAPTATION FOR BINARY MARKETS
   ────────────────────────────────────────────────
   Standard A-S for continuous assets:
     r(s,q,t) = s - q·γ·σ²·(T-t)        [reservation price]
     δ* = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k) [optimal half-spread]

   Binary market adaptation:
     - s ∈ [0,1] is the mid probability (not a price)
     - Payoff is discrete: 1 if YES resolves, 0 if NO
     - σ² = p·(1-p) is the binary variance (Bernoulli)
     - γ: risk aversion (calibrated from max drawdown tolerance)
     - k: arrival rate decay (calibrated from historical fills)
     - T-t: time to resolution in YEARS (maintains dimensional consistency)
     - q: inventory in contracts (signed: long=positive, short=negative)
     - q_max: maximum inventory from risk profile

   Reservation price:
     r = p_mid - q·γ·p·(1-p)·(T-t)

   Optimal half-spread:
     δ* = γ·p·(1-p)·(T-t) + (1/γ)·ln(1 + γ/k)

   Note: Near resolution (T-t → 0), spread collapses → reduce quoting.
   Near p=0.5, σ² is maximized → widest spreads appropriate.

2. FLOW ADJUSTMENT
   ──────────────
   P_fair = P_base + α·CVD + β·OFI_normalized

   where α, β are calibrated from rolling regression of
   aggressive flow on subsequent mid-price moves.

3. FAVORITE-LONGSHOT BIAS CORRECTION (Kalshi)
   ──────────────────────────────────────────
   Empirical finding: retail markets overweight low-probability events.
   Calibration: fit a power-law transform from historical resolution data.
   P_true = P_market^κ / (P_market^κ + (1-P_market)^κ)   [Prelec, 1998]
   κ < 1 corrects for longshot bias; κ = 1 means no correction.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import structlog

from src.data.unified_book import MarketState

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Model Parameters
# ──────────────────────────────────────────────

@dataclass
class ASBinaryParams:
    """
    Calibrated parameters for the A-S binary adaptation.
    Updated by the Calibrator on a rolling basis.
    """
    gamma: float = 0.05       # risk aversion coefficient
    k: float     = 1.5        # order arrival intensity (fills per unit spread)
    alpha: float = 0.002      # CVD → fair value sensitivity
    beta: float  = 0.001      # OFI → fair value sensitivity (normalized)
    kappa: float = 0.85       # Prelec longshot bias exponent (<1 = correct bias)

    # Spread floor/ceil guardrails
    min_half_spread: float = 0.005   # 0.5 cent minimum
    max_half_spread: float = 0.08    # 8 cent maximum

    # Inventory limits (contracts)
    q_max: float = 500.0

    # Resolution threshold: don't quote if T-t < this (seconds)
    min_ttres_s: float = 3600.0  # 1 hour before resolution


# ──────────────────────────────────────────────
# Fair Value Output
# ──────────────────────────────────────────────

@dataclass
class FairValueResult:
    market_id: str
    ts: float

    # Core outputs
    p_fair: float          # Flow-adjusted fair value probability
    p_reservation: float   # Inventory-skewed reservation price
    bid_quote: float       # Where to place bid
    ask_quote: float       # Where to place ask
    half_spread: float     # Optimal half-spread δ*

    # Diagnostics
    p_base: float          # Mid before flow adjustment
    flow_adjustment: float # α·CVD + β·OFI
    inventory_skew: float  # -q·γ·σ²·(T-t)
    binary_vol: float      # p·(1-p) = σ²
    ttres_years: float     # Time to resolution in years
    should_quote: bool     # False if conditions make quoting unprofitable

    # Risk signals
    is_stale: bool = False
    longshot_corrected: bool = False


# ──────────────────────────────────────────────
# Core Pricing Engine
# ──────────────────────────────────────────────

class FairValueEngine:
    """
    Stateless pricing computation. Thread-safe (no mutable state).
    Takes MarketState + current inventory + calibrated params → FairValueResult.
    """

    SECONDS_PER_YEAR: float = 365.25 * 24 * 3600

    def compute(
        self,
        state: MarketState,
        inventory_q: float,        # signed contracts: +long YES, -short YES
        params: ASBinaryParams,
        apply_bias_correction: bool = False,
    ) -> FairValueResult:
        """
        Main pricing entry point.

        Parameters
        ----------
        state           : Latest MarketState from UnifiedBook
        inventory_q     : Current net inventory in this market
        params          : Calibrated A-S binary parameters
        apply_bias_correction : Apply Prelec longshot correction (Kalshi)
        """
        now = time.time()

        # ── 1. Base probability ───────────────
        p_base = state.p_mid

        # ── 2. Longshot bias correction ───────
        if apply_bias_correction:
            p_base = self._prelec_correction(p_base, params.kappa)
            longshot_corrected = True
        else:
            longshot_corrected = False

        # ── 3. Flow adjustment ────────────────
        #
        # Normalize OFI by total depth to get a [-1, +1] signal
        ofi_norm = state.imbalance  # already normalized in UnifiedBook
        flow_adj = params.alpha * state.cvd + params.beta * ofi_norm

        # Clip flow adjustment to prevent it overriding the book entirely
        max_flow_adj = state.spread * 0.5
        flow_adj = max(-max_flow_adj, min(max_flow_adj, flow_adj))

        p_fair = np.clip(p_base + flow_adj, 0.001, 0.999)

        # ── 4. Binary variance ────────────────
        # σ² = p·(1-p), maximized at p=0.5
        sigma_sq = p_fair * (1.0 - p_fair)

        # ── 5. Time to resolution ─────────────
        ttres_s = max(0.0, state.time_to_resolution_s)
        ttres_years = ttres_s / self.SECONDS_PER_YEAR

        # ── 6. Quoting guard ──────────────────
        should_quote = (
            state.is_valid() and
            ttres_s > params.min_ttres_s and
            state.spread < 0.25  # market not completely illiquid
        )

        # ── 7. Reservation price (inventory skew) ─────
        # r = p_fair - q·γ·σ²·(T-t)
        inv_skew = -inventory_q * params.gamma * sigma_sq * ttres_years

        # Clip skew: don't let inventory move quote outside [0.01, 0.99]
        max_skew = 0.10
        inv_skew = max(-max_skew, min(max_skew, inv_skew))

        p_reservation = np.clip(p_fair + inv_skew, 0.001, 0.999)

        # ── 8. Optimal half-spread ────────────
        #
        # δ* = γ·σ²·(T-t) + (1/γ)·ln(1 + γ/k)
        #
        # The first term widens spread as time remaining increases
        # (more uncertainty about path → more inventory risk).
        # The second term is the market-making profit per unit.

        if ttres_years > 0 and params.gamma > 0 and params.k > 0:
            term1 = params.gamma * sigma_sq * ttres_years
            term2 = (1.0 / params.gamma) * math.log(1.0 + params.gamma / params.k)
            half_spread = term1 + term2
        else:
            half_spread = params.min_half_spread

        # Apply guardrails
        half_spread = max(params.min_half_spread, min(params.max_half_spread, half_spread))

        # ── 9. Quote prices ───────────────────
        #
        # Center quotes around reservation price, not fair value.
        # This is the inventory-management asymmetry: if long,
        # we skew both bid and ask DOWN to reduce inventory faster.
        bid_quote = np.clip(p_reservation - half_spread, 0.01, 0.98)
        ask_quote = np.clip(p_reservation + half_spread, 0.02, 0.99)

        # Round to tick size (default 1 cent)
        tick = 0.01
        bid_quote = round(round(bid_quote / tick) * tick, 4)
        ask_quote = round(round(ask_quote / tick) * tick, 4)

        # Ensure we never post a crossed quote
        if bid_quote >= ask_quote:
            mid_q = (bid_quote + ask_quote) / 2
            bid_quote = round(mid_q - tick, 4)
            ask_quote = round(mid_q + tick, 4)

        # ── 10. Staleness check ───────────────
        age_s = now - (state.book_ts_ms / 1000.0) if state.book_ts_ms else 0
        is_stale = age_s > 5.0   # book older than 5s = stale

        return FairValueResult(
            market_id=state.market_id,
            ts=time.monotonic(),
            p_fair=float(p_fair),
            p_reservation=float(p_reservation),
            bid_quote=float(bid_quote),
            ask_quote=float(ask_quote),
            half_spread=float(half_spread),
            p_base=float(p_base),
            flow_adjustment=float(flow_adj),
            inventory_skew=float(inv_skew),
            binary_vol=float(sigma_sq),
            ttres_years=float(ttres_years),
            should_quote=should_quote and not is_stale,
            is_stale=is_stale,
            longshot_corrected=longshot_corrected,
        )

    # ── Bias correction ───────────────────────

    @staticmethod
    def _prelec_correction(p_market: float, kappa: float) -> float:
        """
        Prelec (1998) probability weighting function.
        w(p) = exp(-(-ln(p))^κ)

        For κ < 1:
          - Overweights small probabilities (longshot bias exists in market)
          - Our correction inverts this to get P_true

        Inverse: given w(p) = p_market, solve for p_true ≈ p_market
        using the inverse Prelec function (numerical).
        
        Simple approximation: logit-space power transform
        logit(p_true) = logit(p_market)^(1/κ)  [maintains monotonicity]
        """
        p_clipped = np.clip(p_market, 0.001, 0.999)
        logit_mkt = math.log(p_clipped / (1 - p_clipped))

        # Sign-preserving power transform in logit space
        sign = 1 if logit_mkt >= 0 else -1
        logit_corrected = sign * (abs(logit_mkt) ** (1.0 / kappa))

        p_true = 1.0 / (1.0 + math.exp(-logit_corrected))
        return float(np.clip(p_true, 0.001, 0.999))


# ──────────────────────────────────────────────
# Rolling Calibrator
# ──────────────────────────────────────────────

class ParameterCalibrator:
    """
    Online calibration of (γ, k, α, β) from observed fills and price paths.

    Calibration targets:
      - γ: from max acceptable inventory PnL volatility
      - k: from empirical fill rate vs spread distance histogram
      - α, β: from rolling OLS regression of (CVD, OFI) → Δmid over 1-min windows

    In production this runs in a background task on a 5-min timer.
    """

    WINDOW = 500       # observations for rolling OLS

    def __init__(self, base_params: ASBinaryParams):
        self._params = base_params
        # Circular buffers for regression
        self._X: list = []   # [(cvd, ofi_norm), ...]
        self._y: list = []   # [delta_mid, ...]

    def observe(self, cvd: float, ofi_norm: float, delta_mid_next: float) -> None:
        """Record one observation for calibration."""
        self._X.append((cvd, ofi_norm))
        self._y.append(delta_mid_next)
        if len(self._X) > self.WINDOW:
            self._X.pop(0)
            self._y.pop(0)

    def recalibrate(self) -> ASBinaryParams:
        """
        Re-estimate α, β via OLS.
        Returns updated params (caller replaces current params).
        """
        if len(self._X) < 50:
            return self._params

        X = np.array(self._X)
        y = np.array(self._y)

        # Add intercept column
        Xb = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS: β = (X'X)^{-1} X'y
            coeffs = np.linalg.lstsq(Xb, y, rcond=None)[0]
            _, alpha_new, beta_new = coeffs

            # Regularize: don't let coefficients jump wildly
            alpha_new = 0.7 * self._params.alpha + 0.3 * float(alpha_new)
            beta_new  = 0.7 * self._params.beta  + 0.3 * float(beta_new)

            # Sanity bounds
            alpha_new = np.clip(alpha_new, -0.01, 0.01)
            beta_new  = np.clip(beta_new,  -0.01, 0.01)

            self._params = ASBinaryParams(
                gamma=self._params.gamma,
                k=self._params.k,
                alpha=alpha_new,
                beta=beta_new,
                kappa=self._params.kappa,
                min_half_spread=self._params.min_half_spread,
                max_half_spread=self._params.max_half_spread,
                q_max=self._params.q_max,
                min_ttres_s=self._params.min_ttres_s,
            )

            logger.info(
                "params_calibrated",
                alpha=round(alpha_new, 6),
                beta=round(beta_new, 6),
                n_obs=len(self._X),
            )

        except np.linalg.LinAlgError as exc:
            logger.warning("calibration_failed", error=str(exc))

        return self._params

    def calibrate_gamma_from_drawdown(
        self,
        max_drawdown_usd: float,
        notional_usd: float,
        sigma_p: float = 0.10,   # assumed mid-price volatility
    ) -> float:
        """
        Derive γ from risk tolerance.
        γ = -ΔU/ΔW where U is CARA utility.
        Approximation: γ ≈ 2·max_drawdown / (notional · σ_p²)
        """
        if notional_usd <= 0 or sigma_p <= 0:
            return self._params.gamma

        gamma = 2 * max_drawdown_usd / (notional_usd * sigma_p ** 2)
        gamma = np.clip(gamma, 0.001, 1.0)
        logger.info("gamma_calibrated", gamma=round(gamma, 5))
        return float(gamma)

    @property
    def params(self) -> ASBinaryParams:
        return self._params
