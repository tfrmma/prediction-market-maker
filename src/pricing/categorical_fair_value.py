"""
Softmax fair-value layer, forces sum(p_fair)=1 across a basket of N
outcomes, the consistency half of the categorical A-S extension (see
covariance.py for the other half, inventory skew).

score_i = ln(p_mid_i) [+ flow], softmax across the basket (LMSR-style).
Using ln(p_i) and not logit(p_i) matters, logit distorts a book that's
already consistent, checked numerically before writing this. Zero flow
reduces to plain proportional renormalization; flow tilts each
outcome's weight multiplicatively before renormalizing.

Doesn't try to numerically match fair_value.py's single-market p_fair
for N=2, different functional form (multiplicative here vs additive
there) and N independent order books don't have to move in lockstep.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict

import numpy as np
import structlog

from src.data.categorical_book import CategoricalMarketState

logger = structlog.get_logger(__name__)


def _prelec_correction(p_market: float, kappa: float) -> float:
    """Prelec (1998) longshot-bias transform, duplicated from
    fair_value.py's FairValueEngine._prelec_correction to keep this
    module standalone during the additive rollout, worth consolidating
    once this engine is wired into main.py."""
    p_clipped = float(np.clip(p_market, 0.001, 0.999))
    logit_mkt = math.log(p_clipped / (1 - p_clipped))
    sign = 1 if logit_mkt >= 0 else -1
    logit_corrected = sign * (abs(logit_mkt) ** (1.0 / kappa))
    p_true = 1.0 / (1.0 + math.exp(-logit_corrected))
    return float(np.clip(p_true, 0.001, 0.999))


@dataclass
class CategoricalASParams:
    """Flow-sensitivity coefficients for the softmax layer, same role as
    ASBinaryParams.alpha/beta but in log-probability score space. Needs
    its own calibration once there's basket-level fill data, for now
    seeded off the binary defaults."""
    alpha: float = 0.002        # CVD -> score sensitivity
    beta: float = 0.001         # OFI (normalized) -> score sensitivity
    max_flow_adj: float = 2.0   # clip |alpha*cvd + beta*ofi| before it hits the softmax
    min_ttres_s: float = 3600.0

    # Inventory skew / spread (see categorical_skew.py)
    gamma: float = 0.05
    k: float = 1.5                                  # default arrival intensity
    k_per_outcome: Dict[str, float] = field(default_factory=dict)
    min_half_spread: float = 0.005
    max_half_spread: float = 0.08
    max_skew: float = 0.10

    # Per-outcome Prelec kappa (Kalshi longshot bias), keyed by outcome_id.
    # Outcomes not present here get no bias correction.
    prelec_kappa: Dict[str, float] = field(default_factory=dict)


@dataclass
class CategoricalFairValueResult:
    event_id: str
    ts: float

    p_fair: Dict[str, float]            # softmax output, sums to 1 by construction
    p_base: Dict[str, float]            # market mid per outcome, post bias-correction
    scores: Dict[str, float]            # ln(p_base_i) + flow adjustment, pre-softmax
    flow_adjustment: Dict[str, float]   # alpha*cvd + beta*ofi per outcome, clipped

    should_quote: bool
    is_stale: bool = False


class CategoricalFairValueEngine:
    """Stateless. Basket in, consistent fair-value vector out."""

    STALE_AGE_S = 5.0   # same threshold fair_value.py uses per-market

    def compute(self, basket: CategoricalMarketState, params: CategoricalASParams) -> CategoricalFairValueResult:
        now = time.time()
        outcome_ids = list(basket.outcomes.keys())

        p_base: Dict[str, float] = {}
        flow_adj: Dict[str, float] = {}
        scores: Dict[str, float] = {}

        for oid in outcome_ids:
            state = basket.outcomes[oid]
            p = float(np.clip(state.p_mid, 1e-4, 1.0 - 1e-4))

            kappa = params.prelec_kappa.get(oid)
            if kappa is not None:
                p = _prelec_correction(p, kappa)

            p_base[oid] = p

            adj = params.alpha * state.cvd + params.beta * state.imbalance
            adj = max(-params.max_flow_adj, min(params.max_flow_adj, adj))
            flow_adj[oid] = adj

            scores[oid] = math.log(p) + adj

        # Numerically stable softmax across the basket
        score_arr = np.array([scores[oid] for oid in outcome_ids])
        weights = np.exp(score_arr - score_arr.max())
        p_fair_arr = weights / weights.sum()
        p_fair = {oid: float(p_fair_arr[i]) for i, oid in enumerate(outcome_ids)}

        ttres_s = min(basket.outcomes[oid].time_to_resolution_s for oid in outcome_ids)
        book_ages = [
            now - (basket.outcomes[oid].book_ts_ms / 1000.0)
            for oid in outcome_ids if basket.outcomes[oid].book_ts_ms
        ]
        is_stale = bool(book_ages) and max(book_ages) > self.STALE_AGE_S

        should_quote = basket.is_valid() and ttres_s > params.min_ttres_s and not is_stale

        return CategoricalFairValueResult(
            event_id=basket.event_id,
            ts=time.monotonic(),
            p_fair=p_fair,
            p_base=p_base,
            scores=scores,
            flow_adjustment=flow_adj,
            should_quote=should_quote,
            is_stale=is_stale,
        )


class CategoricalParameterCalibrator:
    """
    Online calibration of CategoricalASParams (alpha, beta, k, gamma)
    for one event. Same three targets as fair_value.py's
    ParameterCalibrator, pooled across every outcome in the basket
    since alpha/beta/k/gamma are shared basket-wide, not per-outcome.

    alpha/beta: rolling OLS, but the target is delta_ln(p_i), not raw
    delta_mid, alpha/beta get applied additively in log-score space
    here (score_i = ln(p_i) + alpha*cvd_i + beta*ofi_i), not raw
    probability space, so that's the unit the regression needs to be in.

    k: empirical fill-rate vs. quote-distance histogram, log-linear fit.
    fair_value.py's ParameterCalibrator docstring promises this
    ("k: from empirical fill rate vs spread distance histogram") but
    never actually implements it, only alpha/beta and gamma are real
    there. This one does.

    gamma: same drawdown-based conversion as the binary calibrator,
    reused as-is, it's a portfolio risk-tolerance formula, nothing
    outcome-specific about it.
    """

    WINDOW = 500
    MIN_OBS_FOR_OLS = 50
    MIN_OBS_FOR_K = 50
    N_DELTA_BUCKETS = 10
    MIN_BUCKET_OBS = 5

    def __init__(self, event_id: str, base_params: CategoricalASParams, fill_obs_window_s: float = 1.0):
        self._event_id = event_id
        self._params = base_params
        self._fill_obs_window_s = fill_obs_window_s   # see calibrate_k_from_fills
        self._X: list = []              # [(cvd_i, ofi_i), ...] pooled across outcomes
        self._y: list = []              # [delta_ln_p_i_next, ...]
        self._fill_obs: list = []       # [(delta, filled), ...]
        self._log = logger.bind(component="categorical_calibrator", event_id=event_id)

    def observe_flow(self, cvd: float, ofi_norm: float, delta_ln_p_next: float) -> None:
        """One (outcome, tick) observation. Call once per outcome per
        basket update, pooling across outcomes is intentional."""
        self._X.append((cvd, ofi_norm))
        self._y.append(delta_ln_p_next)
        if len(self._X) > self.WINDOW:
            self._X.pop(0)
            self._y.pop(0)

    def observe_fill_outcome(self, delta: float, filled: bool) -> None:
        """delta = |quoted_price - mid| for whichever side was live,
        filled = whether it got hit within fill_obs_window_s of being
        posted. Window needs to be roughly constant across observations,
        calibrate_k_from_fills() assumes a single shared window when it
        inverts the Poisson relationship."""
        self._fill_obs.append((delta, filled))
        if len(self._fill_obs) > self.WINDOW:
            self._fill_obs.pop(0)

    def recalibrate_flow(self) -> CategoricalASParams:
        """Re-estimate alpha/beta via OLS. Returns updated params (caller
        replaces current params)."""
        if len(self._X) < self.MIN_OBS_FOR_OLS:
            return self._params

        X = np.array(self._X)
        y = np.array(self._y)
        Xb = np.column_stack([np.ones(len(X)), X])

        try:
            coeffs = np.linalg.lstsq(Xb, y, rcond=None)[0]
            _, alpha_new, beta_new = coeffs

            # Regularize: don't let coefficients jump wildly tick to tick
            alpha_new = 0.7 * self._params.alpha + 0.3 * float(alpha_new)
            beta_new = 0.7 * self._params.beta + 0.3 * float(beta_new)
            alpha_new = float(np.clip(alpha_new, -0.01, 0.01))
            beta_new = float(np.clip(beta_new, -0.01, 0.01))

            self._params = replace(self._params, alpha=alpha_new, beta=beta_new)
            self._log.info(
                "categorical_flow_params_calibrated",
                alpha=round(alpha_new, 6), beta=round(beta_new, 6), n_obs=len(self._X),
            )
        except np.linalg.LinAlgError as exc:
            self._log.warning("flow_calibration_failed", error=str(exc))

        return self._params

    def calibrate_k_from_fills(self) -> float:
        """
        Bucket (delta, filled) observations by distance-from-mid quantile,
        empirical fill rate per bucket, invert the Poisson relationship
        each observation actually came from (fill_rate = 1 - exp(-lambda
        * fill_obs_window_s)) to recover lambda_hat, then log-linear fit:

            ln(lambda_hat) = ln(A) - k * delta

        Taking ln() of the raw fill rate directly (skipping the
        inversion) only works while lambda*dt is small, real fill rates
        saturate toward 1 well before that, and the fit silently
        recovers a k an order of magnitude too small. Caught this by
        validating against a simulator with a known true k before
        trusting the formula, see test_core.py.
        """
        if len(self._fill_obs) < self.MIN_OBS_FOR_K:
            return self._params.k

        deltas = np.array([d for d, _ in self._fill_obs])
        filled = np.array([1.0 if f else 0.0 for _, f in self._fill_obs])

        edges = np.unique(np.quantile(deltas, np.linspace(0, 1, self.N_DELTA_BUCKETS + 1)))
        if len(edges) < 3:
            return self._params.k
        bucket_idx = np.digitize(deltas, edges[1:-1])

        bucket_delta, bucket_lambda = [], []
        for b in range(len(edges) - 1):
            mask = bucket_idx == b
            if mask.sum() < self.MIN_BUCKET_OBS:
                continue
            rate = filled[mask].mean()
            if rate <= 0 or rate >= 1:
                continue   # can't invert a 0% or saturated 100% bucket
            lam_hat = -math.log(1 - rate) / self._fill_obs_window_s
            bucket_delta.append(float(deltas[mask].mean()))
            bucket_lambda.append(lam_hat)

        if len(bucket_delta) < 3:
            return self._params.k

        x = np.array(bucket_delta)
        y = np.log(np.array(bucket_lambda))
        Xb = np.column_stack([np.ones(len(x)), x])

        try:
            _, slope = np.linalg.lstsq(Xb, y, rcond=None)[0]
            k_new = float(np.clip(-slope, 0.1, 10.0))
            k_new = 0.7 * self._params.k + 0.3 * k_new

            self._params = replace(self._params, k=k_new)
            self._log.info("k_calibrated", k=round(k_new, 4), n_buckets=len(bucket_delta))
            return k_new
        except np.linalg.LinAlgError as exc:
            self._log.warning("k_calibration_failed", error=str(exc))
            return self._params.k

    def calibrate_gamma_from_drawdown(
        self, max_drawdown_usd: float, notional_usd: float, sigma_p: float = 0.10,
    ) -> float:
        """gamma ~= 2*max_drawdown / (notional * sigma_p^2), same CARA
        risk-tolerance approximation as the binary calibrator."""
        if notional_usd <= 0 or sigma_p <= 0:
            return self._params.gamma

        gamma = float(np.clip(2 * max_drawdown_usd / (notional_usd * sigma_p ** 2), 0.001, 1.0))
        self._params = replace(self._params, gamma=gamma)
        self._log.info("gamma_calibrated", gamma=round(gamma, 5))
        return gamma

    @property
    def params(self) -> CategoricalASParams:
        return self._params
