"""
Covariance estimator for the categorical inventory skew, feeds the Σ
matrix into the Guéant-style multi-asset A-S reservation price:

    r = p_fair - gamma * (Sigma @ q) * (T-t)

q is signed inventory in contracts, p_fair is a probability, so Sigma
needs to be in probability units for that formula to be dimensionally
sane, same units fair_value.py's scalar p*(1-p) is already in for the
binary case.

Raw probabilities live on a simplex (sum to 1), so their raw N x N
covariance matrix is singular, N-1 degrees of freedom for N numbers.
Two ways we deal with that here, for two different purposes:

  - Structural prior (probability units): for N mutually-exclusive
    outcomes with true probabilities p, the terminal payout vector is
    one-hot Multinomial(1, p), covariance diag(p) - p pᵀ. Dropping the
    reference outcome's row/column from that (same trick as dropping a
    baseline category in dummy-variable regression) gives a
    non-singular (N-1)x(N-1) probability-unit matrix, exactly the
    N-outcome generalization of p*(1-p). Needs zero history.

  - Online EWMA (internally ALR space): tracking realized co-movement
    of the raw probabilities directly runs into the same singularity,
    plus the variance of p_i isn't stationary in p_i (a move near 0 or
    1 behaves differently than one near 0.5). Additive log-ratio space,
    y_i = ln(p_i / p_ref) for every non-reference outcome, is a
    well-behaved bijection off the simplex onto unconstrained R^(N-1)
    and is what we actually run the EWMA on. The catch: ALR variance
    for an outcome scales like 1/p_i, so its magnitude is in log-ratio
    units, not probability units, don't feed it into the skew formula
    directly (tested that the hard way, see test_core.py). We convert
    back to probability space via the local delta-method Jacobian at
    the current prices before handing it to a caller.

covariance() is the one thing external code should call, it's always
probability-space regardless of whether it's serving the structural
prior or the converted EWMA. alr_covariance() is exposed separately
for diagnostics (correlation, being scale-invariant, is directly
readable there without conversion).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import structlog

from src.data.categorical_book import CategoricalMarketState

logger = structlog.get_logger(__name__)


def multinomial_structural_covariance(p: np.ndarray) -> np.ndarray:
    """Cov(X) for a one-hot Multinomial(1, p) draw. diag(p) - p pᵀ, rank N-1."""
    return np.diag(p) - np.outer(p, p)


def structural_probability_covariance(
    p_mid: Dict[str, float],
    outcome_ids: List[str],
    reference_id: str,
) -> np.ndarray:
    """
    Probability-space structural prior for the non-reference outcomes,
    the multinomial covariance with the reference row/column dropped.
    Always available, no history required, this is what
    CategoricalCovarianceEstimator.covariance() falls back to cold.
    """
    p = np.array([p_mid[oid] for oid in outcome_ids], dtype=float)
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()   # renormalize, should already be ~1 but don't trust upstream blindly

    sigma_p = multinomial_structural_covariance(p)

    ref_idx = outcome_ids.index(reference_id)
    keep = [i for i in range(len(outcome_ids)) if i != ref_idx]
    return sigma_p[np.ix_(keep, keep)]


class CategoricalCovarianceEstimator:
    """
    One instance per event. Feed it CategoricalMarketState snapshots as
    they arrive, call covariance(current_p_mid) to get an (N-1)x(N-1)
    probability-space matrix for the non-reference outcomes, EWMA-based
    once warmed up, the structural prior otherwise.

    Reference outcome is fixed at construction (last id in outcome_ids)
    and never changes, the running EWMA's internal ALR coordinate
    system depends on it staying put.

    TODO: if the reference outcome's own probability collapses toward
    0 well into the estimator's life (leading candidate gets knocked
    out by news, say), the ALR->probability conversion below gets
    poorly conditioned right when you'd want it most. Haven't hit this
    in practice yet, worth a CLR/ILR basis instead of a single fixed
    reference if it turns out to actually happen.
    """

    MIN_OBS_FOR_EWMA = 30

    def __init__(self, event_id: str, outcome_ids: List[str], correlation_window_s: float = 1800.0):
        if len(outcome_ids) < 2:
            raise ValueError(f"{event_id}: need at least 2 outcomes, got {outcome_ids}")
        self._event_id = event_id
        self._outcome_ids = list(outcome_ids)
        self._reference_id = outcome_ids[-1]
        self._non_reference_ids = [oid for oid in self._outcome_ids if oid != self._reference_id]
        self._window_s = correlation_window_s

        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None   # native ALR-space EWMA state
        self._last_ts: Optional[float] = None
        self._n_obs = 0

        self._log = logger.bind(component="categorical_covariance", event_id=event_id)

    @property
    def reference_id(self) -> str:
        return self._reference_id

    @property
    def n_obs(self) -> int:
        return self._n_obs

    def _alr_vector(self, p_mid: Dict[str, float]) -> np.ndarray:
        p_ref = max(p_mid[self._reference_id], 1e-6)
        return np.array([math.log(max(p_mid[oid], 1e-6) / p_ref) for oid in self._non_reference_ids])

    def observe(self, basket: CategoricalMarketState) -> None:
        """Record one basket snapshot, updates the running EWMA mean/cov."""
        p_mid = {oid: state.p_mid for oid, state in basket.outcomes.items()}
        if self._reference_id not in p_mid or len(p_mid) < len(self._outcome_ids):
            return   # basket not fully populated for this event yet, skip

        y = self._alr_vector(p_mid)

        if self._mean is None:
            self._mean = y
            self._cov = np.zeros((len(y), len(y)))
            self._last_ts = basket.ts
            self._n_obs = 1
            return

        dt = max(basket.ts - self._last_ts, 1e-6)
        self._last_ts = basket.ts
        decay = math.exp(-dt / self._window_s)

        delta = y - self._mean
        self._mean = decay * self._mean + (1.0 - decay) * y
        self._cov = decay * self._cov + (1.0 - decay) * np.outer(delta, delta)
        self._n_obs += 1

    def alr_covariance(self) -> Optional[np.ndarray]:
        """
        Diagnostic: the raw EWMA in its native log-ratio space, None
        until warmed up. Correlation (scale-invariant) is directly
        readable here, the covariance magnitude itself is in log-ratio
        units, use covariance() for anything going into the skew formula.
        """
        if self._n_obs < self.MIN_OBS_FOR_EWMA:
            return None
        return self._cov

    def _alr_jacobian(self, current_p_mid: Dict[str, float]) -> np.ndarray:
        """
        Local Jacobian d(ALR)/d(p_free) at current_p_mid, diagonal
        (1/p_i) plus a constant (1/p_ref) picked up from p_ref itself
        depending on every free p_j through the simplex constraint.
        """
        p_ref = max(current_p_mid[self._reference_id], 1e-6)
        p_free = np.array([max(current_p_mid[oid], 1e-6) for oid in self._non_reference_ids])
        n = len(p_free)
        return np.diag(1.0 / p_free) + (1.0 / p_ref) * np.ones((n, n))

    def covariance(self, current_p_mid: Dict[str, float]) -> np.ndarray:
        """
        Main entry point, always probability-space, safe to plug into
        the A-S skew formula. EWMA once warmed up, converted from its
        native ALR-space estimate via the local delta-method Jacobian
        at current_p_mid, otherwise the structural prior.
        """
        if self._n_obs >= self.MIN_OBS_FOR_EWMA:
            jac = self._alr_jacobian(current_p_mid)
            jac_inv = np.linalg.inv(jac)
            return jac_inv @ self._cov @ jac_inv.T

        return structural_probability_covariance(current_p_mid, self._outcome_ids, self._reference_id)
