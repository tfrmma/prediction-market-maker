"""
Sigma for the categorical A-S skew: r = p_fair - gamma*(Sigma@q)*(T-t).
Needs probability units, same as p_fair/q, same as fair_value.py's
scalar p*(1-p) for the binary case.

Structural prior (no history needed): multinomial payout covariance
diag(p) - p*pᵀ with the reference outcome's row/column dropped, same
trick as a baseline category in dummy-variable regression.

Online EWMA runs internally in additive log-ratio (ALR) space, raw
probabilities are simplex-constrained so their covariance is singular.
ALR variance scales like 1/p_i though, wrong units for the skew
formula (caught this via a failing test, see test_core.py), so
covariance() always converts back to probability space via the local
delta-method Jacobian before returning.
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
    """Probability-space structural prior, multinomial covariance with the
    reference row/column dropped. No history required, cold-start fallback
    for CategoricalCovarianceEstimator.covariance()."""
    p = np.array([p_mid[oid] for oid in outcome_ids], dtype=float)
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()   # renormalize, should already be ~1 but don't trust upstream blindly

    sigma_p = multinomial_structural_covariance(p)

    ref_idx = outcome_ids.index(reference_id)
    keep = [i for i in range(len(outcome_ids)) if i != ref_idx]
    return sigma_p[np.ix_(keep, keep)]


class CategoricalCovarianceEstimator:
    """One instance per event. Feed it CategoricalMarketState snapshots,
    call covariance(current_p_mid) for an (N-1)x(N-1) probability-space
    matrix, EWMA once warmed up, structural prior otherwise. Reference
    outcome is fixed at construction (last id in outcome_ids), switching
    it mid-stream would break the running EWMA's coordinate system.

    TODO: if the reference outcome's own probability collapses toward 0
    well into the estimator's life, the ALR->probability conversion gets
    poorly conditioned. Haven't hit this in practice, worth a CLR/ILR
    basis instead of a fixed reference if it does.
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
        """Diagnostic: raw EWMA in native log-ratio space, None until
        warmed up. Use covariance() for anything feeding the skew formula."""
        if self._n_obs < self.MIN_OBS_FOR_EWMA:
            return None
        return self._cov

    def _alr_jacobian(self, current_p_mid: Dict[str, float]) -> np.ndarray:
        """Local Jacobian d(ALR)/d(p_free) at current_p_mid: diagonal
        1/p_i plus a constant 1/p_ref from the simplex constraint."""
        p_ref = max(current_p_mid[self._reference_id], 1e-6)
        p_free = np.array([max(current_p_mid[oid], 1e-6) for oid in self._non_reference_ids])
        n = len(p_free)
        return np.diag(1.0 / p_free) + (1.0 / p_ref) * np.ones((n, n))

    def covariance(self, current_p_mid: Dict[str, float]) -> np.ndarray:
        """Probability-space, safe for the skew formula. EWMA once warmed
        up (converted via the local Jacobian at current_p_mid), structural
        prior otherwise."""
        if self._n_obs >= self.MIN_OBS_FOR_EWMA:
            jac = self._alr_jacobian(current_p_mid)
            jac_inv = np.linalg.inv(jac)
            return jac_inv @ self._cov @ jac_inv.T

        return structural_probability_covariance(current_p_mid, self._outcome_ids, self._reference_id)
