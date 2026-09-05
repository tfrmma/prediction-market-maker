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
    outcome stays fixed between resets (last id in outcome_ids by
    default, or the highest-probability one if initial_p_mid is given),
    switching it invalidates the running EWMA's coordinate system, which
    is why that only ever happens through the explicit
    maybe_reset_reference() below, never silently.

    Two defenses against the reference (or any outcome) collapsing
    toward 0 mid-event, which is when the ALR<->probability conversion
    gets numerically dangerous:
      - covariance() checks every probability the conversion actually
        touches against MIN_SAFE_PROB before inverting the Jacobian,
        falls back to the (always well-behaved) structural prior rather
        than trust a near-singular inversion.
      - maybe_reset_reference(), called periodically from outside the
        hot path, swaps to whichever outcome currently has the highest
        probability once the current reference drops below
        MIN_SAFE_PROB, and resets the EWMA, old observations were in
        the old coordinate system and aren't valid in the new one.
    """

    MIN_OBS_FOR_EWMA = 30
    MIN_SAFE_PROB = 0.02   # below this, for any outcome the ALR machinery touches,
                            # covariance() distrusts the conversion and
                            # maybe_reset_reference() will swap the reference out

    def __init__(
        self,
        event_id: str,
        outcome_ids: List[str],
        correlation_window_s: float = 1800.0,
        initial_p_mid: Optional[Dict[str, float]] = None,
    ):
        if len(outcome_ids) < 2:
            raise ValueError(f"{event_id}: need at least 2 outcomes, got {outcome_ids}")
        self._event_id = event_id
        self._outcome_ids = list(outcome_ids)
        self._window_s = correlation_window_s
        self._log = logger.bind(component="categorical_covariance", event_id=event_id)

        if initial_p_mid:
            self._reference_id = max(outcome_ids, key=lambda oid: initial_p_mid.get(oid, 0.0))
        else:
            self._reference_id = outcome_ids[-1]
        self._non_reference_ids = [oid for oid in self._outcome_ids if oid != self._reference_id]

        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None   # native ALR-space EWMA state
        self._last_ts: Optional[float] = None
        self._n_obs = 0

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
        up and every probability the ALR conversion touches is above
        MIN_SAFE_PROB (converted via the local Jacobian at
        current_p_mid), structural prior otherwise, that prior never
        needs a matrix inversion so it's numerically safe regardless of
        how skewed current_p_mid is.

        Checked empirically before picking this over a condition-number
        threshold: the Jacobian's condition number scales with 1/p for
        whichever outcome (reference or not) is smallest, but it's
        already capped by the 1e-6 floor inside _alr_jacobian well
        before a generic condition-number cutoff would ever trigger, a
        direct probability floor is simpler and actually fires when it
        should.
        """
        if self._n_obs >= self.MIN_OBS_FOR_EWMA:
            relevant = [current_p_mid.get(self._reference_id, 0.0)] + [
                current_p_mid.get(oid, 0.0) for oid in self._non_reference_ids
            ]
            min_p = min(relevant)
            if min_p >= self.MIN_SAFE_PROB:
                jac = self._alr_jacobian(current_p_mid)
                jac_inv = np.linalg.inv(jac)
                return jac_inv @ self._cov @ jac_inv.T
            self._log.warning(
                "alr_conversion_unsafe",
                min_probability=min_p,
                threshold=self.MIN_SAFE_PROB,
                reference_id=self._reference_id,
            )

        return structural_probability_covariance(current_p_mid, self._outcome_ids, self._reference_id)

    def maybe_reset_reference(self, current_p_mid: Dict[str, float]) -> bool:
        """
        Call periodically (e.g. from a calibration loop), not from the
        hot path. If the current reference's probability has dropped
        below MIN_SAFE_PROB, switches to whichever outcome currently has
        the highest probability and resets the running EWMA, old
        observations were in the old reference's coordinate system and
        aren't meaningful in the new one. Returns whether a reset
        happened.
        """
        current_ref_p = current_p_mid.get(self._reference_id, 0.0)
        if current_ref_p >= self.MIN_SAFE_PROB:
            return False

        new_reference = max(self._outcome_ids, key=lambda oid: current_p_mid.get(oid, 0.0))
        if new_reference == self._reference_id:
            return False   # nothing better available right now, stay put

        self._log.warning(
            "categorical_covariance_reference_reset",
            old_reference=self._reference_id, old_reference_p=current_ref_p,
            new_reference=new_reference, new_reference_p=current_p_mid.get(new_reference),
        )
        self._reference_id = new_reference
        self._non_reference_ids = [oid for oid in self._outcome_ids if oid != self._reference_id]
        self._mean = None
        self._cov = None
        self._last_ts = None
        self._n_obs = 0
        return True
