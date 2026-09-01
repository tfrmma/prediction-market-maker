"""
Inventory skew + spread, Guéant-style multi-asset extension of
fair_value.py's binary A-S formula:

    r = p_fair - q*gamma*sigma_sq*(T-t)
    half_spread = gamma*sigma_sq*(T-t) + (1/gamma)*ln(1+gamma/k)

Vector form for the non-reference outcomes (Sigma from covariance.py):

    q_tilde_i = q_i - q_ref
    skew_vec = -gamma * (Sigma @ q_tilde) * (T-t)

q_tilde is inventory netted against the reference outcome. Mutually
exclusive outcomes means the payout vector is one-hot, so total P&L is
q_ref + sum_i(q_i - q_ref)*X_i, only the excess over the reference
position carries variance, that's also why Sigma itself only exists
for N-1 free outcomes (see covariance.py).

The reference outcome's own skew comes for free rather than needing
its own matrix row: Sigma is the covariance of a one-hot draw, its
rows sum to zero, which means skew across all N outcomes (reference
included) sums to exactly zero. So skew_ref = -sum(skew_i for i !=
ref), and the whole reservation-price vector ends up summing to 1
automatically, same invariant the softmax layer keeps for p_fair.
Checked this holds exactly for N=2, where it reduces to fair_value.py's
plain binary formula with inventory_q = q_A - q_B, see test_core.py.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import structlog

from src.pricing.categorical_fair_value import CategoricalFairValueResult, CategoricalASParams
from src.pricing.covariance import CategoricalCovarianceEstimator

logger = structlog.get_logger(__name__)


@dataclass
class CategoricalQuoteResult:
    event_id: str
    ts: float

    p_reservation: Dict[str, float]     # p_fair + inventory_skew, sums to 1
    inventory_skew: Dict[str, float]    # sums to 0 across the basket
    marginal_variance: Dict[str, float] # Sigma_ii per outcome, diagnostic
    half_spread: Dict[str, float]
    bid_quote: Dict[str, float]
    ask_quote: Dict[str, float]

    should_quote: bool


class CategoricalSkewEngine:
    """Stateless. Takes the softmax layer's output plus inventory and
    Sigma, produces per-outcome reservation prices and quotes."""

    SECONDS_PER_YEAR: float = 365.25 * 24 * 3600
    TICK: float = 0.01

    def compute(
        self,
        fv_result: CategoricalFairValueResult,
        inventory_q: Dict[str, float],   # per-outcome signed contracts, missing = 0
        covariance_estimator: CategoricalCovarianceEstimator,
        params: CategoricalASParams,
        ttres_s: float,
    ) -> CategoricalQuoteResult:
        outcome_ids = list(fv_result.p_fair.keys())
        ref = covariance_estimator.reference_id
        non_ref_ids = [oid for oid in outcome_ids if oid != ref]
        ttres_years = max(0.0, ttres_s) / self.SECONDS_PER_YEAR

        sigma = covariance_estimator.covariance(fv_result.p_fair)   # (N-1)x(N-1)

        q_ref = inventory_q.get(ref, 0.0)
        q_tilde = np.array([inventory_q.get(oid, 0.0) - q_ref for oid in non_ref_ids])

        skew_vec = -params.gamma * (sigma @ q_tilde) * ttres_years
        skew_vec = np.clip(skew_vec, -params.max_skew, params.max_skew)

        inv_skew = {oid: float(skew_vec[i]) for i, oid in enumerate(non_ref_ids)}
        inv_skew[ref] = -sum(inv_skew.values())   # Sigma's rows sum to zero, see module docstring

        marginal_var = {non_ref_ids[i]: float(sigma[i, i]) for i in range(len(non_ref_ids))}
        # Reference's own marginal variance isn't in the reduced matrix, fall back
        # to the plain structural formula, same one the binary engine uses.
        p_ref = fv_result.p_fair[ref]
        marginal_var[ref] = p_ref * (1.0 - p_ref)

        p_reservation: Dict[str, float] = {}
        half_spread: Dict[str, float] = {}
        bid_quote: Dict[str, float] = {}
        ask_quote: Dict[str, float] = {}

        for oid in outcome_ids:
            r = np.clip(fv_result.p_fair[oid] + inv_skew[oid], 1e-4, 1.0 - 1e-4)
            p_reservation[oid] = float(r)

            k_i = params.k_per_outcome.get(oid, params.k)
            if ttres_years > 0 and params.gamma > 0 and k_i > 0:
                term1 = params.gamma * marginal_var[oid] * ttres_years
                term2 = (1.0 / params.gamma) * math.log(1.0 + params.gamma / k_i)
                hs = term1 + term2
            else:
                hs = params.min_half_spread
            hs = max(params.min_half_spread, min(params.max_half_spread, hs))
            half_spread[oid] = hs

            bid = np.clip(r - hs, 0.01, 0.98)
            ask = np.clip(r + hs, 0.02, 0.99)
            bid = round(round(bid / self.TICK) * self.TICK, 4)
            ask = round(round(ask / self.TICK) * self.TICK, 4)
            if bid >= ask:
                mid_q = (bid + ask) / 2
                bid = round(mid_q - self.TICK, 4)
                ask = round(mid_q + self.TICK, 4)
            bid_quote[oid] = float(bid)
            ask_quote[oid] = float(ask)

        return CategoricalQuoteResult(
            event_id=fv_result.event_id,
            ts=time.monotonic(),
            p_reservation=p_reservation,
            inventory_skew=inv_skew,
            marginal_variance=marginal_var,
            half_spread=half_spread,
            bid_quote=bid_quote,
            ask_quote=ask_quote,
            should_quote=fv_result.should_quote,
        )
