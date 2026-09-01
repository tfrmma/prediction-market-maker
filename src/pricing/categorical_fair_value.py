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
from dataclasses import dataclass, field
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
