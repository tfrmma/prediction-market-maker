"""
Backtest simulator for the categorical (N-outcome) MM engine, same
spirit as backtest_simulator.py: synthetic environment to sanity check
parameter choices, not a replay of real fills.

Path generation: N independent GARCH-vol score processes, softmaxed at
every tick into probabilities that always sum to 1, same mechanism
CategoricalFairValueEngine itself uses. A shared common shock on top of
the idiosyncratic ones injects excess correlation between outcomes
beyond what mutual exclusivity already implies (softmax alone gives
built-in negative correlation, one outcome rising mechanically pulls
the others down), so the EWMA covariance estimator actually has
something beyond the structural prior to pick up.

At resolution_s the outcome with the highest final probability "wins",
every position gets marked out at $1 (winner) or $0 (everyone else).

Runs the real pricing pipeline (CategoricalBookAggregator ->
CategoricalCovarianceEstimator -> CategoricalFairValueEngine ->
CategoricalSkewEngine), not a reimplementation of it, this file is
market simulation and fill bookkeeping only.

Heads up on the PnL number specifically: this fill model has no
adverse selection, arrivals react to quote distance only, never to
where the "true" price is about to move, so a market maker capturing
spread on effectively every tick over a long run produces a PnL figure
that looks unrealistically good. Same caveat backtest_simulator.py
already carries for the binary case. Useful for checking the pipeline
holds together (consistency, no crashes, correct settlement), not for
estimating real strategy performance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.data.unified_book import MarketState, BookSource
from src.data.categorical_book import CategoricalBookAggregator, CategoricalMarketState
from src.pricing.covariance import CategoricalCovarianceEstimator
from src.pricing.categorical_fair_value import CategoricalFairValueEngine, CategoricalASParams
from src.pricing.categorical_skew import CategoricalSkewEngine
from src.inventory.manager import InventoryManager
from config.settings import RiskProfile

logger = structlog.get_logger(__name__)


@dataclass
class CategoricalSimConfig:
    outcome_ids: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    p0: Optional[Dict[str, float]] = None   # defaults to uniform 1/N if omitted

    resolution_s: float = 86400.0
    tick_s: float = 1.0

    score_vol: float = 0.04             # per-tick vol of each outcome's underlying score
    common_shock_weight: float = 0.30   # fraction of variance shared across outcomes

    A_bid: float = 1.0
    A_ask: float = 1.0
    k_arrival: float = 1.8

    order_size_usd: float = 30.0
    max_position_contracts: float = 500.0   # simple per-outcome risk cap, real position
                                             # limits live in InventoryManager/compute_order_size_usd,
                                             # this backtest doesn't wire those in, so we need
                                             # something to keep a long run from ballooning
    random_seed: int = 42


@dataclass
class CategoricalSimPath:
    outcome_ids: List[str]
    probs: np.ndarray        # (n_steps+1, n_outcomes), each row sums to 1
    tick_s: float
    winner: str
    rng: np.random.Generator
    cfg: CategoricalSimConfig

    def simulate_arrivals(self, bid: float, ask: float, mid: float, dt: float) -> Tuple[bool, bool]:
        """Same Poisson fill model as backtest_simulator.py's SimPath,
        applied independently per outcome."""
        cfg = self.cfg
        delta_bid = mid - bid
        delta_ask = ask - mid
        lam_bid = cfg.A_bid * math.exp(-cfg.k_arrival * max(delta_bid, 0))
        lam_ask = cfg.A_ask * math.exp(-cfg.k_arrival * max(delta_ask, 0))
        p_bid = 1 - math.exp(-lam_bid * dt)
        p_ask = 1 - math.exp(-lam_ask * dt)
        u = self.rng.uniform(0, 1, 2)
        return (u[0] < p_bid, u[1] < p_ask)


class CategoricalMarketSimulator:
    """Pure data generation, no strategy logic."""

    def __init__(self, cfg: CategoricalSimConfig):
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.random_seed)

    def generate_path(self) -> CategoricalSimPath:
        cfg = self._cfg
        n_outcomes = len(cfg.outcome_ids)
        n_steps = int(cfg.resolution_s / cfg.tick_s)

        p0 = cfg.p0 or {oid: 1.0 / n_outcomes for oid in cfg.outcome_ids}
        scores = np.zeros((n_steps + 1, n_outcomes))
        scores[0] = [math.log(max(p0[oid], 1e-6)) for oid in cfg.outcome_ids]

        idio_vol = cfg.score_vol * math.sqrt(1.0 - cfg.common_shock_weight)
        common_vol = cfg.score_vol * math.sqrt(cfg.common_shock_weight)

        for t in range(1, n_steps + 1):
            common_shock = self._rng.standard_normal() * common_vol
            idio_shock = self._rng.standard_normal(n_outcomes) * idio_vol
            scores[t] = scores[t - 1] + common_shock + idio_shock

        shifted = scores - scores.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        probs = weights / weights.sum(axis=1, keepdims=True)

        winner = cfg.outcome_ids[int(np.argmax(probs[-1]))]

        return CategoricalSimPath(
            outcome_ids=cfg.outcome_ids, probs=probs, tick_s=cfg.tick_s,
            winner=winner, rng=self._rng, cfg=cfg,
        )


@dataclass
class CategoricalFillRecord:
    outcome_id: str
    t: float
    side: str
    price: float
    size: float


@dataclass
class CategoricalBacktestResult:
    total_pnl: float
    realized_pnl: float
    settlement_pnl: float

    n_fills: int
    fill_rate: float
    n_quotes: int

    max_avg_arb_gap: float    # mean |arb_gap_mint_and_dump| + |arb_gap_buy_basket| over the run
    winner: str

    per_outcome_fills: Dict[str, int]
    sharpe: float
    max_drawdown: float

    def summary(self) -> str:
        return (
            f"PnL: ${self.total_pnl:.2f} | Sharpe: {self.sharpe:.2f} | "
            f"MaxDD: ${self.max_drawdown:.2f} | fills: {self.n_fills} | "
            f"avg |arb_gap|: {self.max_avg_arb_gap:.4f} | winner: {self.winner}"
        )


class CategoricalBacktestRunner:
    """Runs the real categorical pricing pipeline against a simulated
    N-outcome path, tracks fills and settlement PnL at the basket level."""

    def __init__(self, cfg: CategoricalSimConfig):
        self._cfg = cfg
        self._fv_engine = CategoricalFairValueEngine()
        self._skew_engine = CategoricalSkewEngine()

    def run(
        self,
        params: Optional[CategoricalASParams] = None,
        path: Optional[CategoricalSimPath] = None,
    ) -> CategoricalBacktestResult:
        cfg = self._cfg
        params = params or CategoricalASParams()
        if path is None:
            path = CategoricalMarketSimulator(cfg).generate_path()

        outcome_ids = path.outcome_ids
        n_steps = path.probs.shape[0] - 1
        dt = path.tick_s

        aggregator = CategoricalBookAggregator("SIM", outcome_ids)
        cov_est = CategoricalCovarianceEstimator("SIM", outcome_ids, correlation_window_s=1800.0)
        inv = InventoryManager(RiskProfile())
        for oid in outcome_ids:
            inv.register_market(oid, "kalshi", event_id="SIM")

        # CategoricalFairValueEngine's staleness check compares book_ts_ms
        # against wall-clock time.time(), anchor simulated ticks to real
        # "now" so a fast backtest doesn't look infinitely stale to it.
        sim_start_wall = time.time()

        fills: List[CategoricalFillRecord] = []
        per_outcome_fills: Dict[str, int] = {oid: 0 for oid in outcome_ids}
        n_quotes = 0
        arb_gap_samples: List[float] = []
        pnl_series: List[float] = [0.0]
        realized_pnl = 0.0

        for t in range(n_steps):
            ttres_s = max(0.0, cfg.resolution_s - t * dt)
            row = path.probs[t]

            basket = None
            for i, oid in enumerate(outcome_ids):
                p = float(row[i])
                state = MarketState(
                    market_id=oid, source=BookSource.KALSHI, ts=float(t),
                    p_mid=p, p_bid=p - 0.005, p_ask=p + 0.005, spread=0.01,
                    resolution_ts=int(cfg.resolution_s), time_to_resolution_s=ttres_s,
                    book_ts_ms=int((sim_start_wall + t * dt) * 1000),
                )
                inv.update_mid(oid, p)
                basket = aggregator.update(oid, state)

            if basket is None:
                pnl_series.append(pnl_series[-1])
                continue

            cov_est.observe(basket)
            arb_gap_samples.append(abs(basket.arb_gap_mint_and_dump) + abs(basket.arb_gap_buy_basket))

            fv_result = self._fv_engine.compute(basket, params)
            if not fv_result.should_quote:
                pnl_series.append(pnl_series[-1])
                continue

            quote = self._skew_engine.compute(
                fv_result,
                inventory_q={oid: inv.get_net_qty(oid) for oid in outcome_ids},
                covariance_estimator=cov_est,
                params=params,
                ttres_s=ttres_s,
            )
            n_quotes += 1

            for oid, state in basket.outcomes.items():
                bid, ask, mid = quote.bid_quote[oid], quote.ask_quote[oid], state.p_mid
                bid_hit, ask_hit = path.simulate_arrivals(bid, ask, mid, dt)
                net_qty = inv.get_net_qty(oid)

                if bid_hit and net_qty < cfg.max_position_contracts:
                    size = min(cfg.order_size_usd / max(bid, 0.01), cfg.max_position_contracts - net_qty)
                    if size > 0:
                        inv.on_fill(oid, "BUY", bid, size, collateral_used=bid * size)
                        fills.append(CategoricalFillRecord(oid, t * dt, "BUY", bid, size))
                        per_outcome_fills[oid] += 1

                if ask_hit and net_qty > -cfg.max_position_contracts:
                    size = min(cfg.order_size_usd / max(1 - ask, 0.01), cfg.max_position_contracts + net_qty)
                    if size > 0:
                        inv.on_fill(oid, "SELL", ask, size, collateral_used=(1 - ask) * size)
                        fills.append(CategoricalFillRecord(oid, t * dt, "SELL", ask, size))
                        per_outcome_fills[oid] += 1

            unrealized = sum(
                inv.get_position(oid).unrealized_pnl for oid in outcome_ids
                if inv.get_position(oid) is not None
            )
            pnl_series.append(realized_pnl + unrealized)

        # Resolution: winner pays $1, everyone else pays $0
        settlement_pnl = 0.0
        for oid in outcome_ids:
            settlement_pnl += inv.on_resolution(oid, resolved_yes=(oid == path.winner))
        realized_pnl = sum(inv._realized_pnl.values())   # includes settlement, see on_resolution
        final_pnl = realized_pnl
        pnl_series.append(final_pnl)

        pnl_arr = np.array(pnl_series)
        returns = np.diff(pnl_arr)
        sharpe = 0.0
        if np.std(returns) > 1e-12:
            sharpe = float(np.mean(returns) / np.std(returns) * math.sqrt(365.25 * 24 * 3600 / dt))
        running_max = np.maximum.accumulate(pnl_arr)
        max_dd = float(np.min(pnl_arr - running_max))

        return CategoricalBacktestResult(
            total_pnl=final_pnl,
            realized_pnl=realized_pnl,
            settlement_pnl=settlement_pnl,
            n_fills=len(fills),
            fill_rate=len(fills) / max(n_steps * dt, 1),
            n_quotes=n_quotes,
            max_avg_arb_gap=float(np.mean(arb_gap_samples)) if arb_gap_samples else 0.0,
            winner=path.winner,
            per_outcome_fills=per_outcome_fills,
            sharpe=sharpe,
            max_drawdown=max_dd,
        )


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Categorical Prediction MM Backtest")
    parser.add_argument("--n-outcomes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--k", type=float, default=1.5)
    args = parser.parse_args()

    outcome_ids = [chr(ord("A") + i) for i in range(args.n_outcomes)]
    cfg = CategoricalSimConfig(outcome_ids=outcome_ids, random_seed=args.seed)
    params = CategoricalASParams(gamma=args.gamma, k=args.k)

    runner = CategoricalBacktestRunner(cfg)
    result = runner.run(params)
    print(f"\n{args.n_outcomes}-outcome categorical backtest: {result.summary()}")
    print(f"Fills per outcome: {result.per_outcome_fills}")
