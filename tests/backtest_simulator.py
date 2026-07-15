"""
Backtest simulator for the MM engine. Not a replay of real fills, this
generates a synthetic environment to sanity check parameter choices.

Order arrivals: independent Poisson processes per side, intensity decays
with spread distance per Avellaneda-Stoikov (2008):

    lambda(delta) = A * exp(-k*delta)

delta is half-spread in probability units, A is the base fill rate at
delta=0, k is the decay constant.

Mid-price follows a GARCH(1,1) so we get fat tails and vol clustering
instead of boring Brownian motion:

    sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
    eps_t = sigma_t * z_t,  z_t ~ N(0,1)

Toxic flow: with probability p_toxic per arrival, an informed trader
shows up and permanently shifts mid by delta_adverse, this is what drives
adverse selection in the PnL decomposition. Crude but good enough to
catch a gamma that's badly miscalibrated.

At resolution_ts the market settles to 1 or 0 based on which side of 0.5
the final mid landed on, and all open positions get marked out.

Walk-forward validation splits the run into N rolling windows, fits
gamma/k/alpha/beta in-sample, scores out-of-sample, and reports Sharpe,
max drawdown and adverse selection rate per fold.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.data.unified_book import MarketState, BookSource
from src.pricing.fair_value import FairValueEngine, ASBinaryParams, ParameterCalibrator
from src.inventory.manager import InventoryManager, FillProcessor
from src.risk.engine import RiskEngine
from config.settings import RiskProfile

logger = structlog.get_logger(__name__)


# Simulation Parameters
@dataclass
class SimConfig:
    # Market
    p0: float = 0.50            # initial probability
    resolution_s: float = 86400.0  # 24 hours
    tick_s: float = 1.0         # simulation time step (seconds)

    # Arrival model
    A_bid: float = 1.2          # base arrival rate on bid (fills/sec at δ=0)
    A_ask: float = 1.2          # base arrival rate on ask
    k_arrival: float = 1.8      # arrival decay coefficient

    # GARCH vol parameters
    garch_omega: float = 1e-7   # long-run variance contribution
    garch_alpha: float = 0.10   # ARCH term
    garch_beta:  float = 0.85   # GARCH term
    init_vol: float = 0.002     # initial σ per time step

    # Toxic flow
    p_toxic: float = 0.05       # fraction of arrivals that are toxic
    toxic_impact: float = 0.01  # mid-shift from one toxic event

    # Strategy
    order_size_usd: float = 50.0

    # Simulation
    random_seed: int = 42
    n_paths: int = 1            # Monte Carlo paths (set >1 for distribution)

    # Walk-forward
    wf_n_folds: int = 5
    wf_train_frac: float = 0.70


# Market Simulator
class MarketSimulator:
    """
    Generates synthetic market data using GARCH + Poisson arrivals.
    Pure data generation , no strategy logic.
    """

    def __init__(self, cfg: SimConfig):
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.random_seed)

    def generate_path(self) -> "SimPath":
        """
        Generate a complete market path from t=0 to t=resolution_s.
        Returns SimPath object with all tick data.
        """
        cfg = self._cfg
        n_steps = int(cfg.resolution_s / cfg.tick_s)

        # GARCH mid-price path
        mids = np.zeros(n_steps + 1)
        vols = np.zeros(n_steps + 1)
        mids[0] = cfg.p0
        vols[0] = cfg.init_vol

        z = self._rng.standard_normal(n_steps)

        for t in range(1, n_steps + 1):
            # GARCH(1,1) variance update
            eps_prev = vols[t-1] * z[t-1]
            var_t = (cfg.garch_omega +
                     cfg.garch_alpha * eps_prev**2 +
                     cfg.garch_beta  * vols[t-1]**2)
            vols[t] = math.sqrt(max(var_t, 1e-10))

            # Mid-price update (arithmetic Brownian on probability)
            shock = vols[t] * z[t-1] if t > 0 else 0.0
            mids[t] = np.clip(mids[t-1] + shock, 0.001, 0.999)

        # Toxic flow injection
        n_toxic = self._rng.binomial(n_steps, cfg.p_toxic)
        toxic_times = sorted(self._rng.choice(n_steps, size=n_toxic, replace=False))
        toxic_direction = self._rng.choice([-1, 1], size=n_toxic)

        for i, t in enumerate(toxic_times):
            impact = cfg.toxic_impact * toxic_direction[i]
            # Toxic event permanently shifts mid from t onward
            mids[t:] = np.clip(mids[t:] + impact, 0.001, 0.999)

        # Resolution
        resolved_yes = mids[-1] >= 0.5

        return SimPath(
            mids=mids,
            vols=vols,
            tick_s=cfg.tick_s,
            toxic_times=set(toxic_times),
            resolved_yes=resolved_yes,
            rng=self._rng,
            cfg=cfg,
        )


# Simulation Path
@dataclass
class FillRecord:
    t: float
    side: str
    price: float
    size: float
    mid: float
    is_toxic: bool


@dataclass
class SimPath:
    mids:       np.ndarray
    vols:       np.ndarray
    tick_s:     float
    toxic_times: set
    resolved_yes: bool
    rng:         np.random.Generator
    cfg:         SimConfig

    def simulate_arrivals(
        self,
        bid_quote: float,
        ask_quote: float,
        t_idx: int,
        dt: float,
    ) -> Tuple[bool, bool]:
        """
        Poisson arrivals for one time step.
        Returns (bid_filled, ask_filled).
        
        P(fill | δ, dt) = 1 - exp(-λ(δ)·dt)
        λ(δ) = A·exp(-k·δ)
        """
        cfg = self.cfg
        mid = self.mids[t_idx]

        # Spread distances
        delta_bid = mid - bid_quote   # how far below mid we're quoting
        delta_ask = ask_quote - mid   # how far above mid

        # Only positive distances lead to fills
        lambda_bid = cfg.A_bid * math.exp(-cfg.k_arrival * max(delta_bid, 0))
        lambda_ask = cfg.A_ask * math.exp(-cfg.k_arrival * max(delta_ask, 0))

        p_bid_fill = 1 - math.exp(-lambda_bid * dt)
        p_ask_fill = 1 - math.exp(-lambda_ask * dt)

        u = self.rng.uniform(0, 1, 2)
        return (u[0] < p_bid_fill, u[1] < p_ask_fill)


# Backtest Runner
@dataclass
class BacktestResult:
    # Performance
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float

    # PnL decomposition
    spread_capture: float
    inventory_pnl: float
    adverse_selection: float

    # Risk metrics
    sharpe: float
    max_drawdown: float
    calmar: float                # total_pnl / max_drawdown

    # Fill statistics
    n_bid_fills: int
    n_ask_fills: int
    n_toxic_fills: int
    fill_rate: float              # fills / time

    # Quoting statistics
    n_ticks: int
    n_quotes: int
    quote_rate: float

    # Inventory stats
    max_inventory: float
    avg_inventory: float
    time_flat_pct: float          # fraction of time at zero inventory

    # Strategy params used
    params: ASBinaryParams

    def summary(self) -> str:
        return (
            f"PnL: ${self.total_pnl:.2f} | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"MaxDD: ${self.max_drawdown:.2f} | "
            f"SC: ${self.spread_capture:.2f} | "
            f"AS: ${self.adverse_selection:.2f} | "
            f"AS%: {self.adverse_selection/max(abs(self.spread_capture),1e-9)*100:.1f}%"
        )


class BacktestRunner:
    """
    Run full simulation with a given parameter set.
    Used both for single runs and walk-forward validation.
    """

    # Adverse selection measurement window
    AS_WINDOW_TICKS: int = 30   # 30 seconds @ 1s ticks

    def __init__(self, cfg: SimConfig):
        self._cfg = cfg
        self._pricer = FairValueEngine()

    def run(
        self,
        params: ASBinaryParams,
        path: Optional[SimPath] = None,
        seed_override: Optional[int] = None,
    ) -> BacktestResult:
        """
        Run one backtest path with given parameters.
        """
        if path is None:
            sim_cfg = self._cfg
            if seed_override is not None:
                import copy
                sim_cfg = copy.copy(sim_cfg)
                sim_cfg.random_seed = seed_override
            sim = MarketSimulator(sim_cfg)
            path = sim.generate_path()

        n_steps = len(path.mids) - 1
        dt = path.tick_s
        cfg = self._cfg

        # State
        inventory_q: float = 0.0
        realized_pnl: float = 0.0
        spread_capture: float = 0.0
        adverse_selection: float = 0.0

        # PnL time series for Sharpe/drawdown
        pnl_series: List[float] = [0.0]
        inventory_series: List[float] = [0.0]

        # Fill records for AS measurement
        fills: List[FillRecord] = []
        n_bid_fills = n_ask_fills = n_toxic_fills = n_quotes = 0

        # Pending AS measurement: [(fill_idx, fill_record), ...]
        pending_as: List[Tuple[int, FillRecord]] = []

        resolution_ts = cfg.resolution_s

        for t in range(n_steps):
            mid = path.mids[t]
            vol = path.vols[t]
            ttres_s = max(0.0, resolution_ts - t * dt)

            # Build synthetic MarketState
            state = MarketState(
                market_id="sim_market",
                source=BookSource.POLYMARKET,
                ts=float(t),
                p_mid=mid,
                p_bid=mid - 0.005,
                p_ask=mid + 0.005,
                spread=0.010,
                ofi=0.0,
                cvd=0.0,
                realized_vol_1m=vol * math.sqrt(60.0 / dt),
                bid_depth_usd=200.0,
                ask_depth_usd=200.0,
                imbalance=0.0,
                resolution_ts=int(resolution_ts),
                time_to_resolution_s=ttres_s,
                book_ts_ms=int(t * dt * 1000),
            )

            # Compute fair value
            fv = self._pricer.compute(
                state=state,
                inventory_q=inventory_q,
                params=params,
            )

            if not fv.should_quote:
                pnl_series.append(pnl_series[-1])
                inventory_series.append(inventory_q)
                continue

            n_quotes += 1
            bid_q = fv.bid_quote
            ask_q = fv.ask_quote

            # Simulate arrivals
            bid_hit, ask_hit = path.simulate_arrivals(bid_q, ask_q, t, dt)

            is_toxic = t in path.toxic_times

            if bid_hit and abs(inventory_q) < params.q_max:
                fill_size = cfg.order_size_usd / max(bid_q, 0.01)
                fill_size = min(fill_size, params.q_max - inventory_q)

                if fill_size > 0:
                    inventory_q += fill_size
                    sc = (mid - bid_q) * fill_size
                    spread_capture += sc
                    realized_pnl += sc  # approximate: realized at mid

                    fr = FillRecord(t=t*dt, side="BUY", price=bid_q,
                                    size=fill_size, mid=mid, is_toxic=is_toxic)
                    fills.append(fr)
                    pending_as.append((t, fr))
                    n_bid_fills += 1
                    if is_toxic:
                        n_toxic_fills += 1

            if ask_hit and abs(inventory_q) < params.q_max:
                fill_size = cfg.order_size_usd / max(1 - ask_q, 0.01)
                fill_size = min(fill_size, params.q_max + inventory_q)

                if fill_size > 0:
                    inventory_q -= fill_size
                    sc = (ask_q - mid) * fill_size
                    spread_capture += sc
                    realized_pnl += sc

                    fr = FillRecord(t=t*dt, side="SELL", price=ask_q,
                                    size=fill_size, mid=mid, is_toxic=is_toxic)
                    fills.append(fr)
                    pending_as.append((t, fr))
                    n_ask_fills += 1
                    if is_toxic:
                        n_toxic_fills += 1

            # Measure adverse selection
            ready_as = [(idx, f) for idx, f in pending_as if t - idx >= self.AS_WINDOW_TICKS]
            for fill_idx, fr in ready_as:
                mid_then = path.mids[min(fill_idx + self.AS_WINDOW_TICKS, n_steps)]
                sign = 1 if fr.side == "BUY" else -1
                as_cost = (fr.mid - mid_then) * fr.size * sign
                adverse_selection += as_cost
            pending_as = [(idx, f) for idx, f in pending_as
                          if t - idx < self.AS_WINDOW_TICKS]

            # Mark-to-market total PnL
            unrealized = inventory_q * (mid - (
                sum(f.price * f.size for f in fills if f.side=="BUY") /
                max(sum(f.size for f in fills if f.side=="BUY"), 1e-9)
            ))
            total_pnl_t = realized_pnl + unrealized
            pnl_series.append(total_pnl_t)
            inventory_series.append(inventory_q)

        # Resolution settlement
        resolution_price = 1.0 if path.resolved_yes else 0.0
        if inventory_q > 0:
            avg_entry = (sum(f.price * f.size for f in fills if f.side == "BUY") /
                         max(sum(f.size for f in fills if f.side == "BUY"), 1e-9))
            settlement_pnl = (resolution_price - avg_entry) * inventory_q
        elif inventory_q < 0:
            avg_entry = (sum(f.price * f.size for f in fills if f.side == "SELL") /
                         max(sum(f.size for f in fills if f.side == "SELL"), 1e-9))
            settlement_pnl = (avg_entry - resolution_price) * abs(inventory_q)
        else:
            settlement_pnl = 0.0

        realized_pnl += settlement_pnl
        final_pnl = realized_pnl
        pnl_series.append(final_pnl)

        # Risk metrics
        pnl_arr = np.array(pnl_series)
        returns = np.diff(pnl_arr)
        sharpe = (np.mean(returns) / max(np.std(returns), 1e-9)) * math.sqrt(
            365.25 * 24 * 3600 / dt
        )

        running_max = np.maximum.accumulate(pnl_arr)
        drawdowns   = pnl_arr - running_max
        max_dd      = float(np.min(drawdowns))
        calmar = final_pnl / abs(max_dd) if abs(max_dd) > 1e-9 else float("inf")

        inv_arr = np.array(inventory_series)

        return BacktestResult(
            total_pnl=final_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=final_pnl - realized_pnl,
            spread_capture=spread_capture,
            inventory_pnl=settlement_pnl,
            adverse_selection=adverse_selection,
            sharpe=sharpe,
            max_drawdown=max_dd,
            calmar=calmar,
            n_bid_fills=n_bid_fills,
            n_ask_fills=n_ask_fills,
            n_toxic_fills=n_toxic_fills,
            fill_rate=(n_bid_fills + n_ask_fills) / max(n_steps * dt, 1),
            n_ticks=n_steps,
            n_quotes=n_quotes,
            quote_rate=n_quotes / max(n_steps, 1),
            max_inventory=float(np.max(np.abs(inv_arr))),
            avg_inventory=float(np.mean(np.abs(inv_arr))),
            time_flat_pct=float(np.mean(inv_arr == 0)),
            params=params,
        )


# Walk-Forward Validator
@dataclass
class WalkForwardFold:
    fold: int
    train_result: BacktestResult
    test_result:  BacktestResult
    params_used:  ASBinaryParams

    def degradation(self) -> float:
        """Sharpe degradation IS/OOS. > 0.5 suggests overfitting."""
        if self.train_result.sharpe == 0:
            return float("inf")
        return 1.0 - self.test_result.sharpe / self.train_result.sharpe


@dataclass
class WalkForwardReport:
    folds: List[WalkForwardFold]
    base_params: ASBinaryParams
    sim_cfg: SimConfig

    @property
    def mean_oos_sharpe(self) -> float:
        return float(np.mean([f.test_result.sharpe for f in self.folds]))

    @property
    def mean_oos_pnl(self) -> float:
        return float(np.mean([f.test_result.total_pnl for f in self.folds]))

    @property
    def mean_degradation(self) -> float:
        return float(np.mean([f.degradation() for f in self.folds]))

    def print_summary(self) -> None:
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION REPORT")
        print("="*70)
        print(f"{'Fold':<6} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
              f"{'OOS PnL':>10} {'Degrad':>8} {'AS%':>8}")
        print("-"*70)
        for f in self.folds:
            tr = f.train_result
            ts = f.test_result
            as_pct = ts.adverse_selection / max(abs(ts.spread_capture), 1e-9) * 100
            print(
                f"{f.fold:<6} {tr.sharpe:>10.2f} {ts.sharpe:>11.2f} "
                f"${ts.total_pnl:>9.2f} {f.degradation():>8.2%} {as_pct:>7.1f}%"
            )
        print("-"*70)
        print(f"{'MEAN':<6} {'':>10} {self.mean_oos_sharpe:>11.2f} "
              f"${self.mean_oos_pnl:>9.2f} {self.mean_degradation:>8.2%}")
        print("="*70)

        # Quality assessment
        if self.mean_oos_sharpe < 1.0:
            print("⚠  OOS Sharpe < 1.0 , strategy not viable without reparametrization")
        elif self.mean_oos_sharpe < 2.0:
            print("⚡ OOS Sharpe 1-2 , marginal; reduce adverse selection costs")
        else:
            print("✓  OOS Sharpe > 2.0 , strategy viable; target Sharpe > 3.0 in live")

        if self.mean_degradation > 0.5:
            print("⚠  Degradation > 50% , significant overfitting detected")


class WalkForwardValidator:
    """
    Walk-forward parameter stability test.
    
    Protocol:
      1. Split simulation timeline into N folds
      2. For each fold: run calibration on IS, evaluate on OOS
      3. Calibration = grid search over (γ, k) space
      4. Report IS/OOS Sharpe degradation per fold
    """

    # Simple grid for γ and k calibration
    GAMMA_GRID = [0.01, 0.03, 0.05, 0.10, 0.20]
    K_GRID     = [0.5, 1.0, 1.5, 2.0, 3.0]

    def __init__(self, cfg: SimConfig):
        self._cfg = cfg
        self._runner = BacktestRunner(cfg)

    def run(self, base_params: ASBinaryParams) -> WalkForwardReport:
        cfg = self._cfg
        n_folds = cfg.wf_n_folds

        # Generate a shared base path (all folds use same mid-price path)
        sim = MarketSimulator(cfg)
        full_path = sim.generate_path()

        n_total = len(full_path.mids) - 1
        fold_size = n_total // n_folds
        train_size = int(fold_size * cfg.wf_train_frac)
        test_size  = fold_size - train_size

        folds: List[WalkForwardFold] = []

        for i in range(n_folds):
            fold_start = i * fold_size
            fold_end   = fold_start + fold_size

            # Slice path
            train_path = self._slice_path(full_path, fold_start, fold_start + train_size)
            test_path  = self._slice_path(full_path, fold_start + train_size, fold_end)

            # Grid search on IS
            best_params = base_params
            best_sharpe = -float("inf")

            for gamma in self.GAMMA_GRID:
                for k in self.K_GRID:
                    candidate = ASBinaryParams(
                        gamma=gamma,
                        k=k,
                        alpha=base_params.alpha,
                        beta=base_params.beta,
                        kappa=base_params.kappa,
                    )
                    result = self._runner.run(candidate, path=train_path)
                    if result.sharpe > best_sharpe:
                        best_sharpe = result.sharpe
                        best_params = candidate

            # Evaluate on OOS
            train_result = self._runner.run(best_params, path=train_path)
            test_result  = self._runner.run(best_params, path=test_path)

            folds.append(WalkForwardFold(
                fold=i + 1,
                train_result=train_result,
                test_result=test_result,
                params_used=best_params,
            ))

            logger.info(
                "wf_fold_complete",
                fold=i+1,
                is_sharpe=round(train_result.sharpe, 2),
                oos_sharpe=round(test_result.sharpe, 2),
                best_gamma=best_params.gamma,
                best_k=best_params.k,
            )

        return WalkForwardReport(
            folds=folds,
            base_params=base_params,
            sim_cfg=cfg,
        )

    @staticmethod
    def _slice_path(path: SimPath, start: int, end: int) -> SimPath:
        """Extract a contiguous sub-path."""
        return SimPath(
            mids=path.mids[start:end+1],
            vols=path.vols[start:end+1],
            tick_s=path.tick_s,
            toxic_times={t - start for t in path.toxic_times if start <= t < end},
            resolved_yes=path.resolved_yes,
            rng=path.rng,
            cfg=path.cfg,
        )


# Monte Carlo Runner
class MonteCarloRunner:
    """
    Run N independent paths to estimate PnL distribution.
    Reports: mean Sharpe, Sharpe std, P(total_pnl > 0), VaR, CVaR.
    """

    def __init__(self, cfg: SimConfig, n_paths: int = 100):
        self._cfg = cfg
        self._n_paths = n_paths
        self._runner = BacktestRunner(cfg)

    def run(self, params: ASBinaryParams) -> Dict:
        results = []
        for i in range(self._n_paths):
            cfg = self._cfg
            sim = MarketSimulator(
                SimConfig(**{**cfg.__dict__, "random_seed": cfg.random_seed + i})
            )
            path = sim.generate_path()
            r = self._runner.run(params, path=path)
            results.append(r)

        pnls   = np.array([r.total_pnl for r in results])
        sharpes = np.array([r.sharpe for r in results])

        var_95 = float(np.percentile(pnls, 5))
        cvar_95 = float(np.mean(pnls[pnls <= var_95]))

        summary = {
            "n_paths":       self._n_paths,
            "mean_pnl":      float(np.mean(pnls)),
            "std_pnl":       float(np.std(pnls)),
            "mean_sharpe":   float(np.mean(sharpes)),
            "std_sharpe":    float(np.std(sharpes)),
            "prob_profit":   float(np.mean(pnls > 0)),
            "var_95":        var_95,
            "cvar_95":       cvar_95,
            "mean_sc":       float(np.mean([r.spread_capture for r in results])),
            "mean_as":       float(np.mean([r.adverse_selection for r in results])),
            "mean_as_rate":  float(np.mean([
                r.adverse_selection / max(abs(r.spread_capture), 1e-9)
                for r in results
            ])),
        }

        print("\n" + "="*60)
        print(f"MONTE CARLO RESULTS  (N={self._n_paths} paths)")
        print("="*60)
        for k, v in summary.items():
            print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")
        print("="*60)

        return summary


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prediction MM Backtest")
    parser.add_argument("--mode", choices=["single", "wf", "mc"], default="wf")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--k",     type=float, default=1.5)
    parser.add_argument("--p0",    type=float, default=0.50)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--paths", type=int,   default=200)
    args = parser.parse_args()

    cfg = SimConfig(p0=args.p0, random_seed=args.seed)
    params = ASBinaryParams(gamma=args.gamma, k=args.k)

    if args.mode == "single":
        runner = BacktestRunner(cfg)
        result = runner.run(params)
        print(f"\nSingle backtest: {result.summary()}")

    elif args.mode == "wf":
        validator = WalkForwardValidator(cfg)
        report = validator.run(params)
        report.print_summary()

    elif args.mode == "mc":
        mc = MonteCarloRunner(cfg, n_paths=args.paths)
        mc.run(params)
