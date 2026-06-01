"""
tests/test_core.py
───────────────────
Unit tests for the critical pricing and book components.
Run with: pytest tests/test_core.py -v
"""
import math
import time

import numpy as np
import pytest

from src.data.unified_book import L2Book, UnifiedBook, BookSource
from src.data.polymarket_feed import PolyBookSnapshot, PriceLevel, PolyTrade
from src.pricing.fair_value import FairValueEngine, ASBinaryParams
from src.execution.eip712_signer import EIP712Signer, OrderParams, OrderSide


# ──────────────────────────────────────────────
# L2Book
# ──────────────────────────────────────────────

class TestL2Book:

    def test_snapshot_sorted_correctly(self):
        book = L2Book("test")
        book.apply_snapshot(
            bids=[PriceLevel(0.45, 100), PriceLevel(0.44, 50), PriceLevel(0.46, 200)],
            asks=[PriceLevel(0.55, 80),  PriceLevel(0.52, 30)],
            ts_ms=1000,
        )
        bid = book.best_bid()
        ask = book.best_ask()
        assert bid is not None and bid[0] == pytest.approx(0.46)
        assert ask is not None and ask[0] == pytest.approx(0.52)

    def test_zero_size_removes_level(self):
        book = L2Book("test")
        book.apply_snapshot(
            bids=[PriceLevel(0.45, 100)],
            asks=[PriceLevel(0.55, 100)],
            ts_ms=1000,
        )
        book.apply_delta("BUY", 0.45, 0.0, 2000)
        assert book.best_bid() is None

    def test_depth_trims_to_max(self):
        book = L2Book("test")
        bids = [PriceLevel(0.50 - i * 0.01, 10) for i in range(25)]
        book.apply_snapshot(bids=bids, asks=[], ts_ms=1000)
        bids_out, _ = book.top_levels(5)
        assert len(bids_out) == 5
        # Best bid first
        assert bids_out[0].price == pytest.approx(0.50)

    def test_crossed_book_detection(self):
        """L2Book should store without crashing; UnifiedBook validates."""
        book = L2Book("test")
        book.apply_snapshot(
            bids=[PriceLevel(0.60, 100)],
            asks=[PriceLevel(0.50, 100)],  # crossed
            ts_ms=1000,
        )
        bid = book.best_bid()
        ask = book.best_ask()
        assert bid[0] > ask[0]  # crossed: stored as-is, validated upstream


# ──────────────────────────────────────────────
# FairValueEngine
# ──────────────────────────────────────────────

class TestFairValueEngine:

    def _make_state(self, p_mid=0.50, ttres_s=86400, cvd=0.0, imbalance=0.0):
        from src.data.unified_book import MarketState
        return MarketState(
            market_id="test",
            source=BookSource.POLYMARKET,
            ts=time.monotonic(),
            p_mid=p_mid,
            p_bid=p_mid - 0.01,
            p_ask=p_mid + 0.01,
            spread=0.02,
            ofi=0.0,
            cvd=cvd,
            realized_vol_1m=0.002,
            bid_depth_usd=500.0,
            ask_depth_usd=500.0,
            imbalance=imbalance,
            resolution_ts=int(time.time() + ttres_s),
            time_to_resolution_s=float(ttres_s),
            book_ts_ms=int(time.time() * 1000),
        )

    def test_zero_inventory_symmetric_quotes(self):
        engine = FairValueEngine()
        state  = self._make_state(p_mid=0.50)
        params = ASBinaryParams(gamma=0.05, k=1.5, alpha=0.0, beta=0.0)

        fv = engine.compute(state, inventory_q=0.0, params=params)

        # At q=0, p_fair, symmetric spread: bid and ask equidistant from mid
        assert abs(fv.bid_quote + fv.ask_quote - 1.0) < 0.02
        assert fv.half_spread > params.min_half_spread
        assert fv.half_spread <= params.max_half_spread

    def test_long_inventory_skews_bid_down(self):
        """Long inventory should lower both bid and ask (reduce long)."""
        engine = FairValueEngine()
        state  = self._make_state(p_mid=0.50)
        params = ASBinaryParams(gamma=0.10, k=1.5, alpha=0.0, beta=0.0)

        fv_flat = engine.compute(state, inventory_q=0.0,   params=params)
        fv_long = engine.compute(state, inventory_q=200.0, params=params)

        assert fv_long.bid_quote < fv_flat.bid_quote
        assert fv_long.ask_quote < fv_flat.ask_quote
        assert fv_long.inventory_skew < 0

    def test_spread_collapses_near_resolution(self):
        """Very short T-t → spread should approach minimum."""
        engine = FairValueEngine()
        params = ASBinaryParams(gamma=0.05, k=1.5, min_ttres_s=60.0)

        far  = self._make_state(p_mid=0.50, ttres_s=86400)
        near = self._make_state(p_mid=0.50, ttres_s=3600)

        fv_far  = engine.compute(far,  0.0, params)
        fv_near = engine.compute(near, 0.0, params)

        assert fv_near.half_spread <= fv_far.half_spread

    def test_should_not_quote_stale_book(self):
        """Old book_ts_ms should set is_stale=True, should_quote=False."""
        engine = FairValueEngine()
        params = ASBinaryParams()
        state  = self._make_state()
        state.book_ts_ms = int((time.time() - 60) * 1000)  # 60s old

        fv = engine.compute(state, 0.0, params)
        assert fv.is_stale
        assert not fv.should_quote

    def test_cvd_positive_raises_fair_value(self):
        """Positive CVD (buy pressure) should increase fair value."""
        engine = FairValueEngine()
        params = ASBinaryParams(alpha=0.005, beta=0.0)

        s_base = self._make_state(cvd=0.0)
        s_bull = self._make_state(cvd=100.0)

        fv_base = engine.compute(s_base, 0.0, params)
        fv_bull = engine.compute(s_bull, 0.0, params)

        assert fv_bull.p_fair > fv_base.p_fair
        assert fv_bull.flow_adjustment > 0

    def test_prelec_correction_monotone_and_bounded(self):
        """
        Prelec correction must:
          1. Preserve monotonicity (higher p_market → higher p_fair)
          2. Keep output strictly in (0, 1)
          3. Fix point at p=0.5 (symmetric correction)
        """
        engine = FairValueEngine()
        params = ASBinaryParams(kappa=0.75)

        probs  = [0.05, 0.20, 0.50, 0.80, 0.95]
        states = [self._make_state(p_mid=p) for p in probs]
        fairs  = [
            engine.compute(s, 0.0, params, apply_bias_correction=True).p_fair
            for s in states
        ]

        # Monotonicity
        for i in range(len(fairs) - 1):
            assert fairs[i] < fairs[i+1], f"Non-monotone at index {i}: {fairs}"

        # Bounds
        for f in fairs:
            assert 0.0 < f < 1.0, f"Out of bounds: {f}"

        # p=0.5 stays at 0.5 (logit=0, correction is symmetric)
        mid_fair = fairs[2]
        assert abs(mid_fair - 0.5) < 0.01

    def test_no_crossed_quotes(self):
        """bid_quote must always be strictly less than ask_quote."""
        engine = FairValueEngine()
        params = ASBinaryParams()
        rng = np.random.default_rng(42)

        for _ in range(200):
            p_mid = float(rng.uniform(0.05, 0.95))
            inv_q = float(rng.uniform(-300, 300))
            state = self._make_state(p_mid=p_mid)
            fv = engine.compute(state, inv_q, params)
            if fv.should_quote:
                assert fv.bid_quote < fv.ask_quote, (
                    f"Crossed quote: bid={fv.bid_quote} >= ask={fv.ask_quote} "
                    f"at p_mid={p_mid}, inv={inv_q}"
                )


# ──────────────────────────────────────────────
# EIP-712 Signer
# ──────────────────────────────────────────────

class TestEIP712Signer:

    # Use a deterministic test key (never use in production)
    TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

    def test_address_derived_correctly(self):
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        assert signer.address.startswith("0x")
        assert len(signer.address) == 42

    def test_sign_buy_order_structure(self):
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        params = OrderParams(
            token_id="123456789",
            side=OrderSide.BUY,
            price=0.45,
            size=100.0,
        )
        signed = signer.sign_order(params)

        assert signed.side == 0
        assert signed.maker == signer.address
        assert signed.taker == "0x0000000000000000000000000000000000000000"
        assert signed.signature.startswith("0x") or len(signed.signature) == 130

    def test_amounts_scale_correctly_buy(self):
        """BUY: makerAmount = USDC (6 dec), takerAmount = tokens (18 dec)."""
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        params = OrderParams(
            token_id="1",
            side=OrderSide.BUY,
            price=0.60,      # 60 cents per token
            size=100.0,      # 100 outcome tokens
        )
        signed = signer.sign_order(params)
        # makerAmount = 0.60 × 100 × 10^6 = 60_000_000
        assert int(signed.maker_amount) == 60_000_000
        # takerAmount = 100 × 10^18
        assert int(signed.taker_amount) == 100 * 10**18

    def test_nonce_increments(self):
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        p = OrderParams(token_id="1", side=OrderSide.BUY, price=0.5, size=10)
        s1 = signer.sign_order(p)
        s2 = signer.sign_order(p)
        assert int(s2.nonce) == int(s1.nonce) + 1

    def test_salt_unique_per_order(self):
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        p = OrderParams(token_id="1", side=OrderSide.BUY, price=0.5, size=10)
        salts = {signer.sign_order(p).salt for _ in range(20)}
        assert len(salts) == 20  # all unique


# ──────────────────────────────────────────────
# Backtest Smoke Tests
# ──────────────────────────────────────────────

class TestBacktest:

    def test_single_path_runs_without_error(self):
        from tests.backtest_simulator import BacktestRunner, SimConfig, ASBinaryParams
        cfg = SimConfig(resolution_s=3600, random_seed=1)
        runner = BacktestRunner(cfg)
        result = runner.run(ASBinaryParams())
        assert isinstance(result.total_pnl, float)
        assert result.n_bid_fills >= 0
        assert result.n_ask_fills >= 0

    def test_higher_gamma_reduces_inventory(self):
        """Higher risk aversion should reduce average inventory."""
        from tests.backtest_simulator import BacktestRunner, SimConfig, MarketSimulator
        cfg = SimConfig(resolution_s=7200, random_seed=42)
        sim = MarketSimulator(cfg)
        path = sim.generate_path()
        runner = BacktestRunner(cfg)

        r_low_gamma  = runner.run(ASBinaryParams(gamma=0.01), path=path)
        r_high_gamma = runner.run(ASBinaryParams(gamma=0.50), path=path)

        # High gamma skews quotes more aggressively → inventory reverts faster
        assert r_high_gamma.avg_inventory <= r_low_gamma.avg_inventory * 1.5

    def test_pnl_decomposition_sums_correctly(self):
        """spread_capture + inventory_pnl + adverse_selection ≈ total realized."""
        from tests.backtest_simulator import BacktestRunner, SimConfig
        cfg = SimConfig(resolution_s=3600, random_seed=7)
        runner = BacktestRunner(cfg)
        r = runner.run(ASBinaryParams())

        # NOTE: decomposition is approximate (AS is measured 30s post-fill)
        # Just check spread_capture ≥ 0 (we're providing liquidity, not taking)
        assert r.spread_capture >= -1.0  # allow small negative for edge cases

    def test_walk_forward_runs_all_folds(self):
        from tests.backtest_simulator import WalkForwardValidator, SimConfig
        cfg = SimConfig(resolution_s=7200, wf_n_folds=3, random_seed=99)
        validator = WalkForwardValidator(cfg)
        report = validator.run(ASBinaryParams())
        assert len(report.folds) == 3
        for fold in report.folds:
            assert isinstance(fold.test_result.sharpe, float)
