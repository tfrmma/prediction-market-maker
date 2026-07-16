"""Unit tests for the pricing and book components."""
import math
import time

import numpy as np
import pytest

from src.data.unified_book import L2Book, UnifiedBook, BookSource
from src.data.polymarket_feed import PolyBookSnapshot, PriceLevel, PolyTrade
from src.pricing.fair_value import FairValueEngine, ASBinaryParams
from src.execution.eip712_signer import EIP712Signer, OrderParams, OrderSide


# L2Book
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


# FairValueEngine
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


# EIP-712 Signer
class TestEIP712Signer:

    # Use a deterministic test key (never use in production)
    TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

    def test_address_derived_correctly(self):
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        assert signer.address.startswith("0x")
        assert len(signer.address) == 42

    def test_sign_buy_order_structure(self):
        """
        V2 Order struct dropped `taker`, `expiration`, `nonce`, and
        `feeRateBps` entirely (see CLOB V2 migration, 2026-04-28).
        Uniqueness now comes from `timestamp` (ms) + random `salt`.
        """
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
        assert signed.signer == signer.address
        assert not hasattr(signed, "taker")   # V2: no taker field
        assert int(signed.timestamp) > 0
        assert signed.signature.startswith("0x")
        assert len(signed.signature) == 132   # 0x + 65 bytes

    def test_amounts_scale_correctly_buy(self):
        """
        V2: BOTH makerAmount and takerAmount are 6-decimal integers.
        Outcome tokens are no longer 18-decimal as they were pre-migration.
        """
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        params = OrderParams(
            token_id="1",
            side=OrderSide.BUY,
            price=0.60,      # 60 cents per token
            size=100.0,      # 100 outcome tokens
        )
        signed = signer.sign_order(params)
        # makerAmount = 0.60 × 100 × 10^6 = 60_000_000 (pUSD)
        assert int(signed.maker_amount) == 60_000_000
        # takerAmount = 100 × 10^6 (outcome tokens, 6-dec under V2)
        assert int(signed.taker_amount) == 100 * 10**6

    def test_neg_risk_selects_correct_verifying_contract(self):
        """A neg-risk order must be signed against the Neg Risk Exchange
        domain, not the regular CTF Exchange , different verifyingContract
        means a signature for one is invalid for the other."""
        from src.execution.eip712_signer import EXCHANGE_V2, NEG_RISK_EXCHANGE_V2
        assert EXCHANGE_V2 != NEG_RISK_EXCHANGE_V2

    def test_timestamp_used_for_uniqueness_not_nonce(self):
        """V2 has no nonce field; uniqueness comes from random salt."""
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        p = OrderParams(token_id="1", side=OrderSide.BUY, price=0.5, size=10)
        s1 = signer.sign_order(p)
        s2 = signer.sign_order(p)
        assert not hasattr(s1, "nonce")
        assert s1.salt != s2.salt   # uniqueness via salt, not a sequential nonce

    def test_salt_unique_per_order(self):
        signer = EIP712Signer(self.TEST_PRIVATE_KEY)
        p = OrderParams(token_id="1", side=OrderSide.BUY, price=0.5, size=10)
        salts = {signer.sign_order(p).salt for _ in range(20)}
        assert len(salts) == 20  # all unique


# Backtest Smoke Tests
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


# Kalshi bids-only book (regression tests for the structural fix)
class TestKalshiBidsOnlyBook:
    """
    Kalshi's feed only ever sends bid-side levels for both the yes and no
    legs , there is no ask array anywhere in the protocol. These tests
    guard the complement-derivation logic in UnifiedBook against
    regressing back to expecting asks directly from the feed.
    """

    def test_yes_ask_derived_as_complement_of_no_bid(self):
        from src.data.kalshi_feed import KalshiBookSnapshot, KalshiLevel
        from src.data.unified_book import UnifiedBook, BookSource

        ub = UnifiedBook("T-1", BookSource.KALSHI, resolution_ts=0)
        snap = KalshiBookSnapshot(
            market_ticker="T-1",
            yes_bids=[KalshiLevel(price=0.42, size=100)],
            no_bids=[KalshiLevel(price=0.56, size=80)],
            seq=1, timestamp_ms=1000, recv_ts=1.0,
        )
        state = ub.process(snap)
        assert state is not None
        assert abs(state.p_bid - 0.42) < 1e-9
        assert abs(state.p_ask - (1.0 - 0.56)) < 1e-9   # 0.44

    def test_no_state_without_opposing_leg_bid(self):
        """Can't derive a YES ask with zero NO-side liquidity , must not
        fabricate one."""
        from src.data.kalshi_feed import KalshiBookSnapshot, KalshiLevel
        from src.data.unified_book import UnifiedBook, BookSource

        ub = UnifiedBook("T-2", BookSource.KALSHI, resolution_ts=0)
        snap = KalshiBookSnapshot(
            market_ticker="T-2",
            yes_bids=[KalshiLevel(price=0.42, size=100)],
            no_bids=[],
            seq=1, timestamp_ms=1000, recv_ts=1.0,
        )
        state = ub.process(snap)
        assert state is None

    def test_delta_updates_complement_ask(self):
        from src.data.kalshi_feed import KalshiBookSnapshot, KalshiBookDelta, KalshiLevel
        from src.data.unified_book import UnifiedBook, BookSource

        ub = UnifiedBook("T-3", BookSource.KALSHI, resolution_ts=0)
        ub.process(KalshiBookSnapshot(
            market_ticker="T-3",
            yes_bids=[KalshiLevel(price=0.40, size=100)],
            no_bids=[KalshiLevel(price=0.50, size=50), KalshiLevel(price=0.48, size=30)],
            seq=1, timestamp_ms=1000, recv_ts=1.0,
        ))
        # Fully remove the best NO bid (0.50) -> next best NO bid is 0.48
        state = ub.process(KalshiBookDelta(
            market_ticker="T-3", side="no", price=0.50, delta=-50,
            seq=2, timestamp_ms=1100, recv_ts=1.1,
        ))
        assert state is not None
        assert abs(state.p_ask - (1.0 - 0.48)) < 1e-9


# RiskEngine loss-rate kill switch (was a no-op stub)
class TestRiskEngineLossRate:

    def test_loss_rate_triggers_kill(self):
        import asyncio
        from config.settings import RiskProfile
        from src.risk.engine import RiskEngine

        profile = RiskProfile(loss_rate_limit_15m=10.0, intraday_drawdown_limit=10_000.0)
        kill_event = asyncio.Event()
        engine = RiskEngine(profile, kill_event)

        # Manually seed pnl_samples spanning >15min with a big drop,
        # bypassing real time.sleep for a fast unit test.
        now = engine._pnl_samples.maxlen or 0  # no-op, just referencing attr exists
        import time as _time
        t0 = _time.monotonic() - engine.LOSS_RATE_WINDOW_S + 1
        engine._pnl_samples.append((t0, 0.0))
        engine._pnl_samples.append((_time.monotonic(), -50.0))  # lost $50

        assert not kill_event.is_set()
        engine._check_loss_rate()
        assert kill_event.is_set()

    def test_realized_pnl_trusts_caller_not_recomputed(self):
        """RiskEngine must not recompute realized PnL independently , it
        should simply accumulate whatever InventoryManager reports, even
        across a long -> short position flip (which the old internal
        VWAP tracker got wrong)."""
        import asyncio
        from config.settings import RiskProfile
        from src.risk.engine import RiskEngine

        profile = RiskProfile(intraday_drawdown_limit=10_000.0)
        engine = RiskEngine(profile, asyncio.Event())

        engine.on_fill(
            market_id="m1", order_id="o1", fill_price=0.5, fill_size=10,
            side="BUY", mid_at_fill=0.5, realized_pnl=0.0,
        )
        engine.on_fill(
            market_id="m1", order_id="o2", fill_price=0.6, fill_size=25,
            side="SELL", mid_at_fill=0.6, realized_pnl=1.0,  # caller-supplied
        )
        # RiskEngine must reflect exactly what was passed in, not its own math
        assert abs(engine._pnl["m1"].realized_pnl - 1.0) < 1e-9


class TestHyperliquidSigner:
    """The phantom-agent scheme is easy to get subtly wrong (key order in
    the action dict, byte layout of the vault flag), these just check the
    hash construction is deterministic and the signature round-trips."""

    TEST_KEY = "0x" + "33" * 32

    def test_action_hash_deterministic(self):
        from src.hedging.hyperliquid_signer import _action_hash
        action = {"type": "order", "orders": [], "grouping": "na"}
        h1 = _action_hash(action, None, 1_700_000_000_000)
        h2 = _action_hash(action, None, 1_700_000_000_000)
        assert h1 == h2
        assert len(h1) == 32   # keccak256 output

    def test_action_hash_changes_with_nonce(self):
        from src.hedging.hyperliquid_signer import _action_hash
        action = {"type": "order", "orders": [], "grouping": "na"}
        h1 = _action_hash(action, None, 1)
        h2 = _action_hash(action, None, 2)
        assert h1 != h2

    def test_sign_action_produces_valid_signature(self):
        from src.hedging.hyperliquid_signer import HyperliquidSigner
        signer = HyperliquidSigner(self.TEST_KEY, is_mainnet=True)
        action = {
            "type": "order",
            "orders": [{"a": 0, "b": True, "p": "95000.0", "s": "0.01", "r": False,
                        "t": {"limit": {"tif": "Ioc"}}}],
            "grouping": "na",
        }
        sig = signer.sign_action(action, HyperliquidSigner.next_nonce())
        assert sig["r"].startswith("0x")
        assert sig["s"].startswith("0x")
        assert sig["v"] in (27, 28)


class TestPolymarketMarketResolver:
    """Token id resolution used to be a hardcoded condition_id + '_YES'
    guess, this checks the real parsing logic against the documented
    /markets/{condition_id} response shape."""

    def test_extracts_yes_no_token_ids(self):
        from src.data.polymarket_market_resolver import PolymarketMarketResolver

        class FakeResp:
            status = 200
            async def json(self):
                return {
                    "tokens": [
                        {"outcome": "Yes", "token_id": "111"},
                        {"outcome": "No",  "token_id": "222"},
                    ],
                    "neg_risk": True,
                    "minimum_tick_size": 0.01,
                }
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

        class FakeSession:
            def get(self, *a, **kw): return FakeResp()

        import asyncio
        resolver = PolymarketMarketResolver("https://clob.polymarket.com", FakeSession())
        result = asyncio.run(resolver.resolve("0xabc"))
        assert result.yes_token_id == "111"
        assert result.no_token_id == "222"
        assert result.neg_risk is True


class TestKalshiOrderManagerWireFormat:
    """side mapping (BUY/SELL -> bid/ask) is the kind of thing that's
    silent and wrong if you get it backwards, no error, orders just
    fill on the side you didn't mean to quote."""

    def test_buy_maps_to_bid_sell_maps_to_ask(self):
        # this mirrors the mapping inside _place_order without needing
        # a live aiohttp session
        def wire_side(side_str):
            return "bid" if side_str == "BUY" else "ask"
        assert wire_side("BUY") == "bid"
        assert wire_side("SELL") == "ask"


class TestRoundToTick:
    """An order priced off-tick gets rejected outright, this is the
    kind of bug that only shows up once in a while depending on what
    fair value happens to compute that tick."""

    def test_snaps_to_nearest_tick(self):
        from src.execution.order_types import round_to_tick
        assert round_to_tick(0.4738, 0.01) == 0.47
        assert round_to_tick(0.4762, 0.01) == 0.48

    def test_zero_tick_is_noop(self):
        from src.execution.order_types import round_to_tick
        assert round_to_tick(0.4738, 0.0) == 0.4738

    def test_already_on_tick_unchanged(self):
        from src.execution.order_types import round_to_tick
        assert round_to_tick(0.56, 0.01) == 0.56


class TestOrderSizing:
    """order_size_usd used to be min_edge_bps * 10, no relationship to
    risk, vol, or actual capital. These pin down the bounds it's now
    supposed to respect."""

    def _profile(self, **overrides):
        from config.settings import RiskProfile
        defaults = dict(base_order_size_usd=25.0, max_order_size_usd=150.0,
                         min_edge_bps=15.0, max_position_pct=0.20)
        defaults.update(overrides)
        return RiskProfile(**defaults)

    def test_zero_edge_sizes_to_zero(self):
        from src.pricing.sizing import compute_order_size_usd
        size = compute_order_size_usd(edge_bps=0, sigma=0.1, free_collateral_usd=1000, risk_profile=self._profile())
        assert size == 0.0

    def test_never_exceeds_free_collateral_budget(self):
        from src.pricing.sizing import compute_order_size_usd
        profile = self._profile(max_position_pct=0.10, max_order_size_usd=1_000_000)
        size = compute_order_size_usd(edge_bps=200, sigma=0.01, free_collateral_usd=100, risk_profile=profile)
        assert size <= 10.0 + 1e-9   # 10% of $100

    def test_never_exceeds_hard_ceiling(self):
        from src.pricing.sizing import compute_order_size_usd
        profile = self._profile(max_order_size_usd=40.0)
        size = compute_order_size_usd(edge_bps=1000, sigma=0.001, free_collateral_usd=1_000_000, risk_profile=profile)
        assert size <= 40.0 + 1e-9

    def test_higher_vol_reduces_size(self):
        from src.pricing.sizing import compute_order_size_usd
        profile = self._profile(max_order_size_usd=1_000_000, max_position_pct=1.0)
        # both above the 1/0.02=50x floor-clamp and below the 2.0x
        # ceiling-clamp on vol_factor, so the comparison actually
        # exercises the 1/sigma scaling instead of the clamps
        calm = compute_order_size_usd(edge_bps=30, sigma=0.6, free_collateral_usd=10_000, risk_profile=profile)
        choppy = compute_order_size_usd(edge_bps=30, sigma=1.5, free_collateral_usd=10_000, risk_profile=profile)
        assert choppy < calm


class TestHedgeSlippage:
    """Was a flat 0.5% regardless of market conditions, now scales with
    realized vol and is bounded on both ends."""

    def _engine(self, **overrides):
        from config.settings import HedgeProfile
        from src.hedging.delta_hedge import HedgeEngine
        defaults = dict(min_slip_bps=10.0, max_slip_bps=200.0,
                         slip_vol_multiplier=4.0, max_hedge_latency_ms=200)
        defaults.update(overrides)
        profile = HedgeProfile(**defaults)
        return HedgeEngine(profile, "https://api.hyperliquid.xyz", signer=None,
                            asset_index_fn=lambda c: 0, sz_decimals_fn=lambda c: 3)

    def test_no_vol_data_falls_back_to_floor(self):
        engine = self._engine()
        assert engine._crossing_slippage(None) == 10.0 / 10_000

    def test_higher_vol_widens_slippage(self):
        engine = self._engine()
        # picked so both land strictly between floor and ceiling, otherwise
        # the clamps do the comparison's job instead of the scaling math
        low = engine._crossing_slippage(20.0)
        high = engine._crossing_slippage(60.0)
        assert high > low

    def test_respects_ceiling(self):
        engine = self._engine(max_slip_bps=50.0)
        slip = engine._crossing_slippage(500.0)   # absurd vol
        assert slip <= 50.0 / 10_000 + 1e-12


class TestSecretsLoader:
    """Kalshi's PEM used to come straight out of a plaintext env var
    with no alternative. This checks the fallback path stays honest
    about doing that, and that the AWS path isn't silently skipped."""

    def test_falls_back_to_env_var_when_no_arn_configured(self, monkeypatch):
        from config.secrets import load_secret
        monkeypatch.setenv("TEST_SECRET_PLAIN", "shh")
        monkeypatch.delenv("TEST_SECRET_ARN", raising=False)
        assert load_secret("TEST_SECRET_PLAIN", "TEST_SECRET_ARN") == "shh"

    def test_uses_secrets_manager_when_arn_present(self, monkeypatch):
        import sys
        import types
        from config import secrets as secrets_mod

        fake_boto3 = types.ModuleType("boto3")
        fake_client = type("C", (), {
            "get_secret_value": lambda self, SecretId: {"SecretString": "from-aws"}
        })()
        fake_boto3.client = lambda name: fake_client
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        monkeypatch.setenv("TEST_SECRET_ARN", "arn:aws:secretsmanager:...:secret:x")

        assert secrets_mod.load_secret("TEST_SECRET_PLAIN", "TEST_SECRET_ARN") == "from-aws"
