"""
src/main.py
────────────
Main async orchestrator.  Wires all layers together.

Task graph:
  [PolymarketFeed] ─┐
                     ├→ [BookRegistry] → state_queue → [Strategy Loop]
  [KalshiFeed]     ─┘                                       │
                                                      [FairValueEngine]
                                                             │
                                                      [OrderManager] ──→ [Polymarket CLOB]
                                                             │
                                                      [HedgeEngine]  ──→ [Hyperliquid]
                                                             │
  [RiskEngine] ←─────────────────────────────────────────────┘
  (independent, owns kill_event)
"""
from __future__ import annotations

import asyncio
import signal
import sys
import time
from typing import Dict, List

import aiohttp
import structlog
import uvloop

from config.settings import Settings, get_settings, BookSource, Venue
from src.data.polymarket_feed import PolymarketFeed, PolyMarket
from src.data.kalshi_feed import KalshiFeed
from src.data.unified_book import BookRegistry, MarketState
from src.pricing.fair_value import FairValueEngine, ASBinaryParams, ParameterCalibrator
from src.execution.eip712_signer import EIP712Signer
from src.execution.order_manager import OrderManager, FlickeringFilter
from src.hedging.delta_hedge import HedgeEngine
from src.risk.engine import RiskEngine

logger = structlog.get_logger(__name__)


class Orchestrator:
    """
    Lifecycle manager for the full market making stack.
    """

    STRATEGY_LOOP_INTERVAL_S: float = 0.1   # 100ms quoting cycle
    CALIBRATION_INTERVAL_S: float  = 300.0  # re-calibrate params every 5 min

    def __init__(self, settings: Settings):
        self._cfg         = settings
        self._kill_event  = asyncio.Event()
        self._state_queue: asyncio.Queue[MarketState] = asyncio.Queue(maxsize=1000)
        self._feed_queue:  asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._health_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        # Per-market state
        self._inventory: Dict[str, float] = {}      # market_id → net contracts
        self._last_mid:  Dict[str, float] = {}
        self._params:    Dict[str, ASBinaryParams] = {}
        self._calibrators: Dict[str, ParameterCalibrator] = {}

        # Core components (initialized in setup())
        self._book_registry: BookRegistry = None
        self._pricing_engine = FairValueEngine()
        self._risk_engine:   RiskEngine = None
        self._order_manager: OrderManager = None
        self._hedge_engine:  HedgeEngine = None
        self._signer:        EIP712Signer = None

    async def run(self) -> None:
        """Start all subsystems and run until kill signal."""
        await self._setup()

        tasks = [
            asyncio.create_task(self._run_feeds(),      name="feeds"),
            asyncio.create_task(self._strategy_loop(),  name="strategy"),
            asyncio.create_task(self._risk_engine.run(), name="risk"),
            asyncio.create_task(self._calibration_loop(), name="calibrator"),
            asyncio.create_task(self._monitor_kill(),    name="kill_monitor"),
        ]

        logger.info("orchestrator_started", n_markets=len(self._cfg.markets))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("orchestrator_shutdown")
        finally:
            await self._shutdown_all(tasks)

    # ── Setup ─────────────────────────────────

    async def _setup(self) -> None:
        poly_creds = self._cfg.polymarket
        hl_creds   = self._cfg.hyperliquid

        # EIP-712 signer
        if poly_creds:
            self._signer = EIP712Signer(poly_creds.private_key)

        # Shared HTTP session
        self._http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=50, ssl=True),
            headers={"Content-Type": "application/json"},
        )

        # Order manager
        self._order_manager = OrderManager(
            http_session=self._http_session,
            signer=self._signer,
            risk_profile=next(iter(self._cfg.markets.values())).risk,
            rest_url=self._cfg.poly_rest_url,
            flickering_filter=FlickeringFilter(),
        )

        # Book registry
        self._book_registry = BookRegistry(self._state_queue)
        for mid, mconf in self._cfg.markets.items():
            source = (BookSource.POLYMARKET
                      if mconf.venue == Venue.POLYMARKET
                      else BookSource.KALSHI)
            self._book_registry.register(mid, source, mconf.resolution_ts)
            self._inventory[mid] = 0.0
            self._last_mid[mid]  = 0.0
            self._params[mid]    = ASBinaryParams()
            self._calibrators[mid] = ParameterCalibrator(ASBinaryParams())

        # Risk engine
        self._risk_engine = RiskEngine(
            risk_profile=next(iter(self._cfg.markets.values())).risk,
            kill_event=self._kill_event,
        )

        # Hedge engine
        if hl_creds:
            self._hedge_engine = HedgeEngine(
                profile=next(iter(self._cfg.markets.values())).hedge,
                hl_url="https://api.hyperliquid.xyz",
                hl_wallet=hl_creds.wallet_address,
                hl_private_key=hl_creds.private_key,
            )
            for mid in self._cfg.markets:
                self._hedge_engine.register_market(mid)

    # ── Feed runner ───────────────────────────

    async def _run_feeds(self) -> None:
        """Launch WebSocket feeds and route events to BookRegistry."""
        tasks = []

        # Polymarket feed
        poly_markets_cfg = [
            m for m in self._cfg.markets.values()
            if m.venue == Venue.POLYMARKET
        ]
        if poly_markets_cfg and self._cfg.polymarket:
            poly_markets = [
                PolyMarket(
                    condition_id=m.condition_id,
                    yes_token_id=m.condition_id + "_YES",  # placeholder; fetch from REST at startup
                    no_token_id=m.condition_id + "_NO",
                )
                for m in poly_markets_cfg
            ]
            feed = PolymarketFeed(
                ws_url=self._cfg.poly_ws_url,
                markets=poly_markets,
                api_key=self._cfg.polymarket.api_key,
                api_secret=self._cfg.polymarket.api_secret,
                api_passphrase=self._cfg.polymarket.api_passphrase,
                out_queue=self._feed_queue,
                health_queue=self._health_queue,
            )
            tasks.append(asyncio.create_task(feed.run(), name="poly_feed"))

        # Kalshi feed
        kalshi_cfg = [
            m for m in self._cfg.markets.values()
            if m.venue == Venue.KALSHI
        ]
        if kalshi_cfg and self._cfg.kalshi:
            feed = KalshiFeed(
                ws_url=self._cfg.kalshi_ws_url,
                tickers=[m.condition_id for m in kalshi_cfg],
                api_key_id=self._cfg.kalshi.api_key_id,
                private_key_pem=self._cfg.kalshi.private_key_pem,
                out_queue=self._feed_queue,
                health_queue=self._health_queue,
            )
            tasks.append(asyncio.create_task(feed.run(), name="kalshi_feed"))

        # Dispatch loop: feed_queue → BookRegistry
        async def dispatch():
            while not self._kill_event.is_set():
                try:
                    event = await asyncio.wait_for(
                        self._feed_queue.get(), timeout=1.0
                    )
                    await self._book_registry.process(event)
                    self._risk_engine.on_book_update(
                        "", time.monotonic()
                    )
                except asyncio.TimeoutError:
                    continue

        tasks.append(asyncio.create_task(dispatch(), name="dispatch"))
        await asyncio.gather(*tasks)

    # ── Strategy loop ─────────────────────────

    async def _strategy_loop(self) -> None:
        """
        Main quoting loop.
        Drains state_queue and updates quotes for each market.
        """
        while not self._kill_event.is_set():
            try:
                state: MarketState = await asyncio.wait_for(
                    self._state_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if self._kill_event.is_set():
                break

            mid_id = state.market_id
            mconf = self._cfg.markets.get(mid_id)
            if mconf is None:
                continue

            # Update risk engine mark
            self._risk_engine.on_market_update(mid_id, state.p_mid)

            # Flow delta for calibration
            prev_mid = self._last_mid.get(mid_id, state.p_mid)
            delta_mid = state.p_mid - prev_mid
            self._last_mid[mid_id] = state.p_mid
            self._calibrators[mid_id].observe(
                cvd=state.cvd,
                ofi_norm=state.imbalance,
                delta_mid_next=delta_mid,
            )

            # Compute fair value
            params = self._params[mid_id]
            apply_bias = (mconf.venue == Venue.KALSHI)
            fv = self._pricing_engine.compute(
                state=state,
                inventory_q=self._inventory.get(mid_id, 0.0),
                params=params,
                apply_bias_correction=apply_bias,
            )

            # Log key pricing metrics
            logger.debug(
                "quote_computed",
                market_id=mid_id,
                p_fair=round(fv.p_fair, 4),
                bid=round(fv.bid_quote, 4),
                ask=round(fv.ask_quote, 4),
                spread_bps=round(fv.half_spread * 2 * 10_000, 1),
                inv_skew=round(fv.inventory_skew, 5),
                cvd=round(state.cvd, 4),
                ofi=round(state.ofi, 5),
            )

            # Update quotes (order manager handles cancel/replace)
            if not self._kill_event.is_set():
                yes_token_id = mconf.condition_id + "_YES"  # resolved at startup
                await self._order_manager.update_quotes(
                    state=state,
                    fv=fv,
                    yes_token_id=yes_token_id,
                    inventory_q=self._inventory.get(mid_id, 0.0),
                    order_size_usd=mconf.risk.min_edge_bps * 10,  # size proportional to edge
                )

            # Trigger hedge if needed
            if self._hedge_engine and mconf.hedge.enabled and mconf.underlying_symbol:
                S_perp = 95_000.0  # TODO: pull from live CEX feed
                sigma_perp = 0.60  # TODO: compute from realized vol window
                K_strike = 100_000.0  # TODO: parse from market name

                instr = await self._hedge_engine.compute_and_hedge(
                    market_id=mid_id,
                    inventory_q=self._inventory.get(mid_id, 0.0),
                    p_mid=state.p_mid,
                    S_perp=S_perp,
                    K_strike=K_strike,
                    sigma_perp=sigma_perp,
                    T_res_s=state.time_to_resolution_s,
                    perp_symbol=f"{mconf.underlying_symbol}-PERP",
                )
                if instr:
                    await self._hedge_engine.execute_hedge(
                        self._http_session, instr, S_perp
                    )

    # ── Calibration loop ──────────────────────

    async def _calibration_loop(self) -> None:
        while not self._kill_event.is_set():
            await asyncio.sleep(self.CALIBRATION_INTERVAL_S)
            for mid_id, cal in self._calibrators.items():
                updated = cal.recalibrate()
                self._params[mid_id] = updated
                logger.info(
                    "params_updated",
                    market_id=mid_id,
                    alpha=round(updated.alpha, 6),
                    beta=round(updated.beta, 6),
                    gamma=round(updated.gamma, 5),
                )

    # ── Kill monitor ──────────────────────────

    async def _monitor_kill(self) -> None:
        await self._kill_event.wait()
        logger.critical(
            "KILL_SWITCH_ACTIVE",
            reason=self._risk_engine.status.kill_reason,
        )
        # Cancel all orders immediately
        for mid in self._cfg.markets:
            try:
                await self._order_manager._cancel_all(mid, reason="kill_switch")
            except Exception:
                pass

    # ── Shutdown ──────────────────────────────

    async def _shutdown_all(self, tasks) -> None:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._http_session:
            await self._http_session.close()
        logger.info("orchestrator_clean_shutdown")


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

async def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    settings = get_settings()
    orchestrator = Orchestrator(settings)

    loop = asyncio.get_event_loop()

    def handle_signal():
        logger.info("shutdown_signal_received")
        orchestrator._kill_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await orchestrator.run()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
