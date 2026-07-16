"""
Orchestrator. Wires the feeds, book registry, pricing engine, order
manager, hedge engine and risk engine together and runs the strategy loop.

RiskEngine runs independently and owns kill_event; everything else checks
it before acting.
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

from config.settings import Settings, get_settings, Venue
from src.data.polymarket_feed import (
    PolymarketFeed, PolyMarket, PolymarketUserFeed, PolyOwnFill,
)
from src.data.polymarket_market_resolver import PolymarketMarketResolver, ResolvedMarket
from src.data.kalshi_feed import KalshiFeed, KalshiOwnFill
from src.data.unified_book import BookRegistry, MarketState, BookSource
from src.pricing.fair_value import FairValueEngine, ASBinaryParams, ParameterCalibrator
from src.execution.eip712_signer import EIP712Signer
from src.execution.order_manager import OrderManager
from src.execution.order_types import FlickeringFilter
from src.execution.polymarket_auth import PolyL2Auth, PolyL2Credentials
from src.execution.kalshi_auth import KalshiRsaSigner
from src.execution.kalshi_order_manager import KalshiOrderManager
from src.hedging.delta_hedge import HedgeEngine
from src.hedging.hyperliquid_signer import HyperliquidSigner
from src.hedging.hyperliquid_price_feed import HyperliquidPriceFeed
from src.risk.engine import RiskEngine
from src.inventory.manager import InventoryManager

logger = structlog.get_logger(__name__)


class Orchestrator:
    """Owns the lifecycle of the whole market making stack."""

    STRATEGY_LOOP_INTERVAL_S: float = 0.1   # 100ms quoting cycle
    CALIBRATION_INTERVAL_S: float  = 300.0  # re-calibrate params every 5 min

    def __init__(self, settings: Settings):
        self._cfg         = settings
        self._kill_event  = asyncio.Event()
        self._state_queue: asyncio.Queue[MarketState] = asyncio.Queue(maxsize=1000)
        self._feed_queue:  asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._fill_queue:  asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._health_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        # Per-market state
        self._last_mid:  Dict[str, float] = {}
        self._params:    Dict[str, ASBinaryParams] = {}
        self._calibrators: Dict[str, ParameterCalibrator] = {}

        # Core components (initialized in setup())
        self._book_registry: BookRegistry = None
        self._pricing_engine = FairValueEngine()
        self._risk_engine:   RiskEngine = None
        self._order_manager: OrderManager = None
        self._kalshi_order_manager: KalshiOrderManager = None
        self._hedge_engine:  HedgeEngine = None
        self._hl_price_feed: HyperliquidPriceFeed = None
        self._signer:        EIP712Signer = None
        # single source of truth for position/collateral
        self._inventory_mgr: InventoryManager = None
        # condition_id -> ResolvedMarket (real token ids, neg_risk, tick size)
        self._resolved_poly: Dict[str, ResolvedMarket] = {}
        # condition_id / kalshi ticker -> our market_id key. Fill events
        # come back keyed by the venue's own identifier, not ours.
        self._condition_to_mid: Dict[str, str] = {}

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
        if self._hl_price_feed is not None:
            tasks.append(asyncio.create_task(self._hl_price_feed.run(), name="hl_price_feed"))

        logger.info("orchestrator_started", n_markets=len(self._cfg.markets))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("orchestrator_shutdown")
        finally:
            await self._shutdown_all(tasks)

    # Setup
    async def _setup(self) -> None:
        poly_creds = self._cfg.polymarket
        hl_creds   = self._cfg.hyperliquid

        # EIP-712 signer
        if poly_creds:
            self._signer = EIP712Signer(poly_creds.private_key)

        # Shared HTTP session
        self._http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=50, ssl=True),
        )

        # without this OrderManager's requests go out unauthenticated and 401
        l2_auth = None
        if poly_creds and self._signer is not None:
            l2_auth = PolyL2Auth(PolyL2Credentials(
                api_key=poly_creds.api_key,
                secret=poly_creds.api_secret,
                passphrase=poly_creds.api_passphrase,
                address=self._signer.address,
            ))

        # resolve real yes/no token ids + neg_risk off the CLOB, the old
        # condition_id + "_YES" placeholder never matched anything real
        poly_market_ids = [
            mid for mid, mconf in self._cfg.markets.items()
            if mconf.venue == Venue.POLYMARKET
        ]
        if poly_market_ids and poly_creds:
            resolver = PolymarketMarketResolver(self._cfg.poly_rest_url, self._http_session)
            for mid in poly_market_ids:
                resolved = await resolver.resolve(self._cfg.markets[mid].condition_id)
                if resolved is None:
                    logger.error("poly_market_unresolved", market_id=mid)
                    continue
                self._resolved_poly[mid] = resolved

        # Order manager
        self._order_manager = OrderManager(
            http_session=self._http_session,
            signer=self._signer,
            risk_profile=next(iter(self._cfg.markets.values())).risk,
            rest_url=self._cfg.poly_rest_url,
            flickering_filter=FlickeringFilter(),
            l2_auth=l2_auth,
        )

        if self._cfg.kalshi:
            self._kalshi_order_manager = KalshiOrderManager(
                http_session=self._http_session,
                api_key_id=self._cfg.kalshi.api_key_id,
                rsa_signer=KalshiRsaSigner(self._cfg.kalshi.private_key_pem),
                risk_profile=next(iter(self._cfg.markets.values())).risk,
                rest_url=self._cfg.kalshi_rest_url,
            )

        # inventory manager: position size, VWAP cost basis, realized PnL
        self._inventory_mgr = InventoryManager(
            risk_profile=next(iter(self._cfg.markets.values())).risk,
        )
        if poly_creds:
            self._inventory_mgr.register_account("polymarket", "pUSD", balance=0.0)
        if self._cfg.kalshi:
            self._inventory_mgr.register_account("kalshi", "USD", balance=0.0)

        # Book registry
        self._book_registry = BookRegistry(self._state_queue)
        for mid, mconf in self._cfg.markets.items():
            source = (BookSource.POLYMARKET
                      if mconf.venue == Venue.POLYMARKET
                      else BookSource.KALSHI)
            self._book_registry.register(mid, source, mconf.resolution_ts)
            self._inventory_mgr.register_market(mid, mconf.venue.value)
            self._condition_to_mid[mconf.condition_id] = mid
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
            coins = sorted({
                mconf.underlying_symbol for mconf in self._cfg.markets.values()
                if mconf.underlying_symbol
            })
            self._hl_price_feed = HyperliquidPriceFeed(self._cfg.hl_rest_url, coins)
            hl_signer = HyperliquidSigner(
                hl_creds.private_key,
                is_mainnet=next(iter(self._cfg.markets.values())).hedge.is_mainnet,
            )
            self._hedge_engine = HedgeEngine(
                profile=next(iter(self._cfg.markets.values())).hedge,
                hl_url=self._cfg.hl_rest_url,
                signer=hl_signer,
                asset_index_fn=self._hl_price_feed.get_asset_index,
                sz_decimals_fn=self._hl_price_feed.get_sz_decimals,
            )
            for mid in self._cfg.markets:
                self._hedge_engine.register_market(mid)

    # Feed runner
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
                    yes_token_id=self._resolved_poly[mid].yes_token_id,
                    no_token_id=self._resolved_poly[mid].no_token_id,
                    neg_risk=self._resolved_poly[mid].neg_risk,
                )
                for mid, m in self._cfg.markets.items()
                if m.venue == Venue.POLYMARKET and mid in self._resolved_poly
            ]
            if len(poly_markets) < len(poly_markets_cfg):
                logger.warning(
                    "some_poly_markets_unresolved",
                    resolved=len(poly_markets),
                    configured=len(poly_markets_cfg),
                )
            feed = PolymarketFeed(
                ws_url=self._cfg.poly_ws_url,
                markets=poly_markets,
                out_queue=self._feed_queue,
                health_queue=self._health_queue,
            )
            tasks.append(asyncio.create_task(feed.run(), name="poly_feed"))

            # user channel: our own fills. without this the bot never learns
            # about its own fills and inventory just sits at zero forever
            user_feed = PolymarketUserFeed(
                ws_url=self._cfg.poly_ws_url.replace("/ws/market", "/ws/user"),
                condition_ids=[m.condition_id for m in poly_markets_cfg],
                api_key=self._cfg.polymarket.api_key,
                api_secret=self._cfg.polymarket.api_secret,
                api_passphrase=self._cfg.polymarket.api_passphrase,
                out_queue=self._fill_queue,
                health_queue=self._health_queue,
            )
            tasks.append(asyncio.create_task(user_feed.run(), name="poly_user_feed"))
            tasks.append(asyncio.create_task(self._fill_dispatch(), name="fill_dispatch"))

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

        # Dispatch loop: feed_queue -> BookRegistry (or fill dispatch for
        # Kalshi's own-fill events, which arrive on the same connection
        # as book data since Kalshi auths the whole socket up front)
        async def dispatch():
            while not self._kill_event.is_set():
                try:
                    event = await asyncio.wait_for(
                        self._feed_queue.get(), timeout=1.0
                    )
                    if isinstance(event, KalshiOwnFill):
                        await self._handle_kalshi_fill(event)
                        continue
                    await self._book_registry.process(event)
                    self._risk_engine.on_book_update(
                        "", time.monotonic()
                    )
                except asyncio.TimeoutError:
                    continue

        tasks.append(asyncio.create_task(dispatch(), name="dispatch"))
        await asyncio.gather(*tasks)

    # Fill dispatch
    def _apply_fill(self, market_id: str, order_id: str, side: str, price: float, size: float) -> None:
        """Shared by both venues: push the fill through OrderManager,
        InventoryManager (source of truth), and RiskEngine."""
        mconf = self._cfg.markets.get(market_id)
        if mconf is None:
            return

        mgr = self._order_manager if mconf.venue == Venue.POLYMARKET else self._kalshi_order_manager
        mgr.mark_filled(order_id, size)

        collateral_used = price * size if side == "BUY" else (1 - price) * size
        realized = self._inventory_mgr.on_fill(
            market_id=market_id,
            fill_side=side,
            fill_price=price,
            fill_qty=size,
            collateral_used=collateral_used,
        )

        mid_at_fill = self._last_mid.get(market_id, price)
        self._risk_engine.on_fill(
            market_id=market_id,
            order_id=order_id,
            fill_price=price,
            fill_size=size,
            side=side,
            mid_at_fill=mid_at_fill,
            realized_pnl=realized,
        )

        logger.info(
            "fill_dispatched",
            market_id=market_id,
            venue=mconf.venue.value,
            side=side,
            price=round(price, 4),
            size=size,
            realized_pnl=round(realized, 4),
            net_qty=round(self._inventory_mgr.get_net_qty(market_id), 2),
        )

    async def _handle_kalshi_fill(self, ev: KalshiOwnFill) -> None:
        market_id = self._condition_to_mid.get(ev.market_ticker)
        if market_id is None:
            return
        # action "buy"/"sell" on the yes leg maps straight onto our BUY/SELL
        side = "BUY" if ev.action == "buy" else "SELL"
        self._apply_fill(market_id, ev.order_id, side, ev.yes_price, ev.count)

    async def _fill_dispatch(self) -> None:
        """Polymarket own-fill events, drained off _fill_queue since the
        user channel is a separate websocket connection there."""
        while not self._kill_event.is_set():
            try:
                ev = await asyncio.wait_for(self._fill_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if not isinstance(ev, PolyOwnFill):
                continue
            market_id = self._condition_to_mid.get(ev.condition_id)
            if market_id is None:
                continue

            self._apply_fill(market_id, ev.order_id, ev.side, ev.price, ev.size)

    # Strategy loop
    async def _strategy_loop(self) -> None:
        """Main quoting loop, drains state_queue and updates quotes per market."""
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

            # Update risk engine mark + inventory mark-to-market
            self._risk_engine.on_market_update(mid_id, state.p_mid)
            self._inventory_mgr.update_mid(mid_id, state.p_mid)

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
                inventory_q=self._inventory_mgr.get_net_qty(mid_id),
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
                if mconf.venue == Venue.POLYMARKET:
                    resolved = self._resolved_poly.get(mid_id)
                    if resolved is None:
                        continue  # never resolved at startup, can't quote safely
                    await self._order_manager.update_quotes(
                        state=state,
                        fv=fv,
                        yes_token_id=resolved.yes_token_id,
                        inventory_q=self._inventory_mgr.get_net_qty(mid_id),
                        order_size_usd=mconf.risk.min_edge_bps * 10,  # size proportional to edge
                        neg_risk=resolved.neg_risk,
                    )
                elif self._kalshi_order_manager:
                    await self._kalshi_order_manager.update_quotes(
                        state=state,
                        fv=fv,
                        ticker=mconf.condition_id,
                        inventory_q=self._inventory_mgr.get_net_qty(mid_id),
                        order_size_usd=mconf.risk.min_edge_bps * 10,
                    )

            # Trigger hedge if needed
            if self._hedge_engine and mconf.hedge.enabled and mconf.underlying_symbol:
                coin = mconf.underlying_symbol
                S_perp = self._hl_price_feed.get_mid(coin) if self._hl_price_feed else None
                sigma_perp = self._hl_price_feed.get_realized_vol(coin) if self._hl_price_feed else None
                K_strike = mconf.underlying_strike

                if S_perp is None or sigma_perp is None or K_strike is None:
                    # missing live price, not enough vol history yet, or no
                    # strike configured for this market, skip rather than
                    # hedge off a guess
                    logger.debug(
                        "hedge_skipped_missing_data",
                        market_id=mid_id,
                        has_price=S_perp is not None,
                        has_vol=sigma_perp is not None,
                        has_strike=K_strike is not None,
                    )
                    continue

                instr = await self._hedge_engine.compute_and_hedge(
                    market_id=mid_id,
                    inventory_q=self._inventory_mgr.get_net_qty(mid_id),
                    p_mid=state.p_mid,
                    S_perp=S_perp,
                    K_strike=K_strike,
                    sigma_perp=sigma_perp,
                    T_res_s=state.time_to_resolution_s,
                    perp_symbol=coin,
                )
                if instr:
                    await self._hedge_engine.execute_hedge(
                        self._http_session, instr, S_perp
                    )

    # Calibration loop
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

    # Kill monitor
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

    # Shutdown
    async def _shutdown_all(self, tasks) -> None:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._http_session:
            await self._http_session.close()
        logger.info("orchestrator_clean_shutdown")


# Entrypoint
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
