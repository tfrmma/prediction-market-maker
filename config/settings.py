"""
Central config. Hard limits live here, strategy code only ever reads
them. Override via env vars or a .env file in prod.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings  # pydantic v2 compat shim

from config.secrets import load_secret


# Venue Enums
class Venue(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI     = "kalshi"
    HYPERLIQUID = "hyperliquid"
    BINANCE    = "binance"


# Risk Profile (per-market)
class RiskProfile(BaseModel):
    """
    Hard limits evaluated by the independent RiskEngine.
    All dollar amounts in USD-equivalent.
    """
    max_net_delta_usd: float = Field(500.0,  description="Max |long - short| USD exposure")
    max_gross_exposure_usd: float = Field(2_000.0, description="Max |long| + |short|")
    max_position_pct: float = Field(0.20,   description="Max fraction of collateral in one market")
    intraday_drawdown_limit: float = Field(200.0,  description="Kill switch: loss in rolling 24h window")
    loss_rate_limit_15m: float = Field(100.0, description="Kill switch: loss within any rolling 15-min window")
    per_trade_loss_limit: float = Field(50.0,  description="Max loss accepted before cancelling side")
    max_inventory_contracts: int  = Field(500,    description="Max contracts long or short per market")
    min_edge_bps: float = Field(15.0,  description="Minimum spread capture in bps to quote")
    base_order_size_usd: float = Field(25.0, description="Order size at min_edge_bps / average vol")
    max_order_size_usd: float = Field(150.0, description="Hard ceiling regardless of edge/vol scaling")
    toxic_flow_pause_ms: int  = Field(5_000,  description="Quote freeze ms after toxicity detection")
    flickering_window_ms: int  = Field(500,    description="Window to detect sub-500ms cancel patterns")
    flickering_cancel_threshold: int  = Field(3,      description="N cancels in window → freeze side")


class HedgeProfile(BaseModel):
    """
    Parameters for the cross-venue delta-neutral hedging module.
    """
    enabled: bool        = True
    venue: Venue         = Venue.HYPERLIQUID
    min_delta_usd: float = Field(50.0,  description="Minimum delta imbalance before hedging")
    max_hedge_latency_ms: int  = Field(200,   description="Abort hedge if not filled within this window")
    correlation_window: int   = Field(300,   description="Rolling window (seconds) for ρ estimation")
    correlation_min_abs: float = Field(0.60,  description="Minimum |ρ| to allow delta hedge via perp")
    hedge_size_multiplier: float = Field(0.90, description="Hedge ratio < 1 to account for basis risk")
    is_mainnet: bool = Field(True, description="Hyperliquid signing domain: mainnet vs testnet")
    min_slip_bps: float = Field(10.0, description="Floor on the IOC crossing buffer, even in dead-quiet markets")
    max_slip_bps: float = Field(200.0, description="Ceiling on the IOC crossing buffer, however wild vol gets")
    slip_vol_multiplier: float = Field(4.0, description="Std-dev multiplier on the vol-scaled slippage buffer")


# Market Configuration
class MarketConfig(BaseModel):
    """Per-market operational parameters."""
    condition_id: str          # Polymarket condition ID or Kalshi market ticker
    venue: Venue
    resolution_ts: int          # Unix timestamp of expected resolution
    underlying_symbol: Optional[str] = None   # e.g. "BTC" for correlated hedging
    # Strike/threshold for the underlying, e.g. 100_000 for a "BTC > $100k"
    # market. Has to come from config, there's no generic way to parse it
    # out of a market's title/ticker across venues.
    underlying_strike: Optional[float] = None
    tick_size: float = 0.01
    min_order_size: float = 1.0  # USD notional
    risk: RiskProfile = Field(default_factory=RiskProfile)
    hedge: HedgeProfile = Field(default_factory=HedgeProfile)


# API Credentials
class PolymarketCredentials(BaseModel):
    api_key: str    = Field(default_factory=lambda: os.environ["POLY_API_KEY"])
    api_secret: str = Field(default_factory=lambda: os.environ["POLY_API_SECRET"])
    api_passphrase: str = Field(default_factory=lambda: os.environ["POLY_PASSPHRASE"])
    private_key: str    = Field(default_factory=lambda: os.environ["POLY_PRIVATE_KEY"])
    # Polygon address derived from private_key at runtime


class KalshiCredentials(BaseModel):
    api_key_id: str  = Field(default_factory=lambda: os.environ["KALSHI_KEY_ID"])
    # KALSHI_PEM_SECRET_ARN, if set, sources this from AWS Secrets Manager
    # instead of the plaintext KALSHI_PEM env var. See config/secrets.py.
    private_key_pem: str = Field(
        default_factory=lambda: load_secret("KALSHI_PEM", "KALSHI_PEM_SECRET_ARN")
    )
    # Kalshi uses RSA PKCS8 signing as of API v2


class HyperliquidCredentials(BaseModel):
    wallet_address: str = Field(default_factory=lambda: os.environ["HL_WALLET"])
    private_key: str    = Field(default_factory=lambda: os.environ["HL_PRIVATE_KEY"])


# Top-level Settings Singleton
class Settings(BaseModel):
    """
    Loaded once at startup.  Treat as immutable after init.
    """
    env: str = Field("production", description="production | staging | backtest")

    # Feed endpoints
    poly_ws_url: str  = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    poly_rest_url: str = "https://clob.polymarket.com"
    poly_data_api_url: str = "https://data-api.polymarket.com"   # positions, public, no auth
    # Production. For the demo/sandbox environment use:
    #   wss://demo-api.kalshi.co/trade-api/ws/v2
    #   https://demo-api.kalshi.co/trade-api/v2
    kalshi_ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    kalshi_rest_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    hl_ws_url: str    = "wss://api.hyperliquid.xyz/ws"
    hl_rest_url: str  = "https://api.hyperliquid.xyz"

    # Credentials (lazy-loaded from env)
    polymarket: Optional[PolymarketCredentials] = None
    kalshi: Optional[KalshiCredentials]         = None
    hyperliquid: Optional[HyperliquidCredentials] = None

    # Markets to make
    markets: Dict[str, MarketConfig] = Field(default_factory=dict)

    # Global kill switch
    kill_switch_active: bool = False

    # Logging
    log_level: str = "INFO"
    log_file: str  = "logs/mm.jsonl"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        assert v in {"production", "staging", "backtest"}, f"Unknown env: {v}"
        return v


# Module-level singleton , import this everywhere
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings_from_dict(d: dict) -> Settings:
    global _settings
    _settings = Settings(**d)
    return _settings
