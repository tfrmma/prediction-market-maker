"""
Order sizing. Replaces the old order_size_usd = min_edge_bps * 10
placeholder, which had no relationship to risk, volatility, or how much
capital was actually free to deploy.

The idea: size up when edge is fat and vol is low (favorable risk/reward
per quote), size down when vol is high (each fill carries more inventory
risk), and never exceed what the risk profile or free collateral allow.
This is deliberately simple, a proper implementation would fold in
Kelly-style bankroll fractions and per-market correlation, but "simple
and bounded" beats "sophisticated and wrong" for a first pass.
"""
from __future__ import annotations

from config.settings import RiskProfile


def compute_order_size_usd(
    edge_bps: float,
    sigma: float,
    free_collateral_usd: float,
    risk_profile: RiskProfile,
) -> float:
    """
    edge_bps            : current quoted edge over fair value, in bps
    sigma                : realized_vol_1m from MarketState (0 if unknown yet)
    free_collateral_usd  : InventoryManager.get_free_collateral(venue)
    """
    if edge_bps <= 0 or free_collateral_usd <= 0:
        return 0.0

    # more edge than our minimum threshold scales size up, capped at 3x
    # so one juicy quote doesn't blow the whole risk budget on one order
    edge_factor = min(edge_bps / max(risk_profile.min_edge_bps, 1e-6), 3.0)

    # dampen size in choppy markets, floor sigma so a quiet/no-data market
    # doesn't get a division-by-near-zero size explosion
    vol_factor = 1.0 / max(sigma, 0.02)
    vol_factor = min(vol_factor, 2.0)   # cap the other direction too

    size = risk_profile.base_order_size_usd * edge_factor * vol_factor

    # never risk more than max_position_pct of what's actually free
    risk_budget = risk_profile.max_position_pct * free_collateral_usd

    return max(0.0, min(size, risk_budget, risk_profile.max_order_size_usd))
