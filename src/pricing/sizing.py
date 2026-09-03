"""
Order sizing. Replaces the old order_size_usd = min_edge_bps * 10
placeholder, which had no relationship to risk, volatility, or how much
capital was actually free to deploy.

Two things layer on top of the base edge/vol-scaled size, both bounded
on top of it rather than replacing it:

  - A fractional-Kelly cap. Full Kelly (f* = edge / variance) is the
    growth-optimal bet size under a mean-variance approximation, and
    it's also a great way to blow up an account on model error, so we
    cap it at `kelly_fraction_cap` (quarter-Kelly by default, the usual
    practitioner haircut). This isn't a full Kelly implementation with
    real win-probability estimation, it's a bounded proxy using the
    edge and realized vol we already compute, treat it as a sanity cap
    on the aggressive end, not a precise optimal-sizing model.

  - Correlated exposure awareness. Two markets on the same underlying
    (say a BTC-100k and a BTC-105k market) aren't independent risk, if
    we're already sitting on a big directional position in one, sizing
    the next one as if starting from zero understates the real exposure.
    `correlated_exposure_usd` is the caller's job to compute (sum of
    abs(net_delta_usd) across every other market sharing the same
    underlying_symbol), and it eats into the same risk budget.
"""
from __future__ import annotations

from config.settings import RiskProfile


def compute_order_size_usd(
    edge_bps: float,
    sigma: float,
    free_collateral_usd: float,
    risk_profile: RiskProfile,
    correlated_exposure_usd: float = 0.0,
) -> float:
    """
    edge_bps                  : current quoted edge over fair value, in bps
    sigma                      : realized_vol_1m from MarketState (0 if unknown yet)
    free_collateral_usd        : InventoryManager.get_free_collateral(venue)
    correlated_exposure_usd    : abs(net_delta_usd) summed across every OTHER
                                  market sharing this market's underlying_symbol
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

    # fractional-Kelly cap: f* = edge / variance, haircut to kelly_fraction_cap
    edge_frac = edge_bps / 10_000
    kelly_f = min(edge_frac / max(sigma ** 2, 1e-4), risk_profile.kelly_fraction_cap)
    kelly_budget = kelly_f * free_collateral_usd

    # correlated exposure eats into the same budget as free collateral,
    # a market already carrying a lot of same-underlying risk gets less
    # room for a new order regardless of how attractive this one quote looks
    correlation_room = max(0.0, risk_profile.max_correlated_exposure_usd - correlated_exposure_usd)

    return max(0.0, min(
        size,
        risk_budget,
        kelly_budget,
        correlation_room,
        risk_profile.max_order_size_usd,
    ))
