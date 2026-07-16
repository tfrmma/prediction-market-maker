# Prediction Market - Market Maker

Algorithmic market making engine for binary prediction markets. Targets **Polymarket CLOB (V2)** and **Kalshi**, with delta-neutral hedging via **Hyperliquid** perpetuals.

This is a real trading system, not a demo. It signs and submits live orders, tracks real inventory and collateral, and has an independent kill switch. Read the [Known Gaps](#known-gaps) section before pointing it at real capital.

## Architecture

```
                         ORCHESTRATOR (src/main.py)
                    asyncio task graph, uvloop, kill_event

  DATA LAYER              STRATEGY LOOP              RISK ENGINE
                                                       (independent task)
  PolymarketFeed   -->    FairValueEngine
  KalshiFeed               - A-S binary               Drawdown monitor
                            - Flow adjustment           Loss-rate monitor
  UnifiedBook      -->      - Prelec correction        API failure counter
   - YES/NO matrix                                     Book stale watchdog
   - OFI / CVD              OrderManager                Latency spike detect
   - Arb signal              - Cancel/replace
                             - Flickering filter        PnL decomposition:
  BookRegistry               - STP guard                 spread_capture
                                                          inventory_pnl
                            KalshiOrderManager             adverse_selection
                             - same, Kalshi wire format

                            HedgeEngine
                             - BS digital delta    --> Hyperliquid perp
                             - Correlation filter
                             - real phantom-agent signing

                            InventoryManager
                             - VWAP cost basis, both legs
                             - Collateral tracking
                             - Concentration limits

  StartupReconciler: on boot, pulls real positions and cancels any
  resting orders left over from a previous run, before the strategy
  loop places anything new.
```

### Module map

```
prediction-market-maker/
├── config/
│   └── settings.py                    # Pydantic settings: risk profiles, API creds, market configs
├── src/
│   ├── data/
│   │   ├── base_feed.py               # Abstract WS feed: reconnection, gap detection, staleness
│   │   ├── polymarket_feed.py         # Polymarket CLOB V2 connector (YES + NO books, own-fill user feed)
│   │   ├── polymarket_market_resolver.py  # Resolves real YES/NO token ids + neg_risk off the CLOB
│   │   ├── kalshi_feed.py             # Kalshi API v2 connector (RSA-PSS auth, seq tracking, own fills)
│   │   └── unified_book.py            # YES/NO synthetic probability matrix, OFI, CVD
│   ├── pricing/
│   │   └── fair_value.py              # A-S binary adaptation, Prelec correction, OLS calibrator
│   ├── execution/
│   │   ├── eip712_signer.py           # EIP-712 signing for Polymarket CLOB V2 orders
│   │   ├── polymarket_auth.py         # L2 HMAC auth for authenticated CLOB REST calls
│   │   ├── kalshi_auth.py             # RSA-PSS request signing, shared by Kalshi's WS and REST
│   │   ├── order_types.py             # ManagedOrder, FlickeringFilter, round_to_tick, shared across venues
│   │   ├── order_manager.py           # Polymarket cancel/replace, flickering filter, STP guard
│   │   ├── kalshi_order_manager.py    # Kalshi cancel/replace, tick resolution, same guards
│   │   └── reconciliation.py          # Startup: pull real positions, flatten stale resting orders
│   ├── inventory/
│   │   └── manager.py                 # VWAP positions (both legs), collateral accounting, exposure report
│   ├── hedging/
│   │   ├── delta_hedge.py             # BS digital delta, correlation filter, hedge sizing
│   │   ├── hyperliquid_signer.py      # Phantom-agent L1 action signing for Hyperliquid
│   │   └── hyperliquid_price_feed.py  # Mid price + realized vol poller, feeds the hedge engine
│   ├── risk/
│   │   └── engine.py                  # Kill switch, PnL decomposition, AS measurement
│   └── main.py                        # Async orchestrator, signal handlers, task graph
├── tests/
│   ├── backtest_simulator.py          # GARCH path gen, Poisson arrivals, walk-forward validator
│   └── test_core.py                   # 34 unit tests across 10 classes
└── pyproject.toml
```

## Theoretical foundations

### 1. Avellaneda-Stoikov, binary adaptation

Standard A-S was designed for continuous assets. This engine adapts it to binary payoff contracts (0 or 1 at resolution).

The key substitution: instead of geometric Brownian motion variance, we use the Bernoulli variance of a binary outcome:

```
sigma_binary^2 = p * (1 - p)
```

This is maximized at p = 0.5 (maximum uncertainty) and collapses to zero at the extremes, which correctly reflects that a 99%-probability market carries very little inventory risk.

Reservation price (inventory skew):
```
r(p, q, t) = p_fair - q * gamma * p*(1-p) * (T-t)
```

Optimal half-spread:
```
delta* = gamma * p*(1-p) * (T-t) + (1/gamma) * ln(1 + gamma/k)
```

Where `p_fair` is the flow-adjusted fair probability, `q` is net inventory in contracts (signed, +long YES / -short YES), `gamma` is a risk aversion coefficient calibrated from max drawdown tolerance, `k` is arrival rate decay calibrated from the empirical fill histogram, and `T-t` is time to resolution in years.

As resolution approaches (`T-t -> 0`), both the inventory skew and the spread collapse toward zero. There's no inventory risk one second before a binary market resolves.

### 2. Fair value with flow adjustment

Raw mid probability is corrected for observable order flow:

```
P_fair = P_base + alpha * CVD + beta * OFI_normalized
```

`CVD` is Cumulative Volume Delta over a rolling 5-minute window ($-weighted taker direction). `OFI_normalized` is order flow imbalance normalized by total depth: `(bid_depth - ask_depth) / total`. `alpha` and `beta` are calibrated via rolling OLS regression of `(CVD, OFI) -> delta_mid` over a 500-observation window.

### 3. Favorite-longshot bias correction (Kalshi)

Retail prediction markets systematically overweight low-probability events (Thaler & Ziemba, 1988). For Kalshi markets, we apply the Prelec (1998) probability weighting inversion in logit space:

```
logit(P_true) = sign(logit(P_market)) * |logit(P_market)|^(1/kappa)
```

With kappa < 1, this compresses extreme probabilities toward the correct frequentist value. kappa is calibrated from historical contract resolution data.

### 4. Polymarket YES/NO arbitrage constraint

Polymarket trades YES and NO as separate ERC-1155 tokens. The no-arbitrage constraint (buy YES + buy NO < $1.00 is risk-free profit) implies:

```
bid_YES <= 1 - ask_NO
ask_YES >= 1 - bid_NO
```

`UnifiedBook` constructs a synthetic probability from both token books. Kalshi is the mirror case: its feed is bids-only on both legs, so we derive the missing ask side as the complement of the other leg's best bid instead:

```
p_mid   = (bid_YES + ask_YES) / 2
arb_gap = bid_YES + bid_NO - 1.0   # > 0 means an arb exists
```

The arb gap is logged as a warning if it exceeds 0.5 cents.

### 5. Binary option delta (hedging)

A YES token on "BTC > $100K at expiry" is economically equivalent to a cash-or-nothing digital call option. Its delta with respect to the underlying is:

```
delta = phi(d2) / (S * sigma * sqrt(T-t))
d2 = [ln(S/K) + (r - sigma^2/2)*(T-t)] / (sigma * sqrt(T-t))
```

This delta is concentrated near 0.50 when `S ~= K` and blows up near expiry for markets right at the strike (the classic binary pin risk, don't trust it in the last few minutes). The hedge engine only executes when `|corr(dP_pm, dS_perp)| > 0.60` over the rolling window, to avoid adding noise from decorrelated flow. Hyperliquid mid price and realized vol come from a live poller (`HyperliquidPriceFeed`), not a static constant.

### 6. PnL decomposition

Every fill contributes to three distinct PnL buckets, measured independently:

| Component | Formula | Measurement |
|---|---|---|
| Spread capture | `(mid_at_fill - fill_price) * qty * sign` | At fill time |
| Inventory PnL | `(current_mid - avg_entry) * net_qty` | Mark-to-market continuously |
| Adverse selection | `(mid_at_fill - mid_at_t+30s) * qty * sign` | 30 seconds post-fill |

Realized PnL is computed once, by `InventoryManager`, and passed down to `RiskEngine`, there is no second parallel VWAP tracker recomputing it independently.

## Installation

```bash
# Python 3.11+
git clone <this repo>
cd prediction-market-maker

pip install -e ".[test]"
```

## Configuration

All configuration is through `config/settings.py`, driven by environment variables:

```bash
# Polymarket
export POLY_API_KEY="..."
export POLY_API_SECRET="..."
export POLY_PASSPHRASE="..."
export POLY_PRIVATE_KEY="0x..."      # Polygon wallet private key

# Kalshi
export KALSHI_KEY_ID="..."
export KALSHI_PEM="-----BEGIN PRIVATE KEY-----\n..."

# Hyperliquid (hedging)
export HL_WALLET="0x..."
export HL_PRIVATE_KEY="0x..."
```

Market configuration is passed programmatically:

```python
from config.settings import Settings, MarketConfig, RiskProfile, Venue, load_settings_from_dict

settings = load_settings_from_dict({
    "markets": {
        "btc-100k-2025": MarketConfig(
            condition_id="0xabc123...",       # or the Kalshi ticker
            venue=Venue.POLYMARKET,
            resolution_ts=1735689600,          # unix ts
            underlying_symbol="BTC",           # for hedging, must match a Hyperliquid coin name
            underlying_strike=100_000,         # required if hedging is enabled for this market
            risk=RiskProfile(
                max_net_delta_usd=500,
                intraday_drawdown_limit=200,
                max_inventory_contracts=300,
                min_edge_bps=15,
            ),
        )
    }
})
```

Note that YES/NO token ids, `neg_risk`, and tick size are **not** configured manually for Polymarket, they're resolved from the CLOB at startup (`PolymarketMarketResolver`). Kalshi tick size is resolved the same way, lazily, on first quote per market.

### Risk profile reference

| Parameter | Default | Description |
|---|---|---|
| `max_net_delta_usd` | 500 | Max `\|long - short\|` USD exposure |
| `max_gross_exposure_usd` | 2 000 | Max `\|long\| + \|short\|` USD |
| `max_position_pct` | 0.20 | Max single-market share of capital |
| `intraday_drawdown_limit` | 200 | Kill switch: rolling 24h loss |
| `loss_rate_limit_15m` | 100 | Kill switch: loss within any rolling 15-min window |
| `per_trade_loss_limit` | 50 | Cancel side after this loss |
| `max_inventory_contracts` | 500 | Hard inventory ceiling |
| `min_edge_bps` | 15 | Don't quote below 15 bps edge |
| `toxic_flow_pause_ms` | 5 000 | Quote freeze after toxicity trigger |
| `flickering_window_ms` | 500 | Window for cancel pattern detection |
| `flickering_cancel_threshold` | 3 | Cancels in window triggers a freeze |

## Running

```bash
python -m src.main
```

On startup, the orchestrator:

1. Signs into Polymarket (EIP-712 + L2 HMAC) and/or Kalshi (RSA-PSS), whichever credentials are present.
2. Resolves real YES/NO token ids, `neg_risk`, and market metadata off the Polymarket CLOB.
3. **Reconciles state**: pulls current positions from both venues, seeds `InventoryManager` with them, and cancels any resting orders left over from a previous run.
4. Launches the concurrent task graph:

| Task | Role |
|---|---|
| `feeds` | Polymarket + Kalshi WS connectors, and Polymarket's authenticated user feed, into `BookRegistry` |
| `strategy` | Drains the state queue, computes fair value, updates quotes on both venues |
| `risk` | Independent monitor loop, owns `kill_event` |
| `calibrator` | Re-estimates `alpha`, `beta` every 5 minutes via OLS |
| `hl_price_feed` | Polls Hyperliquid mid price + realized vol, feeds the hedge engine |
| `kill_monitor` | Awaits `kill_event`, cancels all orders, graceful shutdown |

## Backtesting

```bash
python -m tests.backtest_simulator --mode single --gamma 0.05 --k 1.5 --p0 0.50
python -m tests.backtest_simulator --mode wf --gamma 0.05 --k 1.5
python -m tests.backtest_simulator --mode mc --paths 500
```

The simulator generates a synthetic environment (GARCH(1,1) mid-price path, Poisson fill arrivals decaying with spread distance, toxic flow injected at a configurable rate) and is meant for sanity-checking parameter choices, not as a source of performance claims. Output varies significantly run to run over short windows, single-run Sharpe ratios in particular are noisy and should not be read as a live-performance estimate. Run it yourself with your own seeds before trusting any number it prints.

GARCH(1,1) mid-price dynamics:
```
sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
eps_t = sigma_t * z_t,   z_t ~ N(0,1)
```

Poisson fill arrivals:
```
lambda(delta) = A * exp(-k*delta)
P(fill in dt) = 1 - exp(-lambda(delta)*dt)
```

With probability `p_toxic` per arrival, an informed trader shows up and permanently shifts mid by `delta_adverse`, this is what generates the adverse selection component in the PnL decomposition.

## Testing

```bash
pytest tests/test_core.py -v
pytest tests/test_core.py --cov=src --cov-report=term-missing
```

34 tests across ten classes:

| Class | What it covers |
|---|---|
| `TestL2Book` | Sort order, delta application, depth trim, crossed book handling |
| `TestFairValueEngine` | Inventory skew, spread collapse at resolution, staleness guard, CVD sensitivity, Prelec monotonicity, no-crossed-quotes invariant |
| `TestEIP712Signer` | V2 struct shape (no taker/nonce), 6-decimal amount scaling, neg-risk domain selection, salt uniqueness |
| `TestBacktest` | Path execution, gamma-to-inventory relationship, walk-forward fold count |
| `TestKalshiBidsOnlyBook` | Complement-ask derivation, missing-leg guard, delta updates against the derived ask |
| `TestRiskEngineLossRate` | Loss-rate kill switch, realized PnL trusts the caller instead of recomputing |
| `TestHyperliquidSigner` | Phantom-agent hash determinism, nonce sensitivity, signature validity |
| `TestPolymarketMarketResolver` | YES/NO token id extraction from the `/markets/{condition_id}` response shape |
| `TestKalshiOrderManagerWireFormat` | BUY/SELL to bid/ask side mapping |
| `TestRoundToTick` | Price snapping to the resolved tick size |

## Kill switch

The risk engine runs as an independent `asyncio.Task` and owns a shared `asyncio.Event` (`kill_event`). Every strategy action is gated on `kill_event.is_set()`.

Kill switch triggers (any one fires it):

| Trigger | Condition |
|---|---|
| Drawdown | Intraday PnL < `-intraday_drawdown_limit` |
| Loss rate | Loss > `loss_rate_limit_15m` within any rolling 15-minute window |
| API failure | 3 or more consecutive order API failures |
| State desync | No book update for more than 10 seconds |
| Latency spike | Fill latency more than 5x the rolling median |
| Manual | `SIGINT` / `SIGTERM` |

On activation, all live quotes are cancelled before the process exits.

## Startup reconciliation

If the process restarts with resting orders still live on either exchange, starting `InventoryManager` at zero would mean quoting on top of a real, untracked position. `StartupReconciler` runs before the strategy loop starts anything:

- **Positions**: pulled from Polymarket's Data API (`GET /positions`, public) and Kalshi's `GET /portfolio/positions`, and used to seed `InventoryManager` directly.
- **Resting orders**: pulled from Polymarket's `GET /data/orders` and Kalshi's `GET /portfolio/orders?status=resting`, and cancelled outright rather than adopted back into tracked state. Queue position and partial-fill history from a previous process can't be recovered reliably, flattening and re-quoting fresh is the safer default.

## Known gaps

Closed as of this revision: Polymarket CLOB V2 order signing (real 6-decimal amounts, correct domain, neg-risk contract routing), the L2 HMAC auth that requests previously went out without, Kalshi's bids-only order book (was being parsed as if it had a real ask side), a working Kalshi execution engine (order placement, cancellation, own-fill feed, none of which existed before), real Hyperliquid phantom-agent signing (was a placeholder `{r, s, v}` that never worked), a live Hyperliquid price/vol feed for the hedge engine (was a hardcoded constant), the own-fill feedback loop on both venues, post-only and tick-size rounding on both venues, a symmetric self-trade guard, and startup reconciliation.

Still open:

- **Order sizing is a placeholder heuristic.** `order_size_usd = risk.min_edge_bps * 10` has no real relationship to edge, volatility, or available capital. Needs a real sizing model.
- **`health_queue` is populated but never consumed.** Feed and hedge health metrics (lag, reconnects) are emitted into a queue in `main.py` that nothing currently drains, so a silently degrading feed won't surface until something downstream breaks.
- **Kalshi PEM loaded from an environment variable.** Should come from a proper secrets manager in production.
- **Hedge execution uses a fixed 0.5% slippage allowance.** Should scale with the realized vol `HyperliquidPriceFeed` already computes, instead of a flat constant.
- **No integration testing against live sandboxes.** Everything above is validated at the unit level with mocks. None of it has been run against Kalshi's demo environment, Hyperliquid testnet, or Polymarket with real size.
- **`ManagedOrder.status` never transitions `PENDING -> OPEN`.** Doesn't break anything today, but any future logic that branches on that distinction will be wrong.

## Design decisions

**Why asyncio and not threads?** The strategy loop, feed ingestion, and risk engine are I/O-bound and latency-sensitive. Python's GIL would serialize CPU work across threads with no benefit. Single-threaded asyncio with `uvloop` gives deterministic execution order and eliminates race conditions on shared state without locks.

**Why a separate risk task?** The risk engine has to be able to fire the kill switch even if the strategy loop is blocked on a slow API call. A separate task with a shared `asyncio.Event` guarantees it can act independently.

**Why not one unified book for YES and NO?** Polymarket's YES and NO tokens trade on separate order books with separate fees and separate execution queues. Treating them as one book would mask the arb gap signal and make order signing ambiguous (wrong token id). `UnifiedBook` folds them synthetically for pricing while keeping the raw books separate for execution. Kalshi's book is architecturally different again (bids-only on both legs), and gets its own derivation path rather than being forced into the Polymarket shape.

**Why cancel-and-flatten instead of adopting stale orders on restart?** We can't recover exact queue position or partial-fill history for an order placed by a previous process. Reconstructing approximate state and being wrong about it is worse than cancelling and re-quoting fresh on the next tick.

**Why Prelec only for Kalshi?** Polymarket's CLOB attracts more sophisticated participants and the longshot bias is empirically weaker there. The correction is applied selectively, and the calibrated `kappa` is worth watching, if it converges to 1.0, the market is already pricing the tails efficiently and the correction is a no-op.

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). *The price impact of order book events.* Journal of Financial Econometrics.
- Prelec, D. (1998). *The probability weighting function.* Econometrica.
- Thaler, R. & Ziemba, W. (1988). *Anomalies: Parimutuel betting markets.* Journal of Economic Perspectives.
- Glosten, L. & Milgrom, P. (1985). *Bid, ask and transaction prices in a specialist market.* Journal of Financial Economics.
