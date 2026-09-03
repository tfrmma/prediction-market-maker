# Prediction Market Market Maker

Algorithmic market making engine for prediction markets, binary and N-outcome categorical (elections, tournaments, anything with mutually-exclusive outcomes that should sum to $1). Targets **Polymarket CLOB (V2)** and **Kalshi**, with delta-neutral hedging via **Hyperliquid** perpetuals.

This signs and submits live orders, tracks real inventory and collateral, and has an independent kill switch. Read [What's left](#whats-left) before you point it at real size.

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
                             - Concentration limits, basket-netted
                               for markets in an event group

  StartupReconciler: on boot, pulls real positions and cancels any
  resting orders left over from a previous run, before the strategy
  loop places anything new.

  CATEGORICAL (N-outcome) PATH, parallel to the binary one above,
  same OrderManager/KalshiOrderManager/HedgeEngine/InventoryManager
  underneath, only the pricing layer differs:

  EventGroupConfig  -->  CategoricalBookAggregator  -->  CategoricalCovarianceEstimator
   (N outcomes,             (buffers per-outcome           (ALR-space online EWMA,
    Settings.event_groups)   MarketState updates            probability-space output,
                              into one basket)                see design decisions below)
                                     |
                                     v
                          CategoricalFairValueEngine        CategoricalSkewEngine
                           (softmax, sum(p_fair)=1)   -->     (Guéant multi-asset,
                                                                sum(p_reservation)=1)
                                     |
                                     v
                     Orchestrator._handle_categorical_update
                      one FairValueResult-shaped quote per outcome,
                      _place_quotes_and_hedge is shared with the
                      binary path unchanged
```

### Module map

```
prediction-market-maker/
├── .github/workflows/
│   └── ci.yml                         # pytest gate (matrix py3.11/3.12) + non-blocking lint job
├── config/
│   ├── settings.py                    # Pydantic settings: risk profiles, API creds, market/event-group configs
│   └── secrets.py                     # Secret loading: AWS Secrets Manager, falls back to env vars
├── src/
│   ├── data/
│   │   ├── base_feed.py               # Abstract WS feed: reconnection, gap detection, staleness
│   │   ├── polymarket_feed.py         # Polymarket CLOB V2 connector (YES + NO books, own-fill user feed)
│   │   ├── polymarket_market_resolver.py  # Resolves real YES/NO token ids + neg_risk off the CLOB
│   │   ├── kalshi_feed.py             # Kalshi API v2 connector (RSA-PSS auth, seq tracking, own fills)
│   │   ├── unified_book.py            # YES/NO synthetic probability matrix, OFI, CVD
│   │   └── categorical_book.py        # N-outcome basket aggregator, inter-outcome arb_gap
│   ├── pricing/
│   │   ├── fair_value.py              # A-S binary adaptation, Prelec correction, OLS calibrator
│   │   ├── sizing.py                  # Order sizing: edge/vol/free-collateral bounded
│   │   ├── covariance.py              # Categorical Sigma: structural prior + online ALR-space EWMA
│   │   ├── categorical_fair_value.py  # Softmax consistency layer, sum(p_fair) = 1
│   │   └── categorical_skew.py        # Guéant multi-asset inventory skew + spread
│   ├── execution/
│   │   ├── eip712_signer.py           # EIP-712 signing for Polymarket CLOB V2 orders
│   │   ├── polymarket_auth.py         # L2 HMAC auth for authenticated CLOB REST calls
│   │   ├── kalshi_auth.py             # RSA-PSS request signing, shared by Kalshi's WS and REST
│   │   ├── order_types.py             # ManagedOrder, FlickeringFilter, round_to_tick, shared across venues
│   │   ├── order_manager.py           # Polymarket cancel/replace, flickering filter, STP guard
│   │   ├── kalshi_order_manager.py    # Kalshi cancel/replace, tick resolution, same guards
│   │   └── reconciliation.py          # Startup: pull real positions, flatten stale resting orders
│   ├── inventory/
│   │   └── manager.py                 # VWAP positions (both legs), collateral accounting, exposure report,
│   │                                  # basket-netted concentration for event-group members
│   ├── hedging/
│   │   ├── delta_hedge.py             # BS digital delta, correlation filter, hedge sizing
│   │   ├── hyperliquid_signer.py      # Phantom-agent L1 action signing for Hyperliquid
│   │   └── hyperliquid_price_feed.py  # Mid price + realized vol poller, feeds the hedge engine
│   ├── risk/
│   │   └── engine.py                  # Kill switch, PnL decomposition, AS measurement
│   └── main.py                        # Async orchestrator, signal handlers, task graph, categorical routing
├── tests/
│   ├── backtest_simulator.py          # Binary: GARCH path gen, Poisson arrivals, walk-forward validator
│   ├── categorical_backtest_simulator.py  # N-outcome: softmaxed correlated paths, same fill model
│   └── test_core.py                   # 106 tests across 21 classes
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

### 7. Categorical (N-outcome) market making

Sections 1-6 above cover a single binary market in isolation. Some events aren't binary, an election with 6 candidates, a tournament bracket, "who wins the award", these trade as N mutually-exclusive outcome markets that are supposed to sum to $1 of combined value but have no mechanism forcing them to, since each outcome's order book updates independently. `config.settings.EventGroupConfig` links a set of outcome markets under one event; `src/pricing/categorical_*.py` and `src/data/categorical_book.py` handle the pricing.

**7a. Consistency layer: softmax fair value**

Given per-outcome mids `p_1, ..., p_N` that don't necessarily sum to 1, score each outcome and renormalize:

```
score_i = ln(p_i) + alpha*CVD_i + beta*OFI_i
p_fair_i = exp(score_i) / sum_j(exp(score_j))
```

This is the same mechanism Hanson's (2003) LMSR uses to price a combinatorial outcome set from an outstanding-shares vector, applied here to order-book mids instead. Using `ln(p_i)`, not the binary engine's `logit(p_i)`, as the base score matters: with zero flow adjustment, `softmax(ln(p))` reduces to plain proportional renormalization `p_i / sum(p_j)`, so an already-consistent book passes through undistorted. Tried `logit(p_i)` first, it doesn't have that property, checked numerically before committing to `ln`. Flow adjustment enters multiplicatively (`p_i * exp(alpha*CVD_i + beta*OFI_i)`, before renormalizing), so buy pressure on one outcome pulls probability mass off the others through the shared denominator, that's the actual point of pricing the basket together instead of market-by-market.

**7b. Risk layer: multi-asset inventory skew**

Guéant, Lehalle & Fernandez-Tapia's (2013) multi-asset extension of A-S replaces the scalar reservation-price skew with a covariance matrix:

```
q_tilde_i = q_i - q_ref
skew_vec = -gamma * (Sigma @ q_tilde) * (T-t)
```

`q_ref` is inventory in a fixed reference outcome, `q_tilde` nets every other position against it, since mutually-exclusive outcomes pay off as a one-hot vector, only the excess over the reference position carries variance. `Sigma` is the covariance of that payout vector, `diag(p) - p*p^T` (the structural, zero-history piece) plus an online EWMA correction for realized co-movement beyond what mutual exclusivity alone implies. `Sigma`'s rows sum to zero by construction (it's a one-hot covariance), which means the reference outcome's own skew falls out for free (`skew_ref = -sum(skew_i for i != ref)`) and the whole reservation-price vector sums to exactly 1, same invariant the softmax layer keeps. Verified this isn't just approximately right: for N=2, `Sigma` collapses to the binary engine's own `p*(1-p)` and `q_tilde` becomes `q_A - q_B`, so the categorical engine's output matches `FairValueEngine` exactly given `inventory_q = q_A - q_B` (`test_n2_matches_binary_engine_exactly`).

**7c. Covariance estimation**

Raw probabilities live on a simplex (`sum(p_i) = 1`), so their covariance matrix is singular, N-1 degrees of freedom for N numbers. The online EWMA runs in additive log-ratio (ALR) space, `y_i = ln(p_i / p_ref)`, a well-behaved bijection off the simplex onto unconstrained `R^(N-1)`. ALR variance scales like `1/p_i` though, wrong units for the skew formula above (caught this via a test that should've held in probability space and didn't in ALR space), so the estimator always converts back to probability space via the local delta-method Jacobian before returning anything a caller can use. Falls back to the structural prior (no ALR involved, `diag(p) - p*p^T` with the reference row/column dropped) until the EWMA has 30 observations.

**7d. Basket-level risk (InventoryManager)**

Two outcomes of the same event aren't independent risk the way two unrelated markets are, but they also don't net the way a naive intuition ("long A, short B, must be hedged") suggests. Checked this by hand against `on_resolution`'s own settlement math: long A / short B in a 3+ outcome event is a leveraged directional bet, not a hedge, worst case is worse than either leg alone. What genuinely reduces risk is being long *several* outcomes at once (some payout is likelier), collapsing to exactly zero risk only in the degenerate uniform case (long every outcome equally, the risk-free "mint the basket" position). `InventoryManager._effective_exposure` replaces a market's naive `gross_exposure` with its share of the event's real worst-case loss across the N possible resolutions, split proportionally by each member's own exposure, for markets that belong to an `EventGroupConfig`. Standalone markets are unaffected.

## Installation

```bash
# Python 3.11+
git clone <this repo>
cd prediction-market-maker

pip install -e ".[test]"

# add the aws extra if you're pulling Kalshi's PEM from Secrets Manager
# (see KALSHI_PEM_SECRET_ARN below), it's a lazy import otherwise so
# don't bother if you're just using the plaintext env var
pip install -e ".[test,aws]"
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
# optional: source the PEM from AWS Secrets Manager instead of the
# plaintext env var above (requires boto3)
export KALSHI_PEM_SECRET_ARN="arn:aws:secretsmanager:...:secret:kalshi-pem"

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

**N-outcome events** are declared separately, linking a set of already-configured markets by `market_id`:

```python
settings = load_settings_from_dict({
    "markets": {
        "elec-candidate-a": MarketConfig(condition_id="ELEC-A", venue=Venue.KALSHI, resolution_ts=1735689600),
        "elec-candidate-b": MarketConfig(condition_id="ELEC-B", venue=Venue.KALSHI, resolution_ts=1735689600),
        "elec-candidate-c": MarketConfig(condition_id="ELEC-C", venue=Venue.KALSHI, resolution_ts=1735689600),
    },
    "event_groups": {
        "election-2028": EventGroupConfig(
            event_id="election-2028",
            outcome_ids=["elec-candidate-a", "elec-candidate-b", "elec-candidate-c"],
            venue=Venue.KALSHI,
            resolution_ts=1735689600,        # must match every member MarketConfig
            correlation_window_s=1800.0,     # EWMA window for the covariance estimator
        )
    },
})
```

Every `outcome_id` must already exist in `markets`, venue and `resolution_ts` must match across the group, and a market can only belong to one event, validated at `Settings` construction. See [Categorical (N-outcome) market making](#7-categorical-n-outcome-market-making) for what this actually changes in the pricing path.

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
| `base_order_size_usd` | 25 | Order size at min_edge_bps / average vol |
| `max_order_size_usd` | 150 | Hard ceiling regardless of edge/vol scaling |
| `kelly_fraction_cap` | 0.25 | Haircut on the fractional-Kelly size cap (quarter-Kelly) |
| `max_correlated_exposure_usd` | 1 000 | Cap on combined exposure across markets sharing an `underlying_symbol` |
| `toxic_flow_pause_ms` | 5 000 | Quote freeze after toxicity trigger |
| `flickering_window_ms` | 500 | Window for cancel pattern detection |
| `flickering_cancel_threshold` | 3 | Cancels in window triggers a freeze |

Order size scales up with edge (capped at 3x `base_order_size_usd`), down with `realized_vol_1m` (capped at 2x), and is always bounded by `max_position_pct * free_collateral` and `max_order_size_usd`, whichever is smaller. See `src/pricing/sizing.py`.

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

### Categorical backtesting

```bash
python -m tests.categorical_backtest_simulator --n-outcomes 4 --gamma 0.05 --k 1.5
```

Generalizes the same idea to N outcomes: independent GARCH-vol score paths, softmaxed every tick into probabilities that always sum to 1 (the same mechanism `CategoricalFairValueEngine` itself uses), plus a shared common shock so outcomes carry excess correlation beyond what mutual exclusivity alone implies, giving the EWMA covariance estimator something real to fit. Runs the actual pricing pipeline (`CategoricalBookAggregator` → `CategoricalCovarianceEstimator` → `CategoricalFairValueEngine` → `CategoricalSkewEngine`), not a reimplementation of it.

This fill model has no adverse selection (arrivals react to quote distance only, never to where the "true" price is about to move), so, same as the binary simulator, its PnL number is not a performance estimate, it's useful for checking the pipeline holds together over an extended run, not for judging strategy quality. `max_position_contracts` caps per-outcome inventory, there's no real risk-limit wiring in this harness otherwise.

## Testing

```bash
pytest tests/test_core.py -v
pytest tests/test_core.py --cov=src --cov-report=term-missing
```

106 tests across twenty-one classes:

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
| `TestOrderSizing` | Zero-edge floor, free-collateral budget cap, hard ceiling, vol dampening, Kelly cap, correlated-exposure cap |
| `TestHedgeSlippage` | No-data floor fallback, vol widens the crossing buffer, ceiling clamp |
| `TestSecretsLoader` | Plain env var fallback, AWS Secrets Manager path when an ARN is configured |
| `TestOrderStatusResolution` | Placement-response status mapping to OPEN/FILLED/PARTIAL_FILL/PENDING on both venues |
| `TestFeedDesyncKillSwitch` | `trigger_feed_desync` actually flips `kill_event`, second trigger is a no-op |
| `TestEventGroupConfig` | Outcome/venue/resolution_ts consistency, double-membership rejection, standalone configs unaffected |
| `TestCategoricalBookAggregator` | Warm-up gating, basket arb_gap signs, cross-outcome staleness gate |
| `TestCategoricalCovarianceEstimator` | Structural-prior PSD/symmetry, EWMA warm-up, probability-space conversion, injected-correlation sign |
| `TestCategoricalFairValueEngine` | `sum(p_fair) == 1` invariant, zero-flow renormalization, N=2 sigmoid identity, Prelec reuse, should_quote gating |
| `TestCategoricalSkewEngine` | Reservation prices sum to 1, skew sums to 0, exact N=2 regression against `FairValueEngine`, spread guardrails |
| `TestOrchestratorCategoricalAdapter` | Categorical-to-`FairValueResult` field mapping, the seam `order_manager.py` consumes |
| `TestInventoryManagerBasketNetting` | Uniform basket nets to zero, flat sibling never gets phantom exposure, long/short isn't treated as a hedge, diversified basket nets to the real worst case |
| `TestCategoricalIntegration` | Full pipeline, no mocks, invariants held every tick across a simulated run; `CategoricalBacktestRunner` settlement correctness |

## Continuous integration

`.github/workflows/ci.yml`, two jobs:

- **`test`** (blocking): matrix on Python 3.11/3.12, `pip install -e ".[test]"`, full suite with coverage. No API keys or secrets required, none of the pricing/inventory/risk/categorical logic touches real credentials at import or test time, so this also runs clean on PRs from forks.
- **`lint`** (non-blocking): `ruff check .`, `black --check .`, `mypy .`, each `continue-on-error`. There's real pre-existing lint/type debt (mostly `pyupgrade`-style modernization, `Dict` → `dict`, `Optional[X]` → `X | None`, plus a handful of untyped test helpers under `mypy --strict`) that predates any of this. Left visible rather than either hiding it or blocking merges on a mass mechanical reformat; tighten it (drop `continue-on-error`, or narrow `ruff check .` to a `--select` of just the rules you want enforced today) once that debt is paid down.

## Kill switch

The risk engine runs as an independent `asyncio.Task` and owns a shared `asyncio.Event` (`kill_event`). Every strategy action is gated on `kill_event.is_set()`.

Kill switch triggers (any one fires it):

| Trigger | Condition |
|---|---|
| Drawdown | Intraday PnL < `-intraday_drawdown_limit` |
| Loss rate | Loss > `loss_rate_limit_15m` within any rolling 15-minute window |
| API failure | 3 or more consecutive order API failures |
| State desync | No book update for more than 10 seconds, or a single feed down for more than `FEED_DOWN_KILL_THRESHOLD_S` (15s) while others keep flowing |
| Latency spike | Fill latency more than 5x the rolling median |
| Manual | `SIGINT` / `SIGTERM` |

On activation, all live quotes are cancelled before the process exits.

## Startup reconciliation

If the process restarts with resting orders still live on either exchange, starting `InventoryManager` at zero would mean quoting on top of a real, untracked position. `StartupReconciler` runs before the strategy loop starts anything:

- **Positions**: pulled from Polymarket's Data API (`GET /positions`, public) and Kalshi's `GET /portfolio/positions`, and used to seed `InventoryManager` directly.
- **Resting orders**: pulled from Polymarket's `GET /data/orders` and Kalshi's `GET /portfolio/orders?status=resting`, and cancelled outright rather than adopted back into tracked state. Queue position and partial-fill history from a previous process can't be recovered reliably, flattening and re-quoting fresh is the safer default.

## What's left

Closed in this revision: real Polymarket CLOB V2 signing (6-decimal amounts, correct domain, neg-risk routing), L2 HMAC auth on every authenticated request, Kalshi's bids-only book (was getting parsed like it had a real ask side), a full Kalshi execution engine that didn't exist before, real Hyperliquid phantom-agent signing (was a placeholder that never actually signed anything), a live Hyperliquid price/vol feed instead of a hardcoded constant, the own-fill feedback loop on both venues, post-only and tick-size rounding on both venues, a symmetric self-trade guard, startup reconciliation, order sizing tied to edge/vol/free collateral instead of a made-up constant, a `health_queue` that actually gets read, Kalshi's PEM off a secrets manager when you want it, hedge slippage that scales with real vol instead of a flat 0.5%, order status resolution off the placement response (`PENDING` used to never move to `OPEN`/`FILLED`/`PARTIAL_FILL`), a fractional-Kelly cap and cross-market correlated-exposure awareness in the sizing model, a `_health_monitor` that trips the kill switch itself when a single feed stays down past `FEED_DOWN_KILL_THRESHOLD_S` (15s default) instead of just logging it, and a full categorical (N-outcome) market making path: `EventGroupConfig`, basket state aggregation, online covariance estimation, softmax fair value, multi-asset inventory skew, wired into the strategy loop, basket-aware concentration limits, a dedicated backtest simulator, and CI.

Also fixed along the way, none of it categorical-specific: `pricing/sizing.py` lived outside `src/`, at a path nothing actually imported (four tests had been failing on that alone); `pyproject.toml`'s `build-backend` was a typo, `pip install -e .` failed outright; `unified_book.py` referenced `asyncio` in a type hint without importing it.

Still open, specifically on the categorical side: `alpha`/`beta`/`gamma`/`k` in `CategoricalASParams` are seeded off the binary engine's defaults, not calibrated against real basket-level fill data, there's no categorical equivalent of `ParameterCalibrator` yet. The covariance estimator's reference outcome is fixed at construction (last id in `outcome_ids`); if that specific outcome's own probability collapses toward 0 well into an event's life, the ALR-to-probability conversion gets poorly conditioned right when you'd want it most, noted as a TODO in `covariance.py`, would need a CLR/ILR basis to fully close. Lint/type debt is real and tracked in CI but non-blocking, see [Continuous integration](#continuous-integration).

Nothing else outstanding right now. If you find something, it's probably in a part of the flow nobody's traded through yet.

## Design decisions

**Why asyncio and not threads?** The strategy loop, feed ingestion, and risk engine are I/O-bound and latency-sensitive. Python's GIL would serialize CPU work across threads with no benefit. Single-threaded asyncio with `uvloop` gives deterministic execution order and eliminates race conditions on shared state without locks.

**Why a separate risk task?** The risk engine has to be able to fire the kill switch even if the strategy loop is blocked on a slow API call. A separate task with a shared `asyncio.Event` guarantees it can act independently.

**Why not one unified book for YES and NO?** Polymarket's YES and NO tokens trade on separate order books with separate fees and separate execution queues. Treating them as one book would mask the arb gap signal and make order signing ambiguous (wrong token id). `UnifiedBook` folds them synthetically for pricing while keeping the raw books separate for execution. Kalshi's book is architecturally different again (bids-only on both legs), and gets its own derivation path rather than being forced into the Polymarket shape.

**Why cancel-and-flatten instead of adopting stale orders on restart?** We can't recover exact queue position or partial-fill history for an order placed by a previous process. Reconstructing approximate state and being wrong about it is worse than cancelling and re-quoting fresh on the next tick.

**Why Prelec only for Kalshi?** Polymarket's CLOB attracts more sophisticated participants and the longshot bias is empirically weaker there. The correction is applied selectively, and the calibrated `kappa` is worth watching, if it converges to 1.0, the market is already pricing the tails efficiently and the correction is a no-op.

**Why `ln(p_i)` and not `logit(p_i)` as the categorical score?** The binary engine's flow adjustment is additive in probability space, the natural first instinct for the categorical version is to reuse the same `logit(p_i)` transform per outcome before softmax-ing. Checked it numerically first: `softmax(logit(p))` doesn't reduce to `p` even when `p` already sums to 1, it distorts an already-consistent book for no reason. `softmax(ln(p))` does reduce correctly (it's exactly proportional renormalization at zero flow), because softmax is invariant to a constant shift and `ln(p_i)` is the right score for that identity to hold. `logit(p_i) = ln(p_i) - ln(1-p_i)`, the extra `-ln(1-p_i)` term doesn't cancel across outcomes the way `ln(p_i)` does.

**Why does the covariance estimator run internally in ALR space instead of directly on probabilities?** Raw probabilities are simplex-constrained (`sum = 1`), so their covariance is singular by construction, and the variance of `p_i` isn't stationary in `p_i` (movement near 0 or 1 behaves differently than movement near 0.5). Additive log-ratio space fixes both, at the cost of the variance magnitude itself being in log-ratio units, not probability units. The estimator hides this: everything it returns to a caller (`covariance()`) is converted back to probability space first, ALR is an implementation detail, not a leaked abstraction.

**Why does `InventoryManager` net baskets by worst-case-loss share instead of by inventory deviation from some reference?** Tried netting `q` against the basket's mean position first, simpler, symmetric, no reference outcome to pick. Checked it by hand against `on_resolution`'s real settlement math and it was wrong two ways: a flat (zero) position could show phantom exposure just for sitting next to an imbalanced sibling, and it didn't net a long-A/short-B position at all, even though that looks like it should be hedged. It isn't, for 3+ outcomes that's a leveraged directional bet, worst case is worse than either leg alone, confirmed by computing the actual resolution PnL for all N possible winners. The version that shipped attributes each event's real worst-case loss (same PnL formula `on_resolution` already trusts) to its members proportionally by their own gross exposure, which gets both cases right.

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance.
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem.* Mathematics and Financial Economics.
- Bergault, P., Evangelista, D., Guéant, O. & Vieira, D. (2018). *Closed-form approximations in multi-asset market making.* arXiv:1810.04383.
- Hanson, R. (2003). *Combinatorial information market design.* Information Systems Frontiers.
- Fortnow, L. & Sami, R. (2012). *Multi-outcome and multidimensional market scoring rules.* arXiv:1202.1712.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). *The price impact of order book events.* Journal of Financial Econometrics.
- Prelec, D. (1998). *The probability weighting function.* Econometrica.
- Thaler, R. & Ziemba, W. (1988). *Anomalies: Parimutuel betting markets.* Journal of Economic Perspectives.
- Glosten, L. & Milgrom, P. (1985). *Bid, ask and transaction prices in a specialist market.* Journal of Financial Economics.
- Saguillo, O., Ghafouri, V., Kiffer, L. & Suárez-Tangil, G. (2025). *Unravelling the probabilistic forest: arbitrage in prediction markets.* AFT 2025, arXiv:2508.03474.
