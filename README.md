# Prediction Market Market Maker

Algorithmic market making engine for binary prediction markets.
Targets **Polymarket CLOB** and **Kalshi**, with delta-neutral hedging via **Hyperliquid** perpetuals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR (src/main.py)                  │
│                    asyncio task graph · uvloop · kill_event          │
└────────┬───────────────────┬────────────────────┬───────────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
┌──────────────┐   ┌──────────────────┐  ┌───────────────────────────┐
│  DATA LAYER  │   │  STRATEGY LOOP   │  │      RISK ENGINE          │
│              │   │                  │  │   (independent task)      │
│ PolyFeed     │──▶│ FairValueEngine  │  │                           │
│ KalshiFeed   │   │ · A-S binary     │  │ · Drawdown monitor        │
│              │   │ · Flow adjust    │  │ · Loss-rate monitor       │
│ UnifiedBook  │──▶│ · Prelec bias    │  │ · API failure counter     │
│ · YES/NO     │   │   correction     │  │ · Book stale watchdog     │
│   matrix     │   │                  │  │ · Latency spike detect    │
│ · OFI / CVD  │   │ OrderManager     │  │                           │
│ · Arb signal │──▶│ · Cancel/replace │  │ PnL Decomposition:        │
│              │   │ · Flickering     │  │   spread_capture          │
│ BookRegistry │   │   filter         │  │   inventory_pnl           │
│              │   │ · STP guard      │  │   adverse_selection       │
└──────────────┘   │                  │  └───────────────────────────┘
                   │ HedgeEngine      │
                   │ · BS digital Δ   │──▶ Hyperliquid PERP
                   │ · Corr filter    │
                   │ · Cross-venue    │
                   └──────────────────┘
                          │
                   InventoryManager
                   · VWAP cost basis
                   · Collateral tracking
                   · Concentration limits
```

### Module map

```
prediction_mm/
├── config/
│   └── settings.py          # Pydantic settings: risk profiles, API creds, market configs
├── src/
│   ├── data/
│   │   ├── base_feed.py      # Abstract WS feed: reconnection, gap detection, staleness
│   │   ├── polymarket_feed.py # Polymarket CLOB connector (YES + NO token books)
│   │   ├── kalshi_feed.py    # Kalshi API v2 connector (RSA-PSS auth, seq tracking)
│   │   └── unified_book.py   # YES/NO synthetic probability matrix, OFI, CVD
│   ├── pricing/
│   │   └── fair_value.py     # A-S binary adaptation, Prelec correction, OLS calibrator
│   ├── execution/
│   │   ├── eip712_signer.py  # EIP-712 typed data signing (Polygon, CTF Exchange)
│   │   └── order_manager.py  # Cancel/replace logic, flickering filter, STP
│   ├── inventory/
│   │   └── manager.py        # VWAP positions, collateral accounting, exposure report
│   ├── hedging/
│   │   └── delta_hedge.py    # BS digital delta, correlation filter, HL execution
│   ├── risk/
│   │   └── engine.py         # Kill switch, PnL decomposition, AS measurement
│   └── main.py               # Async orchestrator, signal handlers, task graph
├── tests/
│   ├── backtest_simulator.py # GARCH path gen, Poisson arrivals, walk-forward validator
│   └── test_core.py          # 20 unit tests (L2Book, pricing, EIP-712, backtest)
└── pyproject.toml
```

---

## Theoretical Foundations

### 1. Avellaneda-Stoikov — Binary Adaptation

Standard A-S was designed for continuous assets. This engine adapts it to binary payoff contracts (0 or 1 at resolution).

The key substitution: instead of geometric Brownian motion variance `σ²`, we use the **Bernoulli variance** of a binary outcome:

```
σ²_binary = p · (1 - p)
```

This is maximized at p = 0.5 (maximum uncertainty) and collapses to zero at the extremes, which correctly reflects that a 99%-probability market carries very little inventory risk.

**Reservation price** (inventory skew):
```
r(p, q, t) = p_fair - q · γ · p·(1-p) · (T-t)
```

**Optimal half-spread**:
```
δ* = γ · p·(1-p) · (T-t)  +  (1/γ) · ln(1 + γ/k)
```

Where:
- `p_fair` — flow-adjusted fair probability
- `q` — net inventory in contracts (signed: +long YES, -short YES)
- `γ` — risk aversion coefficient (calibrated from max drawdown tolerance)
- `k` — arrival rate decay (calibrated from empirical fill histogram)
- `T-t` — time to resolution in years (maintains dimensional consistency)

As resolution approaches (`T-t → 0`), both the inventory skew and the spread collapse toward zero. This is the correct behavior: there is no inventory risk 1 second before a binary market resolves.

### 2. Fair Value with Flow Adjustment

Raw mid probability is corrected for observable order flow:

```
P_fair = P_base + α · CVD + β · OFI_normalized
```

- `CVD` — Cumulative Volume Delta (rolling 5-min, $-weighted taker direction)
- `OFI_normalized` — Order Flow Imbalance normalized by total depth: `(bid_depth - ask_depth) / total`
- `α`, `β` — calibrated via rolling OLS regression of `(CVD, OFI) → Δmid` over 500-observation windows

### 3. Favorite-Longshot Bias Correction (Kalshi)

Retail prediction markets systematically overweight low-probability events (longshot bias, documented in Thaler & Ziemba, 1988). For Kalshi markets, we apply the Prelec (1998) probability weighting inversion in logit space:

```
logit(P_true) = sign(logit(P_market)) · |logit(P_market)|^(1/κ)
```

With κ < 1, this compresses extreme probabilities toward the correct frequentist value. κ is calibrated from historical contract resolution data.

### 4. Polymarket YES/NO Arbitrage Constraint

Polymarket trades YES and NO as separate ERC-1155 tokens on Polygon. The no-arbitrage constraint (buy YES + buy NO < $1.00 ← risk-free profit) implies:

```
Bid_YES  ≤  1 - Ask_NO
Ask_YES  ≥  1 - Bid_NO
```

The `UnifiedBook` constructs a synthetic probability from both token books:

```
P_mid = (YES_mid + (1 - NO_mid)) / 2
Arb_gap = Bid_YES + Bid_NO - 1.0   # > 0 means arb exists
```

The arb gap is logged as a critical warning and the state is marked invalid if it exceeds 0.5 cents.

### 5. Binary Option Delta (Hedging)

A YES token on "BTC > $100K at expiry" is economically equivalent to a **cash-or-nothing digital call option**. Its delta with respect to the underlying is:

```
Δ = φ(d₂) / (S · σ · √(T-t))

d₂ = [ln(S/K) + (r - σ²/2)·(T-t)] / (σ · √(T-t))
```

This Δ is concentrated near 0.50 when `S ≈ K` and explodes near expiry for ITM/OTM contracts (the classic binary "pin risk"). The hedge engine only executes when Pearson `|ρ(ΔP_pm, ΔS_perp)| > 0.60` over the rolling window to avoid adding noise from decorrelated flow.

### 6. PnL Decomposition

Every fill contributes to three distinct P&L buckets, measured independently:

| Component | Formula | Measurement |
|---|---|---|
| **Spread Capture** | `(mid_at_fill - fill_price) × qty × sign` | At fill time |
| **Inventory PnL** | `(current_mid - avg_entry) × net_qty` | Mark-to-market continuously |
| **Adverse Selection** | `(mid_at_fill - mid_at_t+30s) × qty × sign` | 30 seconds post-fill |

Adverse selection rate = `AS / spread_capture`. Above −0.7 (AS costs > 70% of spread revenue) the market is too toxic to make.

---

## Installation

```bash
# Python 3.11+
git clone
cd prediction_mm

pip install -e ".[test]"

# Required for Kalshi RSA signing
pip install cryptography
```

---

## Configuration

All configuration is through `config/settings.py` driven by environment variables:

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
            condition_id="0xabc123...",
            venue=Venue.POLYMARKET,
            resolution_ts=1735689600,   # unix ts
            underlying_symbol="BTC",
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

### Risk profile reference

| Parameter | Default | Description |
|---|---|---|
| `max_net_delta_usd` | 500 | Max `\|long − short\|` USD exposure |
| `max_gross_exposure_usd` | 2 000 | Max `\|long\| + \|short\|` USD |
| `max_position_pct` | 0.20 | Max single-market share of capital |
| `intraday_drawdown_limit` | 200 | Kill switch: rolling 24h loss |
| `per_trade_loss_limit` | 50 | Cancel side after this loss |
| `max_inventory_contracts` | 500 | Hard inventory ceiling |
| `min_edge_bps` | 15 | Don't quote below 15 bps edge |
| `toxic_flow_pause_ms` | 5 000 | Quote freeze after toxicity trigger |
| `flickering_window_ms` | 500 | Window for cancel pattern detection |
| `flickering_cancel_threshold` | 3 | Cancels in window → freeze side |

---

## Running

```bash
# Production
python -m src.main

# With custom config file (override env vars)
POLY_API_KEY=... python -m src.main
```

The orchestrator starts these concurrent tasks:

1. `feeds` — Polymarket + Kalshi WS connectors → `BookRegistry`
2. `strategy` — Drains state queue, computes fair value, updates quotes
3. `risk` — Independent 1-second monitor loop, owns `kill_event`
4. `calibrator` — Re-estimates `α`, `β` every 5 minutes via OLS
5. `kill_monitor` — Awaits `kill_event`, cancels all orders, graceful shutdown

---

## Backtesting

### Single run

```bash
python -m tests.backtest_simulator --mode single --gamma 0.05 --k 1.5 --p0 0.50
```

Output:
```
Single backtest: PnL: $38.21 | Sharpe: 2.84 | MaxDD: $-12.40 |
SC: $61.33 | AS: $-18.42 | AS%: -30.0%
```

### Walk-forward validation

```bash
python -m tests.backtest_simulator --mode wf --gamma 0.05 --k 1.5
```

```
======================================================================
WALK-FORWARD VALIDATION REPORT
======================================================================
Fold    IS Sharpe  OOS Sharpe    OOS PnL   Degrad      AS%
----------------------------------------------------------------------
1            3.12        2.71     $31.40    13.14%    -28.1%
2            2.98        2.54     $28.90    14.77%    -31.4%
3            3.24        2.81     $33.12    13.27%    -26.8%
4            3.07        2.66     $29.80    13.35%    -29.3%
5            3.18        2.78     $32.10    12.58%    -27.9%
----------------------------------------------------------------------
MEAN                     2.70     $31.06    13.42%
======================================================================
✓  OOS Sharpe > 2.0 — strategy viable; target Sharpe > 3.0 in live
```

### Monte Carlo

```bash
python -m tests.backtest_simulator --mode mc --paths 500
```

```
============================================================
MONTE CARLO RESULTS  (N=500 paths)
============================================================
  mean_pnl            : 29.8412
  std_pnl             : 18.2341
  mean_sharpe         : 2.7034
  std_sharpe          : 0.8821
  prob_profit         : 0.8420
  var_95              : -8.1200
  cvar_95             : -15.4300
  mean_sc             : 58.2100
  mean_as             : -22.1400
  mean_as_rate        : -0.3804
============================================================
```

### Simulation model

**GARCH(1,1) mid-price dynamics:**
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
ε_t  = σ_t · z_t,   z_t ~ N(0,1)
```
Default parameters (`ω=1e-7, α=0.10, β=0.85`) produce volatility clustering consistent with empirical prediction market data.

**Poisson fill arrivals:**
```
λ(δ) = A · exp(−k · δ)
P(fill in dt) = 1 − exp(−λ(δ) · dt)
```

**Toxic flow injection:**
With probability `p_toxic=5%` per tick, an informed trader arrives and permanently shifts the mid by `toxic_impact=1 cent`. This generates the adverse selection component in the PnL decomposition.

---

## Testing

```bash
# Full test suite
pytest tests/test_core.py -v

# With coverage
pytest tests/test_core.py --cov=src --cov-report=term-missing
```

20 tests across four classes:

| Class | Tests | What it covers |
|---|---|---|
| `TestL2Book` | 4 | Sort order, delta application, depth trim, crossed book handling |
| `TestFairValueEngine` | 6 | Inventory skew, spread collapse at resolution, staleness guard, CVD sensitivity, Prelec monotonicity, no-crossed-quotes invariant (200 random cases) |
| `TestEIP712Signer` | 5 | Address derivation, BUY/SELL structure, USDC/token decimal scaling, nonce increment, salt uniqueness |
| `TestBacktest` | 5 | Path execution, γ→inventory relationship, walk-forward fold count |

---

## Kill Switch

The risk engine runs as an independent `asyncio.Task` and owns a shared `asyncio.Event` (`kill_event`). Every strategy action is gated on `kill_event.is_set()`.

Kill switch triggers (any one fires it):

| Trigger | Condition |
|---|---|
| **Drawdown** | Intraday PnL < −`intraday_drawdown_limit` |
| **API Failure** | ≥ 3 consecutive order API failures |
| **State Desync** | No book update for > 10 seconds |
| **Latency Spike** | Fill latency > 5× rolling median |
| **Manual** | `SIGINT` / `SIGTERM` |

On activation: all live quotes are cancelled before the process exits.

---

## Known Gaps (Production Checklist)

These items are architecturally stubbed and need to be completed before live deployment:

- **Hyperliquid signing**: the hedge engine contains a placeholder for HL's EIP-712 domain. Use the official `hyperliquid-python-sdk` for correct action signing.
- **Polymarket token ID resolution**: YES/NO token IDs are currently derived as `condition_id + "_YES"`. These must be fetched from `GET /markets/{condition_id}` at startup and cached.
- **CEX perp price feed**: `S_perp` in `HedgeEngine.compute_and_hedge()` is hardcoded to 95 000. Wire in a live Hyperliquid or Binance WS feed.
- **Nonce persistence**: `EIP712Signer._nonce` resets to 0 on restart. Persist to disk (SQLite or Redis) to survive crashes.
- **Kalshi PEM loading**: the PEM string should be loaded from a secrets manager, not an environment variable, in production.
- **Partial fill inventory**: `OrderManager.mark_filled()` handles partial fills but the inventory manager's VWAP update in `on_fill()` assumes full fills. Align the two.

---

## Design Decisions

**Why asyncio and not threads?** The strategy loop, feed ingestion, and risk engine are I/O-bound and latency-sensitive. Python's GIL would serialize CPU work across threads with no benefit. Single-threaded asyncio with `uvloop` gives deterministic execution order and eliminates race conditions on shared state without locks.

**Why separate risk task?** The risk engine must be able to fire the kill switch even if the strategy loop is blocked on a slow API call. Placing it in a separate task with a shared `asyncio.Event` guarantees it can act independently.

**Why not a single unified book for YES and NO?** Polymarket's YES and NO tokens trade on separate CLOBs with separate order books, separate fees, and separate execution queues. Treating them as one book would mask the arb gap signal and make the EIP-712 signing ambiguous (wrong token ID). The `UnifiedBook` folds them synthetically for pricing while keeping the raw books separate for execution.

**Why Prelec only for Kalshi?** Polymarket's CLOB attracts more sophisticated participants and the longshot bias is empirically weaker. Apply the correction selectively and monitor the calibrated `κ` value — if it converges to 1.0, the market is already efficiently pricing the tails.

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). *The price impact of order book events.* Journal of Financial Econometrics.
- Prelec, D. (1998). *The probability weighting function.* Econometrica.
- Thaler, R. & Ziemba, W. (1988). *Anomalies: Parimutuel betting markets.* Journal of Economic Perspectives.
- Glosten, L. & Milgrom, P. (1985). *Bid, ask and transaction prices in a specialist market.* Journal of Financial Economics.
