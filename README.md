# Multi-Asset Systematic Trading Framework

A modular, research-grade framework for systematic trading across **Equities, Rates, FX, Commodities and Credit**.  
The strategy is signal-driven (momentum, mean reversion, carry), combined via portfolio optimization and evaluated through a walk-forward backtest.

---

## Project Structure

```
.
├── equities/
│   ├── data.py          # Fetch prices & fundamentals
│   └── signals.py       # Momentum, mean reversion, carry signals
├── rates/
│   ├── data.py          # Fetch yield curves & rate futures
│   └── signals.py       # Momentum, mean reversion, carry signals
├── fx/
│   ├── data.py          # Fetch spot rates, forwards & IR differentials
│   └── signals.py       # Momentum, mean reversion, carry signals
├── commodities/
│   ├── data.py          # Fetch futures prices & term structure
│   └── signals.py       # Momentum, mean reversion, carry signals
├── credit/
│   ├── data.py          # Fetch CDS spreads & bond yields
│   └── signals.py       # Momentum, mean reversion, carry signals
│
├── optimization.py      # Portfolio optimization (mean-variance / risk parity / BL)
├── constraints.py       # Hard & soft portfolio constraints
├── backtest.py          # Walk-forward backtest engine
├── performance.py       # Performance statistics & reporting
└── main.py              # Orchestration — run everything from here
```

---

## Module Overview

### Asset Class Packages (`equities/`, `rates/`, `fx/`, `commodities/`, `credit/`)

Each package is self-contained and follows the same interface:

| File | Responsibility |
|------|---------------|
| `data.py` | Connect to the data vendor, fetch and clean raw market data, return `pd.DataFrame` objects indexed by date |
| `signals.py` | Compute the three alpha signals and expose a `combined_signal()` function that returns a z-scored cross-sectional signal |

#### Signals

| Signal | Description |
|--------|-------------|
| **Momentum** | Returns over a trailing window (typically 12-1 months); long winners, short losers |
| **Mean Reversion** | Short-term reversal relative to a medium-term moving average |
| **Carry** | Asset-class-specific carry: dividend yield (equities), roll-down (rates), forward discount (FX), roll-yield (commodities), OAS (credit) |

All signals are z-scored cross-sectionally before combination. The `combined_signal()` function accepts a `weights` dict to control the blend.

---

### `optimization.py`

Combines all per-asset-class signal views into target portfolio weights.

Key functions:
- `build_views(signals)` → expected return vector `mu` and uncertainty matrix `omega`
- `estimate_covariance(returns)` → covariance matrix (sample / Ledoit-Wolf / EWM)
- `optimize(mu, cov, constraints)` → optimal weight vector

Supported objectives: `mean_variance`, `risk_parity`, `max_sharpe`.

---

### `constraints.py`

Defines and validates all portfolio constraints fed into the optimizer:

| Constraint | Parameter |
|-----------|-----------|
| Gross leverage | `gross_leverage` (default 1.0) |
| Net exposure | `net_exposure` tuple (min, max) |
| Position limit | `position_limit` per instrument |
| Asset class bounds | `asset_class_limits` dict |
| Turnover limit | `turnover_limit` (requires current weights) |

---

### `backtest.py`

Walk-forward simulation engine:
- Rebalances on a configurable schedule (`rebalance_freq`: daily / weekly / monthly / quarterly)
- Applies one-way transaction costs in basis points
- Records NAV, daily returns, weights and turnover for each period

---

### `performance.py`

Produces a full suite of performance statistics from NAV and return series:

| Metric | Description |
|--------|-------------|
| Annualized Return | Geometric annualized return |
| Annualized Volatility | Realized vol (252-day basis) |
| Sharpe Ratio | Risk-adjusted return vs. risk-free rate |
| Sortino Ratio | Downside-only volatility denominator |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar Ratio | Ann. return / |Max Drawdown| |
| Hit Rate | % of positive return days |
| Avg Win/Loss | Average win over average loss |
| Skewness / Kurtosis | Return distribution shape |
| VaR (95%) | 1-day 95% Value at Risk |
| CVaR / ES (95%) | Conditional VaR (expected shortfall) |

Call `performance.summary(nav)` to get all metrics as a `pd.Series`.

---

### `main.py`

The single entry point. Edit the configuration block at the top to set:
- `START` / `END` — backtest dates
- `REBALANCE_FREQ` — rebalance frequency (pandas offset alias)
- `TRANSACTION_COST_BPS` — one-way cost in basis points
- `RISK_FREE_RATE` — annualized risk-free rate

Then run:

```bash
python main.py
```

---

## Data Vendors

Data sources are not yet configured. Each `data.py` module exposes a clean interface; swap in your preferred vendor by implementing the functions:

- Bloomberg / Refinitiv / FactSet via their Python APIs
- `yfinance` or `pandas-datareader` for quick prototyping
- Proprietary flat files / databases

---

## Dependencies

```
pandas
numpy
scipy          # optimization
scikit-learn   # covariance estimation (Ledoit-Wolf)
```

Install with:

```bash
pip install pandas numpy scipy scikit-learn
```

---

## Performance Evaluation

Once the backtest is implemented, the `performance.py` module provides a full tearsheet via `performance.summary(nav)`. Future additions will include:
- Rolling Sharpe and drawdown charts
- Factor attribution (signal contribution breakdown)
- Benchmark-relative statistics (Information Ratio, Tracking Error, Beta)
- Per-asset-class contribution to overall P&L

---

## Roadmap

- [ ] Implement data connectors (vendor TBD)
- [ ] Implement signal logic per asset class
- [ ] Implement optimizer (mean-variance baseline)
- [ ] Implement backtest engine
- [ ] Add performance charts and attribution
- [ ] Add risk reporting (factor exposures, Greeks where applicable)
