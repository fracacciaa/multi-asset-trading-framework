# Multi-Asset Systematic Trading Framework

A research-grade signal framework for systematic trading across **Equities, Rates, FX and Commodities**.  
Each asset class has a self-contained production pipeline that serialises PnL to parquet; a cross-asset blend layer combines the four sleeves into a single portfolio.

IS/OOS split: **2020-12-31** (all asset classes; rates blend uses 2018-12-31).

---

## Repository Layout

```
.
├── core/
│   ├── asset_blend.py      # Cross-asset inv-vol blend (monthly, 10% vol target)
│   ├── main.py             # Full pipeline: all four sleeves → cross-asset blend
│   ├── optimization.py     # Covariance estimation + mean-variance / risk-parity
│   ├── performance.py      # Sharpe, MDD, Calmar, VaR, CVaR, hit rate …
│   ├── pipeline_report.py  # IS/OOS tearsheet formatter
│   └── signal_metrics.py   # Per-signal Sharpe / turnover helpers
│
├── rates/
│   ├── production/
│   │   ├── run.py                      # ← entry point: generates all rates parquets
│   │   ├── ns_classic/                 # Nelson-Siegel MR signal
│   │   │   ├── bootstrap.py            # OIS curve bootstrapping
│   │   │   ├── ns_fit.py               # DV01-weighted NS regression + z-scores
│   │   │   ├── signal.py               # build_dv01_positions (net-DV01 neutral)
│   │   │   └── pnl.py                  # Full swap repricing backtest
│   │   ├── momentum/                   # Daily TSMOM signal
│   │   ├── carry/                      # Within-curve OIS carry
│   │   └── data.py / ois_data.py       # Curve data loaders
│   ├── signals/                        # ← parquet outputs (read by blend)
│   │   ├── mean_reversion_pnl.parquet      date × market, bp
│   │   ├── mean_reversion_dv01_pos.parquet date × (market, tenor)
│   │   ├── mean_reversion_zscores.parquet  date × (market, tenor)
│   │   ├── momentum_a_daily_pnl.parquet    date × market, bp
│   │   ├── carry_ois_par_pnl.parquet       date × market, bp
│   │   └── ns_factors.parquet / ret_d / rx_m …
│   └── blend.py            # Rolling Ridge MSRR across the three signals
│
├── equities/
│   ├── production/
│   │   ├── pipeline.py     # ← entry point: signals + blend
│   │   ├── signal_lib.py   # TSMOM, threshold MR, earnings revision
│   │   ├── view_formation.py
│   │   └── data.py / sectors.py
│   ├── signals/            # ← parquet outputs
│   └── blend.py
│
├── fx/
│   ├── production/
│   │   ├── run.py          # ← entry point: generates all FX parquets
│   │   ├── signal_a.py     # Momentum (XS, f=9m h=1)
│   │   ├── signal_d.py     # Carry (Dupuy E-only leverage)
│   │   ├── signal_e.py     # Carry (XS + E3 aggregate de-risk)
│   │   ├── ns_rv.py        # MR: NS LOO view-formation TS signal
│   │   ├── ns_fwd.py       # NS forward curve fitting
│   │   └── data.py / bloomberg.py
│   ├── signals/            # ← parquet outputs
│   └── blend.py
│
├── commodities/
│   ├── production/
│   │   ├── backtest.py     # ← entry point: signal comparison harness
│   │   ├── signals.py      # TSMOM-252, XS carry enhanced, curvature
│   │   └── data.py
│   ├── signals/            # ← parquet outputs
│   └── blend.py
│
└── tests/
    ├── rates/
    │   ├── ns_classic/     # 53 tests: bootstrap, swap PV, NS fit, leakage, net-DV01
    │   └── carry/          # signal, returns, orthogonality, no-lookahead
    ├── equities/           # blend tests
    ├── fx/                 # blend tests
    └── test_position_turnover.py   # cross-asset turnover sanity
```

---

## Rates Pipeline

The rates sleeve is the most fully developed.  Run once to populate `rates/signals/`; all downstream queries read parquet.

### 1. Generate parquets

```bash
python -m rates.production.run
```

Outputs (all in `rates/signals/`):

| File | Shape | Description |
|---|---|---|
| `mean_reversion_pnl.parquet` | date × market | NS classic MR daily PnL, full swap repricing (bp) |
| `mean_reversion_dv01_pos.parquet` | date × (market, tenor) | Held DV01 positions — TC proxy |
| `mean_reversion_zscores.parquet` | date × (market, tenor) | NS residual z-scores |
| `momentum_a_daily_pnl.parquet` | date × market | Daily TSMOM PnL, full repricing (bp) |
| `carry_ois_par_pnl.parquet` | date × market | Within-curve OIS carry PnL, full repricing (bp) |
| `ns_factors.parquet` | date × (market, factor) | NS level / slope / curvature |
| `ret_d.parquet` | date × (market, tenor) | Daily constant-maturity returns (bp) |
| `rx_m.parquet` | date × (market, tenor) | Monthly Sihvonen log excess returns (bp) |

### 2. Run the blend

```python
from rates.blend import run_blend
r_is, r_oos = run_blend()   # returns IS and OOS daily return Series
```

The blend applies:
1. **Cross-market**: inverse-vol weights (IS σ, frozen) within each signal
2. **Cross-signal**: rolling Ridge MSRR (lookback 252d, walk-forward) across MR / momentum / carry

Transaction costs are deducted per signal from their respective position parquets:

| Signal | TC formula | Units |
|---|---|---|
| Carry | `\|Δpos_dv01\| × h_bp` | bp |
| Momentum | `\|Δpos\| × h_bp × eff_dur` | bp |
| Mean reversion | `\|Δ(net_dv01)\| × h_bp` | bp — net across tenors, not tenor by tenor |

### Key design decisions

- **Full swap repricing** — PnL is computed from `swap_pv(K, T−dt, curve_t)`, not `−D × Δy`. Carry, roll-down and curve moves are captured in one number.
- **Net-DV01 neutral MR** — z-scores are demeaned row-wise before position sizing so `Σ_i dv01_tgt_i = 0` at every rebalance. The DV01-weighted OLS condition `Σ_i w_i·z_i = 0` is a property of the NS fit and is distinct from the plain-sum neutrality enforced here.
- **TC on net DV01** — within-market tenor rebalancing (NS shape adjustment) is free; cost is charged only on changes in total market allocation.

---

## Other Asset Class Pipelines

### FX

```bash
python -m fx.production.run
```

Signals: momentum (XS f=9m h=1), carry (Dupuy leverage + E3 de-risk), MR (NS leave-one-out view-formation TS + fly).  Monthly and daily outputs in `fx/signals/`.

### Equities

```python
from equities.production.pipeline import run_eq_prod
run_eq_prod()
```

Signals: TSMOM, threshold mean reversion, earnings revision.  Outputs in `equities/signals/`.

### Commodities

```python
from commodities.production.backtest import run_comparison
run_comparison()
```

Signals: TSMOM-252, XS carry enhanced, curvature.  Outputs in `commodities/signals/`.

---

## Full Cross-Asset Pipeline

```bash
python -m core.main
```

Runs all four sleeves, then blends them with static inverse-vol weights (IS σ, frozen) targeting 10% annual vol.

---

## Tests

```bash
python -m pytest tests/
```

214 tests.  The rates suite (`tests/rates/`) covers bootstrap round-trips, swap PV sign conventions, NS fit properties, leakage audits, placebo Sharpe distribution, and net-DV01 neutrality.

---

## Dependencies

```
pandas
numpy
scipy
scikit-learn
```

```bash
pip install pandas numpy scipy scikit-learn
```
