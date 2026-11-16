# Build spec: Nelson-Siegel fair value and curve residual signal



Implement this as a new package inside the existing repo. Do not modify existing
modules. The point is a clean, testable path from raw par yields to a ranked set
of signal specifications, so that spec selection is honest.

Target: `src/nsrv/` (Nelson-Siegel relative value).

---

## 0. Non-negotiable rules

These are correctness requirements, not style preferences. Violating any of them
invalidates the backtest.

1. **No lookahead in the fit.** The curve at date `t` is fitted using only data
   at date `t`. The lambda hyperparameter is tuned once on a training block and
   frozen. Rolling statistics (means, vols) at date `t` use data up to and
   including `t-1`.
2. **Signal lag.** A residual computed from date `t` closing yields can only
   drive a position held from `t+1`. Every position series must be shifted by at
   least one day before being multiplied by a return.
3. **Leave-one-out residuals.** With 8 to 12 tenors the in-sample residual is
   biased toward zero at exactly the dislocated tenor. Always compute and trade
   the LOO residual. Keep the in-sample residual for diagnostics only.
4. **Everything in basis points.** Convert on load. Never mix percent and bp.
5. **Deterministic.** Seed anything stochastic. Same input must give byte-identical
   output.
6. **One spec = one config object.** No spec may be expressible only as a code edit.

---

## 1. Repository layout

```
src/nsrv/
    __init__.py
    config.py          # frozen dataclasses, all tunables
    data.py            # load + validate + align
    ns_core.py         # loadings, fit, LOO residuals
    lambda_tune.py     # global lambda grid search
    fitting.py         # day loop -> residual panel
    structures.py      # tenor residuals -> slope / fly residuals
    signals.py         # residual -> z -> position
    pnl.py             # position -> daily PnL, costs
    evaluate.py        # Sharpe, deflated Sharpe, half-life, consistency
    experiment.py      # spec grid runner
    cli.py
tests/
    test_ns_core.py
    test_structures.py
    test_signals.py
    test_no_lookahead.py
```

Dependencies: `numpy`, `pandas`, `scipy`, `pyyaml`. Nothing else. No sklearn, no
statsmodels, no plotting in the library.

---

## 2. Data contract

### 2.1 Input

A single long-format parquet or CSV with exactly these columns:

| column    | type      | notes                                        |
|-----------|-----------|----------------------------------------------|
| `date`    | date      | business day, no duplicates per (curve,tenor) |
| `curve_id`| string    | e.g. `US`, `DE`, `GB`. One curve per id       |
| `tenor`   | float     | years. 2.0, 5.0, 10.0, not "2Y"               |
| `yield`   | float     | percent on input, converted to bp on load     |

### 2.2 Loader

```python
# data.py
def load_curves(path: str, *, units: str = "pct") -> pd.DataFrame:
    """Returns long frame with yield in bp, sorted, validated."""
```

Validation that must raise, not warn:
- duplicate `(date, curve_id, tenor)` rows
- any tenor <= 0
- any `|yield|` > 5000 bp after conversion
- a `curve_id` with fewer than 4 tenors on any date

### 2.3 Wide accessor

```python
def to_wide(df: pd.DataFrame, curve_id: str) -> pd.DataFrame:
    """index=date, columns=tenor (sorted float), values=yield in bp."""
```

Missing tenors on a given date are allowed and must be dropped from that day's
fit, not filled. Record `n_obs` per day.

### 2.4 Roll handling

Generic on-the-run tickers jump on reroll. Provide:

```python
def flag_rolls(wide: pd.DataFrame, *, z_thresh: float = 6.0) -> pd.DataFrame:
    """Boolean frame, True where that tenor's daily change is a |z| > z_thresh
    outlier against a 250d trailing vol of its own daily changes."""
```

This is a diagnostic flag plus an optional exclusion mask. Do not auto-drop.
The spec grid decides.

---

## 3. `ns_core.py`

### 3.1 Loadings

```python
def ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """(n, 3): [level, slope, curvature]."""
    x = np.maximum(np.asarray(tau, float) * lam, 1e-8)
    d = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), d, d - np.exp(-x)])
```

Optional fourth factor for the Bjork-Christensen variant:

```python
def bc_loadings(tau, lam):
    X = ns_loadings(tau, lam)
    return np.column_stack([X, np.exp(-2.0 * lam * np.asarray(tau, float))])
```

Adjusted Svensson (lam2 fixed at 2*lam, no second free decay):

```python
def svensson_adj_loadings(tau, lam):
    X = ns_loadings(tau, lam)
    x2 = np.maximum(np.asarray(tau, float) * 2.0 * lam, 1e-8)
    d2 = (1.0 - np.exp(-x2)) / x2
    return np.column_stack([X, d2 - np.exp(-x2)])
```

Two-factor control (level + slope only):

```python
def ns2_loadings(tau, lam):
    return ns_loadings(tau, lam)[:, :2]
```

Register these in a dict so the config can name one:

```python
LOADINGS = {
    "ns2": ns2_loadings,
    "ns": ns_loadings,
    "bc": bc_loadings,
    "svensson_adj": svensson_adj_loadings,
}
NFACTORS = {"ns2": 2, "ns": 3, "bc": 4, "svensson_adj": 4}
```

### 3.2 Fit

Generalized ridge toward a prior mean, with per-factor weights.

```python
@dataclass(frozen=True)
class NSFit:
    beta: np.ndarray        # (k,)
    lam: float
    basis: str
    tau: np.ndarray         # (n,)
    y: np.ndarray           # (n,) bp
    yhat: np.ndarray        # (n,) bp
    resid: np.ndarray       # (n,) in-sample, diagnostics only
    resid_loo: np.ndarray   # (n,) leverage-corrected, THIS is the signal input
    leverage: np.ndarray    # (n,) hat matrix diagonal
    sse: float
    n_obs: int

def fit_ns(
    tau, y, lam, *,
    basis: str = "ns",
    prior_mean: np.ndarray | None = None,   # (k,), None = zeros
    prior_w: np.ndarray | None = None,      # (k,), None = pure OLS
    weights: np.ndarray | None = None,      # (n,) observation weights
) -> NSFit:
```

Implementation:

```python
X = LOADINGS[basis](tau, lam)
k = X.shape[1]
W = np.ones(len(tau)) if weights is None else np.asarray(weights, float)
m = np.zeros(k) if prior_mean is None else np.asarray(prior_mean, float)[:k]
A = np.diag(np.zeros(k) if prior_w is None else np.asarray(prior_w, float)[:k])

Xw = X * W[:, None]
G = np.linalg.inv(Xw.T @ X + A)
beta = G @ (Xw.T @ y + A @ m)
yhat = X @ beta
r = y - yhat
h = np.einsum("ij,jk,ik->i", Xw, G, X)      # weighted hat diagonal
resid_loo = r / np.clip(1.0 - h, 1e-6, None)
```

Note on the prior weight units. `prior_w[j]` is the number of equivalent
observations of prior information on factor `j`. A legacy `shrink` in `[0,1]`
with isotropic penalty maps as:

```python
def shrink_to_prior_w(shrink: float, n_obs: int, k: int) -> np.ndarray:
    return np.full(k, shrink / max(1e-6, 1.0 - shrink) * n_obs)
```

Provide this for backward compatibility with the existing `fit_ns_peer_shrinkage`
behaviour, but the config should prefer explicit per-factor weights.

### 3.3 Predict

```python
def predict(fit: NSFit, tau_new) -> np.ndarray:
    return LOADINGS[fit.basis](np.asarray(tau_new, float), fit.lam) @ fit.beta
```

### 3.4 Par-space variant

Generic par yields are not zero rates. Provide an optional fit that treats the
NS curve as the zero curve and matches model par yields to observed.

```python
def model_par_yields(tau, beta, lam, basis="ns", freq=2) -> np.ndarray:
    out = np.empty(len(tau))
    for i, n in enumerate(tau):
        t = np.arange(1, int(round(n * freq)) + 1) / freq
        z = LOADINGS[basis](t, lam) @ beta / 10000.0     # bp -> decimal
        dfs = np.exp(-z * t)
        out[i] = freq * (1.0 - dfs[-1]) / dfs.sum() * 10000.0
    return out

def fit_ns_par(tau, y, lam, *, basis="ns", freq=2, **kw) -> NSFit:
    """scipy.optimize.least_squares on model_par_yields - y.
    x0 = beta from the linear fit_ns on y treated as zeros."""
```

For the LOO residual here, compute the Jacobian at the solution and use the
Gauss-Newton hat matrix `J (J'J + A)^-1 J'`. Do not skip this.

This variant is an axis in the grid, not the default. Test whether its residual
series correlates above 0.95 with the linear one. If it does, drop it.

---

## 4. `lambda_tune.py`

```python
def tune_global_lambda(
    curves: list[tuple[np.ndarray, np.ndarray]],   # (tau, y) per date
    *, basis: str = "ns",
    lo: float = 0.02, hi: float = 1.5, n: int = 60,
) -> float:
    """Grid search minimizing pooled SSE over the training block."""
```

Rules:
- Grid is log-spaced over `[lo, hi]`.
- Uses pure OLS at each lambda (no prior), so lambda and shrinkage do not
  confound.
- Called exactly once, on the training block only. Persist the value in the
  run manifest.
- Provide `tune_lambda_per_curve` returning `dict[curve_id, float]` as a grid
  axis.

Sanity assertion: reject a tuned lambda within 2% of either grid bound and raise,
since that means the grid is wrong.

---

## 5. `fitting.py`

### 5.1 Day loop

```python
def build_residual_panel(
    wide: pd.DataFrame,          # index=date, cols=tenor, bp
    cfg: FitConfig,
) -> ResidualPanel:
```

Returns a container holding:

| frame            | shape            | content                        |
|------------------|------------------|--------------------------------|
| `yhat`           | date x tenor     | fitted yields, bp              |
| `resid`          | date x tenor     | in-sample, bp                  |
| `resid_loo`      | date x tenor     | leverage-corrected, bp         |
| `leverage`       | date x tenor     | hat diagonal                   |
| `betas`          | date x factor    | fitted coefficients            |
| `diagnostics`    | date x {...}     | `n_obs`, `rmse`, `max_lev`     |

Loop body, in this order:

```python
for date in wide.index:
    row = wide.loc[date].dropna()
    if len(row) < cfg.min_tenors:
        record_nan_row(); continue
    tau, y = row.index.values.astype(float), row.values.astype(float)
    prior_m = resolve_prior_mean(cfg, date, state)
    prior_w = resolve_prior_w(cfg, len(tau))
    w = resolve_obs_weights(cfg, tau)
    fit = fit_ns(tau, y, lam_for(cfg, date), basis=cfg.basis,
                 prior_mean=prior_m, prior_w=prior_w, weights=w)
    store(date, fit)
    state.prev_beta_ols = fit_ns(tau, y, lam_for(cfg, date),
                                 basis=cfg.basis).beta   # UNSHRUNK anchor
```

The anchor update uses the unshrunk OLS betas. Anchoring on the shrunk betas
compounds the prior across days and the fit drifts.

### 5.2 Prior mean sources

`cfg.prior_source` is one of:

- `"none"` — pure OLS. **This is the control. Run it first.**
- `"prev_day"` — yesterday's unshrunk betas.
- `"peer_mean"` — cross-sectional mean of today's unshrunk betas across a peer
  group of curves, computed in a pre-pass. Requires two passes over the panel.
- `"ewma"` — EWMA of past unshrunk betas with a configured halflife.

For `"peer_mean"`, the pre-pass must fit every curve OLS first, then the second
pass shrinks each toward the group mean **excluding itself**, or the shrinkage is
partly toward its own estimate.

### 5.3 Prior weight profiles

`cfg.prior_profile`:

| name           | weight vector (times n_obs)      | rationale                        |
|----------------|----------------------------------|----------------------------------|
| `off`          | zeros                            | OLS control                      |
| `isotropic`    | `[s, s, s]`                      | current production behaviour     |
| `curvature`    | `[0.0, 0.25*s, 1.0*s]`           | leave level free, pin curvature  |
| `shape_only`   | `[0.0, s, s]`                    | level entirely unconstrained     |

with `s = shrink / (1 - shrink)` and `shrink` in config.

### 5.4 Observation weights

`cfg.obs_weights`:

- `"equal"`
- `"inv_duration"` — weight proportional to `1 / tau` so long tenors do not
  dominate a price-equivalent objective
- `"huber"` — iteratively reweighted, 2 passes, tuning constant 1.345 times a
  MAD-based scale estimate. Implement inline, do not import statsmodels.

Huber loop:

```python
w = np.ones(n)
for _ in range(cfg.huber_iters):
    fit = fit_ns(tau, y, lam, weights=w, ...)
    s = 1.4826 * np.median(np.abs(fit.resid - np.median(fit.resid)))
    u = fit.resid / max(s, 1e-8)
    w = np.where(np.abs(u) <= 1.345, 1.0, 1.345 / np.abs(u))
```

---

## 6. `structures.py`

Tenor residuals are correlated by construction. The tradeable object is a
structure.

```python
@dataclass(frozen=True)
class Structure:
    name: str                 # "5s10s", "3s5s10s"
    legs: tuple[float, ...]   # tenors
    weights: tuple[float, ...]  # signed, sum to 0 in DV01 terms for a fly
```

Build structure residuals as the same linear combination applied to the LOO
tenor residuals:

```python
def structure_residual(resid_loo: pd.DataFrame, s: Structure) -> pd.Series:
    cols = [resid_loo[t] for t in s.legs]
    return sum(w * c for w, c in zip(s.weights, cols))
```

### 6.1 Weighting conventions

Two modes, both must be implemented.

**Yield-neutral (naive).** Slope `[1, -1]`. Fly `[-0.5, 1, -0.5]`.

**Duration-neutral.** Weights scale by approximate DV01. For a par yield at
tenor `tau` with yield `y` in decimal and frequency `f`:

```python
def approx_dv01(tau, y, f=2):
    c = y
    n = int(round(tau * f))
    disc = (1 + y / f) ** -np.arange(1, n + 1)
    pv_cf = c / f * disc
    pv_cf[-1] += disc[-1]
    t = np.arange(1, n + 1) / f
    return float((t * pv_cf).sum() / (1 + y / f)) / 10000.0
```

Fly weights solve `w3*D3 + w10*D10 = -w5*D5` with `w5 = 1` and a chosen split.
Use a 50/50 DV01 split on the wings by default, and expose a `wing_split`
parameter.

Duration-neutral weights must be recomputed from the **previous day's** yields
when used to form today's position. Using today's yields leaks.

### 6.2 Default structure set

```python
DEFAULT_STRUCTURES = [
    Structure("2s5s",    (2., 5.),       ...),
    Structure("5s10s",   (5., 10.),      ...),
    Structure("10s30s",  (10., 30.),     ...),
    Structure("2s5s10s", (2., 5., 10.),  ...),
    Structure("5s10s30s",(5., 10., 30.), ...),
]
```

Only build a structure on a date where every leg has a non-null residual.

---

## 7. `signals.py`

### 7.1 Residual transform

All rolling statistics use `.shift(1)` before the window so that date `t`'s
z-score uses information through `t-1` only, plus today's residual level.

```python
def z_transform(r: pd.Series, cfg: SignalConfig) -> pd.Series:
    if cfg.demean_window:
        mu = r.shift(1).rolling(cfg.demean_window, min_periods=cfg.min_periods).mean()
    else:
        mu = 0.0
    sd = r.shift(1).rolling(cfg.vol_window, min_periods=cfg.min_periods).std()
    return (r - mu) / sd.clip(lower=cfg.vol_floor)
```

Transforms to expose as a config axis:

| name            | definition                                     |
|-----------------|------------------------------------------------|
| `raw`           | residual in bp, no scaling                     |
| `vol_scaled`    | `r / trailing_sd`                              |
| `demean_vol`    | `(r - trailing_mean) / trailing_sd`            |
| `xs_z`          | cross-sectional z across structures, same date |
| `xs_country_z`  | cross-sectional z across `curve_id`, same structure |
| `diff`          | `r.diff(cfg.diff_lag)` then vol-scaled         |

`xs_country_z` is the one most likely to help. The on-the-run premium is common
across sovereigns and sits in every residual as uncompensated correlated risk.

### 7.2 Position formation

```python
def position_from_z(z: pd.Series, cfg: SignalConfig) -> pd.Series:
```

Two modes:

**Linear.** `pos = -clip(z, -cap, cap)`. The sign is negative because the signal
is reversion: a rich structure (positive residual) is sold.

**Hysteresis.** Enter at `|z| > entry`, exit at `|z| < exit_`, hold otherwise.

```python
def pos_hysteresis(z, entry=1.25, exit_=0.5):
    s = np.sign(z.to_numpy())
    a = np.abs(z.to_numpy())
    out = np.zeros(len(z)); cur = 0.0
    for i in range(len(z)):
        if np.isnan(a[i]):
            out[i] = cur; continue
        if cur == 0.0:
            cur = -s[i] if a[i] > entry else 0.0
        elif a[i] < exit_:
            cur = 0.0
        out[i] = cur
    return pd.Series(out, index=z.index)
```

Then **always**:

```python
pos = pos.shift(cfg.lag)     # cfg.lag >= 1
```

Make `lag` a required field with no default. Forcing it to be named at the call
site is how this stops being forgotten.

---

## 8. `pnl.py`

### 8.1 Structure return

PnL for holding one unit of a structure is the position times the change in the
structure's market value, in bp of yield times the leg weights.

```python
def structure_bp_change(wide: pd.DataFrame, s: Structure) -> pd.Series:
    return sum(w * wide[t].diff() for w, t in zip(s.weights, s.legs))
```

Sign convention: a position of `+1` is receiving the structure, so

```python
pnl = pos * (-structure_bp_change)
```

Write the sign convention into a docstring and assert it with a test: a position
taken when the residual is positive must make money when the residual falls.

### 8.2 Costs

```python
def apply_costs(pnl, pos, *, bid_ask_bp: dict[str, float], structure: Structure):
    tc = sum(abs(w) for w in structure.weights) * bid_ask_bp[structure.name] * 0.5
    return pnl - tc * pos.diff().abs().fillna(0.0)
```

Default bid-ask in bp per leg, overridable per curve: 0.25 for tenors under 10y,
0.5 for 10y to 30y. These are placeholders. Replace with your own marks before
believing any number.

### 8.3 Scaling

Ex-post vol scaling to a 10% annualized target, using an **expanding** window so
it is not forward-looking:

```python
scale = target_vol / (pnl.expanding(min_periods=252).std() * np.sqrt(252))
pnl_scaled = pnl * scale.shift(1)
```

---

## 9. `evaluate.py`

Every spec must produce this record. No spec is ranked on Sharpe alone.

```python
@dataclass(frozen=True)
class SpecResult:
    spec_id: str
    cfg_hash: str
    sharpe: float
    sharpe_deflated: float
    sharpe_oos: float            # on the held-out block
    ir_turnover_adj: float
    half_life: float             # of the structure residual, days
    half_life_stability: float   # std of half-life across 4 subsamples
    hit_rate: float
    max_dd: float
    turnover: float              # mean |dpos| per day
    n_curves_positive: int       # cross-sectional sign consistency
    n_curves: int
    corr_to_level: float         # vs daily change in the 10y
    corr_to_vol: float           # vs change in a rate vol proxy
```

### 9.1 Half-life

Fit `dr_t = a + b * r_{t-1} + e` on the structure residual, then
`half_life = -ln(2) / ln(1 + b)`. Reject with NaN when `b >= 0` or `b <= -2`.

### 9.2 Deflated Sharpe

Bailey and Lopez de Prado (2014). Required because you are selecting a maximum
over a grid.

```python
def deflated_sharpe(sr, n_obs, n_trials, skew, kurt, sr_var_across_trials):
    e = 0.5772156649
    sr0 = np.sqrt(sr_var_across_trials) * (
        (1 - e) * norm.ppf(1 - 1.0 / n_trials)
        + e * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    num = (sr - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr**2)
    return float(norm.cdf(num / den))
```

Pass `n_trials` as the actual number of specs run, and
`sr_var_across_trials` as the sample variance of Sharpe across the grid. Report
it on every spec, not just the winner.

### 9.3 Selection rule, pre-committed

Write this into the README before running the grid.

1. Discard any spec with `sharpe_deflated < 0.95`.
2. Discard any spec with `|corr_to_level| > 0.3`. A curve residual signal that
   correlates with duration is a duration trade.
3. Rank survivors by `n_curves_positive / n_curves`, then by
   `half_life_stability` ascending.
4. Among specs within 0.2 Sharpe of the top, take the one with the fewest free
   parameters.

---

## 10. `config.py`

Frozen dataclasses, one per stage, composed into a `SpecConfig` with a stable
hash.

```python
@dataclass(frozen=True)
class FitConfig:
    basis: str = "ns"                 # ns2 | ns | bc | svensson_adj
    lambda_mode: str = "global"       # global | per_curve | rolling | free
    lambda_value: float | None = None
    rolling_window: int = 250
    prior_source: str = "none"        # none | prev_day | peer_mean | ewma
    prior_profile: str = "off"        # off | isotropic | curvature | shape_only
    shrink: float = 0.0
    obs_weights: str = "equal"        # equal | inv_duration | huber
    huber_iters: int = 2
    min_tenors: int = 5
    tenor_min: float | None = None    # drop short end below this
    tenor_max: float | None = None
    exclude_roll_days: bool = False
    par_space: bool = False

@dataclass(frozen=True)
class SignalConfig:
    transform: str = "demean_vol"
    demean_window: int | None = 60
    vol_window: int = 125
    min_periods: int = 30
    vol_floor: float = 0.5            # bp
    diff_lag: int = 5
    mode: str = "hysteresis"          # linear | hysteresis
    entry: float = 1.25
    exit_: float = 0.5
    cap: float = 3.0
    lag: int = 1                      # required, >= 1

@dataclass(frozen=True)
class SpecConfig:
    fit: FitConfig
    signal: SignalConfig
    structures: tuple[str, ...]
    weighting: str = "duration"       # yield | duration
    def spec_id(self) -> str:
        return hashlib.sha1(repr(asdict(self)).encode()).hexdigest()[:10]
```

---

## 11. `experiment.py`

### 11.1 Sample split

| block    | use                                        |
|----------|--------------------------------------------|
| train    | first 40% of dates. Lambda tuning only.    |
| select   | next 35%. Spec grid runs and ranking here. |
| holdout  | final 25%. Touched exactly once, at the end.|

Enforce with an assertion: the holdout slice must raise if accessed before a
`--unlock-holdout` flag is passed on the CLI.

### 11.2 Grid, run in this order

Move one axis at a time against a fixed baseline. Do not run the full cross.
Expected total: 16 to 20 specs.

**Baseline.** `basis=ns`, `lambda_mode=global`, `prior_source=none`,
`obs_weights=equal`, `transform=demean_vol`, `mode=hysteresis`, `weighting=duration`.

| # | axis            | variants                                           |
|---|-----------------|----------------------------------------------------|
| 1 | tenor set       | full, `tenor_min=2`, `tenor_max=20`                |
| 2 | obs weights     | equal, inv_duration, huber                         |
| 3 | prior           | off, isotropic 0.4, curvature 0.4, shape_only 0.4  |
| 4 | prior source    | prev_day, peer_mean (only if axis 3 wins)          |
| 5 | lambda          | global, per_curve, rolling, free (control)         |
| 6 | basis           | ns2, ns, bc, svensson_adj, svensson_free (control) |
| 7 | transform       | demean_vol, xs_country_z, diff, raw                |
| 8 | roll exclusion  | off, on                                            |
| 9 | par space       | off, on                                            |

Axes 5, 6 contain deliberate negative controls. If `free` lambda or free
Svensson wins, something is wrong upstream and the result should not be trusted.

### 11.3 Runner

```python
def run_spec(data: pd.DataFrame, cfg: SpecConfig, block: str) -> SpecResult
def run_grid(data, specs: list[SpecConfig], block: str) -> pd.DataFrame
```

Each run writes to `runs/{spec_id}/` containing `config.yaml`, `result.json`,
`pnl.parquet`, `positions.parquet`, `residuals.parquet`. Append a row to
`runs/index.csv`. Never overwrite an existing `spec_id` directory; raise instead.

---

## 12. `cli.py`

```
python -m nsrv tune     --data path.parquet --out lambda.json
python -m nsrv run      --data path.parquet --spec specs/baseline.yaml --block select
python -m nsrv grid     --data path.parquet --specs specs/ --block select
python -m nsrv report   --runs runs/ --out report.csv
python -m nsrv holdout  --runs runs/ --spec <spec_id> --unlock-holdout
```

`report` prints the ranked table and the deflated Sharpe computed with
`n_trials` set to the actual row count of `runs/index.csv`.

---

## 13. Tests

These are not optional. Write them before the grid runs.

### `test_ns_core.py`
- Loadings at `tau -> 0`: slope loading tends to 1, curvature to 0. Assert with
  `tau = 1e-9`.
- Curvature loading peaks at `tau = 1.7933 / lam` to within 1%.
- Exact recovery: generate `y = X @ beta_true` with no noise, fit OLS, assert
  `beta` matches to 1e-10 and residuals are zero.
- Leverage sums to `k`: `h.sum() == pytest.approx(n_factors)`.
- LOO equals brute force: refit excluding each point, compare predicted value at
  the excluded tenor to `yhat_i + resid_loo_i`. Must match to 1e-8.
- Prior with `prior_w -> inf` drives `beta -> prior_mean`.
- Prior with `prior_w = 0` equals OLS exactly.

### `test_structures.py`
- A duration-neutral fly has zero PnL under a parallel 1bp shift of all tenors,
  to within 1e-10.
- A yield-neutral fly does not, and the test asserts the difference, so the two
  conventions cannot be silently swapped.

### `test_signals.py`
- Hysteresis: a z path crossing entry then decaying below exit produces exactly
  one position episode of the right sign.
- The sign convention test: construct a synthetic residual that goes from +3 sd
  to 0 over 10 days, run the full pipeline, assert positive PnL.

### `test_no_lookahead.py`
The important one. Take the full panel, truncate it at date `T`, run the
pipeline, and assert that every output value up to `T - 1` is identical to the
corresponding value from the full-sample run. Run this for three different `T`.
Any lookahead anywhere in the chain fails this test.

---

## 14. Build order

1. `data.py` + loader tests. Confirm the panel shape and that validation fires.
2. `ns_core.py` + full test file. Do not proceed until LOO matches brute force.
3. `lambda_tune.py`, tune on the train block, persist the value, inspect the SSE
   curve for a clear interior minimum.
4. `fitting.py` with `prior_source="none"` only. Plot residual series for one
   curve and eyeball that they look like dislocations, not trends.
5. `structures.py` + duration-neutrality test.
6. `signals.py`, `pnl.py`, then `test_no_lookahead.py`. Fix anything it catches
   before computing a single Sharpe.
7. `evaluate.py`.
8. Only now add the prior sources and profiles, and run the grid.

Step 6 before step 8 is deliberate. A clean no-lookahead result on the OLS
baseline is the reference every shrinkage variant gets compared against.

---

## 15. First diagnostic to produce

Before any PnL. For one curve, output a table of:

- mean and sd of `resid_loo` by tenor, in bp
- mean leverage by tenor
- ratio `sd(resid_loo) / sd(resid)` by tenor
- lag-1 autocorrelation of `resid_loo` by tenor
- estimated half-life by tenor

If the LOO/in-sample sd ratio is near 1.0 you have more tenors than you think
and leverage is not binding. If it exceeds 1.5 at some tenor, the in-sample
residual at that tenor was badly understated and any prior work using it was
measuring the wrong thing. If half-lives are above roughly 60 days, the residual
is a persistent premium rather than a dislocation and the trade is a carry trade
with a different risk profile.
