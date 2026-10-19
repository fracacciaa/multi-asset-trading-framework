# Build spec: rates carry signals from generic par yield curves

Companion to `ns_residual_signal_spec.md`. Same repo, same package, same
evaluation harness. This document covers carry only.

Target: `src/nsrv/carry/`.

Reuse without reimplementing: `nsrv.ns_core` (loadings, `fit_ns`, `predict`),
`nsrv.structures.approx_dv01`, `nsrv.evaluate`, `nsrv.config` patterns,
`nsrv.data` loader and validators.

---

## 0. The idea in one paragraph

Carry is the return the position earns if the yield curve does not move. For a
bond it has two parts: income, the yield over the funding rate, and roll-down,
the capital gain from the bond aging into a lower point on an upward-sloping
curve. Ranking maturities within a curve by duration-adjusted carry and going
long high carry against short low carry is the curve carry strategy. It is a
risk premium, not a mispricing, which changes how it must be evaluated.

Reference: Martens, Beekhuizen, Duyvesteyn, Zomerdijk, "Carry Investing on the
Yield Curve," Financial Analysts Journal 75(4), 2019. Definitional framework:
Koijen, Moskowitz, Pedersen, Vrugt, "Carry," JFE 127(2), 2018, section 2.4.

---

## 1. Non-negotiable rules

Carry-specific additions to the rules in the residual spec. All of those still
apply, especially the no-lookahead test and the bp unit convention.

1. **Roll-down is computed off the fitted NS curve, never off raw yields.**
   Raw-yield roll-down contains the rich/cheap residual, which makes the carry
   signal mechanically correlated with the residual signal and double-counts the
   same dislocation. Computing it from `fit.predict` makes the two signals
   orthogonal by construction. The raw variant exists in the grid only to
   demonstrate the contamination.
2. **Carry must be duration-adjusted before cross-sectional ranking.** Raw carry
   rises with maturity because duration rises with maturity. Ranking on raw
   carry produces a leveraged duration bet wearing a curve-trade label.
3. **The book must be DV01-neutral.** Assert it: a 1bp parallel shift of every
   tenor produces zero PnL to within 1e-10.
4. **Backtest PnL must include the carry accrual, not only the price move.**
   Omitting accrual measures only the yield-change leg and the signal will look
   like a curve-shape bet with no premium. Report the decomposition into carry
   and yield-change components separately, as Martens et al do.
5. **Funding rate is an input, not a default.** No funding series, no
   cross-country carry. Within-curve carry nets most of it out and can run
   without one, but the shortest bucket still needs it (see 5.3).

---

## 2. Data

### 2.1 Already available

The curve panel from the residual spec: long frame with `date`, `curve_id`,
`tenor` (years, float), `yield` (bp after load).

### 2.2 Additional series required

One short rate per `curve_id`, daily, in bp, aligned to the same date index.

| curve type | preferred | acceptable fallback |
|---|---|---|
| DM | OIS 3m | 3m T-bill / generic 3m govt |
| EM | onshore 3m depo or 3m bill | policy rate |

Loader:

```python
# carry/data.py
def load_funding(path: str, *, units: str = "pct") -> pd.DataFrame:
    """columns: date, curve_id, rf. Returns rf in bp."""
```

Validation that must raise:
- a `curve_id` present in the curve panel with no funding series, when
  `cfg.require_funding` is True
- any `|rf|` > 10000 bp
- gaps longer than 5 business days (forward fill shorter gaps, record the mask)

Record a `funding_source` string per curve in the run manifest. The choice
materially changes EM results and must be visible in the output, not buried in a
config file.

### 2.3 Not required

FX forwards. This spec builds **currency-hedged** carry only. Unhedged
cross-country carry is a different signal with an FX carry leg in it, and should
not be mixed in here.

---

## 3. Module layout

```
src/nsrv/carry/
    __init__.py
    config.py        # CarryConfig
    data.py          # funding loader
    measures.py      # the carry definitions
    returns.py       # constant-maturity total return decomposition
    signal.py        # carry -> duration-adjusted z -> DV01 positions
    cross_country.py # country-level carry
    experiment.py    # grid runner, reuses nsrv.evaluate
tests/carry/
    test_measures.py
    test_returns.py
    test_signal.py
    test_orthogonality.py
```

---

## 4. `measures.py`: the carry definitions

All functions take a fitted `NSFit` for the date, a tenor array, the funding
rate in bp, and a horizon `h` in years. All return carry in **bp of total
return over the horizon h**.

### 4.1 Yield pickup (control)

```python
def carry_pickup(fit, tau, rf_bp, h):
    """Income only. No roll-down. The simplest measure, and the control."""
    return (fit.predict(tau) - rf_bp) * h
```

Keep this. The Indonesian evidence (JRFM 19(7), 2026) finds the plain term-spread
measure more robust and more persistent than the composite measure in at least
one EM market, so this is not a strawman.

### 4.2 Carry plus roll-down, duration approximation

```python
def carry_rolldown_dur(fit, tau, rf_bp, h, *, roll_floor):
    tau = np.asarray(tau, float)
    y_now = fit.predict(tau)
    tau_f = tau - h
    y_fwd = np.where(tau_f > roll_floor, fit.predict(np.maximum(tau_f, 1e-6)), np.nan)
    # shortest bucket has nothing below it: roll to the funding rate
    y_fwd = np.where(np.isnan(y_fwd), rf_bp, y_fwd)
    dur = np.array([approx_dv01(max(t, 1e-6), y / 1e4) for t, y in zip(tau_f, y_fwd)]) * 1e4
    income = (y_now - rf_bp) * h
    rolldn = dur * (y_now - y_fwd)
    return income + rolldn
```

`roll_floor` is the shortest tenor on the grid. Below it there is no lower
bucket to roll into, so the position rolls to the funding rate. This is the
Martens et al convention and it must be flagged in attribution, because the
shortest bucket's carry then depends entirely on the funding choice.

Units check: `dur` in years times a yield difference in bp gives bp of price
return. `(y_now - rf_bp)` in bp times `h` in years gives bp of income. Both are
bp of return over the horizon. Assert this in a test.

### 4.3 Carry by full repricing (preferred)

The duration approximation misstates roll-down at the long end over a quarterly
horizon by enough to change the ranking. Full repricing handles convexity
automatically.

```python
def carry_reprice(fit, tau, rf_bp, h, *, freq=2):
    """Total return in bp over h, assuming the zero curve (as a function of
    time-to-maturity) is unchanged. Bond is par at t, so coupon = par yield."""
    out = np.empty(len(tau))
    for i, n in enumerate(tau):
        c = fit.predict(np.array([n]))[0] / 1e4          # coupon = today's par yield
        # cash flow times remaining at t+h, measured from t+h
        k = np.arange(1, int(round(n * freq)) + 1) / freq
        k_rem = k[k > h] - h
        if len(k_rem) == 0:
            out[i] = (c * 1e4 - rf_bp) * h
            continue
        z = fit.predict(k_rem) / 1e4
        df = np.exp(-z * k_rem)
        p_fwd = (c / freq) * df.sum() + df[-1]            # per unit par
        coupons_paid = (c / freq) * np.sum(k <= h)
        out[i] = ((p_fwd - 1.0) + coupons_paid) * 1e4 - rf_bp * h
    return out
```

Notes on correctness:
- The bond is at par today by construction of a par yield, so price today is 1.0
  and the coupon equals `y(τ)`.
- "Curve unchanged" means the zero curve as a function of time-to-maturity is
  the same at `t+h`, so remaining cash flows discount at `z(k - h)`.
- Coupons paid during the horizon are added undiscounted. Add a reinvestment
  option at `rf` behind a config flag if you want it; the effect is small at
  `h <= 1`.
- `fit.predict` returns the NS **zero** curve. If `cfg.par_space` was used in the
  fit, the betas describe a zero curve matched to observed par yields, and this
  function is still correct. If the linear fit was used, the betas describe a
  curve fitted directly to par yields treated as zeros, and this function is
  internally consistent but slightly misspecified. Record which in the manifest.

### 4.4 Forward-based carry

Equivalent to 4.3 up to the coupon treatment, stated in yield space, which some
people find easier to sanity-check against a screen.

```python
def carry_forward(fit, tau, rf_bp, h):
    """-D * (forward par yield - spot par yield) + income."""
```

Implement by solving for the `h`-forward par yield off the fitted discount curve
and applying the duration. Include it as a grid variant and verify it matches
4.3 to within 1bp at tenors under 10y.

### 4.5 Registry

```python
CARRY_MEASURES = {
    "pickup":   carry_pickup,
    "roll_dur": carry_rolldown_dur,
    "reprice":  carry_reprice,
    "forward":  carry_forward,
}
```

### 4.6 Raw-yield variant (contamination demo only)

```python
def carry_rolldown_raw(wide_row, tau, rf_bp, h, *, interp="linear"):
    """Same as 4.2 but y_fwd interpolated from RAW observed yields."""
```

This exists to produce one number for the report: the correlation between the
carry signal built this way and the NS residual signal. Expect it to be
materially above the correlation using the fitted curve. Do not ship it.

---

## 5. `returns.py`: constant-maturity total return

Your data is constant-maturity generic yields, not a tracked bond. The standard
daily total return approximation for a constant-maturity position at tenor τ,
all terms in bp of return:

```python
def cm_total_return_bp(y_prev, y_now, yhat_prev, tau, rf_prev, dur_prev, dt=1/252):
    income  = y_prev * dt
    rolldn  = dur_prev * (yhat_prev_at_tau - yhat_prev_at_tau_minus_dt)
    pricech = -dur_prev * (y_now - y_prev)
    funding = -rf_prev * dt
    return income + rolldn + pricech + funding
```

Rules:
- Every input is from `t-1`. The only `t` quantity is `y_now`.
- `dur_prev` is computed from `t-1` yields. Using today's duration leaks.
- The roll-down leg uses the **fitted** curve at `t-1` for both points, for the
  orthogonality reason in rule 1.
- `income + rolldn + funding` is the carry component. `pricech` is the
  yield-change component. Return them as separate series so attribution is
  possible.

Attribution output per spec, matching the Martens et al table format:

| column | meaning |
|---|---|
| `ret_total` | full daily return of the book |
| `ret_carry` | income + rolldown + funding |
| `ret_yldchg` | price change from curve moves |

`ret_carry` should be positive and smooth, `ret_yldchg` should be the volatile
part. If `ret_carry` is negative on average for a long-high-carry book,
something is inverted. Martens et al report curve carry at roughly +0.46% per
annum on the carry component alone.

---

## 6. `signal.py`: carry to positions

### 6.1 Duration adjustment

```python
def carry_per_duration(carry_bp, dur):
    """bp of return per unit duration = carry expressed in yield terms."""
    return carry_bp / np.maximum(dur, 1e-6)
```

Variants for the grid:

| name | transform |
|---|---|
| `none` | raw carry (control, expect a duration bet) |
| `per_dur` | `carry / D` |
| `per_dur_vol` | `carry / (D * trailing_sd_of_dy)` , risk-adjusted carry |

`per_dur_vol` is worth testing. It ranks by carry per unit of realized rate risk
rather than per unit of duration, which matters when the front end is in a
high-vol policy regime.

### 6.2 Cross-sectional score

Within one curve, across tenors, on each date:

```python
def xs_score(c_adj: pd.Series, mode="z"):
    if mode == "z":
        return (c_adj - c_adj.mean()) / c_adj.std(ddof=0)
    if mode == "rank":
        r = c_adj.rank()
        return (r - r.mean()) / r.std(ddof=0)
    if mode == "demean":
        return c_adj - c_adj.mean()
```

Demeaning across tenors is what converts a level signal into a curve signal. Do
not skip it and then try to neutralize afterwards.

`rank` is worth testing because carry is a smooth function of tenor and a single
dislocated point can dominate a z-score on an 8-point cross-section.

### 6.3 Positions

Positions are in DV01 units (currency per bp), DV01-neutral.

```python
def dv01_positions(score: pd.Series, dur: pd.Series, gross=1.0):
    w = score.to_numpy(float)
    d = dur.to_numpy(float)
    dv = w / np.maximum(d, 1e-6)          # convert score weight to DV01 weight
    dv = dv - dv.mean() * 0               # placeholder, see below
    # enforce DV01 neutrality exactly
    dv = dv - np.dot(dv, d) / np.dot(d, d) * d
    dv = dv / np.abs(dv).sum() * gross
    return pd.Series(dv, index=score.index)
```

The projection step removes the component of the position along the duration
vector, which is the exact DV01 neutralization. Assert it with the parallel
shift test.

Optional second neutralization, behind a flag: also project out the slope
loading, so the book is neutral to level and slope and expresses curvature only.
Worth testing, since within-curve carry is close to a slope bet by construction.

### 6.4 Rebalance

Carry is extremely persistent. The ranking barely changes day to day.

```python
cfg.rebalance in {"daily", "weekly", "monthly"}
```

Implementation: compute the score daily, but only update the held position on
rebalance dates. Hold otherwise. Monthly is the Martens et al convention and
should be the baseline. Report turnover for each.

Then, always:

```python
pos = pos.shift(cfg.lag)     # lag >= 1, required field, no default
```

---

## 7. `cross_country.py`

Separate book. Martens et al report the correlation between curve carry and
country carry at roughly 11%, so these should be evaluated and sized
independently and combined afterwards, not merged into one cross-section.

### 7.1 Country carry measure

Per country, compute carry at a fixed duration point, typically the 10y or a
duration-matched 5y to 10y blend:

```python
c_country = carry_measure(fit_c, tau=target, rf_bp=rf_c, h=h) / dur_target
```

Hedged, so the funding rate is the local one and no FX term enters.

### 7.2 Cross-sectional step

```python
z = (c_country - c_country.mean()) / c_country.std()   # across countries, same date
pos = z / trailing_sd_of_10y_change                     # inverse-vol sizing
pos = pos - pos.mean()                                  # zero net
```

Do **not** run DM and EM in one cross-section by default. The carry dispersion
across EM is several multiples of DM, so a pooled z-score is an EM-only
strategy with DM noise attached. Grid axis: `DM only`, `EM only`,
`bloc-demeaned` (z computed within bloc, then stacked).

### 7.3 EM caveats to encode

- `funding_source` must be recorded and at least two must be tested. The
  published betting-against-beta comparison in Martens et al explicitly depends
  on the assumed funding rate, and the DM-EM gap between policy rate, onshore
  depo and implementable repo is wide.
- Capital controls and non-deliverable markets mean the local funding rate may
  not be achievable. Add a `tradeable: bool` column per country-date and allow
  the config to filter on it.
- EM panels have survivorship and entry-date problems. Assert that a country
  entering the panel mid-sample does so on the same date in every spec.

---

## 8. Costs

```python
def apply_costs(pnl_bp, dv01_pos, *, bid_ask_bp: dict[float, float]):
    cost = sum over tenors of 0.5 * bid_ask_bp[tenor] * |dv01_pos[tenor].diff()|
    return pnl_bp - cost
```

Defaults, placeholders to be replaced with your own marks: 0.25bp for tenors
under 10y DM, 0.5bp for 10y to 30y DM, 1.0 to 3.0bp for EM depending on country.
With monthly rebalance the cost drag should be small; if it is not, the signal
is turning over more than a carry signal should and something upstream is noisy.

---

## 9. Evaluation

Reuse `nsrv.evaluate.SpecResult` and add these fields in a `CarrySpecResult`
subclass. Carry is a risk premium, so the diagnostics that matter are different
from the residual signal's.

| field | definition | expectation |
|---|---|---|
| `carry_component_ann` | annualized `ret_carry` | positive, smooth |
| `yldchg_component_ann` | annualized `ret_yldchg` | near zero or negative |
| `beta_to_duration` | regression of PnL on change in the 10y | non-zero is fine here |
| `beta_to_risk` | regression on a risk proxy (VIX change, MOVE, EM spread) | expect negative |
| `worst_month` | | |
| `dd_2013`, `dd_2022` | drawdown in the taper tantrum and the 2022 bear flattening | **must be negative** |
| `corr_to_residual_book` | correlation with the NS residual strategy PnL | should be < 0.3 |
| `turnover` | mean abs DV01 change per day | |
| `pct_pnl_from_shortest` | share of PnL from the roll-to-funding bucket | flag if > 40% |

### 9.1 The sanity checks that matter here

**A carry signal with no drawdown in 2013 or 2022 is a bug.** Curve carry is
long the premium for bearing duration and liquidity risk. It should lose in
sharp bear flattenings and in EM stress. Assert that `dd_2013 < 0` and
`dd_2022 < 0` and fail the spec loudly if not, because the usual cause is a
lookahead or a survivorship hole in the EM panel.

**`corr_to_residual_book`.** If this exceeds roughly 0.3 you are on the
raw-yield roll-down path by mistake and the carry signal is partly re-trading
the NS residual. Check that `cfg.rolldown_source == "fitted"`.

**`pct_pnl_from_shortest`.** If most of the PnL comes from the bucket that rolls
to the funding rate, the strategy is a funding-rate arbitrage artifact, not curve
carry. Rerun with a different `funding_source` and see if it survives.

### 9.2 Selection rule, pre-committed

1. Discard any spec failing the 2013 or 2022 drawdown assertion.
2. Discard any spec with `corr_to_residual_book > 0.3`.
3. Discard any spec with `pct_pnl_from_shortest > 0.4`.
4. Rank survivors by `n_curves_positive / n_curves`, then by
   `carry_component_ann` (the premium you actually intend to harvest), then by
   deflated Sharpe.
5. Among specs within 0.2 Sharpe of the top, take the fewest free parameters.

Deflated Sharpe uses `n_trials` from the combined `runs/index.csv` across both
the residual and carry grids, not the carry grid alone. They are selections from
the same research effort.

---

## 10. `config.py`

```python
@dataclass(frozen=True)
class CarryConfig:
    measure: str = "reprice"            # pickup | roll_dur | reprice | forward
    horizon_years: float = 0.25
    rolldown_source: str = "fitted"     # fitted | raw  (raw = demo only)
    duration_adj: str = "per_dur"       # none | per_dur | per_dur_vol
    xs_mode: str = "z"                  # z | rank | demean
    neutralize: tuple[str, ...] = ("level",)   # level | level,slope
    rebalance: str = "monthly"          # daily | weekly | monthly
    lag: int = 1                        # required, >= 1
    funding_source: str = "ois3m"
    require_funding: bool = True
    coupon_reinvest: bool = False
    tenor_min: float | None = None
    tenor_max: float | None = None
    universe: str = "DM"                # DM | EM | both | bloc_demeaned
```

The fit side is inherited from `FitConfig` unchanged. Carry runs on top of
whichever NS fit won the residual grid, and that `spec_id` must be recorded as
`parent_fit_spec` in the manifest.

---

## 11. Tests

### `test_measures.py`
- **Flat curve, y = rf.** All carry measures return 0 to 1e-10.
- **Flat curve, y > rf.** Carry equals `(y - rf) * h` exactly, roll-down is zero.
- **Upward-sloping curve.** Roll-down is strictly positive at every tenor above
  `roll_floor`.
- **Inverted curve.** Roll-down is strictly negative. Both signs must be
  exercised; an inverted-curve bug is invisible in a bull-steepening sample.
- **Units.** Construct a curve with a known 100bp slope over 1y and a duration of
  exactly 5, assert roll-down equals 500bp of return over a 1y horizon.
- **Reprice vs duration approximation.** Agree to within 1bp at τ <= 10y and
  h = 0.25. Assert they **diverge** by more than 2bp at τ = 30y, h = 1.0, so the
  convexity term cannot be silently dropped.
- **Shortest bucket.** With `tau[0] - h < roll_floor`, the roll target is `rf`,
  verified by changing `rf` and checking only that bucket's carry moves.

### `test_signal.py`
- **DV01 neutrality.** After `dv01_positions`, `np.dot(dv, dur) == approx(0)`.
- **Parallel shift.** Shift every tenor by 1bp, book PnL is 0 to 1e-10.
- **Level+slope neutralization.** With `neutralize=("level","slope")`, the
  position is orthogonal to both the NS level and slope loading vectors.
- **Sign.** A synthetic curve where the 5y has the highest duration-adjusted
  carry produces a long position at 5y.
- **Rebalance.** With `rebalance="monthly"`, the position series changes on at
  most 13 dates per year.

### `test_returns.py`
- **Decomposition adds up.** `ret_total == ret_carry + ret_yldchg` exactly.
- **Static curve.** If the curve does not move for 60 days, `ret_yldchg` is zero
  and the realized return equals the ex-ante carry to within 0.5bp.

### `test_orthogonality.py`
Build both signals on the same panel. Assert
`corr(carry_pnl_fitted, residual_pnl) < corr(carry_pnl_raw, residual_pnl)`.
This is the test that locks in rule 1.

### `test_no_lookahead.py`
Extend the existing one to cover the carry path. Truncate at `T`, rerun, assert
identity up to `T-1`. Three values of `T`.

---

## 12. Build order

1. `carry/data.py` and the funding loader. Plot `y(10y) - rf` per country and
   confirm it looks like a term spread, not garbage.
2. `measures.py` with `carry_pickup` and `carry_rolldown_dur` only, plus the full
   `test_measures.py`. Do not proceed until the inverted-curve sign test passes.
3. `returns.py` and `test_returns.py`. The decomposition identity must hold
   before any PnL is computed.
4. `signal.py` with `duration_adj="per_dur"`, `xs_mode="z"`,
   `rebalance="monthly"`. Run the DV01-neutrality and parallel-shift tests.
5. First backtest: within-curve, one DM country, `measure="pickup"`. Look at the
   attribution table before looking at the Sharpe.
6. `carry_reprice` and the convexity divergence test.
7. `test_orthogonality.py`.
8. Cross-country, DM only.
9. EM, with two funding sources.

Step 5 before step 6 is deliberate. The simplest measure on one country with
honest attribution is the reference everything else is compared against.

---

## 13. Grid

Baseline: `measure=reprice`, `h=0.25`, `rolldown_source=fitted`,
`duration_adj=per_dur`, `xs_mode=z`, `neutralize=("level",)`,
`rebalance=monthly`, `universe=DM`, within-curve.

One axis at a time. Roughly 20 specs.

| # | axis | variants |
|---|---|---|
| 1 | measure | pickup, roll_dur, reprice, forward |
| 2 | horizon | 1m, 3m, 6m, 12m |
| 3 | duration adj | none (control), per_dur, per_dur_vol |
| 4 | xs mode | z, rank, demean |
| 5 | neutralize | level, level+slope |
| 6 | rebalance | daily, weekly, monthly |
| 7 | funding | ois3m, bill3m, policy |
| 8 | universe | DM, EM, both, bloc_demeaned |
| 9 | rolldown source | fitted, raw (contamination demo, not shippable) |

Axes 3 and 9 contain deliberate controls. `duration_adj="none"` should show a
high `beta_to_duration` and should be rejected by the selection rule even if its
Sharpe is the highest in the table.

---

## 14. First diagnostic, before any PnL

For one DM country, one date near the middle of the sample, print:

| tenor | y | yhat | rf | income_bp | rolldown_bp | carry_bp | dur | carry_per_dur |
|---|---|---|---|---|---|---|---|---|

Then the time series version: for each tenor, the mean and sd of `carry_per_dur`
in bp, and its rank autocorrelation at 21 business days.

What to look for:

- **Roll-down should be the larger term at the short end and the smaller term at
  the long end** on an upward-sloping curve. If it is monotone in one direction
  across the whole curve, check the duration leg.
- **`carry_per_dur` should be hump-shaped in tenor**, peaking somewhere in the
  belly where the curve is steepest per unit duration. A monotone profile means
  the duration adjustment is not binding and the signal will be a duration bet.
- **Rank autocorrelation at 21d should be above 0.8.** Carry is persistent. If it
  is not, the carry estimate is dominated by noise, most likely because
  roll-down is being computed off raw yields.
- **Compare the carry profile to the NS residual profile at the same tenors.**
  They should look unrelated. If the tenors with the highest carry are the same
  tenors with the largest residual, rule 1 has been violated somewhere.
