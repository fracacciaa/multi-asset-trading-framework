ok i habe uploaded an md file that explain how to build 3 raw signals. # Build spec: three rates momentum signals on an OIS curve panel

Raw signal construction only. No backtest harness, no PnL, no spec grid. The
output of this document is three signal panels indexed by (date, curve_id,
tenor), plus the return panel they are all built from.

Companion to `ns_residual_signal_spec.md` and `carry_signal_spec.md`. Reuses
their data loader conventions and the bp unit rule. Target: `src/nsrv/momentum/`.

Papers implemented:

| Signal | Paper | What it is |
|---|---|---|
| A | Moskowitz, Ooi & Pedersen (2012), *JFE* 104(2) 228-250, "Time Series Momentum" | Per-instrument time series momentum, vol-scaled |
| B | Sihvonen (2022), Bank of Finland, "Yield Curve Momentum" | Momentum along one curve, no vol scaling, 1-month lookback |
| C | Durham (2013), FRBNY Staff Report 657, "Momentum and the Term Structure of Interest Rates" | Cross-sectional across duration buckets, duration-neutral, linear program |

---

## 0. The central problem: all three papers use bond returns, you have par swap rates

None of these three signals is computed on yields. Every one is computed on a
**holding-period return** series. MOP use futures excess returns, Sihvonen uses
log excess returns on zero-coupon Treasuries, Durham uses Barclays total return
sub-indices.

So Section 1 is not optional preamble. It is the part that decides whether the
three signals mean anything. Build it first and test it before touching Sections
2 to 4.

Two more translation facts that must be encoded:

1. **A swap is unfunded.** For a bond, excess return = total return minus the
   funding rate. For a receiver swap, the floating leg *is* the funding leg, so
   the return of the swap position is already an excess return. Do not subtract
   a funding rate a second time. This is the single easiest way to corrupt all
   three signals at once.
2. **Sign convention.** Define a long position as **receiving fixed**, which
   gains when rates fall. A long receiver has positive duration, exactly like a
   long bond. With this convention every formula below reads the same as it does
   in the papers. Write this into a module docstring and assert it with the test
   in Section 5.1.

---

## 1. `returns.py`: the return panel

### 1.1 Input

The OIS workbook: one sheet per currency, 27 tenors (1M, 2M, ..., 12M, 1Y, 2Y,
..., 10Y, 12Y, 15Y, 20Y, 25Y, 30Y), daily, par OIS rates in percent.

Load to a long frame with columns `date`, `curve_id`, `tenor` (years, float),
`par_rate` (bp after load). Apply the same validators as `nsrv.data.load_curves`.

Encode per-currency daycount, because it enters the discount factors:

```python
DAYCOUNT = {
    "USDSOFR": 360, "USDEFFR": 360, "EUR": 360, "CHF": 360, "SEK": 360,
    "NOK": 360, "TRY": 360, "COP": 360,
    "GBP": 365, "JPY": 365, "CAD": 365, "AUD": 365, "NZD": 365,
    "PLN": 365, "ILS": 365, "INR": 365, "CNY": 365, "PHP": 365,
}
```

Verify each one against Bloomberg DES for the ticker before trusting it. A wrong
daycount shifts the level of every return by a small constant, which matters for
Signal A (it conditions on the sign) far more than it looks.

### 1.2 Bootstrap par OIS to a zero curve

Required exactly for Signal B. Signals A and C can use the duration
approximation in 1.4, but build this anyway: it is the reference the
approximation is checked against.

Conventions: OIS pays a single payment at maturity for tenors under 1Y, and
annually thereafter.

```python
def bootstrap_ois(tenors, par_bp, basis=360):
    """tenors: years, sorted ascending. par_bp: par OIS rates in bp.
    Returns dict {tenor: discount_factor}. Annual fixed leg at and beyond 1Y."""
    s = np.asarray(par_bp, float) / 1e4
    df = {}
    # money-market stub: single payment, simple compounding
    for t, r in zip(tenors, s):
        if t < 1.0:
            df[t] = 1.0 / (1.0 + r * t * 365.0 / basis)
    # annual coupon leg from 1Y out
    annual = [t for t in tenors if t >= 1.0]
    for t in annual:
        pay = np.arange(1.0, t + 1e-9, 1.0)          # annual payment dates
        acc = sum(df_interp(df, u) for u in pay[:-1])  # DFs already solved
        r = s[list(tenors).index(t)]
        df[t] = (1.0 - r * acc) / (1.0 + r * 1.0)
    return df
```

**The gap problem.** Your annual grid runs 1Y to 10Y, then jumps to 12, 15, 20,
25, 30. Years 11, 13, 14, 16-19, 21-24 and 26-29 have no quote, but the annual
coupon leg of the 12Y, 15Y, 20Y, 25Y and 30Y swaps pays on those dates. You must
interpolate before you can bootstrap.

Use **log-linear interpolation on discount factors**, equivalent to piecewise
constant forwards. Solve the long tenors sequentially, each time interpolating
the intermediate DFs from the solved points, and iterate twice to convergence.

```python
def df_interp(df, u):
    """Log-linear in DF = piecewise-constant forward."""
    ks = sorted(df)
    if u <= ks[0]:  return df[ks[0]] ** (u / ks[0])
    if u >= ks[-1]: return df[ks[-1]] ** (u / ks[-1])
    lo = max(k for k in ks if k <= u); hi = min(k for k in ks if k >= u)
    if lo == hi: return df[lo]
    w = (u - lo) / (hi - lo)
    return np.exp((1 - w) * np.log(df[lo]) + w * np.log(df[hi]))
```

Record the interpolation choice in the manifest. Do not use the NS fit from the
residual spec here. NS smooths away exactly the small curve moves that momentum
is reading, which would make the signal a function of the fitting method.

Continuously compounded zero yield, as an **annual** rate:

```python
z(tau) = -log(df[tau]) / tau
```

### 1.3 Monthly log excess returns (Sihvonen convention, exact)

Sihvonen, equation (1):

```
rx^n_{t+1} = -(n-1) y^{n-1}_{t+1} + n y^n_t - y^1_t
```

Units matter and are the most common source of a wrong answer here. In that
equation `n` is in **months** and `y^n_t` is the **monthly** continuously
compounded yield of an n-month zero. Convert from your annual zero curve:

```python
y_monthly(n_months) = z(n_months / 12.0) / 12.0
```

Then, with month-end sampling:

```python
def log_excess_return_bp(z_t, z_tp1, n_months):
    """One-month log excess return of an n-month zero, in bp.
    z_t, z_tp1: callables tau_years -> annual cc zero yield (decimal)."""
    n = n_months
    y_n_t    = z_t(n / 12.0) / 12.0
    y_nm1_t1 = z_tp1((n - 1) / 12.0) / 12.0
    y_1_t    = z_t(1 / 12.0) / 12.0
    return (-(n - 1) * y_nm1_t1 + n * y_n_t - y_1_t) * 1e4
```

`(n-1)/12` is rarely an observed tenor, so this call goes through `df_interp`.
That is fine and is the same interpolation used in the bootstrap.

**Carry / yield-change split**, Sihvonen equation (5). Needed for the Signal B
variants in 3.3:

```
rx^n_{t+1} = c^n_t - yc^n_{t+1}

c^n_t      = -(n-1) y^{n-1}_t + n y^n_t - y^1_t        # observable at t
yc^n_{t+1} = (n-1) (y^{n-1}_{t+1} - y^{n-1}_t)          # the yield-change part
```

Return `c` and `yc` as separate panels. Assert `rx == c - yc` to 1e-10.

### 1.4 Constant-maturity approximation (Signals A and C)

Cheaper, and adequate where the signal only reads the sign or a cross-sectional
ranking. In bp of return over one period `dt` (years):

```python
def cm_return_bp(s_now, s_aged, s_next, dur, rf_bp, dt):
    """Receiver swap at constant maturity tau.
    s_now   : par rate at tau, time t
    s_aged  : par rate at tau - dt, time t     (the roll target)
    s_next  : par rate at tau, time t+1
    dur     : annuity-weighted duration of the tau swap at t, in years
    rf_bp   : overnight / 1M OIS rate at t, bp
    """
    carry  = (s_now - rf_bp) * dt
    roll   = dur * (s_now - s_aged)
    price  = -dur * (s_next - s_now)
    return carry + roll + price
```

Note there is no separate funding term: `rf_bp` enters once, inside `carry`,
because the swap's floating leg is the funding.

Duration of a par swap is the fixed-leg annuity divided by the notional:

```python
def swap_duration(tenors, dfs, tau):
    pay = np.arange(1.0, tau + 1e-9, 1.0)
    return float(sum(u * df_interp(dfs, u) for u in pay)) if tau >= 1 else tau
```

Use `dur` and `s_aged` from **t**, never t+1. Both are known at t.

### 1.5 Sampling

All three papers are monthly. Build both panels:

- `ret_m`: month-end to month-end, the primary. Resample the daily curve to the
  last business day of each month, then compute returns on that series.
- `ret_d`: daily, needed only for the MOP ex ante volatility in 2.1.

Never mix them inside one signal.

---

## 2. Signal A: MOP time series momentum

### 2.1 Ex ante volatility

MOP equation (1), computed on **daily** excess returns, annualized:

```
sigma_t^2 = 261 * sum_{i>=0} (1-delta) delta^i (r_{t-1-i} - rbar_t)^2
```

`delta` is set so the centre of mass of the weights is 60 days:
`delta / (1 - delta) = 60`, so `delta = 60/61 = 0.983607`. `rbar_t` is the
exponentially weighted mean of the same returns with the same `delta`.

```python
DELTA = 60.0 / 61.0

def ex_ante_vol(r_daily_bp):
    """Annualized ex ante vol in bp. Uses data through t-1 only."""
    r = r_daily_bp.shift(1)                       # MOP: sigma at t-1
    mu = r.ewm(alpha=1 - DELTA, adjust=False).mean()
    var = ((r - mu) ** 2).ewm(alpha=1 - DELTA, adjust=False).mean()
    return np.sqrt(261.0 * var)
```

The `.shift(1)` is MOP's own no-lookahead rule: "we use the volatility estimate
at time t-1 applied to time-t returns throughout the analysis."

Resample `sigma` to month-end to align with the monthly signal.

### 2.2 The signal

MOP equation (5):

```
r^{TSMOM,s}_{t,t+1} = sign( r^s_{t-12,t} ) * (40% / sigma^s_t) * r^s_{t,t+1}
```

So the raw signal, before any portfolio step, is a **position size**:

```python
TARGET_VOL_BP = 0.40 * 1e4        # 40% annualized, in bp

def tsmom_position(ret_m_bp, sigma_ann_bp, k=12):
    """Position in units of notional. ret_m_bp monthly, sigma monthly-sampled."""
    past = ret_m_bp.rolling(k).sum()              # cumulative k-month return
    return np.sign(past) * (TARGET_VOL_BP / sigma_ann_bp)
```

Parameters as specified in the paper:

| parameter | MOP value | note |
|---|---|---|
| lookback `k` | 12 months | their headline; grid is {1,3,6,9,12,24,36,48} |
| holding `h` | 1 month | headline; grid is {1,3,6,9,12,24,36,48} |
| skip | none | MOP do not skip the most recent month |
| vol target | 40% annualized per instrument | gives ~12% on the diversified portfolio |
| vol estimator | EWMA, com 60 days, 261 annualization | equation (1) |
| signal shape | `sign()`, not the magnitude | equation (5) |

For `h > 1`, MOP use overlapping portfolios in the Jegadeesh-Titman style:
at each date hold `h` separate vintages opened in each of the last `h` months
and average their returns. Equivalent and simpler to implement as an `h`-period
moving average of the `h=1` position series:

```python
pos_h = tsmom_position(ret_m_bp, sigma, k).rolling(h).mean()
```

### 2.3 What is an "instrument" here

MOP run one signal per futures contract. The analogue is one constant-maturity
swap per (currency, tenor).

Primary set, mirroring their 13 bond futures: tenors {2, 5, 10, 30} years across
the 8 usable DM curves (USD, GBP, EUR, JPY, CAD, AUD, NZD, SEK). 32 instruments.

Variant: all 27 tenors. Note that tenors on one curve are 0.95+ correlated, so
the "diversified" average across all of them is close to a single level bet.
That is a property of the data, not a bug, but it must be visible, so output the
per-instrument signal panel and let the portfolio step happen downstream.

---

## 3. Signal B: Sihvonen yield curve momentum

The two things that make this different from Signal A, both stated explicitly in
the paper, and both easy to lose:

1. **Lookback is 1 month, not 12.** Sihvonen tests h in {1, 3, 6, 12} and finds
   only h=1 statistically significant (Table 1: beta at 1-month lookback is 0.19
   at the 1y with t=3.01, decaying to 0.09 at the 10y with t=1.99; at 3 and 6
   month lookbacks the betas collapse to 0.01-0.03 and go insignificant).
2. **No volatility scaling.** Quoting: "I also ignore the volatility scaling
   applied by Moskowitz et al. (2012) as it can induce return predictability
   unrelated to raw momentum." Do not apply Section 2.1 here, even though it
   is sitting in the same package.

Maturities: Sihvonen uses `N = {12, 24, 36, ..., 120}` months, i.e. 1y to 10y in
annual steps, plus the 1-month bill as `y^1`. Your OIS grid supplies all of them.

### 3.1 B1, own-maturity momentum

Regression form, Sihvonen equation (2):

```
rx^n_{t+1} = alpha + beta * rx^n_{t-h,t} + eps
```

Raw signal:

```python
def sihvonen_own(rx_m_bp, h=1):
    """rx_m_bp: month-end log excess returns by tenor, in bp."""
    return rx_m_bp.rolling(h).sum()
```

Output both forms and let the downstream step pick:

- `signal_linear = rx^n_{t-h,t}`, which is the regression's own predictor
- `signal_sign   = sign(rx^n_{t-h,t})`, which matches the Table 2 subsample cut

Expect the term structure of momentum to slope **downward**: the predictive beta
should fall with maturity. Sihvonen's Figure 1 shows 0.19 at 1y declining
monotonically to 0.09 at 10y. This is a diagnostic, see Section 5.2.

### 3.2 B2, factor momentum

Sihvonen equation (3), the single momentum factor:

```
rxbar_t = (1/10) * sum_{n in N} rx^n_t ,   N = {12, 24, ..., 120} months
```

and equation (4), `rx^n_{t+1} = alpha + beta * rxbar_t + eps`.

```python
def sihvonen_factor(rx_m_bp, maturities=(1,2,3,4,5,6,7,8,9,10)):
    return rx_m_bp[list(maturities)].mean(axis=1)
```

The factor is one series per curve. It predicts every maturity, **with
maturity-specific loadings**, so the signal panel is the factor broadcast across
tenors, and the loading is estimated per tenor downstream. Do not pre-multiply
by anything here.

Sihvonen's Table 3 betas rise with maturity (0.04 at 1y to 0.18 at 10y) while
the own-maturity betas fall. Both patterns should appear in your data; they are
not contradictory, since the factor has a common scale while `rx^n` does not.

### 3.3 B3, yield-change momentum

The economically interesting variant, and the paper's own conclusion about what
drives the result. Table 4 attributes 85 to 93 percent of the covariance between
current and past bond returns to `Cov(yc_{t+1}, yc_t)`, the yield-change channel,
with the carry channel contributing only 3 to 7 percent.

So predict the yield-change component from its own lag, using the split from
Section 1.3:

```python
def sihvonen_yieldchange(yc_m_bp, h=1):
    """Signal for yc^n_{t+1}. Sign flipped so that a positive signal is a
    positive expected RETURN: rx = c - yc, so rising yc lowers return."""
    return -yc_m_bp.rolling(h).sum()
```

Mind the sign. `rx = c - yc`, so a positive past `yc` forecasts a positive future
`yc`, which forecasts a *negative* return contribution. Get this backwards and
the signal inverts. The test in 5.1 catches it.

### 3.4 B4, PC1 momentum

Sihvonen equation (11): `rx^n_{t+1} = alpha + beta (pc^1_t - pc^1_{t-1}) + eps`.
He extracts the first three PCs from all 120 monthly maturities between 1 month
and 10 years, finds PC1 explains ~98.5% of yield variation, and reports that
only the PC1 change is clearly significant (Table 6: t-stats -2.1 to -2.9 across
maturities; PC2 and PC3 insignificant).

```python
def pc1_change(zero_panel_bp, n_pc=3):
    """zero_panel_bp: date x maturity (monthly grid 1..120), zero yields in bp.
    PCA on LEVELS, then take the first difference of the score."""
    X = zero_panel_bp.dropna()
    Xc = X - X.mean()
    _, _, Vt = np.linalg.svd(Xc.values, full_matrices=False)
    scores = Xc.values @ Vt[:n_pc].T
    pc = pd.DataFrame(scores, index=X.index, columns=[f"pc{i+1}" for i in range(n_pc)])
    return pc.diff()
```

Two rules:

- PCA on **levels**, then difference the score. Not PCA on changes. This is what
  equation (11) says.
- The loadings must be estimated on a training block and frozen, or the sign and
  scale of the score flip between windows and the signal is not a time series of
  anything. Fit once on the first 40% of dates and apply thereafter. The paper
  fits in-sample over the whole period, which is fine for a descriptive
  regression and not fine for a signal.

The expected beta is **negative** (Table 6), because PC1 up means yields up
means bond returns down. Flip the sign to make it a return signal:
`signal = -pc1_change`.

---

## 4. Signal C: Durham duration-neutral cross-sectional momentum

Different animal from A and B. This is cross-sectional across the curve on one
date, it is duration-neutral by construction, and the portfolio is the solution
to a linear program rather than a ranking.

### 4.1 Buckets

Durham's six Treasury sub-index buckets: 1-3, 3-5, 5-7, 7-10, 10-20, 20-30
years. Map to your OIS tenors by bucket midpoint:

```python
BUCKETS = [(1,3), (3,5), (5,7), (7,10), (10,20), (20,30)]
MIDPOINT = {b: 0.5*(b[0]+b[1]) for b in BUCKETS}   # 2, 4, 6, 8.5, 15, 25
```

For each bucket, build a constant-maturity swap return series at the midpoint
tenor, interpolating the par rate from the observed grid where the midpoint is
not quoted (8.5, 15 and 25 are; 2, 4 and 6 are quoted directly).

`D_i` in the LP is the **duration midpoint** of bucket `i`, which for a swap is
the annuity duration at the midpoint tenor from Section 1.4, not the maturity
midpoint. Durham uses index durations; the annuity duration is the swap
equivalent.

### 4.2 The momentum measure

"Average total returns for each duration bucket from between two through,
alternatively, five to 13 months prior." So the window **skips the most recent
month**: it runs from `t-2` back to `t-n`.

```python
def durham_momentum(ret_m_bp, n=12):
    """6-vector of average monthly returns over t-2 .. t-n. Skips t-1."""
    return ret_m_bp.shift(2).rolling(n - 1).mean()
```

With `n=12` the window is 11 months long, `t-2` through `t-12`, giving the
187 monthly observations Durham reports. Variants: `n` from 5 to 13.

This skip is a real difference from Signals A and B, which use the most recent
month. Keep it.

### 4.3 The long-only duration-neutral program

Durham equations (1) and (2). Maximise expected momentum subject to no shorts
and a duration match to the benchmark:

```
max_w   w' Rbar_{t-2,t-n}
s.t.    0 <= w_i <= 1
        w' D_t = d_index_t
        sum(w) = 1
```

```python
from scipy.optimize import linprog

def durham_long_only(rbar, dur, d_index):
    """rbar, dur: length-6 arrays. Returns the 6-vector of weights."""
    res = linprog(
        c=-np.asarray(rbar, float),              # linprog minimises
        A_eq=np.vstack([np.asarray(dur, float), np.ones(6)]),
        b_eq=np.array([d_index, 1.0]),
        bounds=[(0.0, 1.0)] * 6,
        method="highs",
    )
    if not res.success:
        return np.full(6, np.nan)
    return res.x
```

`sum(w) = 1` is not written out in the paper but is implied by "the portfolio
weights are bound between zero and one" describing a portfolio allocation with a
benchmark. Flag it in the manifest as an assumption, and run the variant without
it once to confirm it does not change the ranking of allocations.

**`d_index_t`, the benchmark duration.** Durham uses the US Treasury index
duration. There is no swap index. Two defensible choices, make it a parameter:

- `equal`: the equal-weighted average of the six bucket durations. Simple,
  stationary, the natural neutral point.
- `fixed`: a constant, e.g. 6 years, matching the typical govt index duration.

Record which. The choice shifts the whole allocation along the curve and is not
a detail.

**Expect corner solutions.** A linear objective with box constraints and two
equality constraints has its optimum at a vertex, so the LP will typically load
two or three buckets and zero the rest. Durham reports exactly this: allocations
"overweight intermediate maturities, the 5- to 7- and 7- to 10-year buckets" and
turning into barbells late in his sample. If your solver returns smoothly spread
weights, something is wrong with the constraints.

### 4.4 The long-short variant

Section 5 of the paper. Long the maximising allocation, short the minimising
one, box widened to [-1, 1], and the duration constraint becomes zero rather
than the benchmark:

```
max_w  w' Rbar     s.t.  -1 <= w_i <= 1,   w' D_t = 0
```

```python
def durham_long_short(rbar, dur):
    res = linprog(
        c=-np.asarray(rbar, float),
        A_eq=np.asarray(dur, float).reshape(1, -1),
        b_eq=np.array([0.0]),
        bounds=[(-1.0, 1.0)] * 6,
        method="highs",
    )
    return res.x if res.success else np.full(6, np.nan)
```

Note `sum(w) = 0` is not imposed: the constraint is zero *duration*, which is
what makes it market-neutral for a rates book. With swaps this is the natural
version, since an unfunded swap book has no cash constraint at all. Run the
long-short as the primary and the long-only as the constrained comparison.

---

## 5. Minimum tests before any of these is used

### 5.1 Sign and identity tests

- **Receiver sign.** Construct a flat curve, drop every rate by 1bp, assert the
  return from `cm_return_bp` is positive and approximately `dur` bp.
- **Decomposition identity.** `rx == c - yc` to 1e-10 at every tenor and date.
- **Exact vs approximate return.** `log_excess_return_bp` and `cm_return_bp`
  agree to within 2bp per month at tenors up to 10y. Assert they **diverge**
  beyond 10bp at 30y, so the convexity gap cannot be silently ignored.
- **Yield-change sign.** Build a synthetic curve where yields rise steadily.
  Assert `sihvonen_yieldchange` is negative and realised returns are negative.
- **Bootstrap round-trip.** Reprice each input par swap off the bootstrapped
  discount factors; recover the input par rate to within 0.01bp.
- **Duration neutrality.** For both Durham programs, `w' D` equals its target to
  1e-10. For the long-short, shift every bucket's rate by 1bp and assert the
  book PnL is zero to 1e-8.

### 5.2 Replication diagnostics

Each signal has a published pattern. If your panel does not show it, the build
is wrong, not the paper.

| check | expected | source |
|---|---|---|
| Sihvonen B1 beta by maturity | downward sloping, roughly 0.19 at 1y to 0.09 at 10y, h=1 | Table 1, Figure 1 |
| Sihvonen B1 beta at h=3, 6 | collapses to ~0.01-0.03, insignificant | Table 1 |
| Sihvonen B2 beta by maturity | upward sloping, ~0.04 at 1y to 0.18 at 10y | Table 3 |
| Sihvonen B4 PC1 beta | negative, t around -2 to -3 | Table 6 |
| Carry vs yield-change share | `Cov(yc_{t+1}, yc_t)` is 85-93% of total | Table 4 |
| Durham LP allocation | corner solutions, 2-3 buckets, intermediates favoured | Exhibit 4 |
| MOP sign | positive 12-month predictive t-stats for bond futures | Table 2 Panel D |

Run these on USD first, the longest and cleanest series. The Sihvonen numbers
are from US Treasuries 1971-2019 and yours is USD OIS from 2001, so expect the
magnitudes to differ. The **shapes** should survive. A beta term structure that
slopes the wrong way means the maturity axis or the return sign is flipped.

### 5.3 No-lookahead

The same test as the other two specs. Truncate the panel at date T, recompute
every signal, assert identity with the full-sample run at every date up to T-1.
Three values of T.

The two places this will fail if anywhere:

- `ex_ante_vol` without the `.shift(1)`
- `pc1_change` with PCA loadings fitted on the full sample

---

## 6. Output contract

Four panels, each `date x curve_id x tenor` (or `x bucket` for Durham), written
to `signals/`:

| file | content |
|---|---|
| `returns_monthly.parquet` | `rx`, `c`, `yc`, `cm_ret`, all bp |
| `signal_a_tsmom.parquet` | MOP position size, `k` as a column level |
| `signal_b_sihvonen.parquet` | `own`, `factor`, `yieldchg`, `pc1` |
| `signal_c_durham.parquet` | LP weights per bucket, long-only and long-short |

Plus `manifest.json` recording: daycount per curve, interpolation method,
bootstrap convergence tolerance, `delta` and annualization for the MOP vol, PCA
training block dates, `d_index` method, and the `sum(w)=1` assumption flag.

---

## 7. Build order

1. `returns.py` 1.1 and 1.2, then the bootstrap round-trip test. Nothing else
   runs until par rates reprice to 0.01bp.
2. 1.3 and 1.4, then the sign, identity and exact-vs-approximate tests.
3. Signal B first, not A. It is the simplest (one rolling sum, no vol model) and
   it has the richest set of published diagnostics in 5.2 to check against.
4. Signal A. The only new machinery is the vol estimator.
5. Signal C. New machinery is the LP and the bucket mapping.
6. `test_no_lookahead.py` across all four.

Signal B before A is deliberate. B has no free parameters beyond the lookback,
so if its beta term structure replicates, the return panel underneath is sound,
and A and C can be trusted on the same foundation.

---

## 8. Parameter summary, side by side

The three papers disagree with each other on every design choice. That is the
point of building all three rather than one parameterised signal.

| | A: MOP | B: Sihvonen | C: Durham |
|---|---|---|---|
| Lookback | 12 months | 1 month | 11 months (t-2 to t-12) |
| Skip most recent month | no | no | **yes** |
| Vol scaling | **yes**, 40% target | **explicitly no** | no |
| Signal shape | `sign()` | linear in past return | linear objective in an LP |
| Cross-section | none, per instrument | none, per maturity | **across 6 buckets** |
| Duration neutral | no | no | **yes, by construction** |
| Rebalance | monthly | monthly | monthly |
| Underlying | futures excess returns | zero-coupon log excess returns | index total returns |
