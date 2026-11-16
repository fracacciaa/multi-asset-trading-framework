chen now# Build spec: five FX signals from spot, forwards and the option surface

Raw signal construction only. No backtest harness, no PnL attribution, no spec
grid. The deliverable is five signal panels indexed by (date, pair), plus the
return and factor panels they are built from.

Target: `src/fxsig/`. Dependencies: `numpy`, `pandas`, `scipy`, `openpyxl`.

Papers implemented:

| Signal | Paper | What it is |
|---|---|---|
| A | Menkhoff, Sarno, Schmeling & Schrimpf (2012), BIS WP 366 / *JFE* | Cross-sectional momentum, 6 portfolios, high minus low |
| B | Zhang, "Dissecting Currency Momentum", *JFE* | Carry/dollar factor decomposition, then **factor** momentum |
| C | Moskowitz, Ooi & Pedersen (2012), *JFE* 104(2) | Time series momentum, vol-scaled |
| D | Dupuy (2021), *JBF*, "Risk-adjusted Return Managed Carry Trade" | Carry with leverage timed by E/V |
| E | Brunnermeier, Nagel & Pedersen (2008), NBER WP 14473 | Crash-risk conditioning of carry via risk reversals |

Section 8 explains why the sixth attached paper (Heinonen, option implied
eigenportfolios) is **not buildable** from this dataset, and what to do instead.

---

## 0. What the data actually is

I inspected `FXXXXXXXXX.xlsx`. 26 sheets: a `Sheet1` ticker map, 24 pair
sheets, and an empty `Sheet27`. Each pair sheet has 31 three-column blocks
(DATE, PX_LAST, blank) laid out left to right:

| block start col (0-indexed) | series |
|---|---|
| 0 | SPOT |
| 3, 6, 9, 12, 15 | FWD 1M, 3M, 6M, 12M, 2Y |
| 18, 21, 24, 27, 30 | ATM vol 1M, 3M, 6M, 1Y, 2Y |
| 33, 36, 39, 42, 45 | 25Δ RR 1M, 3M, 6M, 1Y, 2Y |
| 48, 51, 54, 57, 60 | 25Δ BF 1M, 3M, 6M, 1Y, 2Y |
| 63, 66, 69, 72, 75 | 10Δ RR 1M, 3M, 6M, 1Y, 2Y |
| 78, 81, 84, 87, 90 | 10Δ BF 1M, 3M, 6M, 1Y, 2Y |

Data rows start at row 8 (1-indexed). Each block carries its **own** date
column, and those dates do not align across blocks. Join on date per block,
never by row position.

### 0.1 Coverage, measured not assumed

| pair | spot | fwd 1M | 25RR 1M |
|---|---|---|---|
| AUDUSD | 1990-01 | 1990-01 | 2003-10 |
| EURUSD | 1990-01 | **1998-12** | 2003-10 |
| GBPUSD | 1990-01 | 1990-01 | 2003-10 |
| NZDUSD | 1990-01 | 1990-01 | 2003-10 |
| USDCAD | 1990-01 | 1990-01 | 2003-10 |
| USDCHF | 1990-01 | 1990-01 | 2005-03 |
| USDDKK | 1990-01 | 1990-01 | 2005-03 |
| USDJPY | 1990-01 | 1990-01 | 2003-10 |
| USDNOK | 1990-01 | 1990-01 | 2003-07 |
| USDSEK | 1990-01 | 1990-01 | 2003-07 |
| USDKRW | 1990-01 | 1998-12 | 2003-10 |
| USDMXN | 1990-01 | 1997-11 | 2002-08 |
| USDSGD | 1990-01 | 1990-01 | 2003-10 |
| USDZAR | 1990-01 | 1990-01 | 2003-10 |
| USDCZK | 1993-06 | 1996-12 | 1999-03 |
| USDCNY | 1990-01 | **2006-07** | 2003-10 |
| USDTHB | 1990-01 | 1995-09 | 2003-10 |
| USDPLN | 1993-06 | 1998-07 | 2003-10 |
| USDHUF | 1993-06 | 1998-07 | 2005-03 |
| USDTRY | 1990-01 | 1996-12 | 2005-02 |
| **USDBRL** | 1992-01 | **MISSING** | 2005-11 |
| **USDTWD** | 1990-01 | **MISSING** | 2003-10 |
| **USDINR** | 1990-01 | **MISSING** | 2005-03 |
| **USDIDR** | 1991-11 | **MISSING** | 2000-06 |

Three consequences that shape every signal below.

1. **Four pairs have no forward data**: BRL, TWD, INR, IDR. These are NDF
   markets and the outright forward sits under different Bloomberg tickers
   (BCN/NTN/IRN/IHN series), which are not in this workbook. Carry and
   forward-based excess returns are not computable for them. Either source the
   NDF points or exclude the four. **Default: exclude, and record the
   exclusion.** Do not silently fill.
2. **The usable carry/momentum universe is 20 pairs**, and only from 1999
   onward does it exceed about 16 pairs. Before that EUR does not exist and
   KRW, CNY, PLN, HUF have no forwards. Start the primary sample at
   **1999-01**, matching Dupuy's own main sample choice.
3. **The option surface starts 2003-10** for most pairs, 2005 for CHF, DKK,
   HUF, INR, TRY. Every option-conditioned signal therefore runs on a
   2003-10 onward sample of about 23 years. Never compare an option-conditioned
   signal's Sharpe to a 1999-start plain carry Sharpe. Re-run the plain carry on
   the matched 2003-10 sample for that comparison.

### 0.2 Three data traps, all of which silently invert or rescale signals

**Trap 1: quotation orientation.** Four pairs are quoted FCU-base (AUDUSD,
EURUSD, GBPUSD, NZDUSD, i.e. USD per foreign unit). The other twenty are
USD-base (USDJPY etc, i.e. foreign units per USD). Every paper here defines
`s = log(foreign currency units per USD)`. So:

```python
FCU_BASE = {"AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"}   # quoted USD per FCU

def to_fcu_per_usd(pair, spot_px):
    return 1.0 / spot_px if pair in FCU_BASE else spot_px
```

Do this **before** anything else, and assert in a test that USDJPY and a
synthetic inverted AUDUSD both produce the same sign of excess return for the
same underlying move.

**Trap 2: forward points are pips, not outright rates.** The workbook holds
Bloomberg forward *points*. AUDUSD 1M shows `-74` against a spot of `0.789`;
USDCAD 1M shows `44` against `1.158`. The outright is

```
F = S + points / divisor
```

and the divisor varies by pair. Do not hardcode a guessed table. Calibrate it:

```python
CANDIDATES = [1.0, 10.0, 100.0, 1000.0, 10000.0]

def calibrate_divisor(spot, points):
    """Pick the divisor making the annualized 1M forward discount land in a
    plausible range. Returns the divisor and the implied annualized discount."""
    best = None
    for dv in CANDIDATES:
        fd = (points / dv) / spot * 12.0          # annualized, 1M tenor
        score = abs(np.nanmedian(np.abs(fd)) - 0.02)   # typical |carry| ~2%
        if np.nanmax(np.abs(fd)) < 0.60 and (best is None or score < best[1]):
            best = (dv, score)
    return best[0] if best else np.nan
```

Run this once per pair on the overlapping spot/forward sample, print the table,
and **eyeball it before proceeding**. A divisor off by 10x makes a carry signal
that is correctly ranked but wrongly scaled, which destroys Dupuy's signal
(Section 6) because that one uses the carry level, not its rank. Sanity check
the output: JPY and CHF should come out persistently negative discount, TRY,
ZAR, BRL, MXN persistently positive.

**Trap 3: risk reversal sign convention follows the pair quoting.** Bloomberg's
`AUDUSD25R` is IV(25Δ AUD call) minus IV(25Δ AUD put). Bloomberg's `USDJPY25R`
is IV(25Δ **USD** call) minus IV(25Δ USD put). Those are opposite orientations
with respect to the foreign currency. BNP measure risk-neutral skewness of the
**foreign currency**, and report AUD at −0.426 and JPY at +1.059 (Table 1), the
high-carry currency negative and the funding currency positive.

```python
def rr_foreign_oriented(pair, rr):
    """Positive = market pays up for foreign-currency APPRECIATION."""
    return rr if pair in FCU_BASE else -rr
```

Assert the BNP pattern as a test: after orientation, the time-series mean RR for
AUD, NZD, ZAR, TRY, MXN should be negative and for JPY and CHF positive. If that
fails, the flip is wrong and Signal E is inverted.

Butterflies are orientation-free (symmetric), ATM vol likewise. Only risk
reversals flip.

### 0.3 No bid/ask

The workbook has `PX_LAST` only. Menkhoff's net-return construction
(`f^b_t − s^a_{t+1}` for longs, `−f^a_t + s^b_{t+1}` for shorts) and Dupuy's
transaction-region rule both need two-sided quotes and cannot be reproduced.
Two consequences, both of which must be written into the manifest:

- Use a flat per-pair round-trip cost in bp as a placeholder. Suggested starting
  values, to be replaced with your own marks: 2bp G10, 5bp CEE and Asia, 10bp
  TRY/ZAR/BRL. These are placeholders, not estimates.
- Dupuy's "attractive currency" filter degenerates. See Section 6.2.

---

## 1. `data.py`: loading and the return panel

### 1.1 Loader

```python
BLOCKS = {
    "SPOT": 0,
    "FWD1M": 3, "FWD3M": 6, "FWD6M": 9, "FWD12M": 12, "FWD2Y": 15,
    "ATM1M": 18, "ATM3M": 21, "ATM6M": 24, "ATM1Y": 27, "ATM2Y": 30,
    "RR25_1M": 33, "RR25_3M": 36, "RR25_6M": 39, "RR25_1Y": 42, "RR25_2Y": 45,
    "BF25_1M": 48, "BF25_3M": 51, "BF25_6M": 54, "BF25_1Y": 57, "BF25_2Y": 60,
    "RR10_1M": 63, "RR10_3M": 66, "RR10_6M": 69, "RR10_1Y": 72, "RR10_2Y": 75,
    "BF10_1M": 78, "BF10_3M": 81, "BF10_6M": 84, "BF10_1Y": 87, "BF10_2Y": 90,
}
```

Read each block as its own (date, value) series, convert the Excel serial with
origin 1899-12-30, drop non-numeric cells (`#N/A N/A` appears), then outer-join
all blocks for a pair on date. Output one tidy frame:

`date, pair, series, value`

Validators that must raise, not warn:
- a pair sheet missing the SPOT block
- any spot value <= 0
- duplicate (date, pair, series)
- an ATM vol outside (0, 200) or a butterfly outside (-10, 20)

### 1.2 Orientation and outrights

Apply Traps 1 to 3 from Section 0.2. Produce, per pair and date:

```
s      = log(FCU per USD)                      # log spot, paper convention
f_tau  = log(FCU per USD forward, tenor tau)   # log outright
fd_tau = f_tau - s                             # log forward discount
rr25_1m, rr10_1m, bf25_1m, atm_1m ...          # foreign-oriented
```

For an FCU-base pair the outright must be inverted **after** adding points:

```python
F_quoted = spot_px + points / divisor          # still USD per FCU for AUDUSD
F_fcu_per_usd = 1.0 / F_quoted if pair in FCU_BASE else F_quoted
```

Inverting the spot first and then adding points is wrong and is the second most
likely bug after the divisor.

### 1.3 Excess returns

Menkhoff equation (1), identical to Zhang equation (1) and Dupuy's:

```
rx_{i,t+1} = i*_{i,t} - i_t - Δs_{i,t+1} ≈ f_{i,t} - s_{i,t+1}
```

with `f` the log 1-month outright and `s` the log spot, both FCU per USD. This
is the return to a US investor **long** foreign currency `i`, funded in USD.

```python
def monthly_excess_return(f_1m_prev, s_now):
    """Both log, FCU per USD. f from month-end t, s from month-end t+1."""
    return f_1m_prev - s_now
```

Sign check, to be asserted: a currency with a positive forward discount whose
spot does not move earns a positive `rx`. A currency that depreciates against
USD by more than its discount earns a negative `rx`.

**Daily excess return**, required for Zhang's beta estimation in Section 4 and
Dupuy's market variance in Section 6:

```python
def daily_excess_return(fd_1m_prev, s_prev, s_now, days_in_month=21):
    return fd_1m_prev / days_in_month - (s_now - s_prev)
```

The carry accrues smoothly and the spot move is the daily innovation. Use the
month's actual business-day count, not a constant 21, and use `fd` from the
previous month end held fixed through the month.

### 1.4 Sampling

All five papers are monthly. Resample to the last business day of each month.
Keep the daily panel only for Section 4's betas and Section 6's variance.

---

## 2. Signal A: Menkhoff cross-sectional momentum

### 2.1 Construction

Verbatim from the paper: at the end of each month, form **six** portfolios on
lagged returns over the previous `f` months. The lowest sixth of available
currencies goes to portfolio 1 ("Low"), the highest sixth to portfolio 6
("High"). `MOM_{f,h}` is High minus Low, and is dollar-neutral because the
dollar component cancels in the difference.

```python
def menkhoff_portfolios(rx_m, f=6, n_port=6):
    """rx_m: date x pair monthly excess returns (logs).
    Returns date x pair integer portfolio labels 1..n_port."""
    past = rx_m.rolling(f).sum()
    def assign(row):
        v = row.dropna()
        if len(v) < n_port:
            return pd.Series(np.nan, index=row.index)
        q = pd.qcut(v.rank(method="first"), n_port, labels=False) + 1
        return q.reindex(row.index)
    return past.apply(assign, axis=1)
```

Signal panel, as weights:

```python
def menkhoff_weights(port, n_port=6):
    hi = (port == n_port).astype(float)
    lo = (port == 1).astype(float)
    w = hi.div(hi.sum(axis=1), axis=0) - lo.div(lo.sum(axis=1), axis=0)
    return w.fillna(0.0)
```

### 2.2 Parameters

| parameter | paper value |
|---|---|
| formation `f` | 1, 3, 6, 9, 12 months |
| holding `h` | 1, 3, 6, 9, 12 months |
| portfolios | 6 |
| skip month | none |
| vol scaling | none |
| rebalance | monthly, end of month |

For `h > 1`, hold `h` overlapping vintages and average, same as the rates spec:
`w_h = w.rolling(h).mean()`.

With only 20 usable pairs the sixths are 3 to 4 currencies each. That is thinner
than the paper's cross-section but matches the industry version they cite
(Deutsche Bank's index trades the best and worst three of G10). Record the
realised portfolio size per date; if it drops below 2, set the date to NaN
rather than trading a single-name portfolio.

### 2.3 Spot-only variant

Menkhoff track spot rate changes of the momentum portfolios separately, to see
whether momentum is an interest-rate-differential effect or a spot effect. Build
it: rerun the whole thing sorting on `-Δs` cumulated rather than on `rx`, and
output it as a separate panel. Cheap, and it answers the first question anyone
will ask of the result.

---

## 3. Signal C: MOP time series momentum

Placed here because Signal B depends on both A and C. This is the same
construction as the rates momentum spec, so reuse that module if it exists.

### 3.1 Ex ante volatility

MOP equation (1), on daily excess returns, annualized:

```
sigma_t^2 = 261 * sum_i (1-delta) delta^i (r_{t-1-i} - rbar_t)^2
```

with `delta/(1-delta) = 60` days, so `delta = 60/61 = 0.983607`.

```python
DELTA = 60.0 / 61.0

def ex_ante_vol(r_daily):
    r = r_daily.shift(1)                       # MOP use sigma at t-1
    mu = r.ewm(alpha=1 - DELTA, adjust=False).mean()
    var = ((r - mu) ** 2).ewm(alpha=1 - DELTA, adjust=False).mean()
    return np.sqrt(261.0 * var)
```

Resample to month-end.

### 3.2 Signal

MOP equation (5):

```
r^{TSMOM,i}_{t,t+1} = sign(rx_{i,t-12,t}) * (40% / sigma_{i,t}) * rx_{i,t,t+1}
```

```python
def tsmom_position(rx_m, sigma_ann, k=12, target=0.40):
    past = rx_m.rolling(k).sum()
    return np.sign(past) * (target / sigma_ann)
```

Lookback grid {1, 3, 6, 9, 12}, holding {1, 3, 6, 9, 12}, 40% per-instrument
vol target, no skip month.

### 3.3 The unscaled variant is mandatory, not optional

Zhang equal-weights the individual-currency TSM at `+1/N` or `-1/N` and
explicitly declines MOP's vol scaling, citing Kim et al (2016) and Huang et al
(2020) that "volatility scaling is a source of potential predictability
different from TSM and can overstate TSM". Sihvonen makes the same objection in
the rates literature. Build both:

```python
def tsmom_equal_weight(rx_m, k=12):
    sgn = np.sign(rx_m.rolling(k).sum())
    return sgn.div(sgn.abs().sum(axis=1), axis=0)
```

Report them side by side. If vol-scaled TSMOM beats equal-weight TSMOM by a
wide margin in your panel, the difference is the vol-timing bet, not momentum,
and should be attributed as such.

---

## 4. Signal B: Zhang factor decomposition and factor momentum

The highest-research-value signal here, and the only one that needs machinery
the others do not.

### 4.1 The two factors

**Dollar factor**, Zhang equation (2), the equal-weighted average excess return:

```python
def dollar_factor(rx):
    return rx.mean(axis=1)
```

**Carry factor**, Zhang equation (3) and (4), rank-weighted and dollar-neutral:

```python
def carry_factor(rx, fd):
    """fd: date x pair forward discounts, observed at the START of the period."""
    r = fd.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1) / 2.0, axis=0)   # c_t: 1 long, 1 short
    return (w * rx).sum(axis=1), w
```

The weights sum to zero by construction. `c_t` scales gross exposure to one
dollar long and one dollar short; Zhang notes this is leverage only and does not
change profitability. Assert `w.sum(axis=1) == 0` and `w.abs().sum(axis=1) == 2`.

Build both at **daily** and **monthly** frequency. The daily versions are needed
next.

### 4.2 Decomposition

Zhang equation (5):

```
rx_{i,t} = b^carry_{i,t} * carry_t + b^dollar_{i,t} * dollar_t + (a_i + e_{i,t})
           \_______________ systematic _______________/   \__ idiosyncratic __/
```

Betas are estimated **each month from that month's daily data**, which gives
time-varying loadings:

```python
def monthly_betas(rx_d, carry_d, dollar_d, min_obs=15):
    """Returns two date(month-end) x pair frames of betas."""
    out_c, out_d = {}, {}
    X = pd.concat([carry_d, dollar_d], axis=1)
    X.columns = ["carry", "dollar"]
    for month, idx in rx_d.groupby(pd.Grouper(freq="ME")).groups.items():
        xm = sm_add_const(X.loc[idx].dropna())
        bc, bd = {}, {}
        for p in rx_d.columns:
            y = rx_d.loc[idx, p].reindex(xm.index).dropna()
            if len(y) < min_obs:
                bc[p], bd[p] = np.nan, np.nan
                continue
            A = xm.loc[y.index].to_numpy()
            beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
            bc[p], bd[p] = beta[1], beta[2]      # beta[0] is the intercept
        out_c[month], out_d[month] = bc, bd
    return pd.DataFrame(out_c).T, pd.DataFrame(out_d).T
```

Implement `sm_add_const` inline as `np.column_stack([ones, X])`; do not import
statsmodels.

Then:

```python
systematic = b_carry * carry_m + b_dollar * dollar_m     # broadcast per pair
idiosyncratic = rx_m - systematic
```

The month-t betas and the month-t factor returns both refer to month t, so
`systematic_t` is known at the end of month t and may be used to form a position
for t+1. That is Zhang's own argument for estimating monthly rather than from a
trailing window. Alternative windows of 3 to 12 months are a grid axis.

### 4.3 Systematic and idiosyncratic momentum

Rerun Signal A and Signal C on `systematic` and on `idiosyncratic` in place of
`rx`. Four extra panels. Zhang's finding, which is the diagnostic to check
against: systematic momentum spreads are 45, 48 and 30bp per month at 1, 3 and
12 month formation and significant, while idiosyncratic spreads are −2, −10 and
5bp and insignificant at every horizon.

### 4.4 Factor momentum, the headline signal

Zhang equation (7): condition on the **sign** of the past factor return.

```python
def factor_momentum(factor_m, h=3):
    """+1 when the past h-month average factor return is positive, else -1.
    Applied to the factor's own weights."""
    past = factor_m.rolling(h).mean().shift(1)
    return np.sign(past)
```

Two tradeable panels follow, one per factor:

```python
w_carry_mom  = factor_momentum(carry_m,  h) * carry_w      # carry_w from 4.1
w_dollar_mom = factor_momentum(dollar_m, h) * (1.0 / N)    # dollar is 1/N long all
```

Formation `h` in {1, 3, 12}, matching the momentum formation periods.

Expected magnitudes from Zhang's Table 4, right panel, as the replication check:
the expected carry return rises by 62, 55 and 15bp following 1, 3 and 12 months
of positive carry returns, and the expected dollar return by 56, 68 and 52bp
following positive dollar returns. All highly significant except 12-month carry.
The asymmetry matters: factors "mostly do not earn significantly positive
returns next month following losses", so the long leg carries the result. Output
the conditional means separately for positive and negative past-sign states so
this asymmetry is visible rather than averaged away.

---

## 5. Signal D: Dupuy risk-adjusted managed carry

A leverage overlay on a standard carry portfolio, not a new cross-section.

### 5.1 Base carry portfolio

Dupuy equation (6): rank by forward discount, go long the top `n` and short the
bottom `n`, equal weight, `n = 4` in the paper.

```python
def carry_weights(fd, n=4):
    r = fd.rank(axis=1, ascending=False)        # rank 1 = highest discount
    k = fd.notna().sum(axis=1)
    lo_cut = k.values[:, None] - n
    w = np.where(r.values <= n, 1.0 / n,
        np.where(r.values > lo_cut, -1.0 / n, 0.0))
    w = pd.DataFrame(w, index=fd.index, columns=fd.columns)
    return w.where(fd.notna(), 0.0)
```

With `n = 4` and a 20-pair universe this holds 4 long and 4 short. The paper's
benchmark fixes the signal at 1, so this unlevered portfolio is the comparison.

### 5.2 Expected profit, E

Dupuy equation (2):

```
E_t(pi_{t+1}) = [ sum_i (f^b_t/s^a_t)_i - 1 | (f^b/s^a)_i > 1 ] / n^buy
              - [ sum_j (f^a_t/s^b_t)_j - 1 | (f^a/s^b)_j < 1 ] / n^sell
```

the average forward discount of the **attractive** investment currencies minus
that of the attractive financing currencies.

**This is where the missing bid/ask bites.** With bid and ask, `f^a/s^b > 1` and
`f^b/s^a < 1` simultaneously defines a genuine no-transaction region where a
currency is attractive on neither side. With mid quotes only, `f/s` is either
above or below 1, so the no-transaction region is empty and every currency is
attractive on exactly one side. The filter becomes vacuous.

Implement the degraded version, and flag it in the manifest:

```python
def expected_profit_mid(fd, cost_bp=None):
    """fd: log forward discounts. Degraded E with mid quotes.
    cost_bp: optional synthetic half-spread in bp to restore the dead zone."""
    fdx = fd.copy()
    if cost_bp is not None:
        c = cost_bp / 1e4
        buy  = fdx.where(fdx >  c)              # attractive long
        sell = fdx.where(fdx < -c)              # attractive short
    else:
        buy  = fdx.where(fdx > 0)
        sell = fdx.where(fdx < 0)
    return buy.mean(axis=1).fillna(0.0) - sell.mean(axis=1).fillna(0.0)
```

Run `cost_bp=None` as the baseline and a synthetic `cost_bp` sweep (2, 5, 10bp)
as a grid axis, since the width of the dead zone is the one thing the mid-only
data cannot tell you and it directly scales E.

### 5.3 Market variance, V

Dupuy equation (3), French-Schwert-Stambaugh with the first-order
autocorrelation correction:

```
MV_t = sum_d r^2_{Mt,d} / D_t + 2 * sum_d r_{Mt,d} * r_{Mt,d-1} / D_t
```

where `r_Mt` is the equally weighted FX market portfolio excess return and `D_t`
the number of trading days in the month.

```python
def market_variance(rx_daily):
    rM = rx_daily.mean(axis=1)                  # 1/N market portfolio
    def mv(x):
        D = len(x)
        if D < 15:
            return np.nan
        return (x ** 2).sum() / D + 2.0 * (x * x.shift(1)).sum() / D
    return rM.groupby(pd.Grouper(freq="ME")).apply(mv)
```

The autocovariance term can make `MV` negative in quiet months with strong
negative daily autocorrelation. Floor it at a small positive value and count how
often the floor binds; if it binds more than a few percent of months, the
universe is too small and `rM` is too noisy.

### 5.4 The signal

Dupuy equation (4), with his explicit extra lag:

```python
def dupuy_signal(E, MV, mode="E_over_V"):
    raw = {"E_over_V": E / MV, "E_only": E, "V_only": 1.0 / MV}[mode]
    sig = raw / raw.expanding(min_periods=36).mean()   # normalise mean exposure to 1
    return sig.shift(1)                                 # the extra month lag
```

Two points that are easy to miss and both come straight from the paper:

- **The extra lag is explicit.** "To ensure no overlap between the signal and
  the future return, we further lag the signal by one month. Therefore, the
  signal obtained at the end of month t is used for the positions held from the
  end of month t+1 to the end of month t+2." The volatility leg is measured over
  month t and so overlaps month t if used immediately. Do not drop this lag to
  "improve" the result.
- **Average exposure is normalised to 1**, matching the unconditional exposure
  of the benchmark. The expanding mean above does this without lookahead. Dupuy
  normalises on the full sample, which is not implementable in real time; he
  says so himself. Use the expanding version and note the difference.

Final weights: `w_dupuy = carry_weights(fd, n=4) * dupuy_signal(...)`.

Dupuy reports exposure varying from 0.25 to 2.7. Check yours lands in a similar
band; if it swings by 10x, `MV` is being floored or `E` is mis-scaled by a bad
forward-point divisor.

Three variants to output, which are the paper's own decomposition:
`E_over_V` (his headline, SR 1.08), `E_only` (0.99), `V_only` (0.84), against
the unlevered benchmark (0.76).

---

## 6. Signal E: BNP crash-risk conditioned carry

BNP is a risk paper, not a trading-rule paper. The tradeable content is the
conditioning, and the workbook's risk reversals are exactly the instrument they
use: "we obtain data from JPMorgan Chase on quotes of 25Δ 1-month risk
reversals."

### 6.1 The two crash-risk measures

**Realised skewness**, BNP's physical measure. Note the minus sign: they measure
the skewness of `-Δs`, i.e. of foreign currency **appreciation**, within each
quarter.

```python
def realised_skew(s_log, freq="QE"):
    ds = -s_log.diff()                          # foreign appreciation
    return ds.groupby(pd.Grouper(freq=freq)).skew()
```

**Risk-neutral skewness**, the 25Δ 1M risk reversal, foreign-oriented per
Section 0.2. This is the one with the shorter history and the better
tradeability.

### 6.2 The cross-sectional fact to verify first

BNP's central finding is a negative cross-sectional relation between the
interest differential and skewness: high-carry currencies have negative skew and
negative risk reversals, funding currencies positive. Their Table 1 reports AUD
RR −0.426, NZD −0.467, CAD −0.099, JPY +1.059, CHF +0.409.

Run this before building anything:

```python
# per pair, time-series means over the common sample
corr = np.corrcoef(fd.mean(), rr25_1m_oriented.mean())[0, 1]
```

Expect clearly negative. If it comes out positive, the RR orientation in
Section 0.2 is flipped and Signal E will be backwards.

### 6.3 Signal construction

Three forms, in increasing aggressiveness. All take the base carry weights from
Section 5.1 and modulate them.

**E1, crash-risk screen.** Drop from the long leg those currencies whose
foreign-oriented RR is in the most negative tail of the cross-section, on the
grounds that the market is charging most for their crash.

```python
def e1_screen(w_carry, rr, pct=0.2):
    thresh = rr.quantile(pct, axis=1)
    drop = rr.lt(thresh, axis=0) & (w_carry > 0)
    w = w_carry.mask(drop, 0.0)
    return w.div(w.abs().sum(axis=1) / 2.0, axis=0)
```

**E2, carry per unit of implied crash risk.** Rank on the forward discount
divided by the price of downside protection rather than on the discount alone.

```python
def e2_adjusted_carry(fd, rr, bf, kind="rr"):
    denom = {"rr": -rr, "bf": bf, "rr_bf": (-rr + bf)}[kind]
    return fd / denom.clip(lower=0.1)           # protection cost floored
```

Then feed `e2_adjusted_carry` into `carry_weights` in place of `fd`. `-rr` is
the cost of the downside skew, `bf` the cost of both tails. Both are in vol
points, so the ratio is carry per vol point, which is dimensionally the point.

**E3, aggregate de-risking.** BNP link unwinds to funding liquidity and the VIX.
You have no TED or VIX here, but the cross-sectional average ATM vol is a usable
global FX volatility state, and the average risk reversal is a usable global
risk-appetite state.

```python
def e3_global_state(atm, rr):
    gvol = atm.mean(axis=1)
    gskew = rr.mean(axis=1)
    z = lambda x: (x - x.expanding(36).mean()) / x.expanding(36).std()
    return (-z(gvol) - z(gskew)).clip(-2, 2)    # scale down when vol high, skew negative
```

Multiply the base carry weights by `e3_global_state(...).shift(1)`.

E3 and Dupuy's `V_only` are close cousins: both scale carry down when FX
volatility is high. The difference is that Dupuy uses realised variance of the
market portfolio and E3 uses implied vol plus implied skew. Compute the
correlation between the two scalers and report it; if it exceeds roughly 0.8
they are one signal and only the cheaper one should be kept.

### 6.4 The option-moment extension

The Chen (2017) abstract in your reading list describes carry sorted on
option-implied volatility, skewness and tail fatness, and reports that if the
implied returns of a pair are more left-skewed than in the past, anti-carry
beats carry. The PDF attached under that heading is Heinonen, not Chen (see
Section 8), so this is built from the described construction rather than from a
text I can read.

Three predictors per pair, all at the 1M tenor, all lagged one month:

```
vol_i   = atm_1m
skew_i  = rr25_1m_oriented                      # left-skew is negative
tail_i  = bf25_1m, or (bf10_1m - bf25_1m) for fat-tail-beyond-25d
```

Two signal forms:

```python
# level form: interact carry with each moment
s_level = fd * z_cross(vol) , fd * z_cross(-skew) , fd * z_cross(tail)

# change form: Chen's anti-carry result
d_skew = rr25_1m_oriented.diff(1)               # more left-skewed = negative
s_change = -np.sign(d_skew) * carry_weights(fd)  # flip to anti-carry on worsening skew
```

Keep these behind a flag and treat them as the most speculative family here.
Three interaction terms on a 23-year sample is already enough trials to need a
deflated Sharpe at selection time.

---

## 7. What is not buildable from this data

**Heinonen, "Information content of option implied eigenportfolios"** (attached
as `ssrn-3228137.pdf`) requires the full cross-section of **45 G10 cross-rate**
implied volatilities, from which he backs out implied covariance matrices per
tenor and takes their first two eigenportfolios as implied dollar and implied
carry factors.

Your workbook has 24 pairs, **all against USD**, with no cross-rate vols
(no EURJPY, AUDJPY, EURGBP and so on). An implied covariance matrix cannot be
recovered from USD-pair vols alone: the cross-rate vol is what identifies the
off-diagonal. With `n` USD pairs you have `n` variances and zero covariances,
and you need `n(n-1)/2` cross vols to fill the matrix.

Three honest options, in order of preference:

1. **Pull the cross-rate vols.** For G10 that is 45 tickers at the 1M tenor
   (`EURJPYV1M BGN Curncy` and so on). This is the only route to the paper as
   written.
2. **Degrade to a correlation assumption.** Use realised correlations of daily
   spot returns with the implied variances on the diagonal. This is a hybrid,
   not Heinonen's measure, and loses the forward-looking property that is the
   paper's entire point.
3. **Drop it.** Signals D and E already use the surface in ways this data
   supports.

Do not build option 2 and call it Heinonen.

---

## 8. Tests

### 8.1 Orientation and units

- **Inversion symmetry.** Build a synthetic pair quoted both ways from the same
  underlying path. Both must produce identical `rx` to 1e-12.
- **Forward divisor.** For each pair, the median absolute annualized 1M forward
  discount must be under 25%, and for G10 under 6%. Print the table and fail any
  pair outside the band.
- **Carry sign.** Over the common sample, mean `fd` must be negative for JPY and
  CHF and positive for TRY, ZAR, BRL, MXN.
- **RR orientation.** Mean foreign-oriented RR negative for AUD, NZD, ZAR, TRY,
  MXN and positive for JPY, CHF. This is the BNP Table 1 pattern.
- **Excess return identity.** `rx == fd_prev - Δs` to 1e-12.

### 8.2 Portfolio construction

- Menkhoff `MOM_{f,h}` weights sum to zero at every date, and gross exposure
  equals 2.
- Zhang carry weights sum to zero and have gross exposure 2.
- Dupuy base weights hold exactly `n` long and `n` short whenever at least `2n`
  pairs are available.
- Factor momentum `sign` takes only the values {-1, 0, +1}.

### 8.3 Replication diagnostics

Each signal has a published pattern. Shapes should survive even though your
sample and universe differ.

| check | expected | source |
|---|---|---|
| Menkhoff portfolio returns | monotone increasing from P1 to P6 | Menkhoff Table |
| Zhang systematic vs idiosyncratic momentum | systematic spreads ~45, 48, 30bp significant; idiosyncratic ~−2, −10, 5bp insignificant | Zhang Table 2 |
| Zhang factor momentum asymmetry | strong positive after positive factor months, nothing after losses | Zhang Table 4 |
| Zhang conditional correlation | MOM vs carry 0.33/0.35/0.12 after positive carry, −0.58/−0.62/−0.69 after negative | Zhang Table 3 |
| Dupuy exposure range | roughly 0.25 to 2.7 | Dupuy Section 3.2 |
| Dupuy signal ranking | E/V > E > 1/V > benchmark on Sharpe | Dupuy Section 3.2.1 |
| BNP carry-skew relation | negative cross-sectional correlation of `fd` and RR | BNP Table 1 |
| BNP carry skewness | carry portfolio returns negatively skewed, around −0.5 to −1.0 | BNP Table 2 |

The BNP skewness check is also the honesty check on the whole carry family: if
your carry portfolio shows positive skew, something upstream is wrong.

### 8.4 No-lookahead

Truncate the panel at date T, recompute every signal, assert identity with the
full-sample run at every date up to T−1. Three values of T.

The four places this will fail if anywhere:

- `dupuy_signal` without the extra `.shift(1)`
- `dupuy_signal` normalised by a full-sample rather than expanding mean
- `e3_global_state` z-scored on the full sample
- Zhang's monthly betas if estimated on a window that extends past month end

---

## 9. Output contract

Written to `signals/`, each `date x pair` monthly unless stated:

| file | content |
|---|---|
| `returns.parquet` | `s`, `fd_1m`, `rx_m`, `rx_d`, oriented surface columns |
| `factors.parquet` | `carry`, `dollar` monthly and daily, plus carry weights |
| `decomposition.parquet` | `b_carry`, `b_dollar`, `systematic`, `idiosyncratic` |
| `signal_a_menkhoff.parquet` | weights, by `(f, h)` |
| `signal_b_zhang.parquet` | `sys_mom`, `idio_mom`, `factor_mom_carry`, `factor_mom_dollar` |
| `signal_c_tsmom.parquet` | vol-scaled and equal-weight, by `k` |
| `signal_d_dupuy.parquet` | `E`, `MV`, `signal`, weights, for all three modes |
| `signal_e_bnp.parquet` | `E1`, `E2` (rr/bf/rr_bf), `E3`, plus the option-moment variants |

Plus `manifest.json`: per-pair forward divisor and how it was calibrated, the
FCU-base list, the RR orientation applied, excluded pairs and why, sample start
per signal, the synthetic cost assumption, and the `MV` floor hit rate.

---

## 10. Build order

1. `data.py`, loader and the three orientation traps. Stop and run every test in
   8.1. Print the forward divisor table and read it before continuing.
2. Monthly and daily excess returns. Verify the identity test.
3. Signal A (Menkhoff). Simplest, and its monotone-portfolio check validates the
   return panel.
4. Signal C (MOP), both scaled and unscaled.
5. The factors in 4.1, then Signal D (Dupuy). Dupuy only needs the carry
   portfolio and two scalars, so it is cheap once the factors exist.
6. Signal E (BNP), starting with the 6.2 cross-sectional check.
7. Signal B (Zhang) last. It is the only one needing daily rolling betas, and it
   consumes A, C and the factors, so everything it depends on is already tested.
8. `test_no_lookahead.py` across all five.

Signal A before the rest is deliberate: its portfolio monotonicity is the
cheapest end-to-end check that orientation, divisors and returns are all right.

---

## 11. Parameter summary

| | A: Menkhoff | B: Zhang | C: MOP | D: Dupuy | E: BNP |
|---|---|---|---|---|---|
| Type | cross-sectional | factor | time series | carry overlay | carry overlay |
| Sort variable | past `rx` | past factor return sign | own past `rx` sign | forward discount | fd and RR |
| Formation | 1,3,6,9,12m | 1,3,12m | 1,3,6,9,12m | n/a | n/a |
| Portfolios | 6 | n/a | n/a | 4 long, 4 short | 4 long, 4 short |
| Vol scaling | no | **no** | **yes**, 40% | via `1/MV` | via implied vol |
| Extra lag | no | no | no | **yes, +1 month** | no |
| Needs options | no | no | no | no | **yes** |
| Usable sample | 1999+ | 1999+ | 1999+ | 1999+ | **2003-10+** |
| Universe | 20 pairs | 20 pairs | 20 pairs | 20 pairs | 20 pairs |
