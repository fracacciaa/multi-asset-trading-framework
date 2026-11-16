# Rates Carry Research: Implementation Specification for a Coding Agent

**Version:** 1.0 | **Scope:** OIS swaps and government cash bonds | **Purpose:** research-quality signal engineering, reproducible backtesting, and paper-inspired variants.

> **Agent instruction:** Treat this document as an implementation specification, not evidence that a particular strategy earns alpha. Inspect the existing repository, reuse its date, market-data, curve, and portfolio conventions, and implement each variant independently. Do not silently substitute a par-rate slope for a repriced trade's holding-period P&L. Do not invent Bloomberg identifiers or historical data. Document every approximation, data gap, and assumption. Do not execute trades.

## 0. Deliverables and priority

Create a modular research pipeline with:

1. Market-data validation and normalized panels for OIS quotes, overnight fixings, government bond reference data, prices, coupons, and funding rates.
2. Date-aware OIS discount/forward curve bootstrap, with instrument calendars and conventions; an explicit approximate par-curve fallback for prototyping only.
3. A reusable position repricer for receiver/payer OIS swaps and long/short government bonds.
4. A scenario engine for **unchanged constant-maturity spot/zero curve**, **unchanged instantaneous forward curve**, and **realized future curve**.
5. Separate signal series: `ois_roll_proxy`, `ois_static_curve_total`, `ois_static_curve_pure_carry`, `ois_static_curve_roll`, `ois_forward_baseline_excess`, `gov_bond_static_total`, `gov_bond_funded_carry`, `gov_bond_roll`, `gov_bond_forward_predictor`, `cross_sectional_carry`, `curve_relative_carry`, `carry_to_risk`, `carry_zscore`, and `carry_momentum_combo`.
6. A reproducible backtester with portfolio DV01 sizing, local-currency vs USD reporting, trading costs, historical-only normalization, and walk-forward validation.
7. Tests, diagnostics, plots, data dictionary, and a concise experiment report for each signal.

**Build order:** Phase A: OIS curve validation and slope proxy; Phase B: full OIS repricing; Phase C: curve and cross-country portfolios; Phase D: cash government bonds; Phase E: forward-rate predictive regressions and combined signals.

---

## 1. Data inventory and validation

### 1.1 Available OIS inputs

The project is intended to use daily Bloomberg OIS **par fixed rates** for developed and emerging markets, with requested tenor coverage 1M–12M, 1Y–10Y, 12Y, 15Y, 20Y, 25Y, 30Y, subject to availability. User-supplied Bloomberg ticker families include US SOFR (`USOSFR...`), EUR €STR (`EESWE...`), GBP (`BPSWS...`), JPY (`JYSO...`), CHF SARON (`SFSNT...`), CAD (`CDSO...`), AUD (`ADSO...`), NZD (`NDSO...`), SEK (`SKSWTN...`), NOK (`NKS...`), plus selected EM currencies. **Do not assume every inferred ticker is valid.** Verify source, quote conventions, effective dates, benchmark transitions, and history per series.

Input schema `ois_quotes`:

| column | type | notes |
|---|---|---|
| `date` | date | observation as-of date and market timezone |
| `currency` | string | ISO currency |
| `index` | string | SOFR, €STR, SONIA, SARON, etc. |
| `tenor` | string | e.g. 1M, 5Y |
| `maturity_years_approx` | float | display/interpolation only, not contractual date |
| `par_rate` | float | **decimal**: 0.04 means 4% |
| `bbg_security` | string | exact verified Bloomberg identifier |
| `quote_source` | string | BGN, BGNL, CMPN, etc. |
| `quality_flag` | string | missing/stale/outlier/source_change |

Additional inputs: overnight index fixings, holiday calendars, business-day conventions, day-count, settlement lags, fixed-leg payment frequency, floating-leg compounding/observation shift, and any historical index transition. Prefer market conventions as of each date. Capture bid/ask quotes and collateral/funding assumptions when available.

### 1.2 Cash-bond inputs

For each **specific bond identifier** (not issuer equity ticker): `security_id`, `currency`, `issuer`, `country`, `coupon_rate`, `coupon_frequency`, `issue_date`, `maturity_date`, `settlement_calendar`, `day_count`, `redemption_terms`, `callability`, `clean_price_bid`, `clean_price_ask`, `clean_price_mid`, `accrued_interest`, `dirty_price`, `yield`, `modified_duration`, `convexity`, `amount_outstanding`, `repo_rate`, `repo_specialness`, and `government_zero_curve_id`. Coupon and principal cash-flow schedules are essential for exact valuation.

### 1.3 Data gates

- Distinguish Bloomberg **par OIS quotes** from zero rates, overnight fixings, forward rates, swap rates referencing IBOR, and cash yields.
- Reject impossible tenors, duplicates, broken dates, and rates whose units are ambiguous. Never mix percentages and decimals.
- Do not forward-fill across missing market days for signal estimation without an explicit, tested rule. Do not treat an unchanged evaluated quote as an executed price.
- Historical signal at time `t` must only use data published by the decision cutoff at `t`.
- Track market holidays and nonsynchronous closing times across currencies.
- Log each curve's fit error, interpolation method, extrapolation flags, negative forwards (not automatically invalid), and coverage.
- For cash bonds, reconstruct historical eligible securities; do not backtest only today's surviving issues.

---

## 2. Units, signs, and risk normalization

**Canonical units:** rates/spreads as decimals, P&L in currency units per notional, DV01 in currency units per **1 bp** move, time as actual year fractions. Let `bp = 1e-4`.

Define a **receiver** OIS swap as receive fixed / pay compounded overnight. Define positive rate DV01 as the *gain* to a receiver when the par rate falls by 1 bp:

\[
DV01_{rec} = -10^{-4}\frac{\partial V_{rec}}{\partial K_{mkt}}>0.
\]

At inception, an at-market receiver swap with fixed strike `K0` has `V0 ≈ 0`. Its approximate mark-to-market when the equivalent remaining par rate moves to `K1` is:

\[
V_{rec}\approx N A(T) (K_0-K_1),
\]

where `A(T)` is a discounted fixed-leg annuity in years. Payer P&L is the negative, ignoring nonlinearities and asymmetries.

**DV01-normalized return:**

\[
R^{DV01}_{t,h}=\frac{\mathrm{PnL}_{t,h}}{DV01_t},
\]

which has units of **basis points of rate-equivalent move**, not a dimensionless percentage return. If DV01 is per bp, a $10,000 P&L on $1,000/bp DV01 is +10 bp-equivalent. Do not call this a percentage return or annualized yield.

For actual portfolio performance, allocate a fixed risk budget (`target_dv01`) and compute currency P&L and return on explicitly defined capital/margin/risk budget. Compare signals at identical horizon and risk basis.

---

## 3. OIS curve construction

### 3.1 Production method: bootstrap OIS discount factors

For each date and currency, build a collateral-consistent OIS curve `P_t(u)` using contractual fixed and overnight compounded floating legs. Short OIS instruments can be used to solve sequentially for discount factors; long maturities require solving par-pricing equations with the correct coupon schedule and compounding conventions. Interpolate **log discount factors** or instantaneous forwards using a documented method; preserve monotone time and positive discount factors where appropriate, while allowing negative yields.

A stylized single-curve par OIS relationship (simplified, no payment lag or lookback) is:

\[
K_t(T) \sum_{j=1}^{n}\alpha_j P_t(T_j) = 1-P_t(T).
\]

This is an approximation to the exact dated OIS swap valuation, not a universal convention-independent formula.

**Diagnostics:** reprice each input instrument to within a chosen tolerance, e.g. 0.1–0.5 bp; fail loudly if no consistent curve can be built. Do not extrapolate 30Y carry from a 5Y-only curve without labeling it.

### 3.2 Research fallback: par-rate interpolation

If only par OIS rates are available and full conventions are missing, interpolate `K_t(T)` using PCHIP or a smooth penalized spline. **Do not treat par rates as zero yields.** The par-rate slope proxy is a separate low-fidelity signal, not exact carry.

---

## 4. Scenario engine: three distinct assumptions

Define horizon `h` in years or actual dated calendar interval, typically 1M/3M/6M. At `t`, enter a new at-market swap at strike `K_t(T)` and fixed initial notional `N`.

**Scenario S (static constant-maturity curve):** At `t+h`, the curve *as a function of time-to-maturity* has the same zero/discount shape as at `t`. Reconstruct the remaining-maturity discount curve from today's zero rates (not by copying absolute-maturity discount factors). This is a hypothetical unchanged yield-curve scenario; it generally differs from forwards and embeds expected roll-down.

**Scenario F (forward evolution / no-arbitrage benchmark):** At `t+h`, conditional discount factors satisfy

\[
P^{F}_{t+h}(u) = P_t(h+u)/P_t(h),
\]

under a deterministic forward realization with consistent collateral discounting. Expected *discounted gains* of a fairly priced self-financing position under the pricing measure are zero, once interim cash flows, funding and collateral remuneration are treated consistently. Do not conflate this baseline with a physical-measure return forecast.

**Scenario R (realized):** Revalue at `t+h` using the actual observed future curve, realized overnight fixings, settlement and cash flows. Use this **only for targets and ex-post P&L**, never for signal construction at `t`.

**Important:** Repricing a swap by simply replacing `T` with `T-h` while leaving its fixed coupon and payment schedule unchanged is generally incorrect. Keep original trade dates and cash flows; use the scenario curve to value its remaining legs.

---

## 5. OIS carry variants

### 5.1 OIS-0: par-curve roll-down proxy (fast baseline)

For receiver direction and `T>h`:

\[
\mathrm{RollRate}^{rec}_{t,h}(T) = K_t(T)-K_t(T-h).
\]

Approximate notional P&L:

\[
\mathrm{RollPnL}^{rec}_{t,h}(T)\approx N A_t(T-h)[K_t(T)-K_t(T-h)].
\]

For payer, multiply by -1. Interpolate `K_t(T-h)` using the **same date's** curve. This proxy ignores fixed/floating accrual, changing annuity, coupons, conventions and nonlinear repricing. Treat it as a curve-slope feature, **not** complete carry and roll.

Example: 5Y=4.00%, 4.75Y=3.95%, horizon 3M => receiver roll-rate proxy +5 bp. With annuity 4.4, roll-only P&L ≈ +22 bp of notional (before all other effects). Validate against full repricing.

### 5.2 OIS-1: full static-curve carry + roll (preferred primary signal)

Enter the swap at `t`, age it to `t+h`, value remaining legs under Scenario S, and include realized or **scenario-projected** fixed/floating settlements occurring during the horizon. At time `t`, the future overnight fixings are unknown; for the *signal* use a clearly specified projected fixing path (e.g. forwards implied by `P_t`). For ex-post P&L use actual fixings.

\[
C^{S}_{t,h}(T)=V_{t+h}^{S}(\text{aged original trade})+\sum_{u\in(t,t+h]} CF_u^{S}\,G(u,t+h)-V_t,
\]

where `G` accumulates interim cash flows to the horizon under a specified collateral/cash reinvestment convention. Report gross P&L, DV01-normalized P&L, and estimated costs separately.

**Sign:** positive means the receiver earns money in Scenario S. Compute the payer explicitly as the opposite position, including any asymmetry in trading costs.

### 5.3 OIS-2: decomposition into pure carry and roll

There is no uniquely model-free decomposition: the definition depends on the chosen baseline. Adopt one consistent *pathwise* attribution and document it:

- `pure_carry`: projected net fixed-minus-floating cash flows, accrual and collateral/funding effects over the horizon, valued consistently under a **no-roll valuation** convention.
- `roll`: residual difference `total_static_curve_PnL - pure_carry` using the same valuation and cash-flow conventions.

An illustrative shortcut is `N*(K_T - expected_avg_overnight)*h` for projected net coupon carry. **Do not** add this shortcut to a full static-curve P&L that already includes it. Build unit tests proving `pure_carry + roll == total_static_curve_PnL` by construction.

### 5.4 OIS-3: forward-curve benchmark / expected return gap

Compute `PnL_S`, `PnL_F`, and their difference:

\[
\mathrm{ForwardGap}_{t,h}=\mathrm{PnL}_{S}-\mathrm{PnL}_{F}.
\]

Interpret as **static-curve-versus-forward-scenario divergence**, not guaranteed alpha. `PnL_F` may include collateral/funding earnings; compare discounted self-financing P&L to get the correct zero-gain benchmark. For each instrument report both scenarios and the baseline convention.

### 5.5 OIS-4: risk-adjusted carry

\[
\mathrm{CarryToRisk}_{t,h}=\frac{C^{S}_{t,h}-\widehat{TC}_{t,h}}{\widehat\sigma_{t,h}^{PnL}},
\]

where numerator and denominator are in identical monetary or DV01-normalized units, and volatility is estimated strictly from past **realized** holding-period P&L of the same instrument or comparable standardized receiver strategy. For overlapping 3M holding-period returns, use dependence-aware standard errors.

Alternative: `C^S / (DV01 × yield_vol_bp)` using forecast rate-change volatility, noting this ignores curve-shape exposures and convexity. For cross-currency comparisons, use a common risk budget and FX convention.

### 5.6 OIS-5: historical carry z-score

\[
Z^C_{c,T,t}=\frac{C_{c,T,t}-\mu_{c,T,t}^{past}}{\sigma_{c,T,t}^{past}}.
\]

Compute rolling mean/std from past observations only (e.g. trailing 3 years; minimum 252 valid dates). A high z-score means unusually high *relative to that instrument's own history*, not necessarily the highest expected risk-adjusted return across markets. Preserve the raw signal alongside the standardized one.

---

## 6. Cash government bond carry variants

### 6.1 Exact bond repricing

For a bond purchased at `t`, calculate dirty purchase price `P_dirty,t`, coupon and principal cash flows during `(t,t+h]`, and hypothetical dirty sale price `P_dirty,t+h^S` obtained by discounting remaining contractual cash flows using a static government **zero** curve (plus a clearly defined security-specific liquidity/specialness spread if modeled).

\[
\mathrm{TR}_{t,h}^{S}=\frac{P^{S}_{dirty,t+h}+\sum CF_u^{S}G(u,t+h)-P_{dirty,t}}{P_{dirty,t}}.
\]

For a fully funded position, compare this total return with a cash benchmark separately. For a repo-financed bond, subtract the **actual modeled repo financing cost** and include haircuts, specials and settlement assumptions:

\[
\mathrm{ExcessPnL}^{S}=P^{S}_{dirty,t+h}+\sum CF_uG-P_{dirty,t}-\mathrm{FinancingCost}.
\]

Do not treat `yield - OIS` as a complete funding or carry estimate. Repo can diverge from OIS, particularly for special collateral.

### 6.2 Approximate bond carry and roll

For a par-like bond and small horizon:

\[
\mathrm{ExcessReturn}_{t,h}^{S}\approx(y_t(T)-r_{fund,t})h+D_{mod,t}[y_t(T)-y_t(T-h)]
\]

with rates in decimals and time in years. This is only a rough duration approximation. It can be materially wrong for off-par, high-coupon, callable, or repo-special bonds. Use it as a feature and cross-check, not as the accounting engine.

### 6.3 Government bond maturity-relative carry

For each country, compute the static-curve carry-and-roll excess return of standardized 2Y/5Y/10Y/20Y benchmark bonds. Rank maturity buckets within country after risk scaling. A long-high-carry / short-low-carry portfolio should have a specified duration hedge, funding convention, and turnover constraint. Rebalance with realistic bid/ask and on-the-run/off-the-run transitions.

### 6.4 Bond-versus-OIS relative carry

For matched currency and approximate DV01/maturity, compare `government_bond_excess_PnL` against `OIS_swap_PnL` after hedging residual rates exposure. This is a **sovereign swap-spread/basis trade** with repo, collateral, credit, liquidity and balance-sheet risks. Never subtract raw carry scores without normalizing risk and currency P&L.

---

## 7. Curve-relative and cross-sectional carry portfolios

### 7.1 Within-currency curve trades

Given instruments `j` with position weights `w_j`, optimize or set weights subject to:

\[
\sum_j w_j DV01_j = 0
\]

for a first-order parallel-shift-neutral trade. For a 2s5s10s butterfly, also impose an appropriate **slope-factor neutrality** or choose predeclared DV01 wing weights; parallel DV01 neutrality alone does **not** make the butterfly slope neutral. Report key-rate DV01 and factor exposures. Portfolio carry is:

\[
C_{portfolio}=\sum_j w_j C_j-\widehat{TC}_{portfolio}.
\]

Test 2s5s, 5s10s, 2s5s10s and 5s10s30s using available tenors. Exclude instruments where interpolation/extrapolation dominates the signal.

### 7.2 Cross-country OIS carry

At each date, standardize by tenor and risk. Candidate ranking universes: (a) G10 within 5Y, (b) DM vs EM separately, (c) country-tenor pairs. Construct dollar-neutral or DV01-neutral long/short positions; be explicit whether FX is hedged. Local rates P&L in foreign currency is **not** automatically USD return. For emerging markets, account for market access, collateral, funding, fixing conventions and transaction costs.

### 7.3 Time-series vs cross-sectional mapping

- **Time-series:** take receiver if `C^S / risk > threshold`, payer if below negative threshold, otherwise flat. Beware unconditional receiver bias in downward rate regimes.
- **Cross-sectional:** rank instruments, receive top quantile and pay bottom quantile, with country/tenor/sector constraints and risk balancing.
- **Hybrid:** first residualize carry against curve level, slope, expected policy moves or historical volatility, then rank residuals. This is a new research adaptation, not necessarily a paper's exact signal.

---

## 8. Paper-inspired specifications and what they actually support

**Citations are conceptual pointers, not claims of exact reproduction.** Obtain the full texts and any supplements before labeling a strategy as a faithful replication. Maintain `paper_replication_status` in experiment metadata: `inspired`, `approximate`, `verified_replication`.

### P1. Koijen, Moskowitz, Pedersen & Vrugt, *Carry* (Journal of Financial Economics, 2018)

**Reference:** Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H., and Vrugt, E. B. (2018), *Carry*, Journal of Financial Economics 127(2), 197–225. DOI: `10.1016/j.jfineco.2017.11.002`.

**Core idea:** Define carry consistently as an expected return under an assumption that market conditions, including prices and yield curves as applicable, do not change. Compare carry across asset classes and investigate its predictive and risk properties.

**Implement:** For government bonds, calculate full static-curve holding-period total/excess returns (Sections 6.1–6.3); for swaps, use the explicit OIS static-curve scenario (Section 5.2) as an **extension**. Normalize across contracts and test time-series and cross-sectional forecasting. The paper's exact asset definitions and return conventions must be checked before claiming direct replication.

**Tests:** raw carry, carry/risk, within-country maturity rank, cross-country rank, interactions with market stress and volatility.

### P2. *Carry Investing on the Yield Curve* (Financial Analysts Journal, 2019)

**Reference:** *Carry Investing on the Yield Curve*, Financial Analysts Journal (2019), DOI: `10.1080/0015198X.2019.1628552`. Verify authors and complete published specifications from full text before replication.

**Core idea:** Exploit differences in carry across maturity segments of sovereign yield curves, rather than only selecting countries with high yield levels.

**Implement:** Create maturity-bucket government-bond carry series (Section 6.3); compare high vs low carry buckets within each country; test global diversification, duration neutrality, and realized transaction costs. **OIS maturity-bucket portfolios are a research adaptation**, not necessarily the published implementation.

**Tests:** 2Y/5Y/10Y/20Y relative ranks, long-short maturity portfolios, alternative duration/key-rate neutrality, cross-country aggregation, curve-level and slope controls.

### P3. Cochrane & Piazzesi, *Bond Risk Premia* (American Economic Review, 2005)

**Reference:** Cochrane, J. H., and Piazzesi, M. (2005), *Bond Risk Premia*, American Economic Review 95(1), 138–160. DOI: `10.1257/0002828053828581`.

**Core idea:** Predict one-year Treasury excess returns using a regression on the one-year spot rate and a set of one-year forward rates; estimated coefficients exhibit a tent-shaped pattern. **This is a forward-rate return-prediction model, not the same thing as static-curve carry.**

**Implement:** From a properly bootstrapped **government Treasury zero-coupon** curve (or an explicitly labeled OIS adaptation), derive one-year log forward rates:

\[
f_t^{(n\to n+1)} = -\log P_t(n+1)+\log P_t(n),
\]

and one-year log bond excess return for an `n`-year zero-coupon bond:

\[
rx_{t+1}^{(n)}=\log P_{t+1}(n-1)-\log P_t(n)-r_t^{(1)}.
\]

Here `P_t(n)` denotes price of a unit-payoff zero-coupon bond maturing in `n` years and `r_t^(1)=-log P_t(1)` for annual compounding in log-price units. Regress average future one-year excess return across maturities on the current 1Y yield and forward-rate vector, using **training-only** coefficients; test on held-out dates. Then test the factor against carry and slope to assess incremental prediction.

**Caution:** Par OIS quotes are not zero-coupon Treasury prices. An OIS implementation is a *new adaptation*, and no-arbitrage swap discount curves do not directly reproduce Treasury bond returns.

### P4. Baz, Granger, Harvey, Le Roux & Rattray, *Dissecting Investment Strategies in the Cross Section and Time Series* (2015)

**Reference:** Baz, J., Granger, N., Harvey, C. R., Le Roux, N., and Rattray, S. (2015), *Dissecting Investment Strategies in the Cross Section and Time Series* (working paper).

**Core idea:** Compare time-series and cross-sectional implementations of systematic signals across asset classes and study their different exposure properties.

**Implement:** Apply identical underlying `carry_to_risk` features to time-series and cross-sectional mappings (Section 7.3); compare performance, exposure concentration, diversification and turnover. This is a **paper-inspired comparison**, not a claim that the exact OIS carry formula below appears in that paper.

### P5. Asness, Moskowitz & Pedersen, *Value and Momentum Everywhere* (Journal of Finance, 2013)

**Reference:** Asness, C. S., Moskowitz, T. J., and Pedersen, L. H. (2013), *Value and Momentum Everywhere*, Journal of Finance 68(3), 929–985. DOI: `10.1111/jofi.12021`.

**Core idea:** Cross-asset value and momentum; useful as a methodological complement rather than a rates-carry construction paper.

**Implement adaptation:** Calculate trailing rates returns / swap P&L momentum (1M, 3M, 12M minus most recent month); combine with carry via fixed-weight z-scores, conditional carry exposure, and double sorts. Attribute whether carry adds incremental information after momentum.

### Optional related literature

- Campbell & Shiller (1991), *Yield Spreads and Interest Rate Movements: A Bird's Eye View*, Review of Economic Studies: expectations-hypothesis benchmarks.
- Ilmanen (1995), *Time-Varying Expected Returns in International Bond Markets*, Journal of Finance: international bond return prediction.
- Martin (2022), *The Credit Spread Curve: Fundamental Concepts, Fitting, Par-Adjusted Spread, and Expected Return* (arXiv:2201.01330): **credit**, not sovereign rates; useful when implementing issuer credit carry.

---

## 9. Signal normalization and research design

For each `date × currency × tenor × horizon × direction` retain **raw components**, not only final score:

`date, currency, index, tenor, horizon, direction, entry_par_rate, entry_dv01, curve_id, curve_fit_error_bp, expected_float_cf, fixed_cf, pure_carry_pnl, roll_pnl, static_total_pnl, forward_scenario_pnl, forward_gap_pnl, funding_pnl, transaction_cost_pnl, carry_dv01_bp, carry_to_risk, carry_zscore, signal_valid, invalid_reason`.

Suggested horizons: **1M, 3M, 6M**. Suggested benchmark tenors: **2Y, 5Y, 10Y**, then 1Y/3Y/7Y/20Y/30Y as data allows. Daily compute, weekly or monthly rebalance. Initial training period >=3 years where available; never require a history that does not exist for recently introduced OIS indices.

Normalize cross-sectionally **within tenor and market group** before combining markets, or explicitly include tenor/currency controls. Avoid allowing high-inflation EM rate levels to dominate a global score. Store separate DM and EM experiments.

Risk estimates: trailing realized swap-rate changes; EWMA or shrinkage covariance across key-rate tenors and countries; optionally factor covariance from PCA. Compare simple diagonal vol targeting against full covariance risk targeting. Cap gross DV01, per-currency DV01, tenor key-rate DV01, and leverage.

**Transaction costs:** swap bid/ask, brokerage, collateral/funding, market impact; cash bonds bid/ask, repo, settlement, and roll between benchmark securities. Costs should scale with turnover and market liquidity, not be assumed zero.

---

## 10. Backtesting: avoid common false positives

1. At signal time `t`, use only data available before trade decision; enter at next executable quote/time, not the same close used to construct the signal unless execution assumptions justify it.
2. Use realized swap cash flows and mark-to-market **with the original trade strike and maturity**. Do not equate a 5Y swap entered at `t` with a newly issued 5Y swap at `t+1`.
3. Mark-to-market using consistent discounting, compounding, accrued and collateral conventions.
4. Explicitly define holding strategy: hold to `h`, roll into constant-maturity swaps each rebalance, or maintain overlapping cohorts. These have different realized P&Ls.
5. If hedging FX, model FX forwards and their cost; otherwise report both local-currency and USD-unhedged P&L.
6. Avoid survivorship bias, interpolation look-ahead, macro data revision bias, and cross-market asynchrony.
7. Split by time (expanding/rolling walk-forward), by currency, and by regime; stress 2008, 2020, 2022 and relevant local market disruptions if covered.
8. Use Newey–West or block bootstrap inference for overlapping horizons; report multiple-testing adjustments when screening many variations.
9. Evaluate gross and net Sharpe, annualized return on defined capital, P&L per DV01, maximum drawdown, turnover, hit rate, tail risk, beta to level/slope/curvature, correlation to momentum, and IC to future returns.
10. Check whether the carry signal is simply a proxy for short rates, curve slope, receiver duration beta, or volatility selling.

---

## 11. Minimum unit and integration tests

**Pricing:** input OIS instruments reprice at par; swap PV at inception approximately zero; receiver/payer P&L antisymmetry (before costs); discount factor consistency; exact cash-flow dates; leap years/holiday calendars; negative-rate cases.

**Scenario:** a flat curve has near-zero **rate roll proxy** but may still have nonzero swap cash-flow/funding components; S and F scenarios generally differ; scenario F discount-factor identity holds numerically; no use of future realized fixings in a signal.

**Accounting:** pure carry + roll = total static P&L under the chosen attribution; cash-flow timing correct; coupons not double counted; clean vs dirty bond prices handled; bond roll and pull-to-par distinguished; repo financing correctly signed.

**Data:** missing observations flagged, not silently filled; `1Y` and `12M` duplicate tenors resolved without counting twice; currency basis and units verified; no invalid extrapolation; Bloomberg field histories checked separately from current BDP fields.

**Risk:** scaling notional by 2 doubles first-order P&L and DV01; dividing P&L by DV01 gives bp-equivalent; hedge weights neutralize intended risk exposures; cross-currency risk units are consistent.

**No-lookahead:** shifting future realized curve must not change historical signal values; rolling z-score excludes current target returns; regression coefficients for an evaluation period trained strictly on earlier dates.

---

## 12. Implementation pseudocode

```python
for date in trading_dates:
    for currency in currencies:
        quotes = get_validated_ois_quotes(date, currency)
        if not sufficient_coverage(quotes):
            log_missing_curve(date, currency)
            continue

        curve = bootstrap_ois_curve(quotes, conventions[currency])
        assert curve.input_repricing_error_bp <= tolerance_bp

        for tenor in tradable_tenors(curve):
            for horizon in ["1M", "3M", "6M"]:
                trade = create_par_ois_trade(
                    asof=date, tenor=tenor, direction="receive_fixed",
                    notional=notional_for_target_dv01(curve, tenor, target_dv01),
                    conventions=conventions[currency]
                )
                # No future data: project interim floating fixings from date-t forwards.
                projected_cf = project_cashflows(trade, curve, horizon)
                aged = age_trade(trade, horizon)

                curve_static = static_constant_maturity_curve(curve, horizon)
                curve_forward = forward_evolved_curve(curve, horizon)

                pnl_static = reprice_and_accumulate(
                    aged, curve_static, projected_cf, funding_model
                ) - trade.initial_pv
                pnl_forward = reprice_and_accumulate(
                    aged, curve_forward, projected_cf, funding_model
                ) - trade.initial_pv

                roll_proxy = interpolate_par_rate(curve, tenor) - \
                             interpolate_par_rate(curve, tenor_minus_horizon(tenor, horizon))
                pure_carry, roll = attribute_static_pnl(
                    trade, curve, curve_static, projected_cf, pnl_static
                )
                assert abs(pure_carry + roll - pnl_static) < pnl_tolerance

                save_signal(date, currency, tenor, horizon,
                    pnl_static=pnl_static, pnl_forward=pnl_forward,
                    pure_carry=pure_carry, roll=roll,
                    roll_proxy=roll_proxy, dv01=trade.initial_dv01)

# Separately, once future dates are observed, calculate realized trade P&L.
# Never use realized future rates/fixings in signal generation.
# After data is assembled, compute trailing-only normalizations and backtest.
```

The `attribute_static_pnl` function must implement a **documented attribution convention**; it is not uniquely defined by the observed par curve.

---

## 13. Suggested repository layout

```text
rates_carry/
  config/markets.yaml
  data/validate_ois.py
  data/validate_bonds.py
  curves/ois_bootstrap.py
  curves/gov_zero_curve.py
  instruments/ois_swap.py
  instruments/gov_bond.py
  scenarios/static_curve.py
  scenarios/forward_curve.py
  signals/ois_carry.py
  signals/bond_carry.py
  signals/curve_relative.py
  signals/forward_rate_predictor.py
  signals/normalize.py
  portfolio/construct.py
  portfolio/risk.py
  backtest/engine.py
  backtest/costs.py
  evaluation/diagnostics.py
  tests/test_curve_repricing.py
  tests/test_cashflow_accounting.py
  tests/test_no_lookahead.py
  tests/test_risk_units.py
  experiments/experiment_config.yaml
  README.md
```

Suggested APIs:

```python
build_ois_curve(asof, currency, par_quotes, conventions) -> DiscountCurve
create_ois_trade(asof, currency, tenor, fixed_rate, direction, notional) -> OISTrade
value_ois(trade, asof, discount_curve, fixings, collateral_model) -> Valuation
scenario_pnl(trade, horizon, scenario, projected_fixings) -> PnLBreakdown
build_gov_curve(asof, country, bond_quotes, conventions) -> ZeroCurve
bond_holding_period_pnl(bond, asof, horizon, scenario, funding) -> PnLBreakdown
compute_carry_panel(config) -> DataFrame
construct_portfolio(signals, risk_model, constraints) -> Portfolio
backtest_portfolio(portfolio, realized_data, costs) -> BacktestResult
```

---

## 14. Acceptance criteria and initial experiments

**First acceptance milestone:** one market (USD SOFR), one date, 2Y/5Y/10Y, 1M/3M horizons. Input quotes reprice correctly, each at-market swap PV ≈ 0, Scenario S/F differ sensibly, receiver/payer signs are correct, and attribution reconciles exactly.

**Second milestone:** historical USD and EUR panels with no look-ahead, plotted carry/roll time series, static vs forward decomposition, and realized P&L comparisons.

**Third milestone:** cross-market panel and DV01-neutral 2s5s10s trades, gross/net results, realistic transaction costs, risk-factor exposures and rolling stability.

**Fourth milestone:** government bond exact dirty-price and coupon engine, maturity-relative carry portfolios, bond/OIS basis research.

**Research matrix:**

| ID | Instrument | Signal | Horizon | Strategy |
|---|---|---|---|---|
| E01 | OIS | Par-curve roll proxy | 1M/3M | Time-series |
| E02 | OIS | Full static-curve total | 1M/3M | Time-series |
| E03 | OIS | Static carry / forecast risk | 3M | Cross-sectional |
| E04 | OIS | Carry z-score | 3M | Time-series |
| E05 | OIS | Forward-scenario gap | 3M | Time-series |
| E06 | OIS | 2s5s10s carry | 3M | Curve RV |
| E07 | OIS | Cross-country carry rank | 3M | Market neutral |
| E08 | Govt bond | Exact funded static carry+roll | 3M | Time-series |
| E09 | Govt bond | Maturity-relative carry | 3M | Within-country RV |
| E10 | Govt zero curve | Cochrane–Piazzesi forward predictor | 1Y | Predictive regression |
| E11 | OIS/Govt | Carry + momentum | 3M | Combined |
| E12 | Govt vs OIS | Matched-DV01 relative carry | 3M | Basis RV |

**Report for every experiment:** data dates, instruments, coverage, missingness, market conventions, signal definition, position sizing, costs, out-of-sample splits, results, and known limitations. Never present in-sample fit as proof of tradable alpha.

---

## 15. Reference links

- Koijen et al., *Carry* (2018): https://doi.org/10.1016/j.jfineco.2017.11.002
- *Carry Investing on the Yield Curve* (2019): https://doi.org/10.1080/0015198X.2019.1628552
- Cochrane & Piazzesi, *Bond Risk Premia* (2005): https://doi.org/10.1257/0002828053828581
- Asness, Moskowitz & Pedersen, *Value and Momentum Everywhere* (2013): https://doi.org/10.1111/jofi.12021
- Baz et al., *Dissecting Investment Strategies in the Cross Section and Time Series* (2015): https://www.cmegroup.com/content/dam/cmegroup/education/files/dissecting-investment-strategies-in-the-cross-section-and-time-series.pdf
- Martin, *The Credit Spread Curve* (2022): https://arxiv.org/abs/2201.01330

**End of specification.**
