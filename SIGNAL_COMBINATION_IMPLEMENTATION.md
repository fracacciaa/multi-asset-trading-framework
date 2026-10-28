# Agent Implementation Specification: Multi-Asset Signal Combination and Risk-Aware Portfolio Construction

**Status:** Implementation-ready research specification  
**Purpose:** Build, evaluate, and compare modular and joint-ML methods for combining systematic trading signals across instruments and asset classes.  
**Scope:** FX, rates, equity indices, commodities, and extensible credit futures/proxies.  
**Primary principle:** Separate forecasting expected returns from estimating risk and constructing portfolios; benchmark against a joint model trained directly for portfolio utility.

> This document is a build specification, not a claim that any strategy will be profitable. An autonomous coding agent must inspect the repository, implement incrementally, run tests, and report what is implemented versus blocked by missing data. Never fabricate results, data, paper findings, or backtests.

---

## 0. Instructions for the coding agent

1. **Inspect before modifying:** Inventory the repository's package manager, modules, data schemas, available price and signal history, tests, configuration, and execution conventions. Reuse existing components and do not overwrite unrelated work.
2. **Produce a short implementation plan:** Map the modules below to actual repository paths; identify missing dependencies and data. Use reasonable documented defaults rather than blocking for nonessential preferences.
3. **Build vertical slices:** First a reproducible baseline on a tiny synthetic panel, then actual data adapters, risk forecasts, portfolio optimizers, ML challengers, and walk-forward experiments.
4. **Run commands and tests:** Record exact commands, dependency versions, seeds, git diff summary, and test outcomes. If no market data exists, deliver a fully tested synthetic-data pipeline and explicit real-data blockers, not imaginary performance statistics.
5. **No leakage:** Every forecast, normalization parameter, model selection, risk estimate, and portfolio decision at date `t` must depend only on information available when the trade would be placed. Treat feature publication lags and executable trade timestamps explicitly.
6. **Do not assume a particular vendor:** Support Bloomberg or existing repository feeds via adapters; never embed credentials or depend on proprietary data being available in CI.
7. **Never place live orders:** Research/backtesting only. No brokerage or order-routing integration without separate authorization.
8. **Produce a final report:** Files changed, architecture, experiment manifest, tests, data coverage, sample counts, caveats, and recommended next steps.

### Acceptance checklist

- [ ] Reproducible data pipeline with adjusted returns, contract roll logic, metadata, and as-of timestamps.
- [ ] Feature and signal normalization without look-ahead bias.
- [ ] Same-instrument blending: equal-weight, Sharpe heuristic, ridge, elastic net, and at least one nonlinear ML method.
- [ ] Instrument-level and sleeve-level signal diagnostics.
- [ ] Cross-asset covariance forecasts: rolling, EWMA, shrinkage, and two-timescale EWMA.
- [ ] PSD checks, numerical conditioning, and risk attribution.
- [ ] Portfolio baselines: inverse volatility, risk parity or equal-risk contribution, and constrained mean-variance with costs.
- [ ] Joint portfolio-level ML challenger if sufficient data; otherwise tested scaffold with explicit block reason.
- [ ] Nested walk-forward evaluation with realistic costs and robust baselines.
- [ ] Unit/integration tests, experiment configs, deterministic output artifacts, and research report.

---

## 1. Problem definition and notation

Let `i = 1,...,N` index tradable instruments, `k = 1,...,K_i` signals for instrument `i`, and `t` decision dates. A raw signal `x[i,k,t]` can be momentum, carry, value, mean reversion, liquidity, or macro state. Signals are features, not necessarily return forecasts. Let `r[i,t+1]` be the next holding-period *tradable excess return per unit of standardized exposure* after correctly accounting for contract multipliers, collateral, funding, currency, and rolls. Define `mu_hat[i,t] = E_t[r[i,t+1]]`, `Sigma_hat[t] = Cov_t(r[t+1])`, and `w[t]` as portfolio exposures in the same return units.

**Two different combination tasks:**

- **Same underlying return:** Several signals forecast the same `r[i,t+1]`. Combine them into *one* forecast per instrument, controlling signal correlation, horizon alignment, estimation uncertainty, and model complexity.
- **Different underlying returns:** Forecasts for EURUSD, Treasury futures, gold, equity futures, etc. form a vector `mu_hat[t]`. Allocate positions jointly using cross-asset covariance, exposure constraints, and costs. Do not collapse them into one scalar forecast.

**Third, optional task:** A joint ML policy learns `w[t]` from the complete feature panel and risk inputs, directly optimizing a portfolio-level objective. Compare it with the modular pipeline on identical data and constraints.

### Time alignment

Specify for each feature: `observation_time`, `published_time`, `available_time`, `decision_time`, `execution_time`, and `return_end_time`. If forecasting horizon `h`, target is the executable forward return from the trade price after `decision_time` to `t+h`. For overlapping `h`-day labels, purge training/validation overlap and use an embargo as appropriate. Explicitly choose rebalance cadence (daily default) and forecast horizon (e.g., 1d, 5d, 21d); do not mix horizon-specific Sharpe statistics without conversion and alignment.

### Data contract

Canonical long-format columns:

`timestamp_utc, instrument_id, asset_class, currency, price, total_return_index_or_pnl, volume, contract_id, multiplier, fx_conversion, trading_calendar, available_at`.

Signal panel: `timestamp_utc, instrument_id, signal_name, raw_value, available_at, signal_horizon, source, version`.

Metadata: tick size, contract multiplier, quotation convention, FX orientation, point value, financing/funding rules, contract rolls, trading hours, minimum trade size, sector/country/duration/FX factor exposures, and estimated spread/commission/slippage.

Store data validation failures explicitly: duplicates, stale prints, missing data, invalid contract mappings, nonmonotone timestamps, unrealistic returns, stale carry inputs, and mismatched units. Do not forward-fill indefinitely. Establish configurable maximum staleness by field.

---

## 2. Signal engineering and normalization

### 2.1 Signal families

Implement pluggable feature generators with explicit economic sign and lag:

- **Momentum/trend:** 1m/3m/6m/12m lookbacks, optionally skip the most recent month; time-series and cross-sectional variants; moving-average and regression-slope variants. Exclude future observations.
- **Carry:** Use instrument-specific, executable definitions. FX forward-implied carry or interest differentials including basis/forward points; futures curve roll yield with correct tenor/annualization; rates rolldown and carry using curve and duration; equity index carry net of funding/dividends. Avoid assuming these are interchangeable.
- **Value/mean reversion:** Rolling deviation from economically justified anchors, standardized price/curve residuals, relative-value spreads, or error-correction features; make stationarity assumptions explicit.
- **Risk/liquidity/macroeconomic:** Realized volatility, drawdown, term structure, spread proxies, market breadth, inflation/growth surprises, and regime indicators, with release-date availability.

Do not force every family on every instrument. Implement feature availability masks and minimum-history rules. Preserve raw signals and transformed signals separately.

### 2.2 Standardization

Use only past data. Options:

- Time-series rolling z-score per instrument/signal: `z = (x - rolling_mean_past) / max(rolling_std_past, eps)`.
- Expanding z-score, robust median/MAD, or percentile/rank transform as challengers.
- Cross-sectional ranks on a date only when the eligible universe and data availability are known at that date.
- Clip extreme standardized signals using thresholds selected on training data; log clipping frequency.
- Volatility-scale forecast targets or signals *only when appropriate*, without inadvertently leaking the risk estimate into the predictive target.

Default: rolling robust or ordinary z-score with minimum 252 daily observations where history permits; parameterize windows and minimum periods. Avoid using a whole-sample scaler. Train all learned transforms inside each walk-forward training fold.

### 2.3 Signal evaluation before blending

For each `(instrument, signal, horizon)` and for pooled panels, report predictive slope, information coefficient (Pearson/Spearman as appropriate), out-of-sample `R^2`, hit rate, forecast calibration, autocorrelation, turnover, trading costs, signal correlation matrix, performance by regime, and naive signal-to-position Sharpe after common risk scaling. Use HAC/Newey-West or block bootstrap uncertainty estimates for overlapping labels and autocorrelated returns. Report effective sample size and multiple-testing caution.

---

## 3. Layer A: Blend signals that forecast the SAME instrument

Let `X[i,t] = [z[i,1,t],...,z[i,K,t]]`. Compare models predicting the same forward return `r[i,t+h]`.

### A0. Equal-weight baseline

`score[i,t] = sum_k a[k] * z[i,k,t] / sum_k abs(a[k])` with default `a[k]=1` after ensuring all signs represent bullish forecasts. Missing signals: renormalize over available signals if a minimum coverage threshold is met; otherwise abstain. A score is not a return forecast until calibrated on training data. For direct portfolio comparisons, map score to exposure using the same position scaling and risk budget as competitors.

### A1. Sharpe heuristic

Construct *standalone tradable signal sleeves* using fixed, comparable ex-ante volatility targeting and costs. Estimate trailing, **past-only**, annualized Sharpe `SR[k,t]` from their net returns. Candidate weights:

- Positive-only: `a[k,t] ∝ max(0, shrink(SR[k,t]))`, with normalization.
- Signed only if permitted by economic rationale and a robust, predeclared rule; otherwise negative Sharpe means weight zero, not automatically reversing a strategy.
- Shrink estimates toward zero or a common mean; impose caps, minimum history, and turnover penalties.

Important: Sharpe-proportional weights ignore *correlation between sleeves*. Add a correlation-aware challenger: `a ∝ (C_sleeve + alpha*I)^(-1) * s`, where `s` is an estimated vector of sleeve expected returns (or comparable risk-adjusted estimates), with constraints and shrinkage. Do not label this closed-form solution optimal without stating assumptions. Use out-of-sample sleeve returns for evaluation; training-window sleeve estimates are permissible only as-of the rebalance.

### A2. Linear and regularized regression

Predict `r[i,t+h]` from `X[i,t]` using OLS (diagnostic), ridge, elastic net, and optionally lasso. Compare:

- Per-instrument fits where history supports it.
- Pooled model across instruments with instrument/asset-class effects and appropriate return normalization.
- Partial pooling / hierarchical regularization to share information while permitting asset-class differences.

Choose penalties using **time-ordered inner validation** only. Use group-aware folds when panel assets overlap in time; prevent same-date leakage. Ridge is the preferred first regularized challenger when features are correlated. Coefficient sign restrictions may be tested but not assumed. Record coefficient stability and collinearity diagnostics.

### A3. Nonlinear ML

At least one of gradient-boosted trees (LightGBM/XGBoost if installed; sklearn HistGradientBoosting fallback), random forest, or a compact MLP. Inputs include standardized signals, their interactions learned by the model, and optionally *lagged* risk/regime features. Use bounded hyperparameter grids, early stopping based on past-only validation, reproducible seeds, and minimum-sample safeguards. Do not introduce a large neural network on a short history.

### A4. Stacking and forecast pooling

Optional second-stage meta-learner combines base-model forecasts. **Train the meta-learner only on out-of-fold, chronologically valid base predictions.** Never train the stacker on in-sample fitted predictions. Baselines: equal-average predictions, winsorized average, ridge stacker, nonnegative sum-to-one weights. Calibrate forecasts to comparable return units.

### A5. Regime-aware blending

Use only observable-at-decision-time regimes: volatility level, curve slope, liquidity, growth/inflation surprises, or a latent state estimated recursively on training history. Compare static coefficients versus gated/interaction models. Require sufficient observations per regime and report state instability. Avoid post-hoc regime labels constructed using future returns.

### Output

At each rebalance produce `mu_hat[i,t]`, forecast uncertainty (optional), signal attribution, availability mask, model version, training cutoff, and as-of timestamp. If using standardized forecast scores instead of expected returns, explicitly convert via past-only calibration before mean-variance optimization.

---

## 4. Layer B: Risk forecasting

### 4.1 Return conventions

Choose a common base currency and P&L unit. For futures, use price changes times contract multiplier divided by chosen capital/notional reference, with realistic roll and collateral assumptions. For FX, orient spot/forward returns consistently and account for funding/forward carry exactly once. For rates, track duration/DV01, convexity when relevant, and contract basis. Risk matrices must be computed from the *same* return convention used in the optimizer. Align holidays and trading calendars without introducing future prices.

### 4.2 Volatility and correlation candidates

Implement:

1. **Rolling sample covariance:** trailing window `L` with sample-count guards; baseline.
2. **Single-timescale EWMA covariance:** `S_t = lambda*S_(t-1) + (1-lambda)*r_(t-1)r_(t-1)^T`, with `lambda = exp(-ln(2)/half_life)` and lagged returns. Explicitly document the indexing and missing-data policy.
3. **Shrinkage:** shrink covariance toward diagonal, constant-correlation, or structured factor target. Use Ledoit-Wolf/OAS only with assumptions and training data compatible with the implementation; compare fixed shrinkage grid as a transparent baseline.
4. **Two-timescale EWMA:** estimate fast marginal volatilities and slow correlations, then reconstruct `Sigma_t = D_fast,t * C_slow,t * D_fast,t`. Initial research defaults: volatility half-life **14 trading days**, correlation half-life **126 trading days**; tune only within training folds. Build `C_slow` from a well-defined standardized-return process, normalize its diagonal to one, and document whether standardization uses lagged fast or slow vol forecasts. Avoid accidentally combining mismatched timestamps.
5. **Optional challengers:** factor covariance (equity beta, duration, USD, commodity, growth/inflation factors), GARCH for marginal volatility, DCC or other dynamic correlations if data warrants.

### 4.3 Numerical safeguards

Check symmetry, finite values, strictly positive marginal variances, and minimum eigenvalue. For an indefinite estimate, project onto the PSD cone (eigenvalue clipping or nearest-correlation procedure), re-normalize correlation diagonal to one, and add a small ridge if needed for inversion. Log adjustment magnitude, condition number, effective rank, and correlation bounds. Prefer convex optimization with regularization rather than a naked matrix inverse. Missing returns require an explicit pairwise/EM/shrinkage policy; naive pairwise deletion can create an indefinite covariance.

### 4.4 Evaluate covariance forecasts

Use one-step-ahead forecast evaluation: Gaussian negative log likelihood where applicable, QLIKE on marginal variance forecasts, realized portfolio-variance calibration, risk-target exceedances, and covariance forecast comparisons with a defensible realized covariance proxy. **Scalar QLIKE on variances is not a full covariance scoring rule.** Test crisis/stress subperiods and estimation stability. Compare actual portfolio outcomes, not just covariance-fit metrics.

### 4.5 Risk attribution

For portfolio `w`, `sigma_p = sqrt(w' Sigma w)` and Euler component risk contributions `RC_i = w_i (Sigma w)_i / sigma_p` (when `sigma_p > 0`). Aggregate by asset class, country, currency, duration, curve bucket, commodity sector, and macro factor. Add sleeve-level attribution by preserving the hypothetical positions produced by each signal sleeve before netting. Warn that overlapping sleeves are not independent: use covariance-aware attribution and clearly specify allocation conventions for cross terms.

---

## 5. Layer B: Cross-asset portfolio construction

Given calibrated `mu_hat[t]` and `Sigma_hat[t]`, produce portfolio positions with identical constraints across comparable models.

### P0. Equal-weight / equal-risk benchmark

Equal notional is generally **not** equal risk across asset classes. Include equal-volatility-risk exposures: `raw_w[i] ∝ sign(mu_hat[i]) / sigma_hat[i]`, optionally weighted by forecast strength; normalize to portfolio volatility target using full covariance. Clearly separate equal-weight forecasts from equal-weight instrument positions.

### P1. Risk parity / equal-risk contribution

For long-only risk budgets or sign-fixed directional portfolios, solve for target contributions under `Sigma_hat`. Risk parity alone does not choose direction: take sign from the forecasts or build a long-only sleeve with clear constraints. Risk contribution may be negative for hedging exposures; document limitations of naive equal-risk-contribution on long-short portfolios.

### P2. Constrained mean-variance with costs

Solve, at each date:

`maximize_w mu_hat' w - (lambda/2) * w' Sigma_hat w - kappa * TC(w, w_prev)`

Subject to risk target/limit, gross/net leverage, instrument position caps, liquidity and ADV caps, class/country/currency/duration/factor limits, and any market-specific restrictions. Use cvxpy if available and deterministic fallback or explicit missing-dependency error. Transaction cost may combine spread, fees, linear slippage, and nonlinear market-impact approximations; calibrate from observed data or clearly label assumptions. Portfolio risk target (e.g., 10% annualized) must be a configurable research parameter, not a universal constant. Use the correct annualization based on holding/rebalance cadence.

**Avoid unbounded optimizer exposures:** expected returns are noisy; shrink `mu_hat`, regularize `Sigma_hat`, cap weights, penalize turnover, and test forecast-strength scaling. Record optimizer status, binding constraints, realized risk, and turnover.

### P3. Strategy/sleeve-level allocation challenger

Construct executable sleeve returns for momentum, carry, value, and mean reversion across instruments, including their overlapping holdings. Estimate a sleeve covariance and allocate risk among sleeves as an alternative to instrument-first blending. Compare the two approaches; they are not mathematically equivalent when sleeves have different horizons, costs, or constraints. Reconcile sleeve exposures to net instrument positions before trading and apply costs on **net trades**, not by double-counting opposing sleeves.

---

## 6. Layer C: Joint ML portfolio challenger

Build only after modular baselines are correct and sufficiently populated with data.

**Inputs:** per-instrument signals, cross-asset macro/regime features, lagged returns, volatility forecasts, covariance/factor exposures, availability masks, previous holdings, and estimated costs.

**Outputs:** raw per-instrument desired exposures or forecasts. Apply a deterministic differentiable or post-model constraint/risk-normalization layer so the same portfolio limits apply as modular competitors.

**Candidate objectives:**

- Supervised forecast loss followed by the *same* optimizer as modular system.
- Direct differentiable mean-variance utility: average `w_t'r_(t+1) - lambda/2 * w_t'Sigma_t w_t - costs`.
- Direct ex-post portfolio Sharpe loss over training episodes: `-mean(net_pnl)/max(std(net_pnl), eps)` plus regularization, turnover and concentration penalties. Compute training Sharpe only on training episodes; avoid look-ahead from batch construction and position timing.
- An **MSRR-inspired** or other risk-adjusted objective only after its exact definition is sourced and independently validated. Do not claim an implementation reproduces a specific published MSRR model without reading the original paper and documenting its assumptions and objective.

Train using walk-forward windows and chronological validation; regularize heavily and use multiple random seeds. Compare net Sharpe, turnover, exposure stability, drawdown, capacity, and risk forecast calibration. If sample length is inadequate, report that the joint model is not estimable with credibility and leave a tested interface rather than claiming victory.

---

## 7. Experimental design: fair horse race

### 7.1 Required model grid

**Signal combination:** equal-weight; rolling Sharpe heuristic; correlation-aware Sharpe heuristic; ridge; elastic net; gradient boosting; optional stacked ensemble and regime gate.

**Risk model:** rolling sample covariance; single EWMA; shrinkage; two-timescale EWMA (14d vol / 126d corr initial settings); optional factor covariance.

**Portfolio construction:** inverse-volatility directional allocation; equal-risk contribution when feasible; constrained mean-variance; optional sleeve-level allocation; optional joint ML.

Run an initial manageable subset rather than full Cartesian explosion. Fix the risk model when comparing signal models, fix the signal model when comparing risk models, and then test selected combinations. Maintain an experiment matrix and record every attempted configuration, including failures.

### 7.2 Walk-forward protocol

1. Choose instruments and date range based on **point-in-time availability**, including historical membership and delistings when relevant.
2. Reserve a final untouched test period; use rolling/expanding outer folds before it.
3. Inside each outer training fold, tune features, normalization, hyperparameters, risk half-lives, shrinkage, and optimizer penalty on chronological inner folds.
4. Fit models and risk estimates only with information available before each rebalance; forecast at `t`, trade after decision time, realize subsequent returns.
5. Purge overlapping forward-return labels and embargo validation boundaries as needed.
6. Refit at a predeclared cadence, e.g. monthly, while updating signals/risk daily; no retrospectively selected refit schedule.
7. Evaluate on the same eligible assets, dates, risk target, financing assumptions, costs, execution lag, and constraints. Where eligibility differs, provide common-universe comparison plus coverage metrics.
8. Use block bootstrap or HAC confidence intervals and account for multiple comparisons/model selection; report paired performance differences rather than only individual Sharpes.
9. Treat the final holdout as one evaluation, not a repeatedly tuned development set.

### 7.3 Required metrics

- Gross/net annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar, skew, kurtosis, tail losses/CVaR.
- Average and distribution of gross/net leverage, realized versus targeted risk, risk-limit violations, concentration, factor exposure, and stress P&L.
- Turnover, trade counts, spread/fees/slippage/impact, capacity estimates and break-even costs.
- Predictive IC, out-of-sample `R^2`, forecast calibration, coefficient/feature stability.
- Covariance forecast scores, PSD adjustment frequency, condition numbers, realized risk calibration.
- By asset class, instrument, signal family, regime, calendar subperiod, and crisis/stress interval.
- Sample sizes, missing-data coverage, model training times, hyperparameters, and confidence intervals.

Annualization and P&L attribution must use explicit trading-day and capital conventions. Avoid reporting a portfolio Sharpe computed on non-investable raw scores.

### 7.4 Robustness tests

Shift execution by 1–2 bars; double and triple trading costs; change training window, feature clipping, signal lookbacks, volatility half-lives, correlation half-lives, and risk targets; remove dominant asset classes and crisis periods; stress correlations toward one; test delayed macro releases and stale signals; compare long/short and constrained directional exposure. If performance disappears under mild changes, flag fragility.

---

## 8. Suggested repository structure

Adapt names to existing conventions; do not create duplicate frameworks unnecessarily.

```text
configs/
  universe.yaml
  data.yaml
  signals.yaml
  risk.yaml
  portfolio.yaml
  experiments.yaml
src/
  data/{schemas,adapters,validation,returns,rolls}.py
  signals/{momentum,carry,value,mean_reversion,normalize}.py
  blending/{equal_weight,sharpe_heuristic,linear,nonlinear,stacking}.py
  risk/{ewma,shrinkage,two_timescale,factors,diagnostics}.py
  portfolio/{baselines,optimizer,costs,attribution,constraints}.py
  ml/{joint_policy,losses,training}.py
  backtest/{walk_forward,execution,metrics,bootstrap}.py
  reporting/{tables,plots,manifest}.py
tests/
  test_asof_no_leakage.py
  test_signal_blending.py
  test_covariance_psd.py
  test_risk_contributions.py
  test_portfolio_constraints.py
  test_costs_and_rolls.py
  test_walk_forward.py
  test_synthetic_end_to_end.py
reports/
  README.md
```

### Core interfaces (illustrative, adapt to repo)

```python
class SignalBlender:
    def fit(self, features, forward_returns, *, asof, groups=None): ...
    def predict(self, features, *, asof): ...  # expected return, not just arbitrary score

class RiskForecaster:
    def fit(self, returns, *, asof): ...
    def predict_covariance(self, *, asof): ...  # aligned N x N PSD matrix

class PortfolioConstructor:
    def construct(self, expected_returns, covariance, previous_weights,
                  metadata, *, asof): ...

class WalkForwardRunner:
    def run(self, config, data): ...  # positions, P&L, diagnostics, manifest
```

Require instrument IDs and dates to align explicitly; do not silently rely on array order. All fitted artifacts must include training-end timestamp, hyperparameters, random seed, and input data version/hash. Write configurations and outputs to versioned, machine-readable formats (Parquet/CSV/JSON) and human-readable Markdown summaries.

---

## 9. Test cases the agent must implement

1. **Leakage sentinel:** A feature only published tomorrow cannot affect today's position; permuting future prices cannot change earlier fitted forecasts or positions.
2. **Correlated signals:** On synthetic correlated predictors, ridge coefficients/forecasts remain better conditioned than unstable OLS; do not assert universal Sharpe dominance.
3. **Equal-weight signs:** Two equally strong opposing signals net near zero after standardized sign conventions.
4. **Sharpe heuristic:** Uses only trailing sleeve returns, handles zero/negative Sharpe and missing histories, and respects weight caps.
5. **Covariance:** Output symmetric, finite, PSD, and aligned with the instrument universe; diagonal equals marginal variance after correlation reconstruction.
6. **Two-timescale EWMA:** Verify half-life-to-decay formula, lagging, diagonal normalization, PSD repair, and sensible response to synthetic volatility shocks.
7. **Risk contributions:** Sum of Euler contributions equals portfolio volatility within tolerance, including long-short cases.
8. **Portfolio constraints:** Gross leverage, individual caps, risk budget, and liquidity limits hold within solver tolerance; infeasibility is surfaced clearly.
9. **Transaction costs:** Zero turnover incurs no trading cost; opposite sleeve trades net before instrument-level cost calculation.
10. **Walk-forward:** Model selection and scalers fit only on permitted training ranges; purge/embargo logic correctly excludes overlapping targets.
11. **Contracts and FX:** Correct return signs, point values, currency conversion, roll accounting, and no double-counted carry.
12. **End-to-end:** A seeded synthetic panel completes data → features → blend → covariance → positions → costed P&L → report with deterministic outputs.

---

## 10. Delivery phases and definition of done

**Phase 1: Foundation.** Inspect repo, build data contracts, return construction, as-of validation, synthetic fixture, signals, normalization, and equal-weight baseline. Deliver tests and a runnable command.

**Phase 2: Signal blending.** Add Sharpe heuristic, ridge/elastic net, nonlinear ML, optional pooling/stacking, and forecast diagnostics. Produce fair common-risk signal comparisons.

**Phase 3: Risk model.** Add rolling, EWMA, shrinkage, and two-timescale covariance, PSD repairs, evaluation, and risk attribution. Compare out-of-sample risk forecast quality.

**Phase 4: Portfolio construction.** Add inverse-vol, risk parity where valid, mean-variance optimizer, costs, constraints, and asset-class/factor exposures. Report net results.

**Phase 5: Joint ML and robustness.** Add direct utility/Sharpe challenger only if data suffices; nested walk-forward, holdout, stress tests, and model comparison report.

A phase is done only when its implementation runs, passes relevant tests, documents assumptions, and produces a reproducible result or an explicit missing-data/dependency blocker. Do not skip basic correctness to deliver an attractive chart.

### Agent's final response format

```text
Implemented:
- ...
Commands run and results:
- ...
Data used and availability caveats:
- ...
Key assumptions and research choices:
- ...
Artifacts and paths:
- ...
Remaining blockers / recommended next action:
- ...
```

---

## 11. Research reading list and how to use it

These are **methodological starting points**, not promises that their results transfer to macro futures. Verify bibliographic details and full paper text before claiming faithful reproduction.

1. **Gu, Kelly & Xiu (2020), _Empirical Asset Pricing via Machine Learning_.** Broad comparison of linear regularization, trees, and neural networks for return prediction. https://www.nber.org/papers/w25398
2. **Tan, Roberts & Zohren (2023), _Spatio-Temporal Momentum: Jointly Learning Time-Series and Cross-Sectional Strategies_.** Multi-asset momentum learning and direct risk-adjusted portfolio objective. https://doi.org/10.3905/jfds.2023.1.130
3. **Banga & Brorsen (2019), _Profitability of Alternative Methods of Combining the Signals from Technical Trading Systems_.** Technical signal combinations and data-snooping caution. https://doi.org/10.1002/isaf.1442
4. **Rapach & Zhou, _Time-Series and Cross-Sectional Stock Return Forecasting: New Machine Learning Methods_.** Regularized forecast combination. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3428095
5. **Mekelburg & Strauss (2024), _Pooling and Winsorizing Machine Learning Forecasts to Predict Stock Returns with High-Dimensional Data_.** Pooling forecasts from diverse ML methods. https://www.sciencedirect.com/science/article/pii/S0927539824000732
6. **_Stock Return Prediction: Stacking a Variety of Models_ (2022).** Stacked ensembles and out-of-fold prediction design. https://www.sciencedirect.com/science/article/pii/S0927539822000342
7. **Giglio, Kelly & Xiu (2022), _Factor Models, Machine Learning, and Asset Pricing_.** Overview and further literature. https://doi.org/10.1146/annurev-financial-101521-104735
8. **Ledoit & Wolf, _Honey, I Shrunk the Sample Covariance Matrix_.** Covariance shrinkage foundation; locate authoritative publication before implementation claims.
9. **Engle (2002), _Dynamic Conditional Correlation_.** Optional dynamic-correlation challenger; do not implement before baseline diagnostics.

**MSRR note:** The exact Bryan Kelly MSRR paper and mathematical objective discussed elsewhere are not specified here. The agent must retrieve and verify that source before implementing an MSRR-labeled algorithm. Until then, use explicitly named mean-variance and direct-Sharpe benchmarks, not a guessed MSRR surrogate.

---

## 12. Initial defaults (all configurable, never silently optimized on test data)

| Parameter | Starting point | Important caveat |
|---|---|---|
| Rebalance | Daily | Execution and costs must match actual market |
| Forecast horizons | 1d, 5d, 21d experiments | Purge overlapping labels |
| Signal standardization | Rolling 252-day z-score | Use minimum history and past-only windows |
| Volatility EWMA half-life | 14 trading days | Benchmark, not universally optimal |
| Correlation EWMA half-life | 126 trading days | Benchmark, not universally optimal |
| Covariance baseline | Rolling 252d + EWMA + shrinkage | Ensure PSD and conditioning |
| Regularized blend | Ridge, then elastic net | Tune in chronological inner CV |
| Nonlinear blend | Gradient boosting | Small search space; compare against ridge |
| ML policy | Small regularized model | Only with sufficient observations |
| Risk target | 10% annualized | Research convention; configurable |
| Trading costs | Instrument-specific spread + fees + slippage | Sensitivity at 1x/2x/3x |
| Evaluation | Nested chronological walk-forward + final holdout | Never optimize on holdout |

## 13. Decisions requiring explicit documentation

Before interpreting performance, document: target horizon; signal definitions and signs; available-at timestamps; target return and funding convention; universe membership; futures rolls; cross-currency translation; risk scaling; missing data; training/validation/test boundaries; feature/parameter tuning; transaction costs; leverage and factor limits; risk-model half-lives; covariance PSD repairs; ML model size; rebalancing timing; and statistical uncertainty.

**Final research question:** Does ML improve the *net, risk-controlled, genuinely out-of-sample* portfolio relative to (a) equal-weight signal combination, (b) Sharpe heuristic, (c) ridge/elastic net, and (d) simple cross-asset risk allocation? Decompose any improvement into forecasting, covariance/risk allocation, timing/regime effects, and trading costs.
