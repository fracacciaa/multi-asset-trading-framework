
# MSRR Signal Blending: Agent Implementation Specification

**Status:** implementation-ready specification  
**Objective:** Build a reproducible, leakage-free research framework comparing simple signal blends, classical maximum-Sharpe optimization, linear/ridge MSRR, nonlinear ML-MSRR, and regime-conditioned variants for systematic FX or cross-asset trading signals.

> **For the coding agent:** Inspect the repository before modifying it. Reuse its conventions, data contracts, dependency manager, tests and backtester. Do not invent available columns, Bloomberg tickers, transaction costs or performance results. If inputs are absent, implement interfaces, synthetic fixtures and tests; document required inputs. Do not trade or submit orders. Produce research code and diagnostics only.

## 0. Research foundation and scope

Primary reading: Bryan T. Kelly and Dacheng Xiu, *Financial Machine Learning*, Sections 5.3 (Maximum Sharpe Ratio Regression) and 5.4 (High Complexity MSRR), SSRN abstract 4501707: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4501707 . Also see Britten-Jones (1999), *The Sampling Error in Estimates of Mean-Variance Efficient Portfolio Weights*, and Brandt, Santa-Clara and Valkanov (2009), *Parametric Portfolio Policies: Exploiting Characteristics in the Cross-Section of Equity Returns*. Read the paper before claiming a particular neural-network architecture is its exact implementation. The ML variants below are engineering proposals consistent with the high-complexity principle, not a verbatim reproduction of every paper specification.

The core unit of optimization is a **tradable signal-managed portfolio return series**, not an arbitrary predictor's correlation with next-period returns. Signals may be FX carry, time-series momentum, cross-sectional momentum, mean reversion, value, or other macro signals. Build signal portfolios first, then learn blend coefficients. Optionally implement a direct characteristic-to-position neural model as a separate experiment.

## 1. Repository reconnaissance and deliverables

1. Inspect existing package structure, configuration, data loaders, instrument metadata, cost model, backtesting infrastructure and tests. Record findings in `docs/msrr_integration_notes.md` (or repository-equivalent).
2. Create modular components corresponding to `data`, `signal_portfolios`, `estimators`, `validation`, `backtest`, `metrics`, `reporting`, and `tests`; conform to existing package structure rather than imposing these literal directories.
3. Provide a configuration-driven CLI entry point, e.g. `python -m <existing_package>.msrr.run --config configs/msrr.yaml`, and a runnable example with synthetic data.
4. Save out-of-sample return series, weights, diagnostics, model configurations, fold dates, costs, summary tables and plots. Prefer Parquet for panel data and JSON/YAML for metadata.
5. Include an experiment manifest with code commit hash (when available), Python/dependency versions, seed, data range, instruments, timestamp convention, cost assumptions and data fingerprint.
6. Never silently fill missing real-market data with simulated values. Synthetic data must be labeled clearly.

## 2. Data contracts and timing

Required normalized input tables (adapt names to repo):

- `prices`: `timestamp`, `asset_id`, `price` or already computed total/excess return; currency, contract identifier and roll convention where applicable.
- `signals`: `asof_timestamp`, `asset_id`, `signal_name`, `value`, optional `availability_timestamp` and `source`.
- `metadata`: instrument currency, asset class, multiplier, liquidity and trading eligibility; for FX, pair orientation, base/quote, quote convention and financing/forward treatment.
- `costs`: explicit commission, spread/slippage assumptions, funding/roll costs when not embedded in returns, and turnover methodology.
- `state_features` (optional): timestamped volatility, trend strength, liquidity, macro releases, etc., with actual public availability timestamps.

Use **availability timestamps**, not merely observation dates. At decision time `t`, only information available at or before `t` is admissible. A position formed at `t` earns return over `(t,t+1]`. Define exact market close/cutoff, rebalance frequency, holding period, and delay. Prevent look-ahead from revised macro series, forward-filled future data, contract rolls, survivorship and universe selection. Do not mix spot FX changes with carry unless the return construction explicitly includes the appropriate forward/financing economics; prefer correctly constructed FX excess returns.

Validate uniqueness of `(timestamp, asset_id, signal_name)`, chronological order, time zones, numerical finiteness, stale data, eligible universe and minimum breadth. Do not treat absent assets as zero returns without a documented convention. Log exclusions and missingness. Use training-only estimates for scaling, winsorization, imputation and covariance regularization. For asset cross-sectional transforms, only use contemporaneously available eligible assets.

## 3. From raw signals to signal-managed returns

Let `x[i,k,t]` be the observed signal for asset `i`, strategy `k`, known at rebalance `t`; let `r[i,t+1]` be its subsequent **net-of-financing but pre-execution-cost** tradable excess return. Each signal generates a fully specified portfolio `p[i,k,t]` and return:

`F[k,t+1] = sum_i p[i,k,t] * r[i,t+1]`.

Build at least two construction modes:

- **Time-series / directional:** transform each signal to desired long/short position, then cap, normalize and scale to an ex-ante risk budget.
- **Cross-sectional:** rank/z-score signal within eligible assets, optionally demean to dollar/FX-factor neutrality, cap concentrations and scale to a common ex-ante risk budget.

Choose normalization explicitly: rank, cross-sectional z-score, robust z-score, or raw economic units. Winsorization thresholds, volatility estimators, shrinkage and exposure caps must be fitted only on the available historical sample. Use lagged volatility forecasts; no contemporaneous realized future volatility. Do not accidentally erase a desired directional USD beta by imposing neutrality indiscriminately.

Store for each `k,t`: signal portfolio positions, gross exposure, risk estimate, realized pre-cost return, transaction-cost estimate, realized net return and turnover. Set a common **ex-ante volatility/risk convention** before blending. A baseline should equal-weight *comparable* signal sleeves, not heterogeneous unscaled raw signals.

**Cost accounting:** For a combined portfolio `P[i,t] = sum_k b[k,t] p[i,k,t]`, compute traded notional and costs on **changes in the final asset-level positions**. Summing standalone sleeve costs can double count offsetting trades. Model spreads/slippage using available liquidity information, and distinguish signal sleeve pre-cost returns from combined-portfolio realized net returns. Never optimize on net returns that incorporate costs calculated using future weights.

## 4. Linear MSRR: mathematical contract and identity checks

Let `F` be a `T x K` matrix of realized managed-portfolio returns over the **training** period. Define `mu = mean(F, axis=0)`, `M = F.T @ F / T`, and population-style covariance `Sigma = M - outer(mu,mu)`. Use consistent denominator conventions throughout identity tests.

### 4.1 OLS MSRR

Minimize `L(b) = (1/T) sum_t (1 - F[t,:] @ b)^2`.

For nonsingular `M`, the solution is `b_OLS = M^{-1} mu` (numerically: use a stable linear solver, never explicit inversion). In the unconstrained case, this has the **same direction** as `Sigma^{-1} mu`, since by Sherman-Morrison:

`M^{-1} mu = (Sigma^{-1} mu) / (1 + mu.T @ Sigma^{-1} @ mu)`.

Thus the raw OLS coefficients and classical tangency coefficients differ by a scalar, not their risk-normalized portfolio direction. This equivalence requires a consistent sample, compatible covariance definition, unconstrained signed weights, nonsingular matrices and no inconsistent intercept or restrictions. When sample moments are singular, use documented pseudoinverse or regularization and do not assert exact identity.

### 4.2 Classical closed-form maximum Sharpe

`b_MV ∝ Sigma^{-1} mu` for unconstrained portfolios and a risk-free/excess-return convention. Estimate both on training data only. This is a **directional** solution: fix scale by ex-ante portfolio risk target, not arbitrary sum-to-one if signed positions are allowed. If `mu` is weak/noisy or predicted risk is zero, apply explicit fallback policy. In small samples use stable `solve`, conditioning checks and configurable covariance shrinkage. The original linear MSRR and closed-form MV are **not independent alpha models**; use their equivalence as a correctness test.

### 4.3 Constraints

Leverage limits, nonnegative blend weights, concentration caps and turnover penalties break the simple closed-form identity. Implement such variants separately and label them `constrained_*`. When constraints apply, solve the actual constrained optimization rather than clipping the unconstrained solution without re-evaluation.

## 5. Benchmark estimators

Implement a uniform interface:

`fit(F_train, context_train=None) -> fitted_model`  
`predict_weights(asof_time, context_asof=None) -> vector[K]`

Weights are applied to **future** signal returns and mapped to asset positions. Separate fitting, weight normalization, portfolio risk targeting and transaction-cost application.

1. **Equal weight:** `b_k = 1/K` for K eligible, risk-normalized sleeves. Define missing/disabled-sleeve handling; optionally require a minimum number of sleeves. Do not inadvertently use absolute equal weight when signs are meaningful.
2. **Standalone Sharpe heuristic:** estimate `SR_k = mean(F_k)/std(F_k)` on training data, consistently annualized if displayed. For long-only allocation across *oriented* sleeves, define `a_k = max(SR_k,0)^p` with `p in {1,2}`, then `b = a/sum(a)`; fallback to equal weight if all zero. This discards negatively estimated sleeves and does not model correlations. Separately permit signed `SR` weights only as an explicitly named variant, with robust normalization and risk limits. Never label heuristic weights mathematically optimal.
3. **Classical maximum Sharpe:** `solve(Sigma_hat,mu_hat)` with documented shrinkage, conditioning and target-risk normalization.
4. **OLS MSRR:** least squares against an all-ones target; no intercept. Use QR/SVD or `lstsq`, report rank and condition number.
5. **Ridge MSRR:** minimize `mean((1-Fb)^2) + lambda * ||b||_2^2`, equivalent to `(M + lambda I)b = mu` with this loss convention. Tune lambda strictly within training history. Standardize sleeve returns using **training** parameters before regularization, then map coefficients back to original units. Report effective exposure and stability. Compare against covariance-shrunk MV but do not claim identical shrinkage parameterizations.
6. **Family/hierarchical blend (optional):** group signals by economic family, first blend within family and then across families. Define whether both stages are estimated with non-overlapping or nested training data; avoid using the same outcomes for unconstrained two-stage optimization without accounting for extra flexibility.

All methods must use exactly the same tradable inputs and common portfolio-level volatility/leverage controls for fair comparison.

## 6. ML-MSRR: nonlinear signal blending

Implement a small, controlled neural approach first, then consider more complex architectures only if justified by data volume.

### 6.1 Recommended first implementation: state-conditioned neural blend

Input `z_t`: lagged regime/state features (e.g. realized vol forecast, trend strength, carry dispersion, liquidity, funding stress), and optionally lagged sleeve diagnostics that were available at `t`. Model `b_t = g_theta(z_t)`, where `g_theta` is a small MLP. Realized strategy return is `R[t+1] = b_t.T @ F[t+1]`. Choose output constraints deliberately: softmax for long-only nonnegative sleeve blends, tanh + normalized gross exposure for signed blends, or unconstrained output with smooth leverage penalty. No look-ahead in `z_t`.

Train on chronological samples using a differentiable portfolio objective. Two options, both reported:

- **Nonlinear MSRR-style quadratic:** minimize `mean((1-R[t+1])^2)` plus explicit L2/weight/turnover penalties. This extends the OLS target but **is not generally identical to maximizing realized Sharpe** for time-varying or constrained portfolios.
- **Direct Sharpe objective:** maximize `mean(R) / sqrt(var(R)+epsilon)` on training batches or whole windows, with risk/leverage/turnover penalties. Ensure batches preserve chronology and sufficiently represent the time-series distribution; small minibatch Sharpe is noisy. Use the full-window objective when feasible.

Use training-only feature scaling, fixed random seeds, early stopping on a strictly later validation segment, dropout/weight decay, gradient clipping and small hyperparameter grids. Fit no features or hyperparameters on test data. Report at least 5 seeds and median/dispersion across seeds. Explicitly measure regime-conditional allocations and whether learned interactions are stable.

### 6.2 Optional direct high-complexity characteristic MSRR

For each asset `i,t`, learn `h_theta(x[i,t], z_t)` as a position score; apply exposure/risk constraints to get `P[i,t]`; optimize realized portfolio returns with the MSRR-style or direct-Sharpe loss. This is distinct from learning time-varying weights across preconstructed sleeves. Implement only after sleeve-level ML passes all validation checks, because direct models have much greater capacity and implementation complexity.

### 6.3 Regime-conditioned linear benchmark

Before a neural network, fit interpretable interactions: `b_t = b_0 + B z_t` with ridge regularization, or predeclared regime buckets with minimum sample counts. This provides a lower-capacity test of whether state dependence helps. Avoid training separate regimes on tiny samples.

## 7. Walk-forward, nested model selection and leakage prevention

Provide configurable expanding and rolling outer windows. Example **illustrative, not mandatory**: 5 years initial training, 1 year outer test, monthly retraining/rebalancing; adapt to signal frequency, history and sample size. For each outer fold:

1. Freeze training cutoff and eligible historical universe; fit preprocessing only on past observations.
2. Split the outer training data into chronological inner train/validation segments. Tune ridge penalties, covariance shrinkage, MLP architecture, regularization, early stopping and risk limits without seeing outer test outcomes.
3. Fit final model using only permitted past data, using a documented refit policy after inner selection.
4. Generate weights at each decision timestamp using only information then available; compute actual next-period returns, exposures, turnover and costs.
5. Persist each prediction, weight vector, data cutoff, fold ID, training period, parameter choices and diagnostics.
6. Concatenate outer test predictions once; compute overall OOS statistics **without** choosing the winning model by repeatedly peeking at the same test. If doing extensive research, reserve a final untouched holdout.

If targets/labels use overlapping forward returns, introduce a purge/embargo at split boundaries at least as long as the overlap horizon; adapt for data release delays. Preserve common test dates and eligible sleeves for apples-to-apples comparisons, or explicitly report sample differences. Estimate ex-ante portfolio risk using only past returns and apply target volatility with sensible maximum gross leverage; use **realized** volatility only for reporting, never for contemporaneous sizing.

## 8. Metrics and diagnostics

Compute from stitched OOS **net** portfolio returns, with gross metrics shown alongside:

- Annualized mean, volatility and Sharpe with explicit frequency; annualized Sharpe `sqrt(periods_per_year) * mean / std` only under clearly stated sampling conventions. Also report HAC/Newey-West-adjusted uncertainty or block bootstrap when serial correlation/overlap is material.
- Sortino, maximum drawdown, Calmar, hit rate, skew, excess kurtosis, worst period, tail loss (VaR/CVaR as descriptive historical metrics), and downside beta where relevant.
- Portfolio-level average and percentile turnover, total transaction costs, gross/net exposure, leverage, realized vs target vol, concentration and fraction of time constraints bind.
- Sleeve weights over time, weight drift, coefficient norms, conditioning/rank, sleeve return correlation heatmap, rolling standalone Sharpe and contribution to risk/return.
- Incremental OOS Sharpe and return vs equal weight, Sharpe heuristic and closed-form MV; paired differences on common dates, block bootstrap confidence intervals, and uncertainty around differences. Do not infer significance from point estimates alone.
- Performance segmented by **predefined, observable** market regimes, not ex-post regime labels used for trading. Analyze crisis/stress periods as descriptive diagnostics, not automatic tuning targets.
- Ablations: remove one sleeve/family at a time, compare correlation clusters, alter estimation window, vary cost assumptions, and test sensitivity to training seed and hyperparameters.

Visual outputs: cumulative gross/net returns; rolling Sharpe/vol; drawdowns; dynamic blend weights; regime allocations; turnover/cost bars; signal correlation and condition number; fold-level scorecards. Clearly mark train/validation/test and risk-target assumptions.

## 9. Required correctness and adversarial tests

Use deterministic synthetic fixtures and repository-native test framework. At minimum:

1. **OLS/MV direction identity:** For a full-rank, well-conditioned synthetic `F`, normalized `lstsq(F,ones)` and `solve(cov(F),mean(F))` agree in direction to numerical tolerance, with consistent covariance convention.
2. **Ridge algebra:** Verify optimizer against `solve(M+lambda*I,mu)` for chosen loss scaling.
3. **Equal-weight and Sharpe heuristics:** Check normalization, zero/negative Sharpe fallback, missing sleeves and sign policy.
4. **No leakage:** Perturb all data strictly after a given cutoff and confirm earlier fitted weights/predictions are unchanged. Perturb availability timestamps and confirm late-released features cannot be used early.
5. **Correlated/duplicate sleeves:** Ensure no crashes, condition diagnostics flag issues, regularization yields finite weights, and risk caps bind as designed.
6. **Singular covariance / K >= T / constant returns / all-zero signals:** Explicit documented fallback or meaningful error; never NaN weights or silent arbitrary inversion.
7. **Cost netting:** Opposing trades in two sleeves cancel at asset level before cost calculation.
8. **Vol targeting:** Forecast uses lagged data; no leverage beyond configured limits; test with zero/near-zero estimated volatility.
9. **ML reproducibility:** Fixed seeds, deterministic fixture, nonempty train/validation/test separation; shuffled chronological features should destroy or reduce artificial predictability on designed synthetic data.
10. **Look-ahead trap:** Create a future-only predictive feature; confirm it is rejected or shifted correctly rather than producing spurious Sharpe.
11. **Synthetic economic scenarios:** Independent positive-alpha sleeves; nearly identical sleeves; regime-switching payoff signs; noisy zero-alpha sleeves; high-cost/high-turnover sleeve. Verify plausible relative behavior, without hard-coding that ML must win.
12. **Accounting:** Verify stitched OOS P&L equals asset-position-level P&L net of modeled costs, with a documented tolerance.

## 10. Experiment matrix and hypotheses

Run an experiment grid, starting parsimoniously:

| Model ID | Specification | Key hypothesis |
|---|---|---|
| EW | Equal weight | Strong robust default |
| SH1 | Positive Sharpe weights, power 1 | Performance ranking helps |
| SH2 | Positive Sharpe weights, power 2 | More concentration helps or hurts |
| MV | Classical tangency | Covariance-aware allocation improves |
| MV-S | Shrunk covariance tangency | Covariance shrinkage improves stability |
| MSRR | Original no-intercept OLS | Must match MV direction under equivalent assumptions |
| MSRR-R | Ridge MSRR | Regularization improves OOS |
| FAM | Hierarchical family blend | Reduces effective complexity |
| REG-L | Linear regime interactions | State dependence helps |
| ML-Q | Neural time-varying weights, quadratic MSRR loss | Nonlinear relationships help |
| ML-S | Neural time-varying weights, direct Sharpe loss | Direct portfolio objective helps |
| ML-D | Direct asset-characteristic neural model (optional) | Additional flexibility earns its costs |

All hyperparameters selected only through nested time-series validation. Compare model performance on identical OOS timestamps and common risk budget. Track number of trained candidates to understand multiple-testing exposure. **Never present a selected winner as statistically established solely because its backtest Sharpe is highest.**

## 11. Suggested configuration schema

```yaml
experiment:
  name: fx_msrr_v1
  seed: 42
  asset_class: fx
  frequency: weekly                 # confirm against source data
  return_type: excess_return       # document FX carry/forward convention
  timezone: UTC
  target_vol_annual: 0.10
  max_gross_leverage: 3.0
  final_holdout_start: null        # populate when dates are known
signals:
  names: [carry, momentum, mean_reversion]
  construction: cross_sectional    # or time_series; configure per sleeve if needed
  normalization: zscore
  risk_normalize_sleeves: true
  minimum_assets: 5
validation:
  scheme: expanding
  initial_train_periods: null      # choose after inspecting data
  outer_test_periods: null
  inner_folds: 3
  purge_periods: 0                 # increase for overlapping forward labels
  rebalance_periods: 1
costs:
  enabled: true
  source: configured              # never invent numeric costs
models:
  equal_weight: {}
  sharpe_heuristic: {powers: [1, 2], positive_only: true}
  mv: {covariance: [sample, shrunk]}
  msrr_ols: {fit_intercept: false}
  msrr_ridge: {lambda_grid: [0.0, 0.000001, 0.0001, 0.01, 1.0]}
  regime_linear: {enabled: true}
  neural_quadratic:
    enabled: true
    hidden_units: [[8], [16, 8]]
    seeds: [11, 22, 33, 44, 55]
    early_stopping: true
  neural_sharpe:
    enabled: true
    hidden_units: [[8], [16, 8]]
    seeds: [11, 22, 33, 44, 55]
reporting:
  output_dir: outputs/msrr
```

The ridge grid is illustrative: its effective scale depends on whether the sleeve returns are expressed as decimals and standardized. Prefer a scale-aware grid derived from training moments; do not blindly use the example values.

## 12. Agent execution order and acceptance criteria

**Phase A: audit + contract.** Inspect repository; produce integration notes, input schemas, timing conventions, FX return accounting and unresolved-data checklist. Do not guess missing inputs.

**Phase B: sleeves + baselines.** Implement leakage-safe sleeve construction and final-position cost accounting; equal weight, SH1/SH2, MV/MV-S. Unit tests and synthetic example must pass.

**Phase C: linear MSRR.** Implement no-intercept OLS and ridge, normalization and covariance diagnostics. Verify OLS/MV directional equivalence numerically before advancing.

**Phase D: validation + reports.** Implement nested walk-forward, risk scaling, out-of-sample metrics, cost sensitivity, attribution and reproducible artifact export.

**Phase E: regime and ML.** Implement regime-linear, then neural quadratic and direct-Sharpe variants with small networks and 5 seeds. Add direct characteristic model only if justified and feasible.

**Phase F: final audit.** Run all tests, verify timestamp leakage guards and P&L reconciliation; produce `README` usage, sample config, experiment summary, caveats, and a list of commands executed. Clearly separate actual-market results from synthetic demonstrations.

### Definition of done

- A fresh environment can run tests and a synthetic end-to-end experiment using documented commands.
- All requested benchmarks and the two neural sleeve-blending objectives are implemented, or any blocker is explicitly documented with a tested interface stub.
- Classical MV and OLS MSRR are directionally equivalent in the defined unconstrained identity test.
- All reported performance is strictly stitched outer-fold OOS, with final-asset-level trading costs and common risk targeting.
- All feature transformations and model selections are trained on historical data only; timestamps and fold membership are auditable.
- Report includes model-by-model net Sharpe, uncertainty, drawdown, turnover, costs, risk exposure, weight stability, correlations and regime breakdowns.
- No unverified claim that a particular model outperforms; document failure cases and sample limitations.

## 13. Research cautions

1. **Correlation of raw signals is not the same as correlation of tradable sleeve returns.** Diagnose both, but optimize and assess portfolio-level return covariance.
2. **Low signal correlation is not a requirement for MSRR.** Incremental predictive return and stable estimation matter more; nearly redundant signals increase conditioning problems.
3. **Sharpe-ratio heuristics are not closed-form maximum-Sharpe solutions.** Keep them as separate benchmarks.
4. **OLS MSRR and unconstrained tangency are directionally equivalent.** Their independent implementation is an algebraic validation, not evidence of distinct alpha.
5. **Static linear MSRR does not automatically detect regimes.** Explicit interactions or nonlinear conditioning are required.
6. **Neural quadratic MSRR is not automatically equivalent to direct Sharpe maximization.** Label and test both objectives.
7. **Financial return samples are small and dependent.** Use parsimonious models, chronological validation, realistic costs and uncertainty quantification.
8. **FX return definitions matter.** Spot price change alone is not the same as implementable FX excess return or carry-inclusive return.
9. **Risk constraints change optimization.** Report their effects and do not claim unconstrained identities after constraints.
10. **Avoid backtest overfitting.** Fix a research protocol, limit repeated peeking and preserve a final untouched holdout.
