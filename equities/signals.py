"""
Equities signals: momentum, tsmom, mean reversion, carry.

Raw signal functions return cross-sectional winsorized DataFrames (date × ticker).
They are not intended to be used directly — always call build_vf_positions() to get
signals that have been processed through ViewFormation and are ready for blending.

Pipeline:
    build_vf_positions()
        → momentum / tsmom / mean_reversion / carry   (raw signals, this file)
        → ViewFormationXS / ViewFormationTS            (equities/view_formation.py)
        → dict[str, pd.DataFrame] of positions        (ready for equities/blend.py)
"""

import pandas as pd
import numpy as np
from typing import Optional

from equities.sectors import GICS, FINANCIALS_TICKERS


def momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    half_life: Optional[int] = 126,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Cross-sectional momentum signal.

    For each date t and ticker i, computes a formation-window return, then
    winsorizes and z-scores cross-sectionally.

    lookback:   formation window in trading days (default 252 = ~12 months)
    skip:       days excluded at the recent end to avoid short-term reversal
                (default 21 = ~1 month, giving classic 12-1m momentum)
    half_life:  if set, uses exponentially weighted log returns within the
                formation window. Set to None for plain cumulative return.
    winsor_pct: symmetric cross-sectional winsorization percentile (default 1%)

    Returns date × ticker DataFrame of winsorized raw returns (not z-scored).
    All normalization is delegated to ViewFormationXS.
    """
    if half_life is not None:
        # Exponentially weighted: recent months count more, old history decays smoothly
        log_ret = np.log(prices / prices.shift(1))
        lagged  = log_ret.shift(skip)
        cumret  = lagged.ewm(halflife=half_life, min_periods=lookback).mean()
    else:
        price_end   = prices.shift(skip)
        price_start = prices.shift(lookback + skip)
        cumret = price_end / price_start - 1

    # Cross-sectional winsorize per date
    q_lo = cumret.quantile(winsor_pct, axis=1)
    q_hi = cumret.quantile(1 - winsor_pct, axis=1)
    cumret = cumret.clip(lower=q_lo, upper=q_hi, axis=0)

    return cumret


def tsmom(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Time-series momentum signal (absolute directional).

    skip=21 (default): mirrors CSMOM's 12-1m convention, removes short-term
    reversal noise — empirically better than skip=0 across train/val periods.
    Returns raw winsorized cumulative returns; normalization by ViewFormationTS.
    """
    cumret = prices.shift(1 + skip) / prices.shift(lookback + 1 + skip) - 1
    q_lo = cumret.quantile(winsor_pct, axis=1)
    q_hi = cumret.quantile(1 - winsor_pct, axis=1)
    return cumret.clip(lower=q_lo, upper=q_hi, axis=0)


def mean_reversion(
    prices: pd.DataFrame,
    lookback: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Short-term reversal signal.

    Signal = negated lookback-day return (1-day lag to avoid bid-ask bounce).
    Recent losers → positive (buy), recent winners → negative (sell).
    Returns raw winsorized signal; normalization delegated to ViewFormationXS.
    """
    ret = prices.shift(1) / prices.shift(lookback + 1) - 1
    signal = -ret
    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def carry(
    prices: pd.DataFrame,
    fundamentals: dict,
    fcf_lag: int = 63,
    sector_adjust: str = "demean",
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Two-component equity carry signal.

    Component 1 — FCF yield = TTM FCF / market cap (lagged fcf_lag days).
      For Financials: earnings yield = BEST_EPS / PX_LAST (no lag).

    Component 2 — Net buyback yield = -Δlog(market_cap/price) over 252 days.
      Positive = buybacks (share count falling), negative = dilution.

    sector_adjust controls how sector bets are handled:
      "zscore"  — z-score within sector (XS signal: relative within-sector bet)
      "demean"  — subtract sector mean only (TS signal: preserves within-sector
                  magnitude; removes energy-always-cheap / tech-always-expensive bias)
      "none"    — no sector adjustment (pure TS: sector-concentrated but directional)

    After sector adjustment, each component is scaled by its cross-sectional std
    so the two are comparable before averaging. Composite is winsorized.
    """
    mkt_cap = fundamentals["market_cap"]
    fcf     = fundamentals["fcf"]
    eps_fwd = fundamentals["eps_fwd"]

    # ── Component 1: yield ──────────────────────────────────────────────────
    fcf_yield = fcf.shift(fcf_lag) / mkt_cap
    eps_yield = eps_fwd / prices

    fins     = [t for t in prices.columns if t in FINANCIALS_TICKERS]
    non_fins = [t for t in prices.columns if t not in FINANCIALS_TICKERS]

    yield_signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    if non_fins:
        yield_signal[non_fins] = fcf_yield.reindex(columns=non_fins)
    if fins:
        yield_signal[fins] = eps_yield.reindex(columns=fins)

    # ── Component 2: net buyback yield ──────────────────────────────────────
    adj_shares    = mkt_cap / prices
    buyback_yield = -np.log(adj_shares / adj_shares.shift(252))

    # ── Winsorize raw components before sector adjustment ───────────────────
    # Essential for TS processing: extreme FCF/price ratios and large share-count
    # swings (mergers, recapitalisations) otherwise survive sector demeaning and
    # produce catastrophic weights after vol scaling.
    def _winsorize(df: pd.DataFrame) -> pd.DataFrame:
        q_lo = df.quantile(winsor_pct, axis=1)
        q_hi = df.quantile(1 - winsor_pct, axis=1)
        return df.clip(lower=q_lo, upper=q_hi, axis=0)

    yield_signal  = _winsorize(yield_signal)
    buyback_yield = _winsorize(buyback_yield)

    # ── Sector adjustment ────────────────────────────────────────────────────
    sector_map = pd.Series(GICS).reindex(prices.columns)

    def _sector_op(df: pd.DataFrame, op: str) -> pd.DataFrame:
        out = df.copy()
        for sector in sector_map.dropna().unique():
            cols = [c for c in sector_map[sector_map == sector].index if c in df.columns]
            if not cols:
                continue
            sub = df[cols]
            mu  = sub.mean(axis=1)
            out[cols] = sub.sub(mu, axis=0)
            if op == "zscore":
                sig = sub.std(axis=1).replace(0, np.nan)
                out[cols] = out[cols].div(sig, axis=0)
        return out

    if sector_adjust == "zscore":
        adj_yield    = _sector_op(yield_signal,  "zscore")
        adj_buyback  = _sector_op(buyback_yield, "zscore")
    elif sector_adjust == "demean":
        adj_yield    = _sector_op(yield_signal,  "demean")
        adj_buyback  = _sector_op(buyback_yield, "demean")
    else:  # "none"
        adj_yield    = yield_signal.copy()
        adj_buyback  = buyback_yield.copy()

    # Scale each component by its cross-sectional std so they're comparable
    # before averaging (FCF yield and buyback yield have different natural scales)
    def _xscale(df: pd.DataFrame) -> pd.DataFrame:
        std = df.std(axis=1).replace(0, np.nan)
        return df.div(std, axis=0)

    adj_yield   = _xscale(adj_yield)
    adj_buyback = _xscale(adj_buyback)

    # ── Composite: average, fall back to available component ────────────────
    composite = (adj_yield + adj_buyback) / 2
    composite = composite.fillna(adj_yield).fillna(adj_buyback)

    # ── Winsorize ────────────────────────────────────────────────────────────
    q_lo = composite.quantile(winsor_pct, axis=1)
    q_hi = composite.quantile(1 - winsor_pct, axis=1)
    return composite.clip(lower=q_lo, upper=q_hi, axis=0)

# ── Research / extended signals ───────────────────────────────────────────────

def residual_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    beta_lookback: int = 252,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Market-orthogonal (residual) cross-sectional momentum.

    Strip the EW-market beta component from each stock's formation-window
    cumulative return. Residual return is a cleaner idiosyncratic signal
    that avoids time-varying factor exposures (Blitz, Huij & Martens 2011).
    """
    log_ret = np.log(prices / prices.shift(1))
    mkt_ret = log_ret.mean(axis=1)

    # Vectorized rolling beta: Cov(stock, mkt) / Var(mkt)
    cross    = log_ret.multiply(mkt_ret, axis=0)
    roll_cov = (
        cross.rolling(beta_lookback).mean()
        .sub(log_ret.rolling(beta_lookback).mean()
             .multiply(mkt_ret.rolling(beta_lookback).mean(), axis=0))
    )
    roll_var = (
        mkt_ret.pow(2).rolling(beta_lookback).mean()
        - mkt_ret.rolling(beta_lookback).mean().pow(2)
    )
    betas = roll_cov.div(roll_var.replace(0, np.nan), axis=0)

    cum_stock = log_ret.shift(skip).rolling(lookback).sum()
    cum_mkt   = mkt_ret.shift(skip).rolling(lookback).sum()

    signal = cum_stock.sub(betas.multiply(cum_mkt, axis=0))

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def fip_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Frog-in-the-Pan momentum (Da, Gurun & Warachka 2014).

    Weights the formation-window cumulative return by how continuous
    (gradual) the price path was. Stocks whose gains arrived in many
    small daily increments rather than a few large jumps show stronger
    and more persistent momentum.

    signal = (cumret + |cumret| × fip) / 2
      where fip = (pos_days − neg_days) / total_days ∈ [-1, 1]

    Consistent winners/losers get full signal strength; stocks with
    contradictory daily-vs-cumulative information are dampened toward zero.
    """
    log_ret = np.log(prices / prices.shift(1))

    signs   = (log_ret > 0).astype(float) - (log_ret < 0).astype(float)
    fip     = signs.shift(skip).rolling(lookback, min_periods=lookback // 2).mean()
    cum_ret = log_ret.shift(skip).rolling(lookback, min_periods=lookback // 2).sum()

    signal  = (cum_ret + cum_ret.abs() * fip) / 2

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def high_52w(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 1,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    52-week high momentum (George & Hwang 2004).

    Signal = price / rolling-maximum-price over the past year.
    Stocks near their annual high are anchoring targets; investors
    under-react to new positive news, creating momentum continuation.
    """
    p_lag  = prices.shift(skip)
    p_high = p_lag.rolling(lookback, min_periods=lookback // 2).max()
    signal = p_lag / p_high.replace(0, np.nan)

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def industry_reversal(
    prices: pd.DataFrame,
    lookback: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Industry-adjusted short-term reversal (Hameed, Huang & Mian 2010).

    Strips the sector-wide component from the lookback-day return before
    negating. Sector-driven moves (fundamental information) are excluded;
    only the idiosyncratic portion that tends to revert is kept.
    """
    ret = prices.shift(1) / prices.shift(lookback + 1) - 1

    sector_map = pd.Series(GICS).reindex(prices.columns)
    mkt_ret    = ret.mean(axis=1)

    # Start from market mean; overwrite with sector mean per group
    sector_ret = pd.DataFrame(
        np.tile(mkt_ret.values[:, None], (1, len(prices.columns))),
        index=prices.index, columns=prices.columns,
    )
    for sect in sector_map.dropna().unique():
        cols = sector_map[sector_map == sect].index.intersection(prices.columns)
        if len(cols) > 0:
            mu = ret[cols].mean(axis=1).values[:, None]
            sector_ret.loc[:, cols] = np.tile(mu, (1, len(cols)))

    idio   = ret - sector_ret
    signal = -idio

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def medium_reversal(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Medium-term mean reversion (6-month horizon, 1-month skip).

    Negates the 6-month return (skipping the most recent month to avoid
    overlap with short-term reversal). Sits between the 21-day reversal
    and the 12-1m momentum window; tests whether medium-horizon returns
    over-shoot fundamental value.
    """
    ret    = prices.shift(1 + skip) / prices.shift(lookback + 1 + skip) - 1
    signal = -ret

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def earnings_yield_all(
    prices: pd.DataFrame,
    eps_fwd: pd.DataFrame,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Forward earnings yield for all stocks, sector-demeaned.

    Purer earnings-based carry vs the composite carry signal (which mixes
    FCF + buyback yield). Using forward EPS / price for every stock (not
    just financials) gives a consistent cross-sectional comparison.
    """
    signal = (eps_fwd / prices.replace(0, np.nan)).copy()

    sector_map = pd.Series(GICS).reindex(prices.columns)
    for sect in sector_map.dropna().unique():
        cols = sector_map[sector_map == sect].index.intersection(prices.columns).tolist()
        if cols:
            mu = signal[cols].mean(axis=1)
            signal[cols] = signal[cols].sub(mu, axis=0)

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def fcf_yield(
    prices: pd.DataFrame,
    market_cap: pd.DataFrame,
    fcf: pd.DataFrame,
    fcf_lag: int = 63,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Pure TTM FCF yield, sector-demeaned.

    Uses realized free cash flow only — no analyst estimates, no buyback
    component. The 63-day lag matches the reporting delay in the production
    carry signal. Sector-demeaning removes the persistent cheap/expensive
    sector bias (energy always cheap, tech always expensive).
    """
    signal = fcf.shift(fcf_lag) / market_cap.replace(0, np.nan)

    sector_map = pd.Series(GICS).reindex(prices.columns)
    for sect in sector_map.dropna().unique():
        cols = sector_map[sector_map == sect].index.intersection(prices.columns).tolist()
        if cols:
            mu = signal[cols].mean(axis=1)
            signal[cols] = signal[cols].sub(mu, axis=0)

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def earnings_revision(
    prices: pd.DataFrame,
    eps_fwd: pd.DataFrame,
    lookback: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Analyst earnings revision momentum: change in forward E/P over lookback days.

    Avoids the level-based regime problem of earnings_yield_all by using the
    *change* in analyst estimates. Rising estimates → long; falling → short.
    Sector-demeaned to focus on relative revision dynamics within sector.
    """
    ep = eps_fwd / prices.shift(1).replace(0, np.nan)
    signal = ep - ep.shift(lookback)

    sector_map = pd.Series(GICS).reindex(prices.columns)
    for sect in sector_map.dropna().unique():
        cols = sector_map[sector_map == sect].index.intersection(prices.columns).tolist()
        if cols:
            mu = signal[cols].mean(axis=1)
            signal[cols] = signal[cols].sub(mu, axis=0)

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def composite_carry(
    prices: pd.DataFrame,
    market_cap: pd.DataFrame,
    fcf: pd.DataFrame,
    eps_fwd: pd.DataFrame,
    fcf_lag: int = 63,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Composite carry: equal-weight average of FCF yield and earnings yield.

    Diversifies across two independent fundamental metrics so neither one
    dominates. Each component is cross-sectionally standardized before
    averaging so they contribute equally regardless of level differences.
    """
    c1 = fcf_yield(prices, market_cap, fcf, fcf_lag=fcf_lag)
    c2 = earnings_yield_all(prices, eps_fwd)

    def _xs_std(df):
        mu  = df.mean(axis=1)
        std = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(std, axis=0)

    signal = (_xs_std(c1) + _xs_std(c2)) / 2.0

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def payout_momentum(
    prices: pd.DataFrame,
    market_cap: pd.DataFrame,
    lookback: int = 252,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Payout momentum: year-over-year change in net buyback yield.

    Rising buyback yield signals management confidence in future cash flows
    (Boudoukh, Michaely, Richardson & Roberts 2007). Increasing repurchase
    activity predicts higher future returns in the cross-section.
    """
    adj_shares    = market_cap / prices.replace(0, np.nan)
    buyback_yield = -np.log(adj_shares / adj_shares.shift(lookback))
    signal        = buyback_yield - buyback_yield.shift(lookback)

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def residual_reversal(
    prices: pd.DataFrame,
    lookback: int = 21,
    beta_lookback: int = 252,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Market-orthogonal short-term reversal (Blitz, Huij, Lansdorp & Verbeek 2013).

    Strips the EW-market beta from the lookback-day cumulative return before
    negating. Removes systematic factor reversals that contaminate the raw
    signal, leaving a cleaner idiosyncratic mean-reversion bet.
    """
    log_ret = np.log(prices / prices.shift(1))
    mkt_ret = log_ret.mean(axis=1)

    cross    = log_ret.multiply(mkt_ret, axis=0)
    roll_cov = (
        cross.rolling(beta_lookback).mean()
        .sub(log_ret.rolling(beta_lookback).mean()
             .multiply(mkt_ret.rolling(beta_lookback).mean(), axis=0))
    )
    roll_var = (
        mkt_ret.pow(2).rolling(beta_lookback).mean()
        - mkt_ret.rolling(beta_lookback).mean().pow(2)
    )
    betas = roll_cov.div(roll_var.replace(0, np.nan), axis=0)

    cum_stock = log_ret.shift(1).rolling(lookback).sum()
    cum_mkt   = mkt_ret.shift(1).rolling(lookback).sum()

    idio   = cum_stock.sub(betas.multiply(cum_mkt, axis=0))
    signal = -idio

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def vol_scaled_reversal(
    prices: pd.DataFrame,
    lookback: int = 21,
    vol_lookback: int = 63,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Volatility-scaled short-term reversal.

    Divides the lookback-day return by the stock's rolling realized volatility
    before negating. Equivalent to reversing a Sharpe-ratio-like measure: a
    large return in a low-vol stock (more statistically unusual) generates a
    stronger signal than the same return in a high-vol stock.
    """
    log_ret  = np.log(prices / prices.shift(1))
    real_vol = log_ret.rolling(vol_lookback).std().replace(0, np.nan)
    cum_ret  = log_ret.shift(1).rolling(lookback).sum()
    signal   = -(cum_ret / real_vol)

    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def threshold_mean_reversion(
    prices: pd.DataFrame,
    lookback: int = 21,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    State-machine mean reversion: enter short when XS z-score > +threshold,
    switch to long when z-score < -threshold, hold otherwise.

    Unlike raw reversal signals this avoids fading small drifts — it only
    takes a position after a statistically significant cross-sectional
    dislocation, then holds until the opposite extreme is reached.
    """
    ret = prices.shift(1) / prices.shift(lookback + 1) - 1
    mu  = ret.mean(axis=1)
    std = ret.std(axis=1).replace(0, np.nan)
    z   = ret.sub(mu, axis=0).div(std, axis=0)

    # Mark threshold crossings; NaN means "no new trigger, hold previous state"
    trigger = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    trigger[z >  threshold] = -1.0   # large up move → short (expect reversal)
    trigger[z < -threshold] = +1.0   # large down move → long (expect reversal)
    return trigger.ffill().fillna(0.0)


_SIGNAL_KINDS: dict[str, str] = {
    # production
    "momentum":           "xs",
    "tsmom":              "ts",
    "mean_reversion":     "xs",
    "carry":              "xs",
    # momentum extensions
    "residual_momentum":  "xs",
    "fip_momentum":       "xs",
    "high_52w":           "xs",
    # mean-reversion extensions
    "industry_reversal":          "xs",
    "medium_reversal":            "xs",
    "residual_reversal":          "xs",
    "vol_scaled_reversal":        "xs",
    "threshold_mean_reversion":   "xs",
    # carry extensions
    "earnings_yield_all": "xs",
    "payout_momentum":    "xs",
    "fcf_yield":          "xs",
    "earnings_revision":  "raw",
    "composite_carry":    "xs",
}


def build_vf_data(
    prices: pd.DataFrame,
    fundamentals: dict,
    returns: pd.DataFrame,
    warmup: int = 504,
    rebal_freq: int = 21,
    cov_lookback: int = 504,
    vol_halflife: int = 21,
    cor_halflife: int = 126,
    extra_signals: Optional[dict] = None,
) -> dict:
    """
    Compute raw signals and covariance matrices at every VF rebalance date.

    This is the slow step (~5 min). Cache the result and reuse with
    apply_vf_params() to test different VF configurations cheaply.

    extra_signals: optional dict of {name: raw_signal_df} to merge with the
                   4 production signals before running VF.

    Returns
    -------
    dict with keys:
        raw_signals  : dict[str, pd.DataFrame]  — raw winsorized signal scores
        cov_matrices : dict[Timestamp, pd.DataFrame]  — cov at each rebal date
        all_dates    : DatetimeIndex
        rebal_dates  : DatetimeIndex
        columns      : Index  — ticker universe
    """
    from optimization import estimate_covariance

    raw_signals: dict[str, pd.DataFrame] = {
        "momentum":                 momentum(prices),
        "tsmom":                    tsmom(prices),
        "mean_reversion":           mean_reversion(prices),
        "carry":                    carry(prices, fundamentals),
        "threshold_mean_reversion": threshold_mean_reversion(prices),
        "earnings_yield_all":       earnings_yield_all(prices, fundamentals["eps_fwd"]),
        "earnings_revision":        earnings_revision(prices, fundamentals["eps_fwd"]),
        **(extra_signals or {}),
    }

    all_dates   = returns.index
    rebal_dates = all_dates[warmup::rebal_freq]

    cov_matrices: dict = {}
    for rd in rebal_dates:
        cov_matrices[rd] = estimate_covariance(
            returns.loc[:rd].iloc[-cov_lookback:],
            vol_halflife=vol_halflife,
            cor_halflife=cor_halflife,
        )

    return {
        "raw_signals":  raw_signals,
        "cov_matrices": cov_matrices,
        "all_dates":    all_dates,
        "rebal_dates":  rebal_dates,
        "columns":      prices.columns,
    }


def apply_vf_params(
    vf_data: dict,
    xs_params: Optional[dict] = None,
    ts_params: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Apply ViewFormation with given params to pre-computed vf_data.

    Fast (~seconds). Use after build_vf_data() to sweep VF configurations
    without repeating the expensive covariance estimation step.

    xs_params : overrides for ViewFormationXS  (use_grinold, use_demean,
                use_ortho, use_tanh, tanh_scale, use_rank, …)
    ts_params : overrides for ViewFormationTS  (use_ortho, desired_vol)
    """
    from equities.view_formation import ViewFormationXS, ViewFormationTS

    raw_signals  = vf_data["raw_signals"]
    cov_matrices = vf_data["cov_matrices"]
    all_dates    = vf_data["all_dates"]
    rebal_dates  = vf_data["rebal_dates"]

    _xs = {"use_rank": False, **(xs_params or {})}

    positions = {
        name: pd.DataFrame(0.0, index=all_dates, columns=vf_data["columns"])
        for name in raw_signals
    }

    for i, rd in enumerate(rebal_dates):
        cov = cov_matrices.get(rd)
        if cov is None:
            continue
        next_rd    = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
        hold_dates = all_dates[(all_dates > rd) & (all_dates <= next_rd)]

        for name, sig_df in raw_signals.items():
            if rd not in sig_df.index:
                continue
            raw    = sig_df.loc[rd].dropna()
            common = raw.index.intersection(cov.index)
            if len(common) < 10:
                continue
            try:
                kind = _SIGNAL_KINDS[name]
                if kind == "raw":
                    s   = raw.loc[common]
                    mu  = s.mean()
                    sd  = s.std()
                    z   = (s - mu) / (sd if sd > 1e-12 else 1.0)
                    gross = z.abs().sum()
                    w   = z / gross if gross > 1e-12 else z
                elif kind == "ts":
                    vf = ViewFormationTS(raw.loc[common], cov.loc[common, common],
                                         params=ts_params)
                    w = vf.generate_view()
                else:
                    vf = ViewFormationXS(raw.loc[common], cov.loc[common, common],
                                         params=_xs)
                    w = vf.generate_view()
            except Exception:
                continue
            # Store at the rebalance date AND hold dates so the backtest
            # picks up new weights immediately (at rd) rather than one full
            # cycle later (at next_rd).  Signal at rd uses prices through
            # rd-1 (via .shift(1)), so there is no look-ahead.
            positions[name].loc[rd, w.index] = w.values
            for d in hold_dates:
                positions[name].loc[d, w.index] = w.values

    return positions


def build_vf_positions(
    prices: pd.DataFrame,
    fundamentals: dict,
    returns: pd.DataFrame,
    warmup: int = 504,
    rebal_freq: int = 21,
    cov_lookback: int = 504,
    vol_halflife: int = 21,
    cor_halflife: int = 126,
    vf_params: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Canonical entry point for equities signals.

    Computes all 4 signals, runs each through ViewFormation on a rolling
    monthly rebalance schedule, and returns pre-formed position DataFrames
    ready to pass directly into equities/blend.py.

    For repeated runs with different VF params, use build_vf_data() +
    apply_vf_params() directly to avoid recomputing covariance matrices.
    """
    vf_data = build_vf_data(
        prices, fundamentals, returns,
        warmup, rebal_freq, cov_lookback, vol_halflife, cor_halflife,
    )
    return apply_vf_params(vf_data, xs_params=vf_params)
