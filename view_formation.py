"""
View Formation Framework for Cross-Sectional Signals.

Transforms a raw XS signal into a risk-consistent, beta-neutral, vol-scaled
portfolio view. Pipeline follows the documented procedure (Francesco Caccia
Projects.pdf — "View Formation Framework for XS Signals").

Steps (all applied per rebalance date):
  1.  Estimate betas via vol-weighted market portfolio
  2.  Estimate idiosyncratic volatilities
  3.  Rank signal (optional)
  4.  Demean (cross-sectional)
  5.  Divide by idiosyncratic vol (Grinold — signal → conviction)
  6.  Orthogonalize to market (remove PC1 exposure)
  7.  Z-score pre-tanh
  8.  tanh compression
  9.  Re-orthogonalize (tanh reintroduces beta)
  10. Normalize to 2× leverage
  11. Combine raw and rank signals (optional)
  12. Re-normalize leverage after blend
  13. Scale to target volatility
"""

import numpy as np
import pandas as pd
from typing import Optional


_DEFAULT_TS_PARAMS = {
    "desired_vol": 0.10,
}

_DEFAULT_PARAMS = {
    "tanh_scale":  0.75,
    "pct_prop":    0.5,
    "desired_vol": 0.10,
    "use_rank":    False,
    "use_grinold": True,
    "use_demean":  True,
}


class ViewFormationXS:
    """
    Convert a raw cross-sectional signal into a beta-neutral, vol-scaled view.

    Parameters
    ----------
    signal : pd.Series
        Raw XS signal indexed by ticker (one rebalance date).
    cov : pd.DataFrame
        Covariance matrix (ticker × ticker), e.g. from optimization.estimate_covariance.
    params : dict, optional
        tanh_scale  — compression strength (default 1.25)
        pct_prop    — weight on rank signal in final blend; 0 = raw only,
                      1 = rank only, 0.5 = equal blend (default 0.5)
        desired_vol — annualized target vol for the output view (default 0.10)
        use_rank    — whether to compute and blend the rank signal (default True)
        use_grinold — whether to divide by idio_vol in step 5 (default True)
    """

    def __init__(
        self,
        signal: pd.Series,
        cov: pd.DataFrame,
        params: Optional[dict] = None,
    ):
        p = {**_DEFAULT_PARAMS, **(params or {})}
        self.tanh_scale  = p["tanh_scale"]
        self.pct_prop    = p["pct_prop"]
        self.desired_vol = p["desired_vol"]
        self.use_rank    = p["use_rank"]
        self.use_grinold = p["use_grinold"]
        self.use_demean  = p["use_demean"]

        # Align signal and covariance to common tickers
        common = signal.dropna().index.intersection(cov.index)
        self.signal = signal.loc[common].astype(float)
        self.cov    = cov.loc[common, common]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_view(self) -> pd.Series:
        """Run the full pipeline and return portfolio weights (pd.Series, ticker index)."""
        betas, idio_vols = self._risk_decomposition()

        raw_view  = self._process_signal(self.signal.copy(), betas, idio_vols)
        if self.use_rank:
            rank_sig  = self._rank_signal(self.signal)
            rank_view = self._process_signal(rank_sig, betas, idio_vols)
            combined  = self.pct_prop * rank_view + (1 - self.pct_prop) * raw_view
        else:
            combined = raw_view

        # Re-normalize leverage after blending (blend of two 2x views may drift at coverage edges)
        combined = self._normalize_leverage(combined)

        # Step 12 → Step 13: Scale to target volatility (desired_vol is annualized → convert to daily)
        combined = self._scale_to_vol(combined)

        return combined

    # ------------------------------------------------------------------
    # Pipeline internals
    # ------------------------------------------------------------------

    def _risk_decomposition(self):
        """Steps 1–2: betas and idiosyncratic vols from the covariance matrix."""
        vols = np.sqrt(np.diag(self.cov.values))
        vols = pd.Series(vols, index=self.cov.index).replace(0, np.nan).dropna()

        # Vol-weighted market portfolio (inverse-vol weights)
        w_mkt = (1.0 / vols)
        w_mkt = w_mkt / w_mkt.sum()
        w_mkt = w_mkt.reindex(self.cov.index).fillna(0.0)

        cov_arr = self.cov.values
        w_arr   = w_mkt.values

        # Portfolio variance
        port_var = float(w_arr @ cov_arr @ w_arr)

        # Beta_i = Cov(r_i, r_mkt) / Var(r_mkt) = (Σ w_mkt)_i / port_var
        betas = pd.Series(cov_arr @ w_arr / port_var, index=self.cov.index)

        # Idiosyncratic vol: sqrt(total_var - systematic_var)
        # Floor at 5% of total variance so high-beta stocks don't blow up in Grinold scaling
        total_var = pd.Series(np.diag(cov_arr), index=self.cov.index)
        sys_var   = (betas ** 2) * port_var
        idio_var  = (total_var - sys_var).clip(lower=total_var * 0.05)
        idio_vols = np.sqrt(idio_var)

        return betas, idio_vols

    def _process_signal(
        self,
        s: pd.Series,
        betas: pd.Series,
        idio_vols: pd.Series,
    ) -> pd.Series:
        """Steps 4–10 applied to a single signal vector."""
        # Step 4: Demean (optional)
        if self.use_demean:
            s = s - s.mean()

        # Step 5: Grinold — divide by idio_vol to convert signal → conviction (optional)
        if self.use_grinold:
            s = s / idio_vols.reindex(s.index).replace(0, np.nan)
            s = s.dropna()

        # Steps 6 + 9 share the same orthogonalization
        s = self._orthogonalize(s, betas.reindex(s.index))

        # Step 7: Z-score pre-tanh
        s = self._zscore(s)

        # Step 8: tanh compression
        s = np.tanh(s * self.tanh_scale)

        # Step 9: Re-orthogonalize
        s = self._orthogonalize(s, betas.reindex(s.index))

        # Step 10: Normalize to 2× leverage
        s = self._normalize_leverage(s)

        return s

    def _rank_signal(self, s: pd.Series) -> pd.Series:
        """Step 3: Ordinal rank signal in [0, 1]."""
        return s.rank(pct=True)

    def _orthogonalize(self, s: pd.Series, betas: pd.Series) -> pd.Series:
        """Project out the beta component: s -= (s·b / b·b) * b."""
        b = betas.reindex(s.index).fillna(0.0)
        b_dot_b = float(b @ b)
        if b_dot_b < 1e-12:
            return s
        coeff = float(s @ b) / b_dot_b
        return s - coeff * b

    def _zscore(self, s: pd.Series) -> pd.Series:
        std = s.std()
        if std < 1e-12:
            return s
        return (s - s.mean()) / std

    def _normalize_leverage(self, s: pd.Series) -> pd.Series:
        longs  = s[s > 0]
        shorts = s[s < 0]
        out = s.copy()
        if longs.sum() > 1e-12:
            out[s > 0] = longs / longs.sum()
        if shorts.sum() < -1e-12:
            out[s < 0] = shorts / shorts.abs().sum()
        return out

    def _scale_to_vol(self, weights: pd.Series) -> pd.Series:
        """Step 13: Scale portfolio to desired_vol (annualized). Cov is in daily units."""
        desired_vol_daily = self.desired_vol / np.sqrt(252)
        w = weights.reindex(self.cov.index).fillna(0.0).values
        port_var = float(w @ self.cov.values @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        return weights * (desired_vol_daily / port_vol)


class ViewFormationTS:
    """
    Time-series momentum view: signal ÷ annualized total vol → beta-ortho → vol scale.

    Parameters
    ----------
    signal : pd.Series
        Raw TSMOM signal indexed by ticker (one rebalance date).
    cov : pd.DataFrame
        Covariance matrix (daily units, EWMA halflife=21). Used for total-vol
        normalization and portfolio vol scaling.
    params : dict, optional
        desired_vol — annualized target vol for the output view (default 0.10)
    """

    def __init__(
        self,
        signal: pd.Series,
        cov: pd.DataFrame,
        params=None,
    ):
        p = {**_DEFAULT_TS_PARAMS, **(params or {})}
        self.desired_vol = p["desired_vol"]
        common = signal.dropna().index.intersection(cov.index)
        self.signal = signal.loc[common].astype(float)
        self.cov    = cov.loc[common, common]

    def generate_view(self) -> pd.Series:
        """Run the TSMOM pipeline and return portfolio weights."""
        # Step 1: signal / annualized total vol from cov diagonal (Sharpe-ratio-like)
        total_vols = pd.Series(
            np.sqrt(np.diag(self.cov.values)) * np.sqrt(252),
            index=self.cov.index,
        ).replace(0, np.nan)
        s = (self.signal / total_vols).dropna()

        # Step 2: orthogonalize to market beta (removes directional market exposure)
        betas, _ = self._risk_decomposition()
        s = self._orthogonalize(s, betas.reindex(s.index))

        # Step 3: portfolio vol scale
        return self._scale_to_vol(s)

    def _risk_decomposition(self):
        """Betas and idiosyncratic vols from the covariance matrix."""
        vols = np.sqrt(np.diag(self.cov.values))
        vols = pd.Series(vols, index=self.cov.index).replace(0, np.nan).dropna()

        w_mkt = (1.0 / vols)
        w_mkt = w_mkt / w_mkt.sum()
        w_mkt = w_mkt.reindex(self.cov.index).fillna(0.0)

        cov_arr  = self.cov.values
        w_arr    = w_mkt.values
        port_var = float(w_arr @ cov_arr @ w_arr)
        betas    = pd.Series(cov_arr @ w_arr / port_var, index=self.cov.index)

        total_var = pd.Series(np.diag(cov_arr), index=self.cov.index)
        sys_var   = (betas ** 2) * port_var
        idio_var  = (total_var - sys_var).clip(lower=total_var * 0.05)
        idio_vols = np.sqrt(idio_var)

        return betas, idio_vols

    def _orthogonalize(self, s: pd.Series, betas: pd.Series) -> pd.Series:
        """Project out the beta component: s -= (s·b / b·b) * b."""
        b = betas.reindex(s.index).fillna(0.0)
        b_dot_b = float(b @ b)
        if b_dot_b < 1e-12:
            return s
        coeff = float(s @ b) / b_dot_b
        return s - coeff * b

    def _scale_to_vol(self, weights: pd.Series) -> pd.Series:
        """Scale portfolio to desired_vol (annualized). Cov is in daily units."""
        desired_vol_daily = self.desired_vol / np.sqrt(252)
        w = weights.reindex(self.cov.index).fillna(0.0).values
        port_var = float(w @ self.cov.values @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        return weights * (desired_vol_daily / port_vol)
