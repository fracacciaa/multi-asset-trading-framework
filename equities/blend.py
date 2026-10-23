"""
Signal blending for equities.

Combines raw signal score DataFrames (date × ticker, output of equities/signals.py,
before ViewFormation) into a single composite signal.

Pipeline position in the stack:
    equities/signals.py  →  equities/blend.py  →  view_formation.py  →  portfolio weights

Static blend methods (return BlendWeights):
    equal_weight        — 1/K on each signal (benchmark)
    sharpe_heuristic    — weight by max(0, Sharpe_k)^p (benchmark)
    msrr_ridge          — Ridge MSRR: b = (M + λI)⁻¹ μ
    enet_blend          — Elastic Net / LASSO blend (Gu et al. 2020)
    c_enet_blend        — Combination ENet: select then equal-weight (Rapach & Zhou 2020)
    e_enet_blend        — Egalitarian ENet: select, then force 1/l (Scholz 2022)
    inv_msfe_blend      — Inverse-MSFE weighting (Mekelburg & Strauss 2023)
    pcr_blend           — Principal Components Regression (Gu et al. 2020)
    pls_blend           — Partial Least Squares (Gu et al. 2020)

Preprocessing utilities:
    winsorize_signals   — clip signal scores at cross-sectional quantile (Mekelburg & Strauss 2023)
    pool_blend          — ensemble pooling of multiple BlendWeights (Mekelburg & Strauss 2023)

Dynamic / ML methods (time-varying weights or direct positions):
    MLMSRR              — PyTorch MLP: z_t → b_t  (objectives: quadratic / sharpe / huber)
    LinearDynamicBlend  — Linear IPCA-style: b_t = Γ₀ + Γ₁ z_t  (Giglio et al. 2022)
    GBRTBlend           — Gradient Boosted trees: z_t → μ̂_k → MSRR weights (Gu et al. 2020)
    RandomForestBlend   — Random Forest analog of GBRTBlend
    DMABlend            — Dynamic Model Averaging with forgetting (Mekelburg & Strauss 2023)
    STMOM               — Multitask SLP/MLP direct portfolio model (Tan et al. 2023)

For the statistical methods (Sharpe, Ridge, ENet …), blend weights are estimated from
managed returns F_k, where each sleeve k earns F_k[t+1] = Σᵢ p[i,k,t] r[i,t+1].
Positions p[i,k,t] are constructed inside this module from raw signal scores via
simple cross-sectional z-scoring (no ViewFormation required here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── Internal position construction ─────────────────────────────────────────────

def _xs_positions(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score → gross-normalised positions.

    Used only internally to construct managed returns for the statistical estimators.
    NaN tickers are excluded from the z-score computation and carry NaN positions.
    """
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1).replace(0.0, np.nan)
    z = signal.sub(mu, axis=0).div(sd, axis=0)
    gross = z.abs().sum(axis=1).replace(0.0, np.nan)
    return z.div(gross, axis=0)


# ── Managed-return construction ─────────────────────────────────────────────────

def managed_return(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Scalar managed portfolio return for one signal sleeve.

    F_k[t+1] = Σᵢ p[i,k,t] · r[i,t+1]

    Positions are constructed from the raw signal via _xs_positions (XS z-score,
    gross-normalised). NaN positions and returns are treated as zero.
    """
    pos = _xs_positions(signal)
    common_dates = pos.index.intersection(forward_returns.index)
    common_tickers = pos.columns.intersection(forward_returns.columns)
    p = pos.loc[common_dates, common_tickers].fillna(0.0)
    r = forward_returns.loc[common_dates, common_tickers].fillna(0.0)
    return (p * r).sum(axis=1).rename("managed_return")


def build_managed_returns(
    signals: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a T × K DataFrame of managed portfolio returns.

    Each column k contains the scalar return series for signal k.
    Rows with any NaN are dropped so all estimators share an identical sample.
    """
    F = {name: managed_return(sig, forward_returns) for name, sig in signals.items()}
    return pd.DataFrame(F).dropna()


# ── Blend weights container ─────────────────────────────────────────────────────

@dataclass
class BlendWeights:
    """Static blend weights with method label and optional diagnostics."""

    weights: pd.Series
    method: str
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        total = self.weights.abs().sum()
        if total > 1e-12:
            self.weights = self.weights / total

    def __repr__(self) -> str:
        w = self.weights.round(4).to_dict()
        return f"BlendWeights({self.method}): {w}"


# ── Static estimators ───────────────────────────────────────────────────────────

def equal_weight(signal_names: list[str]) -> BlendWeights:
    """1/K weight on every signal. No managed returns required."""
    K = len(signal_names)
    w = pd.Series(1.0 / K, index=signal_names)
    return BlendWeights(w, "equal_weight")


def sharpe_heuristic(F: pd.DataFrame, power: int = 1) -> BlendWeights:
    """
    Weight by max(0, SR_k)^power, normalised to sum to 1.

    SR_k = mean(F_k) / std(F_k). Discards sleeves with non-positive estimated Sharpe.
    Falls back to equal_weight if all Sharpe estimates are ≤ 0.

    power=1 (SH1) is proportional to Sharpe; power=2 (SH2) concentrates more.
    """
    mu = F.mean()
    sd = F.std().replace(0.0, np.nan)
    sr = mu / sd
    scores = sr.clip(lower=0.0) ** power
    total = scores.sum()
    fallback = bool(total < 1e-12)
    if fallback:
        w = pd.Series(1.0 / len(F.columns), index=F.columns)
    else:
        w = scores / total
    return BlendWeights(
        w,
        f"sharpe_heuristic_p{power}",
        {"sharpe": sr.to_dict(), "fallback_to_ew": fallback},
    )


def msrr_ridge(F: pd.DataFrame, lam: float = 0.0) -> BlendWeights:
    """
    Ridge MSRR: b = (M + λI)⁻¹ μ,  M = F'F/T,  μ = mean(F).

    When λ=0 this is OLS MSRR (numerically solved via lstsq for stability).
    Directionally equivalent to classical max-Sharpe (Σ⁻¹μ) when λ=0 and
    the consistent sample covariance convention is used — this equivalence
    serves as an algebraic identity check (see tests).

    λ interpretation: regularisation is on the scale of mean(diag(M)).
    A reasonable starting grid: [0, 0.01, 0.1, 1.0] × mean(diag(M)).
    """
    T = len(F)
    Fv = F.values.astype(float)
    mu = Fv.mean(axis=0)
    M = Fv.T @ Fv / T
    K = M.shape[0]

    A = M + lam * np.eye(K)
    b, _, rank, sv = np.linalg.lstsq(A, mu, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else np.inf

    w = pd.Series(b, index=F.columns)
    return BlendWeights(
        w,
        "msrr_ols" if lam == 0.0 else "msrr_ridge",
        {"lambda": lam, "cond": cond, "rank": int(rank)},
    )


# ── Applying blend weights ──────────────────────────────────────────────────────

def combine_signals(
    signals: dict[str, pd.DataFrame],
    weights: BlendWeights,
) -> pd.DataFrame:
    """
    Weighted sum of cross-sectionally z-scored signal DataFrames.

    Each signal is z-scored per date before blending so that signals with
    different natural scales contribute according to the blend weights alone.
    Signals absent from `weights` are silently skipped; weights are renormalised
    on the available subset.

    Output is a combined raw signal (date × ticker) ready to feed ViewFormationXS.
    """
    w = weights.weights
    available = {k: signals[k] for k in w.index if k in signals}
    if not available:
        raise ValueError(
            f"No signals match blend weights. "
            f"Weight keys: {list(w.index)}, signal keys: {list(signals.keys())}"
        )

    w_avail = w.reindex(list(available.keys())).dropna()
    w_avail = w_avail / w_avail.abs().sum()

    idx = sorted(set.union(*(set(df.index) for df in available.values())))
    cols = sorted(set.union(*(set(df.columns) for df in available.values())))

    combo = pd.DataFrame(0.0, index=idx, columns=cols)
    for name, wk in w_avail.items():
        sig = available[name]
        mu = sig.mean(axis=1)
        sd = sig.std(axis=1).replace(0.0, np.nan)
        z = sig.sub(mu, axis=0).div(sd, axis=0)
        z = z.reindex(index=idx, columns=cols).fillna(0.0)
        combo = combo.add(wk * z)

    return combo


def fit_static(
    signals: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ridge_lam: float = 0.01,
    enet_alpha: float = 0.01,
    enet_l1_ratio: float = 0.5,
    n_components: int = 2,
) -> dict[str, BlendWeights]:
    """
    Convenience: fit all static combination methods on a single training window.

    Returns a dict with keys (static methods only; dynamic methods require separate fit):
        equal_weight, sh_p1, sh_p2, msrr_ols, msrr_ridge  (original)
        enet, lasso, c_enet, e_enet, e_lasso, inv_msfe, pcr, pls  (new)
    """
    names = list(signals.keys())
    F = build_managed_returns(signals, forward_returns)
    result: dict[str, BlendWeights] = {
        "equal_weight": equal_weight(names),
        "sh_p1": sharpe_heuristic(F, power=1),
        "sh_p2": sharpe_heuristic(F, power=2),
        "msrr_ols": msrr_ridge(F, lam=0.0),
        "msrr_ridge": msrr_ridge(F, lam=ridge_lam),
    }
    result["enet"] = enet_blend(F, alpha=enet_alpha, l1_ratio=enet_l1_ratio)
    result["lasso"] = enet_blend(F, alpha=enet_alpha, l1_ratio=1.0)
    result["c_enet"] = c_enet_blend(F, alpha=enet_alpha, l1_ratio=enet_l1_ratio)
    result["e_enet"] = e_enet_blend(F, alpha=enet_alpha, l1_ratio=enet_l1_ratio)
    result["e_lasso"] = e_enet_blend(F, alpha=enet_alpha, l1_ratio=1.0)
    result["inv_msfe"] = inv_msfe_blend(F)
    result["pcr"] = pcr_blend(F, n_components=n_components)
    result["pls"] = pls_blend(F, n_components=n_components)
    return result


# ── ML MSRR ────────────────────────────────────────────────────────────────────

def build_state_features(F: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """
    Build lagged state features z_t for ML MSRR.

    All features use F.shift(1) so z_t only contains F[t-1], F[t-2], …
    (F[t] is a forward return not observable until t+1):
      - Rolling Sharpe of each sleeve k  (K features)
      - Rolling cross-sectional dispersion across sleeves  (1 feature)
      - Rolling average return proxy (market direction)  (1 feature)

    Features are standardised using expanding-window mean/std so that parameters
    fit on training data can be applied at test time without data leakage.

    The first `lookback` rows will be NaN and should be dropped before calling fit().
    """
    F_lag = F.shift(1)   # F_lag[t] = F[t-1]: no lookahead
    roll_mean = F_lag.rolling(lookback, min_periods=lookback // 2).mean()
    roll_std = F_lag.rolling(lookback, min_periods=lookback // 2).std().replace(0.0, np.nan)
    sharpe_features = roll_mean / roll_std
    sharpe_features.columns = [f"sr_{c}" for c in F.columns]

    dispersion = F_lag.std(axis=1).rolling(lookback, min_periods=lookback // 2).mean()
    dispersion.name = "dispersion"

    avg_ret = F_lag.mean(axis=1).rolling(lookback, min_periods=lookback // 2).mean()
    avg_ret.name = "avg_ret"

    z = pd.concat([sharpe_features, dispersion, avg_ret], axis=1)

    # Expanding-window standardisation (no future data used)
    exp_mean = z.expanding(min_periods=2).mean()
    exp_std = z.expanding(min_periods=2).std().replace(0.0, np.nan)
    z_std = (z - exp_mean) / exp_std

    return z_std


class _BlendMLP(nn.Module):
    """Inner MLP: z_t → time-varying blend weights b_t."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        n_signals: int,
        output_activation: str,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_signals),
        )
        self.output_activation = output_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        # "norm_abs": signed weights, normalised by sum of absolute values
        return logits / (logits.abs().sum(dim=-1, keepdim=True) + 1e-8)


def _quadratic_loss(
    weights: torch.Tensor,
    F: torch.Tensor,
    lam: float,
    model: nn.Module,
) -> torch.Tensor:
    """Quadratic MSRR: mean((1 - (b_t · F_{t+1}))²) + λ ‖θ‖²."""
    R = (weights * F).sum(dim=-1)
    loss = ((1.0 - R) ** 2).mean()
    if lam > 0:
        l2 = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + lam * l2
    return loss


def _sharpe_loss(
    weights: torch.Tensor,
    F: torch.Tensor,
    lam: float,
    model: nn.Module,
    eps: float,
) -> torch.Tensor:
    """Direct Sharpe: −mean(R)/sqrt(var(R)+ε) + λ ‖θ‖² (negative because we minimise)."""
    R = (weights * F).sum(dim=-1)
    mean_R = R.mean()
    var_R = R.var(unbiased=False)
    sharpe = mean_R / (var_R + eps).sqrt()
    loss = -sharpe
    if lam > 0:
        l2 = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + lam * l2
    return loss


def _huber_loss(
    weights: torch.Tensor,
    F: torch.Tensor,
    lam: float,
    model: nn.Module,
    xi: float,
) -> torch.Tensor:
    """
    Huber robust MSRR (Gu, Kelly & Xiu 2020).

    H(r; ξ) = r²              if |r| ≤ ξ
             = 2ξ|r| − ξ²     otherwise
    applied to the MSRR residual r = 1 − R_t.

    Reduces the influence of extreme managed-return outliers.
    ξ ≈ 0.01 corresponds to the ~99th percentile of daily portfolio returns.
    """
    R = (weights * F).sum(dim=-1)
    residual = 1.0 - R
    abs_r = residual.abs()
    huber = torch.where(abs_r <= xi, residual ** 2, 2.0 * xi * abs_r - xi ** 2)
    loss = huber.mean()
    if lam > 0:
        l2 = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + lam * l2
    return loss


class MLMSRR:
    """
    Small PyTorch MLP producing time-varying blend weights b_t = g_θ(z_t).

    Architecture:
        z_t → Linear(D_z, H) → ReLU → Dropout(p) → Linear(H, K) → activation

    output_activation options:
        "softmax"   long-only, nonneg, sums to 1  (default)
        "norm_abs"  signed: w / |w|.sum()

    Objective options (spec §6.1):
        "quadratic"  minimise mean((1 - b_t' F_{t+1})²) + λ‖θ‖²
        "sharpe"     maximise mean(R)/sqrt(var(R)+ε) − λ‖θ‖²
        "huber"      Huber-robust MSRR: H(1−R_t; ξ) + λ‖θ‖²  (Gu et al. 2020)

    Training: Adam, gradient clipping, early stopping on a chronological
    validation tail. Multiple seeds are run; best-by-validation-loss is kept.
    z_t must be lagged so that z[t] is known strictly before F[t+1] is observed.
    """

    def __init__(
        self,
        n_signals: int,
        hidden_size: int = 16,
        output_activation: str = "softmax",
        objective: str = "quadratic",
        lam: float = 1e-3,
        eps: float = 1e-6,
        xi: float = 0.01,
        dropout: float = 0.1,
        lr: float = 1e-3,
        n_epochs: int = 500,
        patience: int = 30,
        val_frac: float = 0.2,
        grad_clip: float = 1.0,
        seeds: list[int] = (42,),
    ):
        self.n_signals = n_signals
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.objective = objective
        self.lam = lam
        self.eps = eps
        self.xi = xi
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.grad_clip = grad_clip
        self.seeds = list(seeds)

        self._model: Optional[_BlendMLP] = None
        self._z_mean: Optional[np.ndarray] = None
        self._z_std: Optional[np.ndarray] = None
        self._signal_names: Optional[list[str]] = None

    def _loss_fn(
        self,
        weights: torch.Tensor,
        F_t: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        if self.objective == "quadratic":
            return _quadratic_loss(weights, F_t, self.lam, model)
        if self.objective == "huber":
            return _huber_loss(weights, F_t, self.lam, model, self.xi)
        return _sharpe_loss(weights, F_t, self.lam, model, self.eps)

    def _train_one_seed(
        self,
        z_tr: torch.Tensor,
        F_tr: torch.Tensor,
        z_va: torch.Tensor,
        F_va: torch.Tensor,
        input_dim: int,
        seed: int,
    ) -> tuple[_BlendMLP, float]:
        torch.manual_seed(seed)
        model = _BlendMLP(
            input_dim,
            self.hidden_size,
            self.n_signals,
            self.output_activation,
            self.dropout,
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_val = math.inf
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0

        for epoch in range(self.n_epochs):
            model.train()
            opt.zero_grad()
            w = model(z_tr)
            loss = self._loss_fn(w, F_tr, model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()

            model.eval()
            with torch.no_grad():
                w_val = model(z_va)
                val_loss = self._loss_fn(w_val, F_va, model).item()

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        model.load_state_dict(best_state)
        return model, best_val

    def fit(self, F_train: pd.DataFrame, z_train: pd.DataFrame) -> "MLMSRR":
        """
        Fit the MLP on chronological training data.

        F_train : T × K managed returns (output of build_managed_returns)
        z_train : T × D_z state features (output of build_state_features),
                  already aligned with F_train and with NaN rows dropped.

        z_train must be lagged: z_train.iloc[t] must only use information
        available strictly before F_train.iloc[t] is observed.
        """
        common = F_train.index.intersection(z_train.index)
        F_train = F_train.loc[common]
        z_train = z_train.loc[common].dropna()
        common = F_train.index.intersection(z_train.index)
        F_arr = F_train.loc[common].values.astype(np.float32)
        z_arr = z_train.loc[common].values.astype(np.float32)

        self._signal_names = list(F_train.columns)

        # Standardise z using training-only stats (for later predict)
        n_val = max(1, int(len(z_arr) * self.val_frac))
        n_tr = len(z_arr) - n_val
        self._z_mean = z_arr[:n_tr].mean(axis=0)
        self._z_std = z_arr[:n_tr].std(axis=0)
        self._z_std[self._z_std < 1e-8] = 1.0
        z_norm = (z_arr - self._z_mean) / self._z_std

        z_tr_t = torch.tensor(z_norm[:n_tr])
        F_tr_t = torch.tensor(F_arr[:n_tr])
        z_va_t = torch.tensor(z_norm[n_tr:])
        F_va_t = torch.tensor(F_arr[n_tr:])
        input_dim = z_norm.shape[1]

        best_model, best_val = None, math.inf
        for seed in self.seeds:
            model, val = self._train_one_seed(
                z_tr_t, F_tr_t, z_va_t, F_va_t, input_dim, seed
            )
            if val < best_val:
                best_val = val
                best_model = model

        self._model = best_model
        return self

    def predict_weights(self, z: pd.DataFrame) -> pd.DataFrame:
        """
        Return T × K DataFrame of time-varying blend weights.

        Uses stored standardisation stats from fit(). NaN rows in z
        produce NaN weight rows in the output.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_weights().")

        nan_mask = z.isna().any(axis=1)
        z_arr = z.values.astype(np.float32)
        z_norm = (z_arr - self._z_mean) / self._z_std

        self._model.eval()
        with torch.no_grad():
            w = self._model(torch.tensor(z_norm)).numpy()

        out = pd.DataFrame(w, index=z.index, columns=self._signal_names)
        out.loc[nan_mask] = np.nan
        return out

    def combine_signals(
        self,
        signals: dict[str, pd.DataFrame],
        z: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply time-varying blend weights to raw signal DataFrames.

        Each signal is cross-sectionally z-scored before blending.
        Rows where z is NaN (no state features yet) produce NaN output rows.
        """
        weights_df = self.predict_weights(z)

        idx = sorted(set.union(*(set(df.index) for df in signals.values())))
        cols = sorted(set.union(*(set(df.columns) for df in signals.values())))

        combo = pd.DataFrame(0.0, index=idx, columns=cols)
        nan_rows = weights_df.isna().any(axis=1)

        for t in weights_df.index:
            if nan_rows.loc[t]:
                combo.loc[t] = np.nan
                continue
            b_t = weights_df.loc[t]
            row = pd.Series(0.0, index=cols)
            for name, wk in b_t.items():
                if name not in signals:
                    continue
                sig = signals[name]
                if t not in sig.index:
                    continue
                cross = sig.loc[t]
                mu, sd = cross.mean(), cross.std()
                if sd > 1e-12:
                    cross = (cross - mu) / sd
                row = row.add(wk * cross.reindex(cols).fillna(0.0))
            combo.loc[t] = row.values

        return combo


# ── Penalized linear blends ────────────────────────────────────────────────────

def enet_blend(
    F: pd.DataFrame,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    non_negative: bool = False,
) -> BlendWeights:
    """
    Elastic Net blend (Gu, Kelly & Xiu 2020).

    Fits ElasticNet(alpha, l1_ratio) with no intercept to the MSRR target y=1,
    identical in spirit to OLS MSRR but with an explicit penalty:
        α · (l1_ratio · ‖b‖₁  +  (1−l1_ratio)/2 · ‖b‖₂²)

    l1_ratio=1 → LASSO (sparse); l1_ratio=0 → Ridge; 0<l1_ratio<1 → ENet.
    non_negative=True enforces b_k ≥ 0 (required by C-ENet variant).
    Falls back to equal weight if regularization zeros all coefficients.
    """
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as e:
        raise ImportError("sklearn is required for enet_blend.") from e

    y = np.ones(len(F))
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        positive=non_negative,
        max_iter=10_000,
    )
    model.fit(F.values, y)
    b = model.coef_.copy()
    n_selected = int((np.abs(b) > 1e-8).sum())

    w = pd.Series(b, index=F.columns)
    if w.abs().sum() < 1e-12:
        w = pd.Series(1.0 / len(F.columns), index=F.columns)
        fallback = True
        if l1_ratio >= 1.0:
            method = "lasso_ew_fallback"
        else:
            method = "enet_ew_fallback"
    else:
        fallback = False
        method = "lasso" if l1_ratio >= 1.0 else ("enet_ridge" if l1_ratio <= 0.0 else "enet")

    return BlendWeights(
        w, method,
        {"alpha": alpha, "l1_ratio": l1_ratio, "n_selected": n_selected, "fallback": fallback},
    )


def c_enet_blend(
    F: pd.DataFrame,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> BlendWeights:
    """
    Combination Elastic Net (C-ENet, Rapach & Zhou 2020).

    Step 1 — ENet with non-negative constraint selects J = {k : coef_k > 0}.
    Step 2 — Equal-weight among J (regression coefficients discarded after selection).

    The distinction from enet_blend: C-ENet uses penalized regression only for
    signal *selection*, then reverts to equal weight among the selected subset.
    Falls back to equal weight over all signals if no signal is selected.
    """
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as e:
        raise ImportError("sklearn is required for c_enet_blend.") from e

    y = np.ones(len(F))
    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, positive=True, max_iter=10_000
    )
    model.fit(F.values, y)

    selected = [name for name, c in zip(F.columns, model.coef_) if c > 1e-8]
    fallback = len(selected) == 0
    if fallback:
        selected = list(F.columns)

    n_sel = len(selected)
    w = pd.Series(0.0, index=F.columns)
    for name in selected:
        w[name] = 1.0 / n_sel

    return BlendWeights(
        w, "c_enet",
        {"alpha": alpha, "l1_ratio": l1_ratio, "selected": selected,
         "n_selected": n_sel, "fallback": fallback},
    )


def e_enet_blend(
    F: pd.DataFrame,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> BlendWeights:
    """
    Egalitarian Elastic Net (eENet / eLasso, Scholz 2022).

    Step 1 — ENet (no non-negativity) selects l = |{k : |coef_k| > 0}| active signals.
    Step 2 — Assign weight 1/l to each active signal, 0 to the rest.

    Decouples selection (ENet) from weighting (always egalitarian among selected).
    l1_ratio=1.0 → eLasso; l1_ratio=0.0 → eRidge.
    Falls back to equal weight over all signals if all coefficients are zeroed.
    """
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as e:
        raise ImportError("sklearn is required for e_enet_blend.") from e

    y = np.ones(len(F))
    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10_000
    )
    model.fit(F.values, y)

    active = [name for name, c in zip(F.columns, model.coef_) if abs(c) > 1e-8]
    fallback = len(active) == 0
    if fallback:
        active = list(F.columns)

    l = len(active)
    w = pd.Series(0.0, index=F.columns)
    for name in active:
        w[name] = 1.0 / l

    if l1_ratio >= 1.0:
        method = "e_lasso"
    elif l1_ratio <= 0.0:
        method = "e_ridge"
    else:
        method = "e_enet"

    return BlendWeights(
        w, method,
        {"alpha": alpha, "l1_ratio": l1_ratio, "selected": active,
         "n_selected": l, "fallback": fallback},
    )


def inv_msfe_blend(F: pd.DataFrame) -> BlendWeights:
    """
    Inverse Mean Squared Forecast Error weighting (Mekelburg & Strauss 2023).

    MSFE_k = mean((F_k − F̄)²)  where F̄ = mean(F, axis=1) is the cross-signal mean.

    w_k ∝ 1/MSFE_k — signals closest to ensemble consensus receive higher weight.
    A floor of 1e-10 prevents division by zero; if MSFE_k = 0 (perfect agreement)
    the signal dominates.
    """
    F_mean = F.mean(axis=1)
    msfe = ((F.sub(F_mean, axis=0)) ** 2).mean()
    inv = 1.0 / msfe.clip(lower=1e-10)
    w = inv / inv.sum()
    return BlendWeights(w, "inv_msfe", {"msfe": msfe.to_dict()})


def pcr_blend(F: pd.DataFrame, n_components: int = 2) -> BlendWeights:
    """
    Principal Components Regression blend (PCR, Gu et al. 2020).

    Step 1 — PCA on F → n_comp principal-component scores (T × n_comp).
    Step 2 — OLS: lstsq(scores, ones(T)) → component coefficients c.
    Step 3 — Map back to signal space: b = V.T @ c  (V = pca.components_, n_comp × K).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("sklearn is required for pcr_blend.") from e

    Fv = F.values.astype(float)
    T, K = Fv.shape
    n_comp = min(n_components, K, T - 1)

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Fv)  # T × n_comp

    c, _, _, _ = np.linalg.lstsq(scores, np.ones(T), rcond=None)
    b = pca.components_.T @ c  # K

    w = pd.Series(b, index=F.columns)
    return BlendWeights(
        w, f"pcr_{n_comp}",
        {"n_components": n_comp, "var_explained": float(pca.explained_variance_ratio_.sum())},
    )


def pls_blend(F: pd.DataFrame, n_components: int = 2) -> BlendWeights:
    """
    Partial Least Squares blend (PLS, Gu et al. 2020).

    PLSRegression with target y=ones(T) (MSRR target).
    PLS maximises covariance between F components and y, weighting directions
    that best predict the portfolio return unit vector.
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
    except ImportError as e:
        raise ImportError("sklearn is required for pls_blend.") from e

    Fv = F.values.astype(float)
    T, K = Fv.shape
    n_comp = min(n_components, K, T - 1)

    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(Fv, np.ones(T).reshape(-1, 1))

    # coef_ shape varies across sklearn versions: can be (K, 1) or (1, K)
    coef = pls.coef_
    b = coef.ravel()
    if len(b) != K:
        b = np.zeros(K)

    w = pd.Series(b, index=F.columns)
    return BlendWeights(w, f"pls_{n_comp}", {"n_components": n_comp})


# ── Preprocessing utility ──────────────────────────────────────────────────────

def winsorize_signals(
    signals: dict[str, pd.DataFrame],
    q: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """
    Forecast winsorization pre-processing (Mekelburg & Strauss 2023).

    Clips each raw signal score at the cross-sectional [q, 1-q] quantiles.
    Applied *before* combine_signals.

    Distinct from the per-signal winsorization in equities/signals.py (which clips
    raw values per signal). This clips the cross-sectional distribution of
    already-computed signal scores, controlling outlier influence on blend weights.
    Returns a new signals dict; originals are unchanged.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, sig in signals.items():
        lo = sig.quantile(q, axis=1)
        hi = sig.quantile(1.0 - q, axis=1)
        out[name] = sig.clip(lower=lo, upper=hi, axis=0)
    return out


def pool_blend(
    blend_list: list[BlendWeights],
    method: str = "mean",
    trim_frac: float = 0.1,
) -> BlendWeights:
    """
    Ensemble pooling of multiple BlendWeights (Mekelburg & Strauss 2023).

    method:
        "mean"          — arithmetic mean of weights per signal
        "median"        — median per signal
        "trimmed_mean"  — trimmed mean (trim_frac from each tail)

    Useful for aggregating across seeds, hyperparameter grids, or methods.
    """
    if not blend_list:
        raise ValueError("blend_list is empty.")

    weights_df = pd.DataFrame([bw.weights for bw in blend_list]).fillna(0.0)

    if method == "mean":
        pooled = weights_df.mean()
    elif method == "median":
        pooled = weights_df.median()
    elif method == "trimmed_mean":
        try:
            from scipy.stats import trim_mean as _trim_mean
        except ImportError as e:
            raise ImportError("scipy is required for pool_blend with method='trimmed_mean'.") from e
        pooled = weights_df.apply(lambda col: _trim_mean(col.values, trim_frac))
    else:
        raise ValueError(f"Unknown pool method {method!r}. Choose 'mean', 'median', 'trimmed_mean'.")

    return BlendWeights(pooled, f"pool_{method}", {"n_models": len(blend_list), "method": method})


# ── Tree-based time-varying blends ─────────────────────────────────────────────

class _TreeBlend:
    """
    Base class for tree-based time-varying blend estimators.

    For each signal k, trains a separate tree ensemble: z_t → F_k[t+1].
    predict_weights computes μ̂_k = model_k(z_t) (expected managed returns) and
    derives blend weights via MSRR: b_t = (M + λI)⁻¹ μ̂_t.
    """

    def __init__(self, ridge_lam: float = 0.01):
        self.ridge_lam = ridge_lam
        self._models: dict = {}
        self._signal_names: Optional[list[str]] = None
        self._A_inv: Optional[np.ndarray] = None

    def _fit_impl(self, F_train: pd.DataFrame, z_train: pd.DataFrame, regressor_fn) -> None:
        common = F_train.index.intersection(z_train.dropna().index)
        F = F_train.loc[common]
        z = z_train.loc[common]
        self._signal_names = list(F.columns)

        for name in self._signal_names:
            model = regressor_fn()
            model.fit(z.values, F[name].values)
            self._models[name] = model

        Fv = F.values.astype(float)
        T, K = Fv.shape
        M = Fv.T @ Fv / T
        A = M + self.ridge_lam * np.eye(K)
        try:
            self._A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._A_inv = np.linalg.pinv(A)

    def predict_weights(self, z: pd.DataFrame) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("Call fit() before predict_weights().")

        nan_mask = z.isna().any(axis=1)
        valid_z = z[~nan_mask]

        mu_mat = np.column_stack([
            self._models[name].predict(valid_z.values)
            for name in self._signal_names
        ])  # (T_valid, K)

        B = mu_mat @ self._A_inv.T  # (T_valid, K)
        row_norms = np.abs(B).sum(axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
        B = B / row_norms

        out = pd.DataFrame(np.nan, index=z.index, columns=self._signal_names)
        out.loc[valid_z.index] = B
        return out


class GBRTBlend(_TreeBlend):
    """
    Gradient Boosted Regression Trees blend (Gu, Kelly & Xiu 2020).

    Trains K GBRTs (one per signal) to predict F_k[t+1] from state features z_t.
    Expected managed returns μ̂_k(z_t) are plugged into MSRR weights:
        b_t = (M + λI)⁻¹ μ̂_t,  M = F_train'F_train/T.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        ridge_lam: float = 0.01,
    ):
        super().__init__(ridge_lam=ridge_lam)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, F_train: pd.DataFrame, z_train: pd.DataFrame) -> "GBRTBlend":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError as e:
            raise ImportError("sklearn is required for GBRTBlend.") from e

        def _factory():
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
            )

        self._fit_impl(F_train, z_train, _factory)
        return self


class RandomForestBlend(_TreeBlend):
    """
    Random Forest blend (Gu, Kelly & Xiu 2020).

    Same structure as GBRTBlend but uses bagged decision trees.
    Provides implicit ensemble uncertainty via the RF variance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        ridge_lam: float = 0.01,
        seed: int = 42,
    ):
        super().__init__(ridge_lam=ridge_lam)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, F_train: pd.DataFrame, z_train: pd.DataFrame) -> "RandomForestBlend":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as e:
            raise ImportError("sklearn is required for RandomForestBlend.") from e

        def _factory():
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
            )

        self._fit_impl(F_train, z_train, _factory)
        return self


# ── Linear dynamic blend (IPCA-style) ──────────────────────────────────────────

class LinearDynamicBlend:
    """
    Linear time-varying blend: b_t = Γ₀ + Γ₁ z_t  (Giglio, Kelly & Xiu 2022).

    Estimated via ridge regression on expanded MSRR features:
        X_exp[t] = [F_k[t], F_k[t]·z_j[t]  for all k, j]
    Target: y = ones(T)  (MSRR convention).
    Coefficients [γ₀_k, γ₁_{kj}] decode to Γ₀ ∈ ℝ^K and Γ₁ ∈ ℝ^(K×D_z).

    Provides a linear, interpretable alternative to MLMSRR — a direct test of
    whether state-conditioned *linear* weighting helps before going nonlinear.
    """

    def __init__(self, ridge_lam: float = 0.01):
        self.ridge_lam = ridge_lam
        self._gamma0: Optional[np.ndarray] = None   # (K,)
        self._gamma1: Optional[np.ndarray] = None   # (K, D_z)
        self._signal_names: Optional[list[str]] = None

    def fit(self, F_train: pd.DataFrame, z_train: pd.DataFrame) -> "LinearDynamicBlend":
        common = F_train.index.intersection(z_train.dropna().index)
        F = F_train.loc[common].values.astype(float)
        z = z_train.loc[common].values.astype(float)
        T, K = F.shape
        D_z = z.shape[1]

        self._signal_names = list(F_train.columns)

        # Expanded features: [F_k] + [F_k * z_j for all k, j]
        interactions = (F[:, :, None] * z[:, None, :]).reshape(T, K * D_z)
        X = np.hstack([F, interactions])  # (T, K + K*D_z)

        y = np.ones(T)
        n_feat = X.shape[1]
        lam = self.ridge_lam * K
        A = X.T @ X / T + lam * np.eye(n_feat)
        b_rhs = X.T @ y / T
        theta, _, _, _ = np.linalg.lstsq(A, b_rhs, rcond=None)

        self._gamma0 = theta[:K]                # (K,)
        self._gamma1 = theta[K:].reshape(K, D_z)  # (K, D_z)
        return self

    def predict_weights(self, z: pd.DataFrame) -> pd.DataFrame:
        if self._gamma0 is None:
            raise RuntimeError("Call fit() before predict_weights().")

        z_arr = z.values.astype(float)
        nan_mask = np.isnan(z_arr).any(axis=1)

        # b_t = gamma0 + gamma1 @ z_t  →  (T, K)
        B = self._gamma0[None, :] + z_arr @ self._gamma1.T
        row_norms = np.abs(B).sum(axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
        B = B / row_norms
        B[nan_mask] = np.nan

        return pd.DataFrame(B, index=z.index, columns=self._signal_names)

    def combine_signals(
        self, signals: dict[str, pd.DataFrame], z: pd.DataFrame
    ) -> pd.DataFrame:
        weights_df = self.predict_weights(z)
        idx = sorted(set.union(*(set(df.index) for df in signals.values())))
        cols = sorted(set.union(*(set(df.columns) for df in signals.values())))
        combo = pd.DataFrame(0.0, index=idx, columns=cols)
        nan_rows = weights_df.isna().any(axis=1)
        for t in weights_df.index:
            if nan_rows.loc[t]:
                combo.loc[t] = np.nan
                continue
            b_t = weights_df.loc[t]
            row = pd.Series(0.0, index=cols)
            for name, wk in b_t.items():
                if name not in signals or t not in signals[name].index:
                    continue
                cross = signals[name].loc[t]
                mu_c, sd_c = cross.mean(), cross.std()
                if sd_c > 1e-12:
                    cross = (cross - mu_c) / sd_c
                row = row.add(wk * cross.reindex(cols).fillna(0.0))
            combo.loc[t] = row.values
        return combo


# ── Dynamic Model Averaging ────────────────────────────────────────────────────

class DMABlend:
    """
    Dynamic Model Averaging blend (Mekelburg & Strauss 2023; Raftery et al. 2010).

    Maintains time-varying model probabilities π_k,t updated at each period t:

        prediction step:  π̄_k,t = π_{k,t-1}^α / Σⱼ π_{j,t-1}^α   (forgetting)
        likelihood:       L_k,t = N(F_k[t] | μ_k, σ_k²)            (predictive)
        update step:      π_k,t ∝ π̄_k,t · L_k,t

    where μ_k and σ_k² are expanding-window statistics of past F_k observations.
    α ∈ (0, 1]: forgetting factor; α=1 → no forgetting (full Bayesian update).

    The blend weight at time t is b_k,t = π_k,t (already sums to 1).
    """

    def __init__(self, alpha: float = 0.99, prior_scale: float = 1.0):
        self.alpha = alpha
        self.prior_scale = prior_scale
        self._signal_names: Optional[list[str]] = None
        self._final_state: Optional[dict] = None
        self._weights_history: Optional[pd.DataFrame] = None

    def _run_filter(
        self,
        F: pd.DataFrame,
        initial_state: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        Fv = F.values.astype(float)
        T, K = Fv.shape

        if initial_state is None:
            log_pi = np.full(K, -np.log(K))   # log(1/K) — uniform prior
            mu = np.zeros(K)
            # Initialise variance to the empirical variance of first 21 obs (or prior_scale)
            var = np.full(K, self.prior_scale)
            count = np.zeros(K)
        else:
            log_pi = initial_state["log_pi"].copy()
            mu = initial_state["mu"].copy()
            var = initial_state["var"].copy()
            count = initial_state["count"].copy()

        weights = np.empty((T, K))

        for t in range(T):
            # Prediction step: apply forgetting in log-space (prevents underflow)
            # π̄_k = π_k^α / Σ π_j^α  ↔  log π̄_k = α·log π_k − log Σ exp(α·log π_j)
            log_pi_bar = self.alpha * log_pi
            m = log_pi_bar.max()
            log_pi_bar -= m + np.log(np.sum(np.exp(log_pi_bar - m)))

            # Predictive likelihood N(F_k[t] | mu_k, var_k) in log-space
            std = np.sqrt(np.maximum(var, 1e-10))
            residual = Fv[t] - mu
            log_L = -0.5 * (residual / std) ** 2 - np.log(std)

            # Update in log-space: log π_k ∝ log π̄_k + log L_k
            log_pi_new = log_pi_bar + log_L
            m2 = log_pi_new.max()
            log_pi = log_pi_new - (m2 + np.log(np.sum(np.exp(log_pi_new - m2))))

            weights[t] = np.exp(log_pi)

            # Update running stats (Welford)
            for k in range(K):
                if np.isfinite(Fv[t, k]):
                    count[k] += 1
                    delta = Fv[t, k] - mu[k]
                    mu[k] += delta / count[k]
                    delta2 = Fv[t, k] - mu[k]
                    if count[k] > 1:
                        var[k] = (var[k] * (count[k] - 2) + delta * delta2) / (count[k] - 1)
                    var[k] = max(var[k], 1e-10)

        state = {"log_pi": log_pi, "mu": mu, "var": var, "count": count}
        return weights, state

    def fit(self, F_train: pd.DataFrame) -> "DMABlend":
        self._signal_names = list(F_train.columns)
        weights_arr, self._final_state = self._run_filter(F_train)
        self._weights_history = pd.DataFrame(
            weights_arr, index=F_train.index, columns=self._signal_names
        )
        return self

    def predict_weights(self, F_new: pd.DataFrame) -> pd.DataFrame:
        if self._final_state is None:
            raise RuntimeError("Call fit() before predict_weights().")
        weights_arr, _ = self._run_filter(
            F_new[self._signal_names], initial_state=self._final_state
        )
        return pd.DataFrame(weights_arr, index=F_new.index, columns=self._signal_names)

    def combine_signals(
        self, signals: dict[str, pd.DataFrame], F: pd.DataFrame
    ) -> pd.DataFrame:
        weights_df = self.predict_weights(F)
        idx = sorted(set.union(*(set(df.index) for df in signals.values())))
        cols = sorted(set.union(*(set(df.columns) for df in signals.values())))
        combo = pd.DataFrame(0.0, index=idx, columns=cols)
        for t in weights_df.index:
            b_t = weights_df.loc[t]
            row = pd.Series(0.0, index=cols)
            for name, wk in b_t.items():
                if name not in signals or t not in signals[name].index:
                    continue
                cross = signals[name].loc[t]
                mu_c, sd_c = cross.mean(), cross.std()
                if sd_c > 1e-12:
                    cross = (cross - mu_c) / sd_c
                row = row.add(wk * cross.reindex(cols).fillna(0.0))
            combo.loc[t] = row.values
        return combo


# ── STMOM — direct portfolio model ────────────────────────────────────────────

class _STMOMLP(nn.Module):
    """Shared-weight per-asset position model (SLP or MLP → Tanh)."""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        if hidden_size > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Tanh(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Tanh(),
            )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)  # (batch, 1)


class STMOM:
    """
    Sparse Trend Momentum — direct asset-level portfolio model (Tan et al. 2023).

    Unlike the blend methods above (which combine K pre-computed signal sleeves into
    a composite score), STMOM maps per-asset signal values directly to positions for
    ALL assets simultaneously using a shared-weight network.

    Architecture (weights shared across all N assets):
        SLP (hidden_size=0): Linear(τ·d, 1) → Tanh
        MLP (hidden_size>0): Linear(τ·d, H) → ReLU → Linear(H, 1) → Tanh
    where d = number of signals and τ = number of lags.

    Input at time t for asset i: u_t[i] = concat(sig_k[t-lag, i]) for k in signals, lag in [0..τ-1]
    Output: X_t[i] ∈ (−1, 1) — long-short position for asset i.

    Portfolio return:  R_t = Σᵢ X_t[i] · r_{t+1}[i]
    Loss: −Sharpe(R) + λ₁·mean(|X_t|) + λ₂·turnover   (Tan et al. 2023)

    Multi-seed training; best by validation Sharpe is retained.
    """

    def __init__(
        self,
        n_signals: int,
        tau: int = 1,
        hidden_size: int = 0,
        lambda_l1: float = 1e-3,
        lambda_turnover: float = 1e-3,
        lr: float = 1e-3,
        n_epochs: int = 500,
        patience: int = 30,
        val_frac: float = 0.2,
        grad_clip: float = 1.0,
        seeds: list[int] = (42,),
    ):
        self.n_signals = n_signals
        self.tau = tau
        self.hidden_size = hidden_size
        self.lambda_l1 = lambda_l1
        self.lambda_turnover = lambda_turnover
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.grad_clip = grad_clip
        self.seeds = list(seeds)

        self._model: Optional[_STMOMLP] = None
        self._signal_names: Optional[list[str]] = None
        self._tickers: Optional[list] = None

    def _build_features(
        self,
        signals: dict[str, pd.DataFrame],
        common_dates: pd.DatetimeIndex,
        common_tickers: list,
    ) -> np.ndarray:
        """Build U array of shape (T, N, τ·K) with lagged signal features."""
        K = len(signals)
        T = len(common_dates)
        N = len(common_tickers)

        sig_arr = np.stack([
            sig.reindex(index=common_dates, columns=common_tickers).fillna(0.0).values
            for sig in signals.values()
        ], axis=-1).astype(np.float32)  # (T, N, K)

        U = np.zeros((T, N, self.tau * K), dtype=np.float32)
        for t in range(T):
            for lag in range(self.tau):
                t_lag = t - lag
                if t_lag >= 0:
                    U[t, :, lag * K:(lag + 1) * K] = sig_arr[t_lag]
        return U

    def _train_one_seed(
        self,
        U_tr: torch.Tensor,   # (T_tr, N, input_dim)
        R_tr: torch.Tensor,   # (T_tr, N) forward returns
        U_va: torch.Tensor,
        R_va: torch.Tensor,
        input_dim: int,
        seed: int,
    ) -> tuple["_STMOMLP", float]:
        torch.manual_seed(seed)
        model = _STMOMLP(input_dim, self.hidden_size)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        T_tr, N, _ = U_tr.shape

        best_val, best_state = math.inf, {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0

        for _ in range(self.n_epochs):
            model.train()
            opt.zero_grad()
            X_flat = model(U_tr.view(T_tr * N, -1))   # (T_tr*N, 1)
            X = X_flat.view(T_tr, N)
            port_ret = (X * R_tr).sum(dim=1)          # (T_tr,)
            mean_r = port_ret.mean()
            var_r = port_ret.var(unbiased=False) + 1e-6
            loss = -(mean_r / var_r.sqrt())
            if self.lambda_l1 > 0:
                loss = loss + self.lambda_l1 * X.abs().mean()
            if self.lambda_turnover > 0 and T_tr > 1:
                loss = loss + self.lambda_turnover * (X[1:] - X[:-1]).abs().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()

            model.eval()
            with torch.no_grad():
                T_va, N_va, _ = U_va.shape
                X_va = model(U_va.view(T_va * N_va, -1)).view(T_va, N_va)
                pret_va = (X_va * R_va).sum(dim=1)
                mean_va = pret_va.mean()
                var_va = pret_va.var(unbiased=False) + 1e-6
                val_loss = -(mean_va / var_va.sqrt()).item()

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        model.load_state_dict(best_state)
        return model, best_val

    def fit(
        self,
        signals: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> "STMOM":
        self._signal_names = list(signals.keys())
        self._tickers = list(forward_returns.columns)
        all_dates = forward_returns.index

        K = len(signals)
        input_dim = self.tau * K

        U = self._build_features(signals, all_dates, self._tickers)  # (T, N, input_dim)
        R = forward_returns.reindex(
            index=all_dates, columns=self._tickers
        ).fillna(0.0).values.astype(np.float32)

        T, N, _ = U.shape
        n_val = max(1, int(T * self.val_frac))
        n_tr = T - n_val

        U_tr = torch.tensor(U[:n_tr])
        R_tr = torch.tensor(R[:n_tr])
        U_va = torch.tensor(U[n_tr:])
        R_va = torch.tensor(R[n_tr:])

        best_model, best_val = None, math.inf
        for seed in self.seeds:
            model, val = self._train_one_seed(U_tr, R_tr, U_va, R_va, input_dim, seed)
            if val < best_val:
                best_val = val
                best_model = model

        self._model = best_model
        return self

    def predict_positions(
        self,
        signals: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Return T × N DataFrame of positions in (−1, 1). NaN where signal unavailable."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_positions().")

        all_dates_set: set = set()
        for sig in signals.values():
            all_dates_set.update(sig.index)
        all_dates = pd.DatetimeIndex(sorted(all_dates_set))

        U = self._build_features(signals, all_dates, self._tickers)  # (T, N, input_dim)
        T, N, input_dim = U.shape

        self._model.eval()
        with torch.no_grad():
            X_flat = self._model(torch.tensor(U.reshape(T * N, input_dim))).numpy()
        X = X_flat.reshape(T, N)
        return pd.DataFrame(X, index=all_dates, columns=self._tickers)
