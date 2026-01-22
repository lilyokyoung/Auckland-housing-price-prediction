from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import FORECAST_DIR


# ---------------------------------
# Config
# ---------------------------------
@dataclass
class BVARConfig:
    # Input (your extracted level macro table)
    macro_level_path: Path = FORECAST_DIR / "macro_variables_level_monthly.csv"

    # Output (forecast in transformed space)
    output_forecast_path: Path = FORECAST_DIR / "var_macro" / "bvar_forecast.csv"

    # Column name for time index in the CSV
    month_col: str = "Month"

    # Level columns that exist in macro_level_path
    level_cols: Tuple[str, ...] = (
        "CPI",
        "CGPI_Dwelling",
        "OCR",
        "2YearFixedRate",
        "unemployment_rate",
    )

    # VAR input columns (after transform)
    var_cols: Tuple[str, ...] = (
        "dlog_CPI",
        "dlog_CGPI_Dwelling",
        "OCR",
        "2YearFixedRate",
        "unemployment_rate",
    )

    # -----------------------------
    # Recommended settings for your case
    # -----------------------------
    # With ~81 monthly samples and 5 variables:
    # - keep the model parsimonious (p=1 is the safest)
    # - forecast 12 months ahead is reasonable
    p: int = 1
    horizon: int = 12

    # Minnesota prior hyperparameters (shrinkage)
    # Stronger shrinkage helps stability under short samples
    lambda1: float = 0.18   # overall tightness (try 0.15~0.22)
    lambda2: float = 0.50   # cross-variable tightness
    lambda3: float = 1.00   # lag decay power

    include_intercept: bool = True
    verbose: bool = True


# ---------------------------------
# Utilities
# ---------------------------------
def load_macro_level_table(path: Path, month_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if month_col not in df.columns:
        raise ValueError(f"Month column '{month_col}' not found. Columns={df.columns.tolist()}")

    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    if df[month_col].isna().any():
        bad = df.loc[df[month_col].isna(), month_col].head(5).tolist()
        raise ValueError(f"Unparseable Month values found: {bad}")

    df = df.sort_values(month_col).set_index(month_col)
    return df


def make_var_inputs_from_levels(df_level: pd.DataFrame) -> pd.DataFrame:
    """
    Build the VAR input dataframe:
      - dlog_CPI = diff(log(CPI))
      - dlog_CGPI_Dwelling = diff(log(CGPI_Dwelling))
      - OCR, 2YearFixedRate, unemployment_rate kept in levels
    """
    df = df_level.copy()

    # Ensure numeric
    for c in ["CPI", "CGPI_Dwelling", "OCR", "2YearFixedRate", "unemployment_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Guard against non-positive values before log
    if (df["CPI"] <= 0).any() or (df["CGPI_Dwelling"] <= 0).any():
        raise ValueError("CPI / CGPI_Dwelling contain non-positive values; cannot apply log().")

    # log-diff for price indices
    df["dlog_CPI"] = np.log(df["CPI"]).diff() # type: ignore
    df["dlog_CGPI_Dwelling"] = np.log(df["CGPI_Dwelling"]).diff() # type: ignore

    out = df[["dlog_CPI", "dlog_CGPI_Dwelling", "OCR", "2YearFixedRate", "unemployment_rate"]].dropna()
    return out


def build_lagged_matrices(
    y: np.ndarray, p: int, include_intercept: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    y: (T, k)
    Returns:
      Y: (T-p, k)
      X: (T-p, m) where m = 1 + k*p if intercept else k*p
    """
    T, k = y.shape
    if T <= p:
        raise ValueError(f"Not enough observations T={T} for p={p}")

    Y = y[p:, :]  # (T-p, k)

    X_lags = []
    for lag in range(1, p + 1):
        X_lags.append(y[p - lag : T - lag, :])  # (T-p, k)

    X = np.concatenate(X_lags, axis=1)  # (T-p, k*p)

    if include_intercept:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    return Y, X


def estimate_sigma_ar1(y: np.ndarray) -> np.ndarray:
    """
    Estimate per-series scale sigma_i using AR(1) residual std (common for Minnesota prior scaling).
    y: (T, k)
    returns sigma: (k,)
    """
    T, k = y.shape
    sigma = np.zeros(k)

    for i in range(k):
        yi = y[:, i]
        Y = yi[1:]
        X = np.column_stack([np.ones(T - 1), yi[:-1]])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        sigma[i] = np.std(resid, ddof=2)

    # Fallback if any sigma is 0
    sigma = np.where(sigma <= 1e-12, np.std(y, axis=0, ddof=1), sigma)
    return sigma


def bvar_fit(
    Y: np.ndarray,
    X: np.ndarray,
    p: int,
    sigma: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    include_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit BVAR with Minnesota prior (posterior mean for coefficients).
    Returns:
      B_post: (m, k) posterior mean coefficient matrix
      Sigma_u: (k, k) residual covariance from posterior-mean fit
    """
    Tn, k = Y.shape
    m = X.shape[1]

    # Prior mean (m, k): own first lag = 1, others 0
    B0 = np.zeros((m, k))
    offset = 1 if include_intercept else 0
    if p >= 1:
        for i in range(k):
            B0[offset + i, i] = 1.0

    B_post = np.zeros((m, k))
    E = np.zeros_like(Y)

    XtX = X.T @ X
    XtY = X.T @ Y

    def row_index(l: int, j: int) -> int:
        # l in 1..p; ordering: [const], lag1 vars, lag2 vars, ...
        return offset + (l - 1) * k + j

    for i in range(k):
        # Prior precision for equation i (diagonal)
        prec = np.zeros(m)

        # Intercept weak prior
        if include_intercept:
            prec[0] = 1.0 / (10.0**2)  # Var=100

        # Lag coeffs
        for l in range(1, p + 1):
            for j in range(k):
                r = row_index(l, j)

                base = (lambda1**2) / (l ** (2.0 * lambda3))
                scale = (sigma[i] ** 2) / (sigma[j] ** 2)

                if i == j:
                    var_ij = base * scale
                else:
                    var_ij = base * (lambda2**2) * scale

                prec[r] = 1.0 / max(var_ij, 1e-12)

        Omega_inv_i = np.diag(prec)

        A = XtX + Omega_inv_i
        b = XtY[:, i] + Omega_inv_i @ B0[:, i]
        B_post[:, i] = np.linalg.solve(A, b)

        E[:, i] = Y[:, i] - X @ B_post[:, i]

    Sigma_u = (E.T @ E) / max(Tn - m, 1)
    return B_post, Sigma_u


def bvar_forecast(
    y_hist: np.ndarray,
    B: np.ndarray,
    p: int,
    horizon: int,
    include_intercept: bool = True,
) -> np.ndarray:
    """
    Forecast using posterior-mean coefficients B.
    y_hist: (T, k) history in transformed space
    Returns forecasts: (horizon, k)
    """
    T, k = y_hist.shape
    if T < p:
        raise ValueError(f"Need at least p={p} observations in history, got T={T}")

    y_ext = y_hist.copy().tolist()
    forecasts = []

    for _ in range(horizon):
        x_parts = []
        if include_intercept:
            x_parts.append([1.0])

        for lag in range(1, p + 1):
            x_parts.append(y_ext[-lag])

        x_t = np.array([v for part in x_parts for v in part], dtype=float)  # (m,)
        y_next = x_t @ B  # (k,)

        forecasts.append(y_next)
        y_ext.append(y_next.tolist())

    return np.vstack(forecasts)


# ---------------------------------
# main
# ---------------------------------
def main() -> None:
    cfg = BVARConfig()

    # 1) Load level macro table
    df_level = load_macro_level_table(cfg.macro_level_path, cfg.month_col)

    # 2) Build VAR inputs (transformed space)
    df_var = make_var_inputs_from_levels(df_level)
    if cfg.verbose:
        print("VAR input head:\n", df_var.head())
        print("\nVAR input tail:\n", df_var.tail())
        print("\nVAR input shape:", df_var.shape)

    # 3) BUG FIX: pandas needs a LIST for multi-column selection (tuple will cause KeyError)
    # Also add a column existence check for safety.
    missing = [c for c in cfg.var_cols if c not in df_var.columns]
    if missing:
        raise ValueError(f"Missing VAR columns: {missing}. Available: {df_var.columns.tolist()}")

    y = df_var.loc[:, list(cfg.var_cols)].to_numpy(dtype=float)  # (T, k)
    T, k = y.shape

    if cfg.verbose:
        print(f"\n[T, k] = [{T}, {k}], using p={cfg.p}, horizon={cfg.horizon}")
        print(
            f"Prior hyperparams: lambda1={cfg.lambda1}, lambda2={cfg.lambda2}, lambda3={cfg.lambda3}"
        )

    Y, X = build_lagged_matrices(y, p=cfg.p, include_intercept=cfg.include_intercept)

    # 4) Minnesota prior scaling (sigma)
    sigma = estimate_sigma_ar1(y)
    if cfg.verbose:
        print("\nSigma (AR1 residual std) per series:")
        for name, s in zip(cfg.var_cols, sigma):
            print(f"  {name}: {s:.6f}")

    # 5) Fit BVAR (posterior mean)
    B_post, Sigma_u = bvar_fit(
        Y=Y,
        X=X,
        p=cfg.p,
        sigma=sigma,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        lambda3=cfg.lambda3,
        include_intercept=cfg.include_intercept,
    )

    if cfg.verbose:
        print("\nB_post shape:", B_post.shape)
        print("Sigma_u (residual covariance) shape:", Sigma_u.shape)

    # 6) Forecast in transformed space
    fc = bvar_forecast(
        y_hist=y,
        B=B_post,
        p=cfg.p,
        horizon=cfg.horizon,
        include_intercept=cfg.include_intercept,
    )

    # 7) Build forecast index (monthly start)
    last_month = df_var.index.max()
    future_index = pd.date_range(
        start=(last_month + pd.offsets.MonthBegin(1)),
        periods=cfg.horizon,
        freq="MS",
    )

    df_fc = pd.DataFrame(fc, index=future_index, columns=list(cfg.var_cols))
    df_fc.index.name = cfg.month_col

    # 8) Save
    cfg.output_forecast_path.parent.mkdir(parents=True, exist_ok=True)
    df_fc.to_csv(cfg.output_forecast_path, index=True)

    print("\nBVAR forecast completed.")
    print("Saved to:", cfg.output_forecast_path)


if __name__ == "__main__":
    main()
