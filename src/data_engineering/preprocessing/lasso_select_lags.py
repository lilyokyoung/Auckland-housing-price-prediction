from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# ============================================================
# 0) Config
# ============================================================

TARGET_COL = "log_Median_Price"
ID_COLS = ["Month", "District"]

# ✅ Only allow these variable families (current + lags) into LASSO
ALLOWED_BASES = [
    "log_sales_count",
    "log_Consents",
    "CGPI_Dwelling",
    "unemployment_rate",
    "OCR",
    "weeklyrent",
    "Net_migration_monthly",
]

# If True: only allow lag terms (e.g. *_lag1, *_lag3...) and exclude contemporaneous base itself
USE_LAG_ONLY_FEATURES = False


# ============================================================
# 1) I/O
# ============================================================

def load_train_for_lasso(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Input must be .csv or .parquet")


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ============================================================
# 2) Prep helpers
# ============================================================

def coerce_month_to_period(df: pd.DataFrame, month_col: str = "Month") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="coerce", dayfirst=True)
    if dt.isna().any():
        bad = out.loc[dt.isna(), month_col].head(5).tolist()
        raise ValueError(f"Month parse failed, examples: {bad}")
    out[month_col] = dt.dt.to_period("M")
    return out


def pick_allowed_feature_cols(
    columns: list[str],
    allowed_bases: list[str],
    lag_only: bool,
) -> list[str]:
    """
    Keep only:
      - base columns in allowed_bases (if lag_only=False)
      - lag columns like f"{base}_lag{...}"
    """
    allowed: list[str] = []
    for c in columns:
        for base in allowed_bases:
            if lag_only:
                # only keep lag columns
                if c.startswith(base + "_lag"):
                    allowed.append(c)
                    break
            else:
                # keep base itself + its lags
                if c == base or c.startswith(base + "_lag"):
                    allowed.append(c)
                    break

    # preserve original order and unique
    seen: set[str] = set()
    out: list[str] = []
    for c in allowed:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    X: only allowed families (current + lags) defined in ALLOWED_BASES
    y: log_Median_Price
    """
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns (first 50): {list(df.columns)[:50]}"
        )

    # --- y ---
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # --- X candidate columns (excluding IDs + target) ---
    cols = [c for c in df.columns if c not in set(ID_COLS + [TARGET_COL])]

    # --- whitelist filter ---
    feature_cols = pick_allowed_feature_cols(
        columns=cols,
        allowed_bases=ALLOWED_BASES,
        lag_only=USE_LAG_ONLY_FEATURES,
    )

    if len(feature_cols) == 0:
        raise ValueError(
            "No allowed features found. Check ALLOWED_BASES / USE_LAG_ONLY_FEATURES "
            "and your actual column names."
        )

    # --- numeric coercion ---
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop rows with NA in X or y (lags create NA at top)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    return X, y


# ============================================================
# 3) LASSO training / selection
# ============================================================

def fit_lasso_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 50_000,
) -> tuple[Pipeline, dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lasso", LassoCV(
                cv=tscv,
                random_state=random_state,
                max_iter=max_iter,
                n_alphas=200,
            )),
        ]
    )

    model.fit(X, y)
    best_alpha = float(model.named_steps["lasso"].alpha_)

    # Diagnostic RMSE across outer folds (optional but useful)
    rmses: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(
                    cv=TimeSeriesSplit(n_splits=3),
                    random_state=random_state,
                    max_iter=max_iter,
                    n_alphas=200,
                )),
            ]
        )
        fold_model.fit(X_tr, y_tr)
        pred = fold_model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))  # type: ignore
        rmses.append(rmse)

    metrics = {
        "best_alpha": best_alpha,
        "tscv_rmse_mean": float(np.mean(rmses)),
        "tscv_rmse_std": float(np.std(rmses)),
        "n_splits": n_splits,
        "n_rows_used": int(len(X)),
        "n_features": int(X.shape[1]),
    }
    return model, metrics


def extract_selected_features(
    fitted_model: Pipeline,
    feature_names: list[str],
    coef_threshold: float = 1e-12,
) -> pd.DataFrame:
    coefs = fitted_model.named_steps["lasso"].coef_
    out = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    })
    selected = out[out["abs_coef"] > coef_threshold].copy()
    selected = selected.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return selected


# ============================================================
# 4) main
# ============================================================

def main() -> None:
    from src.config import PREPROCESSED_DIR, TABLE_DIR

    in_csv = PREPROCESSED_DIR / "LASSO" / "train_for_lasso.csv"
    in_parquet = PREPROCESSED_DIR / "LASSO" / "train_for_lasso.parquet"
    in_path = in_csv if in_csv.exists() else in_parquet

    if not in_path.exists():
        raise FileNotFoundError(
            "train_for_lasso file not found. Expected one of:\n"
            f" - {in_csv}\n"
            f" - {in_parquet}\n"
        )

    df = load_train_for_lasso(in_path)
    df = coerce_month_to_period(df, "Month")
    df = df.sort_values(["Month", "District"]).reset_index(drop=True)

    X, y = build_X_y(df)

    print(f"[INFO] Allowed bases: {ALLOWED_BASES}")
    print(f"[INFO] USE_LAG_ONLY_FEATURES = {USE_LAG_ONLY_FEATURES}")
    print(f"[INFO] X shape: {X.shape} | y length: {len(y)}")
    print(f"[INFO] First 30 X columns: {list(X.columns)[:30]}")

    model, metrics = fit_lasso_timeseries_cv(
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
        max_iter=50_000,
    )

    selected = extract_selected_features(model, feature_names=list(X.columns))

    out_dir = TABLE_DIR / "LASSO_summary"
    save_csv(selected, out_dir / "lasso_selected_features.csv")
    save_csv(pd.DataFrame([metrics]), out_dir / "lasso_metrics.csv")

    print("\n=== LASSO Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Selected features: {len(selected)}")
    print("\nTop 20 selected features:")
    print(selected.head(20))


if __name__ == "__main__":
    main()
