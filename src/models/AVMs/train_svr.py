from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


# ============================================================
# Config
# ============================================================
@dataclass
class SVRTrainConfig:
    train_path: Path
    out_dir: Path

    month_col: str = "Month"          # used ONLY for time-based CV split
    district_col: str = "District"    # categorical feature
    target_col: str = "log_Median_Price"

    drop_raw_cols: Tuple[str, ...] = ("sales_count", "Consents")
    lag_suffix: str = "_lag"

    # Time-based CV inside TRAIN
    n_splits: int = 5
    min_train_months: int = 24
    val_months_per_split: int = 6

    # SVR grid
    kernel: str = "rbf"
    C_grid: Tuple[float, ...] = (1.0, 10.0, 100.0)
    epsilon_grid: Tuple[float, ...] = (0.05, 0.1, 0.2)
    gamma_grid: Tuple[float, ...] = (0.01, 0.1, 1.0)  # only for RBF

    scoring: str = "neg_root_mean_squared_error"


# ============================================================
# I/O
# ============================================================
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Preprocessing
# ============================================================
def parse_month_to_month_start(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """
    Normalize Month to month-start Timestamp. Accepts 'YYYY-MM' or any datetime-like.
    """
    if month_col not in df.columns:
        raise KeyError(f"Missing month column: {month_col}")

    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    out[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return out


def drop_raw_columns(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df.drop(columns=present)


def drop_rows_with_na_lags(df: pd.DataFrame, lag_suffix: str = "_lag") -> pd.DataFrame:
    lag_cols = [c for c in df.columns if lag_suffix in c]
    if not lag_cols:
        raise ValueError(f"No lag columns found (suffix='{lag_suffix}').")
    return df.dropna(subset=lag_cols).reset_index(drop=True)


def sort_panel(df: pd.DataFrame, month_col: str, district_col: str) -> pd.DataFrame:
    if district_col in df.columns:
        return df.sort_values([month_col, district_col]).reset_index(drop=True)
    return df.sort_values([month_col]).reset_index(drop=True)


# ============================================================
# Time-series CV splitter by Month (no leakage)
# ============================================================
class MonthSeriesSplit:
    """
    Time-series CV using unique months in X[month_col].
    Each fold:
      Train: earliest months ... up to train_end
      Val  : next val_months_per_split months
    """

    def __init__(self, n_splits: int, min_train_months: int, val_months_per_split: int, month_col: str):
        self.n_splits = n_splits
        self.min_train_months = min_train_months
        self.val_months_per_split = val_months_per_split
        self.month_col = month_col

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.month_col not in X.columns:
            raise KeyError(f"MonthSeriesSplit requires '{self.month_col}' in X.")

        months = pd.to_datetime(X[self.month_col]).dt.to_period("M").astype(str)
        uniq_months = np.array(sorted(months.unique()))
        total_months = len(uniq_months)

        need = self.min_train_months + self.val_months_per_split
        if total_months < need:
            raise ValueError(f"Not enough months for CV: total={total_months}, need at least {need}.")

        max_train_end = total_months - self.val_months_per_split
        train_end_positions = np.linspace(self.min_train_months, max_train_end, num=self.n_splits, dtype=int)

        for train_end in train_end_positions:
            train_months = set(uniq_months[:train_end])
            val_months = set(uniq_months[train_end: train_end + self.val_months_per_split])

            train_idx = np.where(months.isin(train_months))[0]
            val_idx = np.where(months.isin(val_months))[0]
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ============================================================
# Modeling
# ============================================================
def build_svr_pipeline(
    df: pd.DataFrame,
    month_col: str,
    district_col: str,
    target_col: str,
) -> Pipeline:
    """
    SVR pipeline:
      - Drop Month from model inputs (used only for CV ordering)
      - OneHot encode District
      - Standardize numeric features
      - SVR
    """
    forbidden = {target_col, month_col}
    feature_cols = [c for c in df.columns if c not in forbidden]

    cat_cols = [district_col] if district_col in feature_cols else []
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("svr", SVR())])
    return pipe


def tune_svr(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: SVRTrainConfig,
    pipeline: Pipeline,
) -> GridSearchCV:
    param_grid: Dict[str, Any] = {
        "svr__kernel": [cfg.kernel],
        "svr__C": list(cfg.C_grid),
        "svr__epsilon": list(cfg.epsilon_grid),
    }
    if cfg.kernel == "rbf":
        param_grid["svr__gamma"] = list(cfg.gamma_grid)

    splitter = MonthSeriesSplit(
        n_splits=cfg.n_splits,
        min_train_months=cfg.min_train_months,
        val_months_per_split=cfg.val_months_per_split,
        month_col=cfg.month_col,
    )

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=cfg.scoring,
        cv=splitter.split(X),
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    gs.fit(X, y)
    return gs

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def save_artifacts(
    gs: GridSearchCV,
    cfg: SVRTrainConfig,
    train_fit_metrics: Dict[str, float],
    rows_used: int,
    month_min: str,
    month_max: str,
) -> None:
    ensure_dir(cfg.out_dir)

    # 1) CV table
    cv_df = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    cv_path = cfg.out_dir / "svr_cv_results.csv"
    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")

    # 2) A readable metrics table (one-row CSV)
    best_rmse_cv = float(-gs.best_score_)  # convert neg RMSE -> RMSE
    metrics_row = {
        "CV_RMSE(best)": best_rmse_cv,
        "Train_RMSE(in_sample)": train_fit_metrics["RMSE"],
        "Train_MAE(in_sample)": train_fit_metrics["MAE"],
        "Train_R2(in_sample)": train_fit_metrics["R2"],
        "rows_used_after_dropna": rows_used,
        "train_month_min": month_min,
        "train_month_max": month_max,
        "best_params": str(gs.best_params_),
    }
    metrics_csv_path = cfg.out_dir / "svr_metrics_train.csv"
    pd.DataFrame([metrics_row]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    # 3) Human-readable TXT report
    txt_path = cfg.out_dir / "svr_metrics_train.txt"
    txt = (
        "SVR training summary (TRAIN ONLY)\n"
        "==================================================\n"
        f"Best CV RMSE: {best_rmse_cv:,.4f}\n"
        f"Train RMSE (in-sample): {train_fit_metrics['RMSE']:,.4f}\n"
        f"Train MAE  (in-sample): {train_fit_metrics['MAE']:,.4f}\n"
        f"Train R^2  (in-sample): {train_fit_metrics['R2']:.6f}\n"
        "\n"
        f"Rows used after dropping NA lags: {rows_used}\n"
        f"Train month range: {month_min}  ->  {month_max}\n"
        "\n"
        f"Best params: {gs.best_params_}\n"
    )
    txt_path.write_text(txt, encoding="utf-8")

    # 4) JSON (keep for programmatic use)
    summary = {
        "best_score_cv_neg_rmse": float(gs.best_score_),
        "best_rmse_cv": best_rmse_cv,
        "best_params": gs.best_params_,
        "train_in_sample_metrics": train_fit_metrics,
        "rows_used_after_dropna": rows_used,
        "train_month_min": month_min,
        "train_month_max": month_max,
        "scoring": cfg.scoring,
        "kernel": cfg.kernel,
        "n_splits": cfg.n_splits,
        "min_train_months": cfg.min_train_months,
        "val_months_per_split": cfg.val_months_per_split,
    }
    json_path = cfg.out_dir / "svr_summary.json"
    json_path.write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5) Model file
    model_path = cfg.out_dir / "svr_best_model.joblib"
    joblib.dump(gs.best_estimator_, model_path)

    print(f"Saved CV results      : {cv_path}")
    print(f"Saved metrics (CSV)   : {metrics_csv_path}")
    print(f"Saved metrics (TXT)   : {txt_path}")
    print(f"Saved summary (JSON)  : {json_path}")
    print(f"Saved best model      : {model_path}")



# ============================================================
# Pipeline runner
# ============================================================
def train_svr_from_train_for_avms(cfg: SVRTrainConfig) -> GridSearchCV: # type: ignore
    df = load_table(cfg.train_path)

    # Normalize month
    df = parse_month_to_month_start(df, cfg.month_col)

    # Drop raw columns (keep log_* and lags)
    df = drop_raw_columns(df, cfg.drop_raw_cols)

    # Drop rows with NA lag features
    df = drop_rows_with_na_lags(df, cfg.lag_suffix)

    # Sort for stable time-based CV
    df = sort_panel(df, cfg.month_col, cfg.district_col)

    if cfg.target_col not in df.columns:
        raise KeyError(f"Target column not found: {cfg.target_col}")

    y = df[cfg.target_col].astype(float).values
    X = df.drop(columns=[cfg.target_col]).copy()  # keep Month for splitter

    pipeline = build_svr_pipeline(df, cfg.month_col, cfg.district_col, cfg.target_col)
    gs = tune_svr(X, y, cfg, pipeline)  # type: ignore

    print("\n=== SVR Training (GridSearchCV) Completed ===")
    print("Best CV score (neg RMSE):", float(gs.best_score_))
    print("Best CV RMSE            :", float(-gs.best_score_))
    print("Best params             :", gs.best_params_)

    # -------------------------

    # In-sample (TRAIN) fitted metrics (diagnostic)
    y_hat_train = gs.best_estimator_.predict(X) # type: ignore
    train_fit_metrics = regression_metrics(y, y_hat_train) # type: ignore

    save_artifacts(
        gs,
        cfg,
        train_fit_metrics=train_fit_metrics,
        rows_used=len(df),
        month_min=str(df[cfg.month_col].min()),
        month_max=str(df[cfg.month_col].max()),
    )




# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import PREPROCESSED_DIR,MODEL_DIR

    train_path = PREPROCESSED_DIR / "avms" / "train_for_AVMs.csv"
    out_dir = MODEL_DIR / "avms" / "svr_best_model"

    cfg = SVRTrainConfig(
        train_path=train_path,
        out_dir=out_dir,
        # Adjust if needed
        n_splits=5,
        min_train_months=24,
        val_months_per_split=6,
        kernel="rbf",
        C_grid=(1.0, 10.0, 100.0),
        epsilon_grid=(0.05, 0.1, 0.2),
        gamma_grid=(0.01, 0.1, 1.0),
        scoring="neg_root_mean_squared_error",
    )

    train_svr_from_train_for_avms(cfg)


if __name__ == "__main__":
    main()
