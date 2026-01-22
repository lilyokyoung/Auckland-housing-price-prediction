# src/explain/build_forecast_shap_long.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import warnings

import numpy as np
import pandas as pd
import joblib


# ============================================================
# Config
# ============================================================
@dataclass
class ForecastShapLongConfig:
    # --- model + prediction inputs ---
    rf_model_path: Path

    # the same feature-engineered future datasets used for /api/predict
    future_base_path: Path
    future_low_path: Path
    future_high_path: Path

    # optional: if you want to merge history too (not required for shap-long)
    # historical_path: Optional[Path] = None

    # --- output ---
    out_dir: Path

    # --- columns ---
    month_col: str = "Month"
    district_col: str = "District"
    scenario_col: str = "Scenario"

    # --- behavior ---
    drop_cols: Tuple[str, ...] = ("Scenario",)  # Scenario usually NOT a model feature
    max_rows_per_scenario: int = 2000
    random_state: int = 42

    # output filename
    out_csv_name: str = "forecast_shap_long.csv"


# ============================================================
# Utils
# ============================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _is_pipeline(model_obj) -> bool:
    return hasattr(model_obj, "steps") and hasattr(model_obj, "__len__")


def _split_pipeline(model_obj):
    """
    Returns: (preprocess, estimator)
    """
    if not _is_pipeline(model_obj):
        return None, model_obj
    if len(model_obj.steps) < 1:
        return None, model_obj
    preprocess = model_obj[:-1]
    estimator = model_obj.steps[-1][1]
    return preprocess, estimator


def _safe_to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    For non-pipeline models: coerce non-numeric to codes.
    """
    X2 = X.copy()
    for c in X2.columns:
        if not np.issubdtype(X2[c].dtype, np.number):  # type: ignore
            X2[c] = pd.factorize(X2[c].astype(str))[0].astype(float)
    return X2


def _get_transformed_feature_names(preprocess, input_features: List[str]) -> List[str]:
    """
    Best-effort transformed feature names.
    """
    if preprocess is None:
        return input_features

    if hasattr(preprocess, "get_feature_names_out"):
        try:
            names = preprocess.get_feature_names_out(input_features)
            return [str(x) for x in names]
        except Exception:
            pass

    # Pipeline last step
    if hasattr(preprocess, "steps"):
        try:
            last = preprocess.steps[-1][1]
            if hasattr(last, "get_feature_names_out"):
                names = last.get_feature_names_out(input_features)
                return [str(x) for x in names]
        except Exception:
            pass

    return [f"f_{i}" for i in range(len(input_features))]


def _transform_X_if_needed(model_obj, X_raw: pd.DataFrame) -> Tuple[np.ndarray, List[str], Any]:
    """
    If pipeline: preprocess.transform(X_raw) then shap on estimator.
    Else: numeric matrix for estimator.
    """
    preprocess, estimator = _split_pipeline(model_obj)

    if preprocess is None:
        X_num = _safe_to_numeric_matrix(X_raw)
        return X_num.to_numpy(), list(X_num.columns), estimator

    Xt = preprocess.transform(X_raw)

    # sparse -> dense (for shap)
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except Exception:
        pass

    feat_names = _get_transformed_feature_names(preprocess, list(X_raw.columns))

    if isinstance(Xt, np.ndarray) and Xt.ndim == 2 and len(feat_names) != Xt.shape[1]:
        feat_names = [f"f_{i}" for i in range(Xt.shape[1])]

    return np.asarray(Xt), feat_names, estimator


def _compute_shap(estimator, X_mat: np.ndarray) -> np.ndarray:
    """
    TreeExplainer for RF.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        raise RuntimeError("SHAP is not installed. Run: pip install shap") from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_mat)

    if isinstance(shap_values, list):
        return np.asarray(shap_values[0])
    return np.asarray(shap_values)


def _read_and_tag_future(path: Path, scenario: str) -> pd.DataFrame:
    df = _load_df(path)
    df = df.copy()
    df["Scenario"] = scenario
    return df


def _standardize_meta(df: pd.DataFrame, month_col: str, district_col: str, scenario_col: str) -> pd.DataFrame:
    """
    Keep Month/District/Scenario present & clean, but don't force any particular casing.
    """
    if month_col in df.columns:
        df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    if district_col in df.columns:
        df[district_col] = df[district_col].astype(str).str.strip()
    if scenario_col in df.columns:
        df[scenario_col] = df[scenario_col].astype(str).str.strip().str.lower()
    return df


def _sample_each_scenario(df: pd.DataFrame, scenario_col: str, max_rows: int, random_state: int) -> pd.DataFrame:
    """
    Sample rows per scenario to control compute size.
    """
    out = []
    for scen, g in df.groupby(scenario_col, dropna=False):
        if len(g) <= max_rows:
            out.append(g.copy())
        else:
            out.append(g.sample(n=max_rows, random_state=random_state).copy())
    return pd.concat(out, ignore_index=True) if out else df.head(0)


# ============================================================
# Core
# ============================================================
def run_build_forecast_shap_long(cfg: ForecastShapLongConfig) -> Dict[str, Any]:
    _ensure_dir(cfg.out_dir)

    model_obj = _load_model(cfg.rf_model_path)

    # 1) load future datasets (base/low/high) and concat
    df_base = _read_and_tag_future(cfg.future_base_path, "base")
    df_low = _read_and_tag_future(cfg.future_low_path, "low")
    df_high = _read_and_tag_future(cfg.future_high_path, "high")

    df_all = pd.concat([df_low, df_base, df_high], ignore_index=True)
    df_all = _standardize_meta(df_all, cfg.month_col, cfg.district_col, cfg.scenario_col)

    # guard
    required_meta = {cfg.month_col, cfg.district_col, cfg.scenario_col}
    missing_meta = sorted(list(required_meta - set(df_all.columns)))
    if missing_meta:
        raise ValueError(f"Future dataset missing meta columns: {missing_meta}")

    # 2) sample (avoid explosion)
    df_all = _sample_each_scenario(df_all, cfg.scenario_col, cfg.max_rows_per_scenario, cfg.random_state)

    # 3) build X_raw for transform
    #    IMPORTANT: if model is pipeline(ColumnTransformer), X_raw must include columns like District.
    X_raw = df_all.drop(columns=[], errors="ignore").copy()

    # drop Scenario if model didn't use it
    for c in cfg.drop_cols:
        if c in X_raw.columns:
            X_raw = X_raw.drop(columns=[c])

    # ✅ CRITICAL FIX: pipeline needs District/Month columns if used in preprocess
    if _is_pipeline(model_obj):
        must_have = [cfg.district_col, cfg.month_col]
        for col in must_have:
            if col in df_all.columns and col not in X_raw.columns:
                X_raw[col] = df_all[col]

    # 4) transform -> compute shap
    X_mat, feat_names, estimator = _transform_X_if_needed(model_obj, X_raw)
    shap_mat = _compute_shap(estimator, X_mat)

    # 5) build long file (align each row's meta with shap row)
    # shap_mat: [n_rows, n_features]
    if shap_mat.shape[0] != len(df_all):
        raise RuntimeError(
            f"SHAP row mismatch: shap_mat rows={shap_mat.shape[0]} but df rows={len(df_all)}"
        )

    meta_df = df_all[[cfg.month_col, cfg.district_col, cfg.scenario_col]].copy()
    meta_df = meta_df.reset_index(drop=True)

    # long: Month, District, Scenario, feature, shap_value
    # (This can be large; but n_rows <= 3*max_rows_per_scenario)
    out_rows: List[pd.DataFrame] = []
    for j, fname in enumerate(feat_names):
        tmp = meta_df.copy()
        tmp["feature"] = str(fname)
        tmp["shap_value"] = shap_mat[:, j].astype(float)
        out_rows.append(tmp)

    long_df = pd.concat(out_rows, ignore_index=True)

    out_csv = cfg.out_dir / cfg.out_csv_name
    long_df.to_csv(out_csv, index=False)

    meta = {
        "model_path": str(cfg.rf_model_path),
        "inputs": {
            "future_base_path": str(cfg.future_base_path),
            "future_low_path": str(cfg.future_low_path),
            "future_high_path": str(cfg.future_high_path),
        },
        "rows_per_scenario": int(cfg.max_rows_per_scenario),
        "rows_used_total": int(len(df_all)),
        "n_features_after_transform": int(shap_mat.shape[1]),
        "is_pipeline": bool(_is_pipeline(model_obj)),
        "output_csv": str(out_csv),
    }

    # optional meta json
    meta_path = cfg.out_dir / "forecast_shap_long_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


# ============================================================
# Main
# ============================================================
def main() -> None:
    from src.config import FORECAST_DIR, MODEL_DIR, OUTPUT_DIR  # type: ignore

    # match your predict endpoint inputs:
    future_dir = FORECAST_DIR / "future datasets"

    cfg = ForecastShapLongConfig(
        rf_model_path=MODEL_DIR / "avms" / "rf_final_model" / "rf_final_model.joblib",
        future_base_path=future_dir / "future_for_AVMs_base.csv",
        future_low_path=future_dir / "future_for_AVMs_low.csv",
        future_high_path=future_dir / "future_for_AVMs_high.csv",
        out_dir=OUTPUT_DIR / "shap",
        drop_cols=("Scenario",),
        max_rows_per_scenario=2000,
        random_state=42,
        out_csv_name="forecast_shap_long.csv",
    )

    meta = run_build_forecast_shap_long(cfg)

    print("\n[forecast_shap_long] Done.")
    print(f"- output_csv: {meta['output_csv']}")
    print(f"- is_pipeline: {meta['is_pipeline']}")
    print(f"- rows_used_total: {meta['rows_used_total']}")
    print(f"- n_features_after_transform: {meta['n_features_after_transform']}\n")


if __name__ == "__main__":
    main()
