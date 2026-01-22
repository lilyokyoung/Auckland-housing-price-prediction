# src/explain/compute_shap_rf.py
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
class ShapRFConfig:
    """
    Compute SHAP for a Random Forest (or any tree estimator / pipeline ending with a tree estimator),
    and export outputs usable for:
      1) global importance (mean abs SHAP)
      2) per-row SHAP with identifiers (Month/District/Scenario)
      3) long format SHAP for easy API querying (district+scenario+month -> top drivers)

    Pipeline-safe:
    - If model is a sklearn Pipeline(preprocess + estimator), we apply preprocess.transform(X_raw).
    - We try hard to align X_raw columns to what the fitted pipeline expects.
    """

    # ---- inputs ----
    rf_model_path: Path
    data_path: Path
    out_dir: Path

    # ---- identifiers for grouping / filtering ----
    id_col_candidates: Tuple[str, ...] = ("Month", "District", "Scenario")

    # ---- target and feature handling ----
    target_col_candidates: Tuple[str, ...] = ("log_Median_Price", "Median_Price", "Target", "y")
    drop_cols: Tuple[str, ...] = ("Scenario",)  # dropped from FEATURES only (not from id cols)

    # ---- sampling ----
    max_rows: int = 2000
    random_state: int = 42

    # ---- plots / top display ----
    max_display: int = 20

    # ---- outputs ----
    write_summary_plot: bool = True
    write_wide_by_row: bool = True
    write_long_by_row: bool = True

    # ---- long format extras (for agent) ----
    write_long_with_feature_values: bool = True   # ✅ add feature_value column
    include_pred_and_base_value: bool = True      # ✅ add pred_log + base_value


# ============================================================
# Utilities
# ============================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def _load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    suf = data_path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(data_path)
    if suf == ".csv":
        return pd.read_csv(data_path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(data_path)
    raise ValueError(f"Unsupported file type: {data_path.suffix}")


def _pick_target_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_id_cols(df: pd.DataFrame, candidates: Tuple[str, ...]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _sample_df(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0:
        return df.copy()
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


def _safe_to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    SHAP TreeExplainer expects numeric inputs.
    If non-numeric columns exist, factorize them deterministically.
    """
    X2 = X.copy()
    for c in X2.columns:
        if not np.issubdtype(X2[c].dtype, np.number):  # type: ignore
            X2[c] = pd.factorize(X2[c].astype(str))[0].astype(float)
    return X2


def _is_pipeline(model_obj) -> bool:
    return hasattr(model_obj, "steps") and hasattr(model_obj, "__len__")


def _split_pipeline(model_obj):
    """
    Returns: (preprocess, estimator)
    - preprocess: pipeline without last step (may be None)
    - estimator: last step estimator
    """
    if not _is_pipeline(model_obj):
        return None, model_obj
    if len(model_obj.steps) < 1:
        return None, model_obj
    estimator = model_obj.steps[-1][1]
    preprocess = model_obj[:-1]
    return preprocess, estimator


def _get_transformed_feature_names(preprocess, input_features: List[str]) -> List[str]:
    """
    Try to recover feature names after transformation.
    """
    if preprocess is None:
        return input_features

    if hasattr(preprocess, "get_feature_names_out"):
        try:
            names = preprocess.get_feature_names_out(input_features)
            return [str(x) for x in names]
        except Exception:
            pass

    if hasattr(preprocess, "steps"):
        try:
            last = preprocess.steps[-1][1]
            if hasattr(last, "get_feature_names_out"):
                names = last.get_feature_names_out(input_features)
                return [str(x) for x in names]
        except Exception:
            pass

    return [f"f_{i}" for i in range(len(input_features))]


def _get_expected_input_features_from_pipeline(model_obj) -> Optional[List[str]]:
    """
    Try to retrieve expected input feature names for preprocessing transform.

    Works for many sklearn pipelines where preprocess step was fitted with pandas DataFrame.
    """
    if hasattr(model_obj, "feature_names_in_") and getattr(model_obj, "feature_names_in_", None) is not None:
        return [str(x) for x in list(model_obj.feature_names_in_)]

    if hasattr(model_obj, "steps"):
        for _, step in model_obj.steps:
            if hasattr(step, "feature_names_in_") and getattr(step, "feature_names_in_", None) is not None:
                return [str(x) for x in list(step.feature_names_in_)]

    try:
        est = model_obj.steps[-1][1] if hasattr(model_obj, "steps") else None
        if est is not None and hasattr(est, "feature_names_in_") and getattr(est, "feature_names_in_", None) is not None:
            return [str(x) for x in list(est.feature_names_in_)]
    except Exception:
        pass

    return None


def _infer_feature_columns_fallback(
    model_obj,
    df: pd.DataFrame,
    target_col: Optional[str],
    drop_cols: Tuple[str, ...],
    id_cols: List[str],
) -> List[str]:
    """
    Fallback: decide which columns are features when we can't retrieve pipeline expected inputs.
    Prefer numeric columns excluding target/id/drop.
    """
    cols = df.columns.tolist()
    exclude = set(drop_cols) | set(id_cols)
    if target_col:
        exclude.add(target_col)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in exclude]
    if feats:
        return feats

    return [c for c in cols if c not in exclude]


def _align_columns(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Ensure df contains exactly required_cols (in that order):
    - missing cols -> filled with 0
    - extra cols -> dropped
    """
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = 0.0
    extra = [c for c in out.columns if c not in required_cols]
    if extra:
        out = out.drop(columns=extra)
    return out[required_cols]


def _transform_X_if_needed(model_obj, X_raw: pd.DataFrame) -> Tuple[np.ndarray, List[str], Any]:
    """
    If model is Pipeline(preprocess + estimator), apply preprocess.transform(X_raw)
    and return (X_transformed, feature_names_transformed, estimator_only).
    Else return (X_raw_numeric_values, feature_names, estimator).
    """
    preprocess, estimator = _split_pipeline(model_obj)

    if preprocess is None:
        X_num = _safe_to_numeric_matrix(X_raw)
        return X_num.to_numpy(), list(X_num.columns), estimator

    Xt = preprocess.transform(X_raw)

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


def _compute_shap_and_base(estimator, X_mat: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns:
      shap_mat: (n, p)
      base_value: scalar
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
        shap_mat = np.asarray(shap_values[0])
    else:
        shap_mat = np.asarray(shap_values)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).ravel()[0])
    else:
        base_value = float(base_value) # type: ignore

    return shap_mat, base_value


def _save_summary_plot(estimator, X_mat: np.ndarray, feature_names: List[str], out_png: Path, max_display: int) -> None:
    import matplotlib.pyplot as plt
    import shap  # type: ignore

    X_df = pd.DataFrame(X_mat, columns=feature_names)

    plt.figure()
    shap.summary_plot(
        shap.TreeExplainer(estimator).shap_values(X_df),
        X_df,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def _mean_abs_shap(shap_mat: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    imp = np.mean(np.abs(shap_mat), axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _format_month_col(df: pd.DataFrame, month_col: str) -> pd.Series:
    """
    Normalize month column to string 'YYYY-MM'.
    """
    s = df[month_col]
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().any():
        return dt.dt.strftime("%Y-%m")
    return s.astype(str).str.strip().str.slice(0, 7)


# ============================================================
# Main runner
# ============================================================
def run_compute_shap_rf(cfg: ShapRFConfig, *, expected_inputs_override: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    expected_inputs_override:
      - If provided, force X_raw to use exactly these input columns (aligned + ordered).
      - This is the key to making base/low/high strictly consistent.
    """
    _ensure_dir(cfg.out_dir)

    model_obj = _load_model(cfg.rf_model_path)
    df0 = _load_data(cfg.data_path)

    # ---- sample rows ONCE so ids and features stay aligned ----
    df = _sample_df(df0, cfg.max_rows, cfg.random_state).copy()

    # ---- identify id columns ----
    id_cols = _pick_id_cols(df, cfg.id_col_candidates)

    # normalize ids
    if "Month" in id_cols:
        df["Month"] = _format_month_col(df, "Month")
    if "Scenario" in id_cols:
        df["Scenario"] = df["Scenario"].astype(str).str.strip().str.lower()
    if "District" in id_cols:
        df["District"] = df["District"].astype(str).str.strip()

    id_df = df[id_cols].copy() if id_cols else None

    # ---- choose X_raw columns ----
    target_col = _pick_target_col(df, cfg.target_col_candidates)

    expected_inputs = expected_inputs_override or _get_expected_input_features_from_pipeline(model_obj)

    if expected_inputs:
        # align + order
        missing = [c for c in expected_inputs if c not in df.columns]
        if missing:
            raise KeyError(
                "Input data is missing columns required by the fitted pipeline / expected inputs.\n"
                f"Missing: {missing}\n"
                f"Available: {df.columns.tolist()}"
            )
        X_raw = df[expected_inputs].copy()
    else:
        # fallback inference
        feature_cols = _infer_feature_columns_fallback(model_obj, df, target_col, cfg.drop_cols, id_cols)
        X_raw = df[feature_cols].copy()
        for c in cfg.drop_cols:
            if c in X_raw.columns:
                X_raw = X_raw.drop(columns=[c])

    # ---- pipeline-safe transform ----
    try:
        X_mat, feat_names, estimator = _transform_X_if_needed(model_obj, X_raw)
    except Exception as e:
        raise RuntimeError(
            "Preprocess transform failed.\n"
            f"- is_pipeline: {bool(_is_pipeline(model_obj))}\n"
            f"- expected_inputs_found: {bool(expected_inputs)}\n"
            f"- X_raw shape: {X_raw.shape}\n"
            f"- X_raw columns (first 50): {list(X_raw.columns)[:50]}\n"
            f"- Error: {type(e).__name__}: {e}"
        ) from e

    # ---- compute shap ----
    shap_mat, base_value = _compute_shap_and_base(estimator, X_mat)

    # ---- derived prediction (in log space) ----
    pred_log = base_value + shap_mat.sum(axis=1)

    # ---- outputs ----
    out_png = cfg.out_dir / "shap_summary_rf.png"
    out_imp = cfg.out_dir / "shap_importance_rf.csv"
    out_values_wide = cfg.out_dir / "shap_values_rf.parquet"  # features only
    out_by_row_wide = cfg.out_dir / "shap_by_row_rf.parquet"  # id + features
    out_long = cfg.out_dir / "shap_long_rf.parquet"           # id + feature + shap_value (+ extras)
    out_long_csv = cfg.out_dir / "shap_long_rf.csv"
    out_meta = cfg.out_dir / "shap_meta_rf.json"

    if cfg.write_summary_plot:
        _save_summary_plot(estimator, X_mat, feat_names, out_png, cfg.max_display)

    imp_df = _mean_abs_shap(shap_mat, feat_names)
    imp_df.to_csv(out_imp, index=False, encoding="utf-8-sig")

    shap_df = pd.DataFrame(shap_mat, columns=feat_names)
    shap_df.to_parquet(out_values_wide, index=False)

    # wide by row
    if cfg.write_wide_by_row:
        if id_df is not None and len(id_cols) > 0:
            wide_by_row = pd.concat([id_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
        else:
            wide_by_row = shap_df.copy()
        wide_by_row.to_parquet(out_by_row_wide, index=False)

    # long by row (+ feature_value + pred/base)
    if cfg.write_long_by_row:
        # build a transformed-X dataframe for feature_value (works for both pipeline/non-pipeline)
        X_df = pd.DataFrame(X_mat, columns=feat_names)

        if id_df is not None and len(id_cols) > 0:
            # melt SHAP
            wide_for_melt = pd.concat([id_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
            long_df = wide_for_melt.melt(id_vars=id_cols, var_name="feature", value_name="shap_value")

            if cfg.write_long_with_feature_values:
                wide_x = pd.concat([id_df.reset_index(drop=True), X_df.reset_index(drop=True)], axis=1)
                long_x = wide_x.melt(id_vars=id_cols, var_name="feature", value_name="feature_value")
                long_df = long_df.merge(long_x, on=id_cols + ["feature"], how="left")
        else:
            long_df = shap_df.melt(var_name="feature", value_name="shap_value")
            if cfg.write_long_with_feature_values:
                long_df["feature_value"] = X_df.to_numpy().reshape(-1)  # fallback (rarely used)

        long_df["abs_shap"] = long_df["shap_value"].abs()
        long_df["direction"] = np.where(long_df["shap_value"] >= 0, "up", "down")

        if cfg.include_pred_and_base_value:
            # attach per-row pred/base
            if id_df is not None and len(id_cols) > 0:
                row_meta = id_df.copy().reset_index(drop=True)
                row_meta["pred_log"] = pred_log
                row_meta["base_value"] = base_value
                long_df = long_df.merge(row_meta, on=id_cols, how="left")
            else:
                long_df["pred_log"] = pred_log.repeat(len(feat_names))
                long_df["base_value"] = base_value

        long_df.to_parquet(out_long, index=False)
        long_df.to_csv(out_long_csv, index=False, encoding="utf-8-sig")

    meta: Dict[str, Any] = {
        "model_path": str(cfg.rf_model_path),
        "data_path": str(cfg.data_path),
        "rows_used": int(X_mat.shape[0]),
        "n_features": int(X_mat.shape[1]),
        "target_col_detected": target_col,
        "id_cols_detected": id_cols,
        "expected_inputs_override": expected_inputs_override if expected_inputs_override else None,
        "expected_inputs_found": bool(expected_inputs),
        "expected_inputs_n": int(len(expected_inputs)) if expected_inputs else 0,
        "is_pipeline": bool(_is_pipeline(model_obj)),
        "base_value": float(base_value),
        "outputs": {
            "summary_plot": str(out_png) if cfg.write_summary_plot else None,
            "importance_csv": str(out_imp),
            "shap_values_wide_features_only": str(out_values_wide),
            "shap_by_row_wide": str(out_by_row_wide) if cfg.write_wide_by_row else None,
            "shap_long": str(out_long) if cfg.write_long_by_row else None,
            "shap_long_csv": str(out_long_csv) if cfg.write_long_by_row else None,
        },
        "top_features": imp_df.head(10).to_dict(orient="records"),
        "notes": {
            "long_format_usage": (
                "Filter by District/Scenario/Month, then aggregate per feature: "
                "mean(shap_value) and mean(abs_shap) to get ranked drivers."
            ),
            "month_format": "If Month exists, it is normalized to 'YYYY-MM'.",
            "pred_definition": "pred_log = base_value + sum(shap_value over all features).",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ============================================================
# CLI: FUTURE scenario SHAP ONLY (base/low/high) + merged output
# ============================================================
def _mean_abs_shap_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    need = {"feature", "abs_shap"}
    missing = need - set(df_long.columns)
    if missing:
        raise ValueError(f"long df missing columns: {sorted(missing)}")

    out = (
        df_long
        .groupby("feature", as_index=False)
        .agg(mean_abs_shap=("abs_shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return out

def main() -> None:
    """
    Run:
      python -m src.explain.compute_shap_rf

    Outputs (recommended for API/Agent):
      outputs/shap_forecast/forecast_shap_long.parquet
      outputs/shap_forecast/forecast_shap_long.csv
      outputs/shap_forecast/shap_importance_rf.csv
    """
    from src.config import MODEL_DIR, FORECAST_DIR, OUTPUT_DIR  # type: ignore

    rf_model_path = MODEL_DIR / "avms" / "rf_final_model" / "rf_final_model.joblib"

    # Your folder name contains a space
    future_dir = FORECAST_DIR / "future datasets"
    future_paths = {
        "base": future_dir / "future_for_AVMs_base.csv",
        "low": future_dir / "future_for_AVMs_low.csv",
        "high": future_dir / "future_for_AVMs_high.csv",
    }

    out_root = OUTPUT_DIR / "shap_forecast"
    _ensure_dir(out_root)

    # ✅ IMPORTANT: lock the expected input columns using BASE as reference
    df_base = _load_data(future_paths["base"])
    if "Month" in df_base.columns:
        df_base["Month"] = _format_month_col(df_base, "Month")
    if "District" not in df_base.columns:
        raise KeyError("Base future dataset must contain 'District' column.")
    expected_inputs_locked = [c for c in df_base.columns if c not in ("Month", "District", "Scenario", "log_Median_Price", "Median_Price", "Target", "y")]

    # If your pipeline expects specific inputs, we prefer that:
    model_obj = _load_model(rf_model_path)
    pipeline_expected = _get_expected_input_features_from_pipeline(model_obj)
    if pipeline_expected:
        expected_inputs_locked = pipeline_expected

    all_long: List[pd.DataFrame] = []

    for scen, data_path in future_paths.items():
        df_s = _load_data(data_path)

        # Ensure Scenario exists for ids (even if not in file)
        if "Scenario" not in df_s.columns:
            df_s["Scenario"] = scen
        else:
            df_s["Scenario"] = scen

        # ✅ align feature columns to locked set (so 3 scenarios are identical in input space)
        # if locked is pipeline_expected -> keep strict
        # else locked is base-derived features -> keep strict
        missing = [c for c in expected_inputs_locked if c not in df_s.columns]
        if missing:
            # fill missing with 0 (safe default for scenario projections)
            for c in missing:
                df_s[c] = 0.0
        extra = [c for c in df_s.columns if c not in set(expected_inputs_locked) | {"Month", "District", "Scenario"}]
        if extra:
            df_s = df_s.drop(columns=extra)

        # write a temp aligned parquet in output (so run_compute_shap_rf reads aligned view)
        aligned_path = out_root / f"_aligned_{scen}.parquet"
        df_s.to_parquet(aligned_path, index=False)

        cfg = ShapRFConfig(
            rf_model_path=rf_model_path,
            data_path=aligned_path,
            out_dir=out_root / scen,
            drop_cols=("Scenario",),
            max_rows=0,                # ✅ do NOT sample future rows
            random_state=42,
            max_display=20,
            write_summary_plot=False,  # keep light
            write_wide_by_row=False,   # keep light
            write_long_by_row=True,    # ✅ we need long format
            write_long_with_feature_values=True,  # ✅ add feature_value
            include_pred_and_base_value=True,     # ✅ add pred_log/base_value
        )

        meta = run_compute_shap_rf(cfg, expected_inputs_override=expected_inputs_locked)

        long_parquet = Path(meta["outputs"]["shap_long"])
        df_long = pd.read_parquet(long_parquet)

        # Force Scenario + Month formatting again (id-safety)
        df_long["Scenario"] = scen
        if "Month" in df_long.columns:
            df_long["Month"] = df_long["Month"].astype(str).str.slice(0, 7)

        all_long.append(df_long)

        print(f"[OK] future SHAP ({scen}) -> {long_parquet}")

    df_all = pd.concat(all_long, ignore_index=True)

    out_all_parquet = out_root / "forecast_shap_long.parquet"
    out_all_csv = out_root / "forecast_shap_long.csv"
    df_all.to_parquet(out_all_parquet, index=False)
    df_all.to_csv(out_all_csv, index=False, encoding="utf-8-sig")

    # ✅ merged global importance for agent
    imp = _mean_abs_shap_from_long(df_all)
    out_imp = out_root / "shap_importance_rf.csv"
    imp.to_csv(out_imp, index=False, encoding="utf-8-sig")

    print("\n[SHAP Forecast] Done.")
    print(f"- merged parquet: {out_all_parquet}")
    print(f"- merged csv:     {out_all_csv}")
    print(f"- importance:     {out_imp}")
    print(f"- rows:           {len(df_all):,}")


if __name__ == "__main__":
    main()
