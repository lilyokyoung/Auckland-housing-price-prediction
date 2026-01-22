# src/forecast_variables/predict_future_with_rf_final.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd



# ============================================================
# Config
# ============================================================
@dataclass
class PredictConfig:
    # --- model ---
    rf_final_model_path: Path

    # --- future scenario datasets (already feature-engineered: logs + lags) ---
    future_base_path: Path
    future_low_path: Path
    future_high_path: Path

    # --- historical dataset for continuity plot (e.g., PREPROCESSED_DIR/avms/final_for_AVMs.csv) ---
    historical_path: Path

    # --- output ---
    out_dir: Path

    # --- column names ---
    month_col: str = "Month"
    district_col: str = "District"
    target_col: str = "log_Median_Price"  # training target is log

    # columns that must NOT go into model.predict
    drop_feature_cols: List[str] = field(default_factory=lambda: ["Scenario"])

    # month parsing
    month_to_month_start: bool = True

    # performance toggles (for API)
    make_plots: bool = True
    write_csv: bool = True

    # how much history to show per district in continuity plots
    last_history_months: int = 36


# ============================================================
# I/O helpers
# ============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")


def parse_month(df: pd.DataFrame, month_col: str, to_month_start: bool) -> pd.DataFrame:
    if month_col not in df.columns:
        raise KeyError(f"Missing month column: {month_col}")
    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="raise")
    if to_month_start:
        out[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    else:
        out[month_col] = dt
    return out


# ============================================================
# Model input alignment
# ============================================================
def get_expected_input_columns(model) -> Optional[List[str]]:
    """
    If the saved Pipeline/Estimator exposes feature_names_in_, we use it.
    Otherwise return None and we will use "all columns except target".
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(getattr(model, "feature_names_in_"))
        return cols if cols else None
    return None


def drop_forbidden_feature_cols(df: pd.DataFrame, forbidden: Optional[List[str]]) -> pd.DataFrame:
    if not forbidden:
        return df
    present = [c for c in forbidden if c in df.columns]
    return df.drop(columns=present) if present else df


def build_X_for_model(df: pd.DataFrame, model, cfg: PredictConfig) -> pd.DataFrame:
    work = df.copy()

    # drop target if accidentally present
    if cfg.target_col in work.columns:
        work = work.drop(columns=[cfg.target_col])

    # drop forbidden feature cols (e.g. Scenario)
    work = drop_forbidden_feature_cols(work, cfg.drop_feature_cols)

    # required columns exist
    for c in [cfg.month_col, cfg.district_col]:
        if c not in work.columns:
            raise KeyError(f"Missing required column '{c}' in input table.")

    expected_cols = get_expected_input_columns(model)

    if expected_cols is None:
        # fallback: use everything that remains
        return work

    missing = [c for c in expected_cols if c not in work.columns]
    if missing:
        raise KeyError(
            "Input table is missing columns expected by the saved model.\n"
            f"Missing ({len(missing)}): {missing}\n\n"
            "Common cause: model was trained with a column you now dropped (e.g., Scenario).\n"
            "Fix: retrain the RF final model with Scenario removed from training features."
        )

    return work.loc[:, expected_cols].copy()


# ============================================================
# Prediction
# ============================================================
def predict_one_scenario(df_future: pd.DataFrame, model, cfg: PredictConfig, scenario_name: str) -> pd.DataFrame:
    df_future = parse_month(df_future, cfg.month_col, to_month_start=cfg.month_to_month_start)

    X = build_X_for_model(df_future, model, cfg)
    y_pred_log = np.asarray(model.predict(X), dtype=float)

    out = df_future[[cfg.month_col, cfg.district_col]].copy()
    out["Scenario"] = scenario_name
    out["pred_log_Median_Price"] = y_pred_log
    out["pred_Median_Price"] = np.exp(y_pred_log)  # back-transform to NZD
    return out.sort_values([cfg.district_col, cfg.month_col]).reset_index(drop=True)


def merge_scenarios_for_fanchart(
    base_df: pd.DataFrame,
    low_df: pd.DataFrame,
    high_df: pd.DataFrame,
    cfg: PredictConfig,
) -> pd.DataFrame:
    key = [cfg.month_col, cfg.district_col]

    b = base_df[key + ["pred_log_Median_Price", "pred_Median_Price"]].copy()
    l = low_df[key + ["pred_log_Median_Price", "pred_Median_Price"]].copy()
    h = high_df[key + ["pred_log_Median_Price", "pred_Median_Price"]].copy()

    b = b.rename(columns={
        "pred_log_Median_Price": "pred_log_Median_Price_base",
        "pred_Median_Price": "pred_Median_Price_base",
    })
    l = l.rename(columns={
        "pred_log_Median_Price": "pred_log_Median_Price_low",
        "pred_Median_Price": "pred_Median_Price_low",
    })
    h = h.rename(columns={
        "pred_log_Median_Price": "pred_log_Median_Price_high",
        "pred_Median_Price": "pred_Median_Price_high",
    })

    m = b.merge(l, on=key, how="inner").merge(h, on=key, how="inner")

    # Ensure low <= high (swap if needed)
    low = m["pred_Median_Price_low"].to_numpy()
    high = m["pred_Median_Price_high"].to_numpy()
    swap = low > high
    if swap.any():
        m.loc[swap, ["pred_Median_Price_low", "pred_Median_Price_high"]] = m.loc[
            swap, ["pred_Median_Price_high", "pred_Median_Price_low"]
        ].to_numpy()
        m.loc[swap, ["pred_log_Median_Price_low", "pred_log_Median_Price_high"]] = m.loc[
            swap, ["pred_log_Median_Price_high", "pred_log_Median_Price_low"]
        ].to_numpy()

    return m.sort_values([cfg.district_col, cfg.month_col]).reset_index(drop=True)


# ============================================================
# Historical continuity
# ============================================================
def prepare_historical_for_continuity(df_hist: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    df_hist = parse_month(df_hist, cfg.month_col, to_month_start=cfg.month_to_month_start)

    needed = [cfg.month_col, cfg.district_col]
    for c in needed:
        if c not in df_hist.columns:
            raise KeyError(f"Historical table missing: {c}")

    if "Median_Price" in df_hist.columns:
        out = df_hist[needed + ["Median_Price"]].copy()
        out["Median_Price"] = pd.to_numeric(out["Median_Price"], errors="coerce")
    elif cfg.target_col in df_hist.columns:
        out = df_hist[needed + [cfg.target_col]].copy()
        out[cfg.target_col] = pd.to_numeric(out[cfg.target_col], errors="coerce")
        out["Median_Price"] = np.exp(out[cfg.target_col])
        out = out.drop(columns=[cfg.target_col])
    else:
        raise KeyError("Historical table must contain either 'Median_Price' or the log target column.")

    out = out.dropna(subset=["Median_Price"]).copy()
    out = out.sort_values([cfg.district_col, cfg.month_col]).reset_index(drop=True)

    if cfg.last_history_months and cfg.last_history_months > 0:
        out = out.groupby(cfg.district_col, as_index=False, sort=False).tail(cfg.last_history_months)

    return out


# ============================================================
# Plotting
# ============================================================
def set_big_bold_title(title: str) -> None:
    plt.title(title, fontsize=18, fontweight="bold")


def plot_7districts_lines(pred_df: pd.DataFrame, cfg: PredictConfig, out_dir: Path, scen: str) -> None:
    ensure_dir(out_dir)

    dfp = pred_df.copy().sort_values([cfg.district_col, cfg.month_col])
    fig = plt.figure(figsize=(16, 6))

    for d, g in dfp.groupby(cfg.district_col, sort=True):
        plt.plot(g[cfg.month_col], g["pred_Median_Price"], label=str(d))

    set_big_bold_title(f"Future Predictions ({scen.upper()}) — 7 Districts")
    plt.xlabel("Month", fontsize=13)
    plt.ylabel("Predicted Median Price (NZD)", fontsize=13)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_path = out_dir / f"future_{scen}_7districts_line.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def plot_fanchart_by_district(merged: pd.DataFrame, cfg: PredictConfig, out_dir: Path) -> None:
    ensure_dir(out_dir)
    for d, g in merged.groupby(cfg.district_col, sort=True):
        g = g.sort_values(cfg.month_col)
        x = g[cfg.month_col]
        y_base = g["pred_Median_Price_base"]
        y_low = g["pred_Median_Price_low"]
        y_high = g["pred_Median_Price_high"]

        fig = plt.figure(figsize=(12, 5))
        plt.plot(x, y_base, label="Base")
        plt.fill_between(x, y_low, y_high, alpha=0.25, label="Low–High range")
        set_big_bold_title(f"Scenario Interval (Fan Chart) — {d}")
        plt.xlabel("Month", fontsize=13)
        plt.ylabel("Predicted Median Price (NZD)", fontsize=13)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"fan_{d}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_continuity_by_district(
    hist: pd.DataFrame,
    merged_future: pd.DataFrame,
    cfg: PredictConfig,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    hist_map: Dict[str, pd.DataFrame] = {
        str(d): g.sort_values(cfg.month_col) for d, g in hist.groupby(cfg.district_col, sort=True)
    }
    fut_map: Dict[str, pd.DataFrame] = {
        str(d): g.sort_values(cfg.month_col) for d, g in merged_future.groupby(cfg.district_col, sort=True)
    }

    districts = sorted(set(hist_map.keys()) | set(fut_map.keys()))
    for d in districts:
        h = hist_map.get(d)
        f = fut_map.get(d)
        if h is None or f is None:
            continue

        fig = plt.figure(figsize=(12, 5))
        plt.plot(h[cfg.month_col], h["Median_Price"], label="Historical")
        plt.plot(f[cfg.month_col], f["pred_Median_Price_base"], label="Future Base")
        plt.fill_between(
            f[cfg.month_col],
            f["pred_Median_Price_low"],
            f["pred_Median_Price_high"],
            alpha=0.25,
            label="Future Low–High range",
        )
        set_big_bold_title(f"Continuity: Historical → Future (Base/Low/High) — {d}")
        plt.xlabel("Month", fontsize=13)
        plt.ylabel("Predicted Median Price (NZD)", fontsize=13)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"continuity_{d}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================
def run_predict_future(cfg: PredictConfig) -> pd.DataFrame:
    """
    Core prediction pipeline.
    - Always computes predictions and returns a long DataFrame (for API/UI)
    - Optionally writes CSV outputs (cfg.write_csv)
    - Optionally generates plots (cfg.make_plots)
    """
    ensure_dir(cfg.out_dir)

    if not cfg.rf_final_model_path.exists():
        raise FileNotFoundError(f"RF final model not found: {cfg.rf_final_model_path}")

    model = joblib.load(cfg.rf_final_model_path)

    # Load future scenario tables
    df_base = load_table(cfg.future_base_path)
    df_low = load_table(cfg.future_low_path)
    df_high = load_table(cfg.future_high_path)

    # Predict
    pred_base = predict_one_scenario(df_base, model, cfg, scenario_name="base")
    pred_low = predict_one_scenario(df_low, model, cfg, scenario_name="low")
    pred_high = predict_one_scenario(df_high, model, cfg, scenario_name="high")

    # Merge for fan chart (also useful for plots)
    merged = merge_scenarios_for_fanchart(pred_base, pred_low, pred_high, cfg)

    # Long table for API/UI downstream use
    pred_all_long: pd.DataFrame = pd.concat([pred_base, pred_low, pred_high], ignore_index=True)

    # Save outputs (optional)
    if cfg.write_csv:
        pred_base.to_csv(cfg.out_dir / "pred_base.csv", index=False, encoding="utf-8-sig")
        pred_low.to_csv(cfg.out_dir / "pred_low.csv", index=False, encoding="utf-8-sig")
        pred_high.to_csv(cfg.out_dir / "pred_high.csv", index=False, encoding="utf-8-sig")
        merged.to_csv(cfg.out_dir / "pred_base_low_high_merged.csv", index=False, encoding="utf-8-sig")
        pred_all_long.to_csv(cfg.out_dir / "pred_all_scenarios_long.csv", index=False, encoding="utf-8-sig")

    # Plots (optional)
    if cfg.make_plots:
        plot_root = cfg.out_dir / "plots"
        ensure_dir(plot_root)

        # 1) 7 districts × time (base/low/high)
        line_dir = plot_root / "lines_7districts"
        plot_7districts_lines(pred_base, cfg, line_dir, "base")
        plot_7districts_lines(pred_low, cfg, line_dir, "low")
        plot_7districts_lines(pred_high, cfg, line_dir, "high")

        # 2) fan chart per district
        plot_fanchart_by_district(merged, cfg, plot_root / "fan_charts")

        # 3) continuity vs historical (only needed when plotting)
        df_hist = load_table(cfg.historical_path)
        hist_small = prepare_historical_for_continuity(df_hist, cfg)
        plot_continuity_by_district(hist_small, merged, cfg, plot_root / "continuity")

    print("[OK] Predictions computed.")
    if cfg.write_csv:
        print("[OK] Saved predictions to :", cfg.out_dir.resolve())
    if cfg.make_plots:
        print("[OK] Saved plots to       :", (cfg.out_dir / "plots").resolve())

    return pred_all_long


def main() -> None:
    """
    Run:
      python -m src.forecast_variables.predict_future_with_rf_final
    """
    from src.config import FORECAST_DIR, PREPROCESSED_DIR, MODEL_DIR

    future_dir = FORECAST_DIR / "future datasets"  # your folder name has a space

    cfg = PredictConfig(
        rf_final_model_path=MODEL_DIR / "avms" / "rf_final_model" / "rf_final_model.joblib",
        future_base_path=future_dir / "future_for_AVMs_base.csv",
        future_low_path=future_dir / "future_for_AVMs_low.csv",
        future_high_path=future_dir / "future_for_AVMs_high.csv",
        historical_path=PREPROCESSED_DIR / "avms" / "final_for_AVMs.csv",
        out_dir=MODEL_DIR / "rf_final_predictions",
        drop_feature_cols=["Scenario"],  # IMPORTANT: do not feed Scenario into model

        # local run: keep these True if you want csv + plots
        make_plots=True,
        write_csv=True,

        last_history_months=36,
        month_to_month_start=True,
    )

    _ = run_predict_future(cfg)


if __name__ == "__main__":
    main()
