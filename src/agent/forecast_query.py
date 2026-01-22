# src/agent/forecast_query.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd


# ============================================================
# utils
# ============================================================
def norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip().lower()
    for ch in [" ", "_", "-", ".", "\t", "\n", "\r"]:
        s = s.replace(ch, "")
    return s


def norm_scenario(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().lower()


def norm_month(x: Any) -> str:
    """
    Normalize month to 'YYYY-MM'.

    Supports:
    - pd.Timestamp
    - 'YYYY-MM', 'YYYY-MM-DD', 'YYYY/MM/DD'
    - Excel-ish '1/07/2025' -> parsed with dayfirst=True
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""

    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m")

    s = str(x).strip()
    if not s:
        return ""

    # ISO-like quick match
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # try parse excel-ish
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return pd.Timestamp(dt).strftime("%Y-%m")

    return s[:7] if len(s) >= 7 else s


def _to_float_safe(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


# ============================================================
# config
# ============================================================
@dataclass(frozen=True)
class ForecastQueryConfig:
    """
    Forecast prediction file location.

    Default:
      outputs/models/rf_final_predictions/pred_all_scenarios_long.(parquet|csv|xlsx)
    """
    project_root: Path
    model_subdir: Path = Path("outputs") / "models" / "rf_final_predictions"
    preferred_stems: Tuple[str, ...] = (
        "pred_all_scenarios_long",
        "pred_base_low_high_merged",
        "pred_all_scenarios",
        "pred_merged",
    )

    month_cols: Tuple[str, ...] = ("Month", "month", "Date", "date")
    scenario_cols: Tuple[str, ...] = ("Scenario", "scenario")
    district_cols: Tuple[str, ...] = ("District", "district")

    # restored price columns (your long table uses pred_Median_Price)
    price_cols: Tuple[str, ...] = (
        "pred_Median_Price",
        "Pred_Median_Price",
        "pred_median_price",
        "Predicted_Price",
        "predicted_price",
        "PredPrice",
        "pred_price",
        "price",
        "Price",
        "y_hat",
        "yhat",
    )

    # log columns (your long table uses pred_log_Median_Price)
    pred_log_cols: Tuple[str, ...] = (
        "pred_log_Median_Price",
        "Pred_log_Median_Price",
        "pred_log",
        "Pred_log",
        "log_pred",
        "log_price",
        "Predicted_log",
    )


class ForecastQueryEngine:
    def __init__(self, cfg: ForecastQueryConfig) -> None:
        self.cfg = cfg
        self.pred_dir = (cfg.project_root / cfg.model_subdir).resolve()
        self.path = self._pick_forecast_path()
        self.df = self._load(self.path)

        # detect columns
        self.col_month = self._detect(self.df.columns, cfg.month_cols, required=False)
        self.col_scenario = self._detect(self.df.columns, cfg.scenario_cols, required=False)

        # district is required -> cast to str to avoid Optional[str] red lines
        self.col_district = cast(str, self._detect(self.df.columns, cfg.district_cols, required=True))

        self.col_price = self._detect(self.df.columns, cfg.price_cols, required=False)
        self.col_pred_log = self._detect(self.df.columns, cfg.pred_log_cols, required=False)

        # normalize key columns
        self.df = self._normalize_keys(self.df)

    # -------------------------
    # internal
    # -------------------------
    def _pick_forecast_path(self) -> Path:
        if not self.pred_dir.exists():
            raise FileNotFoundError(f"Prediction folder not found: {self.pred_dir}")

        candidates: List[Path] = []

        # 1) prefer known stems
        for stem in self.cfg.preferred_stems:
            for ext in (".parquet", ".csv", ".xlsx", ".xls"):
                p = self.pred_dir / f"{stem}{ext}"
                if p.exists():
                    candidates.append(p)

        # 2) fallback: heuristic search
        if not candidates:
            for ext in ("*.parquet", "*.csv", "*.xlsx", "*.xls"):
                for p in self.pred_dir.glob(ext):
                    name = p.stem.lower()
                    if any(tok in name for tok in ["scenario", "merged", "forecast", "pred"]):
                        candidates.append(p)

        # 3) fallback: newest file
        if not candidates:
            all_files: List[Path] = []
            for ext in ("*.parquet", "*.csv", "*.xlsx", "*.xls"):
                all_files.extend(list(self.pred_dir.glob(ext)))
            if not all_files:
                raise FileNotFoundError(f"No prediction files found under: {self.pred_dir}")
            all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return all_files[0]

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _load(self, path: Path) -> pd.DataFrame:
        suf = path.suffix.lower()
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf == ".csv":
            return pd.read_csv(path)
        if suf in (".xlsx", ".xls"):
            return pd.read_excel(path)
        raise ValueError(f"Unsupported forecast file type: {path}")

    def _detect(self, cols: Any, candidates: Tuple[str, ...], required: bool) -> Optional[str]:
        cols_list = list(cols)

        for c in candidates:
            if c in cols_list:
                return c

        cols_norm = {norm_text(c): c for c in cols_list}
        for c in candidates:
            key = norm_text(c)
            if key in cols_norm:
                return cols_norm[key]

        if required:
            raise KeyError(f"Cannot detect required column from {candidates}. Available: {cols_list}")
        return None

    def _normalize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out[self.col_district] = out[self.col_district].astype(str).str.strip()

        if self.col_scenario:
            out[self.col_scenario] = out[self.col_scenario].map(norm_scenario)

        if self.col_month:
            out[self.col_month] = out[self.col_month].map(norm_month)

        return out

    def _value_series(self, df: pd.DataFrame) -> Tuple[pd.Series, str]:
        """
        Return numeric prediction values to use for ranking, and a mode string.
        Preference:
          1) restored price column
          2) exp(pred_log) if only log exists
        """
        if self.col_price and self.col_price in df.columns:
            s = pd.to_numeric(df[self.col_price], errors="coerce")
            return s, f"price({self.col_price})"

        if self.col_pred_log and self.col_pred_log in df.columns:
            logv = pd.to_numeric(df[self.col_pred_log], errors="coerce")
            # IMPORTANT: ensure we return pd.Series (avoid ndarray type warnings)
            restored = pd.Series(np.exp(logv.to_numpy(dtype=float)), index=logv.index)
            return restored, f"exp(pred_log={self.col_pred_log})"

        raise KeyError("No price or pred_log column found in forecast file.")

    # ============================================================
    # public APIs
    # ============================================================
    def list_available(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ok": True, "source_path": str(self.path)}
        out["columns"] = list(self.df.columns)

        out["districts"] = sorted(
            [d for d in self.df[self.col_district].dropna().unique().tolist() if str(d).strip()]
        )

        if self.col_month:
            out["months"] = sorted(
                [m for m in self.df[self.col_month].dropna().unique().tolist() if str(m).strip()]
            )

        if self.col_scenario:
            out["scenarios"] = sorted(
                [s for s in self.df[self.col_scenario].dropna().unique().tolist() if str(s).strip()]
            )

        return out

    def best_district(
        self,
        month: Optional[str],
        scenario: Optional[str],
        k: int = 3,
        highest: bool = True,
        agg_method: str = "mean",
    ) -> Dict[str, Any]:
        """
        Rank districts by prediction value (restored price preferred).

        If month is provided (and Month column exists), we filter to that month.
        If month is not provided, we aggregate across months by district.
        """
        df_f = self.df

        month_n = norm_month(month) if (month and self.col_month) else ""
        scen_n = norm_scenario(scenario) if (scenario and self.col_scenario) else ""

        if self.col_month and month_n:
            df_f = df_f[df_f[self.col_month] == month_n]

        if self.col_scenario and scen_n:
            df_f = df_f[df_f[self.col_scenario] == scen_n]

        if df_f.empty:
            return {
                "ok": False,
                "message": f"No forecast data found for month={month_n or 'ALL'} scenario={scen_n or 'ALL'}.",
                "debug": {"source_path": str(self.path)},
            }

        try:
            values, value_mode = self._value_series(df_f)
        except KeyError as e:
            return {
                "ok": False,
                "message": str(e),
                "debug": {"source_path": str(self.path), "columns": list(self.df.columns)},
            }

        work = df_f.copy()
        work["_pv"] = pd.to_numeric(values, errors="coerce")
        work = work.dropna(subset=["_pv"])

        if work.empty:
            return {
                "ok": False,
                "message": f"No valid numeric prediction values for month={month_n or 'ALL'} scenario={scen_n or 'ALL'}.",
                "debug": {"value_mode": value_mode, "source_path": str(self.path)},
            }

        # Aggregate if we didn't filter a specific month (and month col exists)
        if self.col_month and not month_n:
            agg_method_n = (agg_method or "mean").strip().lower()

            if agg_method_n == "last":
                tmp = work.sort_values(self.col_month)
                agg_df = tmp.groupby(self.col_district, as_index=False).tail(1)[[self.col_district, "_pv"]]
            elif agg_method_n == "max":
                agg_df = work.groupby(self.col_district)["_pv"].max().reset_index()
            elif agg_method_n == "min":
                agg_df = work.groupby(self.col_district)["_pv"].min().reset_index()
            else:
                agg_df = work.groupby(self.col_district)["_pv"].mean().reset_index()
        else:
            # month specified (or Month col absent) => just dedupe safely
            agg_df = work.groupby(self.col_district)["_pv"].mean().reset_index()

        agg_df = agg_df.sort_values("_pv", ascending=not highest).head(int(max(1, k)))

        records: List[Dict[str, Any]] = []
        for _, row in agg_df.iterrows():
            pv = _to_float_safe(row["_pv"])
            if pv is None:
                continue
            records.append(
                {
                    "district": str(row[self.col_district]).strip(),
                    "pred_price": float(pv),
                }
            )

        if not records:
            return {"ok": False, "message": "No valid ranking records after aggregation."}

        return {
            "ok": True,
            "month": month_n or None,
            "scenario": scen_n or None,
            "mode": value_mode,
            "highest": bool(highest),
            "k": int(max(1, k)),
            "agg_method": agg_method if (self.col_month and not month_n) else None,
            "ranking": records,
            "source_path": str(self.path),
        }
