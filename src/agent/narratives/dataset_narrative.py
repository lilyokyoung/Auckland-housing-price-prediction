# src/agent/narratives/dataset_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class DatasetNarrativeConfig:
    modeling_table_path: Path  # e.g., data/preprocessed/avms/final_for_AVMs.csv
    month_col: str = "Month"
    district_col: str = "District"


def _to_month_start(x: pd.Series) -> pd.Series:
    dt = pd.to_datetime(x, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp() # type: ignore


def _infer_max_lag_from_columns(cols: List[str]) -> int:
    # detect patterns like OCR_lag12, unemployment_lag6, etc.
    max_lag = 0
    for c in cols:
        s = str(c)
        if "_lag" in s:
            try:
                tail = s.split("_lag")[-1]
                n = int("".join(ch for ch in tail if ch.isdigit()))
                max_lag = max(max_lag, n)
            except Exception:
                continue
    return max_lag


def _month_range(months: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    months = months.dropna()
    if months.empty:
        return None, None
    return months.min(), months.max()


def _detect_missing_months(months: pd.Series) -> List[str]:
    months = months.dropna().sort_values().unique() # type: ignore
    if len(months) == 0:
        return []
    start = pd.Timestamp(months[0])
    end = pd.Timestamp(months[-1])
    full = pd.date_range(start=start, end=end, freq="MS")
    missing = sorted(set(full) - set(months))
    return [m.strftime("%Y-%m") for m in missing]


def build_dataset_meta(cfg: DatasetNarrativeConfig) -> Dict[str, Any]:
    if not cfg.modeling_table_path.exists():
        return {"ok": False, "error": f"Missing file: {cfg.modeling_table_path}"}

    df = pd.read_csv(cfg.modeling_table_path)

    # basic counts
    rows, cols = int(df.shape[0]), int(df.shape[1])

    # months
    if cfg.month_col in df.columns:
        months = _to_month_start(df[cfg.month_col])
    else:
        months = pd.Series([], dtype="datetime64[ns]")

    eff_start, eff_end = _month_range(months)
    missing_eff = _detect_missing_months(months)

    # districts
    if cfg.district_col in df.columns:
        districts = sorted(df[cfg.district_col].dropna().unique().tolist())
    else:
        districts = []

    # max lag inference
    max_lag = _infer_max_lag_from_columns(list(df.columns))

    # "raw start" approximation: effective_start - max_lag months
    raw_start = None
    if eff_start is not None and max_lag > 0:
        raw_start = (eff_start - pd.DateOffset(months=max_lag)).to_period("M").to_timestamp()

    meta = {
        "ok": True,
        "modeling_table": str(cfg.modeling_table_path),
        "rows": rows,
        "cols": cols,
        "district_count": len(districts),
        "districts": districts,
        "effective_start": eff_start.strftime("%Y-%m") if eff_start is not None else None,
        "effective_end": eff_end.strftime("%Y-%m") if eff_end is not None else None,
        "unique_months": int(months.dropna().nunique()) if len(months) else 0,
        "missing_months_effective": missing_eff,
        "max_lag_inferred": int(max_lag),
        "raw_start_inferred": raw_start.strftime("%Y-%m") if raw_start is not None else None,
        "raw_end_inferred": eff_end.strftime("%Y-%m") if eff_end is not None else None,
    }
    return meta


def build_dataset_narrative(meta: Dict[str, Any]) -> str:
    if not meta.get("ok"):
        return f"Dataset summary unavailable: {meta.get('error', 'unknown error')}"

    eff_start = meta.get("effective_start")
    eff_end = meta.get("effective_end")
    raw_start = meta.get("raw_start_inferred")
    raw_end = meta.get("raw_end_inferred")
    max_lag = meta.get("max_lag_inferred", 0)
    rows = meta.get("rows", 0)
    cols = meta.get("cols", 0)
    dcount = meta.get("district_count", 0)
    miss_eff = meta.get("missing_months_effective", [])

    parts: List[str] = []

    # coverage
    if raw_start and raw_end and max_lag:
        parts.append(
            f"**Raw coverage (inferred from lag design)**: {raw_start} → {raw_end}. "
            f"The modeling table uses lagged features up to **lag-{max_lag}**, so the first {max_lag} months are naturally truncated in the effective sample."
        )
    if eff_start and eff_end:
        parts.append(f"**Effective modeling sample**: {eff_start} → {eff_end}.")

    # size
    parts.append(f"The modeling table contains **{rows} rows** and **{cols} columns** across **{dcount} districts** (monthly frequency).")

    # missing
    if miss_eff:
        show = ", ".join(miss_eff[:12])
        more = "" if len(miss_eff) <= 12 else f" … (+{len(miss_eff)-12} more)"
        parts.append(
            f"**Missing months in the effective table** were detected (e.g., {show}{more}). "
            f"These gaps can reflect original data availability issues and/or lag-induced missing values around breaks."
        )

    return "\n\n".join(parts)
