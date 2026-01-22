# src/agent/shap_query.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

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
    """Normalize month to 'YYYY-MM'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m")
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return pd.Timestamp(dt).strftime("%Y-%m")
    return s[:7] if len(s) >= 7 else s


# ============================================================
# config
# ============================================================
@dataclass(frozen=True)
class ShapQueryConfig:
    """
    Default SHAP long file:
      outputs/shap/shap_long_rf.parquet

    This file should be a long format with at least:
      - Month
      - District
      - Scenario
      - feature
      - shap_value (signed) OR abs_shap (magnitude)
    """
    project_root: Path
    shap_subdir: Path = Path("outputs") / "shap"
    preferred_stems: Tuple[str, ...] = ("shap_long_rf", "shap_long", "shap_long_tree")

    month_cols: Tuple[str, ...] = ("Month", "month", "Date", "date")
    scenario_cols: Tuple[str, ...] = ("Scenario", "scenario")
    district_cols: Tuple[str, ...] = ("District", "district")
    feature_cols: Tuple[str, ...] = ("feature", "Feature", "name", "var", "variable")
    shap_cols: Tuple[str, ...] = ("shap_value", "shap", "SHAP", "value")
    abs_cols: Tuple[str, ...] = ("abs_shap", "abs_shap_value", "mean_abs_shap", "magnitude")


class ShapQueryEngine:
    def __init__(self, cfg: ShapQueryConfig) -> None:
        self.cfg = cfg
        self.shap_dir = (cfg.project_root / cfg.shap_subdir).resolve()
        self.path = self._pick_shap_path()

        self.df = self._load(self.path)

        self.col_month = self._detect(self.df.columns, cfg.month_cols, required=False)
        self.col_scenario = self._detect(self.df.columns, cfg.scenario_cols, required=False)
        self.col_district = cast(str, self._detect(self.df.columns, cfg.district_cols, required=True))
        self.col_feature = cast(str, self._detect(self.df.columns, cfg.feature_cols, required=True))

        # shap magnitude: prefer abs_shap if present; else derive abs from shap_value
        self.col_abs = self._detect(self.df.columns, cfg.abs_cols, required=False)
        self.col_shap = self._detect(self.df.columns, cfg.shap_cols, required=False)

        self.df = self._normalize_keys(self.df)

    # -------------------------
    # internal
    # -------------------------
    def _pick_shap_path(self) -> Path:
        if not self.shap_dir.exists():
            raise FileNotFoundError(f"SHAP folder not found: {self.shap_dir}")

        candidates: List[Path] = []
        for stem in self.cfg.preferred_stems:
            p = self.shap_dir / f"{stem}.parquet"
            if p.exists():
                candidates.append(p)

        if not candidates:
            # fallback: newest parquet with "shap" in name
            all_files = list(self.shap_dir.glob("*.parquet"))
            all_files = [p for p in all_files if "shap" in p.stem.lower()]
            if not all_files:
                raise FileNotFoundError(f"No SHAP parquet found under: {self.shap_dir}")
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
        raise ValueError(f"Unsupported SHAP file type: {path}")

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

    # ============================================================
    # public
    # ============================================================
    def top_drivers(
        self,
        district: str,
        scenario: Optional[str],
        month: Optional[str],
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """
        Return top_k drivers for (district, scenario, month).
        Ranking priority:
        - if abs_shap exists: use abs_shap descending
        - else if shap exists: use abs(shap) descending and keep sign from shap
        """
        d = (district or "").strip()
        s = norm_scenario(scenario) if scenario else ""
        m = norm_month(month) if month else ""

        df_f = self.df
        df_f = df_f[df_f[self.col_district] == d]

        if self.col_scenario and s:
            df_f = df_f[df_f[self.col_scenario] == s]

        if self.col_month and m:
            df_f = df_f[df_f[self.col_month] == m]

        if df_f.empty:
            return {
                "ok": False,
                "message": f"No SHAP rows for district={d}, scenario={s or 'ALL'}, month={m or 'ALL'}.",
                "debug": {"source_path": str(self.path)},
            }

        work = df_f.copy()
        work[self.col_feature] = work[self.col_feature].astype(str).str.strip()

        # numeric columns
        if self.col_abs and self.col_abs in work.columns:
            work["_mag"] = pd.to_numeric(work[self.col_abs], errors="coerce")
            # signed shap optional
            if self.col_shap and self.col_shap in work.columns:
                work["_shap"] = pd.to_numeric(work[self.col_shap], errors="coerce")
            else:
                work["_shap"] = pd.NA
            mode = f"abs({self.col_abs})"
        elif self.col_shap and self.col_shap in work.columns:
            work["_shap"] = pd.to_numeric(work[self.col_shap], errors="coerce")
            work["_mag"] = work["_shap"].abs()
            mode = f"abs({self.col_shap})"
        else:
            return {
                "ok": False,
                "message": "No SHAP/abs_SHAP column found in SHAP file.",
                "debug": {"columns": list(self.df.columns), "source_path": str(self.path)},
            }

        work = work.dropna(subset=["_mag"])
        if work.empty:
            return {"ok": False, "message": "No valid numeric SHAP magnitudes after cleaning."}

        # in case you have multiple rows per feature (rare), aggregate mean magnitude
        agg = (
            work.groupby(self.col_feature, as_index=False)
            .agg(_mag=("_mag", "mean"), _shap=("_shap", "mean"))
            .sort_values("_mag", ascending=False)
            .head(int(max(1, top_k)))
        )

        drivers: List[Dict[str, Any]] = []
        for _, r in agg.iterrows():
            feat = str(r[self.col_feature]).strip()
            mag = float(r["_mag"]) if pd.notna(r["_mag"]) else None
            shap = float(r["_shap"]) if pd.notna(r["_shap"]) else None
            drivers.append(
                {
                    "feature": feat,
                    "magnitude": mag,
                    "shap_value": shap,
                    "direction": ("up" if (shap is not None and shap > 0) else ("down" if (shap is not None and shap < 0) else None)),
                }
            )

        return {
            "ok": True,
            "district": d,
            "scenario": s or None,
            "month": m or None,
            "mode": mode,
            "top_k": int(max(1, top_k)),
            "drivers": drivers,
            "source_path": str(self.path),
        }
