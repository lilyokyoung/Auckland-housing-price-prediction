# src/investment/investment_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import OUTPUT_DIR


# ============================================================
# Config
# ============================================================
@dataclass
class InvestmentNarrativeConfig:
    metrics_path: Path = OUTPUT_DIR / "investment" / "metrics_by_district.csv"
    top_k: int = 3

    # Tone controls: "neutral" | "cautious" | "confident"
    tone: str = "neutral"

    # Investor profiles
    include_profiles: bool = True

    # Optional: hard cap for max district lines in highlights
    max_highlights: int = 6


# ============================================================
# Helpers
# ============================================================
def _clean_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}%"


def _nzd(x: Optional[float], digits: int = 0) -> str:
    if x is None:
        return "—"
    try:
        return f"{x:,.{digits}f}"
    except Exception:
        return "—"


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Investment metrics file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Metrics missing columns: {missing}. Found: {list(df.columns)}")


def _rank_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create consistent rankings with fallbacks.
    Expected columns from compute_investment_metrics.py:
      district, anchor_month, horizon_months,
      current_price_base, forecast_price_base_h,
      expected_return_base, volatility_base,
      downside_max_drawdown_low, scenario_spread_high_minus_low,
      stability_score, expected_return_base_pct, downside_max_drawdown_low_pct
    """
    req = [
        "district",
        "anchor_month",
        "horizon_months",
        "current_price_base",
        "forecast_price_base_h",
        "expected_return_base",
        "expected_return_base_pct",
        "volatility_base",
        "downside_max_drawdown_low",
        "downside_max_drawdown_low_pct",
        "scenario_spread_high_minus_low",
        "stability_score",
    ]
    _ensure_cols(df, req)

    out = df.copy()

    # numeric coercion
    num_cols = [
        "current_price_base",
        "forecast_price_base_h",
        "expected_return_base",
        "expected_return_base_pct",
        "volatility_base",
        "downside_max_drawdown_low",
        "downside_max_drawdown_low_pct",
        "scenario_spread_high_minus_low",
        "stability_score",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # rankings
    out["rank_stability"] = out["stability_score"].rank(ascending=False, method="min")
    out["rank_return"] = out["expected_return_base"].rank(ascending=False, method="min")

    # risk proxies: volatility, downside magnitude, scenario spread
    out["downside_abs"] = out["downside_max_drawdown_low"].abs()
    out["rank_vol"] = out["volatility_base"].rank(ascending=True, method="min")  # lower vol is better
    out["rank_downside"] = out["downside_abs"].rank(ascending=True, method="min")  # smaller downside is better
    out["rank_spread"] = out["scenario_spread_high_minus_low"].rank(ascending=True, method="min")  # smaller is better

    # simple composite "defensive score" (lower is better)
    out["defensive_rank_sum"] = out["rank_vol"] + out["rank_downside"] + out["rank_spread"]

    return out


# ============================================================
# Meta builder
# ============================================================
def build_investment_meta(df_metrics: pd.DataFrame, cfg: InvestmentNarrativeConfig) -> Dict[str, Any]:
    df = _rank_df(df_metrics)

    if df.empty:
        return {"ok": False, "error": "metrics table is empty"}

    # global context
    anchor_month = _clean_str(df["anchor_month"].iloc[0], "—")
    horizon = int(_safe_float(df["horizon_months"].iloc[0]) or 0)

    # winners / leaders
    df_stable = df.sort_values(["stability_score", "expected_return_base"], ascending=[False, False])
    df_return = df.sort_values(["expected_return_base", "stability_score"], ascending=[False, False])
    df_defensive = df.sort_values(["defensive_rank_sum", "stability_score"], ascending=[True, False])

    best_stability = df_stable.iloc[0].to_dict()
    best_return = df_return.iloc[0].to_dict()
    best_defensive = df_defensive.iloc[0].to_dict()

    # highest risk / uncertainty (large spread or deep downside)
    df_risk = df.sort_values(
        ["downside_abs", "scenario_spread_high_minus_low", "volatility_base"],
        ascending=[False, False, False],
    )
    highest_risk = df_risk.iloc[0].to_dict()

    # top-k lists
    top_stable = df_stable.head(cfg.top_k).to_dict(orient="records")
    top_return = df_return.head(cfg.top_k).to_dict(orient="records")
    top_defensive = df_defensive.head(cfg.top_k).to_dict(orient="records")

    # highlights: pick a small set of districts to comment on (avoid repetition)
    highlight_df = (
        pd.concat(
            [
                df_stable.head(cfg.top_k),
                df_return.head(cfg.top_k),
                df_defensive.head(cfg.top_k),
                df_risk.head(1),
            ],
            axis=0,
            ignore_index=True,
        )
        .drop_duplicates(subset=["district"])
        .head(cfg.max_highlights)
    )
    highlights = highlight_df.to_dict(orient="records")

    # full metrics payload for UI (structured) built from the ranked dataframe `df`
    metrics_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        metrics_rows.append(
            {
                "district": _clean_str(r.get("district"), "—"),
                # use ratio values (not pct) so UI formatting is consistent
                "base_return": _safe_float(r.get("expected_return_base")),
                "volatility": _safe_float(r.get("volatility_base")),
                "low_scenario_drawdown": _safe_float(r.get("downside_max_drawdown_low")),
                "scenario_spread": _safe_float(r.get("scenario_spread_high_minus_low")),
                "stability_score": _safe_float(r.get("stability_score")),
            }
        )

    meta: Dict[str, Any] = {
        "ok": True,
        "anchor_month": anchor_month,
        "horizon_months": horizon,
        "n_districts": int(df["district"].nunique()),
        "best_stability": best_stability,
        "best_return": best_return,
        "best_defensive": best_defensive,
        "highest_risk": highest_risk,
        "top_stable": top_stable,
        "top_return": top_return,
        "top_defensive": top_defensive,
        "highlights": highlights,
        "metrics": metrics_rows,
    }

    return meta


# ============================================================
# Narrative builder
# ============================================================
def _tone_prefix(tone: str) -> str:
    t = (tone or "").strip().lower()
    if t == "cautious":
        return "This section provides decision-support insights rather than financial advice. "
    if t == "confident":
        return "Based on the model-implied scenario forecasts, the following insights summarise relative district profiles. "
    return "The following insights summarise relative district profiles using model-implied scenario forecasts. "


def _row_snippet(row: Dict[str, Any]) -> str:
    d = _clean_str(row.get("district"), "—")
    ret = _safe_float(row.get("expected_return_base_pct"))
    vol = _safe_float(row.get("volatility_base"))
    dd = _safe_float(row.get("downside_max_drawdown_low_pct"))
    spread = _safe_float(row.get("scenario_spread_high_minus_low"))
    stab = _safe_float(row.get("stability_score"))

    return (
        f"{d}: base return {_pct(ret)}, "
        f"volatility {('—' if vol is None else f'{vol:.4f}')}, "
        f"low-scenario drawdown {_pct(dd)}, "
        f"scenario spread {_nzd(spread)}, "
        f"stability score {('—' if stab is None else f'{stab:.2f}')}"
    )

# metrics_rows is built per-call inside build_investment_meta to avoid using undefined globals



def build_investment_narrative(meta: Dict[str, Any], cfg: InvestmentNarrativeConfig) -> Dict[str, Any]:
    if not meta.get("ok"):
        return {
            "tone": "error",
            "text": "Investment insight unavailable (metrics meta failed).",
            "profiles": {},
        }

    anchor = meta.get("anchor_month", "—")
    horizon = meta.get("horizon_months", 0)
    n = meta.get("n_districts", 0)

    # Key picks
    best_stability = meta.get("best_stability", {}) or {}
    best_return = meta.get("best_return", {}) or {}
    best_defensive = meta.get("best_defensive", {}) or {}
    highest_risk = meta.get("highest_risk", {}) or {}

    prefix = _tone_prefix(cfg.tone)

    # Main summary paragraph
    summary_lines: List[str] = []
    summary_lines.append(
        f"{prefix}Evaluation uses a {horizon}-month horizon starting from **{anchor}** across **{n} districts**."
    )
    summary_lines.append(
        f"Overall, **{_clean_str(best_stability.get('district'))}** shows the strongest *risk-adjusted stability* "
        f"(stability score {(_safe_float(best_stability.get('stability_score')) or 0):.2f}) "
        f"with base return {_pct(_safe_float(best_stability.get('expected_return_base_pct')))}."
    )
    summary_lines.append(
        f"For *growth orientation*, **{_clean_str(best_return.get('district'))}** has the highest base expected return "
        f"({_pct(_safe_float(best_return.get('expected_return_base_pct')))}), "
        f"while **{_clean_str(best_defensive.get('district'))}** appears most defensive across volatility/downside/spread indicators."
    )
    summary_lines.append(
        f"The most elevated downside/uncertainty signals are observed in **{_clean_str(highest_risk.get('district'))}**, "
        f"with low-scenario drawdown {_pct(_safe_float(highest_risk.get('downside_max_drawdown_low_pct')))} "
        f"and scenario spread {_nzd(_safe_float(highest_risk.get('scenario_spread_high_minus_low')))}."
    )

    # Highlights (bullet style)
    highlights: List[str] = []
    for row in (meta.get("highlights") or []):
        highlights.append(f"- {_row_snippet(row)}")

    # Investor profiles (optional)
    profiles: Dict[str, str] = {}
    if cfg.include_profiles:
        stable_d = _clean_str(best_stability.get("district"))
        def_d = _clean_str(best_defensive.get("district"))
        grow_d = _clean_str(best_return.get("district"))

        profiles["conservative"] = (
            f"**Conservative profile (stability-first):** Prioritise districts with strong stability scores and limited scenario dispersion. "
            f"On this horizon, **{stable_d}** and **{def_d}** appear relatively defensive, combining modest base returns with comparatively lower volatility/downside risk."
        )

        profiles["balanced"] = (
            f"**Balanced profile (risk–return trade-off):** Consider districts that deliver positive base returns while keeping scenario spread moderate. "
            f"A practical shortlist is drawn from the top stability set and the top return set; begin with **{stable_d}** and compare against **{grow_d}** for incremental upside versus risk."
        )

        profiles["growth"] = (
            f"**Growth-oriented profile (upside-first):** Focus on districts with the highest expected base appreciation, while monitoring larger scenario spreads and potential drawdowns. "
            f"**{grow_d}** leads on expected return, but investors should pair this with downside checks under the low scenario."
        )

    # Compliance-style note (safe wording)
    caveats = (
        "Note: These insights are **model-implied** and scenario-dependent. They are intended for analytical decision support "
        "and should be interpreted alongside market context, transaction costs, and individual risk tolerance."
    )

    # Assemble final text
    text = "\n\n".join(
    [
        "### Investment insight (district stability vs upside)",
        "\n".join(summary_lines),
        "### Investor-style interpretations" if cfg.include_profiles else "",
        "\n\n".join([profiles[k] for k in ["conservative", "balanced", "growth"] if k in profiles]) if cfg.include_profiles else "",
        f"**{caveats}**",
    ]
).strip()

    # Tone label
    tone = "info"
    if cfg.tone.strip().lower() == "cautious":
        tone = "warning"
    elif cfg.tone.strip().lower() == "confident":
        tone = "success"

    return {"tone": tone, "text": text, "profiles": profiles}


# ============================================================
# Public entrypoint (for API / Agent)
# ============================================================
def generate_investment_insight(
    metrics_path: Optional[Path] = None,
    top_k: int = 3,
    tone: str = "neutral",
    include_profiles: bool = True,
) -> Dict[str, Any]:
    """
    One-call helper:
      - loads metrics CSV
      - builds meta
      - returns {tone, text, profiles, meta}
    """
    cfg = InvestmentNarrativeConfig(
        metrics_path=metrics_path or (OUTPUT_DIR / "investment" / "metrics_by_district.csv"),
        top_k=top_k,
        tone=tone,
        include_profiles=include_profiles,
    )

    df = _read_metrics(cfg.metrics_path)
    meta = build_investment_meta(df, cfg)
    narr = build_investment_narrative(meta, cfg)

    return {
        "ok": True if meta.get("ok") else False,
        "tone": narr.get("tone", "info"),
        "text": narr.get("text", ""),
        "profiles": narr.get("profiles", {}),
        "meta": meta,
        "source": str(cfg.metrics_path),
    }


# ============================================================
# CLI quick test
# ============================================================
def main() -> None:
    out = generate_investment_insight(tone="neutral", top_k=3, include_profiles=True)
    print(out["text"])


if __name__ == "__main__":
    main()
