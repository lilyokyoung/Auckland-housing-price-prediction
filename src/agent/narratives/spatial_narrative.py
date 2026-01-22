# src/agent/narratives/spatial_narrative.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


# ============================================================
# Config
# ============================================================
@dataclass
class SpatialNarrativeConfig:
    top_n: int = 3
    max_bullets: int = 6

    # New switches
    length: str = "long"      # "short" | "long"
    tone: str = "academic"    # "academic" | "plain"


# ============================================================
# Helpers
# ============================================================
def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{(x - 1.0) * 100:.1f}%"


def _format_top_premium(top_items: List[Dict[str, Any]], top_n: int) -> str:
    items = top_items[:top_n]
    parts: List[str] = []
    for it in items:
        district = str(it.get("district", "Unknown"))
        ratio = _safe_float(it.get("avg_ratio_to_region"))
        parts.append(f"{district} ({_pct(ratio)} premium vs region)")
    return ", ".join(parts) if parts else ""


def _extract_span(summary: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> str:
    if data:
        span = data.get("time_span")
        if span:
            return str(span)
    span = summary.get("time_span") or summary.get("time span")
    return str(span) if span else ""


def _extract_findings(summary: Dict[str, Any]) -> List[str]:
    findings = (
        summary.get("key_findings")
        or summary.get("key findings")
        or summary.get("findings")
        or summary.get("highlights")
        or []
    )
    out: List[str] = []
    for x in _as_list(findings):
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _extract_interpretation(summary: Dict[str, Any]) -> str:
    notes = summary.get("notes") or {}
    interp = notes.get("interpretation") or summary.get("interpretation") or ""
    return str(interp).strip()


# ============================================================
# Short narrative
# ============================================================
def _build_short_narrative(
    summary: Dict[str, Any],
    figure_path: Optional[str],
    cfg: SpatialNarrativeConfig,
    payload: Dict[str, Any],
) -> str:
    span = _extract_span(summary, payload)

    top_items = summary.get("top_3_premium_vs_region") or summary.get("top_premium_vs_region") or []
    top_items = _as_list(top_items)
    top_text = _format_top_premium(top_items, cfg.top_n)

    findings = _extract_findings(summary)
    findings = findings[: cfg.max_bullets]

    # Tone-adjusted phrasing (still all-English)
    if cfg.tone == "plain":
        lead = "District median prices differ from the Auckland regional median, indicating spatial differences across districts."
        training_note = "This comparison is for context only; it is not used for model training."
    else:
        lead = "District-level median prices deviate from the Auckland regional median benchmark, indicating spatial heterogeneity across districts."
        training_note = "This comparison provides background context only and is not used for model training."

    lines: List[str] = []
    if span:
        lines.append(f"Spatial context ({span}).")
    else:
        lines.append("Spatial context.")

    lines.append(lead)
    lines.append(training_note)

    if top_text:
        lines.append(f"Largest average premiums vs region: {top_text}.")
    else:
        lines.append("Premium/discount ranking vs region is not available in the summary output.")

    # In short mode: at most 2 bullets, optional
    if findings:
        lines.append("Key observations:")
        for x in findings[:2]:
            lines.append(f"- {x}")

    if figure_path:
        lines.append(f"Figure asset: {figure_path}")

    return "\n".join(lines).strip()


# ============================================================
# Main builder
# ============================================================
def build_spatial_narrative(
    spatial_context_payload: Dict[str, Any],
    cfg: Optional[SpatialNarrativeConfig] = None,
) -> str:
    """
    Build an English narrative from the FastAPI /spatial_context response.

    Payload shape (flexible):
    - summary: dict
      - time_span: str
      - key_findings: list[str] (optional)
      - notes.interpretation: str (optional)
      - top_3_premium_vs_region: list[{"district":..., "avg_ratio_to_region":...}] (optional)
    - figure_path: str (optional)
    """
    cfg = cfg or SpatialNarrativeConfig()

    summary = spatial_context_payload.get("summary") or {}
    figure_path = (
        spatial_context_payload.get("figure_path")
        or spatial_context_payload.get("figure")
        or spatial_context_payload.get("fig_path")
        or None
    )

    # ---- Short mode ----
    if str(cfg.length).lower() == "short":
        return _build_short_narrative(summary, figure_path, cfg, spatial_context_payload)

    # ---- Long mode (your current output, kept) ----
    span = _extract_span(summary, spatial_context_payload)
    findings = _extract_findings(summary)
    interpretation = _extract_interpretation(summary)

    top_items = summary.get("top_3_premium_vs_region") or summary.get("top_premium_vs_region") or []
    top_items = _as_list(top_items)
    top_premium_text = _format_top_premium(top_items, cfg.top_n)

    lines: List[str] = []

    if span:
        lines.append(
            f"Spatial context analysis ({span}) compares district-level median prices against the Auckland regional median benchmark."
        )
    else:
        lines.append(
            "Spatial context analysis compares district-level median prices against the Auckland regional median benchmark."
        )

    lines.append(
        "This section is used to motivate spatial heterogeneity and provide background context; it is not used for model training."
    )

    if top_items and top_premium_text:
        lines.append(
            f"On average, the largest persistent premiums relative to the regional benchmark are observed in: {top_premium_text}."
        )
    else:
        lines.append(
            "District-level price levels diverge from the regional benchmark, indicating meaningful cross-district heterogeneity."
        )

    if findings:
        lines.append("")
        lines.append("Key empirical observations:")
        for x in findings[: cfg.max_bullets]:
            lines.append(f"- {x}")

    if interpretation:
        lines.append("")
        lines.append(f"Interpretation: {interpretation}")

    if figure_path:
        lines.append("")
        lines.append(f"Figure asset: {figure_path}")

    return "\n".join(lines).strip()


# ============================================================
# File helpers
# ============================================================
def load_summary_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_spatial_narrative_from_summary_file(
    summary_json_path: str,
    figure_path: Optional[str] = None,
    cfg: Optional[SpatialNarrativeConfig] = None,
) -> str:
    summary = load_summary_json(summary_json_path)
    payload: Dict[str, Any] = {"summary": summary}
    if figure_path:
        payload["figure_path"] = figure_path
    return build_spatial_narrative(payload, cfg=cfg)


# ============================================================
# Optional CLI test
# ============================================================
def main() -> None:
    # Adjust these paths to your local outputs if you want CLI testing
    # Example:
    # summary_path = "outputs/spatial_context/summary.json"
    # fig_path = "outputs/spatial_context/district_vs_region.png"
    summary_path = "outputs/spatial_context/summary.json"
    fig_path = "outputs/spatial_context/district_vs_region.png"

    cfg = SpatialNarrativeConfig(length="short", tone="academic")
    text = build_spatial_narrative_from_summary_file(summary_path, figure_path=fig_path, cfg=cfg)
    print(text)


if __name__ == "__main__":
    main()

