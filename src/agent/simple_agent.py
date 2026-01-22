# src/agent/simple_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Config
# ============================================================

@dataclass
class SimpleAgentConfig:
    # trend thresholds (pct change; using level pct change)
    strong_change_pct: float = 0.08     # ±8% strong
    moderate_change_pct: float = 0.03   # ±3% moderate

    # driver thresholds (share of total abs contribution)
    high_share: float = 0.25
    medium_share: float = 0.10

    # how many drivers to present
    top_n: int = 6

    # narrative formatting
    max_bullets: int = 6


# ============================================================
# Agent
# ============================================================

class SimpleForecastAgent:
    """
    Deterministic agent that turns:
      - predicted movement (pct_change)
      - SHAP month-specific drivers (up/down lists)
    into a structured "analyst report" + a natural-language narrative string.
    """

    def __init__(self, config: Optional[SimpleAgentConfig] = None):
        self.cfg = config or SimpleAgentConfig()

    # -------------------------
    # Public entry
    # -------------------------
    def run(
        self,
        *,
        district: str,
        scenario: str,
        month: str,
        current_price: Optional[float],
        future_price: Optional[float],
        pct_change: Optional[float],
        shap_up: Optional[List[Dict[str, Any]]] = None,
        shap_down: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        trend = self._interpret_trend(pct_change)

        compiled = self._compile_driver_cards(shap_up or [], shap_down or [])
        cards_all = compiled["cards_all"]
        cards_up = compiled["cards_up"]
        cards_down = compiled["cards_down"]

        risks = self._infer_risks(trend=trend, drivers=cards_all, scenario=scenario)

        headline = self._headline(district=district, scenario=scenario, month=month, trend=trend)

        bullets = self._summary_bullets(
            trend=trend,
            current_price=current_price,
            future_price=future_price,
            drivers_all=cards_all,
            drivers_up=cards_up,
            drivers_down=cards_down,
        )

        story = self._economic_story(trend=trend, drivers_all=cards_all, scenario=scenario)

        narrative = self._render_narrative(
            headline=headline,
            bullets=bullets,
            drivers_up=cards_up[: self.cfg.top_n],
            drivers_down=cards_down[: self.cfg.top_n],
            risks=risks,
            story=story,
        )

        return {
            "meta": {
                "district": district,
                "scenario": scenario,
                "month": month,
            },
            "headline": headline,
            "summary_bullets": bullets[: self.cfg.max_bullets],
            "top_drivers": {
                "up": cards_up[: self.cfg.top_n],
                "down": cards_down[: self.cfg.top_n],
                "all": cards_all[: self.cfg.top_n],
            },
            "risk_notes": risks,
            "economic_story": story,
            # ⭐关键：给 UI 直接显示的一段文字
            "narrative": narrative,
        }

    # -------------------------
    # Trend
    # -------------------------
    def _interpret_trend(self, pct_change: Optional[float]) -> Dict[str, Any]:
        if pct_change is None:
            return {"direction": "unknown", "strength": "unknown", "pct_change": None}

        p = float(pct_change)

        if p >= self.cfg.strong_change_pct:
            return {"direction": "increase", "strength": "strong", "pct_change": p}
        if p >= self.cfg.moderate_change_pct:
            return {"direction": "increase", "strength": "moderate", "pct_change": p}
        if p <= -self.cfg.strong_change_pct:
            return {"direction": "decrease", "strength": "strong", "pct_change": p}
        if p <= -self.cfg.moderate_change_pct:
            return {"direction": "decrease", "strength": "moderate", "pct_change": p}

        return {"direction": "stable", "strength": "weak", "pct_change": p}

    def _headline(self, *, district: str, scenario: str, month: str, trend: Dict[str, Any]) -> str:
        d = trend.get("direction", "unknown")
        s = trend.get("strength", "unknown")

        if d == "stable":
            return f"{district} ({scenario}, {month}): Prices are expected to remain broadly stable."
        if d in {"increase", "decrease"}:
            arrow = "▲" if d == "increase" else "▼"
            return f"{district} ({scenario}, {month}): {arrow} {s.capitalize()} {d} expected."
        return f"{district} ({scenario}, {month}): Forecast direction is uncertain."

    # -------------------------
    # Drivers
    # -------------------------
    def _compile_driver_cards(
        self,
        up: List[Dict[str, Any]],
        down: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Input format (typical):
          up:   [{feature, mean_shap, mean_abs_shap}, ...]
          down: [{feature, mean_shap, mean_abs_shap}, ...]

        Output:
          {
            "cards_all": [...sorted by abs_shap desc...],
            "cards_up":  [...],
            "cards_down":[...],
          }
        """
        rows: List[Tuple[str, str, float]] = []

        for x in up:
            feat = str(x.get("feature", "")).strip()
            abs_sh = self._safe_float(x.get("mean_abs_shap", 0.0))
            if feat:
                rows.append((feat, "up", abs_sh))

        for x in down:
            feat = str(x.get("feature", "")).strip()
            abs_sh = self._safe_float(x.get("mean_abs_shap", 0.0))
            if feat:
                rows.append((feat, "down", abs_sh))

        total = sum(r[2] for r in rows) if rows else 0.0

        def _importance(share: float) -> str:
            if share >= self.cfg.high_share:
                return "high"
            if share >= self.cfg.medium_share:
                return "medium"
            return "low"

        # sort by magnitude
        rows_sorted = sorted(rows, key=lambda t: t[2], reverse=True)

        cards_all: List[Dict[str, Any]] = []
        for idx, (feat, direction, abs_sh) in enumerate(rows_sorted, start=1):
            share = (abs_sh / total) if total > 0 else 0.0
            imp = _importance(share)

            label = f"{feat} ({imp}, {direction})"
            cards_all.append(
                {
                    "rank": idx,
                    "feature": feat,
                    "direction": direction,                 # up/down
                    "importance": imp,                      # high/medium/low
                    "contribution_share": round(share, 3),
                    "abs_shap": round(abs_sh, 6),
                    "label": label,
                }
            )

        cards_up = [c for c in cards_all if c["direction"] == "up"]
        cards_down = [c for c in cards_all if c["direction"] == "down"]

        return {"cards_all": cards_all, "cards_up": cards_up, "cards_down": cards_down}

    def _safe_float(self, x: Any) -> float:
        try:
            if x is None:
                return 0.0
            return float(x)
        except Exception:
            return 0.0

    # -------------------------
    # Summary bullets
    # -------------------------
    def _summary_bullets(
        self,
        *,
        trend: Dict[str, Any],
        current_price: Optional[float],
        future_price: Optional[float],
        drivers_all: List[Dict[str, Any]],
        drivers_up: List[Dict[str, Any]],
        drivers_down: List[Dict[str, Any]],
    ) -> List[str]:
        bullets: List[str] = []

        pct = trend.get("pct_change")
        if pct is not None:
            bullets.append(
                f"Projected change: {pct*100:.2f}% (direction: {trend.get('direction')}, strength: {trend.get('strength')})."
            )
        else:
            bullets.append("Projected change: unavailable (missing prediction inputs).")

        if current_price is not None and future_price is not None:
            bullets.append(f"Price level: {current_price:,.0f} → {future_price:,.0f}.")

        if drivers_all:
            top = drivers_all[:3]
            names = ", ".join([d["feature"] for d in top])
            bullets.append(f"Top drivers (by SHAP magnitude): {names}.")
        else:
            bullets.append("Top drivers: not available for this selection (no SHAP slice).")

        # quick balance signal
        if drivers_up and drivers_down:
            bullets.append("Drivers show both upward and downward forces in this month (net effect depends on balance).")

        return bullets

    # -------------------------
    # Risks
    # -------------------------
    def _infer_risks(
        self,
        *,
        trend: Dict[str, Any],
        drivers: List[Dict[str, Any]],
        scenario: str,
    ) -> List[str]:
        risks: List[str] = []
        risks.append("Model-based forecast: sensitive to feature assumptions and historical regime stability.")

        sc = (scenario or "").lower().strip()
        if sc in {"low", "high"}:
            risks.append(
                f"Scenario '{sc}' is conditional: interpretation depends on how scenario variables were constructed."
            )

        if trend.get("direction") == "stable" and any(d["importance"] in {"high", "medium"} for d in drivers[:6]):
            risks.append("Apparent stability may reflect offsetting forces (positives and negatives cancelling out).")

        if trend.get("strength") == "strong" and not any(d["importance"] == "high" for d in drivers[:6]):
            risks.append("Large predicted move with no dominant SHAP driver suggests diffuse influence; higher uncertainty.")

        if not drivers:
            risks.append("Attribution missing: treat explanation as incomplete until SHAP drivers are available.")

        return risks

    # -------------------------
    # Story (economic narrative)
    # -------------------------
    def _economic_story(
        self,
        *,
        trend: Dict[str, Any],
        drivers_all: List[Dict[str, Any]],
        scenario: str,
    ) -> str:
        d = trend.get("direction")
        s = trend.get("strength")

        if not drivers_all:
            return (
                f"Under the {scenario} scenario, the model suggests {s} {d} dynamics, "
                "but driver attribution is unavailable for this selection. "
                "Interpretation should rely on broader macro context and the forecast assumptions."
            )

        top_up = [x["feature"] for x in drivers_all if x["direction"] == "up"][:2]
        top_down = [x["feature"] for x in drivers_all if x["direction"] == "down"][:2]

        parts: List[str] = []
        if d == "stable":
            parts.append(
                f"Under the {scenario} scenario, prices are projected to remain broadly stable. "
                "This commonly occurs when positive and negative pressures balance within the month."
            )
        elif d in {"increase", "decrease"}:
            parts.append(
                f"Under the {scenario} scenario, the forecast indicates a {s} {d} in prices. "
                "This reflects the model’s net assessment of demand, supply adjustment, and financing conditions."
            )
        else:
            parts.append(
                f"Under the {scenario} scenario, the forecast direction is uncertain due to missing or weak signals."
            )

        if top_up:
            parts.append(f"Key upward contributors include {', '.join(top_up)}.")
        if top_down:
            parts.append(f"Key downward contributors include {', '.join(top_down)}.")

        parts.append(
            "SHAP attributions are conditional on the trained model and inputs; interpret them as directional signals, not causal effects."
        )
        return " ".join(parts).strip()

    # -------------------------
    # Final narrative string (for UI)
    # -------------------------
    def _render_narrative(
        self,
        *,
        headline: str,
        bullets: List[str],
        drivers_up: List[Dict[str, Any]],
        drivers_down: List[Dict[str, Any]],
        risks: List[str],
        story: str,
    ) -> str:
        lines: List[str] = []

        lines.append(headline)
        lines.append("")

        # bullets
        if bullets:
            lines.append("Key takeaways:")
            for b in bullets[: self.cfg.max_bullets]:
                lines.append(f"- {b}")
            lines.append("")

        # drivers
        def _fmt_driver(d: Dict[str, Any]) -> str:
            feat = d.get("feature", "—")
            imp = d.get("importance", "low")
            share = d.get("contribution_share", 0.0)
            return f"{feat} ({imp}, share={share})"

        if drivers_up or drivers_down:
            lines.append("Drivers (month-specific SHAP):")
            if drivers_up:
                lines.append("Upward:")
                for d in drivers_up[: self.cfg.top_n]:
                    lines.append(f"- {_fmt_driver(d)}")
            else:
                lines.append("Upward: —")

            if drivers_down:
                lines.append("Downward:")
                for d in drivers_down[: self.cfg.top_n]:
                    lines.append(f"- {_fmt_driver(d)}")
            else:
                lines.append("Downward: —")

            lines.append("")

        # risks
        if risks:
            lines.append("Risk notes:")
            for r in risks[:6]:
                lines.append(f"- {r}")
            lines.append("")

        # story
        if story:
            lines.append("Interpretation:")
            lines.append(story)

        return "\n".join(lines).strip()
