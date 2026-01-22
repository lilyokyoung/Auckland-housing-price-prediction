from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
@dataclass
class SpatialCfg:
    # input
    region_path: Path
    districts_path: Path

    # columns
    month_col: str = "Month"
    region_price_col: str = "Median_Price"
    district_col: str = "District"
    district_price_col: str = "Median_Price"

    # analysis window (inclusive, month-start)
    start: Optional[str] = "2018-08-01"
    end: Optional[str] = "2025-06-01"

    # output assets
    out_dir: Path = Path("outputs/spatial_context")
    fig_name: str = "district_vs_region.png"
    summary_name: str = "summary.json"

    # plot options (Statista-like)
    normalize_index: bool = True
    label_every_n_months: int = 3

    show_legend: bool = True
    legend_right_pad: float = 0.82      # 给右侧 legend 留空间（0~1）

    show_end_labels: bool = False       # ✅ 关键：关掉右侧末端文字
    end_label_top_n: int = 0            # 双保险
    right_pad_months: int = 4

    title: str = "Auckland Region vs 7 Districts — Historical Median Price Comparison"
    subtitle: Optional[str] = None


# =========================================================
# Helpers
# =========================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def parse_month_to_mstart(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing month column: {col}")
    out = df.copy()
    dt = pd.to_datetime(out[col], errors="raise")
    out[col] = dt.dt.to_period("M").dt.to_timestamp(how="start")
    return out


def _to_mstart(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).to_period("M").to_timestamp(how="start")


def filter_window(df: pd.DataFrame, month_col: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if start:
        s0 = _to_mstart(start)
        out = out[out[month_col] >= s0]
    if end:
        e0 = _to_mstart(end)
        out = out[out[month_col] <= e0]
    return out


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def to_index_100(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    first = s.dropna().iloc[0]
    return (s / first) * 100.0


# =========================================================
# Build tables
# =========================================================
def build_region_series(cfg: SpatialCfg) -> pd.DataFrame:
    df = load_csv(cfg.region_path)
    df = parse_month_to_mstart(df, cfg.month_col)

    if cfg.region_price_col not in df.columns:
        raise KeyError(f"Region file missing price column: {cfg.region_price_col}")

    df[cfg.region_price_col] = safe_numeric(df[cfg.region_price_col])
    df = df.dropna(subset=[cfg.region_price_col])

    df = filter_window(df, cfg.month_col, cfg.start, cfg.end)
    return df[[cfg.month_col, cfg.region_price_col]].sort_values(cfg.month_col).reset_index(drop=True)


def build_district_panel(cfg: SpatialCfg) -> pd.DataFrame:
    df = load_csv(cfg.districts_path)
    df = parse_month_to_mstart(df, cfg.month_col)

    for c in [cfg.district_col, cfg.district_price_col]:
        if c not in df.columns:
            raise KeyError(f"District file missing column: {c}")

    df[cfg.district_price_col] = safe_numeric(df[cfg.district_price_col])
    df = df.dropna(subset=[cfg.district_col, cfg.district_price_col])

    df = filter_window(df, cfg.month_col, cfg.start, cfg.end)
    df = df[[cfg.month_col, cfg.district_col, cfg.district_price_col]]
    return df.sort_values([cfg.district_col, cfg.month_col]).reset_index(drop=True)


def align_common_months(region: pd.DataFrame, district: pd.DataFrame, cfg: SpatialCfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(region[cfg.month_col]).intersection(set(district[cfg.month_col])))
    region2 = region[region[cfg.month_col].isin(common)].copy()
    district2 = district[district[cfg.month_col].isin(common)].copy()
    return region2, district2


# =========================================================
# Summary metrics (for JSON)
# =========================================================
def compute_spatial_summary(region: pd.DataFrame, districts: pd.DataFrame, cfg: SpatialCfg) -> Dict:
    mcol = cfg.month_col
    rcol = cfg.region_price_col
    dcol = cfg.district_col
    pcol = cfg.district_price_col

    region_s = region.set_index(mcol)[rcol].sort_index()
    region_lr = np.log(region_s).diff()  # type: ignore

    out_rows = []
    for d, g in districts.groupby(dcol):
        s = g.set_index(mcol)[pcol].sort_index()
        idx = region_s.index.intersection(s.index)
        if len(idx) < 6:
            continue

        rr = region_s.loc[idx]
        dd = s.loc[idx]

        ratio_mean = float((dd / rr).mean())
        ratio_median = float((dd / rr).median())

        lr_d = np.log(dd).diff()  # type: ignore
        vol = float(lr_d.std(skipna=True))
        corr = float(lr_d.corr(region_lr.loc[idx], min_periods=6))

        out_rows.append(
            {
                "district": d,
                "avg_ratio_to_region": ratio_mean,
                "median_ratio_to_region": ratio_median,
                "volatility_log_return_std": vol,
                "corr_with_region_log_return": corr,
            }
        )

    dfm = pd.DataFrame(out_rows)
    if dfm.empty:
        top_premium = []
        top_vol = []
    else:
        top_premium = (
            dfm.sort_values("avg_ratio_to_region", ascending=False)
            .head(3)[["district", "avg_ratio_to_region"]]
            .to_dict("records")
        )
        top_vol = (
            dfm.sort_values("volatility_log_return_std", ascending=False)
            .head(3)[["district", "volatility_log_return_std"]]
            .to_dict("records")
        )

    time_span = f"{region[mcol].min().strftime('%Y-%m')} to {region[mcol].max().strftime('%Y-%m')}"
    return {
        "time_span": time_span,
        "notes": {
            "normalize_index": cfg.normalize_index,
            "interpretation": "Spatial context (region vs districts). Descriptive only; not used for training.",
        },
        "top_3_premium_vs_region": top_premium,
        "top_3_volatility": top_vol,
        "district_metrics": dfm.to_dict("records") if not dfm.empty else [],
    }


# =========================================================
# Plot (Statista-like)
# =========================================================
def plot_district_vs_region(region: pd.DataFrame, districts: pd.DataFrame, cfg: SpatialCfg) -> plt.Figure:  # type: ignore
    import matplotlib.dates as mdates

    mcol = cfg.month_col
    rcol = cfg.region_price_col
    dcol = cfg.district_col
    pcol = cfg.district_price_col

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 7))

    # --- region
    reg_series = region.set_index(mcol)[rcol].sort_index()
    if cfg.normalize_index:
        reg_y = to_index_100(reg_series)
        y_label = "Price Index (2018-08 = 100)"
        reg_label = "AucklandRegion"
    else:
        reg_y = reg_series
        y_label = "Median Price (NZD)"
        reg_label = "AucklandRegion"

    ax.plot(reg_y.index, reg_y.values, linewidth=3.2, label=reg_label, zorder=3)  # type: ignore

    # --- districts
    district_last_values = []
    for d, g in districts.groupby(dcol, sort=True):
        s = g.set_index(mcol)[pcol].sort_index()
        y = to_index_100(s) if cfg.normalize_index else s
        ax.plot(y.index, y.values, linewidth=1.4, alpha=0.95, label=str(d), zorder=2)  # type: ignore

        y_non_na = y.dropna()
        if not y_non_na.empty:
            district_last_values.append((str(d), float(y_non_na.iloc[-1])))

    # --- title centered
    ax.set_title(cfg.title, loc="center", pad=18)
    ax.set_xlabel("Month")
    ax.set_ylabel(y_label)

    # --- ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=cfg.label_every_n_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- grid
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.grid(axis="x", alpha=0.0)

    # --- spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- xlim with padding (still useful even without end labels)
    xmin = reg_y.index.min()
    xmax = reg_y.index.max()
    if pd.notna(xmin) and pd.notna(xmax):
        ax.set_xlim(xmin, xmax + pd.offsets.MonthBegin(cfg.right_pad_months))

    # --- end labels OFF by default (avoid your red-box text)
    if cfg.show_end_labels and cfg.end_label_top_n > 0:
        district_last_values.sort(key=lambda x: x[1], reverse=True)
        to_label = district_last_values[: int(cfg.end_label_top_n)]
        x_text = xmax + pd.offsets.MonthBegin(max(1, cfg.right_pad_months - 1))
        for name, yv in to_label:
            ax.text(x_text, yv, name, fontsize=10, va="center", ha="left", color="#333333")

    # --- legend outside right (white box)
    if cfg.show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=10,
            ncol=1,
        )

        # reserve space on the right for legend
        plt.subplots_adjust(right=cfg.legend_right_pad)

    fig.tight_layout()
    return fig


# =========================================================
# Save assets
# =========================================================
def save_assets(fig: plt.Figure, summary: Dict, cfg: SpatialCfg) -> None:  # type: ignore
    ensure_dir(cfg.out_dir)

    fig_path = cfg.out_dir / cfg.fig_name
    json_path = cfg.out_dir / cfg.summary_name

    fig.savefig(fig_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved figure : {fig_path.resolve()}")
    print(f"[OK] saved summary: {json_path.resolve()}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    cfg = SpatialCfg(
        region_path=PROJECT_ROOT / "data" / "processed" / "housing_price" / "AucklandRegion_Price.csv",
        districts_path=PROJECT_ROOT / "data" / "processed" / "housing_price" / "HousingPrice_7districts.csv",
        out_dir=PROJECT_ROOT / "outputs" / "spatial_context",
        start="2018-08-01",
        end="2025-06-01",
        normalize_index=True,
        label_every_n_months=3,

        show_legend=True,          
        show_end_labels=False,     
        end_label_top_n=0,         

        right_pad_months=4,
        legend_right_pad=0.82,
    )

    region = build_region_series(cfg)
    districts = build_district_panel(cfg)
    region, districts = align_common_months(region, districts, cfg)

    fig = plot_district_vs_region(region, districts, cfg)
    summary = compute_spatial_summary(region, districts, cfg)

    save_assets(fig, summary, cfg)


if __name__ == "__main__":
    main()
