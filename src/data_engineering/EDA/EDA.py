# src/data_engineering/EDA/EDA.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
@dataclass
class EDAConfig:
    train_path: Path
    out_dir: Path

    month_col: str = "Month"
    district_col: str = "District"
    price_col: str = "Median_Price"

    numeric_cols: Optional[list[str]] = None
    skew_cols: Optional[list[str]] = None

    plot_box_by_district: bool = True
    plot_mean_median_std_by_district: bool = True
    plot_multiline_price_by_district: bool = True
    plot_multiline_sales_by_district: bool = True
    plot_multiline_consents_by_district: bool = True
    plot_ocr_time_series: bool = True
    plot_real_mortgage_rate_time_series: bool = True

    sales_col: str = "sales_count"
    consents_col: str = "Consents"
    ocr_col: str = "OCR"
    real_mortgage_rate_col: str = "RealMortgageRate"

    # Global title styling
    title_fontsize: int = 16
    suptitle_fontsize: int = 18
    title_fontweight: str = "bold"


# =========================
# I/O + parsing
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_train_dataset(path: Path, month_col: str = "Month") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"train file not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {suf}. Use .csv or .parquet")

    if month_col not in df.columns:
        raise KeyError(f"Column '{month_col}' not found in train dataset.")

    dt = pd.to_datetime(df[month_col], errors="raise")
    df[month_col] = dt.dt.to_period("M").dt.to_timestamp(how="start")

    df = df.sort_values(month_col, kind="mergesort").reset_index(drop=True)
    return df


def pick_numeric_cols(df: pd.DataFrame, numeric_cols: Optional[list[str]], month_col: str) -> list[str]:
    if numeric_cols is not None:
        missing = [c for c in numeric_cols if c not in df.columns]
        if missing:
            raise KeyError(f"numeric_cols contains missing columns: {missing}")
        return numeric_cols

    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != month_col]
    return cols


def pick_skew_cols(
    df: pd.DataFrame,
    skew_cols: Optional[list[str]],
    exclude_patterns: tuple[str, ...] = ("_lag", "lag_", "L1", "L3", "L6", "L9", "L12"),
) -> list[str]:
    if skew_cols is not None:
        missing = [c for c in skew_cols if c not in df.columns]
        if missing:
            raise KeyError(f"skew_cols contains missing columns: {missing}")
        cols = list(skew_cols)
    else:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def is_lag_col(name: str) -> bool:
        s = name.lower()
        return any(p.lower() in s for p in exclude_patterns)

    cols = [c for c in cols if not is_lag_col(c)]
    return cols


# =========================
# EDA tables
# =========================
def dataset_overview(df: pd.DataFrame, month_col: str) -> dict[str, object]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "month_min": df[month_col].min(),
        "month_max": df[month_col].max(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = (miss_cnt / len(df)).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"missing_count": miss_cnt, "missing_pct": miss_pct}).sort_values(
        "missing_count", ascending=False
    )
    return out


def describe_numeric(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    return df[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T


def skewness_table(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    sk = df[num_cols].skew(numeric_only=True).sort_values(ascending=False)
    return sk.to_frame("skewness")


def correlation_matrix(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    return df[num_cols].corr()


# =========================
# Plot styling helpers (Infographic / Statista-like)
# =========================
def apply_infographic_style(ax: plt.Axes, grid_axis: str = "y") -> None:  # type: ignore
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.25)  # type: ignore
    ax.tick_params(axis="both", labelsize=11)


def add_title_subtitle(
    fig: plt.Figure,  # type: ignore
    title: str,
    subtitle: Optional[str],
    title_fontsize: int,
    title_fontweight: str,
    *,
    title_y: float = 0.995,
    subtitle_gap: Optional[float] = None,
) -> float:
    """
    Add figure-level title & subtitle without overlapping.
    Returns reserved_top for tight_layout(rect=...) to keep axes away from titles.
    """
    fig.suptitle(title, fontsize=title_fontsize, fontweight=title_fontweight, y=title_y)

    if subtitle_gap is None:
        subtitle_gap = 0.04 + 0.003 * float(title_fontsize)  # 18 -> ~0.094

    if subtitle:
        subtitle_y = title_y - subtitle_gap
        fig.text(0.5, subtitle_y, subtitle, fontsize=12, alpha=0.85, ha="center")
        reserved_top = subtitle_y - 0.02
    else:
        reserved_top = title_y - 0.02

    reserved_top = min(max(reserved_top, 0.70), 0.92)
    return reserved_top


# =========================
# Plots (matplotlib only)
# =========================
def plot_corr_heatmap(
    corr: pd.DataFrame,
    out_path: Path,
    title: str,
    subtitle: Optional[str],
    title_fontsize: int,
    title_fontweight: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    apply_infographic_style(ax, grid_axis="y")
    fig.colorbar(im, ax=ax, shrink=0.8)

    reserved_top = add_title_subtitle(
        fig,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
    )
    fig.tight_layout(rect=(0, 0, 1, reserved_top))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_boxplot_by_district(
    df: pd.DataFrame,
    district_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    subtitle: Optional[str],
    ylabel: str,
    title_fontsize: int,
    title_fontweight: str,
) -> None:
    for col in (district_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")

    tmp = df[[district_col, value_col]].dropna()
    districts = sorted(tmp[district_col].astype(str).unique().tolist())
    data = [tmp.loc[tmp[district_col].astype(str) == d, value_col].to_numpy(dtype=float) for d in districts]  # type: ignore

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.boxplot(data, labels=districts, showfliers=True, patch_artist=True)  # type: ignore

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)

    apply_infographic_style(ax, grid_axis="y")

    reserved_top = add_title_subtitle(
        fig,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
    )
    fig.tight_layout(rect=(0, 0, 1, reserved_top))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def district_level_stats(df: pd.DataFrame, district_col: str, value_col: str) -> pd.DataFrame:
    tmp = df[[district_col, value_col]].dropna().copy()
    tmp[district_col] = tmp[district_col].astype(str)

    def iqr(x: pd.Series) -> float:
        return float(x.quantile(0.75) - x.quantile(0.25))

    g = tmp.groupby(district_col)[value_col]
    stats = pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "std": g.std(ddof=1),
            "min": g.min(),
            "max": g.max(),
            "q25": g.quantile(0.25),
            "q75": g.quantile(0.75),
            "iqr": g.apply(iqr),
        }
    ).reset_index()

    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    return stats


def plot_mean_median_std_errorbars(
    stats: pd.DataFrame,
    district_col: str,
    out_path: Path,
    title: str,
    subtitle: Optional[str],
    ylabel: str,
    title_fontsize: int,
    title_fontweight: str,
) -> None:
    stats = stats.copy()
    stats[district_col] = stats[district_col].astype(str)
    stats = stats.sort_values("median", ascending=False).reset_index(drop=True)

    x = np.arange(len(stats))
    y_mean = stats["mean"].to_numpy(dtype=float)
    yerr = stats["std"].fillna(0).to_numpy(dtype=float)
    y_median = stats["median"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.errorbar(x, y_mean, yerr=yerr, fmt="o", capsize=6, elinewidth=2)
    ax.scatter(x, y_median, marker="D")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("District (sorted by median, high → low)")
    ax.set_xticks(x)
    ax.set_xticklabels(stats[district_col].tolist(), rotation=25)

    apply_infographic_style(ax, grid_axis="y")
    ax.legend(["Mean ± Std", "Median"], loc="best", frameon=False)

    reserved_top = add_title_subtitle(
        fig,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
    )
    fig.tight_layout(rect=(0, 0, 1, reserved_top))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multiline_by_district(
    df: pd.DataFrame,
    month_col: str,
    district_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    subtitle: Optional[str],
    ylabel: str,
    agg: str = "median",
    show_end_labels: bool = True,
    end_label_fmt: Optional[str] = None,
    legend: bool = False,
    legend_ncol: int = 4,
    title_fontsize: int = 18,
    title_fontweight: str = "bold",
    
) -> None:
    """
    Statista-like multi-line district comparison (one chart).

    Improvements:
    1) End-of-line dots match the corresponding line color.
    2) Only top-N districts get end labels (default N=4) to reduce clutter.
    3) Labels are placed in the right blank area (fixed x_text) with collision-avoid.
    """
    for col in (month_col, district_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")

    tmp = df[[month_col, district_col, value_col]].dropna().copy()
    tmp[district_col] = tmp[district_col].astype(str)

    if agg not in ("mean", "median", "sum", "first"):
        raise ValueError("agg must be one of: mean, median, sum, first")

    tmp = tmp.sort_values([district_col, month_col])

    districts = sorted(tmp[district_col].unique().tolist())
    series_map: dict[str, pd.Series] = {}

    for d in districts:
        sub = tmp[tmp[district_col] == d].sort_values(month_col)
        g = sub.groupby(month_col)[value_col]
        if agg == "mean":
            s = g.mean()
        elif agg == "median":
            s = g.median()
        elif agg == "sum":
            s = g.sum()
        else:
            s = g.first()
        series_map[d] = s

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=160)

    # Axis style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=11)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    # Plot lines and store (last value, mean value, color)
    info: dict[str, dict[str, float | str | object]] = {}
    for d in districts:
        s = series_map[d]
        x = s.index.to_numpy()
        y = s.to_numpy(dtype=float)
        if len(x) == 0:
            continue

        line, = ax.plot(x, y, linewidth=3.0, label=d)
        color = line.get_color()

        info[d] = {
            "x_last": x[-1],
            "y_last": float(y[-1]),
            "y_mean": float(np.nanmean(y)),
            "color": color,
        }

    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(legend_ncol, len(districts)),
            frameon=False,
            fontsize=11,
            handlelength=2.2,
            columnspacing=1.2,
        )

    # Expand xlim to create right-side whitespace for labels
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.18)

    # End labels
    if show_end_labels and info:
        if end_label_fmt is None:
            if "price" in value_col.lower():
                end_label_fmt = "{name}  {val:,.0f}"
            else:
                end_label_fmt = "{name}  {val:.2f}"

        # ✅ label ALL districts (no ranking / no truncation)
        all_items = list(info.items())

        # sort labels by y_last low->high to reduce collisions
        top_sorted = sorted(all_items, key=lambda kv: float(kv[1]["y_last"]))  # type: ignore

        # vertical collision control
        y0, y1 = ax.get_ylim()
        min_gap = (y1 - y0) * 0.03

        placed_ys: list[float] = []
        x_span = (x_max - x_min)
        x_text = x_max + x_span * 0.02

        for name, meta in top_sorted:
            x_last = meta["x_last"]
            y_last = float(meta["y_last"])  # type: ignore
            color = str(meta["color"])

            y_text = y_last
            while any(abs(y_text - py) < min_gap for py in placed_ys):
                y_text += min_gap * 0.6
            placed_ys.append(y_text)

            ax.scatter(
                [x_last], [y_last],  # type: ignore
                s=55,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

            ax.text(
                x_text,
                y_text,
                end_label_fmt.format(name=name, val=y_last),
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec="none",
                    alpha=0.85,
                ),
                clip_on=False,
            )


    reserved_top = add_title_subtitle(
        fig,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
    )

    fig.tight_layout(rect=(0, 0, 0.98, reserved_top))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_infographic_time_series(
    df: pd.DataFrame,
    month_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    subtitle: str,
    agg: str,
    ylabel: str,
    highlight_period: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    annotate_points: Optional[list[tuple[pd.Timestamp, str]]] = None,
    title_fontsize: int = 18,
    title_fontweight: str = "bold",
) -> None:
    for col in (month_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")

    tmp = df[[month_col, value_col]].dropna().sort_values(month_col)

    if agg not in ("mean", "median", "sum", "first"):
        raise ValueError("agg must be one of: mean, median, sum, first")

    g = tmp.groupby(month_col)[value_col]
    if agg == "mean":
        series = g.mean()
    elif agg == "median":
        series = g.median()
    elif agg == "sum":
        series = g.sum()
    else:
        series = g.first()

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)

    x = series.index.to_numpy()
    y = series.to_numpy(dtype=float)
    y_min = float(series.min())

    ax.plot(x, y, linewidth=3.2)
    ax.fill_between(x, y, y_min, alpha=0.12)

    if highlight_period is not None:
        start, end = highlight_period
        ax.axvspan(start, end, alpha=0.10)  # type: ignore

    last_x = x[-1]
    last_y = y[-1]
    ax.scatter([last_x], [last_y], s=45, zorder=3)
    ax.text(last_x, last_y, f"  {last_y:.2f}", va="center", fontsize=12, fontweight="bold")

    if annotate_points:
        idx = pd.Index(series.index)
        for x_pt, text in annotate_points:
            if x_pt in idx:
                y_pt = float(series.loc[x_pt])
                ax.annotate(
                    text,
                    xy=(x_pt, y_pt),  # type: ignore
                    xytext=(15, 20),
                    textcoords="offset points",
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", lw=1.2),
                )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    apply_infographic_style(ax, grid_axis="y")

    reserved_top = add_title_subtitle(
        fig,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
    )
    fig.tight_layout(rect=(0, 0, 1, reserved_top))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================
# Pipeline
# =========================
def run_train_eda(cfg: EDAConfig) -> dict[str, object]:
    ensure_dir(cfg.out_dir)

    df = load_train_dataset(cfg.train_path, month_col=cfg.month_col)
    num_cols = pick_numeric_cols(df, cfg.numeric_cols, month_col=cfg.month_col)

    overview = dataset_overview(df, month_col=cfg.month_col)
    print("\n[Overview]")
    for k, v in overview.items():
        print(f"{k}: {v}")

    miss = missing_summary(df)
    miss_path = cfg.out_dir / "missing_summary_train.csv"
    miss.to_csv(miss_path, encoding="utf-8-sig")
    print(f"\nSaved: {miss_path}")

    desc = describe_numeric(df, num_cols)
    desc_path = cfg.out_dir / "describe_numeric_train.csv"
    desc.to_csv(desc_path, encoding="utf-8-sig")
    print(f"Saved: {desc_path}")

    skew_cols = pick_skew_cols(df, cfg.skew_cols)
    sk = skewness_table(df, skew_cols)
    sk_path = cfg.out_dir / "skewness_train.csv"
    sk.to_csv(sk_path, encoding="utf-8-sig")
    print(f"Saved: {sk_path}")

    corr = correlation_matrix(df, num_cols)
    corr_path = cfg.out_dir / "corr_train.csv"
    corr.to_csv(corr_path, encoding="utf-8-sig")
    print(f"Saved: {corr_path}")

    heatmap_path = cfg.out_dir / "Correlation_Matrix_Train.png"
    plot_corr_heatmap(
        corr=corr,
        out_path=heatmap_path,
        title="Correlation structure across training variables",
        subtitle="Pearson correlation matrix (train set)",
        title_fontsize=cfg.suptitle_fontsize,
        title_fontweight=cfg.title_fontweight,
    )
    print(f"Saved: {heatmap_path}")

    boxplot_path = None
    if cfg.plot_box_by_district:
        boxplot_path = cfg.out_dir / "Boxplot_MedianPrice_by_District.png"
        plot_boxplot_by_district(
            df=df,
            district_col=cfg.district_col,
            value_col=cfg.price_col,
            out_path=boxplot_path,
            title="Median prices vary widely by district",
            subtitle="Boxplot of monthly median house prices (train set)",
            ylabel=cfg.price_col,
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
        )
        print(f"Saved: {boxplot_path}")

    stats_csv_path = None
    errbar_path = None
    if cfg.plot_mean_median_std_by_district:
        stats = district_level_stats(df, cfg.district_col, cfg.price_col)
        stats_csv_path = cfg.out_dir / "district_price_stats.csv"
        stats.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {stats_csv_path}")

        errbar_path = cfg.out_dir / "District_MeanMedian_Std_Errorbars.png"
        plot_mean_median_std_errorbars(
            stats=stats,
            district_col=cfg.district_col,
            out_path=errbar_path,
            title="District price levels and volatility",
            subtitle="Mean ± Std and Median (train set)",
            ylabel=cfg.price_col,
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
        )
        print(f"Saved: {errbar_path}")

    # Multi-line comparisons (EDA on observed data)
    multi_price_path = None
    if cfg.plot_multiline_price_by_district:
        multi_price_path = cfg.out_dir / "MedianPrice_Trends_by_District_MultiLine.png"
        plot_multiline_by_district(
            df=df,
            month_col=cfg.month_col,
            district_col=cfg.district_col,
            value_col=cfg.price_col,
            out_path=multi_price_path,
            title="Median prices moved together—but gaps remain across districts",
            subtitle="7 districts on one chart (train set)",
            ylabel="Median Price (NZD)",
            agg="median",
            show_end_labels=True,
            end_label_fmt="{name}",
            legend=False,
            legend_ncol=4,
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
            
        )
        print(f"Saved: {multi_price_path}")

    multi_sales_path = None
    if cfg.plot_multiline_sales_by_district:
        multi_sales_path = cfg.out_dir / "SalesCount_Trends_by_District_MultiLine.png"
        plot_multiline_by_district(
            df=df,
            month_col=cfg.month_col,
            district_col=cfg.district_col,
            value_col=cfg.sales_col,
            out_path=multi_sales_path,
            title="Sales activity varies notably across districts",
            subtitle="Monthly sales count (train set)",
            ylabel="Sales Count",
            agg="sum",
            show_end_labels=True,
            end_label_fmt="{name}",
            legend=False,
            legend_ncol=4,
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
          
        )
        print(f"Saved: {multi_sales_path}")

    multi_consents_path = None
    if cfg.plot_multiline_consents_by_district:
        multi_consents_path = cfg.out_dir / "Consents_Trends_by_District_MultiLine.png"
        plot_multiline_by_district(
            df=df,
            month_col=cfg.month_col,
            district_col=cfg.district_col,
            value_col=cfg.consents_col,
            out_path=multi_consents_path,
            title="Building consents show different supply cycles by district",
            subtitle="Monthly residential consents (train set)",
            ylabel="Building Consents",
            agg="sum",
            show_end_labels=True,
            end_label_fmt="{name}",
            legend=False,
            legend_ncol=4,
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
           
        )
        print(f"Saved: {multi_consents_path}")

    # OCR: infographic time series (national series)
    ocr_ts_path = None
    if cfg.plot_ocr_time_series:
        ocr_ts_path = cfg.out_dir / "TimeSeries_OCR.png"
        plot_infographic_time_series(
            df=df,
            month_col=cfg.month_col,
            value_col=cfg.ocr_col,
            out_path=ocr_ts_path,
            title="OCR surged sharply after 2021",
            subtitle="Monthly Official Cash Rate (New Zealand)",
            agg="first",
            ylabel="OCR (%)",
            highlight_period=(pd.Timestamp("2021-10-01"), pd.Timestamp("2023-07-01")),
            annotate_points=[
                (pd.Timestamp("2020-03-01"), "COVID easing: OCR cut to record low"),
                (pd.Timestamp("2022-10-01"), "Rapid tightening cycle"),
            ],
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
        )
        print(f"Saved: {ocr_ts_path}")

    # Real Mortgage Rate: infographic time series
    rmr_ts_path = None
    if cfg.plot_real_mortgage_rate_time_series:
        rmr_ts_path = cfg.out_dir / "TimeSeries_RealMortgageRate.png"
        plot_infographic_time_series(
            df=df,
            month_col=cfg.month_col,
            value_col=cfg.real_mortgage_rate_col,
            out_path=rmr_ts_path,
            title="Real mortgage rates turned positive post-tightening",
            subtitle="Mortgage rate minus CPI inflation (monthly)",
            agg="mean",
            ylabel="Real Mortgage Rate (%)",
            highlight_period=(pd.Timestamp("2021-10-01"), pd.Timestamp("2023-07-01")),
            annotate_points=None,
            title_fontsize=cfg.suptitle_fontsize,
            title_fontweight=cfg.title_fontweight,
        )
        print(f"Saved: {rmr_ts_path}")

    return {
        "overview": overview,
        "num_cols": num_cols,
        "outputs": {
            "missing_summary": miss_path,
            "describe": desc_path,
            "skewness": sk_path,
            "corr_csv": corr_path,
            "corr_heatmap": heatmap_path,
            "boxplot_by_district": boxplot_path,
            "district_stats_csv": stats_csv_path,
            "mean_median_std_errorbars": errbar_path,
            "multiline_price_by_district": multi_price_path,
            "multiline_sales_by_district": multi_sales_path,
            "multiline_consents_by_district": multi_consents_path,
            "ocr_time_series": ocr_ts_path,
            "real_mortgage_rate_time_series": rmr_ts_path,
        },
    }


def main() -> None:
    from src.config import TRAIN_DIR, OUTPUT_DIR

    train_path = TRAIN_DIR / "train.csv"
    out_dir = OUTPUT_DIR / "eda_train"

    cfg = EDAConfig(
        train_path=train_path,
        out_dir=out_dir,
        month_col="Month",
        district_col="District",
        price_col="Median_Price",
        numeric_cols=None,
        skew_cols=[
            "Median_Price",
            "sales_count",
            "Consents",
            "OCR",
            "CGPI_Dwelling",
            "unemployment_rate",
            "weeklyrent",
            "RealMortgageRate",
            "Net_migration_monthly",
        ],
        plot_box_by_district=True,
        plot_mean_median_std_by_district=True,
        plot_multiline_price_by_district=True,
        plot_multiline_sales_by_district=True,
        plot_multiline_consents_by_district=True,
        plot_ocr_time_series=True,
        plot_real_mortgage_rate_time_series=True,
        title_fontsize=16,
        suptitle_fontsize=18,
        title_fontweight="bold",
    )

    run_train_eda(cfg)


if __name__ == "__main__":
    main()
