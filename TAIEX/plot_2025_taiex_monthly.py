from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import requests


API_URL = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"
DEFAULT_OUTPUT = "outputs/taiex_10y_monthly_close_overlay.png"
DEFAULT_MONTHLY_RETURNS_OUTPUT = "outputs/taiex_10y_monthly_returns.csv"
DEFAULT_MONTHLY_SUMMARY_OUTPUT = "outputs/taiex_10y_calendar_month_summary.csv"
CLOSE_COL = "收盤指數"
DATE_COL = "日期"
DEFAULT_END_YEAR = date.today().year
DEFAULT_START_YEAR = DEFAULT_END_YEAR - 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch TWSE TAIEX close history and draw 12 monthly overlays for the last 10 years."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="First year to fetch. Defaults to current year minus 9.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Last year to fetch. Defaults to current year.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--returns-output",
        default=DEFAULT_MONTHLY_RETURNS_OUTPUT,
        help="Output CSV path for each year-month return.",
    )
    parser.add_argument(
        "--summary-output",
        default=DEFAULT_MONTHLY_SUMMARY_OUTPUT,
        help="Output CSV path for calendar-month summary statistics.",
    )
    parser.add_argument(
        "--cache",
        default="outputs/cache/twse_mi_5mins_hist",
        help="Folder for monthly TWSE API JSON cache.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always fetch from TWSE even when a cached monthly JSON exists.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay in seconds between uncached TWSE requests.",
    )
    return parser.parse_args()


def parse_number(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if not text or text in {"--", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_twse_date(value: object) -> pd.Timestamp | None:
    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()

    parts = text.split("/")
    if len(parts) == 3 and all(part.strip().isdigit() for part in parts):
        year, month, day = (int(part.strip()) for part in parts)
        if year < 1911:
            year += 1911
        return pd.Timestamp(year=year, month=month, day=day)

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def fetch_month(year: int, month: int, cache_dir: Path, use_cache: bool) -> dict:
    cache_path = cache_dir / f"{year}{month:02d}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = requests.get(
        API_URL,
        params={"response": "json", "date": f"{year}{month:02d}01"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def month_rows(payload: dict) -> list[list[object]]:
    if "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    tables = payload.get("tables")
    if isinstance(tables, list):
        for table in tables:
            fields = table.get("fields", [])
            if CLOSE_COL in fields and isinstance(table.get("data"), list):
                return table["data"]
    return []


def load_taiex_close_history(
    start_year: int,
    end_year: int,
    cache_dir: Path,
    use_cache: bool = True,
    delay: float = 0.35,
    include_previous_month: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    today = pd.Timestamp.today().normalize()
    first_month = pd.Period(year=start_year, month=1, freq="M")
    if include_previous_month:
        first_month -= 1
    last_month = min(
        pd.Period(year=end_year, month=12, freq="M"),
        today.to_period("M"),
    )

    for period in pd.period_range(first_month, last_month, freq="M"):
        year, month = period.year, period.month
        cache_path = cache_dir / f"{year}{month:02d}.json"
        was_cached = use_cache and cache_path.exists()
        payload = fetch_month(year, month, cache_dir, use_cache=use_cache)
        fields = payload.get("fields", [])
        rows = month_rows(payload)
        if not rows:
            continue

        date_idx = fields.index(DATE_COL) if DATE_COL in fields else 0
        close_idx = fields.index(CLOSE_COL) if CLOSE_COL in fields else 4

        for row in rows:
            if len(row) <= max(date_idx, close_idx):
                continue
            trade_date = parse_twse_date(row[date_idx])
            close = parse_number(row[close_idx])
            if trade_date is None or close is None:
                continue
            if first_month <= trade_date.to_period("M") <= last_month:
                records.append({"trade_date": trade_date, "close": close})

        if not was_cached and delay > 0:
            time.sleep(delay)

    if not records:
        raise RuntimeError(
            f"No TWSE {CLOSE_COL} rows found for {start_year}-{end_year}."
        )

    data = pd.DataFrame(records)
    data = data.drop_duplicates(subset=["trade_date"], keep="last")
    return data.sort_values("trade_date").reset_index(drop=True)


def prepare_overlay_data(
    data: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    prepared = data.sort_values("trade_date").copy()
    prepared["trade_month"] = prepared["trade_date"].dt.to_period("M")
    month_end_close = prepared.groupby("trade_month")["close"].last()
    previous_close_by_month = {
        period + 1: close for period, close in month_end_close.items()
    }
    prepared["previous_month_close"] = prepared["trade_month"].map(previous_close_by_month)
    prepared["year"] = prepared["trade_date"].dt.year
    prepared["month"] = prepared["trade_date"].dt.month
    prepared["day"] = prepared["trade_date"].dt.day
    prepared["month_trading_day"] = prepared.groupby(["year", "month"]).cumcount() + 1
    prepared["gain_pct"] = (
        prepared["close"] / prepared["previous_month_close"] - 1.0
    ) * 100.0

    if start_year is not None:
        prepared = prepared[prepared["year"] >= start_year]
    if end_year is not None:
        prepared = prepared[prepared["year"] <= end_year]
    return prepared.dropna(subset=["previous_month_close", "gain_pct"]).reset_index(drop=True)


def calculate_monthly_returns(
    data: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    prepared = prepare_overlay_data(data, start_year=start_year, end_year=end_year)
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "month",
                "month_end_date",
                "previous_month_close",
                "month_end_close",
                "monthly_return_pct",
                "is_complete_month",
            ]
        )

    monthly = prepared.groupby(["year", "month"], as_index=False).tail(1).copy()
    monthly = monthly.rename(
        columns={
            "trade_date": "month_end_date",
            "close": "month_end_close",
            "gain_pct": "monthly_return_pct",
        }
    )
    monthly["is_complete_month"] = monthly["trade_month"] < pd.Timestamp.today().to_period("M")
    return monthly[
        [
            "year",
            "month",
            "month_end_date",
            "previous_month_close",
            "month_end_close",
            "monthly_return_pct",
            "is_complete_month",
        ]
    ].reset_index(drop=True)


def summarize_calendar_months(
    monthly_returns: pd.DataFrame,
    complete_only: bool = True,
) -> pd.DataFrame:
    source = monthly_returns.copy()
    if complete_only:
        source = source[source["is_complete_month"]]

    summary = source.groupby("month")["monthly_return_pct"].agg(
        observations="count",
        average_return_pct="mean",
        median_return_pct="median",
        best_return_pct="max",
        worst_return_pct="min",
    )
    win_rate = source.groupby("month")["monthly_return_pct"].apply(
        lambda values: values.gt(0).mean() * 100.0
    )
    summary["win_rate_pct"] = win_rate
    return summary.reset_index()[
        [
            "month",
            "observations",
            "average_return_pct",
            "median_return_pct",
            "win_rate_pct",
            "best_return_pct",
            "worst_return_pct",
        ]
    ]


def plot_monthly_year_overlay(
    data: pd.DataFrame,
    output: Path,
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    overlay = prepare_overlay_data(data, start_year=start_year, end_year=end_year)
    monthly_returns = calculate_monthly_returns(
        data,
        start_year=start_year,
        end_year=end_year,
    )
    monthly_summary = summarize_calendar_months(monthly_returns)
    years = sorted(overlay["year"].unique())
    if not years:
        raise RuntimeError("No years to plot.")

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharey=False)
    axes_flat = axes.ravel()
    colors = plt.get_cmap("tab10").colors

    for month in range(1, 13):
        ax = axes_flat[month - 1]
        month_frame = overlay[overlay["month"] == month]
        ax.axhline(0, color="#6b7280", linewidth=0.8)
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.grid(True, axis="x", color="#f3f4f6", linewidth=0.5)
        ax.set_title(f"{month:02d} 月", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)

        if month_frame.empty:
            ax.text(0.5, 0.5, "無資料", ha="center", va="center", transform=ax.transAxes)
            continue

        complete_years = monthly_returns.loc[
            (monthly_returns["month"] == month)
            & monthly_returns["is_complete_month"],
            "year",
        ].tolist()

        complete_paths: list[np.ndarray] = []
        progress_grid = np.linspace(0.0, 100.0, 101)
        for idx, year in enumerate(years):
            if year not in complete_years:
                continue
            frame = month_frame[month_frame["year"] == year]
            if frame.empty:
                continue
            color = colors[idx % len(colors)]
            progress = np.linspace(0.0, 100.0, len(frame) + 1)
            gains = np.array([0.0, *frame["gain_pct"].tolist()])
            ax.plot(
                progress,
                gains,
                color=color,
                linewidth=1.05,
                alpha=0.78,
                label=str(year),
            )
            ax.plot(
                100.0,
                gains[-1],
                marker="o",
                markersize=3.0,
                color=color,
                alpha=0.9,
            )
            complete_paths.append(np.interp(progress_grid, progress, gains))

        if complete_paths:
            average_path = np.mean(complete_paths, axis=0)
            ax.plot(
                progress_grid,
                average_path,
                color="#111827",
                linewidth=3.0,
                label="完整月份平均",
                zorder=10,
            )

            scale_values = np.concatenate(complete_paths)
            y_min = min(0.0, float(scale_values.min()))
            y_max = max(0.0, float(scale_values.max()))
            y_padding = max(0.75, (y_max - y_min) * 0.10)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        summary_row = monthly_summary[monthly_summary["month"] == month]
        if not summary_row.empty:
            stats = summary_row.iloc[0]
            stats_text = (
                f"月底平均 {stats['average_return_pct']:+.2f}%  |  "
                f"中位數 {stats['median_return_pct']:+.2f}%  |  "
                f"上漲機率 {stats['win_rate_pct']:.0f}%  |  "
                f"n={int(stats['observations'])}"
            )
            ax.text(
                0.015,
                0.965,
                stats_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color="#374151",
                bbox={
                    "boxstyle": "square,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                zorder=20,
            )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="年度",
        ncol=11,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"{years[0]}-{years[-1]} 發行量加權股價指數每月收盤報酬疊圖",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel(
        "月內進度（0% = 上月底收盤，100% = 本月底收盤）",
        fontsize=11,
        y=0.036,
    )
    fig.supylabel("相對上月底收盤指數報酬 (%)", fontsize=11)
    fig.text(
        0.5,
        0.012,
        "粗線：完整月份依月內進度標準化後的平均；框內數字：月底報酬 CSV 摘要",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.93))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    data = load_taiex_close_history(
        args.start_year,
        args.end_year,
        Path(args.cache),
        use_cache=not args.no_cache,
        delay=args.delay,
        include_previous_month=True,
    )
    output = Path(args.output)
    plot_monthly_year_overlay(
        data,
        output,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    monthly_returns = calculate_monthly_returns(
        data,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    monthly_summary = summarize_calendar_months(monthly_returns)
    returns_output = Path(args.returns_output)
    summary_output = Path(args.summary_output)
    returns_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_returns.to_csv(returns_output, index=False, encoding="utf-8-sig")
    monthly_summary.to_csv(summary_output, index=False, encoding="utf-8-sig")

    print(f"Wrote {output.resolve()}")
    print(f"Wrote {returns_output.resolve()}")
    print(f"Wrote {summary_output.resolve()}")
    print(f"Rows: {len(data):,}")
    print(f"Years: {args.start_year}-{args.end_year}")


if __name__ == "__main__":
    main()
