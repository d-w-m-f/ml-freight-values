"""Step 4 - Data exploration and cleaning.

Run from the repository root:

    poetry run python -m src.data_preparation

What it does:
1. Loads the three raw files.
2. Applies deterministic cleaning only (no statistic learned from the data).
3. Runs data-quality checks.
4. Explores the development data (Jan-Sep; October is the test set and is never explored).
5. Completes the December chart inputs (coordinates, market fields).
6. Saves cleaned files to src/data/ and tables, figures and a summary to src/outputs/data_preparation/.

Anything learned from the data (imputation values, outlier cut-offs, target
statistics) is deliberately NOT applied here: it belongs inside each model's
pipeline so it is fit on training folds only.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.split import development_data

LOAD_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date", "market_index", "quote_signal",
]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# Reporting thresholds only: rows are flagged in the summary, never removed here.
LOW_RATE_PER_MILE = 1.0
HIGH_RATE_PER_MILE = 5.0
MIN_DISTANCE = 70.0  # distances appear floored at this value
SUSPECT_DISTANCE_RATIO = 2.0  # road distance / straight-line distance
CORRUPT_LABEL_LOW = 0.6  # posted rate / lane-equipment median expectation
CORRUPT_LABEL_HIGH = 1.8

# Chart styling (categorical slots 1-3 of the reference palette, fixed order).
EQUIPMENT_COLORS = {"Dry Van": "#2a78d6", "Flatbed": "#eb6834", "Reefer": "#1baf7a"}
SINGLE_SERIES_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- loading


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_test = pd.read_csv(config.RAW_TRAIN_TEST)
    validation = pd.read_csv(config.RAW_VALIDATION)
    december = pd.read_csv(config.RAW_DECEMBER)

    _require_columns(train_test, LOAD_COLUMNS + [config.TARGET], "train-test.csv")
    _require_columns(validation, LOAD_COLUMNS, "validation.csv")
    _require_columns(december, DECEMBER_COLUMNS, "december-chart-inputs.csv")
    return train_test, validation, december


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


# --------------------------------------------------------------------------- cleaning


def clean_loads(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-by-row cleaning shared by every input file."""
    df = df.copy()
    for column in ["pickup", "delivery", "equipment"]:
        df[column] = df[column].str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    unknown = set(df["equipment"].dropna()) - set(EQUIPMENT)
    if unknown:
        raise ValueError(f"unexpected equipment values: {sorted(unknown)}")

    # Negative weights mirror the positive distribution exactly, so they are sign errors.
    df["weight_was_negative"] = df["weight"] < 0
    df["weight"] = df["weight"].abs()
    return df


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per city with its coordinates, collected from pickup and delivery columns."""
    parts = []
    for df in frames:
        for side in ["pickup", "delivery"]:
            part = df[[side, f"{side}_lat", f"{side}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    cities = pd.concat(parts).drop_duplicates()
    if cities["city"].duplicated().any():
        raise ValueError("a city appears with more than one coordinate pair")
    return cities.set_index("city").sort_index()


def complete_december(december: pd.DataFrame, cities: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Add the inputs the December file lacks, so it has the same features as validation.csv.

    - Coordinates: looked up from the city table (fixed per city).
    - market_index / quote_signal: median of the validation loads on the same date.
      These are model inputs Spotter provides for those dates, not labels, so no
      target information is involved.
    """
    df = clean_loads(december)
    for side in ["pickup", "delivery"]:
        df[f"{side}_lat"] = df[side].map(cities["lat"])
        df[f"{side}_lon"] = df[side].map(cities["lon"])
    daily = validation.groupby("date")[["market_index", "quote_signal"]].median()
    df["market_index"] = df["date"].map(daily["market_index"])
    df["quote_signal"] = df["date"].map(daily["quote_signal"])

    if df[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].isna().any().any():
        raise ValueError("could not complete every December input")
    return df


# --------------------------------------------------------------------------- quality checks


def straight_line_miles(df: pd.DataFrame) -> pd.Series:
    """Great-circle (haversine) distance between pickup and delivery, in miles."""
    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def quality_report(df: pd.DataFrame) -> dict:
    """Counts of every data-quality issue found during exploration."""
    distance_ratio = df["distance"] / straight_line_miles(df)
    report = {
        "rows": len(df),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "duplicate_load_ids": int(df["load_id"].duplicated().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns="load_id").duplicated().sum()),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "negative_weight_fixed": int(df["weight_was_negative"].sum()),
        "same_pickup_and_delivery": int((df["pickup"] == df["delivery"]).sum()),
        "distance_min": float(df["distance"].min()),
        f"distance_at_{MIN_DISTANCE:g}_mile_floor": int((df["distance"] == MIN_DISTANCE).sum()),
        f"distance_over_{SUSPECT_DISTANCE_RATIO:g}x_straight_line": int((distance_ratio > SUSPECT_DISTANCE_RATIO).sum()),
    }
    if config.TARGET in df:
        rate_per_mile = df[config.TARGET] / df["distance"]
        ratio = label_ratio_to_lane(df)
        report["non_positive_rate"] = int((df[config.TARGET] <= 0).sum())
        report[f"rate_per_mile_below_{LOW_RATE_PER_MILE:g}"] = int((rate_per_mile < LOW_RATE_PER_MILE).sum())
        report[f"rate_per_mile_above_{HIGH_RATE_PER_MILE:g}"] = int((rate_per_mile > HIGH_RATE_PER_MILE).sum())
        report["suspect_labels_vs_lane_median"] = {
            f"below_{CORRUPT_LABEL_LOW:g}x": int((ratio < CORRUPT_LABEL_LOW).sum()),
            f"above_{CORRUPT_LABEL_HIGH:g}x": int((ratio > CORRUPT_LABEL_HIGH).sum()),
        }
    return report


def label_ratio_to_lane(df: pd.DataFrame) -> pd.Series:
    """Posted rate divided by what the lane-equipment median rate per mile predicts.

    Exploration only: a target statistic computed on the whole frame, so it must
    never be used as a model feature or a cleaning rule outside a training fold.
    """
    rate_per_mile = df[config.TARGET] / df["distance"]
    lane_median = rate_per_mile.groupby([df["pickup"], df["delivery"], df["equipment"]]).transform("median")
    return rate_per_mile / lane_median


# --------------------------------------------------------------------------- exploration


def explore(dev: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Tables and figures on the development months (Jan-Sep) only."""
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    dev = dev.assign(
        rate_per_mile=dev[config.TARGET] / dev["distance"],
        month=dev["date"].dt.month,
    )

    numeric = ["distance", "weight", "market_index", "quote_signal", config.TARGET, "rate_per_mile"]
    dev[numeric].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(3).to_csv(config.EDA_DIR / "numeric_summary.csv")

    by_month = dev.pivot_table(index="month", columns="equipment", values="rate_per_mile", aggfunc="median").round(3)
    by_month.to_csv(config.EDA_DIR / "rate_per_mile_by_month_equipment.csv")

    buckets = pd.cut(dev["distance"], [0, 250, 500, 1000, 2000, 4000])
    by_distance = dev.groupby(buckets, observed=True)["rate_per_mile"].agg(["median", "count"]).round(3)
    by_distance.to_csv(config.EDA_DIR / "rate_per_mile_by_distance.csv")

    missing = dev.groupby("month")[["weight", "market_index"]].apply(lambda g: g.isna().sum())
    missing.to_csv(config.EDA_DIR / "missing_by_month.csv")

    correlations = dev[["rate_per_mile", "distance", "weight", "market_index", "quote_signal"]].corr(method="spearman")
    correlations.round(3).to_csv(config.EDA_DIR / "spearman_correlations.csv")

    by_weekday = dev.groupby(dev["date"].dt.day_name())[["market_index", "rate_per_mile"]].agg({"market_index": "mean", "rate_per_mile": "median"})
    by_weekday = by_weekday.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).rename_axis("weekday").round(3)
    by_weekday.to_csv(config.EDA_DIR / "market_index_and_rate_by_weekday.csv")

    daily_market = dev.groupby("date")["market_index"].agg(["mean", "std"])
    daily_rate = dev.groupby("date")["rate_per_mile"].median()
    unseen = ~validation["pickup"].isin(dev["pickup"]) | ~validation["delivery"].isin(dev["delivery"])

    _plot_rate_per_mile_distribution(dev)
    _plot_rate_per_mile_by_month(by_month)
    _plot_rate_per_mile_by_distance(by_distance)
    _plot_market_index(dev, validation)
    _plot_label_ratio(label_ratio_to_lane(dev))

    return {
        "rate_per_mile_median_by_equipment": dev.groupby("equipment")["rate_per_mile"].median().round(3).to_dict(),
        "rate_per_mile_median_by_distance": {str(k): v for k, v in by_distance["median"].items()},
        "spearman_with_rate_per_mile": correlations["rate_per_mile"].drop("rate_per_mile").round(3).to_dict(),
        "market_index_std_within_day": round(float(daily_market["std"].mean()), 3),
        "market_index_std_between_days": round(float(daily_market["mean"].std()), 3),
        "spearman_daily_market_index_vs_daily_rate_per_mile": round(float(daily_market["mean"].rank().corr(daily_rate.rank())), 3),
        "cities_in_development": int(pd.concat([dev["pickup"], dev["delivery"]]).nunique()),
        "lanes_in_development": int(dev.groupby(["pickup", "delivery"]).ngroups),
        "validation_cities_unseen": sorted(set(validation["pickup"]) - set(dev["pickup"])),
        "validation_rows_with_unseen_city": round(float(unseen.mean()), 3),
    }


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY, pad=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ae")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)


def _save(fig, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(config.EDA_DIR / name, dpi=150)
    plt.close(fig)


def _plot_rate_per_mile_distribution(dev: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = dev["rate_per_mile"].clip(upper=8)
    ax.hist(values, bins=np.arange(0, 8.1, 0.1), color=SINGLE_SERIES_COLOR, edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Rate per mile ($), values above 8 shown at 8", color=TEXT_SECONDARY)
    _style(ax, "Rate per mile, Jan–Sep 2025", "Loads (log scale)")
    _save(fig, "rate_per_mile_distribution.png")


def _plot_rate_per_mile_by_month(by_month: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for equipment in EQUIPMENT:
        series = by_month[equipment]
        ax.plot(series.index, series.values, color=EQUIPMENT_COLORS[equipment], linewidth=2, marker="o", markersize=5, label=equipment)
        ax.annotate(equipment, (series.index[-1], series.values[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(by_month.index, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"][: len(by_month)])
    ax.set_xlim(by_month.index.min() - 0.3, by_month.index.max() + 0.9)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_SECONDARY)
    _style(ax, "Median rate per mile by month and equipment", "Rate per mile ($)")
    _save(fig, "rate_per_mile_by_month.png")


def _plot_rate_per_mile_by_distance(by_distance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{int(i.left)}–{int(i.right)}" for i in by_distance.index]
    bars = ax.bar(labels, by_distance["median"], color=SINGLE_SERIES_COLOR, width=0.6)
    ax.bar_label(bars, fmt="$%.2f", padding=3, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("Distance (miles)", color=TEXT_SECONDARY)
    _style(ax, "Median rate per mile falls with distance", "Rate per mile ($)")
    _save(fig, "rate_per_mile_by_distance.png")


def _plot_market_index(dev: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Daily mean market_index: development months, then the Nov-Dec model inputs."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    development = dev.groupby("date")["market_index"].mean()
    future = validation.groupby("date")["market_index"].mean()
    ax.plot(development.index, development.values, color=SINGLE_SERIES_COLOR, linewidth=2)
    ax.plot(future.index, future.values, color=SINGLE_SERIES_COLOR, linewidth=2, linestyle=(0, (3, 2)))
    ax.annotate("Jan–Sep (development)", (development.index[20], development.max()), color=TEXT_PRIMARY, fontsize=9)
    ax.annotate("Nov–Dec (validation inputs)", (future.index[0], future.max() + 0.03), color=TEXT_PRIMARY, fontsize=9)
    ax.annotate("Oct: test month,\nnot explored", (pd.Timestamp("2025-10-03"), 1.2), color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Daily mean market index", "market_index")
    _save(fig, "market_index_daily.png")


def _plot_label_ratio(ratio: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(ratio, bins=np.logspace(np.log10(0.1), np.log10(8), 80), color=SINGLE_SERIES_COLOR, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.2, 0.33, 0.5, 1, 2, 3, 5], ["0.2×", "0.33×", "0.5×", "1×", "2×", "3×", "5×"])
    ax.minorticks_off()
    ax.set_xlabel("Posted rate ÷ lane-equipment median expectation", color=TEXT_SECONDARY)
    _style(ax, "Suspect labels form separate clusters far from 1×", "Loads (log scale)")
    _save(fig, "label_ratio_to_lane_median.png")


# --------------------------------------------------------------------------- main


def main() -> None:
    raw_train_test, raw_validation, raw_december = load_raw()

    train_test = clean_loads(raw_train_test)
    validation = clean_loads(raw_validation)
    cities = city_coordinates(train_test, validation)
    december = complete_december(raw_december, cities, validation)

    dev = development_data(train_test)
    summary = {
        "quality_development_jan_sep": quality_report(dev),
        "quality_validation_nov_dec": quality_report(validation),
        "exploration": explore(dev, validation),
        "december_inputs": december[["date", "market_index", "quote_signal"]]
        .assign(date=december["date"].dt.date.astype(str))
        .round(4)
        .to_dict(orient="records"),
    }

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_test.to_csv(config.CLEAN_TRAIN_TEST, index=False)
    validation.to_csv(config.CLEAN_VALIDATION, index=False)
    december.to_csv(config.CLEAN_DECEMBER, index=False)
    (config.EDA_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "december_inputs"}, indent=2))
    print(f"\nCleaned files written to {config.PROCESSED_DIR}")
    print(f"Exploration outputs written to {config.EDA_DIR}")


if __name__ == "__main__":
    main()
