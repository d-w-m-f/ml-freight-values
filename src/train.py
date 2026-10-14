"""Step 5 - Train every model with the same time-based cross-validation and compare them.

Run from the repository root:

    poetry run python -m src.train

Uses Jan-Sep only (October is the test set and stays untouched). Outputs go to
src/outputs/models/:

- cv_folds.csv                 metrics per model and fold
- cv_summary.csv               mean and std across folds, plus the September (validation) fold
- segments.csv                 out-of-fold MAPE by equipment and distance band
- lightgbm_feature_importance.csv
- december_preview.csv / .png  December chart inputs predicted by models fit on Jan-Sep
- cv_mape_by_fold.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.features import DISTANCE_BANDS, load_clean
from src.metrics import evaluate, mape
from src.models.baseline import BaselineModel
from src.models.label_filter import CorruptedLabelFilter
from src.models.lgbm_model import LightGBMModel
from src.models.regression import RegressionModel
from src.split import cv_folds, development_data

MODELS_DIR = config.OUTPUTS_DIR / "models"

# (name, model factory). The "no trimming" variants measure the effect of
# removing corrupted labels from the training folds.
VARIANTS = [
    ("baseline", lambda: BaselineModel()),
    ("regression", lambda: RegressionModel()),
    ("lightgbm", lambda: LightGBMModel()),
    ("regression_no_trimming", lambda: RegressionModel(trim_labels=False)),
    ("lightgbm_no_trimming", lambda: LightGBMModel(trim_labels=False)),
]
MAIN_MODELS = ["baseline", "regression", "lightgbm"]

COLORS = {"baseline": "#2a78d6", "regression": "#eb6834", "lightgbm": "#1baf7a"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def cross_validate(dev: pd.DataFrame, variants=VARIANTS, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics per fold for every (name, factory) variant, and the out-of-fold predictions."""
    rows, predictions = [], []
    for month, train, validation in cv_folds(dev):
        # Flag validation rows with implausible labels, judged by the training months only.
        # Used for reporting; these rows are always kept in the evaluation.
        plausible = CorruptedLabelFilter().fit(train).is_plausible(validation)
        y = validation[config.TARGET]
        for name, factory in variants:
            model = factory().fit(train)
            prediction = model.predict(validation)
            rows.append({
                "model": name,
                "validation_month": month,
                **evaluate(y, prediction),
                "mape_plausible_labels": mape(y[plausible], prediction[plausible.to_numpy()]),
                "training_rows_trimmed": model.trimmed_rows_,
            })
            predictions.append(pd.DataFrame({
                "model": name,
                "load_id": validation["load_id"].to_numpy(),
                "validation_month": month,
                "equipment": validation["equipment"].to_numpy(),
                "distance": validation["distance"].to_numpy(),
                "actual": y.to_numpy(),
                "predicted": prediction,
                "plausible_label": plausible.to_numpy(),
            }))
        if verbose:
            print(f"fold validated on month {month} done")
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def summarize(folds: pd.DataFrame, variants=VARIANTS) -> pd.DataFrame:
    metrics = ["mape", "wape", "mae", "rmse", "mape_plausible_labels"]
    summary = folds.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    september = folds[folds["validation_month"] == config.VALIDATION_MONTH].set_index("model")
    summary["mape_september"] = september["mape"]
    order = [name for name, _ in variants]
    return summary.loc[order].round(3)


def segment_errors(predictions: pd.DataFrame, models=MAIN_MODELS) -> pd.DataFrame:
    predictions = predictions[predictions["model"].isin(models)].assign(
        distance_band=pd.cut(predictions["distance"], DISTANCE_BANDS).astype(str),
        ape=(predictions["actual"] - predictions["predicted"]).abs() / predictions["actual"] * 100,
    )
    by_equipment = predictions.pivot_table(index="equipment", columns="model", values="ape", aggfunc="mean")
    by_distance = predictions.pivot_table(index="distance_band", columns="model", values="ape", aggfunc="mean")
    by_equipment.index = "equipment: " + by_equipment.index
    by_distance.index = "distance: " + by_distance.index
    return pd.concat([by_equipment, by_distance])[list(models)].round(2)


def december_preview(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Fit the main models on Jan-Sep and predict the fixed December load.

    A preview only: the submitted chart comes from the final model refit on Jan-Oct.
    """
    december = load_clean(config.CLEAN_DECEMBER)
    preview = december[["date"]].copy()
    importance = None
    for name in MAIN_MODELS:
        model = dict(VARIANTS)[name]().fit(dev)
        preview[name] = model.predict(december).round(2)
        if name == "lightgbm":
            importance = model.feature_importance()
    return preview, importance


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY, pad=28)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ae")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)


def _line_chart(frame: pd.DataFrame, x, title: str, ylabel: str, filename: str, xticks=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name in MAIN_MODELS:
        ax.plot(x, frame[name], color=COLORS[name], linewidth=2, marker="o", markersize=4, label=name)
        ax.annotate(name, (x[-1], frame[name].iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9)
    if xticks is not None:
        ax.set_xticks(*xticks)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(MAIN_MODELS), labelcolor=TEXT_SECONDARY)
    ax.margins(x=0.12)
    _style(ax, title, ylabel)
    fig.patch.set_facecolor(SURFACE)
    if xticks is None:
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(MODELS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dev = development_data(load_clean(config.CLEAN_TRAIN_TEST))

    folds, predictions = cross_validate(dev)
    summary = summarize(folds)
    segments = segment_errors(predictions)
    preview, importance = december_preview(dev)

    folds.round(4).to_csv(MODELS_DIR / "cv_folds.csv", index=False)
    summary.to_csv(MODELS_DIR / "cv_summary.csv")
    segments.to_csv(MODELS_DIR / "segments.csv")
    importance.round(4).to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", header=["gain_share"])
    preview.to_csv(MODELS_DIR / "december_preview.csv", index=False)

    by_fold = folds.pivot(index="validation_month", columns="model", values="mape")
    months = list(by_fold.index)
    _line_chart(by_fold, months, "MAPE by validation month", "MAPE (%)", "cv_mape_by_fold.png",
                xticks=(months, ["Jun", "Jul", "Aug", "Sep"]))
    _line_chart(preview.set_index("date"), list(preview["date"]), "December preview (models fit on Jan–Sep)",
                "Predicted rate ($)", "december_preview.png")

    pd.set_option("display.width", 200)
    print("\nCross-validation summary (Jan-Sep, expanding window):")
    print(summary.to_string())
    print("\nOut-of-fold MAPE by segment:")
    print(segments.to_string())
    print("\nLightGBM feature importance (share of gain):")
    print(importance.round(3).to_string())
    print("\nDecember preview:")
    print(preview.describe().loc[["min", "mean", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
