"""Phase 2 - Tuning, using the same expanding-window folds as Step 5 (Jan-Sep only).

Run from the repository root:

    poetry run python -m src.tuning

Stages, each reported to src/outputs/tuning/:

1. drift      - fixes for the upward time drift (trend feature, hybrid model)
2. trimming   - corrupted-label trimming on / off for the stage-1 candidates
3. ridge      - Ridge regularization strength
4. lightgbm   - LightGBM hyperparameters for the hybrid's correction model
"""

import itertools

import pandas as pd

from src import config
from src.features import load_clean
from src.models.baseline import BaselineModel
from src.models.hybrid import HybridModel
from src.models.lgbm_model import LightGBMModel
from src.models.regression import RegressionModel
from src.split import development_data
from src.train import cross_validate, segment_errors, summarize

TUNING_DIR = config.OUTPUTS_DIR / "tuning"
# Selection rule: lowest mean MAPE plus one standard deviation across folds, so a
# setting must be both accurate and stable from month to month to win.
COLUMNS = ["mape_mean", "mape_std", "mape_plausible_labels_mean", "wape_mean", "mae_mean", "mape_september"]


def run_stage(name: str, dev: pd.DataFrame, variants: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds, predictions = cross_validate(dev, variants, verbose=False)
    summary = summarize(folds, variants)
    by_month = folds.pivot(index="model", columns="validation_month", values="mape").loc[summary.index].round(2)
    by_month.columns = [f"mape_month_{m}" for m in by_month.columns]
    table = summary[COLUMNS].join(by_month)
    table.to_csv(TUNING_DIR / f"{name}.csv")
    print(f"\n=== {name} ===")
    print(table.to_string())
    return table, predictions


def best(table: pd.DataFrame) -> str:
    return (table["mape_mean"] + table["mape_std"]).idxmin()


def median_bias(predictions: pd.DataFrame) -> pd.DataFrame:
    """Median predicted / actual on plausible labels, per model and month (1.0 = unbiased)."""
    plausible = predictions[predictions["plausible_label"]]
    ratio = plausible["predicted"] / plausible["actual"]
    return ratio.groupby([plausible["model"], plausible["validation_month"]]).median().unstack().round(3)


def main() -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    dev = development_data(load_clean(config.CLEAN_TRAIN_TEST))

    # Stage 1 - drift.
    drift_variants = [
        ("baseline", lambda: BaselineModel()),
        ("regression", lambda: RegressionModel()),
        ("lightgbm", lambda: LightGBMModel()),
        ("regression_trend", lambda: RegressionModel(use_trend=True)),
        ("lightgbm_trend", lambda: LightGBMModel(use_trend=True)),
        ("hybrid", lambda: HybridModel()),
        ("regression_trend_held", lambda: RegressionModel(use_trend=True, extrapolate_trend=False)),
        ("hybrid_trend_held", lambda: HybridModel(extrapolate_trend=False)),
    ]
    _, predictions = run_stage("1_drift", dev, drift_variants)
    bias = median_bias(predictions)
    bias.to_csv(TUNING_DIR / "1_drift_bias.csv")
    print("\nMedian predicted / actual (plausible labels):")
    print(bias.to_string())
    segments = segment_errors(predictions, ["regression_trend", "lightgbm_trend", "hybrid", "hybrid_trend_held"])
    segments.to_csv(TUNING_DIR / "1_drift_segments.csv")
    print(segments.to_string())

    # Stage 2 - corrupted-label trimming, on the stage-1 winners.
    run_stage("2_trimming", dev, [
        ("regression_trend_held", lambda: RegressionModel(use_trend=True, extrapolate_trend=False)),
        ("regression_trend_held_no_trimming", lambda: RegressionModel(use_trend=True, extrapolate_trend=False, trim_labels=False)),
        ("hybrid_trend_held", lambda: HybridModel(extrapolate_trend=False)),
        ("hybrid_trend_held_no_trimming", lambda: HybridModel(extrapolate_trend=False, trim_labels=False)),
    ])

    # Stage 3 - Ridge regularization strength inside the hybrid.
    alphas = [1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
    ridge_table, _ = run_stage("3_ridge_alpha", dev, [
        (f"alpha={a:g}", lambda a=a: HybridModel(extrapolate_trend=False, alpha=a)) for a in alphas
    ])
    best_alpha = float(best(ridge_table).split("=")[1])

    # Stage 4 - LightGBM correction-model hyperparameters, with the best alpha.
    grid = list(itertools.product([300, 600, 1000], [15, 31, 63], [20, 40, 100]))
    lgbm_table, _ = run_stage("4_lightgbm", dev, [
        (
            f"trees={n},leaves={l},min_samples={m}",
            lambda n=n, l=l, m=m: HybridModel(
                extrapolate_trend=False, alpha=best_alpha, n_estimators=n, num_leaves=l, min_child_samples=m
            ),
        )
        for n, l, m in grid
    ])
    print(f"\nBest alpha: {best_alpha:g}")
    print(f"Best LightGBM settings by the selection rule: {best(lgbm_table)}")
    print("Spread of mean MAPE across the whole grid:", round(lgbm_table["mape_mean"].max() - lgbm_table["mape_mean"].min(), 3))
    print(lgbm_table.assign(score=lgbm_table["mape_mean"] + lgbm_table["mape_std"]).sort_values("score").head(5).to_string())

if __name__ == "__main__":
    main()
