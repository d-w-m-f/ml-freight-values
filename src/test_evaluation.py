"""Phase 3 - One-time evaluation on the October test set.

Run from the repository root:

    poetry run python -m src.test_evaluation

Every model is trained on Jan-Sep and scored on October, which no previous step
has looked at. The final configuration was frozen before this ran; the other
models are reported for context only, not to choose between them. Outputs go to
src/outputs/test/.
"""

import pandas as pd

from src import config
from src.features import DISTANCE_BANDS, load_clean
from src.final_model import build_final_model
from src.metrics import evaluate, mape
from src.models.baseline import BaselineModel
from src.models.label_filter import CorruptedLabelFilter
from src.models.lgbm_model import LightGBMModel
from src.models.regression import RegressionModel
from src.split import development_data, train_validation_test_split

TEST_DIR = config.OUTPUTS_DIR / "test"

MODELS = [
    ("baseline", lambda: BaselineModel()),
    ("regression", lambda: RegressionModel()),
    ("lightgbm", lambda: LightGBMModel()),
    ("final_hybrid", build_final_model),
]


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean(config.CLEAN_TRAIN_TEST)
    train = development_data(data)  # Jan-Sep
    _, _, test = train_validation_test_split(data)  # Oct
    plausible = CorruptedLabelFilter().fit(train).is_plausible(test)
    y = test[config.TARGET]

    rows, predictions = [], {}
    for name, factory in MODELS:
        prediction = factory().fit(train).predict(test)
        predictions[name] = prediction
        rows.append({
            "model": name,
            **evaluate(y, prediction),
            "mape_plausible_labels": mape(y[plausible], prediction[plausible.to_numpy()]),
            "median_predicted_over_actual": float(pd.Series(prediction[plausible.to_numpy()] / y[plausible].to_numpy()).median()),
        })
    results = pd.DataFrame(rows).set_index("model").round(3)

    final = predictions["final_hybrid"]
    ape = (y - final).abs() / y * 100
    segments = pd.concat([
        ape.groupby(test["equipment"]).mean(),
        ape.groupby(pd.cut(test["distance"], DISTANCE_BANDS).astype(str)).mean(),
        ape.groupby(plausible.map({True: "plausible label", False: "suspect label"})).mean(),
    ]).round(2).rename("final_hybrid_mape")

    results.to_csv(TEST_DIR / "october_test.csv")
    segments.to_csv(TEST_DIR / "october_test_segments.csv")
    pd.DataFrame({"load_id": test["load_id"], "actual": y, **{k: v.round(2) for k, v in predictions.items()}}).to_csv(
        TEST_DIR / "october_test_predictions.csv", index=False
    )

    pd.set_option("display.width", 200)
    print(f"October test set: {len(test):,} loads, {int((~plausible).sum())} with suspect labels\n")
    print(results.to_string())
    print("\nFinal hybrid MAPE by segment:")
    print(segments.to_string())


if __name__ == "__main__":
    main()
