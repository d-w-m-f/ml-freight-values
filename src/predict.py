"""Phase 3 - Final predictions for submission.

Run from the repository root (after src.data_preparation):

    poetry run python -m src.predict

Refits the final configuration on all development data (Jan-Oct) and writes:

- validation_predictions.csv     load_id,predicted_rate for the 12,000 validation loads,
                                 in the template's order
- december-chart-inputs.csv      the provided file with its predicted_rate column filled
                                 (original seven columns kept unchanged)

Then validate and draw the chart with:

    poetry run python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
"""

import pandas as pd

from src import config
from src.features import load_clean
from src.final_model import build_final_model

TEMPLATE = config.ROOT_DIR / "validation-predictions-template.csv"
VALIDATION_PREDICTIONS = config.ROOT_DIR / "validation_predictions.csv"


def main() -> None:
    model = build_final_model().fit(load_clean(config.CLEAN_TRAIN_TEST))
    print(f"Final model fit on Jan-Oct ({model.trimmed_rows_} suspect labels trimmed from training)")

    validation = load_clean(config.CLEAN_VALIDATION)
    rates = pd.Series(model.predict(validation).round(2), index=validation["load_id"])
    template = pd.read_csv(TEMPLATE)
    template["predicted_rate"] = template["load_id"].map(rates)
    if template["predicted_rate"].isna().any() or (template["predicted_rate"] <= 0).any():
        raise ValueError("missing or non-positive validation predictions")
    template.to_csv(VALIDATION_PREDICTIONS, index=False)
    print(f"Wrote {len(template):,} predictions to {VALIDATION_PREDICTIONS.name}")

    # Predict from the completed inputs, write into the provided file so its columns stay as given.
    december = load_clean(config.CLEAN_DECEMBER)
    provided = pd.read_csv(config.RAW_DECEMBER)
    if not (pd.to_datetime(provided["date"]) == december["date"]).all():
        raise ValueError("December rows are not aligned with the provided file")
    provided["predicted_rate"] = model.predict(december).round(2)
    provided.to_csv(config.RAW_DECEMBER, index=False)
    print(f"Filled predicted_rate in {config.RAW_DECEMBER.name}")

    print("\nValidation predictions:")
    print(template["predicted_rate"].describe().round(2).to_string())
    print("\nDecember predictions:")
    print(provided[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
