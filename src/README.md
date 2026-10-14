# Freight Rate Prediction — Implementation

This directory contains the solution to the Spotter Freight Rate Prediction assessment: a supervised regression model that predicts the `posted_rate` (USD) of a truckload from its lane, distance, equipment, weight, date, and market signals.

Problem framing, metrics, and validation rationale are documented in [`docs/problem-framing.md`](../docs/problem-framing.md).

## Approach at a glance

1. **Target:** rate per mile (`posted_rate / distance`), converted back to dollars at prediction time.
2. **Validation:** out-of-time — development data (Jan–Oct 2025) is split chronologically to mirror the hidden Nov–Dec evaluation, with rolling-origin cross-validation for model selection.
3. **Data quality:** negative weights, missing values, and extreme rate-per-mile labels are handled inside the training pipeline, fit on training folds only.
4. **Models:** median baselines → regularized linear model → gradient-boosted trees, compared on the same splits.
5. **Inference:** the final model is refit on all development data and predicts `validation.csv` and the December chart inputs.

## Project structure


```
src/
├── README.md            # this file
├── POETRY.md            # setup with Poetry or plain Python/pip
├── manifest.md          # project decisions, step by step (private, git-ignored)
├── config.py            # paths, constants, split months, random seed
├── metrics.py           # MAPE (primary), WAPE, MAE, RMSE
├── split.py             # out-of-time split and expanding-window CV folds
├── data_preparation.py  # exploration + deterministic cleaning
├── features.py          # loading cleaned data, deterministic features
├── train.py             # cross-validation of every model (Step 5)
├── tuning.py            # Phase 2 experiments and hyperparameter search (Step 6)
├── final_model.py       # selected configuration
├── test_evaluation.py   # one-time October test (Step 7)
├── predict.py           # submission files (Step 7)
├── report.py            # submission report PDF (Step 8)
├── sync_requirements.py # regenerates requirements.txt from poetry.lock
├── models/              # base interface, label filter, baseline, regression, lgbm_model, hybrid
├── steps/               # one markdown document per project step
├── freight-rate-report/ # the submission report PDF
├── data/                # cleaned datasets (generated)
└── outputs/             # tables and figures (generated)
docs/
├── problem-framing.md
└── storyline.md         # presentation script
```

## Setup

Python ≥ 3.14 (the locked packages support 3.12+). Use either Poetry or plain pip; both install the same versions and produce identical outputs. Details, and how to switch between the two, are in [`POETRY.md`](POETRY.md).

```bash
# Poetry
poetry install --no-root --with report

# or plain Python
python -m pip install -r requirements.txt
```

The commands below use `poetry run`; with plain Python, drop that prefix.

## How to run

```bash
# 1. Explore and clean the data (Step 4)
poetry run python -m src.data_preparation
```

```bash
# 2. Cross-validate every model (Step 5)
poetry run python -m src.train

# 3. Tuning experiments (Step 6, ~3.5 minutes)
poetry run python -m src.tuning
```

```bash
# 4. One-time evaluation on the October test set (Step 7)
poetry run python -m src.test_evaluation

# 5. Refit on Jan–Oct and write validation_predictions.csv + fill december-chart-inputs.csv (Step 7)
poetry run python -m src.predict

# 6. Validate outputs and render the December chart
poetry run python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv

# 7. Build the report: src/freight-rate-report/freight-rate-report.pdf (Step 8)
poetry run python -m src.report
```

## Data

| File | Rows | Period | Purpose |
|---|---|---|---|
| `train-test.csv` | 48,000 | Jan–Oct 2025 | Labeled development data |
| `validation.csv` | 12,000 | Nov–Dec 2025 | Final predictions (scored by Spotter) |
| `december-chart-inputs.csv` | 31 | Dec 2025 | Fixed load, one row per day |

Data files live at the repository root (hyphenated names), not under `data/` as the original instructions state.

## Data cleaning

See [`steps/04-data-preparation.md`](steps/04-data-preparation.md). Summary: negative weights are sign errors (absolute value); missing `weight` and `market_index` are imputed inside model pipelines; about 1.3% of labels are corrupted (clusters far from their lane median) and are handled in training folds only.

## Features

Rate per mile is the target. Features: 1 / distance and log distance (a load costs a fixed charge plus a per-mile charge), day of week, weight, market index, quote signal, coordinates, equipment, pickup and delivery city. Month is excluded because December is never seen in training. Details in [`steps/05-models.md`](steps/05-models.md).

## Models

| Model | File | Summary |
|---|---|---|
| Baseline | `models/baseline.py` | Median rate per mile per equipment × distance |
| Ridge regression | `models/regression.py` | Imputation, scaling, one-hot encoding, L2-regularized linear regression |
| LightGBM | `models/lgbm_model.py` | Gradient-boosted trees, L1 objective, native missing values |
| **Hybrid (final)** | `models/hybrid.py` | Ridge with a time trend as the base, LightGBM learning a relative correction on top |

All share `models/base.py` (predict rate per mile, convert to dollars) and can drop corrupted training labels with `models/label_filter.py`, fit on training rows only.

**Final choice: the hybrid** (`final_model.py`). Ridge carries the upward time trend (held at the last training day) and the 1 / distance shape of rate per mile; LightGBM adds the interactions Ridge cannot express. Details in [`steps/06-tuning.md`](steps/06-tuning.md).

## Validation results

Expanding-window cross-validation on Jan–Sep (4 folds):

| Model | MAPE (%) | MAPE, plausible labels (%) | WAPE (%) | MAE ($) | RMSE ($) |
|---|---|---|---|---|---|
| Baseline | 10.31 ± 0.49 | 8.28 | 9.28 | 224 | 676 |
| Ridge regression | 5.29 ± 1.22 | 3.20 | 5.53 | 134 | 652 |
| LightGBM | 5.42 ± 1.62 | 3.34 | 5.43 | 131 | 645 |
| **Hybrid (final, tuned)** | **4.15 ± 0.32** | **1.99** | **4.30** | **104** | **639** |

October test (trained on Jan–Sep, evaluated once after the final configuration was frozen):

| Model | MAPE (%) | MAPE, plausible labels (%) | WAPE (%) | MAE ($) |
|---|---|---|---|---|
| Baseline | 10.93 | 8.10 | 9.72 | 231 |
| Ridge regression | 6.29 | 3.41 | 5.81 | 138 |
| LightGBM | 5.61 | 2.71 | 5.25 | 125 |
| **Hybrid (final)** | **4.54** | **1.56** | **4.04** | **96** |

About 3 of the final model's 4.5 MAPE points come from the 1.6% of October loads with corrupted labels. Details in [`steps/07-final-evaluation-and-submission.md`](steps/07-final-evaluation-and-submission.md).

## December chart

![December chart](../scorer_results/candidate_december.png)

$824–841 for the fixed load, slightly above the lane's historical Dry Van median ($807), consistent with the upward trend held at its October level. The weekly cycle (peak on Thursdays) comes from the market index inputs for each date.

## Limitations and next steps

- About 1.5% of labels are corrupted; they set an error floor of ~2–3 MAPE points no model can remove.
- The time trend is held at its last level, not extrapolated: safer, but slightly low if prices keep rising.
- No holiday effects: Thanksgiving and Christmas are outside the training period.
- `market_index` and `quote_signal` are assumed known at quote time; their meaning is undocumented.
- Next steps: prediction intervals (a price range per load), monitoring the trend as new months arrive, holiday features once holiday data exists.
