# Freight Rate Prediction Challenge — Solution

Predicts the posted rate of a truckload from its route, distance, equipment, weight, date and market signals.

| | |
|---|---|
| **Final model** | Hybrid: Ridge regression with a time trend, corrected by LightGBM |
| **Cross-validated MAPE** (Jan–Sep, 4 time-based folds) | **4.15% ± 0.32** (baseline 10.31%) |
| **October test MAPE** (held out until the end) | **4.54%** (1.56% on loads with plausible labels) |
| **Report** | [`src/freight-rate-report/freight-rate-report.pdf`](src/freight-rate-report/freight-rate-report.pdf) |
| **Implementation and results** | [`src/README.md`](src/README.md) |
| **Step-by-step documentation** | [`src/steps/`](src/steps/) |

## Submission files

| Assessment item | Where |
|---|---|
| Code, dependencies, run instructions | `src/`, `pyproject.toml` + `poetry.lock` (Poetry), `requirements.txt` (pip), [`src/POETRY.md`](src/POETRY.md) |
| `validation_predictions.csv` | Repository root |
| December predictions | `predicted_rate` filled in `december-chart-inputs.csv` |
| Report with the split / validation approach and the December chart | [`src/freight-rate-report/freight-rate-report.pdf`](src/freight-rate-report/freight-rate-report.pdf) |
| December chart from `score.py` | `scorer_results/candidate_december.png` |

## Quick start (plain Python)

```bash
python -m pip install -r requirements.txt
python -m src.data_preparation   # clean the data
python -m src.predict            # refit the final model, write both submission files
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

With Poetry, prefix each command with `poetry run`. See [`src/POETRY.md`](src/POETRY.md) for both setups and the full pipeline.

## File names

The data files are at the repository root, with hyphens, rather than under `data/` with underscores as the original instructions below say:

| Original instructions | This repository |
|---|---|
| `data/train_test.csv` | `train-test.csv` |
| `data/validation.csv` | `validation.csv` |
| `data/validation_predictions_template.csv` | `validation-predictions-template.csv` |
| `data/december_chart_inputs.csv` | `december-chart-inputs.csv` |
| `Freight_Rate_ML_Assessment.pdf` | `freight-rate-ml-assessment.pdf` |

`score.py` is kept exactly as provided.

---

## Original assessment instructions

*Kept as provided with the assessment.*

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

### What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

### Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
