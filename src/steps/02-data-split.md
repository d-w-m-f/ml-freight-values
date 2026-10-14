# Step 2 — Data split

**Code:** [`src/split.py`](../split.py) · constants in [`src/config.py`](../config.py)

## Decision

`train-test.csv` (Jan–Oct 2025) is split by month — an **out-of-time** split:

| Set | Months | Rows | Use |
|---|---|---|---|
| Train | Jan–Aug | 38,477 | Fit models |
| Validation | Sep | 4,670 | Compare models, check decisions, early stopping |
| Test | Oct | 4,853 | One final evaluation per final model |

Functions:

- `train_validation_test_split(df)` → `(train, validation, test)`.
- `development_data(df)` → every row except October; the only data used for exploration and cross-validation.

## Why split by time

Spotter scores the model on Nov–Dec 2025 loads, all *after* the development data. The internal split reproduces that: models are always evaluated on months later than the ones they were trained on.

## Why Sep = validation and Oct = test (not a random mix of Sep–Oct)

- Keeping the test set as the latest month preserves the forward-in-time ordering at every stage.
- A random mix would put loads from the same week in both sets. Market conditions are shared within a week (see Step 4: the market index has a strong weekly cycle), so the test score would look better than Nov–Dec performance.

## Test-set rule

October is not explored, plotted, or used for any decision until the end of the project (after tuning). Step 4's exploration uses `development_data()` and therefore never sees it.

## Final model

For the submission, the chosen configuration is refit on all of Jan–Oct before predicting `validation.csv` and the December chart inputs.
