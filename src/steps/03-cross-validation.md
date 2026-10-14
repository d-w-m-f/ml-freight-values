# Step 3 — Cross-validation

**Code:** `cv_folds()` in [`src/split.py`](../split.py)

## Decision

Expanding-window (rolling-origin) time-series cross-validation over Jan–Sep. October never enters any fold.

| Fold | Train | Train rows | Validate | Validation rows |
|---|---|---|---|---|
| 1 | Jan–May | 24,023 | Jun | 4,783 |
| 2 | Jan–Jun | 28,806 | Jul | 4,912 |
| 3 | Jan–Jul | 33,718 | Aug | 4,759 |
| 4 | Jan–Aug | 38,477 | Sep | 4,670 |

`cv_folds(df)` yields `(validation_month, train, validation)` for each fold.

## Why this strategy

- **No future data in training:** each fold trains only on months before its validation month, like the real task.
- **Consistent with Step 2:** fold 4 is exactly the train/validation split.
- **Stability, not just accuracy:** four scores instead of one. Models are ranked by **mean MAPE** across folds; the **standard deviation** shows whether a model is consistent month to month. The folds cover very different market conditions (June is the rate peak, August–September are low), so a model that only works in one regime will show it.
- **Random k-fold is not used:** it mixes past and future loads and overstates accuracy.

## Rule for every model

Every preprocessing step that learns from data — imputation values, encoders, outlier thresholds, lane statistics — is part of the model's pipeline and is refit inside each fold on that fold's training rows only.
