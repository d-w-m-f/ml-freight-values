# Step 1 — Evaluation metric

**Code:** [`src/metrics.py`](../metrics.py)

## Decision

| Metric | Role | Formula |
|---|---|---|
| **MAPE** | Primary — used for every model-selection decision | mean(\|y − ŷ\| / y) × 100 |
| WAPE | Secondary — interpretation | Σ\|y − ŷ\| / Σy × 100 |
| MAE | Secondary — interpretation, in dollars | mean(\|y − ŷ\|) |
| RMSE | Secondary — detects large misses | √mean((y − ŷ)²) |

`evaluate(y_true, y_pred)` returns all four, primary first.

## Why MAPE

- Rates range from ~$150 to ~$8,000. A percentage error compares short and long loads on the same footing, which matches how pricing errors are judged: being 10% off is equally bad on any load.
- All posted rates are positive (minimum $57), so MAPE is always defined.

## Known limits of MAPE, and why the secondary metrics are reported

- **Equal weight per load:** cheap, short loads dominate the score. WAPE weights errors by load value and shows the share of dollars actually mispriced.
- **Asymmetry:** over-prediction is penalized more than under-prediction, so a model tuned for MAPE tends to predict slightly low. Comparing MAPE with WAPE makes this visible.
- **Readability:** MAE and RMSE stay in dollars for the report. A large gap between RMSE and MAE means a few big misses dominate.

## Check

On the worked example (actual 500 / 2,000 / 5,000; predicted 600 / 1,900 / 4,500), the module returns MAPE 11.67%, WAPE 9.33%, MAE $233.33, RMSE $300.00 — matching the hand calculation.
