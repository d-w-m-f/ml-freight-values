# Step 6 — Tuning (Phase 2)

**Code:** [`src/tuning.py`](../tuning.py) · [`src/models/hybrid.py`](../models/hybrid.py) · [`src/final_model.py`](../final_model.py)

```bash
poetry run python -m src.tuning   # ~3.5 minutes
```

Outputs in `src/outputs/tuning/`: one CSV per stage, plus `1_drift_bias.csv` and `1_drift_segments.csv`.

All experiments use the same four expanding-window folds as Step 5 (Jan–Sep). October stays untouched.

**Selection rule:** lowest mean MAPE **plus one standard deviation** across folds. A setting has to be accurate *and* stable from month to month to win.

## Stage 1 — The upward drift

### Diagnosis

Step 5 showed both models predicting too low in later months. Two explanations were tested on daily median rates (relative to their equipment-distance band, plausible labels only):

| Explanation | Share of day-to-day variation explained |
|---|---|
| Market index (same day) | 57% |
| Market index + lagged market index (prices reacting with a delay, 28–84 days) | 63–72% |
| **Market index + time trend** | **84%** |

The drift is a time trend of roughly **+0.75% per month**, independent of the market index.

### Fixes tested

| Variant | What it does |
|---|---|
| `regression_trend` | Ridge with a `days_since_start` feature |
| `lightgbm_trend` | LightGBM with the same feature (trees cannot extrapolate it) |
| `hybrid` | Ridge with the trend as a base; LightGBM learns a relative correction (log-ratio) on top |
| `*_trend_held` | Same, but at prediction time the trend is frozen at the last training day: the level reached is kept, the slope is not extrapolated forward |

### Results

| Model | MAPE | Jun | Jul | Aug | Sep | Short loads (< 150 mi) |
|---|---|---|---|---|---|---|
| Ridge (Step 5) | 5.29 ± 1.22 | 4.77 | 4.19 | 5.18 | 7.01 | 5.53 |
| LightGBM (Step 5) | 5.42 ± 1.62 | 4.89 | 3.89 | 7.69 | 5.19 | 9.21 |
| Ridge + trend | 4.89 ± 0.98 | 4.09 | 6.32 | 4.49 | 4.66 | 5.62 |
| LightGBM + trend | 5.48 ± 1.17 | 6.06 | 4.88 | 6.80 | 4.18 | 8.64 |
| Hybrid | 4.53 ± 1.00 | 3.56 | 5.89 | 4.60 | 4.05 | 4.68 |
| Ridge + trend held | 4.77 ± 0.55 | 4.36 | 5.53 | 4.37 | 4.84 | — |
| **Hybrid, trend held** | **4.35 ± 0.57** | 3.66 | 5.03 | 4.48 | 4.23 | — |

Findings:

- **The trend fixes the September under-prediction** (Ridge: 7.01 → 4.66).
- **Extrapolating the trend is risky.** In the July fold, the training months (Jan–Jun) end at the market peak, so the learned slope is too steep: predictions overshoot by ~3.5% and July MAPE jumps (4.19 → 6.32). Holding the trend at the last training day removes most of that risk and halves the fold-to-fold spread.
- **The hybrid combines both models' strengths.** Ridge carries the trend and the 1 / distance shape (short-load MAPE 4.68% vs LightGBM's 9.21%). LightGBM adds the interactions Ridge cannot express.

## Stage 2 — Corrupted-label trimming

| Model | With trimming | Without trimming |
|---|---|---|
| Ridge + trend held | **4.77 ± 0.55** | 5.63 ± 1.46 |
| Hybrid, trend held | **4.35 ± 0.57** | 4.65 ± 1.05 |

**Trimming is kept.** In Step 5 it made no difference. Once a trend is estimated, the corrupted prices distort the slope, and removing them from the training folds matters.

## Stage 3 — Ridge regularization (alpha)

| alpha | 1 | 100 | 1,000 | **3,000** | 10,000 | 30,000 | 100,000 |
|---|---|---|---|---|---|---|---|
| MAPE | 4.35 ± 0.57 | 4.31 ± 0.50 | 4.17 ± 0.36 | **4.12 ± 0.40** | 4.10 ± 0.51 | 4.25 ± 0.76 | 4.69 ± 1.18 |

Selected: **alpha = 3,000**. Strong regularization mainly shrinks the ~130 one-hot city columns, which fits Step 5's finding that lanes barely move prices; the trend and 1 / distance coefficients stay intact (trend 0.0253 at alpha 3,000 vs 0.0248 at alpha 1). Beyond ~10,000 those coefficients start shrinking too (market index 0.060 → 0.033 at 30,000) and errors rise.

## Stage 4 — LightGBM correction model

Grid: 300 / 600 / 1,000 trees × 15 / 31 / 63 leaves × 20 / 40 / 100 minimum samples per leaf (27 settings).

- The whole grid spans only **0.04 points** of MAPE (4.11–4.15), far inside the fold-to-fold spread (~0.3–0.5). The model is not sensitive to these settings.
- The selection rule picks the **smallest, most regularized** setting: 300 trees, 15 leaves, 100 minimum samples. Equal accuracy, lowest spread (± 0.32).

## Final configuration — `src/final_model.py`

| Setting | Value |
|---|---|
| Model | Hybrid: Ridge base + LightGBM relative correction |
| Target | Rate per mile |
| Corrupted-label trimming | On (training folds only) |
| Trend | `days_since_start` in Ridge, held at the last training day |
| Ridge alpha | 3,000 |
| LightGBM | 300 trees, 15 leaves, 100 min samples per leaf, learning rate 0.05, L1 objective |

### Cross-validation (Jan–Sep)

| Model | MAPE | MAPE, plausible labels | WAPE | MAE | Sep MAPE |
|---|---|---|---|---|---|
| Baseline | 10.31 ± 0.49 | 8.28 | 9.28 | $224 | 10.04 |
| Ridge (Step 5) | 5.29 ± 1.22 | 3.20 | 5.53 | $134 | 7.01 |
| LightGBM (Step 5) | 5.42 ± 1.62 | 3.34 | 5.43 | $131 | 5.19 |
| **Final hybrid** | **4.15 ± 0.32** | **1.99 ± 0.20** | **4.30** | **$104** | **4.22** |

- **60% less error than the baseline**, and about 22% less than the best Step 5 model.
- **Stable:** monthly MAPE between 3.8% and 4.6%.
- **Unbiased:** median predicted / actual per month is 0.991, 1.014, 1.011, 0.984 (Step 5 Ridge in September: 0.951).
- On plausible labels the error is about **2%**. The remaining ~2 points are the corrupted labels.

### December preview (fit on Jan–Sep)

$818–835, a weekly rhythm peaking on Thursdays, for Lexington → Fort Wayne, Dry Van, 360 miles. Historical Dry Van loads on this lane: median $807 (range $758–934); August–September scaled to 360 miles: ~$795–798. The preview sits slightly above recent months, consistent with the upward trend held at its latest level.

## Next

1. Evaluate the final configuration **once** on October (train Jan–Sep, test Oct).
2. Refit on Jan–Oct and produce `validation_predictions.csv` and the December chart.
