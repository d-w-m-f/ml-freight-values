"""The configuration selected in Phase 2 (see src/steps/06-tuning.md).

Hybrid model: Ridge regression with a time trend held at the last training day
(alpha = 3000), corrected by a small LightGBM model, trained with corrupted
labels trimmed. Cross-validated MAPE on Jan-Sep: 4.15% +/- 0.32.
"""

from src.models.hybrid import HybridModel

FINAL_PARAMS = {
    "trim_labels": True,
    "alpha": 3000.0,
    "extrapolate_trend": False,
    "n_estimators": 300,
    "num_leaves": 15,
    "min_child_samples": 100,
}


def build_final_model() -> HybridModel:
    return HybridModel(**FINAL_PARAMS)
