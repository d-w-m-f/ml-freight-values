"""Hybrid: Ridge regression as the base, LightGBM correcting what Ridge misses.

Phase 2 combines the strengths found in Step 5:
- Ridge extrapolates the upward time trend and captures the 1 / distance shape
  of rate per mile exactly (strong on short loads).
- LightGBM captures interactions and non-linear effects, but cannot extrapolate.

LightGBM is trained on the log-ratio between the actual rate per mile and
Ridge's prediction, so it only learns a relative correction around Ridge.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.base import RatePerMileModel
from src.models.lgbm_model import PARAMS, feature_list, learn_categories, lgbm_frame
from src.models.regression import ALPHA, TREND_FEATURE, build_pipeline, hold_trend

MIN_BASE_RATE_PER_MILE = 0.1  # guards the log-ratio against a degenerate base prediction


class HybridModel(RatePerMileModel):
    name = "hybrid"

    def __init__(self, trim_labels: bool = True, alpha: float = ALPHA, extrapolate_trend: bool = True, **params):
        super().__init__(trim_labels)
        self.alpha = alpha
        self.extrapolate_trend = extrapolate_trend
        self.params = {**PARAMS, **params}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.last_training_day_ = int(X[TREND_FEATURE].max())
        self.ridge_ = build_pipeline(self.alpha, use_trend=True).fit(X, y)
        base = self._base(X)
        # The trend is left to Ridge; LightGBM sees the same features as on its own.
        self.features_ = feature_list(use_trend=False)
        self.categories_ = learn_categories(X)
        correction = np.log(y.to_numpy()) - np.log(base)
        self.lgbm_ = LGBMRegressor(**self.params).fit(lgbm_frame(X, self.features_, self.categories_), correction)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        correction = self.lgbm_.predict(lgbm_frame(X, self.features_, self.categories_))
        return self._base(X) * np.exp(correction)

    def _base(self, X: pd.DataFrame) -> np.ndarray:
        if not self.extrapolate_trend:
            X = hold_trend(X, self.last_training_day_)
        return np.maximum(self.ridge_.predict(X), MIN_BASE_RATE_PER_MILE)
