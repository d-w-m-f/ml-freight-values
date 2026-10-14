"""LightGBM (gradient-boosted decision trees) on rate per mile.

Named lgbm_model.py so it does not shadow the lightgbm package. LightGBM
handles missing values natively, so no imputation is needed. City and equipment
categories are fixed to those seen in training; an unseen city is treated as
missing. The absolute-error (L1) objective keeps any remaining noisy labels from
pulling the fit.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import RANDOM_SEED
from src.models.base import RatePerMileModel

NUMERIC_FEATURES = [
    "inverse_distance", "log_distance", "weight", "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "day_of_week",
]
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
TREND_FEATURE = "days_since_start"

PARAMS = {
    "objective": "l1",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


def feature_list(use_trend: bool) -> list[str]:
    return NUMERIC_FEATURES + ([TREND_FEATURE] if use_trend else []) + CATEGORICAL_FEATURES


def learn_categories(X: pd.DataFrame) -> dict[str, list]:
    return {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_FEATURES}


def lgbm_frame(X: pd.DataFrame, features: list[str], categories: dict[str, list]) -> pd.DataFrame:
    """Select features and fix categorical levels to those seen in training."""
    X = X[features].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = pd.Categorical(X[column], categories=categories[column])
    return X


class LightGBMModel(RatePerMileModel):
    name = "lightgbm"

    def __init__(self, trim_labels: bool = True, use_trend: bool = False, **params):
        super().__init__(trim_labels)
        self.use_trend = use_trend
        self.params = {**PARAMS, **params}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.features_ = feature_list(self.use_trend)
        self.categories_ = learn_categories(X)
        self.model_ = LGBMRegressor(**self.params).fit(lgbm_frame(X, self.features_, self.categories_), y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(lgbm_frame(X, self.features_, self.categories_))

    def feature_importance(self) -> pd.Series:
        """Total gain per feature, normalized to sum to 1."""
        gain = pd.Series(self.model_.booster_.feature_importance("gain"), index=self.features_)
        return (gain / gain.sum()).sort_values(ascending=False)
