"""Ridge regression (L2-regularized linear regression) on rate per mile.

Pipeline: median imputation with missing-value flags and standardization for
numeric features, one-hot encoding for categorical features, then Ridge. Every
step is fit on the training rows only. Cities unseen in training get an all-zero
encoding, i.e. the average city effect.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.base import RatePerMileModel

NUMERIC_FEATURES = [
    "inverse_distance", "log_distance", "weight", "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery", "day_of_week"]
TREND_FEATURE = "days_since_start"
ALPHA = 1.0


def hold_trend(X: pd.DataFrame, last_training_day: int) -> pd.DataFrame:
    """Freeze the trend at the last training day: the level is kept, the slope is not extrapolated."""
    return X.assign(**{TREND_FEATURE: X[TREND_FEATURE].clip(upper=last_training_day)})


def build_pipeline(alpha: float = ALPHA, use_trend: bool = False) -> Pipeline:
    numeric_features = NUMERIC_FEATURES + ([TREND_FEATURE] if use_trend else [])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha))])


class RegressionModel(RatePerMileModel):
    name = "regression"

    def __init__(
        self, trim_labels: bool = True, alpha: float = ALPHA, use_trend: bool = False, extrapolate_trend: bool = True
    ):
        super().__init__(trim_labels)
        self.alpha = alpha
        self.use_trend = use_trend
        self.extrapolate_trend = extrapolate_trend

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.last_training_day_ = int(X[TREND_FEATURE].max())
        self.pipeline_ = build_pipeline(self.alpha, self.use_trend).fit(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.extrapolate_trend:
            X = hold_trend(X, self.last_training_day_)
        return self.pipeline_.predict(X)
