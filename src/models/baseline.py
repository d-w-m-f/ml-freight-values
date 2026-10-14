"""Baseline: median rate per mile of the load's equipment type, times distance.

This is how a broker could quote by hand. Every other model must beat it.
"""

import numpy as np
import pandas as pd

from src.models.base import RatePerMileModel


class BaselineModel(RatePerMileModel):
    name = "baseline"

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.medians_ = y.groupby(X["equipment"]).median()
        self.global_median_ = float(y.median())

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["equipment"].map(self.medians_).fillna(self.global_median_).to_numpy()
