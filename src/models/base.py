"""Common interface shared by every model.

Each model predicts rate per mile and converts it back to a posted rate. The
corrupted-label filter is fit on the rows passed to fit(), so inside cross-
validation it only ever sees the fold's training months.
"""

import numpy as np
import pandas as pd

from src.features import add_features, rate_per_mile
from src.models.label_filter import CorruptedLabelFilter


class RatePerMileModel:
    name = "model"

    def __init__(self, trim_labels: bool = True):
        self.trim_labels = trim_labels

    def fit(self, df: pd.DataFrame) -> "RatePerMileModel":
        if self.trim_labels:
            keep = CorruptedLabelFilter().fit(df).is_plausible(df)
            self.trimmed_rows_ = int((~keep).sum())
            df = df[keep]
        else:
            self.trimmed_rows_ = 0
        self._fit(add_features(df), rate_per_mile(df))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted posted rate in dollars."""
        return self._predict(add_features(df)) * df["distance"].to_numpy()

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
