"""Detection of corrupted labels, learned on training rows only.

Step 4 showed that ~1.3% of posted rates sit in clusters far from what similar
loads pay (about 0.2-0.6x and 2.2-5x), separated from normal loads by empty gaps.
Normal loads stay within ~0.85-1.17x of the median rate per mile of their
equipment and distance band, so that median is a reliable reference.
"""

import pandas as pd

from src.features import distance_band, rate_per_mile

LOW_RATIO = 0.6
HIGH_RATIO = 1.8


class CorruptedLabelFilter:
    def __init__(self, low: float = LOW_RATIO, high: float = HIGH_RATIO):
        self.low = low
        self.high = high

    def fit(self, df: pd.DataFrame) -> "CorruptedLabelFilter":
        rpm = rate_per_mile(df)
        self.reference_ = rpm.groupby([df["equipment"], distance_band(df["distance"])]).median()
        self.equipment_reference_ = rpm.groupby(df["equipment"]).median()
        return self

    def is_plausible(self, df: pd.DataFrame) -> pd.Series:
        """True for rows whose label is within [low, high] x the learned reference."""
        keys = pd.MultiIndex.from_arrays([df["equipment"], distance_band(df["distance"])])
        reference = pd.Series(self.reference_.reindex(keys).to_numpy(), index=df.index)
        reference = reference.fillna(df["equipment"].map(self.equipment_reference_))
        ratio = rate_per_mile(df) / reference
        return ratio.between(self.low, self.high)
