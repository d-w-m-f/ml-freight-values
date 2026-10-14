"""Loading of cleaned data and deterministic (row-by-row) feature engineering.

Nothing here learns from the data, so these features can be computed on any
file without leakage. Learned transformations live inside each model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TARGET

# Distance bands used to compare a load with similar loads (see label_filter.py).
DISTANCE_BANDS = [0, 150, 250, 400, 600, 900, 1300, 2000, 4000]
# Origin of the time-trend feature (first day of the development data).
TREND_START = pd.Timestamp("2025-01-01")


def load_clean(path: Path) -> pd.DataFrame:
    """Read a file written by src/data_preparation.py."""
    return pd.read_csv(path, parse_dates=["date"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # A load's price behaves like a fixed charge plus a per-mile charge, so its
    # rate per mile falls roughly like 1 / distance.
    df["inverse_distance"] = 1.0 / df["distance"]
    df["log_distance"] = np.log(df["distance"])
    df["day_of_week"] = df["date"].dt.dayofweek
    # Rates drift upward over the year beyond what the market index explains (Phase 2).
    df["days_since_start"] = (df["date"] - TREND_START).dt.days
    return df


def rate_per_mile(df: pd.DataFrame) -> pd.Series:
    """Modeling target: posted rate divided by distance."""
    return df[TARGET] / df["distance"]


def distance_band(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, DISTANCE_BANDS, labels=False)
