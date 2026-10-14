"""Out-of-time data splits: train / validation / test and expanding-window cross-validation."""

from collections.abc import Iterator

import pandas as pd

from src.config import CV_VALIDATION_MONTHS, TEST_MONTH, TRAIN_MONTHS, VALIDATION_MONTH


def _month(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date"])
    if (dates.dt.year != 2025).any():
        raise ValueError("development data is expected to cover 2025 only")
    return dates.dt.month


def train_validation_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train = Jan-Aug, validation = Sep, test = Oct.

    The test set is only used once, for the final evaluation.
    """
    month = _month(df)
    train = df[month.isin(TRAIN_MONTHS)]
    validation = df[month == VALIDATION_MONTH]
    test = df[month == TEST_MONTH]
    return train, validation, test


def development_data(df: pd.DataFrame) -> pd.DataFrame:
    """Every row except the test month (Jan-Sep): the only data used for exploration and CV."""
    return df[_month(df) != TEST_MONTH]


def cv_folds(df: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds over Jan-Sep.

    Yields (validation_month, train, validation): the fold trains on every month
    before the validation month. The test month never appears in any fold.
    """
    month = _month(df)
    for validation_month in CV_VALIDATION_MONTHS:
        train = df[month < validation_month]
        validation = df[month == validation_month]
        yield validation_month, train, validation
