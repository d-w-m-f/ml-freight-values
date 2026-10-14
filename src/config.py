"""Project-wide paths and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

# Raw files provided with the assessment (repository root).
RAW_TRAIN_TEST = ROOT_DIR / "train-test.csv"
RAW_VALIDATION = ROOT_DIR / "validation.csv"
RAW_DECEMBER = ROOT_DIR / "december-chart-inputs.csv"

# Outputs of the data preparation step.
PROCESSED_DIR = SRC_DIR / "data"
CLEAN_TRAIN_TEST = PROCESSED_DIR / "train_test_clean.csv"
CLEAN_VALIDATION = PROCESSED_DIR / "validation_clean.csv"
CLEAN_DECEMBER = PROCESSED_DIR / "december_chart_inputs_clean.csv"
OUTPUTS_DIR = SRC_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "data_preparation"

RANDOM_SEED = 42
TARGET = "posted_rate"

# Out-of-time split of the development data (months of 2025).
TRAIN_MONTHS = list(range(1, 9))  # Jan-Aug
VALIDATION_MONTH = 9  # Sep
TEST_MONTH = 10  # Oct: untouched until the final evaluation

# Expanding-window cross-validation: each fold validates on one month and
# trains on every month before it.
CV_VALIDATION_MONTHS = [6, 7, 8, 9]
