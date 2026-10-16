"""
M5 Demand Forecasting - Preprocessing Pipeline

Creates leakage-safe train, validation, and final test datasets
from the representative M5 series selected earlier.

Split:
    Train      : d_1 ... d_1885
    Validation : d_1886 ... d_1913  (28 days)
    Test       : d_1914 ... d_1941  (28 days)

The test period comes from sales_train_evaluation.csv and should
remain untouched until model selection is complete.
"""

from pathlib import Path
import re
import json

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPRESENTATIVE_FILE = PROCESSED_DIR / "representative_series.csv"

VALIDATION_SALES_FILE = RAW_DIR / "sales_train_validation.csv"
EVALUATION_SALES_FILE = RAW_DIR / "sales_train_evaluation.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_END = 1885
VALIDATION_START = 1886
VALIDATION_END = 1913

TEST_START = 1914
TEST_END = 1941

FORECAST_HORIZON = 28

ID_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

DAY_PATTERN = re.compile(r"^d_(\d+)$")


# ============================================================================
# HELPERS
# ============================================================================

def extract_day_number(day_column: str) -> int:
    """Convert d_123 -> 123."""
    match = DAY_PATTERN.match(day_column)

    if not match:
        raise ValueError(f"Invalid day column: {day_column}")

    return int(match.group(1))


def normalize_series_id(series_id: str) -> str:
    """
    Remove M5 validation/evaluation suffix.

    Example:
        FOODS_3_090_CA_3_validation
        -> FOODS_3_090_CA_3
    """
    return re.sub(r"_(validation|evaluation)$", "", series_id)


def get_day_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted M5 day columns."""
    day_columns = [
        column
        for column in df.columns
        if DAY_PATTERN.match(str(column))
    ]

    day_columns.sort(key=extract_day_number)

    return day_columns


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    filename: str,
) -> None:
    """Check that required columns exist."""
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )


# ============================================================================
# LOAD DATA
# ============================================================================

def load_calendar() -> pd.DataFrame:
    """Load and validate M5 calendar."""
    print("Loading calendar...")

    calendar = pd.read_csv(
        CALENDAR_FILE,
        parse_dates=["date"],
    )

    validate_required_columns(
        calendar,
        ["d", "date"],
        "calendar.csv",
    )

    if calendar["d"].duplicated().any():
        raise ValueError("Duplicate d values found in calendar.")

    calendar = calendar.sort_values("date").reset_index(drop=True)

    return calendar


def load_representative_series() -> pd.DataFrame:
    """Load representative series selected in series_selection.py."""
    print("Loading representative series...")

    representative = pd.read_csv(REPRESENTATIVE_FILE)

    validate_required_columns(
        representative,
        ID_COLUMNS,
        "representative_series.csv",
    )

    representative["series_id"] = (
        representative["id"]
        .astype(str)
        .map(normalize_series_id)
    )

    if representative["series_id"].duplicated().any():
        duplicates = representative.loc[
            representative["series_id"].duplicated(),
            "series_id",
        ].tolist()

        raise ValueError(
            f"Duplicate representative series found: {duplicates}"
        )

    return representative


def load_sales_file(filepath: Path) -> pd.DataFrame:
    """Load an M5 sales file."""
    print(f"Loading {filepath.name}...")

    sales = pd.read_csv(filepath)

    validate_required_columns(
        sales,
        ID_COLUMNS,
        filepath.name,
    )

    day_columns = get_day_columns(sales)

    if not day_columns:
        raise ValueError(
            f"No M5 day columns found in {filepath.name}."
        )

    return sales


# ============================================================================
# WIDE -> LONG
# ============================================================================

def wide_to_long(
    sales: pd.DataFrame,
    selected_series: set[str],
    calendar: pd.DataFrame,
    start_day: int,
    end_day: int,
    split_name: str,
) -> pd.DataFrame:
    """
    Convert selected M5 series from wide format to long format.

    Only the requested day range is converted.
    """

    day_columns = [
        f"d_{day}"
        for day in range(start_day, end_day + 1)
    ]

    missing_days = [
        column
        for column in day_columns
        if column not in sales.columns
    ]

    if missing_days:
        raise ValueError(
            f"Missing day columns in sales file: "
            f"{missing_days[:10]}"
        )

    working = sales.copy()

    working["series_id"] = (
        working["id"]
        .astype(str)
        .map(normalize_series_id)
    )

    working = working[
        working["series_id"].isin(selected_series)
    ].copy()

    if working.empty:
        raise ValueError(
            f"No selected series found for {split_name}."
        )

    # Keep only metadata + requested days.
    columns_to_keep = [
        column
        for column in ID_COLUMNS
        if column in working.columns
    ]

    columns_to_keep += ["series_id"]
    columns_to_keep += day_columns

    working = working[columns_to_keep]

    long_df = working.melt(
        id_vars=[
            column
            for column in columns_to_keep
            if column not in day_columns
        ],
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    long_df["day_number"] = (
        long_df["d"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    long_df["split"] = split_name

    # Merge calendar information.
    long_df = long_df.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one",
    )

    if long_df["date"].isna().any():
        raise ValueError(
            f"Calendar join failed for {split_name}."
        )

    return long_df


# ============================================================================
# TIME FEATURES
# ============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that are known in advance.

    IMPORTANT:
    No lag or rolling features are created here.

    Lag/rolling features will be created later inside the model-specific
    forecasting pipeline to ensure multi-step forecasts do not accidentally
    use future actual demand.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Basic calendar features.
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Days elapsed since beginning of dataset.
    df["days_since_start"] = (
        df["day_number"] - 1
    )

    # Cyclical day-of-week features.
    import numpy as np

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Cyclical month features.
    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    # Event indicators.
    event_columns = [
        column
        for column in [
            "event_name_1",
            "event_name_2",
        ]
        if column in df.columns
    ]

    if event_columns:
        df["is_event"] = (
            df[event_columns]
            .notna()
            .any(axis=1)
            .astype(int)
        )
    else:
        df["is_event"] = 0

    # SNAP indicator according to state.
    snap_mapping = {
        "CA": "snap_CA",
        "TX": "snap_TX",
        "WI": "snap_WI",
    }

    def get_snap(row):
        state = row["state_id"]
        snap_column = snap_mapping.get(state)

        if snap_column is None:
            return 0

        return row.get(snap_column, 0)

    df["snap"] = df.apply(get_snap, axis=1)

    # Ensure numeric.
    df["snap"] = pd.to_numeric(
        df["snap"],
        errors="coerce",
    ).fillna(0).astype(int)

    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_split(
    df: pd.DataFrame,
    split_name: str,
    expected_days: int,
    expected_series: int,
) -> None:
    """Validate a processed split."""

    expected_rows = expected_days * expected_series

    actual_rows = len(df)

    if actual_rows != expected_rows:
        raise ValueError(
            f"{split_name}: expected {expected_rows} rows, "
            f"found {actual_rows}."
        )

    unique_series = df["series_id"].nunique()

    if unique_series != expected_series:
        raise ValueError(
            f"{split_name}: expected {expected_series} series, "
            f"found {unique_series}."
        )

    duplicate_count = df.duplicated(
        subset=["series_id", "d"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"{split_name}: found {duplicate_count} "
            f"duplicate series/day observations."
        )

    # Every series should have exactly the expected number of observations.
    counts = df.groupby("series_id").size()

    if not (counts == expected_days).all():
        raise ValueError(
            f"{split_name}: some series do not have "
            f"{expected_days} observations."
        )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():

    print("=" * 72)
    print("M5 DEMAND FORECASTING - PREPROCESSING")
    print("=" * 72)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Load inputs
    # ----------------------------------------------------------------------

    calendar = load_calendar()

    representative = load_representative_series()

    selected_series = set(
        representative["series_id"]
    )

    print()
    print(f"Selected series: {len(selected_series)}")

    # ----------------------------------------------------------------------
    # Load sales files
    # ----------------------------------------------------------------------

    validation_sales = load_sales_file(
        VALIDATION_SALES_FILE
    )

    evaluation_sales = load_sales_file(
        EVALUATION_SALES_FILE
    )

    # ----------------------------------------------------------------------
    # Create datasets
    # ----------------------------------------------------------------------

    print()
    print("Creating train dataset...")

    train = wide_to_long(
        validation_sales,
        selected_series,
        calendar,
        1,
        TRAIN_END,
        "train",
    )

    print("Creating validation dataset...")

    validation = wide_to_long(
        validation_sales,
        selected_series,
        calendar,
        VALIDATION_START,
        VALIDATION_END,
        "validation",
    )

    print("Creating final test dataset...")

    test = wide_to_long(
        evaluation_sales,
        selected_series,
        calendar,
        TEST_START,
        TEST_END,
        "test",
    )

    # ----------------------------------------------------------------------
    # Add time features
    # ----------------------------------------------------------------------

    print()
    print("Adding time-based features...")

    train = add_time_features(train)
    validation = add_time_features(validation)
    test = add_time_features(test)

    # ----------------------------------------------------------------------
    # Validate splits
    # ----------------------------------------------------------------------

    expected_series = len(selected_series)

    validate_split(
        train,
        "TRAIN",
        TRAIN_END,
        expected_series,
    )

    validate_split(
        validation,
        "VALIDATION",
        FORECAST_HORIZON,
        expected_series,
    )

    validate_split(
        test,
        "TEST",
        FORECAST_HORIZON,
        expected_series,
    )

    # ----------------------------------------------------------------------
    # Check chronological separation
    # ----------------------------------------------------------------------

    train_max_day = train["day_number"].max()
    validation_min_day = validation["day_number"].min()
    validation_max_day = validation["day_number"].max()
    test_min_day = test["day_number"].min()

    if not train_max_day < validation_min_day:
        raise ValueError(
            "Train/validation chronological separation failed."
        )

    if not validation_max_day < test_min_day:
        raise ValueError(
            "Validation/test chronological separation failed."
        )

    # ----------------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------------

    all_data = pd.concat(
        [
            train,
            validation,
            test,
        ],
        ignore_index=True,
    )

    all_data = all_data.sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # Save datasets
    # ----------------------------------------------------------------------

    train_path = PROCESSED_DIR / "forecast_train.csv"
    validation_path = PROCESSED_DIR / "forecast_validation.csv"
    test_path = PROCESSED_DIR / "forecast_test.csv"
    all_path = PROCESSED_DIR / "forecast_all.csv"

    train.to_csv(
        train_path,
        index=False,
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    test.to_csv(
        test_path,
        index=False,
    )

    all_data.to_csv(
        all_path,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Save metadata
    # ----------------------------------------------------------------------

    manifest = {
        "project": "M5 Demand Forecasting",
        "selected_series": expected_series,
        "forecast_horizon": FORECAST_HORIZON,
        "splits": {
            "train": {
                "start_day": 1,
                "end_day": TRAIN_END,
                "days": TRAIN_END,
                "rows": len(train),
            },
            "validation": {
                "start_day": VALIDATION_START,
                "end_day": VALIDATION_END,
                "days": FORECAST_HORIZON,
                "rows": len(validation),
            },
            "test": {
                "start_day": TEST_START,
                "end_day": TEST_END,
                "days": FORECAST_HORIZON,
                "rows": len(test),
            },
        },
        "feature_policy": {
            "time_features": True,
            "lag_features": False,
            "rolling_features": False,
            "reason": (
                "Lag and rolling features will be generated "
                "inside model-specific forecasting pipelines "
                "to prevent multi-step leakage."
            ),
        },
    }

    manifest_path = (
        PROCESSED_DIR / "preprocessing_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    # ----------------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("PREPROCESSING SUMMARY")
    print("=" * 72)

    print(f"Selected series : {expected_series}")
    print()

    print(
        f"TRAIN       : {len(train):>6,} rows "
        f"| d_1 -> d_{TRAIN_END}"
    )

    print(
        f"VALIDATION  : {len(validation):>6,} rows "
        f"| d_{VALIDATION_START} -> d_{VALIDATION_END}"
    )

    print(
        f"TEST        : {len(test):>6,} rows "
        f"| d_{TEST_START} -> d_{TEST_END}"
    )

    print()
    print("Files created:")

    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")
    print(f"  {all_path}")
    print(f"  {manifest_path}")

    print()
    print("PASS | chronological split")
    print("PASS | series completeness")
    print("PASS | duplicate series/day check")
    print("PASS | calendar join")
    print("PASS | leakage-safe time features")

    print("=" * 72)
    print("PREPROCESSING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()