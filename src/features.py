"""
M5 Demand Forecasting
Feature Engineering

Creates leakage-safe lag and rolling features.

IMPORTANT:
- `series_id` identifies the underlying time series.
- `id` identifies the individual row/forecast record.
- Validation/evaluation suffixes in `id` must NOT create separate series.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = PROCESSED_DIR / "features"

TRAIN_FILE = PROCESSED_DIR / "forecast_train.csv"
VALIDATION_FILE = PROCESSED_DIR / "forecast_validation.csv"
TEST_FILE = PROCESSED_DIR / "forecast_test.csv"

SERIES_COLUMN = "series_id"
TARGET_COLUMN = "sales"
TIME_COLUMN = "date"

LAG_WINDOWS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():

    print("Loading training data...")
    train = pd.read_csv(TRAIN_FILE)

    print("Loading validation data...")
    validation = pd.read_csv(VALIDATION_FILE)

    print("Loading test data...")
    test = pd.read_csv(TEST_FILE)

    for df in [train, validation, test]:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    return train, validation, test


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_input_schema(df):

    required_columns = [
        "id",
        "series_id",
        "date",
        "sales",
        "split",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ============================================================================
# FEATURE CREATION
# ============================================================================

def create_features(df):

    df = df.copy()

    validate_input_schema(df)

    # ------------------------------------------------------------------------
    # Sort by TRUE series identifier
    # ------------------------------------------------------------------------

    df = (
        df
        .sort_values(
            [SERIES_COLUMN, TIME_COLUMN]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------------
    # Existing calendar features
    # ------------------------------------------------------------------------

    if "day_of_week" not in df.columns:
        df["day_of_week"] = (
            df[TIME_COLUMN].dt.dayofweek
        )

    if "day_of_month" not in df.columns:
        df["day_of_month"] = (
            df[TIME_COLUMN].dt.day
        )

    if "week_of_year" not in df.columns:
        df["week_of_year"] = (
            df[TIME_COLUMN]
            .dt.isocalendar()
            .week
            .astype(int)
        )

    if "month" not in df.columns:
        df["month"] = (
            df[TIME_COLUMN].dt.month
        )

    if "year" not in df.columns:
        df["year"] = (
            df[TIME_COLUMN].dt.year
        )

    if "quarter" not in df.columns:
        df["quarter"] = (
            df[TIME_COLUMN].dt.quarter
        )

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

    # ------------------------------------------------------------------------
    # Group by TRUE time series
    # ------------------------------------------------------------------------

    grouped = (
        df
        .groupby(
            SERIES_COLUMN,
            sort=False
        )[TARGET_COLUMN]
    )

    # ------------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------------

    print("Creating lag features...")

    for lag in LAG_WINDOWS:

        df[f"lag_{lag}"] = grouped.shift(lag)

        print(
            f"  Created lag_{lag}"
        )

    # ------------------------------------------------------------------------
    # Shifted sales
    #
    # shift(1) ensures the current day's sales are never used.
    # ------------------------------------------------------------------------

    shifted_sales = grouped.shift(1)

    # ------------------------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------------------------

    print("Creating rolling features...")

    for window in ROLLING_WINDOWS:

        rolling_group = shifted_sales.groupby(
            df[SERIES_COLUMN],
            sort=False
        )

        df[f"rolling_mean_{window}"] = (
            rolling_group
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window
                ).mean()
            )
        )

        df[f"rolling_std_{window}"] = (
            rolling_group
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window
                ).std()
            )
        )

        print(
            f"  Created rolling_mean_{window} "
            f"and rolling_std_{window}"
        )

    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_features(df, dataset_name):

    print()
    print(
        f"Validating {dataset_name} features..."
    )

    required_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
    ]

    required_columns = [
        "id",
        "series_id",
        "date",
        "sales",
    ] + required_features

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name}: missing columns: {missing}"
        )

    # ------------------------------------------------------------------------
    # Chronological ordering
    # ------------------------------------------------------------------------

    expected = (
        df
        .sort_values(
            [SERIES_COLUMN, TIME_COLUMN]
        )
        .reset_index(drop=True)
    )

    actual = df.reset_index(drop=True)

    if not expected[
        [SERIES_COLUMN, TIME_COLUMN]
    ].equals(
        actual[
            [SERIES_COLUMN, TIME_COLUMN]
        ]
    ):
        raise ValueError(
            f"{dataset_name}: chronological ordering failed."
        )

    print(
        f"PASS | {dataset_name}: chronological ordering verified"
    )

    # ------------------------------------------------------------------------
    # Duplicate series/date check
    # ------------------------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            SERIES_COLUMN,
            TIME_COLUMN
        ]
    ).sum()

    if duplicates > 0:

        raise ValueError(
            f"{dataset_name}: found "
            f"{duplicates} duplicate "
            f"series/date combinations."
        )

    print(
        f"PASS | {dataset_name}: duplicate check passed"
    )

    print(
        f"PASS | {dataset_name}: required features present"
    )


# ============================================================================
# TRAINING HISTORY FILTER
# ============================================================================

def prepare_training_data(df):

    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
    ]

    before = len(df)

    df = (
        df
        .dropna(
            subset=feature_columns
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    print(
        f"Training history filter removed "
        f"{removed:,} rows."
    )

    return df


# ============================================================================
# FORECAST DATA VALIDATION
# ============================================================================

def validate_forecast_history(
    df,
    dataset_name
):

    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
    ]

    missing_counts = (
        df[feature_columns]
        .isna()
        .sum()
    )

    total_missing = (
        missing_counts.sum()
    )

    if total_missing > 0:

        print()
        print(
            f"WARNING | {dataset_name} contains "
            f"{total_missing} missing feature values."
        )

        print(
            missing_counts[
                missing_counts > 0
            ].to_string()
        )

        raise ValueError(
            f"{dataset_name} has incomplete "
            f"forecasting history."
        )

    print(
        f"PASS | {dataset_name}: "
        f"all forecasting features available"
    )


# ============================================================================
# SAVE
# ============================================================================

def save_dataset(df, filename):

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FEATURE_DIR / filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print(
        "M5 DEMAND FORECASTING - FEATURE ENGINEERING"
    )
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    train, validation, test = load_data()

    print()
    print(
        f"Training rows:   {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows:       {len(test):,}"
    )

    # ------------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------------

    print()
    print("Validating source schema...")

    for df in [train, validation, test]:

        validate_input_schema(df)

    print(
        "PASS | source schema validated"
    )

    # ------------------------------------------------------------------------
    # Combine all periods
    # ------------------------------------------------------------------------

    all_data = pd.concat(
        [
            train,
            validation,
            test
        ],
        ignore_index=True
    )

    all_data = (
        all_data
        .sort_values(
            [SERIES_COLUMN, TIME_COLUMN]
        )
        .reset_index(drop=True)
    )

    print()
    print(
        f"Combined rows: {len(all_data):,}"
    )

    print()
    print(
        "Unique time series:",
        all_data[SERIES_COLUMN].nunique()
    )

    # ------------------------------------------------------------------------
    # Verify each series has 1941 observations
    # ------------------------------------------------------------------------

    series_counts = (
        all_data
        .groupby(SERIES_COLUMN)
        .size()
    )

    print(
        "Series observations:",
        series_counts.unique().tolist()
    )

    if not (series_counts == 1941).all():

        raise ValueError(
            "Unexpected number of observations "
            "for one or more series."
        )

    print(
        "PASS | every series contains "
        "the expected 1,941 observations"
    )

    # ------------------------------------------------------------------------
    # Create features
    # ------------------------------------------------------------------------

    print()
    print(
        "Creating time-series features..."
    )

    all_features = create_features(
        all_data
    )

    # ------------------------------------------------------------------------
    # Split using split column
    # ------------------------------------------------------------------------

    train_features = all_features[
        all_features["split"] == "train"
    ].copy()

    validation_features = all_features[
        all_features["split"] == "validation"
    ].copy()

    test_features = all_features[
        all_features["split"] == "test"
    ].copy()

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    validate_features(
        train_features,
        "training"
    )

    validate_features(
        validation_features,
        "validation"
    )

    validate_features(
        test_features,
        "test"
    )

    # ------------------------------------------------------------------------
    # Training requires complete history
    # ------------------------------------------------------------------------

    print()
    print(
        "Preparing training data..."
    )

    train_features = prepare_training_data(
        train_features
    )

    # ------------------------------------------------------------------------
    # Validation/test MUST retain all forecast rows
    # ------------------------------------------------------------------------

    print()
    print(
        "Checking validation history..."
    )

    validate_forecast_history(
        validation_features,
        "validation"
    )

    print()
    print(
        "Checking test history..."
    )

    validate_forecast_history(
        test_features,
        "test"
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    print()
    print(
        "Saving feature datasets..."
    )

    save_dataset(
        train_features,
        "train_features.csv"
    )

    save_dataset(
        validation_features,
        "validation_features.csv"
    )

    save_dataset(
        test_features,
        "test_features.csv"
    )

    # ------------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------------

    manifest = {

        "target": TARGET_COLUMN,

        "series_identifier": SERIES_COLUMN,

        "row_identifier": "id",

        "time_column": TIME_COLUMN,

        "lag_features": [
            f"lag_{lag}"
            for lag in LAG_WINDOWS
        ],

        "rolling_features": (

            [
                f"rolling_mean_{window}"
                for window in ROLLING_WINDOWS
            ]

            +

            [
                f"rolling_std_{window}"
                for window in ROLLING_WINDOWS
            ]
        ),

        "calendar_features": [
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "month",
            "year",
            "quarter",
            "is_weekend",
        ],

        "preprocessing_features": [
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_event",
            "snap",
        ],

        "maximum_history_required": 28,

        "leakage_protection": {

            "rolling_features_shifted": True,

            "future_sales_used": False,

            "validation_future_sales_used": False,

            "test_future_sales_used": False,
        },

        "split_sizes": {

            "train": len(train_features),

            "validation": len(
                validation_features
            ),

            "test": len(
                test_features
            ),
        },
    }

    manifest_path = (
        FEATURE_DIR /
        "feature_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "FEATURE ENGINEERING SUMMARY"
    )
    print("=" * 72)

    print(
        f"Training rows:   "
        f"{len(train_features):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_features):,}"
    )

    print(
        f"Test rows:       "
        f"{len(test_features):,}"
    )

    print()
    print(
        "Unique series:",
        train_features[SERIES_COLUMN].nunique()
    )

    print()
    print("Forecasting features:")

    for feature in (
        manifest["lag_features"]
        +
        manifest["rolling_features"]
    ):

        print(
            f"  - {feature}"
        )

    print()
    print(
        "Manifest:",
        manifest_path
    )

    print()
    print("=" * 72)
    print(
        "PASS | FEATURE ENGINEERING COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()