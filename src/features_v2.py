"""
M5 Demand Forecasting - Feature Engineering V2

Extends the original feature set with:
    - Short-term lags
    - Longer seasonal lags
    - Additional rolling statistics
    - Demand trend features
    - Recent-demand change features

IMPORTANT:
All lag and rolling features are calculated using only
strictly previous observations.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = INPUT_DIR / "features_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = INPUT_DIR / "forecast_train.csv"
VALIDATION_FILE = INPUT_DIR / "forecast_validation.csv"
TEST_FILE = INPUT_DIR / "forecast_test.csv"


TARGET_COLUMN = "sales"
SERIES_COLUMN = "series_id"
DATE_COLUMN = "date"


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

SHORT_LAGS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    14,
    21,
    28,
    35,
    56,
]


ROLLING_WINDOWS = [
    3,
    5,
    7,
    14,
    21,
    28,
    56,
]


CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "quarter",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_event",
    "snap",
]


STATIC_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


# ============================================================================
# HELPERS
# ============================================================================

def create_lag_features(df):
    """Create lag features grouped by underlying series."""

    print("Creating lag features...")

    grouped_sales = df.groupby(
        SERIES_COLUMN,
        sort=False
    )[TARGET_COLUMN]

    for lag in SHORT_LAGS:

        column = f"lag_{lag}"

        df[column] = grouped_sales.shift(lag)

        print(f"  Created {column}")

    return df


def create_rolling_features(df):
    """Create leakage-safe rolling means and standard deviations."""

    print("Creating rolling features...")

    # Shift by one day before calculating rolling statistics.
    # This prevents the current day's sales from entering its features.
    shifted_sales = (
        df.groupby(SERIES_COLUMN, sort=False)[TARGET_COLUMN]
        .shift(1)
    )

    for window in ROLLING_WINDOWS:
        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"

        grouped_rolling = (
            shifted_sales
            .groupby(df[SERIES_COLUMN], sort=False)
            .rolling(window=window, min_periods=window)
        )

        df[mean_column] = (
            grouped_rolling.mean()
            .reset_index(level=0, drop=True)
        )

        df[std_column] = (
            grouped_rolling.std()
            .reset_index(level=0, drop=True)
        )

        print(f"  Created {mean_column} and {std_column}")

    return df


def create_trend_features(df):
    """Create leakage-safe demand trend and relative-change features."""

    print("Creating trend features...")

    # Recent versus longer-term demand
    df["trend_mean_7_28"] = (
        df["rolling_mean_7"] - df["rolling_mean_28"]
    )

    df["trend_mean_7_56"] = (
        df["rolling_mean_7"] - df["rolling_mean_56"]
    )

    # Ratios of rolling means.
    # Use zero when the denominator is zero.
    df["ratio_mean_7_28"] = np.divide(
        df["rolling_mean_7"].to_numpy(dtype=float),
        df["rolling_mean_28"].to_numpy(dtype=float),
        out=np.zeros(len(df), dtype=float),
        where=df["rolling_mean_28"].to_numpy(dtype=float) > 0,
    )

    df["ratio_mean_7_56"] = np.divide(
        df["rolling_mean_7"].to_numpy(dtype=float),
        df["rolling_mean_56"].to_numpy(dtype=float),
        out=np.zeros(len(df), dtype=float),
        where=df["rolling_mean_56"].to_numpy(dtype=float) > 0,
    )

    # Absolute demand changes
    df["change_lag_1_7"] = df["lag_1"] - df["lag_7"]
    df["change_lag_7_14"] = df["lag_7"] - df["lag_14"]
    df["change_lag_14_28"] = df["lag_14"] - df["lag_28"]

    # Relative demand changes
    df["ratio_lag_1_7"] = np.divide(
        df["lag_1"].to_numpy(dtype=float),
        df["lag_7"].to_numpy(dtype=float),
        out=np.zeros(len(df), dtype=float),
        where=df["lag_7"].to_numpy(dtype=float) > 0,
    )

    df["ratio_lag_7_14"] = np.divide(
        df["lag_7"].to_numpy(dtype=float),
        df["lag_14"].to_numpy(dtype=float),
        out=np.zeros(len(df), dtype=float),
        where=df["lag_14"].to_numpy(dtype=float) > 0,
    )

    print("  Created trend/change features")

    return df



# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print("M5 DEMAND FORECASTING - FEATURE ENGINEERING V2")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------

    print("Loading training data...")
    train_df = pd.read_csv(
        TRAIN_FILE,
        parse_dates=[DATE_COLUMN],
    )

    print("Loading validation data...")
    validation_df = pd.read_csv(
        VALIDATION_FILE,
        parse_dates=[DATE_COLUMN],
    )

    print("Loading test data...")
    test_df = pd.read_csv(
        TEST_FILE,
        parse_dates=[DATE_COLUMN],
    )

    print()

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows:       {len(test_df):,}")

    # ----------------------------------------------------------------------
    # Validate source schema
    # ----------------------------------------------------------------------

    print()
    print("Validating source schema...")

    required_columns = [
        "id",
        SERIES_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
        "split",
    ]

    for df_name, df in [
        ("training", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{df_name}: missing columns {missing}"
            )

    print("PASS | source schema validated")

    # ----------------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------------

    combined = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [SERIES_COLUMN, DATE_COLUMN]
    ).reset_index(drop=True)

    print()
    print(
        f"Combined rows: {len(combined):,}"
    )

    # ----------------------------------------------------------------------
    # Series validation
    # ----------------------------------------------------------------------

    series_counts = (
        combined
        .groupby(SERIES_COLUMN)
        .size()
    )

    print()
    print(
        f"Unique time series: {len(series_counts)}"
    )

    print(
        f"Series observations: "
        f"{sorted(series_counts.unique().tolist())}"
    )

    if not (series_counts == 1941).all():

        raise ValueError(
            "Not every series contains exactly 1,941 observations."
        )

    print(
        "PASS | every series contains the expected 1,941 observations"
    )

    # ----------------------------------------------------------------------
    # Feature creation
    # ----------------------------------------------------------------------

    print()
    print("Creating V2 time-series features...")

    combined = create_lag_features(combined)

    combined = create_rolling_features(combined)

    combined = create_trend_features(combined)

    # ----------------------------------------------------------------------
    # Calendar feature check
    # ----------------------------------------------------------------------

    missing_calendar = [
        feature
        for feature in CALENDAR_FEATURES
        if feature not in combined.columns
    ]

    if missing_calendar:

        raise ValueError(
            f"Missing calendar features: {missing_calendar}"
        )

    # ----------------------------------------------------------------------
    # Static feature check
    # ----------------------------------------------------------------------

    missing_static = [
        feature
        for feature in STATIC_FEATURES
        if feature not in combined.columns
    ]

    if missing_static:

        raise ValueError(
            f"Missing static features: {missing_static}"
        )

    # ----------------------------------------------------------------------
    # Split
    # ----------------------------------------------------------------------

    train_features = combined[
        combined["split"] == "train"
    ].copy()

    validation_features = combined[
        combined["split"] == "validation"
    ].copy()

    test_features = combined[
        combined["split"] == "test"
    ].copy()

    print()
    print("Validating feature datasets...")

    # ----------------------------------------------------------------------
    # Forecasting feature list
    # ----------------------------------------------------------------------

    lag_features = [
        f"lag_{lag}"
        for lag in SHORT_LAGS
    ]

    rolling_features = []

    for window in ROLLING_WINDOWS:

        rolling_features.extend(
            [
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
            ]
        )

    trend_features = [
        "trend_mean_7_28",
        "trend_mean_7_56",
        "ratio_mean_7_28",
        "ratio_mean_7_56",
        "change_lag_1_7",
        "change_lag_7_14",
        "change_lag_14_28",
        "ratio_lag_1_7",
        "ratio_lag_7_14",
    ]

    forecasting_features = (
        lag_features
        + rolling_features
        + trend_features
    )

    all_model_features = (
        forecasting_features
        + CALENDAR_FEATURES
        + STATIC_FEATURES
    )

    # ----------------------------------------------------------------------
    # Required feature validation
    # ----------------------------------------------------------------------

    for dataset_name, dataset in [
        ("training", train_features),
        ("validation", validation_features),
        ("test", test_features),
    ]:

        missing = [
            feature
            for feature in all_model_features
            if feature not in dataset.columns
        ]

        if missing:

            raise ValueError(
                f"{dataset_name}: missing features {missing}"
            )

        print(
            f"PASS | {dataset_name}: required features present"
        )

    # ----------------------------------------------------------------------
    # Training history
    # ----------------------------------------------------------------------

    print()
    print("Preparing training data...")

    before = len(train_features)

    train_features = train_features.dropna(
        subset=forecasting_features
    ).copy()

    removed = before - len(train_features)

    print(
        f"Training history filter removed {removed} rows."
    )

    # Expected:
    # 14 series × 56 maximum lag/history = 784 rows
    #
    # However, rolling windows are also present and the maximum
    # history requirement is 56 observations.
    expected_removed = 14 * 56

    if removed != expected_removed:

        raise ValueError(
            f"Unexpected training history removal: "
            f"{removed}; expected {expected_removed}"
        )

    # ----------------------------------------------------------------------
    # Validation / test history
    # ----------------------------------------------------------------------

    print()
    print("Checking validation history...")

    validation_missing = (
        validation_features[
            forecasting_features
        ]
        .isna()
        .sum()
        .sum()
    )

    if validation_missing != 0:

        raise ValueError(
            "Validation contains missing forecasting features."
        )

    print(
        "PASS | validation: all forecasting features available"
    )

    print()
    print("Checking test history...")

    test_missing = (
        test_features[
            forecasting_features
        ]
        .isna()
        .sum()
        .sum()
    )

    if test_missing != 0:

        raise ValueError(
            "Test contains missing forecasting features."
        )

    print(
        "PASS | test: all forecasting features available"
    )

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------

    print()
    print("Saving V2 feature datasets...")

    train_output = OUTPUT_DIR / "train_features.csv"
    validation_output = OUTPUT_DIR / "validation_features.csv"
    test_output = OUTPUT_DIR / "test_features.csv"

    train_features.to_csv(
        train_output,
        index=False,
    )

    validation_features.to_csv(
        validation_output,
        index=False,
    )

    test_features.to_csv(
        test_output,
        index=False,
    )

    print(f"Saved: {train_output}")
    print(f"Saved: {validation_output}")
    print(f"Saved: {test_output}")

    # ----------------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------------

    manifest = {
        "version": "v2",
        "target": TARGET_COLUMN,
        "series_column": SERIES_COLUMN,
        "time_column": DATE_COLUMN,
        "maximum_history": 56,
        "short_lags": SHORT_LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "forecasting_features": forecasting_features,
        "calendar_features": CALENDAR_FEATURES,
        "static_features": STATIC_FEATURES,
        "total_model_features": all_model_features,
        "leakage_protection": (
            "All lag features use shift(lag). "
            "Rolling statistics use shift(1) before rolling. "
            "Current target is never included."
        ),
    }

    manifest_file = (
        OUTPUT_DIR / "feature_manifest_v2.json"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FEATURE ENGINEERING V2 SUMMARY")
    print("=" * 72)

    print(
        f"Training rows:   {len(train_features):,}"
    )

    print(
        f"Validation rows: {len(validation_features):,}"
    )

    print(
        f"Test rows:       {len(test_features):,}"
    )

    print(
        f"Unique series:   "
        f"{combined[SERIES_COLUMN].nunique()}"
    )

    print()
    print(
        f"Lag features:       {len(lag_features)}"
    )

    print(
        f"Rolling features:   {len(rolling_features)}"
    )

    print(
        f"Trend features:     {len(trend_features)}"
    )

    print(
        f"Calendar features:  {len(CALENDAR_FEATURES)}"
    )

    print(
        f"Static features:    {len(STATIC_FEATURES)}"
    )

    print(
        f"Total model features: "
        f"{len(all_model_features)}"
    )

    print()
    print(
        f"Manifest: {manifest_file}"
    )

    print()
    print("=" * 72)
    print("PASS | FEATURE ENGINEERING V2 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()