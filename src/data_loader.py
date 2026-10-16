"""
M5 Forecasting - Accuracy
Data loading and validation utilities.

This module intentionally uses chunked reads for the large M5 CSV files so
validation does not require loading the entire dataset into memory at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class ValidationResult:
    """Container for validation results."""

    file: str
    exists: bool
    rows: Optional[int] = None
    columns: Optional[int] = None
    issues: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.issues is None:
            self.issues = []

    @property
    def passed(self) -> bool:
        return self.exists and not self.issues


def get_data_paths(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Path]:
    """Return paths to the M5 files."""
    return {
        "calendar": raw_dir / "calendar.csv",
        "sales_validation": raw_dir / "sales_train_validation.csv",
        "sales_evaluation": raw_dir / "sales_train_evaluation.csv",
        "sell_prices": raw_dir / "sell_prices.csv",
        "sample_submission": raw_dir / "sample_submission.csv",
    }


def load_calendar(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load the small calendar table."""
    path = get_data_paths(raw_dir)["calendar"]
    return pd.read_csv(path, parse_dates=["date"])


def load_sales(
    raw_dir: Path = RAW_DATA_DIR,
    validation: bool = True,
) -> pd.DataFrame:
    """
    Load a sales file.

    This is suitable for controlled analysis, but for the full M5 workflow
    prefer chunked reads or selective loading because the file is large.
    """
    key = "sales_validation" if validation else "sales_evaluation"
    return pd.read_csv(get_data_paths(raw_dir)[key])


def load_sell_prices(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load the sell-price table."""
    return pd.read_csv(get_data_paths(raw_dir)["sell_prices"])


def _validate_required_columns(
    actual: Iterable[str],
    required: Iterable[str],
) -> list[str]:
    actual_set = set(actual)
    return [column for column in required if column not in actual_set]


def _expected_day_columns(last_day: int) -> list[str]:
    return [f"d_{i}" for i in range(1, last_day + 1)]


def validate_calendar(path: Path) -> ValidationResult:
    """Validate calendar.csv."""
    result = ValidationResult(file=path.name, exists=path.exists())

    if not result.exists:
        result.issues.append("File does not exist.")
        return result

    df = pd.read_csv(path, parse_dates=["date"])

    required = [
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "d",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    missing = _validate_required_columns(df.columns, required)
    if missing:
        result.issues.append(f"Missing columns: {missing}")

    result.rows, result.columns = df.shape

    if df["date"].isna().any():
        result.issues.append("Missing dates found.")

    if not df["date"].is_monotonic_increasing:
        result.issues.append("Dates are not sorted in increasing order.")

    if df["date"].duplicated().any():
        result.issues.append("Duplicate calendar dates found.")

    if df["d"].duplicated().any():
        result.issues.append("Duplicate M5 day identifiers found.")

    expected_d = _expected_day_columns(len(df))
    if df["d"].tolist() != expected_d:
        result.issues.append(
            "Calendar day identifiers are not exactly d_1 ... d_N."
        )

    for column in ["snap_CA", "snap_TX", "snap_WI"]:
        if column in df and not df[column].isin([0, 1]).all():
            result.issues.append(f"{column} contains values other than 0/1.")

    return result


def validate_sales(
    path: Path,
    expected_last_day: int,
    chunksize: int = 5_000,
) -> ValidationResult:
    """
    Validate a sales file using chunks.

    Checks:
      - schema
      - duplicate series IDs
      - missing values
      - negative demand
      - integer-like demand
      - all-zero series
      - row count
    """
    result = ValidationResult(file=path.name, exists=path.exists())

    if not result.exists:
        result.issues.append("File does not exist.")
        return result

    required_id_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]
    expected_days = _expected_day_columns(expected_last_day)

    first_chunk = pd.read_csv(path, nrows=5)
    result.columns = len(first_chunk.columns)

    missing_ids = _validate_required_columns(first_chunk.columns, required_id_columns)
    if missing_ids:
        result.issues.append(f"Missing ID columns: {missing_ids}")

    missing_days = _validate_required_columns(first_chunk.columns, expected_days)
    if missing_days:
        result.issues.append(
            f"Missing/incorrect number of day columns; first missing examples: "
            f"{missing_days[:10]}"
        )

    actual_day_columns = [
        column for column in first_chunk.columns if str(column).startswith("d_")
    ]
    if len(actual_day_columns) != expected_last_day:
        result.issues.append(
            f"Expected {expected_last_day} daily columns, found "
            f"{len(actual_day_columns)}."
        )

    seen_ids: set[str] = set()
    total_rows = 0
    null_count = 0
    negative_count = 0
    non_integer_count = 0
    all_zero_series = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)

        null_count += int(chunk.isna().sum().sum())

        ids = chunk["id"].astype(str)
        duplicate_in_chunk = ids[ids.duplicated()].tolist()
        if duplicate_in_chunk:
            result.issues.append("Duplicate IDs found within a chunk.")

        duplicate_across_chunks = set(ids).intersection(seen_ids)
        if duplicate_across_chunks:
            result.issues.append(
                "Duplicate IDs found across chunks."
            )
        seen_ids.update(ids)

        values = chunk[actual_day_columns]

        negative_count += int((values < 0).sum().sum())

        # M5 demand is integer-valued. Use a numeric comparison that tolerates
        # normal pandas integer/float representation.
        non_integer_count += int(
            ((values.notna()) & ((values % 1) != 0)).sum().sum()
        )

        all_zero_series += int((values.sum(axis=1) == 0).sum())

    result.rows = total_rows

    if null_count:
        result.issues.append(f"Found {null_count:,} missing cells.")

    if negative_count:
        result.issues.append(f"Found {negative_count:,} negative demand values.")

    if non_integer_count:
        result.issues.append(
            f"Found {non_integer_count:,} non-integer demand values."
        )

    if len(seen_ids) != total_rows:
        result.issues.append(
            f"Unique IDs ({len(seen_ids):,}) do not equal row count "
            f"({total_rows:,})."
        )

    print(f"\n{path.name}")
    print(f"  Rows:                 {total_rows:,}")
    print(f"  Columns:              {result.columns:,}")
    print(f"  Unique series IDs:    {len(seen_ids):,}")
    print(f"  Missing cells:        {null_count:,}")
    print(f"  Negative demand:      {negative_count:,}")
    print(f"  Non-integer demand:   {non_integer_count:,}")
    print(f"  All-zero series:      {all_zero_series:,}")

    return result


def validate_sell_prices(
    path: Path,
    chunksize: int = 100_000,
) -> ValidationResult:
    """Validate sell_prices.csv using chunks."""
    result = ValidationResult(file=path.name, exists=path.exists())

    if not result.exists:
        result.issues.append("File does not exist.")
        return result

    first_chunk = pd.read_csv(path, nrows=5)
    required = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing = _validate_required_columns(first_chunk.columns, required)

    if missing:
        result.issues.append(f"Missing columns: {missing}")

    result.columns = len(first_chunk.columns)

    total_rows = 0
    null_count = 0
    negative_count = 0
    duplicate_key_count = 0
    min_price = float("inf")
    max_price = float("-inf")

    seen_keys: set[tuple[str, str, int]] = set()
    stores: set[str] = set()
    items: set[str] = set()

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)
        null_count += int(chunk.isna().sum().sum())
        negative_count += int((chunk["sell_price"] < 0).sum())

        min_price = min(min_price, float(chunk["sell_price"].min()))
        max_price = max(max_price, float(chunk["sell_price"].max()))

        stores.update(chunk["store_id"].astype(str).unique())
        items.update(chunk["item_id"].astype(str).unique())

        keys = list(
            zip(
                chunk["store_id"].astype(str),
                chunk["item_id"].astype(str),
                chunk["wm_yr_wk"].astype(int),
            )
        )
        duplicate_key_count += len(keys) - len(set(keys))

        cross_duplicates = sum(key in seen_keys for key in set(keys))
        duplicate_key_count += cross_duplicates
        seen_keys.update(keys)

    result.rows = total_rows

    if null_count:
        result.issues.append(f"Found {null_count:,} missing cells.")

    if negative_count:
        result.issues.append(
            f"Found {negative_count:,} negative sell prices."
        )

    if duplicate_key_count:
        result.issues.append(
            f"Found {duplicate_key_count:,} duplicate "
            "(store_id, item_id, wm_yr_wk) keys."
        )

    print(f"\n{path.name}")
    print(f"  Rows:                 {total_rows:,}")
    print(f"  Columns:              {result.columns:,}")
    print(f"  Stores:               {len(stores):,}")
    print(f"  Items:                {len(items):,}")
    print(f"  Missing cells:        {null_count:,}")
    print(f"  Negative prices:      {negative_count:,}")
    print(f"  Duplicate price keys: {duplicate_key_count:,}")
    print(f"  Price range:          {min_price:.2f} - {max_price:.2f}")

    return result


def validate_sample_submission(
    path: Path,
    expected_series: int = 30_490,
    horizon: int = 28,
) -> ValidationResult:
    """Validate sample_submission.csv."""
    result = ValidationResult(file=path.name, exists=path.exists())

    if not result.exists:
        result.issues.append("File does not exist.")
        return result

    df = pd.read_csv(path)
    result.rows, result.columns = df.shape

    expected_forecast_columns = [f"F{i}" for i in range(1, horizon + 1)]
    expected_columns = ["id"] + expected_forecast_columns

    if list(df.columns) != expected_columns:
        result.issues.append(
            "Submission columns do not match id + F1 ... F28."
        )

    expected_rows = expected_series * 2
    if len(df) != expected_rows:
        result.issues.append(
            f"Expected {expected_rows:,} submission rows "
            f"(validation + evaluation), found {len(df):,}."
        )

    if df["id"].duplicated().any():
        result.issues.append("Duplicate submission IDs found.")

    print(f"\n{path.name}")
    print(f"  Rows:                 {result.rows:,}")
    print(f"  Columns:              {result.columns:,}")
    print(f"  Forecast horizon:     {horizon} days")
    print(f"  Duplicate IDs:        {int(df['id'].duplicated().sum()):,}")

    return result


def validate_m5_dataset(raw_dir: Path = RAW_DATA_DIR) -> dict[str, ValidationResult]:
    """Run the complete M5 validation suite."""
    paths = get_data_paths(raw_dir)

    print("=" * 72)
    print("M5 FORECASTING DATA VALIDATION")
    print("=" * 72)
    print(f"Raw data directory: {raw_dir}")

    results: dict[str, ValidationResult] = {}

    results["calendar"] = validate_calendar(paths["calendar"])

    # sales_train_validation ends at d_1913.
    results["sales_validation"] = validate_sales(
        paths["sales_validation"],
        expected_last_day=1913,
    )

    # sales_train_evaluation contains the additional 28 observed days.
    results["sales_evaluation"] = validate_sales(
        paths["sales_evaluation"],
        expected_last_day=1941,
    )

    results["sell_prices"] = validate_sell_prices(paths["sell_prices"])

    results["sample_submission"] = validate_sample_submission(
        paths["sample_submission"]
    )

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    overall_passed = True

    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status:>4} | {name:<20} | "
            f"rows={result.rows if result.rows is not None else 'N/A'} | "
            f"columns={result.columns if result.columns is not None else 'N/A'}"
        )

        if result.issues:
            overall_passed = False
            for issue in result.issues:
                print(f"       - {issue}")

    print("-" * 72)
    print("OVERALL:", "PASS" if overall_passed else "FAIL")

    return results


if __name__ == "__main__":
    validate_m5_dataset()
