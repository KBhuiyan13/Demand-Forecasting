"""
Representative M5 series selection.

Selects item-store series representing different demand behaviors:
- high volume
- low/intermittent demand
- typical demand
- high volatility
- strong weekly seasonality
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


ID_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def load_sales():
    """Load the M5 validation sales table."""
    path = RAW_DIR / "sales_train_validation.csv"
    return pd.read_csv(path)


def calculate_series_statistics(sales):
    """Calculate demand statistics for every item-store series."""

    day_cols = [
        column for column in sales.columns
        if column.startswith("d_")
    ]

    values = sales[day_cols]

    stats = sales[ID_COLS].copy()

    stats["total_sales"] = values.sum(axis=1)
    stats["mean_sales"] = values.mean(axis=1)
    stats["median_sales"] = values.median(axis=1)
    stats["std_sales"] = values.std(axis=1)

    stats["zero_days"] = (values == 0).sum(axis=1)

    stats["zero_ratio"] = (
        stats["zero_days"] / len(day_cols)
    )

    stats["cv"] = (
        stats["std_sales"] /
        stats["mean_sales"].replace(0, np.nan)
    )

    return stats


def select_representative_series(stats):
    """Select representative demand patterns."""

    selections = {}

    # Highest-volume series
    selections["high_volume"] = (
        stats
        .sort_values("total_sales", ascending=False)
        .head(3)
    )

    # Typical series: closest to median total demand
    median_total = stats["total_sales"].median()

    typical_idx = (
        (stats["total_sales"] - median_total)
        .abs()
        .sort_values()
        .head(3)
        .index
    )

    selections["typical"] = stats.loc[typical_idx]

    # Highly intermittent
    selections["intermittent"] = (
        stats[
            stats["zero_ratio"] >= 0.50
        ]
        .sort_values(
            ["mean_sales", "total_sales"],
            ascending=False
        )
        .head(3)
    )

    # Highly volatile
    selections["volatile"] = (
        stats[
            stats["mean_sales"] > 0
        ]
        .sort_values("cv", ascending=False)
        .head(3)
    )

    # Low-volume
    selections["low_volume"] = (
        stats[
            stats["total_sales"] > 0
        ]
        .sort_values("total_sales")
        .head(3)
    )

    return selections


def main():

    print("=" * 70)
    print("M5 REPRESENTATIVE SERIES SELECTION")
    print("=" * 70)

    sales = load_sales()

    print(f"Sales shape: {sales.shape}")

    stats = calculate_series_statistics(sales)

    print("\nSeries statistics calculated.")

    selections = select_representative_series(stats)

    for name, selected in selections.items():

        print("\n" + "-" * 70)
        print(name.upper())
        print("-" * 70)

        print(
            selected[
                [
                    "id",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "total_sales",
                    "mean_sales",
                    "zero_ratio",
                    "cv",
                ]
            ].to_string(index=False)
        )

    # Combine selected series
    selected_ids = []

    for selected in selections.values():
        selected_ids.extend(selected["id"].tolist())

    selected_ids = list(dict.fromkeys(selected_ids))

    selected_series = sales[
        sales["id"].isin(selected_ids)
    ].copy()

    output_path = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "representative_series.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_series.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)

    print(f"Selected series: {len(selected_series)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()