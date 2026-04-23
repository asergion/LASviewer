from __future__ import annotations

import pandas as pd


def numeric_curve_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns:
        if column == "__INDEX__":
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        valid = int(series.notna().sum())

        if valid == 0:
            continue

        total = int(series.shape[0])

        rows.append(
            {
                "curve": column,
                "total_values": total,
                "valid_values": valid,
                "null_values": total - valid,
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    return pd.DataFrame(rows).sort_values(by="curve").reset_index(drop=True)


def curves_with_valid_data(stats_df: pd.DataFrame) -> list[str]:
    if stats_df.empty:
        return []
    return stats_df["curve"].tolist()


def compare_curves_between_wells(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    curve_name: str,
) -> pd.DataFrame:
    left = pd.DataFrame({
        "index": pd.to_numeric(df1["__INDEX__"], errors="coerce"),
        "well_1": pd.to_numeric(df1[curve_name], errors="coerce"),
    })

    right = pd.DataFrame({
        "index": pd.to_numeric(df2["__INDEX__"], errors="coerce"),
        "well_2": pd.to_numeric(df2[curve_name], errors="coerce"),
    })

    merged = pd.merge(left, right, on="index", how="inner")
    return merged.sort_values("index").reset_index(drop=True)