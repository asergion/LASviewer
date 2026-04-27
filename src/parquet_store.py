from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_metadata(metadata_path: str | Path) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_selected_curves(
    parquet_path: str | Path,
    fixed_variable: str,
    selected_variables: list[str],
) -> pd.DataFrame:
    columns = [fixed_variable] + selected_variables

    if "__INDEX_DATETIME__" in columns and "__INDEX__" not in columns:
        columns.append("__INDEX__")

    columns = list(dict.fromkeys(columns))

    return pd.read_parquet(
        parquet_path,
        columns=columns,
        engine="pyarrow",
    )


def downsample_df(df: pd.DataFrame, max_points: int = 10_000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df

    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()