from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_metadata(metadata_path: str | Path) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_parquet_path(
    parquet_path: str | Path,
    metadata_path: str | Path | None = None,
) -> Path:
    path = Path(parquet_path)

    if path.exists() and path.is_file():
        return path

    if metadata_path is not None:
        metadata_dir = Path(metadata_path)

        if metadata_dir.is_file():
            metadata_dir = metadata_dir.parent

        candidate = metadata_dir / path.name
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Arquivo Parquet não encontrado. "
        f"Caminho recebido: {parquet_path}"
    )


def read_selected_curves(
    parquet_path: str | Path,
    fixed_variable: str,
    selected_variables: list[str],
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    columns = [fixed_variable] + selected_variables

    if "__INDEX_DATETIME__" in columns and "__INDEX__" not in columns:
        columns.append("__INDEX__")

    columns = list(dict.fromkeys(columns))
    resolved_parquet_path = resolve_parquet_path(parquet_path, metadata_path)

    return pd.read_parquet(
        resolved_parquet_path,
        columns=columns,
        engine="pyarrow",
    )


def downsample_df(df: pd.DataFrame, max_points: int = 10_000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df

    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()
