from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.analysis import curves_with_valid_data, numeric_curve_stats
from src.las_parser import (
    curves_metadata,
    extract_header_info,
    las_to_filtered_dataframe,
    read_las,
)


def detect_unix_time_unit(series: pd.Series) -> str | None:
    valores = pd.to_numeric(series, errors="coerce").dropna()

    if valores.empty:
        return None

    mediana = float(valores.median())

    if mediana > 1e15:
        return "ns"
    if mediana > 1e12:
        return "ms"
    if mediana > 1e9:
        return "s"

    return None


def normalize_time_index(df: pd.DataFrame, index_type: str) -> tuple[pd.DataFrame, str | None]:
    if index_type != "tempo":
        return df, None

    time_unit = detect_unix_time_unit(df["__INDEX__"])

    if time_unit is None:
        return df, None

    df = df.copy()
    df["__INDEX_DATETIME__"] = pd.to_datetime(
        pd.to_numeric(df["__INDEX__"], errors="coerce"),
        unit=time_unit,
        errors="coerce",
        utc=True,
    )

    return df, time_unit


def index_las_file(las_path: str | Path, output_dir: str | Path) -> dict:
    las_path = Path(las_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las = read_las(las_path)

    header = extract_header_info(las, las_path.name)
    metadata_df = curves_metadata(las, str(las_path))

    df = las_to_filtered_dataframe(las, header.null_value)
    df, time_unit = normalize_time_index(df, header.index_type)

    stats_df = numeric_curve_stats(df)
    valid_curves = curves_with_valid_data(stats_df)

    index_series = pd.to_numeric(df["__INDEX__"], errors="coerce").dropna()

    index_min = float(index_series.min()) if not index_series.empty else None
    index_max = float(index_series.max()) if not index_series.empty else None

    md_min = None
    md_max = None

    if "MD" in df.columns:
        md_series = pd.to_numeric(df["MD"], errors="coerce").dropna()

        if not md_series.empty:
            md_min = float(md_series.min())
            md_max = float(md_series.max())

    tempo_min = None
    tempo_max = None

    if "__INDEX_DATETIME__" in df.columns:
        tempo_min_value = df["__INDEX_DATETIME__"].min()
        tempo_max_value = df["__INDEX_DATETIME__"].max()

        tempo_min = tempo_min_value.isoformat() if pd.notna(tempo_min_value) else None
        tempo_max = tempo_max_value.isoformat() if pd.notna(tempo_max_value) else None

    parquet_path = output_dir / f"{las_path.stem}.parquet"
    metadata_path = output_dir / f"{las_path.stem}.metadata.json"

    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    metadata = {
        "source_file": las_path.name,
        "source_absolute_path": str(las_path.resolve()),
        "parquet_file": parquet_path.name,
        "parquet_absolute_path": str(parquet_path.resolve()),
        "metadata_file": metadata_path.name,
        "metadata_absolute_path": str(metadata_path.resolve()),
        "header": asdict(header),
        "time_unit": time_unit,
        "curves_metadata": metadata_df.to_dict(orient="records"),
        "stats": stats_df.to_dict(orient="records"),
        "valid_curves": valid_curves,
        "columns": list(df.columns),
        "total_records": int(len(df)),
        "index_min": index_min,
        "index_max": index_max,
        "md_min": md_min,
        "md_max": md_max,
        "tempo_min": tempo_min,
        "tempo_max": tempo_max,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)

    return metadata
