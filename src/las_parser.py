from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd


TIME_KEYWORDS = {"TIME", "DATE", "DATETIME", "ETIM", "TIME_DATE", "TIMEINDEX"}
DEPTH_KEYWORDS = {"DEPT", "DEPTH", "MD", "TVD", "TVDSS"}

TIME_UNITS = {
    "S", "SEC", "SECOND", "SECONDS",
    "MS", "MSEC", "MILLISECOND", "MILLISECONDS",
    "MIN", "MINS", "MINUTE", "MINUTES",
    "H", "HR", "HOUR", "HOURS",
    "D", "DAY", "DAYS",
}

DEPTH_UNITS = {"M", "FT", "F", "METER", "METERS", "FEET"}


@dataclass
class HeaderInfo:
    file_name: str
    las_version: str | None
    company: str | None
    well_name: str | None
    field_name: str | None
    null_value: float | None
    index_curve: str | None
    index_unit: str | None
    index_type: str
    start: Any
    stop: Any
    step: Any
    total_records: int


def _safe_header_value(item: Any) -> Any:
    if item is None:
        return None
    value = getattr(item, "value", item)
    if value in (None, "", " "):
        return None
    return value


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_index_type(index_mnemonic: str | None, index_unit: str | None) -> str:
    mnemonic = (index_mnemonic or "").strip().upper()
    unit = (index_unit or "").strip().upper()

    if mnemonic in TIME_KEYWORDS or unit in TIME_UNITS:
        return "tempo"
    if mnemonic in DEPTH_KEYWORDS or unit in DEPTH_UNITS:
        return "profundidade"
    return "indeterminado"


def read_las(file_path: str | Path):
    return lasio.read(file_path)


def get_index_curve_info(las) -> tuple[str | None, str | None]:
    index_curve_obj = getattr(las, "index_curve", None)
    if index_curve_obj is not None:
        return (
            getattr(index_curve_obj, "mnemonic", None),
            getattr(index_curve_obj, "unit", None),
        )

    curves = getattr(las, "curves", None)
    if curves is not None and len(curves) > 0:
        first_curve = curves[0]
        return (
            getattr(first_curve, "mnemonic", None),
            getattr(first_curve, "unit", None),
        )

    return None, None


def extract_header_info(las, file_name: str) -> HeaderInfo:
    version = _safe_header_value(getattr(las.version, "VERS", None))
    company = _safe_header_value(getattr(las.well, "COMP", None))
    well_name = _safe_header_value(getattr(las.well, "WELL", None))
    field_name = _safe_header_value(getattr(las.well, "FLD", None))
    null_value = _to_float(_safe_header_value(getattr(las.well, "NULL", None)))

    index_curve, index_unit = get_index_curve_info(las)
    index_type = detect_index_type(index_curve, index_unit)

    start = _safe_header_value(getattr(las.well, "STRT", None))
    stop = _safe_header_value(getattr(las.well, "STOP", None))
    step = _safe_header_value(getattr(las.well, "STEP", None))

    total_records = len(getattr(las, "index", []))

    return HeaderInfo(
        file_name=file_name,
        las_version=str(version) if version is not None else None,
        company=str(company) if company is not None else None,
        well_name=str(well_name) if well_name is not None else None,
        field_name=str(field_name) if field_name is not None else None,
        null_value=null_value,
        index_curve=index_curve,
        index_unit=index_unit,
        index_type=index_type,
        start=start,
        stop=stop,
        step=step,
        total_records=total_records,
    )


def curves_metadata(las, file_path: str) -> pd.DataFrame:
    raw_descriptions = extract_curve_descriptions_raw(file_path)

    rows = []

    for curve in las.curves:
        mnemonic = getattr(curve, "mnemonic", None)

        descr_lasio = getattr(curve, "descr", None)
        descr_raw = raw_descriptions.get(mnemonic)

        # prioridade: descrição bruta do arquivo
        description = descr_raw or descr_lasio

        rows.append(
            {
                "mnemonic": mnemonic,
                "unit": getattr(curve, "unit", None),
                "description": description,
            }
        )

    return pd.DataFrame(rows)


def get_valid_curve_names(las, null_value: float | None) -> list[str]:
    valid_curves = []

    index_curve_name, _ = get_index_curve_info(las)

    for curve in las.curves:
        mnemonic = getattr(curve, "mnemonic", None)
        if not mnemonic:
            continue

        if mnemonic == index_curve_name:
            continue

        raw = np.asarray(las[mnemonic])

        try:
            values = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue

        mask = ~np.isnan(values)

        if null_value is not None:
            mask &= values != null_value

        if np.any(mask):
            valid_curves.append(mnemonic)

    return valid_curves


def las_to_filtered_dataframe(las, null_value: float | None) -> pd.DataFrame:
    data = {}

    index_curve_name, _ = get_index_curve_info(las)

    index_values = np.asarray(getattr(las, "index", []))
    data["__INDEX__"] = index_values

    valid_curve_names = get_valid_curve_names(las, null_value)

    for mnemonic in valid_curve_names:
        raw = np.asarray(las[mnemonic])
        values = pd.to_numeric(pd.Series(raw), errors="coerce")

        if null_value is not None:
            values = values.replace(null_value, np.nan)

        data[mnemonic] = values.to_numpy()

    return pd.DataFrame(data)

def extract_curve_descriptions_raw(file_path: str) -> dict:
    descriptions = {}
    in_curve_section = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # detectar início da seção
            if line.upper().startswith("~C"):
                in_curve_section = True
                continue

            # saiu da seção
            if line.startswith("~") and in_curve_section:
                break

            if not in_curve_section:
                continue

            if ":" in line:
                left, descr = line.split(":", 1)

                left = left.strip()
                descr = descr.strip()

                # extrair mnemonic antes do ponto
                if "." in left:
                    mnemonic = left.split(".", 1)[0].strip()
                else:
                    mnemonic = left.strip()

                descriptions[mnemonic] = descr

    return descriptions
