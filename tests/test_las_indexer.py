from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src import las_indexer


@dataclass
class FakeHeader:
    file_name: str = "teste.las"
    las_version: str = "2.0"
    company: str = "COMP"
    well_name: str = "POCO"
    field_name: str = "CAMPO"
    null_value: float = -999.25
    index_curve: str = "TIME"
    index_unit: str = "s"
    index_type: str = "tempo"
    start: str = "0"
    stop: str = "2"
    step: str = "1"
    total_records: int = 3


def test_detect_unix_time_unit_deve_retornar_none_quando_series_vazia():
    result = las_indexer.detect_unix_time_unit(pd.Series([None, "abc"]))

    assert result is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1_700_000_000], "s"),
        ([1_700_000_000_000], "ms"),
        ([1_700_000_000_000_000_000], "ns"),
    ],
)
def test_detect_unix_time_unit_deve_detectar_s_ms_ns(values, expected):
    result = las_indexer.detect_unix_time_unit(pd.Series(values))

    assert result == expected


def test_normalize_time_index_deve_retornar_df_original_quando_nao_for_tempo():
    df = pd.DataFrame({"__INDEX__": [1, 2, 3]})

    result_df, unit = las_indexer.normalize_time_index(df, "profundidade")

    assert result_df is df
    assert unit is None
    assert "__INDEX_DATETIME__" not in result_df.columns


def test_normalize_time_index_deve_retornar_df_original_quando_nao_detectar_unix():
    df = pd.DataFrame({"__INDEX__": [1, 2, 3]})

    result_df, unit = las_indexer.normalize_time_index(df, "tempo")

    assert result_df is df
    assert unit is None
    assert "__INDEX_DATETIME__" not in result_df.columns


def test_normalize_time_index_deve_criar_coluna_datetime_para_tempo_unix():
    df = pd.DataFrame({"__INDEX__": [1_700_000_000, 1_700_000_001]})

    result_df, unit = las_indexer.normalize_time_index(df, "tempo")

    assert unit == "s"
    assert "__INDEX_DATETIME__" in result_df.columns
    assert str(result_df["__INDEX_DATETIME__"].dt.tz) == "UTC"
    assert result_df is not df


def test_index_las_file_deve_gerar_metadata_parquet_e_json_para_indexacao_tempo(monkeypatch, tmp_path):
    las_path = tmp_path / "poco_tempo.las"
    las_path.write_text("fake", encoding="utf-8")

    fake_header = FakeHeader(
        file_name="poco_tempo.las",
        index_type="tempo",
        index_curve="TIME",
        index_unit="s",
        total_records=3,
    )

    df = pd.DataFrame(
        {
            "__INDEX__": [1_700_000_000, 1_700_000_001, 1_700_000_002],
            "MD": [1000.0, 1001.0, 1002.0],
            "ROP": [10.0, 20.0, 30.0],
        }
    )

    metadata_df = pd.DataFrame(
        [
            {"mnemonic": "TIME", "unit": "s", "description": "Tempo"},
            {"mnemonic": "MD", "unit": "m", "description": "Profundidade"},
            {"mnemonic": "ROP", "unit": "m/h", "description": "Taxa"},
        ]
    )

    monkeypatch.setattr(las_indexer, "read_las", lambda path: object())
    monkeypatch.setattr(las_indexer, "extract_header_info", lambda las, name: fake_header)
    monkeypatch.setattr(las_indexer, "curves_metadata", lambda las, path: metadata_df)
    monkeypatch.setattr(las_indexer, "las_to_filtered_dataframe", lambda las, null_value: df)

    result = las_indexer.index_las_file(las_path, tmp_path)

    parquet_path = tmp_path / "poco_tempo.parquet"
    metadata_path = tmp_path / "poco_tempo.metadata.json"

    assert parquet_path.exists()
    assert metadata_path.exists()

    assert result["source_file"] == "poco_tempo.las"
    assert result["parquet_file"] == "poco_tempo.parquet"
    assert result["metadata_file"] == "poco_tempo.metadata.json"
    assert result["header"]["index_type"] == "tempo"
    assert result["time_unit"] == "s"
    assert result["total_records"] == 3
    assert result["index_min"] == 1_700_000_000
    assert result["index_max"] == 1_700_000_002
    assert result["md_min"] == 1000.0
    assert result["md_max"] == 1002.0
    assert result["tempo_min"] is not None
    assert result["tempo_max"] is not None
    assert "__INDEX_DATETIME__" in result["columns"]
    assert set(result["valid_curves"]) == {"MD", "ROP"}

    parquet_df = pd.read_parquet(parquet_path)
    assert parquet_df.columns.tolist() == result["columns"]

    metadata_json = metadata_path.read_text(encoding="utf-8")
    assert '"source_file": "poco_tempo.las"' in metadata_json


def test_index_las_file_deve_gerar_metadata_sem_tempo_e_sem_md_quando_nao_existirem(monkeypatch, tmp_path):
    las_path = tmp_path / "poco_depth.las"
    las_path.write_text("fake", encoding="utf-8")

    fake_header = FakeHeader(
        file_name="poco_depth.las",
        index_type="profundidade",
        index_curve="DEPT",
        index_unit="m",
        total_records=3,
    )

    df = pd.DataFrame(
        {
            "__INDEX__": [100.0, 101.0, 102.0],
            "ROP": [10.0, 20.0, 30.0],
        }
    )

    metadata_df = pd.DataFrame(
        [{"mnemonic": "ROP", "unit": "m/h", "description": "Taxa"}]
    )

    monkeypatch.setattr(las_indexer, "read_las", lambda path: object())
    monkeypatch.setattr(las_indexer, "extract_header_info", lambda las, name: fake_header)
    monkeypatch.setattr(las_indexer, "curves_metadata", lambda las, path: metadata_df)
    monkeypatch.setattr(las_indexer, "las_to_filtered_dataframe", lambda las, null_value: df)

    result = las_indexer.index_las_file(las_path, tmp_path)

    assert result["time_unit"] is None
    assert result["tempo_min"] is None
    assert result["tempo_max"] is None
    assert result["md_min"] is None
    assert result["md_max"] is None
    assert result["index_min"] == 100.0
    assert result["index_max"] == 102.0
    assert "__INDEX_DATETIME__" not in result["columns"]
    assert result["valid_curves"] == ["ROP"]


def test_index_las_file_deve_retornar_index_min_max_none_quando_indice_invalido(monkeypatch, tmp_path):
    las_path = tmp_path / "poco_invalido.las"
    las_path.write_text("fake", encoding="utf-8")

    fake_header = FakeHeader(index_type="profundidade", index_curve="DEPT", index_unit="m")

    df = pd.DataFrame(
        {
            "__INDEX__": ["abc", None],
            "ROP": [10.0, 20.0],
        }
    )

    metadata_df = pd.DataFrame(
        [{"mnemonic": "ROP", "unit": "m/h", "description": "Taxa"}]
    )

    monkeypatch.setattr(las_indexer, "read_las", lambda path: object())
    monkeypatch.setattr(las_indexer, "extract_header_info", lambda las, name: fake_header)
    monkeypatch.setattr(las_indexer, "curves_metadata", lambda las, path: metadata_df)
    monkeypatch.setattr(las_indexer, "las_to_filtered_dataframe", lambda las, null_value: df)

    result = las_indexer.index_las_file(las_path, tmp_path)

    assert result["index_min"] is None
    assert result["index_max"] is None
    assert result["total_records"] == 2