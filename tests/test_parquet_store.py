import json

import pandas as pd
import pytest

from src.parquet_store import (
    downsample_df,
    load_metadata,
    read_selected_curves,
    resolve_parquet_path,
)


def test_load_metadata_deve_ler_json(tmp_path):
    metadata_path = tmp_path / "arquivo.metadata.json"
    metadata_path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    result = load_metadata(metadata_path)

    assert result == {"a": 1}


def test_resolve_parquet_path_deve_retornar_caminho_quando_arquivo_existe(tmp_path):
    parquet_path = tmp_path / "arquivo.parquet"
    parquet_path.write_bytes(b"fake")

    result = resolve_parquet_path(parquet_path)

    assert result == parquet_path


def test_resolve_parquet_path_deve_procurar_na_pasta_do_metadata(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    metadata_path = metadata_dir / "arquivo.metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    parquet_path = metadata_dir / "arquivo.parquet"
    parquet_path.write_bytes(b"fake")

    result = resolve_parquet_path("arquivo.parquet", metadata_path)

    assert result == parquet_path


def test_resolve_parquet_path_deve_lancar_file_not_found_quando_nao_encontrar(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_parquet_path("inexistente.parquet", tmp_path / "arquivo.metadata.json")


def test_read_selected_curves_deve_ler_apenas_colunas_solicitadas(tmp_path):
    parquet_path = tmp_path / "arquivo.parquet"

    df = pd.DataFrame({
        "__INDEX__": [1, 2, 3],
        "__INDEX_DATETIME__": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "ROP": [10, 20, 30],
        "WOB": [1, 2, 3],
    })

    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    result = read_selected_curves(
        parquet_path=parquet_path,
        fixed_variable="__INDEX_DATETIME__",
        selected_variables=["ROP"],
    )

    assert result.columns.tolist() == ["__INDEX_DATETIME__", "ROP", "__INDEX__"]
    assert result["ROP"].tolist() == [10, 20, 30]


def test_downsample_df_deve_retornar_mesmo_dataframe_quando_menor_que_limite():
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = downsample_df(df, max_points=10)

    assert result is df


def test_downsample_df_deve_reduzir_dataframe_quando_maior_que_limite():
    df = pd.DataFrame({"a": list(range(100))})

    result = downsample_df(df, max_points=10)

    assert len(result) == 10
    assert result["a"].tolist() == list(range(0, 100, 10))