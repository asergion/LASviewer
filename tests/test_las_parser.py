from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.las_parser import (
    _safe_header_value,
    _to_float,
    curves_metadata,
    detect_index_type,
    extract_curve_descriptions_raw,
    extract_header_info,
    get_index_curve_info,
    get_valid_curve_names,
    las_to_filtered_dataframe,
)


class FakeLas:
    def __init__(self, curves, data, index, index_curve=None):
        self.curves = curves
        self._data = data
        self.index = index
        self.index_curve = index_curve
        self.version = SimpleNamespace(VERS=SimpleNamespace(value="2.0"))
        self.well = SimpleNamespace(
            COMP=SimpleNamespace(value="PETROBRAS"),
            WELL=SimpleNamespace(value="POCO-01"),
            FLD=SimpleNamespace(value="CAMPO-01"),
            NULL=SimpleNamespace(value="-999.25"),
            STRT=SimpleNamespace(value="100"),
            STOP=SimpleNamespace(value="200"),
            STEP=SimpleNamespace(value="1"),
        )

    def __getitem__(self, mnemonic):
        return self._data[mnemonic]


def curve(mnemonic, unit="", descr=""):
    return SimpleNamespace(mnemonic=mnemonic, unit=unit, descr=descr)


def test_safe_header_value_deve_retornar_none_para_valores_vazios():
    assert _safe_header_value(None) is None
    assert _safe_header_value(SimpleNamespace(value="")) is None
    assert _safe_header_value(SimpleNamespace(value=" ")) is None


def test_safe_header_value_deve_retornar_value_quando_existir():
    assert _safe_header_value(SimpleNamespace(value="ABC")) == "ABC"


def test_to_float_deve_converter_valor_valido_e_retornar_none_para_invalido():
    assert _to_float("10.5") == 10.5
    assert _to_float(None) is None
    assert _to_float("abc") is None


def test_detect_index_type_deve_identificar_tempo_por_mnemonic_ou_unidade():
    assert detect_index_type("TIME", "") == "tempo"
    assert detect_index_type("qualquer", "s") == "tempo"


def test_detect_index_type_deve_identificar_profundidade_por_mnemonic_ou_unidade():
    assert detect_index_type("MD", "") == "profundidade"
    assert detect_index_type("qualquer", "m") == "profundidade"


def test_detect_index_type_deve_retornar_indeterminado():
    assert detect_index_type("ABC", "un") == "indeterminado"


def test_get_index_curve_info_deve_usar_index_curve_quando_existir():
    las = FakeLas(
        curves=[curve("MD", "m")],
        data={"MD": [1, 2]},
        index=[1, 2],
        index_curve=curve("TIME", "s"),
    )

    assert get_index_curve_info(las) == ("TIME", "s")


def test_get_index_curve_info_deve_usar_primeira_curva_quando_nao_houver_index_curve():
    las = FakeLas(
        curves=[curve("MD", "m"), curve("ROP", "m/h")],
        data={"MD": [1, 2], "ROP": [10, 20]},
        index=[1, 2],
        index_curve=None,
    )

    assert get_index_curve_info(las) == ("MD", "m")


def test_get_index_curve_info_deve_retornar_none_quando_nao_houver_curvas():
    las = FakeLas(curves=[], data={}, index=[], index_curve=None)

    assert get_index_curve_info(las) == (None, None)


def test_extract_header_info_deve_montar_header_com_tipo_de_indexacao():
    las = FakeLas(
        curves=[curve("TIME", "s")],
        data={"TIME": [1, 2, 3]},
        index=[1, 2, 3],
        index_curve=curve("TIME", "s"),
    )

    result = extract_header_info(las, "arquivo.las")

    assert result.file_name == "arquivo.las"
    assert result.las_version == "2.0"
    assert result.company == "PETROBRAS"
    assert result.well_name == "POCO-01"
    assert result.field_name == "CAMPO-01"
    assert result.null_value == -999.25
    assert result.index_curve == "TIME"
    assert result.index_unit == "s"
    assert result.index_type == "tempo"
    assert result.total_records == 3


def test_extract_curve_descriptions_raw_deve_ler_descricao_completa_da_secao_curve(tmp_path):
    las_file = tmp_path / "teste.las"
    las_file.write_text(
        """
~Version
VERS. 2.0
~Curve Information
ROPINS.min/m : Taxa de perfuração nos últimos XXs (Ex.: 5 seg)
MD.m : Profundidade medida
~Ascii Data
1 2
""",
        encoding="utf-8",
    )

    result = extract_curve_descriptions_raw(str(las_file))

    assert result["ROPINS"] == "Taxa de perfuração nos últimos XXs (Ex.: 5 seg)"
    assert result["MD"] == "Profundidade medida"


def test_curves_metadata_deve_priorizar_descricao_bruta_do_arquivo(tmp_path):
    las_file = tmp_path / "teste.las"
    las_file.write_text(
        """
~Curve Information
ROPINS.min/m : Descrição bruta correta
~Ascii Data
1
""",
        encoding="utf-8",
    )

    las = FakeLas(
        curves=[curve("ROPINS", "min/m", "descrição lasio truncada")],
        data={"ROPINS": [1]},
        index=[1],
    )

    result = curves_metadata(las, str(las_file))

    assert result.iloc[0]["mnemonic"] == "ROPINS"
    assert result.iloc[0]["unit"] == "min/m"
    assert result.iloc[0]["description"] == "Descrição bruta correta"


def test_get_valid_curve_names_deve_ignorar_indice_nulls_e_curvas_sem_dados():
    las = FakeLas(
        curves=[
            curve("MD", "m"),
            curve("ROP", "m/h"),
            curve("WOB", "klb"),
            curve("TXT", "un"),
        ],
        data={
            "MD": [1, 2, 3],
            "ROP": [10, -999.25, 20],
            "WOB": [-999.25, -999.25, -999.25],
            "TXT": ["a", "b", "c"],
        },
        index=[1, 2, 3],
        index_curve=curve("MD", "m"),
    )

    result = get_valid_curve_names(las, null_value=-999.25)

    assert result == ["ROP"]


def test_las_to_filtered_dataframe_deve_criar_dataframe_com_indice_e_curvas_validas():
    las = FakeLas(
        curves=[curve("MD", "m"), curve("ROP", "m/h"), curve("WOB", "klb")],
        data={
            "MD": [1, 2, 3],
            "ROP": [10, -999.25, 20],
            "WOB": [-999.25, -999.25, -999.25],
        },
        index=np.array([1, 2, 3]),
        index_curve=curve("MD", "m"),
    )

    result = las_to_filtered_dataframe(las, null_value=-999.25)

    assert result.columns.tolist() == ["__INDEX__", "ROP"]
    assert result["__INDEX__"].tolist() == [1, 2, 3]
    assert result["ROP"].isna().sum() == 1
    assert result["ROP"].dropna().tolist() == [10, 20]