import pandas as pd
import pytest

from src.analysis import (
    classify_las_quality,
    compare_curves_between_wells,
    compare_wells_metrics,
    compute_time_metrics,
    curves_with_valid_data,
    numeric_curve_stats,
)


def test_numeric_curve_stats_deve_ignorar_indices_e_curvas_sem_valores_validos():
    df = pd.DataFrame({
        "__INDEX__": [1, 2, 3],
        "__INDEX_DATETIME__": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "ROP": [10, 20, None],
        "WOB": [None, None, None],
        "SPP": ["1.5", "2.5", "erro"],
    })

    result = numeric_curve_stats(df)

    assert result["curve"].tolist() == ["ROP", "SPP"]

    rop = result[result["curve"] == "ROP"].iloc[0]
    assert rop["total_values"] == 3
    assert rop["valid_values"] == 2
    assert rop["null_values"] == 1
    assert rop["min"] == 10
    assert rop["max"] == 20

    spp = result[result["curve"] == "SPP"].iloc[0]
    assert spp["valid_values"] == 2
    assert spp["min"] == 1.5
    assert spp["max"] == 2.5


def test_curves_with_valid_data_deve_retornar_lista_vazia_quando_dataframe_vazio():
    result = curves_with_valid_data(pd.DataFrame())

    assert result == []


def test_curves_with_valid_data_deve_retornar_curvas_na_ordem_do_dataframe():
    stats = pd.DataFrame({"curve": ["ROP", "WOB", "SPP"]})

    result = curves_with_valid_data(stats)

    assert result == ["ROP", "WOB", "SPP"]


def test_compare_curves_between_wells_deve_fazer_inner_join_pelo_indice():
    df1 = pd.DataFrame({"__INDEX__": [1, 2, 3], "ROP": [10, 20, 30]})
    df2 = pd.DataFrame({"__INDEX__": [2, 3, 4], "ROP": [200, 300, 400]})

    result = compare_curves_between_wells(df1, df2, "ROP")

    assert result.to_dict(orient="records") == [
        {"index": 2, "well_1": 20, "well_2": 200},
        {"index": 3, "well_1": 30, "well_2": 300},
    ]


def test_compute_time_metrics_deve_retornar_vazio_quando_menos_de_dois_indices():
    df = pd.DataFrame({"__INDEX__": [100]})

    result = compute_time_metrics(df)

    assert result == {}


def test_compute_time_metrics_deve_calcular_cobertura_gaps_e_deltas():
    df = pd.DataFrame({"__INDEX__": [1, 2, 3, 7]})

    result = compute_time_metrics(df)

    assert result["total_registros"] == 4
    assert result["duracao"] == 6
    assert result["cobertura_percentual"] == pytest.approx(57.142857)
    assert result["delta_medio"] == pytest.approx(2.0)
    assert result["delta_mediano"] == pytest.approx(1.0)
    assert result["delta_min"] == 1
    assert result["delta_max"] == 4
    assert result["gaps"] == 1
    assert result["maior_gap"] == 4


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({}, "Sem dados"),
        ({"cobertura_percentual": 80, "gaps": 0, "maior_gap": 1}, "🔴 Baixa qualidade (subamostrado)"),
        ({"cobertura_percentual": 95, "gaps": 0, "maior_gap": 11}, "🔴 Gaps grandes"),
        ({"cobertura_percentual": 95, "gaps": 2, "maior_gap": 2}, "🟡 Gaps moderados"),
        ({"cobertura_percentual": 100, "gaps": 0, "maior_gap": 1}, "🟢 Boa qualidade"),
    ],
)
def test_classify_las_quality_deve_classificar_conforme_metricas(metrics, expected):
    result = classify_las_quality(metrics)

    assert result == expected


def test_compare_wells_metrics_deve_comparar_metricas_e_curvas():
    m1 = {
        "total_registros": 100,
        "cobertura_percentual": 90.0,
        "delta_medio": 1.0,
        "maior_gap": 2.0,
    }
    m2 = {
        "total_registros": 80,
        "cobertura_percentual": 75.0,
        "delta_medio": 1.5,
        "maior_gap": 5.0,
    }

    result = compare_wells_metrics(
        m1,
        m2,
        curves1=["ROP", "WOB", "SPP"],
        curves2=["ROP", "SPP", "RPM", "MD"],
    )

    assert result == {
        "dif_total_registros": 20,
        "dif_cobertura": 15.0,
        "dif_delta_medio": 0.5,
        "dif_maior_gap": 3.0,
        "dif_qtd_curvas_validas": 1,
        "curvas_comuns": 2,
    }