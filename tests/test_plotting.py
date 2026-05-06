import pandas as pd
import plotly.graph_objects as go

from src.plotting import (
    plot_compare_wells,
    plot_crossplot,
    plot_curves_side_by_side,
    plot_histogram,
    plot_single_curve,
)


def label_formatter(value: str) -> str:
    labels = {
        "__INDEX__": "Profundidade (m)",
        "__INDEX_DATETIME__": "Tempo",
        "ROP": "ROP (m/h)",
        "WOB": "WOB (klb)",
        "SPP": "SPP (psi)",
    }
    return labels.get(value, value)


def test_plot_curves_side_by_side_deve_retornar_none_quando_nao_houver_variaveis():
    df = pd.DataFrame({"__INDEX__": [1, 2, 3]})

    result = plot_curves_side_by_side(
        df=df,
        fixed_axis="Eixo X",
        fixed_variable="__INDEX__",
        selected_variables=[],
        title="Teste",
        index_type="tempo",
        label_formatter=label_formatter,
    )

    assert result is None


def test_plot_curves_side_by_side_deve_criar_um_trace_por_variavel_com_eixo_x_fixo():
    df = pd.DataFrame(
        {
            "__INDEX__": [1, 2, 3],
            "ROP": [10, 20, 30],
            "WOB": [1, 2, 3],
        }
    )

    fig = plot_curves_side_by_side(
        df=df,
        fixed_axis="Eixo X",
        fixed_variable="__INDEX__",
        selected_variables=["ROP", "WOB"],
        title="Curvas",
        index_type="tempo",
        label_formatter=label_formatter,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Curvas"
    assert fig.layout.height == 650
    assert fig.data[0].name == "ROP (m/h)"
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [10, 20, 30]


def test_plot_curves_side_by_side_deve_criar_linhas_em_multiplas_fileiras():
    df = pd.DataFrame({"__INDEX__": list(range(5))})

    selected_variables = [f"C{i}" for i in range(11)]

    for variable in selected_variables:
        df[variable] = list(range(5))

    fig = plot_curves_side_by_side(
        df=df,
        fixed_axis="Eixo X",
        fixed_variable="__INDEX__",
        selected_variables=selected_variables,
        title="Muitas curvas",
        index_type="tempo",
        label_formatter=lambda value: value,
    )

    assert len(fig.data) == 11
    assert fig.layout.height == 1040


def test_plot_curves_side_by_side_deve_inverter_eixo_y_para_profundidade_com_eixo_y_fixo():
    df = pd.DataFrame(
        {
            "__INDEX__": [1000, 1001, 1002],
            "ROP": [10, 20, 30],
        }
    )

    fig = plot_curves_side_by_side(
        df=df,
        fixed_axis="Eixo Y",
        fixed_variable="__INDEX__",
        selected_variables=["ROP"],
        title="Profundidade",
        index_type="profundidade",
        label_formatter=label_formatter,
    )

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [10, 20, 30]
    assert list(fig.data[0].y) == [1000, 1001, 1002]

    # yaxes = [axis for name, axis in fig.layout.items() if name.startswith("yaxis")]
    # assert any(getattr(axis, "autorange", None) == "reversed" for axis in yaxes)
    assert fig.layout.yaxis.autorange == "reversed"

def test_plot_curves_side_by_side_deve_configurar_eixo_datetime_como_data():
    df = pd.DataFrame(
        {
            "__INDEX__": [1, 2, 3],
            "__INDEX_DATETIME__": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 00:00:01", "2024-01-01 00:00:02"]
            ),
            "ROP": [10, 20, 30],
        }
    )

    fig = plot_curves_side_by_side(
        df=df,
        fixed_axis="Eixo X",
        fixed_variable="__INDEX_DATETIME__",
        selected_variables=["ROP"],
        title="Tempo",
        index_type="tempo",
        label_formatter=label_formatter,
    )

    assert len(fig.data) == 1
    assert fig.layout.xaxis.type == "date"
    assert fig.layout.xaxis.tickformat == "%d/%m/%Y<br>%H:%M:%S"


def test_plot_curves_side_by_side_deve_remover_valores_invalidos_do_trace():
    df = pd.DataFrame(
        {
            "__INDEX__": [1, 2, 3],
            "ROP": [10, "erro", 30],
        }
    )

    fig = plot_curves_side_by_side(
        df=df,
        fixed_axis="Eixo X",
        fixed_variable="__INDEX__",
        selected_variables=["ROP"],
        title="Com inválidos",
        index_type="tempo",
        label_formatter=label_formatter,
    )

    assert list(fig.data[0].x) == [1, 3]
    assert list(fig.data[0].y) == [10, 30]


def test_plot_single_curve_deve_criar_linha_e_inverter_y_quando_y_for_indice_de_profundidade():
    df = pd.DataFrame(
        {
            "__INDEX__": [1000, 1001, 1002],
            "ROP": [10, 20, 30],
        }
    )

    fig = plot_single_curve(
        df=df,
        x_column="ROP",
        y_column="__INDEX__",
        title="Curva simples",
        index_type="profundidade",
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Curva simples"
    assert fig.layout.yaxis.autorange == "reversed"


def test_plot_single_curve_deve_usar_label_tempo_quando_indexacao_for_tempo():
    df = pd.DataFrame(
        {
            "__INDEX__": [1, 2, 3],
            "ROP": [10, 20, 30],
        }
    )

    fig = plot_single_curve(
        df=df,
        x_column="__INDEX__",
        y_column="ROP",
        title="Tempo",
        index_type="tempo",
    )

    assert fig.layout.xaxis.title.text == "Tempo"


def test_plot_crossplot_deve_criar_scatter_convertendo_valores_numericos():
    df = pd.DataFrame(
        {
            "ROP": ["10", "20", "erro"],
            "WOB": [1, 2, 3],
        }
    )

    fig = plot_crossplot(df, "ROP", "WOB")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Crossplot: ROP x WOB"
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [10, 20]
    assert list(fig.data[0].y) == [1, 2]


def test_plot_histogram_deve_criar_histograma_com_valores_validos():
    df = pd.DataFrame({"ROP": ["10", "20", "erro", None]})

    fig = plot_histogram(df, "ROP")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Histograma - ROP"
    assert list(fig.data[0].x) == [10, 20]


def test_plot_compare_wells_deve_criar_duas_series_e_inverter_eixo_y():
    compare_df = pd.DataFrame(
        {
            "index": [1000, 1001, 1002],
            "well_1": [10, 20, 30],
            "well_2": [15, 25, 35],
        }
    )

    fig = plot_compare_wells(compare_df, "ROP")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Poço 1"
    assert fig.data[1].name == "Poço 2"
    assert list(fig.data[0].x) == [10, 20, 30]
    assert list(fig.data[0].y) == [1000, 1001, 1002]
    assert fig.layout.title.text == "Comparação da curva ROP"
    assert fig.layout.xaxis.title.text == "ROP"
    assert fig.layout.yaxis.title.text == "Índice"
    assert fig.layout.yaxis.autorange == "reversed"