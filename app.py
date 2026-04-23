from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.analysis import (
    compare_curves_between_wells,
    curves_with_valid_data,
    numeric_curve_stats,
)
from src.las_parser import (
    curves_metadata,
    extract_header_info,
    las_to_filtered_dataframe,
    read_las,
)
from src.plotting import (
    plot_compare_wells,
    plot_crossplot,
    plot_histogram,
    plot_single_curve,
)


st.set_page_config(page_title="Leitor LAS", layout="wide")

st.title("Leitor e Analisador de Arquivos LAS")

uploaded_files = st.file_uploader(
    "Selecione um ou dois arquivos LAS",
    type=["las"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Envie pelo menos um arquivo LAS para começar.")
    st.stop()

if len(uploaded_files) > 2:
    st.warning("Por enquanto, envie no máximo dois arquivos.")
    st.stop()


def load_uploaded_las(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".las"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    las = read_las(tmp_path)
    header = extract_header_info(las, uploaded_file.name)
    metadata_df = curves_metadata(las, tmp_path)
    df = las_to_filtered_dataframe(las, header.null_value)
    stats_df = numeric_curve_stats(df)
    valid_curves = curves_with_valid_data(stats_df)

    return {
        "las": las,
        "header": header,
        "metadata_df": metadata_df,
        "df": df,
        "stats_df": stats_df,
        "valid_curves": valid_curves,
    }


datasets = [load_uploaded_las(f) for f in uploaded_files]

abas = st.tabs(["Resumo", "Curvas", "Gráficos", "Comparação"])


with abas[0]:
    for i, item in enumerate(datasets, start=1):
        st.subheader(f"Arquivo {i}: {item['header'].file_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Versão LAS", item["header"].las_version or "-")
        col2.metric("Companhia", item["header"].company or "-")
        col3.metric("Poço", item["header"].well_name or "-")

        col4, col5, col6 = st.columns(3)
        col4.metric("Tipo de indexação", item["header"].index_type)
        col5.metric("Curva índice", item["header"].index_curve or "-")
        col6.metric("Total de registros", item["header"].total_records)

        st.write("**Cabeçalho principal**")
        st.json({
            "field_name": item["header"].field_name,
            "null_value": item["header"].null_value,
            "start": item["header"].start,
            "stop": item["header"].stop,
            "step": item["header"].step,
            "index_unit": item["header"].index_unit,
        })

        st.write("**Curvas presentes**")
        st.dataframe(item["metadata_df"], use_container_width=True)

        st.write("**Estatísticas das curvas (apenas com dados válidos)**")

        stats_filtrado = item["stats_df"][item["stats_df"]["valid_values"] > 0]

        if stats_filtrado.empty:
            st.warning("Nenhuma curva com valores válidos foi encontrada.")
        else:
            st.dataframe(stats_filtrado, use_container_width=True)

        st.divider()


with abas[1]:
    dataset_idx = st.selectbox(
        "Escolha o arquivo",
        options=range(len(datasets)),
        format_func=lambda x: datasets[x]["header"].file_name,
        key="curvas_dataset",
    )

    item = datasets[dataset_idx]
    st.write("**Curvas com valores válidos**")
    st.dataframe(
        item["stats_df"][item["stats_df"]["valid_values"] > 0],
        use_container_width=True,
    )


with abas[2]:
    dataset_idx = st.selectbox(
        "Arquivo para análise gráfica",
        options=range(len(datasets)),
        format_func=lambda x: datasets[x]["header"].file_name,
        key="grafico_dataset",
    )

    item = datasets[dataset_idx]

    if not item["valid_curves"]:
        st.warning("Esse arquivo não possui curvas numéricas válidas para plot.")
    else:
        curve = st.selectbox("Selecione a curva", item["valid_curves"])

        # Buscar descrição da curva
        metadata_df = item["metadata_df"]

        curve_info = metadata_df[metadata_df["mnemonic"] == curve]

        if not curve_info.empty:
            description = curve_info.iloc[0]["description"]
            unit = curve_info.iloc[0]["unit"]

            st.markdown(f"**Descrição:** {description or '-'}")
            st.markdown(f"**Unidade:** {unit or '-'}")
        else:
            st.markdown("**Descrição:** -")
            st.markdown("**Unidade:** -")

        fig = plot_single_curve(item["df"], curve, item["header"].index_type)
        st.plotly_chart(fig, width="stretch")

        st.write("**Histograma**")
        hist_fig = plot_histogram(item["df"], curve)
        st.plotly_chart(hist_fig, width="stretch")

        st.write("**Crossplot**")
        curve_x = st.selectbox("Curva X", item["valid_curves"], key="curve_x")
        curve_y = st.selectbox("Curva Y", item["valid_curves"], key="curve_y")
        if curve_x != curve_y:
            cross_fig = plot_crossplot(item["df"], curve_x, curve_y)
            st.plotly_chart(cross_fig, width="stretch")
        else:
            st.info("Selecione curvas diferentes para o crossplot.")


with abas[3]:
    if len(datasets) < 2:
        st.info("Envie dois arquivos para habilitar a comparação entre poços.")
    else:
        common_curves = sorted(
            set(datasets[0]["valid_curves"]).intersection(set(datasets[1]["valid_curves"]))
        )

        if not common_curves:
            st.warning("Não há curvas válidas em comum entre os dois arquivos.")
        else:
            curve_name = st.selectbox("Curva para comparar", common_curves)
            compare_df = compare_curves_between_wells(
                datasets[0]["df"],
                datasets[1]["df"],
                curve_name,
            )

            if compare_df.empty:
                st.warning("Não foi possível alinhar os índices das duas curvas.")
            else:
                st.dataframe(compare_df.head(200), use_container_width=True)
                fig = plot_compare_wells(compare_df, curve_name)
                st.plotly_chart(fig, width="stretch")