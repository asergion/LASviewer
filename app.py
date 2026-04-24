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


# -----------------------------
# CACHE
# -----------------------------
@st.cache_data(show_spinner=True)
def load_uploaded_las(file_bytes: bytes, file_name: str):
    suffix = Path(file_name).suffix or ".las"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    las = read_las(tmp_path)

    header = extract_header_info(las, file_name)

    metadata_df = curves_metadata(las, tmp_path)

    df = las_to_filtered_dataframe(las, header.null_value)

    stats_df = numeric_curve_stats(df)

    valid_curves = curves_with_valid_data(stats_df)

    return {
        "header": header,
        "metadata_df": metadata_df,
        "df": df,
        "stats_df": stats_df,
        "valid_curves": valid_curves,
    }


datasets = [
    load_uploaded_las(f.getvalue(), f.name)
    for f in uploaded_files
]


abas = st.tabs(["Resumo", "Curvas", "Gráficos", "Comparação"])


# ==========================================================
# RESUMO
# ==========================================================
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
        st.dataframe(item["metadata_df"].head(200), width="stretch")

        st.write("**Estatísticas das curvas (somente válidas)**")
        stats_filtrado = item["stats_df"][item["stats_df"]["valid_values"] > 0].copy()

        # Ajustar índice para começar em 1
        stats_filtrado.index = stats_filtrado.index + 1

        if stats_filtrado.empty:
            st.warning("Nenhuma curva com dados válidos.")
        else:
            st.dataframe(stats_filtrado.head(200), width="stretch")

        st.divider()


# ==========================================================
# CURVAS
# ==========================================================
with abas[1]:
    dataset_idx = st.selectbox(
        "Escolha o arquivo",
        options=range(len(datasets)),
        format_func=lambda x: datasets[x]["header"].file_name,
    )

    item = datasets[dataset_idx]

    st.write("**Curvas com dados válidos**")

    stats_validas = item["stats_df"][item["stats_df"]["valid_values"] > 0].copy()

    # Ajustar índice para começar em 1
    stats_validas.index = stats_validas.index + 1

    st.dataframe(
        stats_validas.head(200),
        width="stretch",
    )

    st.write("**Unidades e descrições das curvas válidas**")

    curvas_validas = stats_validas["curve"].tolist()

    metadata_validas = item["metadata_df"][
        item["metadata_df"]["mnemonic"].isin(curvas_validas)
    ].copy()

    metadata_validas = metadata_validas.rename(
        columns={
            "mnemonic": "curve",
            "unit": "unit",
            "description": "description",
        }
    )

    st.dataframe(
        metadata_validas.head(200),
        width="stretch",
    )

# ==========================================================
# GRÁFICOS
# ==========================================================
with abas[2]:
    dataset_idx = st.selectbox(
        "Arquivo para análise gráfica",
        options=range(len(datasets)),
        format_func=lambda x: datasets[x]["header"].file_name,
        key="grafico_dataset",
    )

    item = datasets[dataset_idx]

    if not item["valid_curves"]:
        st.warning("Nenhuma curva válida para plot.")
    else:
        st.markdown("### Configuração dos eixos")

        axis_options = ["__INDEX__"] + item["valid_curves"]

        primeira_curva = item["valid_curves"][0]

        if item["header"].index_type == "tempo":
            default_x = "__INDEX__"
            default_y = primeira_curva
        elif item["header"].index_type == "profundidade":
            default_x = primeira_curva
            default_y = "__INDEX__"
        else:
            default_x = "__INDEX__"
            default_y = primeira_curva

        def format_axis_label(value):
            if value == "__INDEX__":
                if item["header"].index_type == "tempo":
                    return "Tempo"
                elif item["header"].index_type == "profundidade":
                    return "Profundidade"
                return "Índice"
            return value

        col_x, col_y = st.columns(2)

        x_column = col_x.selectbox(
            "Eixo X",
            axis_options,
            index=axis_options.index(default_x),
            format_func=format_axis_label,
        )

        y_column = col_y.selectbox(
            "Eixo Y",
            axis_options,
            index=axis_options.index(default_y),
            format_func=format_axis_label,
        )

        # -----------------------------
        # DESCRIÇÃO DAS CURVAS ESCOLHIDAS
        # -----------------------------
        metadata_df = item["metadata_df"]

        selected_curves = [
            col for col in [x_column, y_column]
            if col != "__INDEX__"
        ]

        for selected_curve in selected_curves:
            curve_info = metadata_df[metadata_df["mnemonic"] == selected_curve]

            if not curve_info.empty:
                description = curve_info.iloc[0]["description"]
                unit = curve_info.iloc[0]["unit"]

                st.markdown(
                    f"**{selected_curve}** — "
                    f"Descrição: {description or '-'} | "
                    f"Unidade: {unit or '-'}"
                )

        if x_column == y_column:
            st.warning("Escolha variáveis diferentes para X e Y.")
        else:
            fig = plot_single_curve(
                item["df"],
                x_column,
                y_column,
                title=f"{format_axis_label(x_column)} x {format_axis_label(y_column)} - {item['header'].file_name}",
                index_type=item["header"].index_type,
            )

            st.plotly_chart(fig, width="stretch")

        # -----------------------------
        # HISTOGRAMA
        # -----------------------------
        curva_histograma = y_column if y_column != "__INDEX__" else x_column

        if curva_histograma != "__INDEX__":
            st.write("**Histograma**")
            hist_fig = plot_histogram(item["df"], curva_histograma)
            st.plotly_chart(hist_fig, width="stretch")

        # -----------------------------
        # CROSSPLOT
        # -----------------------------
        st.write("**Crossplot**")

        curve_x = st.selectbox("Curva X", item["valid_curves"], key="cx")
        curve_y = st.selectbox("Curva Y", item["valid_curves"], key="cy")

        if curve_x != curve_y:
            cross_fig = plot_crossplot(item["df"], curve_x, curve_y)
            st.plotly_chart(cross_fig, width="stretch")
        else:
            st.info("Escolha curvas diferentes.")

# ==========================================================
# COMPARAÇÃO ENTRE POÇOS
# ==========================================================
with abas[3]:
    if len(datasets) < 2:
        st.info("Envie dois arquivos para comparação.")
    else:
        common_curves = sorted(
            set(datasets[0]["valid_curves"]).intersection(
                set(datasets[1]["valid_curves"])
            )
        )

        if not common_curves:
            st.warning("Nenhuma curva comum.")
        else:
            curve_name = st.selectbox("Curva", common_curves)

            compare_df = compare_curves_between_wells(
                datasets[0]["df"],
                datasets[1]["df"],
                curve_name,
            )

            if compare_df.empty:
                st.warning("Não foi possível alinhar índices.")
            else:
                st.dataframe(compare_df.head(200), width="stretch")

                fig = plot_compare_wells(compare_df, curve_name)
                st.plotly_chart(fig, width="stretch")