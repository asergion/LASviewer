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
    plot_curves_side_by_side,
)


st.set_page_config(page_title="Leitor LAS", layout="wide")
st.title("Leitor e Analisador de Arquivos LAS")

def preview_las_file(file_path: str, extra_lines_after_ascii: int = 5) -> str:
    preview_lines = []
    found_ascii = False
    lines_after_ascii = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            preview_lines.append(line.rstrip("\n"))

            line_upper = line.strip().upper()

            if line_upper.startswith("~A") or "~ASCII" in line_upper:
                found_ascii = True
                continue

            if found_ascii:
                lines_after_ascii += 1

                if lines_after_ascii >= extra_lines_after_ascii:
                    break

    return "\n".join(preview_lines)


@st.cache_data(show_spinner=True)
def load_uploaded_las(file_bytes: bytes, file_name: str):
    suffix = Path(file_name).suffix or ".las"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    preview_text = preview_las_file(tmp_path)

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
        "preview_text": preview_text,
    }

# SIDEBAR CONFIGURACOES
with st.sidebar:
    st.header("Configurações")

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


datasets = [
    load_uploaded_las(f.getvalue(), f.name)
    for f in uploaded_files
]

with st.sidebar:
    dataset_idx_global = st.selectbox(
        "Arquivo para análise gráfica",
        options=range(len(datasets)),
        format_func=lambda x: datasets[x]["header"].file_name,
        key="sidebar_dataset",
    )

    item_global = datasets[dataset_idx_global]

    st.markdown("### Configuração dos eixos")

    axis_options_global = ["__INDEX__"] + item_global["valid_curves"]

    def format_axis_label_global(value):
        if value == "__INDEX__":
            if item_global["header"].index_type == "tempo":
                return "Tempo (s)"
            if item_global["header"].index_type == "profundidade":
                return f"Profundidade ({item_global['header'].index_unit or 'm'})"
            return "Índice"
        # Buscar unidade no metadata
        curve_info = item_global["metadata_df"][
            item_global["metadata_df"]["mnemonic"] == value
        ]

        if not curve_info.empty:
            unit = curve_info.iloc[0]["unit"]
            if unit:
                return f"{value} ({unit})"

        return value

    eixo_fixo = st.radio(
        "Eixo fixo",
        ["Eixo X", "Eixo Y"],
        index=1 if item_global["header"].index_type == "profundidade" else 0,
    )

    variavel_fixa_default = "__INDEX__"

    variavel_fixa = st.selectbox(
        "Variável fixa",
        axis_options_global,
        index=axis_options_global.index(variavel_fixa_default),
        format_func=format_axis_label_global,
    )

    variaveis_disponiveis = [
        curva for curva in axis_options_global
        if curva != variavel_fixa
    ]

    st.markdown("**Variáveis do outro eixo**")

    estado_key = f"variaveis_multiplas_{dataset_idx_global}_{variavel_fixa}"

    checkbox_prefix = f"var_outro_eixo_{dataset_idx_global}_{variavel_fixa}"

    default_variaveis = variaveis_disponiveis[: min(3, len(variaveis_disponiveis))]

    if estado_key not in st.session_state:
        st.session_state[estado_key] = default_variaveis

    for variavel in variaveis_disponiveis:
        checkbox_key = f"{checkbox_prefix}_{variavel}"

        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = variavel in st.session_state[estado_key]


    def selecionar_todas_variaveis():
        st.session_state[estado_key] = variaveis_disponiveis

        for variavel in variaveis_disponiveis:
            checkbox_key = f"{checkbox_prefix}_{variavel}"
            st.session_state[checkbox_key] = True


    def limpar_variaveis():
        st.session_state[estado_key] = []

        for variavel in variaveis_disponiveis:
            checkbox_key = f"{checkbox_prefix}_{variavel}"
            st.session_state[checkbox_key] = False


    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        st.button(
            "Selecionar todas",
            key=f"btn_selecionar_todas_{estado_key}",
            on_click=selecionar_todas_variaveis,
        )

    with col_sel2:
        st.button(
            "Limpar",
            key=f"btn_limpar_{estado_key}",
            on_click=limpar_variaveis,
        )

    with st.container(height=520):
        variaveis_multiplas = []

        for variavel in variaveis_disponiveis:
            checkbox_key = f"{checkbox_prefix}_{variavel}"

            selecionada = st.checkbox(
                format_axis_label_global(variavel),
                key=checkbox_key,
            )

            if selecionada:
                variaveis_multiplas.append(variavel)

    st.session_state[estado_key] = variaveis_multiplas


abas = st.tabs(["Resumo", "Gráficos", "Comparação"])

# RESUMO
# Exibe os metadados principais de cada arquivo LAS, a pré-visualização textual
# do arquivo e as primeiras tabelas de curvas/metadados.
with abas[0]:
    for i, item in enumerate(datasets, start=1):
        col_resumo, col_preview = st.columns([0.62, 0.38])

        with col_resumo:
            st.subheader(f"Arquivo {i}: {item['header'].file_name}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Versão LAS", item["header"].las_version or "-")
            col2.metric("Companhia", item["header"].company or "-")
            col3.metric("Poço", item["header"].well_name or "-")

            col4, col5, col6 = st.columns(3)
            col4.metric("Tipo de indexação", item["header"].index_type)
            col5.metric("Curva índice", item["header"].index_curve or "-")
            col6.metric("Total de registros", item["header"].total_records)

            # st.write("**Cabeçalho principal**")
            # st.json({
            #     "field_name": item["header"].field_name,
            #     "null_value": item["header"].null_value,
            #     "start": item["header"].start,
            #     "stop": item["header"].stop,
            #     "step": item["header"].step,
            #     "index_unit": item["header"].index_unit,
            # })

            st.write("**Curvas presentes**")
            st.dataframe(item["metadata_df"].head(200), width="stretch")

            st.write("**Estatísticas das curvas (somente válidas)**")
            stats_filtrado = item["stats_df"][item["stats_df"]["valid_values"] > 0].copy()
            stats_filtrado.index = stats_filtrado.index + 1

            if stats_filtrado.empty:
                st.warning("Nenhuma curva com dados válidos.")
            else:
                st.dataframe(stats_filtrado.head(200), width="stretch")

            st.write("**Unidades e descrições das curvas válidas**")

            curvas_validas = stats_filtrado["curve"].tolist()

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

            metadata_validas.index = range(1, len(metadata_validas) + 1)

            st.dataframe(metadata_validas.head(200), width="stretch")

        with col_preview:
            st.markdown("**Pré-visualização do arquivo LAS**")
            st.code(item["preview_text"], language="text")

        st.divider()

# GRAFICOS
# Usa a configuração da sidebar para montar gráficos lado a lado.
# O eixo fixo pode ser Tempo/Profundidade ou qualquer curva escolhida.
with abas[1]:
    item = item_global

    if not item["valid_curves"]:
        st.warning("Nenhuma curva válida para plot.")
    elif not variaveis_multiplas:
        st.warning("Selecione pelo menos uma variável no painel lateral.")
    else:
        metadata_df = item["metadata_df"]

        tamanho_grupo = 10

        grupos_de_curvas = [
            variaveis_multiplas[i:i + tamanho_grupo]
            for i in range(0, len(variaveis_multiplas), tamanho_grupo)
        ]

        for grupo_idx, grupo_curvas in enumerate(grupos_de_curvas, start=1):
            inicio = ((grupo_idx - 1) * tamanho_grupo) + 1
            fim = inicio + len(grupo_curvas) - 1
            curva_inicio = inicio

            st.write(f"**Curvas selecionadas {inicio} a {fim}**")

            col_desc_1, col_desc_2 = st.columns(2)

            for idx_curva, selected_curve in enumerate(grupo_curvas):
                if selected_curve == "__INDEX__":
                    continue

                curve_info = metadata_df[metadata_df["mnemonic"] == selected_curve]

                if curve_info.empty:
                    continue

                description = curve_info.iloc[0]["description"]
                unit = curve_info.iloc[0]["unit"]

                texto_curva = (
                    f"{curva_inicio}: "
                    f"**{selected_curve}** — "
                    f"Descrição: {description or '-'} | "
                    f"Unidade: {unit or '-'}"
                )

                curva_inicio = curva_inicio + 1

                if idx_curva < 5:
                    col_desc_1.markdown(texto_curva)
                else:
                    col_desc_2.markdown(texto_curva)

            fig = plot_curves_side_by_side(
                df=item["df"],
                fixed_axis=eixo_fixo,
                fixed_variable=variavel_fixa,
                selected_variables=grupo_curvas,
                title=f"Curvas {inicio} a {fim} - {item['header'].file_name}",
                index_type=item["header"].index_type,
                label_formatter=format_axis_label_global,
            )

            if fig is not None:
                st.plotly_chart(fig, width="stretch")

            st.divider()

        st.write("**Crossplot**")

        curve_x = st.selectbox("Curva X", item["valid_curves"], key="cx")
        curve_y = st.selectbox("Curva Y", item["valid_curves"], key="cy")

        if curve_x != curve_y:
            cross_fig = plot_crossplot(item["df"], curve_x, curve_y)
            st.plotly_chart(cross_fig, width="stretch")
        else:
            st.info("Escolha curvas diferentes.")

# COMPARACAO ENTRE POCOS
# Quando dois arquivos são carregados, compara curvas em comum entre eles.
with abas[2]:
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