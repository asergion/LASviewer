from __future__ import annotations

import json
import os
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np

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

from src.parquet_store import (
    read_selected_curves,
    downsample_df,
)

from src.las_indexer import index_las_file

def get_env() -> str:
    try:
        return str(st.secrets["LASVIEWER_ENV"])
    except Exception:
        return os.getenv("LASVIEWER_ENV", "local")


ENV = get_env()


st.set_page_config(page_title="Leitor LAS", layout="wide")

st.markdown(
    """
    <style>
    div.stDownloadButton > button {
        min-height: 24px !important;
        height: 24px !important;
        padding: 0px 10px !important;
        font-size: 12px !important;
        line-height: 1 !important;
        border-radius: 6px !important;
    }

    div.stDownloadButton > button p {
        margin: 0px !important;
        line-height: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Leitor e Analisador de Arquivos LAS")

APP_DATA_DIR = Path("app_data")
INDEXED_DIR = APP_DATA_DIR / "indexed"
UPLOADED_PARQUET_DIR = APP_DATA_DIR / "uploaded_parquet"

INDEXED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def get_execution_environment() -> str:
    value = str(ENV or "local").strip().lower()

    local_values = {"local", "servidor", "server", "caminho_local", "local_server"}
    cloud_values = {"streamlit_cloud", "cloud", "streamlit"}

    if value in local_values:
        return "local"
    if value in cloud_values:
        return "streamlit_cloud"

    return "local"


EXECUTION_ENV = get_execution_environment()
IS_LOCAL_ENV = EXECUTION_ENV == "local"
IS_STREAMLIT_CLOUD_ENV = EXECUTION_ENV == "streamlit_cloud"


def create_indexed_files_zip(metadata_path: Path, parquet_path: Path) -> bytes:
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(metadata_path, arcname=metadata_path.name)
        zip_file.write(parquet_path, arcname=parquet_path.name)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def save_uploaded_file(uploaded_file, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(uploaded_file.name).name

    with open(destination, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return destination


def candidate_parquet_names(metadata: dict, metadata_file_name: str) -> list[str]:
    names: list[str] = []

    parquet_file = metadata.get("parquet_file")
    if parquet_file:
        names.append(Path(str(parquet_file)).name)

    if metadata_file_name.endswith(".metadata.json"):
        names.append(metadata_file_name.replace(".metadata.json", ".parquet"))

    source_file = metadata.get("source_file")
    if source_file:
        names.append(f"{Path(str(source_file)).stem}.parquet")

    header = metadata.get("header", {})
    if isinstance(header, dict) and header.get("file_name"):
        names.append(f"{Path(str(header['file_name'])).stem}.parquet")

    return list(dict.fromkeys(names))


def find_parquet_path(
    metadata: dict,
    metadata_file_name: str,
    parquet_paths_by_name: dict[str, Path],
    metadata_dir: Path | None = None,
) -> Path | None:
    for absolute_key in ("parquet_absolute_path", "parquet_path"):
        absolute_value = metadata.get(absolute_key)
        if absolute_value:
            absolute_path = Path(str(absolute_value))
            if absolute_path.exists() and absolute_path.is_file():
                return absolute_path

    for parquet_name in candidate_parquet_names(metadata, metadata_file_name):
        parquet_path = parquet_paths_by_name.get(parquet_name)
        if parquet_path is not None and parquet_path.exists():
            return parquet_path

    parquet_file = metadata.get("parquet_file")
    if parquet_file:
        parquet_path = Path(str(parquet_file))
        file_name = parquet_path.name

        candidates: list[Path] = []

        if parquet_path.exists() and parquet_path.is_file():
            return parquet_path

        if metadata_dir is not None:
            candidates.extend([
                metadata_dir / parquet_path,
                metadata_dir / file_name,
            ])

        project_root = Path(__file__).resolve().parent
        candidates.extend([
            project_root / parquet_path,
            Path.cwd() / parquet_path,
            project_root / file_name,
            Path.cwd() / file_name,
            INDEXED_DIR / file_name,
            UPLOADED_PARQUET_DIR / file_name,
        ])

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

    return None


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


def get_header_value(item: dict, key: str):
    header = item["header"]

    if isinstance(header, dict):
        return header.get(key)

    return getattr(header, key, None)


def is_indexed_mode(item: dict) -> bool:
    return item.get("modo_indexado", False)

def dataframe_to_image(df: pd.DataFrame):
    df_export = df.copy()
    df_export.insert(0, "#", df_export.index)

    n_rows, n_cols = df_export.shape

    fig_width = n_cols * 1.2
    fig_height = n_rows * 0.3

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_export.values,
        colLabels=df_export.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    table.auto_set_column_width(col=list(range(n_cols)))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )
    buf.seek(0)
    plt.close(fig)

    return buf

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

    return {
        "header": header,
        "metadata_df": metadata_df,
        "df": df,
        "stats_df": stats_df,
        "valid_curves": valid_curves,
        "preview_text": preview_text,
        "index_min": index_min,
        "index_max": index_max,
        "md_min": md_min,
        "md_max": md_max,
        "time_unit": time_unit,
        "modo_indexado": False,
    }


@st.dialog("Indexar arquivo LAS", width="large")
def abrir_dialogo_indexacao():
    st.write(
        "Use esta opção para converter um arquivo LAS em arquivos otimizados "
        "`Parquet` e `metadata.json`."
    )

    las_path: Path | None = None
    output_dir_final: Path | None = None
    origem_upload = False

    if IS_STREAMLIT_CLOUD_ENV:
        st.info(
            "Ambiente configurado: **Streamlit Cloud**. "
            "Envie o arquivo LAS pelo navegador. Ao final, baixe um único `.zip` "
            "contendo o `.metadata.json` e o `.parquet`."
        )
    else:
        st.info(
            "Ambiente configurado: **Caminho local/servidor**. "
            "Selecione o arquivo LAS pelo navegador. Os arquivos indexados serão "
            "gerados na pasta local `app_data/indexed`."
        )

    uploaded_las = st.file_uploader(
        "Selecione o arquivo LAS para indexar",
        type=["las"],
        accept_multiple_files=False,
    )

    if uploaded_las is None:
        st.info("Envie um arquivo LAS para iniciar a indexação.")
    else:
        origem_upload = True
        las_path = save_uploaded_file(uploaded_las, INDEXED_DIR)
        output_dir_final = INDEXED_DIR

        st.write("**Arquivo LAS recebido:**")
        st.code(uploaded_las.name)

    if st.button("Indexar LAS", disabled=las_path is None or output_dir_final is None):
        with st.spinner("Indexando arquivo LAS..."):
            metadata = index_las_file(
                las_path=las_path,
                output_dir=output_dir_final,
            )

        metadata_path = output_dir_final / f"{las_path.stem}.metadata.json"
        parquet_path = output_dir_final / f"{las_path.stem}.parquet"

        st.success("Indexação concluída.")

        st.write("**Arquivo Parquet gerado:**")
        st.code(str(parquet_path))

        st.write("**Arquivo de metadados gerado:**")
        st.code(str(metadata_path))

        if IS_STREAMLIT_CLOUD_ENV or origem_upload:
            zip_bytes = create_indexed_files_zip(metadata_path, parquet_path)
            zip_name = f"{las_path.stem}.indexado.zip"

            st.download_button(
                "Baixar arquivos indexados (.zip)",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
            )
        else:
            st.info("Os arquivos foram salvos automaticamente na mesma pasta do LAS original.")

        st.write("**Total de registros:**")
        st.metric("Registros", metadata["total_records"])

        st.write("**Curvas válidas encontradas:**")
        st.dataframe(
            pd.DataFrame({"curves": metadata["valid_curves"]}),
            width="stretch",
        )


# SIDEBAR CONFIGURACOES
with st.sidebar:
    st.sidebar.caption(f"Ambiente: {EXECUTION_ENV}")
    st.sidebar.caption(f"Diretório: {Path.cwd()}")
    st.header("Configurações")

    if st.button("Indexar arquivo LAS", type="secondary"):
        abrir_dialogo_indexacao()

    st.divider()

    modo_leitura = st.radio(
        "Modo de visualização",
        [
            "Ler LAS diretamente",
            "Visualizar LAS indexado",
        ],
    )

    datasets = []

    if modo_leitura == "Ler LAS diretamente":
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

    else:
        if IS_STREAMLIT_CLOUD_ENV:
            st.info(
                "Ambiente configurado: **Streamlit Cloud**. Envie o `.metadata.json` "
                "e o `.parquet` correspondente."
            )

            uploaded_metadata_files = st.file_uploader(
                "Selecione um ou dois arquivos .metadata.json",
                type=["json"],
                accept_multiple_files=True,
                key="metadata_json_files",
            )

            uploaded_parquet_files = st.file_uploader(
                "Selecione os arquivos .parquet correspondentes",
                type=["parquet"],
                accept_multiple_files=True,
                key="metadata_parquet_files",
            )

            if not uploaded_metadata_files:
                st.info("Envie pelo menos um arquivo .metadata.json indexado.")
                st.stop()

            if not uploaded_parquet_files:
                st.info("Envie também o `.parquet` correspondente ao metadata.")
                st.stop()

            if len(uploaded_metadata_files) > 2:
                st.warning("Por enquanto, envie no máximo dois arquivos indexados.")
                st.stop()

            parquet_paths_by_name = {
                parquet_file.name: save_uploaded_file(parquet_file, UPLOADED_PARQUET_DIR)
                for parquet_file in uploaded_parquet_files
            }

            metadata_items = []
            for metadata_file in uploaded_metadata_files:
                metadata = json.loads(metadata_file.getvalue().decode("utf-8"))
                metadata_items.append((metadata_file.name, metadata, None))

        else:
            st.info(
                "Ambiente configurado: **Caminho local/servidor**. "
                "Selecione apenas o arquivo `.metadata.json`. "
                "O app localizará automaticamente o `.parquet` correspondente "
                "pelo caminho salvo no metadata ou na mesma pasta do arquivo indexado."
            )

            uploaded_metadata_files = st.file_uploader(
                "Selecione um ou dois arquivos .metadata.json",
                type=["json"],
                accept_multiple_files=True,
                key="local_metadata_json_files",
            )

            if not uploaded_metadata_files:
                st.info("Selecione pelo menos um arquivo .metadata.json indexado.")
                st.stop()

            if len(uploaded_metadata_files) > 2:
                st.warning("Por enquanto, selecione no máximo dois arquivos indexados.")
                st.stop()

            parquet_paths_by_name = {}
            metadata_items = []

            for metadata_file in uploaded_metadata_files:
                metadata = json.loads(metadata_file.getvalue().decode("utf-8"))

                metadata_tmp = NamedTemporaryFile(delete=False, suffix=".metadata.json")
                metadata_tmp.write(metadata_file.getvalue())
                metadata_tmp.close()
                metadata_path = Path(metadata_tmp.name)

                metadata_items.append((metadata_file.name, metadata, metadata_path.parent))

        for metadata_file_name, metadata, metadata_dir in metadata_items:
            parquet_path = find_parquet_path(
                metadata=metadata,
                metadata_file_name=metadata_file_name,
                parquet_paths_by_name=parquet_paths_by_name,
                metadata_dir=metadata_dir,
            )

            if parquet_path is None:
                st.error(
                    "Não encontrei o arquivo Parquet correspondente ao metadata "
                    f"`{metadata_file_name}`. Nomes esperados: "
                    f"{', '.join(candidate_parquet_names(metadata, metadata_file_name))}."
                )
                st.stop()

            metadata_df = pd.DataFrame(metadata["curves_metadata"])
            stats_df = pd.DataFrame(metadata["stats"])

            datasets.append(
                {
                    "header": metadata["header"],
                    "metadata_df": metadata_df,
                    "stats_df": stats_df,
                    "valid_curves": metadata["valid_curves"],
                    "preview_text": f"Arquivo indexado: {metadata.get('source_file', '-')}",
                    "parquet_file": str(parquet_path),
                    "metadata_path": metadata_dir,
                    "time_unit": metadata.get("time_unit"),
                    "index_min": metadata.get("index_min"),
                    "index_max": metadata.get("index_max"),
                    "md_min": metadata.get("md_min"),
                    "md_max": metadata.get("md_max"),
                    "tempo_min": metadata.get("tempo_min"),
                    "tempo_max": metadata.get("tempo_max"),
                    "columns": metadata.get("columns", []),
                    "modo_indexado": True,
                }
            )


with st.sidebar:
    dataset_idx_global = st.selectbox(
        "Arquivo para análise gráfica",
        options=range(len(datasets)),
        format_func=lambda x: get_header_value(datasets[x], "file_name") or f"Arquivo {x + 1}",
        key="sidebar_dataset",
    )

    item_global = datasets[dataset_idx_global]

    st.markdown("### Configuração dos eixos")

    index_type_global = get_header_value(item_global, "index_type")
    columns_global = item_global.get("columns", [])

    if (
        index_type_global == "tempo"
        and (
            (
                not is_indexed_mode(item_global)
                and "__INDEX_DATETIME__" in item_global["df"].columns
            )
            or (
                is_indexed_mode(item_global)
                and "__INDEX_DATETIME__" in columns_global
            )
        )
    ):
        axis_options_global = ["__INDEX_DATETIME__"] + item_global["valid_curves"]
        variavel_fixa_default = "__INDEX_DATETIME__"
    else:
        axis_options_global = ["__INDEX__"] + item_global["valid_curves"]
        variavel_fixa_default = "__INDEX__"

    def format_axis_label_global(value):
        index_type = get_header_value(item_global, "index_type")
        index_unit = get_header_value(item_global, "index_unit")

        if value == "__INDEX_DATETIME__":
            return "Tempo"

        if value == "__INDEX__":
            if index_type == "tempo":
                unidade = item_global.get("time_unit") or index_unit or "s"
                return f"Tempo ({unidade})"

            if index_type == "profundidade":
                return f"Profundidade ({index_unit or 'm'})"

            return "Índice"

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
        index=1 if index_type_global == "profundidade" else 0,
    )

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
            st.subheader(f"Arquivo {i}: {get_header_value(item, 'file_name')}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Versão LAS", get_header_value(item, "las_version") or "-")
            col2.metric("Companhia", get_header_value(item, "company") or "-")
            col3.metric("Poço", get_header_value(item, "well_name") or "-")

            col4, col5, col6 = st.columns(3)
            col4.metric("Tipo de indexação", get_header_value(item, "index_type") or "-")
            col5.metric("Curva índice", get_header_value(item, "index_curve") or "-")
            col6.metric("Total de registros", get_header_value(item, "total_records") or "-")

            col7, col8 = st.columns(2)

            if get_header_value(item, "index_type") == "tempo":
                tempo_min_formatado = "-"
                tempo_max_formatado = "-"

                if not is_indexed_mode(item) and "__INDEX_DATETIME__" in item["df"].columns:
                    tempo_min = item["df"]["__INDEX_DATETIME__"].min()
                    tempo_max = item["df"]["__INDEX_DATETIME__"].max()

                    tempo_min_formatado = (
                        tempo_min.strftime("%d/%m/%Y %H:%M:%S")
                        if pd.notna(tempo_min)
                        else "-"
                    )

                    tempo_max_formatado = (
                        tempo_max.strftime("%d/%m/%Y %H:%M:%S")
                        if pd.notna(tempo_max)
                        else "-"
                    )

                elif is_indexed_mode(item):
                    tempo_min = item.get("tempo_min")
                    tempo_max = item.get("tempo_max")

                    if tempo_min and tempo_max:
                        tempo_min_formatado = pd.to_datetime(tempo_min).strftime("%d/%m/%Y %H:%M:%S")
                        tempo_max_formatado = pd.to_datetime(tempo_max).strftime("%d/%m/%Y %H:%M:%S")

                    elif item.get("index_min") is not None and item.get("index_max") is not None:
                        unidade = item.get("time_unit") or "s"

                        tempo_min_formatado = pd.to_datetime(
                            item["index_min"],
                            unit=unidade,
                            utc=True,
                        ).strftime("%d/%m/%Y %H:%M:%S")

                        tempo_max_formatado = pd.to_datetime(
                            item["index_max"],
                            unit=unidade,
                            utc=True,
                        ).strftime("%d/%m/%Y %H:%M:%S")

                col7.metric("Tempo mínimo", tempo_min_formatado)
                col8.metric("Tempo máximo", tempo_max_formatado)

            elif get_header_value(item, "index_type") == "profundidade":
                profundidade_min = item.get("index_min")
                profundidade_max = item.get("index_max")

                if profundidade_min is None or profundidade_max is None:
                    stats_df = item.get("stats_df")

                    if stats_df is not None and not stats_df.empty:
                        md_stats = stats_df[stats_df["curve"] == "MD"]

                        if not md_stats.empty:
                            profundidade_min = float(md_stats.iloc[0]["min"])
                            profundidade_max = float(md_stats.iloc[0]["max"])

                col7.metric(
                    "Profundidade mínima",
                    f"{profundidade_min:.2f}" if profundidade_min is not None else "-",
                )
                col8.metric(
                    "Profundidade máxima",
                    f"{profundidade_max:.2f}" if profundidade_max is not None else "-",
                )

            if item.get("md_min") is not None:
                col9, col10 = st.columns(2)
                col9.metric("MD mínimo (m)", f"{item['md_min']:.2f}")
                col10.metric("MD máximo (m)", f"{item['md_max']:.2f}")

            st.write("**Curvas presentes**")
            st.dataframe(item["metadata_df"].head(200), width="stretch")

            stats_filtrado = item["stats_df"][item["stats_df"]["valid_values"] > 0].copy()
            stats_filtrado.index = stats_filtrado.index + 1

            if stats_filtrado.empty:
                st.warning("Nenhuma curva com dados válidos.")
            else:
                df_export = stats_filtrado.head(200)
                img = dataframe_to_image(df_export)

                col_titulo, col_botao, col_fill = st.columns([0.5, 0.2, 0.3], gap="small")

                with col_titulo:
                    st.markdown("**Estatísticas das curvas (somente válidas)**")

                with col_botao:
                    st.download_button(
                        label="⭳ PNG",
                        data=img,
                        file_name="tabela_curvas.png",
                        mime="image/png",
                        key=f"download_stats_png_{i}",
                    )

                with col_fill:
                    st.write(" ")

                st.dataframe(df_export, width="stretch")

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

            df_metadata_export = metadata_validas.head(200)
            img_metadata = dataframe_to_image(df_metadata_export)

            col_titulo_metadata, col_botao_metadata, col_fill_metadata = st.columns(
                [0.5, 0.2, 0.3],
                gap="small",
            )

            with col_titulo_metadata:
                st.markdown("**Unidades e descrições das curvas válidas**")

            with col_botao_metadata:
                st.download_button(
                    label="⭳ PNG",
                    data=img_metadata,
                    file_name="tabela_unidades_descricoes.png",
                    mime="image/png",
                    key=f"download_metadata_png_{i}",
                )

            with col_fill_metadata:
                st.write(" ")

            st.dataframe(df_metadata_export, width="stretch")

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
        if is_indexed_mode(item):
            df_plot = read_selected_curves(
                parquet_path=item["parquet_file"],
                fixed_variable=variavel_fixa,
                selected_variables=variaveis_multiplas,
                metadata_path=item.get("metadata_path"),
            )

            df_plot = downsample_df(df_plot, max_points=10_000)
        else:
            df_plot = item["df"]

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
                if selected_curve in ["__INDEX__", "__INDEX_DATETIME__"]:
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
                df=df_plot,
                fixed_axis=eixo_fixo,
                fixed_variable=variavel_fixa,
                selected_variables=grupo_curvas,
                title=f"Curvas {inicio} a {fim} - {get_header_value(item, 'file_name')}",
                index_type=get_header_value(item, "index_type"),
                label_formatter=format_axis_label_global,
            )

            if fig is not None:
                st.plotly_chart(fig, width="stretch")

            st.divider()

        st.write("**Crossplot**")

        curve_x = st.selectbox("Curva X", item["valid_curves"], key="cx")
        curve_y = st.selectbox("Curva Y", item["valid_curves"], key="cy")

        if curve_x != curve_y:
            if is_indexed_mode(item):
                crossplot_df = read_selected_curves(
                    parquet_path=item["parquet_file"],
                    fixed_variable=curve_x,
                    selected_variables=[curve_y],
                    metadata_path=item.get("metadata_path"),
                )
                crossplot_df = downsample_df(crossplot_df, max_points=10_000)
            else:
                crossplot_df = item["df"]

            cross_fig = plot_crossplot(crossplot_df, curve_x, curve_y)
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

            if is_indexed_mode(datasets[0]):
                df1 = read_selected_curves(
                    parquet_path=datasets[0]["parquet_file"],
                    fixed_variable="__INDEX__",
                    selected_variables=[curve_name],
                    metadata_path=datasets[0].get("metadata_path"),
                )
            else:
                df1 = datasets[0]["df"]

            if is_indexed_mode(datasets[1]):
                df2 = read_selected_curves(
                    parquet_path=datasets[1]["parquet_file"],
                    fixed_variable="__INDEX__",
                    selected_variables=[curve_name],
                    metadata_path=datasets[1].get("metadata_path"),
                )
            else:
                df2 = datasets[1]["df"]

            compare_df = compare_curves_between_wells(
                df1,
                df2,
                curve_name,
            )

            if compare_df.empty:
                st.warning("Não foi possível alinhar índices.")
            else:
                compare_df = downsample_df(compare_df, max_points=10_000)

                st.dataframe(compare_df.head(200), width="stretch")

                fig = plot_compare_wells(compare_df, curve_name)
                st.plotly_chart(fig, width="stretch")
