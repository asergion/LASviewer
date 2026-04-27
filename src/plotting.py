from __future__ import annotations
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_curves_side_by_side(
    df: pd.DataFrame,
    fixed_axis: str,
    fixed_variable: str,
    selected_variables: list[str],
    title: str,
    index_type: str,
    label_formatter,
):
    if not selected_variables:
        return None

    max_cols = 10
    total_plots = len(selected_variables)
    total_rows = (total_plots + max_cols - 1) // max_cols
    total_cols = min(max_cols, total_plots)

    subplot_titles = [label_formatter(v) for v in selected_variables]

    fig = make_subplots(
        rows=total_rows,
        cols=total_cols,
        shared_yaxes=fixed_axis == "Eixo Y",
        shared_xaxes=fixed_axis == "Eixo X",
        subplot_titles=subplot_titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.08,
    )

    for idx, variable in enumerate(selected_variables):
        row = (idx // max_cols) + 1
        col = (idx % max_cols) + 1

        if fixed_axis == "Eixo Y":
            x_column = variable
            y_column = fixed_variable
        else:
            x_column = fixed_variable
            y_column = variable

        chart_df = pd.DataFrame({
            x_column: df[x_column] if x_column == "__INDEX_DATETIME__" else pd.to_numeric(df[x_column], errors="coerce"),
            y_column: df[y_column] if y_column == "__INDEX_DATETIME__" else pd.to_numeric(df[y_column], errors="coerce"),
        }).dropna()

        fig.add_trace(
            go.Scatter(
                x=chart_df[x_column],
                y=chart_df[y_column],
                mode="lines",
                name=label_formatter(variable),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text=label_formatter(x_column),
            row=row,
            col=col,
        )

        if col == 1:
            fig.update_yaxes(
                title_text=label_formatter(y_column),
                row=row,
                col=col,
            )

    if fixed_axis == "Eixo Y" and fixed_variable == "__INDEX__" and index_type == "profundidade":
        fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title=title,
        height=max(650, total_rows * 520),
    )

    # Se estiver usando datetime, ajustar eixo como data
    if any(v == "__INDEX_DATETIME__" for v in [fixed_variable] + selected_variables):
        fig.update_xaxes(
            type="date",
            tickformat="%d/%m/%Y<br>%H:%M:%S",
        )
        
    return fig


def plot_single_curve(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    index_type: str,
):
    chart_df = pd.DataFrame({
        x_column: pd.to_numeric(df[x_column], errors="coerce"),
        y_column: pd.to_numeric(df[y_column], errors="coerce"),
    }).dropna()

    fig = px.line(
        chart_df,
        x=x_column,
        y=y_column,
        title=title,
        labels={
            "__INDEX__": "Tempo" if index_type == "tempo" else "Profundidade/Índice",
        },
    )

    if y_column == "__INDEX__" and index_type == "profundidade":
        fig.update_yaxes(autorange="reversed")

    return fig


def plot_crossplot(df: pd.DataFrame, curve_x: str, curve_y: str):
    chart_df = pd.DataFrame({
        curve_x: pd.to_numeric(df[curve_x], errors="coerce"),
        curve_y: pd.to_numeric(df[curve_y], errors="coerce"),
    }).dropna()

    fig = px.scatter(
        chart_df,
        x=curve_x,
        y=curve_y,
        title=f"Crossplot: {curve_x} x {curve_y}",
    )
    return fig


def plot_histogram(df: pd.DataFrame, curve_name: str):
    chart_df = pd.DataFrame({
        curve_name: pd.to_numeric(df[curve_name], errors="coerce"),
    }).dropna()

    fig = px.histogram(
        chart_df,
        x=curve_name,
        title=f"Histograma - {curve_name}",
    )
    return fig


def plot_compare_wells(compare_df: pd.DataFrame, curve_name: str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=compare_df["well_1"],
        y=compare_df["index"],
        mode="lines",
        name="Poço 1",
    ))

    fig.add_trace(go.Scatter(
        x=compare_df["well_2"],
        y=compare_df["index"],
        mode="lines",
        name="Poço 2",
    ))

    fig.update_layout(
        title=f"Comparação da curva {curve_name}",
        xaxis_title=curve_name,
        yaxis_title="Índice",
    )
    fig.update_yaxes(autorange="reversed")
    return fig