from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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