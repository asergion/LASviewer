from __future__ import annotations

import pandas as pd


def numeric_curve_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns:
        if column in ["__INDEX__", "__INDEX_DATETIME__"]:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        valid = int(series.notna().sum())

        if valid == 0:
            continue

        total = int(series.shape[0])

        rows.append(
            {
                "curve": column,
                "total_values": total,
                "valid_values": valid,
                "null_values": total - valid,
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    return pd.DataFrame(rows).sort_values(by="curve").reset_index(drop=True)

def curves_with_valid_data(stats_df: pd.DataFrame) -> list[str]:
    if stats_df.empty:
        return []
    return stats_df["curve"].tolist()


def compare_curves_between_wells(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    curve_name: str,
) -> pd.DataFrame:
    left = pd.DataFrame({
        "index": pd.to_numeric(df1["__INDEX__"], errors="coerce"),
        "well_1": pd.to_numeric(df1[curve_name], errors="coerce"),
    })

    right = pd.DataFrame({
        "index": pd.to_numeric(df2["__INDEX__"], errors="coerce"),
        "well_2": pd.to_numeric(df2[curve_name], errors="coerce"),
    })

    merged = pd.merge(left, right, on="index", how="inner")
    return merged.sort_values("index").reset_index(drop=True)

def compute_time_metrics(df: pd.DataFrame) -> dict:
    series = pd.to_numeric(df["__INDEX__"], errors="coerce").dropna()

    if len(series) < 2:
        return {}

    series = series.sort_values().reset_index(drop=True)
    delta = series.diff().dropna()

    total = len(series)
    duration = series.max() - series.min()

    expected = int(duration) + 1 if duration > 0 else total

    coverage = (total / expected) * 100 if expected > 0 else None

    return {
        "total_registros": total,
        "duracao": float(duration),
        "cobertura_percentual": coverage,
        "delta_medio": float(delta.mean()),
        "delta_mediano": float(delta.median()),
        "delta_max": float(delta.max()),
        "delta_min": float(delta.min()),
        "gaps": int((delta > 1).sum()),
        "maior_gap": float(delta.max()),
    }

def classify_las_quality(metrics: dict) -> str:
    if not metrics:
        return "Sem dados"

    cobertura = metrics.get("cobertura_percentual", 0)
    gaps = metrics.get("gaps", 0)
    max_gap = metrics.get("maior_gap", 0)

    if cobertura < 85:
        return "🔴 Baixa qualidade (subamostrado)"

    if max_gap > 10:
        return "🔴 Gaps grandes"

    if gaps > 0:
        return "🟡 Gaps moderados"

    return "🟢 Boa qualidade"

def compare_wells_metrics(m1: dict, m2: dict, curves1: list, curves2: list) -> dict:
    return {
        "dif_total_registros": abs(m1["total_registros"] - m2["total_registros"]),
        "dif_cobertura": abs(m1["cobertura_percentual"] - m2["cobertura_percentual"]),
        "dif_delta_medio": abs(m1["delta_medio"] - m2["delta_medio"]),
        "dif_maior_gap": abs(m1["maior_gap"] - m2["maior_gap"]),
        "dif_qtd_curvas_validas": abs(len(curves1) - len(curves2)),
        "curvas_comuns": len(set(curves1).intersection(set(curves2))),
    }