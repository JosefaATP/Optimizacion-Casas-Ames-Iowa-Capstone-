"""
Analiza impactos por barrio usando `summary_by_neighborhood_budget.csv` (generado por analysis_summary.py).

Calcula:
- Net gain promedio por barrio y presupuesto.
- Deltas: 50k-20k, 100k-50k, 100k-20k.
- Top/bottom barrios más y menos impactados para cada delta.

Salidas:
- CSV con net_gain y deltas por barrio: `optimization/sensibilidad_remodelacion/analisis/neighborhood_deltas.csv`
- Reporte en texto por consola con top/bottom para cada delta.

Uso:
  python3 optimization/sensibilidad_remodelacion/analisis_scripts/top_bottom_neighborhoods.py
    --summary optimization/sensibilidad_remodelacion/analisis/summary_by_neighborhood_budget.csv
    --outdir  optimization/sensibilidad_remodelacion/analisis
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# asegurar repo en path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("optimization/sensibilidad_remodelacion/analisis/summary_by_neighborhood_budget.csv"),
        help="CSV generado por analysis_summary.py con promedios por barrio y presupuesto.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("optimization/sensibilidad_remodelacion/analisis"),
        help="Carpeta donde se guardan los CSV de salida."
    )
    ap.add_argument(
        "--topn",
        type=int,
        default=5,
        help="Número de barrios a mostrar en top/bottom por delta."
    )
    return ap.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"neighborhood", "budget", "net_gain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")
    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    # pivot a budgets -> columnas para permitir deltas
    pivot = df.pivot(index="neighborhood", columns="budget", values="net_gain")
    # aseguramos columnas estándar
    for b in (20000, 50000, 100000):
        if b not in pivot.columns:
            pivot[b] = pd.NA
    pivot = pivot[[20000, 50000, 100000]]
    pivot = pivot.rename(columns={20000: "net_gain_20k", 50000: "net_gain_50k", 100000: "net_gain_100k"})
    pivot["delta_50k_20k"] = pivot["net_gain_50k"] - pivot["net_gain_20k"]
    pivot["delta_100k_50k"] = pivot["net_gain_100k"] - pivot["net_gain_50k"]
    pivot["delta_100k_20k"] = pivot["net_gain_100k"] - pivot["net_gain_20k"]
    return pivot.reset_index()


def rank_and_print(df: pd.DataFrame, delta_col: str, topn: int) -> None:
    sorted_df = df.sort_values(delta_col, ascending=False)
    top = sorted_df.head(topn)
    bottom = sorted_df.tail(topn)
    print(f"\n== {delta_col} ==")
    print(f"Top {topn}:")
    print(top[["neighborhood", delta_col]].to_string(index=False))
    print(f"\nBottom {topn}:")
    print(bottom[["neighborhood", delta_col]].to_string(index=False))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_summary(args.summary)
    deltas = compute_deltas(df)

    out_csv = args.outdir / "neighborhood_deltas.csv"
    deltas.to_csv(out_csv, index=False)
    print(f"Guardado {out_csv}")

    for delta_col in ["delta_50k_20k", "delta_100k_50k", "delta_100k_20k"]:
        rank_and_print(deltas, delta_col, args.topn)


if __name__ == "__main__":
    main()
