#!/usr/bin/env python3
"""
Crea un gráfico de barras apiladas por barrio con las áreas principales construidas
(1er piso, 2do piso, sótano, garage, porches/piscina si existen).

Uso:
  python3 resumenes_ejecutivos/grafico_stack_areas_nomindims.py \
    --runs analysis/construction_batch/results_nomindims/construction_runs.csv \
    --out resumenes_ejecutivos/graficos_nomindims/stack_areas_por_barrio.png
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="CSV construction_runs.csv")
    ap.add_argument("--out", type=Path, default=Path("resumenes_ejecutivos/graficos_nomindims/stack_areas_por_barrio.png"))
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.runs)
    # selecciona columnas de área presentes
    cols = [c for c in ["area_1st", "area_2nd", "bsmt", "garage_area", "screen_porch", "pool_area"] if c in df.columns]
    if not cols:
        raise SystemExit("No se encontraron columnas de área en el CSV.")
    plot_df = df[["neigh"] + cols].set_index("neigh")
    # ordenar por gr_liv_area si existe
    if "gr_liv_area" in df.columns:
        plot_df = plot_df.loc[df.sort_values("gr_liv_area")["neigh"]]
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(14, 6), colormap="tab20")
    ax.set_ylabel("ft²")
    ax.set_xlabel("Barrio")
    ax.set_title("Áreas construidas por barrio (suma de componentes)")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
