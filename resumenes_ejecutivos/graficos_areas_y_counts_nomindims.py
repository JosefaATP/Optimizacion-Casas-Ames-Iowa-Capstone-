#!/usr/bin/env python3
"""
Genera dos gráficos para la corrida results_nomindims (o similar):
- Histograma de áreas (ft²) por feature: baños completos, medios baños, cocinas, dormitorios.
- Histograma de conteos por feature: #fullbath, #halfbath, #kitchen, #beds.

Uso típico:
  python3 resumenes_ejecutivos/graficos_areas_y_counts_nomindims.py \
    --runs analysis/construction_batch/results_nomindims/construction_runs.csv \
    --xinputs analysis/construction_batch/results_nomindims/x_inputs \
    --outdir resumenes_ejecutivos/graficos_nomindims
"""

import argparse
from pathlib import Path
import glob
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="CSV de construction_runs.csv")
    ap.add_argument("--xinputs", type=Path, required=True, help="Carpeta con cost_breakdown_*.csv")
    ap.add_argument("--outdir", type=Path, default=Path("resumenes_ejecutivos/graficos_nomindims"))
    return ap.parse_args()


def load_areas(xdir: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(xdir / "cost_breakdown_*.csv")):
        df = pd.read_csv(f)
        neigh = Path(f).name.split("cost_breakdown_")[1].split("_lot")[0]
        def grab(name: str):
            r = df[df["var"] == name]
            if r.empty:
                return None
            try:
                return float(r["value"].iloc[0])
            except Exception:
                return None
        rows.append({
            "neigh": neigh,
            "AreaFullBath": grab("AreaFullBath"),
            "AreaHalfBath": grab("AreaHalfBath"),
            "AreaKitchen": grab("AreaKitchen"),
            "AreaBedroom": grab("AreaBedroom"),
        })
    return pd.DataFrame(rows)


def plot_hist_grid(df: pd.DataFrame, cols, title: str, outpath: Path, bins=10):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, col, color in zip(axes, cols, colors):
        vals = df[col].dropna()
        if vals.empty:
            continue
        # histograma clásico para valores continuos
        ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="black", histtype="bar")
        ax.set_title(col)
        ax.set_xlabel("ft²")
        ax.set_ylabel("Frecuencia")
    fig.suptitle(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_hist_counts_grid(df: pd.DataFrame, cols, title: str, outpath: Path, bins=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, col, color in zip(axes, cols, colors):
        vals = df[col].dropna()
        if vals.empty:
            continue
        counts = vals.value_counts().sort_index()
        x = counts.index
        ax.bar(x, counts.values, color=color, alpha=0.8, edgecolor="black", width=0.6)
        ax.set_xticks(x)
        ax.set_title(col)
        ax.set_xlabel("Cantidad")
        ax.set_ylabel("Frecuencia")
    fig.suptitle(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    areas = load_areas(args.xinputs)
    runs = pd.read_csv(args.runs)
    # histogramas de áreas
    plot_hist_grid(
        areas,
        ["AreaFullBath", "AreaHalfBath", "AreaKitchen", "AreaBedroom"],
        "Distribución de áreas (ft²) por ambiente",
        args.outdir / "hist_areas_ft2.png",
        bins=10,
    )
    # histogramas de conteos
    plot_hist_counts_grid(
        runs,
        ["fullbath", "halfbath", "kitchen", "beds"],
        "Distribución de cantidades de ambientes",
        args.outdir / "hist_counts.png",
        bins=None,
    )


if __name__ == "__main__":
    main()
