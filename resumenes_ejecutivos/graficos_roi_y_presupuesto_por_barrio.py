#!/usr/bin/env python3
"""
Grafica ROI y costo utilizado por barrio, coloreando según categoría (baja/media/alta).

Uso recomendado (resultados fullfeatures):
  python3 resumenes_ejecutivos/graficos_roi_y_presupuesto_por_barrio.py \
    --runs analysis/construction_batch/results_fullfeatures/construction_runs.csv \
    --outdir resumenes_ejecutivos/graficos_por_barrio
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Categorías de barrio
LOW = {"MeadowV", "IDOTRR", "BrDale", "OldTown", "BrkSide", "Edwards", "SWISU", "Sawyer", "NPkVill", "Blueste"}
MID = {"Landmrk", "No aplicames", "Mitchel", "SawyerW", "NWAmes", "Gilbert", "Greens", "Blmngtn", "CollgCr"}
HIGH = {"Crawfor", "ClearCr", "Somerst", "Timber", "Veenker", "GrnHill", "NridgHt", "StoneBr", "NoRidge"}

# Colores por categoría
COLORS: Dict[str, str] = {"baja": "#2ca02c", "media": "#1f77b4", "alta": "#d62728"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Graficos ROI y costo por barrio, coloreados por categoría.")
    ap.add_argument("--runs", type=Path, required=True, help="CSV de runs (construction_runs.csv).")
    ap.add_argument("--outdir", type=Path, default=Path("resumenes_ejecutivos/graficos_por_barrio"), help="Carpeta de salida.")
    return ap.parse_args()


def cat(nb: str) -> str:
    if nb in LOW:
        return "baja"
    if nb in MID:
        return "media"
    if nb in HIGH:
        return "alta"
    return "otro"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "roi" not in df.columns:
        df["roi"] = (df["y_price"] - df["cost"]) / df["cost"]
    df["cat"] = df["neigh"].apply(cat)
    df = df[df["cat"] != "otro"].copy()
    # Si hay múltiples corridas por barrio, promediamos ROI y costo
    agg = df.groupby(["neigh", "cat"], as_index=False).agg({"roi": "mean", "cost": "mean"})
    return agg


def plot_bar(df: pd.DataFrame, ycol: str, ylabel: str, outpath: Path, ascending: bool = False):
    # ordenar globalmente por valor, manteniendo colores por categoría
    df = df.sort_values(ycol, ascending=ascending)
    colors = [COLORS.get(c, "#7f7f7f") for c in df["cat"]]

    plt.figure(figsize=(10, max(3, len(df) * 0.4)))
    plt.barh(df["neigh"], df[ycol], color=colors, edgecolor="black")
    plt.xlabel(ylabel)
    plt.ylabel("Barrio")
    # anotaciones con valores
    for i, (val, neigh) in enumerate(zip(df[ycol], df["neigh"])):
        plt.text(val, i, f" {val:.3f}" if ycol == "roi" else f" {val:,.0f}", va="center", ha="left", fontsize=8)
    # leyenda manual
    handles = [plt.Line2D([0], [0], marker="s", color="w", label=lab, markerfacecolor=col, markersize=10) for lab, col in COLORS.items()]
    plt.legend(handles=handles, title="Categoría", loc="lower right")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    print(f"Guardado {outpath}")
    plt.close()


def main():
    args = parse_args()
    df = load_data(args.runs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Orden descendente (mayor a menor) por claridad; cambiar ascending=True para invertir.
    plot_bar(df, "roi", "ROI ( (precio - costo)/costo )", args.outdir / "roi_por_barrio.png", ascending=False)
    plot_bar(df, "cost", "Costo (capex ejecutado)", args.outdir / "costo_por_barrio.png", ascending=False)


if __name__ == "__main__":
    main()
