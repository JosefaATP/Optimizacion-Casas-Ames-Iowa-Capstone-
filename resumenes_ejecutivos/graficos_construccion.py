#!/usr/bin/env python3
"""
Genera varios gráficos para analizar qué se construye en las corridas de construcción.
Soporta filtrar por categoría de barrio (baja/media/alta) y usa tanto construction_runs.csv
como los X_input completos exportados (carpeta x_inputs).

Gráficos generados (se guardan en --outdir):
 - barras horizontales de distribución (%) para beds, fullbath, halfbath, kitchen
 - boxplots de áreas clave (GrLiv, 1st/2nd, Bsmt, Garage, Porches/Deck)
 - barras de frecuencia (top 10) para materiales/estilos: Exterior1st/2nd, Roof Style/Matl,
   Foundation, Garage Finish, House Style
 - heatmap de % de presencia (>0) por feature numérica (omitidas las obligatorias y dummies)

Ejemplo:
  python3 resumenes_ejecutivos/graficos_construccion.py \
    --runs analysis/construction_batch/results_fullfeatures/construction_runs.csv \
    --xinputs analysis/construction_batch/results_fullfeatures/x_inputs \
    --category all \
    --outdir resumenes_ejecutivos/graficos_all
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LOW = {"MeadowV", "IDOTRR", "BrDale", "OldTown", "BrkSide", "Edwards", "SWISU", "Sawyer", "NPkVill", "Blueste"}
MID = {"Landmrk", "No aplicames", "Mitchel", "SawyerW", "NWAmes", "Gilbert", "Greens", "Blmngtn", "CollgCr"}
HIGH = {"Crawfor", "ClearCr", "Somerst", "Timber", "Veenker", "GrnHill", "NridgHt", "StoneBr", "NoRidge"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Gráficos de sensibilidad de construcción")
    ap.add_argument("--runs", type=Path, required=True, help="Ruta a construction_runs.csv")
    ap.add_argument("--xinputs", type=Path, required=True, help="Carpeta con x_input_*.csv")
    ap.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "baja", "media", "alta"],
        help="Filtra por categoría de barrio (all = sin filtro)",
    )
    ap.add_argument("--outdir", type=Path, default=Path("resumenes_ejecutivos/graficos_all"))
    return ap.parse_args()


def load_runs(runs_path: Path, category: str) -> pd.DataFrame:
    df = pd.read_csv(runs_path)
    if "neigh" not in df.columns:
        raise ValueError("runs CSV debe incluir columna 'neigh'.")
    if category == "baja":
        df = df[df["neigh"].isin(LOW)]
    elif category == "media":
        df = df[df["neigh"].isin(MID)]
    elif category == "alta":
        df = df[df["neigh"].isin(HIGH)]
    return df.reset_index(drop=True)


def load_xinputs(xdir: Path, neighborhoods: List[str]) -> pd.DataFrame:
    files = []
    for f in xdir.glob("x_input_*.csv"):
        nb = f.name.split("_")[2]  # x_input_<nb>_lot...
        if nb in neighborhoods:
            files.append(f)
    if not files:
        raise FileNotFoundError("No se encontraron x_input_* para los barrios filtrados.")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["neigh"] = f.name.split("_")[2]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_bar_distribution(df: pd.DataFrame, outdir: Path):
    feats = [c for c in ["beds", "fullbath", "halfbath", "kitchen"] if c in df.columns]
    if not feats:
        return
    n = len(feats)
    fig, axes = plt.subplots(n, 1, figsize=(6, 1.6 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats):
        vc = df[feat].value_counts().sort_index()
        perc = vc / len(df) * 100
        ax.barh([str(v) for v in perc.index], perc.values, color="#4caf50")
        for y, v in enumerate(perc.values):
            ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_title(f"Distribución % de {feat}")
        ax.set_xlabel("% de casas")
    fig.savefig(outdir / "barras_programa.png", dpi=180)
    plt.close(fig)


def plot_box_areas(df: pd.DataFrame, outdir: Path):
    area_cols = [
        ("gr_liv_area", "Gr Liv Area"),
        ("area_1st", "1st Flr SF"),
        ("area_2nd", "2nd Flr SF"),
        ("bsmt", "Total Bsmt SF"),
        ("garage_area", "Garage Area"),
        ("wood_deck", "Wood Deck SF"),
        ("open_porch", "Open Porch SF"),
        ("enclosed_porch", "Enclosed Porch"),
        ("screen_porch", "Screen Porch"),
    ]
    cols = [c for c, _ in area_cols if c in df.columns]
    if not cols:
        return
    data = [df[c].dropna() for c in cols]
    labels = [label for c, label in area_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(cols) + 2))
    ax.boxplot(data, labels=labels, vert=False, showmeans=True, meanline=True)
    ax.set_title("Distribución de áreas (ft²)")
    ax.set_xlabel("ft²")
    fig.tight_layout()
    fig.savefig(outdir / "box_areas.png", dpi=180)
    plt.close(fig)


def plot_categorical_freq(df: pd.DataFrame, outdir: Path):
    # dummies en x_inputs suelen tener prefix "Exterior 1st_", etc.
    categories: Dict[str, str] = {
        "Exterior1": "Exterior 1st_",
        "Exterior2": "Exterior 2nd_",
        "RoofStyle": "Roof Style_",
        "RoofMatl": "Roof Matl_",
        "Foundation": "Foundation_",
        "GarageFinish": "Garage Finish_",
        "HouseStyle": "House Style_",
    }
    for name, prefix in categories.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        freq = (df[cols] > 0.5).sum().sort_values(ascending=False)
        top = freq.head(10)
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(top) + 1))
        ax.barh(top.index.str.replace(prefix, ""), top.values, color="#2196f3")
        ax.invert_yaxis()
        ax.set_title(f"Top {len(top)} {name} (conteo de casas)")
        ax.set_xlabel("N° de casas")
        fig.tight_layout()
        fig.savefig(outdir / f"freq_{name.lower()}.png", dpi=180)
        plt.close(fig)


def plot_heatmap(df: pd.DataFrame, outdir: Path):
    # omitir obligatorias y dummies
    omit_prefix = ("Neighborhood_",)
    omit_cols = {"fullbath", "kitchen", "beds", "floor1", "overall_qual", "kitchen_qual", "overall_cond", "heating_qc"}
    cols = [
        c
        for c in df.columns
        if c.lower() not in omit_cols and not any(c.startswith(p) for p in omit_prefix) and df[c].nunique() > 1
    ]
    if not cols:
        return
    stats = {}
    for c in cols:
        s_num = pd.to_numeric(df[c], errors="coerce")
        pct = (s_num.fillna(0) > 0).mean() * 100.0
        stats[c] = pct
    heat_df = pd.DataFrame.from_dict(stats, orient="index", columns=["% > 0"]).sort_values("% > 0", ascending=False)
    data = heat_df["% > 0"].values.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(4, len(heat_df) * 0.25 + 1))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="% > 0")
    ax.set_yticks(range(len(heat_df)))
    ax.set_yticklabels(heat_df.index, fontsize=7)
    ax.set_xticks([0]); ax.set_xticklabels(["%"])
    for i, v in enumerate(heat_df["% > 0"].values):
        ax.text(0, i, f"{v:.1f}%", ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Presencia de features (>0)")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_features.png", dpi=180)
    plt.close(fig)


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs_df = load_runs(args.runs, args.category)
    neighborhoods = runs_df["neigh"].unique().tolist()
    xin_df = load_xinputs(args.xinputs, neighborhoods)

    # Unir conteos del runs (beds/fullbath/halfbath/kitchen) por si no están en x_inputs
    for col in ["beds", "fullbath", "halfbath", "kitchen", "gr_liv_area", "garage_area"]:
        if col in runs_df.columns and col not in xin_df.columns:
            xin_df[col] = runs_df.set_index("neigh")[col].reindex(xin_df["neigh"]).values

    plot_bar_distribution(xin_df, args.outdir)
    plot_box_areas(xin_df, args.outdir)
    plot_categorical_freq(xin_df, args.outdir)
    plot_heatmap(xin_df, args.outdir)
    print(f"[INFO] Gráficos guardados en {args.outdir}")


if __name__ == "__main__":
    main()
