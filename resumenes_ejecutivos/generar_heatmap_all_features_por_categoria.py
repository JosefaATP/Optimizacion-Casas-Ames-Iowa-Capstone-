#!/usr/bin/env python3
"""
Genera un heatmap combinado (features x categorías baja/media/alta) con el %
de casos en que cada feature es >0, usando los X_input completos.

Uso:
  python3 resumenes_ejecutivos/generar_heatmap_all_features_por_categoria.py \
    --xinputs analysis/construction_batch/results_fullfeatures/x_inputs \
    --out resumenes_ejecutivos/heatmap_all_features_por_cat.png
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
    ap = argparse.ArgumentParser(description="Heatmap por categoría de features >0.")
    ap.add_argument(
        "--xinputs",
        type=Path,
        default=Path("analysis/construction_batch/results_fullfeatures/x_inputs"),
        help="Carpeta con los x_input_*.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("resumenes_ejecutivos/heatmap_all_features_por_cat.png"),
        help="Ruta del PNG de salida.",
    )
    ap.add_argument(
        "--omit",
        nargs="*",
        default=["fullbath", "kitchen", "beds", "floor1", "overall_qual", "kitchen_qual", "overall_cond", "heating_qc"],
        help="Columnas a omitir (obligatorias/calidades).",
    )
    ap.add_argument(
        "--omit-prefix",
        nargs="*",
        default=["Neighborhood_"],
        help="Prefijos a omitir (p.ej. dummies de barrio).",
    )
    return ap.parse_args()


def load_xinputs(xdir: Path) -> pd.DataFrame:
    files = sorted(xdir.glob("x_input_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay x_input_*.csv en {xdir}")
    dfs = []
    for f in files:
        nb = f.name.split("_")[2]
        df = pd.read_csv(f)
        df["neigh"] = nb
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    def cat(nb: str) -> str:
        if nb in LOW:
            return "baja"
        if nb in MID:
            return "media"
        if nb in HIGH:
            return "alta"
        return "otro"

    df["cat"] = df["neigh"].apply(cat)
    return df[df["cat"] != "otro"].reset_index(drop=True)


def filter_cols(df: pd.DataFrame, omit: List[str], omit_prefix: List[str]) -> List[str]:
    omit_set = {o.lower() for o in omit}
    omit_set.add("cat")  # columna auxiliar de categoría
    prefixes = tuple(omit_prefix)
    forbidden_substrings = [
        "qual",  # calidades (overall_qual, kitchen_qual, etc.)
        "1st",  # 1st flr / first floor
        "2nd",  # 2nd flr
        "lot frontage",
        "garage cars",
        "bedroom abvgr",
        "neigh",  # columna auxiliar de barrio
        "total bsmt sf",
        "bsmtfin sf2",
        "bsmtfin sf 2",
        "fireplace qu",
        "totrms abvgrd",
        "heating qc",
    ]
    cols = [
        c for c in df.columns if c.lower() not in omit_set and not any(c.startswith(pref) for pref in prefixes)
    ]
    # descarta columnas de áreas explícitas si no se quieren (gr_liv_area, area_1st/2nd, etc. contienen "area")
    cols = [c for c in cols if "area" not in c.lower()]
    # descarta columnas que contengan subcadenas prohibidas
    cols = [c for c in cols if not any(sub in c.lower() for sub in forbidden_substrings)]
    # descarta constantes
    cols = [c for c in cols if df[c].nunique(dropna=False) > 1]
    return cols


def compute_presence(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    stats: Dict[str, Dict[str, float]] = {}
    for c in cols:
        stats[c] = {}
        for cat, sub in df.groupby("cat"):
            s_num = pd.to_numeric(sub[c], errors="coerce")
            pct = (s_num.fillna(0) > 0).mean() * 100.0
            stats[c][cat] = pct
    out = pd.DataFrame(stats).T.fillna(0)
    # ordenar por media total
    out["mean_all"] = out.mean(axis=1)
    out = out.sort_values("mean_all", ascending=False).drop(columns=["mean_all"])
    return out


def plot_heatmap(mat: pd.DataFrame, outpath: Path):
    # asegura el orden de columnas baja→media→alta si existen
    cols_order = [c for c in ["baja", "media", "alta"] if c in mat.columns] + [
        c for c in mat.columns if c not in ("baja", "media", "alta")
    ]
    mat = mat[cols_order]

    plt.figure(figsize=(5, len(mat) * 0.25 + 1))
    im = plt.imshow(mat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, label="% > 0")
    plt.yticks(range(len(mat)), mat.index, fontsize=7)
    plt.xticks(range(mat.shape[1]), mat.columns)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=6, color="black")
    plt.title("Presencia de features (>0) por categoría")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    print(f"Guardado {outpath}")


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    args = parse_args()
    df = load_xinputs(args.xinputs)
    df = add_category(df)
    cols = filter_cols(df, args.omit, args.omit_prefix)
    mat = compute_presence(df, cols)
    plot_heatmap(mat, args.out)


if __name__ == "__main__":
    main()
