#!/usr/bin/env python3
"""
Genera un heatmap con el % de casos donde cada feature optimizada es > 0,
usando los X_input completos exportados por run_opt (carpeta x_inputs).
Omite las columnas "obligatorias" (p.ej., fullbath/kitchen/beds/floor1).

Uso:
  python3 resumenes_ejecutivos/generar_heatmap_all_features.py \
      --xinputs-dir analysis/construction_batch/results_fullfeatures/x_inputs \
      --out resumenes_ejecutivos/heatmap_all_features.png
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Heatmap de % de presencia por feature (X_input).")
    ap.add_argument(
        "--xinputs-dir",
        type=Path,
        default=Path("analysis/construction_batch/results_fullfeatures/x_inputs"),
        help="Carpeta con los CSV x_input_* por barrio.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("resumenes_ejecutivos/heatmap_all_features.png"),
        help="Ruta del PNG de salida.",
    )
    ap.add_argument(
        "--omit",
        nargs="*",
        default=["fullbath", "kitchen", "beds", "floor1", "overall_qual", "kitchen_qual", "overall_cond", "heating_qc"],
        help="Columnas obligatorias a omitir del heatmap.",
    )
    ap.add_argument(
        "--omit-prefix",
        nargs="*",
        default=["Neighborhood_"],
        help="Prefijos de columnas a omitir (ej. dummies de barrio).",
    )
    return ap.parse_args()


def load_all_xinputs(xdir: Path) -> pd.DataFrame:
    files = sorted(xdir.glob("x_input_*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {xdir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    args = parse_args()

    df = load_all_xinputs(args.xinputs_dir)
    # filtra columnas omitidas y prefijos (ej. Neighborhood_*)
    omit_set = {o.lower() for o in args.omit}
    prefix_set = tuple(args.omit_prefix)
    cols = [
        c for c in df.columns
        if c.lower() not in omit_set and not any(c.startswith(pref) for pref in prefix_set)
    ]
    df = df[cols]

    # descarta columnas constantes (sin variación)
    cols = [c for c in cols if df[c].nunique(dropna=False) > 1]
    df = df[cols]

    # porcentaje de casos con valor > 0 (num) o distinto de 0 (si no num)
    stats = {}
    for c in cols:
        s = df[c]
        try:
            s_num = pd.to_numeric(s, errors="coerce")
            pct = (s_num.fillna(0) > 0).mean() * 100.0
        except Exception:
            pct = (s.astype(str) != "0").mean() * 100.0
        stats[c] = pct

    heat_df = pd.DataFrame.from_dict(stats, orient="index", columns=["% > 0"]).sort_values("% > 0", ascending=False)

    plt.figure(figsize=(4, len(heat_df) * 0.25 + 1))
    data = heat_df["% > 0"].values.reshape(-1, 1)
    im = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, label="% > 0")
    plt.yticks(range(len(heat_df)), heat_df.index)
    plt.xticks([0], ["%"])
    for i, v in enumerate(heat_df["% > 0"].values):
        plt.text(0, i, f"{v:.1f}%", ha="center", va="center", fontsize=7, color="black")
    plt.title("Presencia de features en X_input (>0)")
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
