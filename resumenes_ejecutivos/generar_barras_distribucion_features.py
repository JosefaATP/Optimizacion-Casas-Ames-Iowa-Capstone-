#!/usr/bin/env python3
"""
Genera gráficos de barras horizontales con la distribución (%) de valores
para features discretas (beds, fullbath, halfbath, kitchen) en
analysis/construction_batch/results/construction_runs.csv.

Salida: resumenes_ejecutivos/barras_distribucion_features.png

Ejecutar:
  python3 resumenes_ejecutivos/generar_barras_distribucion_features.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    path = Path("analysis/construction_batch/results/construction_runs.csv")
    df = pd.read_csv(path)

    features = ["beds", "fullbath", "halfbath", "kitchen"]
    features = [f for f in features if f in df.columns]
    if not features:
        raise SystemExit("No se encontraron columnas beds/fullbath/halfbath/kitchen en construction_runs.csv")

    n = len(features)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6, 1.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        vc = df[feat].value_counts(dropna=False).sort_index()
        perc = vc / len(df) * 100.0
        ax.barh([str(v) for v in perc.index], perc.values, color="#4caf50")
        for y, v in enumerate(perc.values):
            ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_title(f"Distribución % de {feat}")
        ax.set_xlabel("% de casas")
        ax.set_ylabel(f"valor de {feat}")
        ax.set_xlim(0, max(perc.values) * 1.2 if len(perc) else 1)

    outdir = Path("resumenes_ejecutivos")
    outdir.mkdir(exist_ok=True)
    out_png = outdir / "barras_distribucion_features.png"
    fig.savefig(out_png, dpi=180)
    print(f"Guardado {out_png}")


if __name__ == "__main__":
    main()
