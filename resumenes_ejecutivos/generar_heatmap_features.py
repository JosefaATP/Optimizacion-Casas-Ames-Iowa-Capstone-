#!/usr/bin/env python3
"""
Genera un heatmap de porcentaje de presencia (>0) de cada feature en las
soluciones de construcción y lo guarda en resumenes_ejecutivos/heatmap_freq_features.png.

Uso:
  python3 resumenes_ejecutivos/generar_heatmap_features.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Evita problemas de cache de fonts en entornos restringidos
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    path = Path("analysis/construction_batch/results/construction_runs.csv")
    df = pd.read_csv(path)
    cols = [
        "beds",
        "fullbath",
        "halfbath",
        "gr_liv_area",
        "garage_area",
        "screen_porch",
        "pool_area",
        "kitchen",
        "heating_qc",
        "floor1",
        "floor2",
        "area_1st",
        "area_2nd",
        "bsmt",
    ]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols]

    # Umbrales mínimos por feature (0 si solo queremos “>0”)
    thresholds = {
        "beds": 1.0,
        "fullbath": 1.0,
        "halfbath": 0.0,
        "kitchen": 1.0,
        "gr_liv_area": 0.0,
        "garage_area": 0.0,
        "screen_porch": 0.0,
        "pool_area": 0.0,
        "heating_qc": 0.0,
        "floor1": 1.0,   # primer piso activo
        "floor2": 0.0,
        "area_1st": 0.0,
        "area_2nd": 0.0,
        "bsmt": 0.0,
    }

    rows = []
    for c in cols:
        thr = thresholds.get(c, 0.0)
        pct = (sub[c].astype(float) >= thr).mean() * 100.0
        rows.append({"feature": c, "% cumple mínimo": pct, "umbral": thr})

    heat_df = pd.DataFrame(rows).set_index("feature")[["% cumple mínimo"]]

    plt.figure(figsize=(4, len(heat_df) * 0.35 + 1))
    data = heat_df["% cumple mínimo"].values.reshape(-1, 1)
    im = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, label="% cumple mínimo")
    plt.yticks(range(len(heat_df)), heat_df.index)
    plt.xticks([0], ["%"])
    for i, v in enumerate(heat_df["% cumple mínimo"].values):
        plt.text(0, i, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="black")
    plt.title("Cumplimiento de mínimos por feature")
    plt.tight_layout()

    outdir = Path("resumenes_ejecutivos")
    outdir.mkdir(exist_ok=True)
    out_png = outdir / "heatmap_freq_features.png"
    plt.savefig(out_png, dpi=180)
    print(f"Guardado {out_png}")


if __name__ == "__main__":
    main()
