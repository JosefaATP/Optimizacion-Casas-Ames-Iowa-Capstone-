#!/usr/bin/env python3
"""
Grafica Gr Liv Area descompuesta por componentes (cocina, dormitorios, baños,
otros) usando cost_breakdown + construction_runs.

Uso:
  python3 resumenes_ejecutivos/grafico_stack_grliv_components.py \
    --runs analysis/construction_batch/results_nomindims/construction_runs.csv \
    --xinputs analysis/construction_batch/results_nomindims/x_inputs \
    --out resumenes_ejecutivos/graficos_nomindims/stack_grliv_components.png
"""

import argparse
from pathlib import Path
import glob
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMP_VARS = [
    "AreaKitchen",
    "AreaBedroom",
    "AreaFullBath",
    "AreaHalfBath",
    "AreaOther1 @construction",
    "AreaOther2 @construction",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--xinputs", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("resumenes_ejecutivos/graficos_nomindims/stack_grliv_components.png"),
    )
    return ap.parse_args()


def load_components(xdir: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(xdir / "cost_breakdown_*.csv")):
        neigh = Path(f).name.split("cost_breakdown_")[1].split("_lot")[0]
        df = pd.read_csv(f)
        row = {"neigh": neigh}
        for var in COMP_VARS:
            r = df[df["var"] == var]
            row[var] = float(r["value"].iloc[0]) if not r.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    runs = pd.read_csv(args.runs)
    comps = load_components(args.xinputs)
    df = runs[["neigh", "gr_liv_area"]].merge(comps, on="neigh", how="left")
    # calcula residual sobre rasante
    df["residual"] = df["gr_liv_area"] - df[COMP_VARS].sum(axis=1)
    # ordenar por gr_liv_area
    df = df.sort_values("gr_liv_area", ascending=True).set_index("neigh")
    stack_cols = COMP_VARS + ["residual"]
    ax = df[stack_cols].plot(kind="bar", stacked=True, figsize=(14, 6), colormap="tab20")
    ax.set_ylabel("ft²")
    ax.set_xlabel("Barrio")
    ax.set_title("Gr Liv Area descompuesta por componentes")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
