"""
Resumen de sensibilidad:
 - Agrega métricas por presupuesto/barrio/percentil a partir de detalles.jsonl.
 - Cuenta qué variables se remodelan con más/menos frecuencia (overall y por barrio/percentil).
Genera CSVs en `optimization/sensibilidad_remodelacion/analisis/`.

Uso:
  python3 optimization/sensibilidad_remodelacion/analysis_summary.py \
      --detalles optimization/sensibilidad_remodelacion/detalles.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import pandas as pd

# asegurar repo en path (por si se importa algo más)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detalles", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("optimization/sensibilidad_remodelacion/analisis"))
    return ap.parse_args()


def load_details(det_path: Path) -> List[dict]:
    rows = []
    changes_rows = []
    with det_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            if not meta:
                continue
            meta["opt_row"] = rec.get("extra", {}).get("opt_row")
            rows.append(meta)
            for ch in meta.get("changes", []):
                changes_rows.append({
                    "pid": meta.get("pid"),
                    "neighborhood": meta.get("neighborhood"),
                    "percentile_label": meta.get("percentile_label"),
                    "budget": meta.get("budget"),
                    "status": meta.get("status"),
                    "col": ch.get("col"),
                    "base": ch.get("base"),
                    "new": ch.get("new"),
                })
    df_meta = pd.DataFrame(rows)
    df_ch = pd.DataFrame(changes_rows)
    return df_meta, df_ch


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_meta, df_ch = load_details(args.detalles)
    # Solo OPTIMAL
    df_opt = df_meta[df_meta["status"] == "OPTIMAL"].copy()
    if df_opt.empty:
        print("No hay registros OPTIMAL en detalles.jsonl")
        return

    # ---- Métricas agregadas ----
    def _agg(df):
        return df.agg({
            "base_price": "mean",
            "opt_price": "mean",
            "cost": "mean",
            "net_gain": "mean",
            "pid": "count",
        }).rename({"pid": "n"})
    grp_nb_budget = df_opt.groupby(["neighborhood", "budget"]).apply(_agg).reset_index()
    grp_pct_budget = df_opt.groupby(["percentile_label", "budget"]).apply(_agg).reset_index()
    grp_budget = df_opt.groupby(["budget"]).apply(_agg).reset_index()

    grp_nb_budget.to_csv(args.outdir / "summary_by_neighborhood_budget.csv", index=False)
    grp_pct_budget.to_csv(args.outdir / "summary_by_percentile_budget.csv", index=False)
    grp_budget.to_csv(args.outdir / "summary_by_budget.csv", index=False)

    # ---- Frecuencia de cambios ----
    df_ch_opt = df_ch[df_ch["status"] == "OPTIMAL"].copy()
    freq_overall = df_ch_opt.groupby("col").size().reset_index(name="count").sort_values("count", ascending=False)
    freq_by_nb = df_ch_opt.groupby(["neighborhood", "col"]).size().reset_index(name="count")
    freq_by_pct = df_ch_opt.groupby(["percentile_label", "col"]).size().reset_index(name="count")

    freq_overall.to_csv(args.outdir / "changes_freq_overall.csv", index=False)
    freq_by_nb.to_csv(args.outdir / "changes_freq_by_neighborhood.csv", index=False)
    freq_by_pct.to_csv(args.outdir / "changes_freq_by_percentile.csv", index=False)

    # ---- Reporte rápido ----
    top5 = freq_overall.head(5)
    rep = [
        f"Total casos OPTIMAL: {len(df_opt)}",
        "",
        "Top 5 cambios más frecuentes (overall):",
    ]
    for _, row in top5.iterrows():
        rep.append(f" - {row['col']}: {int(row['count'])} veces")
    rep_path = args.outdir / "resumen.txt"
    rep_path.write_text("\n".join(rep), encoding="utf-8")
    print(f"Listo. Escritos resúmenes en {args.outdir}")


if __name__ == "__main__":
    main()
