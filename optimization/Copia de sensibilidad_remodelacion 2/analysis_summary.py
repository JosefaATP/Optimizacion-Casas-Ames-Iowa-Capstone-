"""
Resumen de sensibilidad:
 - Métricas agregadas por presupuesto/barrio/percentil (medias, mediana, p10/p90, std, ratios).
 - Tasas de éxito (OPTIMAL) por grupo.
 - Frecuencia relativa de cambios (overall y por barrio/percentil/presupuesto).
 - Impacto en net_gain por tipo de cambio y combinaciones más comunes.
Genera CSVs en `optimization/sensibilidad_remodelacion/analisis/`.

Uso:
  python3 optimization/sensibilidad_remodelacion/analysis_summary.py \
      --detalles optimization/sensibilidad_remodelacion/detalles.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
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
    # Métricas derivadas para ratios
    df_meta["net_gain_per_cost"] = df_meta["net_gain"] / df_meta["cost"].where(
        df_meta["cost"].notna() & (df_meta["cost"] != 0)
    )
    df_meta["price_uplift"] = df_meta["opt_price"] / df_meta["base_price"].where(
        df_meta["base_price"].notna() & (df_meta["base_price"] != 0)
    )
    # Solo OPTIMAL
    df_opt = df_meta[df_meta["status"] == "OPTIMAL"].copy()
    if df_opt.empty:
        print("No hay registros OPTIMAL en detalles.jsonl")
        return

    # ---- Métricas agregadas ----
    def _num_stats(df, col: str):
        s = df[col]
        return {
            f"{col}_mean": s.mean(),
            f"{col}_median": s.median(),
            f"{col}_p10": s.quantile(0.1),
            f"{col}_p90": s.quantile(0.9),
            f"{col}_std": s.std(),
        }

    def _agg(df):
        metrics = {"n": len(df)}
        for col in ["base_price", "opt_price", "cost", "net_gain", "net_gain_per_cost", "price_uplift"]:
            metrics.update(_num_stats(df, col))
        return pd.Series(metrics)

    # Tasas de éxito (usa todos los status)
    def _success_rate(df, group_cols):
        grp = df.groupby(group_cols).agg(
            total=("pid", "count"),
            optimal=("status", lambda s: (s == "OPTIMAL").sum()),
        ).reset_index()
        grp["optimal_rate"] = grp["optimal"] / grp["total"]
        return grp

    grp_nb_budget = df_opt.groupby(["neighborhood", "budget"]).apply(_agg).reset_index()
    grp_pct_budget = df_opt.groupby(["percentile_label", "budget"]).apply(_agg).reset_index()
    grp_budget = df_opt.groupby(["budget"]).apply(_agg).reset_index()

    grp_nb_budget.to_csv(args.outdir / "summary_by_neighborhood_budget.csv", index=False)
    grp_pct_budget.to_csv(args.outdir / "summary_by_percentile_budget.csv", index=False)
    grp_budget.to_csv(args.outdir / "summary_by_budget.csv", index=False)

    succ_nb_budget = _success_rate(df_meta, ["neighborhood", "budget"])
    succ_pct_budget = _success_rate(df_meta, ["percentile_label", "budget"])
    succ_budget = _success_rate(df_meta, ["budget"])

    succ_nb_budget.to_csv(args.outdir / "success_rate_by_neighborhood_budget.csv", index=False)
    succ_pct_budget.to_csv(args.outdir / "success_rate_by_percentile_budget.csv", index=False)
    succ_budget.to_csv(args.outdir / "success_rate_by_budget.csv", index=False)

    # ---- Frecuencia de cambios ----
    df_ch_opt = df_ch[df_ch["status"] == "OPTIMAL"].copy()
    total_opt = len(df_opt)
    freq_overall = (
        df_ch_opt.groupby("col")
        .size()
        .reset_index(name="count")
        .assign(rate_overall=lambda d: d["count"] / total_opt)
        .sort_values("count", ascending=False)
    )
    freq_by_nb = df_ch_opt.groupby(["neighborhood", "col"]).size().reset_index(name="count")
    freq_by_pct = df_ch_opt.groupby(["percentile_label", "col"]).size().reset_index(name="count")
    freq_by_budget = df_ch_opt.groupby(["budget", "col"]).size().reset_index(name="count")

    nb_sizes = df_opt.groupby("neighborhood").size().rename("n_opt").reset_index()
    pct_sizes = df_opt.groupby("percentile_label").size().rename("n_opt").reset_index()
    budget_sizes = df_opt.groupby("budget").size().rename("n_opt").reset_index()

    freq_by_nb = freq_by_nb.merge(nb_sizes, on="neighborhood", how="left")
    freq_by_nb["rate_in_group"] = freq_by_nb["count"] / freq_by_nb["n_opt"]

    freq_by_pct = freq_by_pct.merge(pct_sizes, on="percentile_label", how="left")
    freq_by_pct["rate_in_group"] = freq_by_pct["count"] / freq_by_pct["n_opt"]

    freq_by_budget = freq_by_budget.merge(budget_sizes, on="budget", how="left")
    freq_by_budget["rate_in_group"] = freq_by_budget["count"] / freq_by_budget["n_opt"]

    freq_overall.to_csv(args.outdir / "changes_freq_overall.csv", index=False)
    freq_by_nb.to_csv(args.outdir / "changes_freq_by_neighborhood.csv", index=False)
    freq_by_pct.to_csv(args.outdir / "changes_freq_by_percentile.csv", index=False)
    freq_by_budget.to_csv(args.outdir / "changes_freq_by_budget.csv", index=False)

    # Impacto por tipo de cambio (usa métricas de las casas OPTIMAL)
    impact_cols = ["pid", "net_gain", "net_gain_per_cost", "price_uplift", "cost", "base_price", "opt_price"]
    df_impact = df_ch_opt.merge(df_opt[impact_cols], on="pid", how="left")
    impact_overall = (
        df_impact.groupby("col")
        .agg(
            n_props=("pid", "nunique"),
            n_changes=("pid", "count"),
            net_gain_mean=("net_gain", "mean"),
            net_gain_median=("net_gain", "median"),
            net_gain_per_cost_mean=("net_gain_per_cost", "mean"),
            net_gain_per_cost_median=("net_gain_per_cost", "median"),
        )
        .reset_index()
        .sort_values("net_gain_mean", ascending=False)
    )
    impact_overall.to_csv(args.outdir / "changes_impact_overall.csv", index=False)

    # Combinaciones de cambios (por solución)
    changes_per_pid = df_ch_opt.groupby("pid")["col"].agg(lambda cols: sorted(set(cols))).reset_index()
    changes_per_pid["n_changes"] = changes_per_pid["col"].apply(len)
    changes_per_pid[["pid", "n_changes"]].to_csv(args.outdir / "changes_per_solution.csv", index=False)

    combo_counter = Counter()
    for cols in changes_per_pid["col"]:
        for r in (2, 3):
            for combo in combinations(cols, r):
                combo_counter[combo] += 1
    combos_rows = [
        {"combo": " + ".join(combo), "count": count} for combo, count in combo_counter.items()
    ]
    df_combos = pd.DataFrame(combos_rows).sort_values("count", ascending=False)
    df_combos.to_csv(args.outdir / "changes_combinations.csv", index=False)

    # ---- Reporte rápido ----
    top5 = freq_overall.head(5)
    rep = [
        f"Total casos: {len(df_meta)}",
        f"Total casos OPTIMAL: {len(df_opt)}",
        f"Tasa OPTIMAL global: {len(df_opt) / len(df_meta):.2%}" if len(df_meta) else "Tasa OPTIMAL global: -",
        "",
        "Top 5 cambios más frecuentes (overall):",
    ]
    for _, row in top5.iterrows():
        rep.append(f" - {row['col']}: {int(row['count'])} veces")
    best_gain = impact_overall.head(3)
    rep.append("")
    rep.append("Top 3 cambios con mayor net_gain medio:")
    for _, row in best_gain.iterrows():
        rep.append(f" - {row['col']}: {row['net_gain_mean']:.1f} net_gain medio")
    rep_path = args.outdir / "resumen.txt"
    rep_path.write_text("\n".join(rep), encoding="utf-8")
    print(f"Listo. Escritos resúmenes en {args.outdir}")


if __name__ == "__main__":
    main()
