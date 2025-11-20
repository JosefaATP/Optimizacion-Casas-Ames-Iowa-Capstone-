#!/usr/bin/env python3
"""
Ejecuta el optimizador de construcción (`optimization.construction.run_opt`)
para todos los barrios disponibles con los mismos parámetros (lote/budget) y
genera un análisis comparativo por barrio y por categoría socioeconómica.

Uso básico:
    python3 analysis/construction_batch/batch_run_by_neighborhood.py \
        --basecsv data/processed/base_completa_sin_nulos.csv \
        --lot 7000 \
        --budget 500000 \
        --outdir analysis/construction_batch/results

El script:
 1. Calcula las categorías (baja/media/alta) de cada barrio según el precio
    medio (`SalePrice_Present`) usando terciles.
 2. Ejecuta `run_opt` una vez por barrio con los parámetros deseados
    (opcionalmente se puede omitir la ejecución con `--skip-exec`).
 3. Consolida los resultados guardados vía `--outcsv` en:
        - `neighborhood_comparison.csv`
        - `category_summary.csv`
        - `analysis.txt` (hallazgos principales)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def compute_neighborhood_categories(base_csv: Path) -> Tuple[List[str], Dict[str, str], pd.Series]:
    df = pd.read_csv(base_csv)
    if "Neighborhood" not in df.columns or "SalePrice_Present" not in df.columns:
        raise ValueError("El CSV necesita columnas 'Neighborhood' y 'SalePrice_Present'.")

    series = (
        df.dropna(subset=["Neighborhood", "SalePrice_Present"])
        .groupby("Neighborhood")["SalePrice_Present"]
        .mean()
        .sort_values()
    )

    if series.empty:
        raise ValueError("No se encontraron barrios válidos en el CSV.")

    q1 = series.quantile(1.0 / 3.0)
    q2 = series.quantile(2.0 / 3.0)

    def categorize(price: float) -> str:
        if price <= q1:
            return "baja"
        if price <= q2:
            return "media"
        return "alta"

    categories = {neigh: categorize(price) for neigh, price in series.items()}
    neighborhoods = list(series.index)
    return neighborhoods, categories, series


def run_for_neighborhoods(
    neighborhoods: List[str],
    args: argparse.Namespace,
    out_csv: Path,
) -> None:
    if args.skip_exec:
        print("[INFO] --skip-exec activo: se omite la ejecución de run_opt.")
        return

    for idx, neigh in enumerate(neighborhoods, start=1):
        cmd = [
            sys.executable,
            "-m",
            "optimization.construction.run_opt",
            "--neigh",
            neigh,
            "--lot",
            str(args.lot),
            "--budget",
            str(args.budget),
            "--quiet",
            "--outcsv",
            str(out_csv),
            "--tag",
            f"batch_{args.budget:.0f}",
        ]
        if args.basecsv:
            cmd.extend(["--basecsv", str(args.basecsv)])
        if args.bldg:
            cmd.extend(["--bldg", args.bldg])
        if args.profile:
            cmd.extend(["--profile", args.profile])

        print(f"[RUN {idx:02d}/{len(neighborhoods)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def _available_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def analyze_results(
    out_csv: Path,
    categories: Dict[str, str],
    outdir: Path,
) -> None:
    if not out_csv.exists():
        raise FileNotFoundError(f"No existe {out_csv}. Ejecuta la fase de corridas primero.")

    df = pd.read_csv(out_csv)
    if df.empty:
        raise ValueError(f"{out_csv} está vacío, no hay corridas para analizar.")

    df["category"] = df["neigh"].map(categories).fillna("desconocido")
    df["profit"] = df["y_price"] - df["cost"]
    df["roi"] = df["profit"] / df["cost"]
    df["budget_utilization"] = df["cost"] / df["budget"]
    if "slack" in df.columns:
        df["slack_pct"] = df["slack"] / df["budget"]

    cols = _available_columns(
        df,
        [
            "timestamp",
            "neigh",
            "category",
            "budget",
            "cost",
            "slack",
            "slack_pct",
            "profit",
            "roi",
            "budget_utilization",
            "y_price",
            "delta_reg_abs",
            "delta_reg_pct",
            "gr_liv_area",
            "garage_area",
            "screen_porch",
            "beds",
            "fullbath",
            "halfbath",
            "kitchen",
            "overall_qual",
            "overall_cond",
            "kitchen_qual",
            "heating_qc",
        ],
    )
    df_out = df[cols].sort_values(["category", "neigh"])
    neighborhood_csv = outdir / "neighborhood_comparison.csv"
    df_out.to_csv(neighborhood_csv, index=False)

    agg_cols = {
        "y_price": "mean",
        "cost": "mean",
        "profit": "mean",
        "roi": "mean",
        "slack": "mean",
        "budget_utilization": "mean",
        "gr_liv_area": "mean",
        "garage_area": "mean",
        "beds": "mean",
        "fullbath": "mean",
        "halfbath": "mean",
        "kitchen": "mean",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
    category_summary = (
        df.groupby("category")
        .agg(agg_cols)
        .rename(
            columns={
                "y_price": "avg_price",
                "cost": "avg_cost",
                "profit": "avg_profit",
                "roi": "avg_roi",
                "slack": "avg_slack",
                "budget_utilization": "avg_budget_util",
                "gr_liv_area": "avg_gr_liv_area",
            }
        )
    )
    category_summary.insert(0, "n_barrios", df.groupby("category").size())
    category_csv = outdir / "category_summary.csv"
    category_summary.to_csv(category_csv)

    # Hallazgos clave
    lines: List[str] = []
    lines.append("Resumen batch run_opt por barrio\n")
    lines.append(f"Total de corridas: {len(df)}")
    lines.append(f"Barrios analizados: {', '.join(sorted(df['neigh'].unique()))}\n")

    lines.append("Promedios por categoría (ver category_summary.csv):")
    for cat, row in category_summary.iterrows():
        lines.append(
            f" - {cat.title()}: n={int(row['n_barrios'])}, "
            f"precio promedio={row.get('avg_price', float('nan')):,.0f}, "
            f"costo={row.get('avg_cost', float('nan')):,.0f}, "
            f"ROI medio={row.get('avg_roi', float('nan')):.2f}, "
            f"slack medio={row.get('avg_slack', float('nan')):,.0f}"
        )
    lines.append("")

    top_roi = df.sort_values("roi", ascending=False).head(3)
    lines.append("Top 3 barrios por ROI individual:")
    for _, row in top_roi.iterrows():
        lines.append(
            f" - {row['neigh']} ({row['category']}): ROI={row['roi']:.2f}, "
            f"profit={row['profit']:,.0f}, costo={row['cost']:,.0f}"
        )
    lines.append("")

    bottom_roi = df.sort_values("roi", ascending=True).head(3)
    lines.append("Bottom 3 barrios por ROI individual:")
    for _, row in bottom_roi.iterrows():
        lines.append(
            f" - {row['neigh']} ({row['category']}): ROI={row['roi']:.2f}, "
            f"profit={row['profit']:,.0f}, costo={row['cost']:,.0f}"
        )
    lines.append("")

    if "gr_liv_area" in df.columns:
        top_area = df.sort_values("gr_liv_area", ascending=False).head(3)
        lines.append("Barrios con mayor área habitable diseñada:")
        for _, row in top_area.iterrows():
            lines.append(
                f" - {row['neigh']}: GrLivArea={row['gr_liv_area']:,.0f} ft², "
                f"beds={row.get('beds', 'na')}, baths={row.get('fullbath', 'na')}"
            )
        lines.append("")

    if "slack" in df.columns:
        top_slack = df.sort_values("slack", ascending=False).head(3)
        lines.append("Barrios con mayor presupuesto sin usar (slack):")
        for _, row in top_slack.iterrows():
            lines.append(
                f" - {row['neigh']}: slack={row['slack']:,.0f} ({row.get('slack_pct', float('nan')):.2%})"
            )
        lines.append("")

    analysis_txt = outdir / "analysis.txt"
    analysis_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[INFO] Comparativo por barrio -> {neighborhood_csv}")
    print(f"[INFO] Resumen por categoría -> {category_csv}")
    print(f"[INFO] Reporte narrativo -> {analysis_txt}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch de run_opt por barrio.")
    ap.add_argument("--basecsv", type=Path, default=Path("data/processed/base_completa_sin_nulos.csv"))
    ap.add_argument("--lot", type=float, default=7000.0)
    ap.add_argument("--budget", type=float, default=500000.0)
    ap.add_argument("--outdir", type=Path, default=Path("analysis/construction_batch/results"))
    ap.add_argument("--profile", type=str, default="balanced", help="Perfil de solver para run_opt.")
    ap.add_argument("--bldg", type=str, default=None, help="Valor de --bldg (opcional).")
    ap.add_argument("--skip-exec", action="store_true", help="Solo analiza resultados existentes (no ejecuta run_opt).")
    ap.add_argument("--only", nargs="*", default=None, help="Lista opcional de barrios específicos a procesar.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "construction_runs.csv"

    neighborhoods, categories, _ = compute_neighborhood_categories(args.basecsv)
    if args.only:
        neighborhoods = [nb for nb in neighborhoods if nb in set(args.only)]
        if not neighborhoods:
            raise ValueError("Ninguno de los barrios de --only coincide con la base.")

    run_for_neighborhoods(neighborhoods, args, out_csv)
    analyze_results(out_csv, categories, args.outdir)


if __name__ == "__main__":
    main()
