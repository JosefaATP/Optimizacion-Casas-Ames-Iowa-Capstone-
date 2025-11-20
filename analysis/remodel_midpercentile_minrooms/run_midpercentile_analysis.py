#!/usr/bin/env python3
"""
Batch para ejecutar sensibilidad de remodelación solo en el percentil 50 (precio medio)
y generar un reporte que destaque qué se construye y qué se sacrifica cuando se
exige un mínimo de ambientes (p. ej. 2 dormitorios).

Este script crea su propia carpeta de resultados:
    analysis/remodel_midpercentile_minrooms/<escenario>/
con los archivos producidos por sensitivity.py + un reporte adicional.

Uso:
  python3 analysis/remodel_midpercentile_minrooms/run_midpercentile_analysis.py \
      --scenario min2 \
      --budgets 20000 50000 100000 \
      --outdir analysis/remodel_midpercentile_minrooms/results_minrooms
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

from optimization.sensibilidad_remodelacion import sensitivity as sens_mod
from optimization.sensibilidad_remodelacion.sensitivity import CaseResult, run_case, _json_default


def _sanitize_base_score(bundle: sens_mod.XGBBundle) -> None:
    try:
        booster = bundle.reg.get_booster()
        base_score = booster.attr("base_score")
        if base_score and isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
            booster.set_attr(base_score=base_score.strip("[]"))
    except Exception:
        pass


def run_sensitivity(args: argparse.Namespace, scenario_dir: Path) -> Path:
    """Ejecuta sensitivity.py solo para percentil 50 por barrio."""
    detalles_path = scenario_dir / "detalles.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "optimization.sensibilidad_remodelacion.sensitivity",
        "--neighborhood",
        args.neighborhood,
        "--budgets",
        str(args.budget),
        "--percentiles",
        "0.5",
        "--outdir",
        str(scenario_dir),
        "--basecsv",
        str(args.basecsv),
    ]
    if args.model_path:
        cmd.extend(["--model-path", str(args.model_path)])
    if args.max_base_bedrooms is not None:
        cmd.extend(["--max-base-bedrooms", str(args.max_base_bedrooms)])
    if args.neighborhood.lower() == "all":
        cmd.extend(["--min-neighborhood-count", str(args.min_neighborhood_count)])
    print(f"[INFO] Ejecutando sensitivity: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return detalles_path


def run_analysis_summary(detalles_path: Path, scenario_dir: Path) -> None:
    """Reutiliza analysis_summary.py para dejar los CSV clásicos."""
    anal_dir = scenario_dir / "analisis"
    anal_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        "optimization/sensibilidad_remodelacion/analysis_summary.py",
        "--detalles",
        str(detalles_path),
        "--outdir",
        str(anal_dir),
    ]
    print(f"[INFO] Generando resúmenes estándar: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _to_float(val):
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except Exception:
        return np.nan


def summarize_changes(detalles_path: Path) -> Dict[str, pd.DataFrame]:
    rows: List[dict] = []
    changes: List[dict] = []
    with detalles_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            if not meta:
                continue
            rows.append(meta)
            for ch in meta.get("changes", []):
                base = ch.get("base")
                new = ch.get("new")
                base_num = _to_float(base)
                new_num = _to_float(new)
                diff = np.nan
                if not np.isnan(base_num) or not np.isnan(new_num):
                    diff = (new_num or 0.0) - (base_num or 0.0)
                changes.append(
                    {
                        "pid": meta.get("pid"),
                        "neighborhood": meta.get("neighborhood"),
                        "col": ch.get("col"),
                        "base": base,
                        "new": new,
                        "diff": diff,
                    }
                )
    df_meta = pd.DataFrame(rows)
    df_changes = pd.DataFrame(changes)
    return {"meta": df_meta, "changes": df_changes}


def build_report(df_meta: pd.DataFrame, df_changes: pd.DataFrame, scenario_dir: Path) -> None:
    report_lines: List[str] = []
    df_meta_opt = df_meta[df_meta["status"] == "OPTIMAL"].copy()
    report_lines.append("Análisis percentil 50 - mínimo de ambientes\n")
    report_lines.append(f"Total casos OPTIMAL: {len(df_meta_opt)}")
    budgets = sorted(df_meta_opt["budget"].unique())
    report_lines.append(f"Presupuesto evaluado: {', '.join(str(int(b)) for b in budgets)}\n")

    # ROI promedio por barrio
    if {"neighborhood", "net_gain", "cost"}.issubset(df_meta_opt.columns):
        df_meta_opt["roi"] = df_meta_opt["net_gain"] / df_meta_opt["cost"].replace(0, np.nan)
        roi_nb = (
            df_meta_opt.groupby("neighborhood")["roi"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        report_lines.append("Top 5 ROI por barrio (solo percentil medio):")
        for nb, roi in roi_nb.items():
            report_lines.append(f" - {nb}: ROI medio {roi:.2f}")
        report_lines.append("")

    # Cambios construidos vs sacrificados
    if not df_changes.empty:
        inc = (
            df_changes[df_changes["diff"] > 0]
            .groupby("col")["diff"]
            .agg(["count", "mean", "sum"])
            .sort_values("sum", ascending=False)
            .head(8)
        )
        dec = (
            df_changes[df_changes["diff"] < 0]
            .groupby("col")["diff"]
            .agg(["count", "mean", "sum"])
            .sort_values("sum")
            .head(8)
        )
        report_lines.append("Principales elementos construidos/ampliados (Δ positivo):")
        for col, row in inc.iterrows():
            report_lines.append(
                f" - {col}: +{row['sum']:.1f} ft² totales (en {int(row['count'])} casas)"
            )
        report_lines.append("")
        report_lines.append("Principales elementos sacrificados/reducidos (Δ negativo):")
        for col, row in dec.iterrows():
            report_lines.append(
                f" - {col}: {row['sum']:.1f} ft² menos (en {int(row['count'])} casas)"
            )
        report_lines.append("")

    # Resumen por presupuesto
    if "budget" in df_meta_opt.columns and "net_gain" in df_meta_opt.columns:
        budget_summary = (
            df_meta_opt.groupby("budget")
            .agg(
                net_gain_mean=("net_gain", "mean"),
                cost_mean=("cost", "mean"),
                roi_mean=("roi", "mean"),
            )
            .reset_index()
        )
        report_lines.append("Resumen por presupuesto:")
        for _, row in budget_summary.iterrows():
            report_lines.append(
                f" - Budget {int(row['budget'])}: net_gain={row['net_gain_mean']:,.0f}, "
                f"costo={row['cost_mean']:,.0f}, ROI medio={row['roi_mean']:.2f}"
            )
        report_lines.append("")

    report_path = scenario_dir / "midpercentile_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[INFO] Reporte listo en {report_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch percentil 50 remodel")
    ap.add_argument("--scenario", type=str, required=True, help="Nombre para la carpeta de resultados.")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/remodel_midpercentile_minrooms"),
        help="Carpeta base donde se crearán los escenarios.",
    )
    ap.add_argument("--budget", type=float, default=50000.0, help="Presupuesto único para todas las corridas.")
    ap.add_argument("--model-path", type=Path, default=None, help="Ruta a model_xgb.joblib para sensitivity.")
    ap.add_argument("--max-base-bedrooms", type=float, default=None,
                    help="Filtro opcional: sólo pids con Bedroom AbvGr base <= valor dado.")
    ap.add_argument("--pids-csv", type=Path, default=None,
                    help="CSV manual con columnas neighborhood y pid para usar esas casas.")
    ap.add_argument("--min-neighborhood-count", type=int, default=10,
                    help="Para --neighborhood all, mínimo de casas requeridas (sensibilidad).")
    ap.add_argument("--basecsv", type=Path, default=Path("data/processed/base_completa_sin_nulos.csv"))
    ap.add_argument("--neighborhood", type=str, default="all")
    return ap.parse_args()


def run_manual_pids(args: argparse.Namespace, scenario_dir: Path) -> Path:
    df_pids = pd.read_csv(args.pids_csv)
    if not {"pid", "neighborhood"}.issubset(df_pids.columns):
        raise ValueError("El CSV debe contener columnas 'pid' y 'neighborhood'.")

    bundle = sens_mod.XGBBundle(Path(args.model_path)) if args.model_path else sens_mod.XGBBundle()
    _sanitize_base_score(bundle)
    detalles_path = scenario_dir / "detalles.jsonl"
    if detalles_path.exists():
        detalles_path.unlink()

    summary_rows: List[CaseResult] = []

    for _, row in df_pids.iterrows():
        pid = int(row["pid"])
        nb = row["neighborhood"]
        print(f"== Barrio {nb} PID {pid} ==")
        res, extra = run_case(pid, args.budget, bundle, None, args.basecsv, scenario_dir)
        res.neighborhood = nb
        label = row.get("percentile_label") or f"manual_{int(row.get('percentile_target', 0)*100)}"
        res.percentile_label = label
        res.percentile_value = float(row.get("percentile_actual", row.get("base_price", float("nan"))))
        summary_rows.append(res)
        with detalles_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": asdict(res), "extra": extra}, default=_json_default) + "\n")

    summary_df = pd.DataFrame([asdict(r) for r in summary_rows])
    summary_df.to_csv(scenario_dir / "resumen.csv", index=False)
    return detalles_path


def main() -> None:
    args = parse_args()
    scenario_dir = args.outdir / args.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    detalles_path = scenario_dir / "detalles.jsonl"
    if args.pids_csv:
        detalles_path = run_manual_pids(args, scenario_dir)
    else:
        if not detalles_path.exists():
            detalles_path = run_sensitivity(args, scenario_dir)
        else:
            print(f"[INFO] Reutilizando {detalles_path}")

    run_analysis_summary(detalles_path, scenario_dir)
    data = summarize_changes(detalles_path)
    build_report(data["meta"], data["changes"], scenario_dir)


if __name__ == "__main__":
    main()
def _sanitize_base_score(bundle: sens_mod.XGBBundle) -> None:
    try:
        booster = bundle.reg.get_booster()
        base_score = booster.attr("base_score")
        if base_score and isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
            booster.set_attr(base_score=base_score.strip("[]"))
    except Exception:
        pass
