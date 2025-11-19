#!/usr/bin/env python3
"""
Ejecuta barridos de construcción (neighborhood × lot × budget) para construir
un CSV con métricas comparables (XGB vs regresión, costos, tiempos, etc.).

Uso:
    PYTHONPATH=. python scripts/benchmark_construction.py \
        --out bench_out/benchmark_runs.csv \
        --neigh-all \
        --lots 1700 3200 5000 7500 11500 \
        --budgets 200000 400000 600000 800000 1000000 \
        --profile balanced \
        --quiet
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from datetime import datetime
from pathlib import Path

from optimization.construction.config import PATHS
from optimization.construction.io import load_base_df


def list_neighborhoods(base_csv: str | None) -> list[str]:
    df = load_base_df(base_csv)
    if "Neighborhood" not in df.columns:
        raise RuntimeError("CSV base no tiene columna Neighborhood")
    vals = sorted(df["Neighborhood"].dropna().unique())
    return [str(v) for v in vals]


def run_single(neigh: str, lot: float, budget: float, *,
               profile: str, quiet: bool, reg_model: str | None,
               reg_basecsv: str | None, xgbdir: str | None,
               out_csv: str,
               fast: bool = False, deep: bool = False) -> dict | None:
    cmd = [
        sys.executable, "-m", "optimization.construction.run_opt",
        "--neigh", str(neigh),
        "--lot", str(lot),
        "--budget", str(budget),
        "--profile", profile,
    ]
    if quiet:
        cmd.append("--quiet")
    if fast:
        cmd.append("--fast")
    if deep:
        cmd.append("--deep")
    if reg_model:
        cmd += ["--reg-model", reg_model]
    if reg_basecsv:
        cmd += ["--reg-basecsv", reg_basecsv]
    if xgbdir:
        cmd += ["--xgbdir", xgbdir]
    cmd += ["--outcsv", out_csv]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PATHS.repo_root))

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = proc.stdout
    stderr = proc.stderr

    result = {
        "neighborhood": neigh,
        "lot_area": lot,
        "budget": budget,
        "profile": profile,
        "quiet": bool(quiet),
        "status": proc.returncode,
        "timestamp": datetime.utcnow().isoformat(),
    }
    result["stdout"] = stdout
    result["stderr"] = stderr
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV de salida para los resultados (se hace append si existe)")
    ap.add_argument("--neigh", nargs="*", default=None, help="Lista de neighborhoods a evaluar")
    ap.add_argument("--neigh-all", action="store_true", help="Si se especifica, recorre todos los neighborhoods del CSV base")
    ap.add_argument("--basecsv", type=str, default=None, help="CSV base para obtener neighborhoods")
    ap.add_argument("--lots", nargs="*", type=float, required=True, help="Lista de lot areas a probar")
    ap.add_argument("--budgets", nargs="*", type=float, required=True, help="Lista de presupuestos a probar")
    ap.add_argument("--profile", type=str, default="balanced", choices=["balanced", "feasible", "bound"])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--reg-model", type=str, default=None, help="Modelo de regresión alternativo")
    ap.add_argument("--reg-basecsv", type=str, default=None, help="CSV limpio de la regresión")
    ap.add_argument("--xgbdir", type=str, default=None, help="Carpeta alternativa con model_xgb.joblib")
    args = ap.parse_args()

    if args.neigh_all:
        neighborhoods = list_neighborhoods(args.basecsv)
    else:
        neighborhoods = args.neigh or []
        if not neighborhoods:
            raise ValueError("Debes indicar --neigh ... o --neigh-all")

    lots = sorted(set(float(x) for x in args.lots))
    budgets = sorted(set(float(x) for x in args.budgets))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    try:
        total_runs = len(neighborhoods) * len(lots) * len(budgets)
        idx = 0
        for neigh in neighborhoods:
            for lot in lots:
                for budget in budgets:
                    idx += 1
                    print(f"[{idx}/{total_runs}] running neigh={neigh} lot={lot} budget={budget}")
                    res = run_single(
                        neigh, lot, budget,
                        profile=args.profile,
                        quiet=args.quiet,
                        reg_model=args.reg_model,
                        reg_basecsv=args.reg_basecsv,
                        xgbdir=args.xgbdir,
                        out_csv=args.out,
                        fast=args.fast,
                        deep=args.deep,
                    )
                    if res is None:
                        continue
                    if res["status"] != 0:
                        print(f"[WARN] run failed (status={res['status']}) neigh={neigh} lot={lot} budget={budget}")
    except KeyboardInterrupt:
        print("[INFO] Benchmark interrumpido por el usuario")


if __name__ == "__main__":
    main()
