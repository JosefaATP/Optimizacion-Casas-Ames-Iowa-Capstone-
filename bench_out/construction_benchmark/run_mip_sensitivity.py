#!/usr/bin/env python3
"""
Ejecuta barridos del optimizador de construcción (MIP) variando presupuesto,
perfil de solver y tipo de edificio para un caso base concreto.

Ejemplos (modo grid):
PYTHONPATH=. python bench_out/construction_benchmark/run_mip_sensitivity.py \
    --out bench_out/construction_benchmark/mip_sensitivity.csv \
    --neigh Veenker --lot 7000 \
    --budgets 400000 600000 800000 \
    --profiles balanced feasible \
    --bldgs 1Fam TwnhsE \
    --quiet

Modo escenarios (usa defaults):
PYTHONPATH=. python bench_out/construction_benchmark/run_mip_sensitivity.py \
    --out bench_out/construction_benchmark/mip_scenarios.csv \
    --neigh Veenker --lot 7000 \
    --use-default-scenarios

o bien con archivo JSON personalizado:
PYTHONPATH=. python bench_out/construction_benchmark/run_mip_sensitivity.py \
    --out bench_out/construction_benchmark/mip_scenarios.csv \
    --scenario-file bench_out/construction_benchmark/mip_scenarios.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from optimization.construction.config import PATHS

CONSTRAINT_FLAG_MAP = {
    "min_beds": "--min-beds",
    "max_beds": "--max-beds",
    "min_fullbath": "--min-fullbath",
    "max_fullbath": "--max-fullbath",
    "min_halfbath": "--min-halfbath",
    "max_halfbath": "--max-halfbath",
    "min_kitchen": "--min-kitchen",
    "max_kitchen": "--max-kitchen",
    "min_overallqual": "--min-overallqual",
    "max_overallqual": "--max-overallqual",
    "min_grliv": "--min-grliv",
    "max_grliv": "--max-grliv",
    "min_garage_area": "--min-garage-area",
    "max_garage_area": "--max-garage-area",
    "min_totalbsmt": "--min-totalbsmt",
    "max_totalbsmt": "--max-totalbsmt",
}

DEFAULT_SCENARIOS = [
    {
        "name": "starter",
        "budgets": [400000, 500000],
        "profile": "balanced",
        "constraints": {"min_beds": 2, "min_fullbath": 1, "min_overallqual": 6},
    },
    {
        "name": "family_moveup",
        "budgets": [600000, 700000],
        "profile": "balanced",
        "constraints": {"min_beds": 3, "min_fullbath": 2, "min_overallqual": 7},
    },
    {
        "name": "premium",
        "budgets": [800000, 900000, 1000000],
        "profile": "balanced",
        "constraints": {"min_beds": 4, "min_fullbath": 3, "min_kitchen": 2, "min_overallqual": 9, "min_grliv": 2000},
    },
    {
        "name": "townhouse_mid",
        "budgets": [500000, 600000],
        "profile": "balanced",
        "bldg": "TwnhsE",
        "constraints": {"min_beds": 3, "min_fullbath": 2, "min_overallqual": 7},
    },
    {
        "name": "compact_highqual",
        "budgets": [450000, 550000],
        "profile": "balanced",
        "constraints": {"min_beds": 2, "min_fullbath": 2, "min_overallqual": 9, "max_grliv": 1500},
    },
    {
        "name": "investor_low",
        "budgets": [350000, 400000],
        "profile": "balanced",
        "constraints": {"min_beds": 1, "min_fullbath": 1, "max_grliv": 1200, "min_overallqual": 6},
    },
]


def build_constraint_cli(constraints: dict[str, float | int | None]) -> list[str]:
    cli: list[str] = []
    for key, flag in CONSTRAINT_FLAG_MAP.items():
        if key in constraints and constraints[key] is not None:
            cli += [flag, str(constraints[key])]
    return cli


def collect_constraints_from_args(args) -> dict[str, float | int | None]:
    data = {}
    for key in CONSTRAINT_FLAG_MAP.keys():
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                data[key] = val
    return data


def load_scenario_list(path: str | None) -> list[dict]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("El archivo de escenarios debe contener una lista de escenarios.")
        return data
    return DEFAULT_SCENARIOS


def run_scenarios_mode(
    args,
    scenarios: list[dict],
    base_constraints: dict[str, float | int | None],
) -> None:
    if args.pid is None and args.neigh is None and args.lot is None:
        raise ValueError("Debes indicar --pid o bien --neigh y --lot (o definirlos en cada escenario).")

    total_runs = 0
    for scen in scenarios:
        budgets = scen.get("budgets")
        if budgets is None:
            if "budget" in scen:
                budgets = [scen["budget"]]
            else:
                budgets = []
        profiles = scen.get("profiles")
        if profiles is None:
            profiles = [scen.get("profile") or (args.profiles[0] if args.profiles else "balanced")]
        elif isinstance(profiles, str):
            profiles = [profiles]
        bldgs = scen.get("bldgs")
        if bldgs is None:
            bldgs = [scen.get("bldg", scen.get("building"))]
        elif isinstance(bldgs, str):
            bldgs = [bldgs]
        total_runs += len(budgets) * len(profiles) * max(len(bldgs), 1)
    if total_runs == 0:
        print("[WARN] No hay presupuestos definidos en los escenarios.")
        return
    idx = 0
    for scen in scenarios:
        name = scen.get("name", "scenario")
        budgets = scen.get("budgets")
        if budgets is None:
            if "budget" in scen:
                budgets = [scen["budget"]]
            else:
                raise ValueError(f"Escenario {name} no tiene 'budgets' ni 'budget'.")
        profiles = scen.get("profiles")
        if profiles is None:
            default_profile = scen.get("profile") or (args.profiles[0] if args.profiles else "balanced")
            profiles = [default_profile]
        elif isinstance(profiles, str):
            profiles = [profiles]
        bldgs = scen.get("bldgs")
        if bldgs is None:
            default_bldg = scen.get("bldg", scen.get("building"))
            bldgs = [default_bldg]
        elif isinstance(bldgs, str):
            bldgs = [bldgs]
        pid = scen.get("pid", args.pid)
        neigh = scen.get("neigh", args.neigh)
        lot = scen.get("lot", args.lot)
        if pid is None and (neigh is None or lot is None):
            raise ValueError(f"Escenario {name} no especifica PID ni (neigh+lot).")
        constraints = dict(base_constraints)
        constraints.update(scen.get("constraints", {}))
        constraint_cli = build_constraint_cli(constraints)
        base_tag = f"scenario={name}"
        for profile in profiles:
            for bldg in bldgs:
                for budget in budgets:
                    idx += 1
                    tag = f"{base_tag}|profile={profile}|bldg={bldg or 'default'}|budget={int(budget)}"
                    print(f"[{idx}/{total_runs}] {tag}")
                    status = run_single(
                        neigh,
                        lot,
                        pid,
                        float(budget),
                        profile,
                        bldg,
                        out_csv=args.out,
                        basecsv=args.basecsv,
                        reg_model=args.reg_model,
                        reg_basecsv=args.reg_basecsv,
                        xgbdir=args.xgbdir,
                        quiet=args.quiet,
                        fast=args.fast,
                        deep=args.deep,
                        extra_tag=tag,
                        constraint_cli=constraint_cli,
                    )
                    if status != 0:
                        print(f"[WARN] run_opt terminó con status={status} para {tag}")


def run_single(
    neigh: str | None,
    lot: float | None,
    pid: int | None,
    budget: float,
    profile: str,
    bldg: str | None,
    *,
    out_csv: str,
    basecsv: str | None,
    reg_model: str | None,
    reg_basecsv: str | None,
    xgbdir: str | None,
    quiet: bool,
    fast: bool,
    deep: bool,
    extra_tag: str,
    constraint_cli: list[str] | None = None,
) -> int:
    cmd = [sys.executable, "-m", "optimization.construction.run_opt", "--budget", str(budget), "--profile", profile]
    if pid is not None:
        cmd += ["--pid", str(pid)]
    else:
        if neigh is None or lot is None:
            raise ValueError("Debes especificar --pid o bien --neigh y --lot")
        cmd += ["--neigh", str(neigh), "--lot", str(lot)]
    if bldg:
        cmd += ["--bldg", bldg]
    if basecsv:
        cmd += ["--basecsv", basecsv]
    if reg_model:
        cmd += ["--reg-model", reg_model]
    if reg_basecsv:
        cmd += ["--reg-basecsv", reg_basecsv]
    if xgbdir:
        cmd += ["--xgbdir", xgbdir]
    if quiet:
        cmd.append("--quiet")
    if fast:
        cmd.append("--fast")
    if deep:
        cmd.append("--deep")
    if constraint_cli:
        cmd += constraint_cli
    cmd += ["--outcsv", out_csv, "--tag", extra_tag]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PATHS.repo_root))

    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Barrido de sensibilidad usando run_opt.")
    ap.add_argument("--out", required=True, help="Ruta al CSV donde se anexarán los resultados")
    ap.add_argument("--pid", type=int, default=None, help="PID base (opcional).")
    ap.add_argument("--neigh", type=str, default=None, help="Barrio base si no hay PID.")
    ap.add_argument("--lot", type=float, default=None, help="LotArea base si no hay PID.")
    ap.add_argument("--budgets", nargs="+", type=float, help="Lista de presupuestos a probar (modo grid).")
    ap.add_argument("--profiles", nargs="+", default=["balanced"], help="Perfiles de solver (balanced/feasible/bound).")
    ap.add_argument("--bldgs", nargs="*", default=[None], help="Tipos de edificio a evaluar (ej: 1Fam, TwnhsE).")
    ap.add_argument("--basecsv", type=str, default=None, help="CSV base alternativo.")
    ap.add_argument("--reg-model", type=str, default=None, help="Modelo de regresión alternativo.")
    ap.add_argument("--reg-basecsv", type=str, default=None, help="CSV limpio usado por la regresión.")
    ap.add_argument("--xgbdir", type=str, default=None, help="Carpeta con model_xgb.joblib alternativo.")
    ap.add_argument("--scenario-file", type=str, help="Archivo JSON con escenarios detallados.")
    ap.add_argument("--use-default-scenarios", action="store_true", help="Ejecuta el set de escenarios predefinido.")
    ap.add_argument("--scenario-names", nargs="*", help="Si se especifica, filtra los escenarios por nombre.")
    ap.add_argument("--min-beds", type=float, default=None, help="Cota inferior global para Bedrooms (modo grid).")
    ap.add_argument("--max-beds", type=float, default=None, help="Cota superior global para Bedrooms.")
    ap.add_argument("--min-fullbath", type=float, default=None, help="Mínimo de baños completos.")
    ap.add_argument("--max-fullbath", type=float, default=None, help="Máximo de baños completos.")
    ap.add_argument("--min-halfbath", type=float, default=None, help="Mínimo de medios baños.")
    ap.add_argument("--max-halfbath", type=float, default=None, help="Máximo de medios baños.")
    ap.add_argument("--min-kitchen", type=float, default=None, help="Mínimo de cocinas.")
    ap.add_argument("--max-kitchen", type=float, default=None, help="Máximo de cocinas.")
    ap.add_argument("--min-overallqual", type=float, default=None, help="Mínimo de Overall Qual.")
    ap.add_argument("--max-overallqual", type=float, default=None, help="Máximo de Overall Qual.")
    ap.add_argument("--min-grliv", type=float, default=None, help="Mínimo de Gr Liv Area (ft2).")
    ap.add_argument("--max-grliv", type=float, default=None, help="Máximo de Gr Liv Area (ft2).")
    ap.add_argument("--min-garage-area", type=float, default=None, help="Mínimo de Garage Area (ft2).")
    ap.add_argument("--max-garage-area", type=float, default=None, help="Máximo de Garage Area (ft2).")
    ap.add_argument("--min-totalbsmt", type=float, default=None, help="Mínimo de Total Bsmt SF.")
    ap.add_argument("--max-totalbsmt", type=float, default=None, help="Máximo de Total Bsmt SF.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()

    scenario_mode = bool(args.scenario_file or args.use_default_scenarios)
    if not scenario_mode and args.pid is None and (args.neigh is None or args.lot is None):
        ap.error("Debes indicar --pid o bien --neigh junto con --lot")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    base_constraints = collect_constraints_from_args(args)

    if scenario_mode:
        scenarios = load_scenario_list(args.scenario_file)
        if args.scenario_names:
            scenarios = [s for s in scenarios if s.get("name") in args.scenario_names]
        run_scenarios_mode(args, scenarios, base_constraints)
        return

    if not args.budgets:
        ap.error("Debes indicar --budgets ... (modo grid)")

    base_constraint_cli = build_constraint_cli(base_constraints)

    total = len(args.budgets) * len(args.profiles) * len(args.bldgs or [None])
    idx = 0
    for profile in args.profiles:
        for bldg in (args.bldgs or [None]):
            for budget in args.budgets:
                idx += 1
                tag = f"profile={profile}|bldg={bldg or 'default'}|budget={int(budget)}"
                print(f"[{idx}/{total}] {tag}")
                status = run_single(
                    args.neigh,
                    args.lot,
                    args.pid,
                    budget,
                    profile,
                    bldg,
                    out_csv=args.out,
                    basecsv=args.basecsv,
                    reg_model=args.reg_model,
                    reg_basecsv=args.reg_basecsv,
                    xgbdir=args.xgbdir,
                    quiet=args.quiet,
                    fast=args.fast,
                    deep=args.deep,
                    extra_tag=tag,
                    constraint_cli=base_constraint_cli,
                )
                if status != 0:
                    print(f"[WARN] run_opt terminó con status={status} para {tag}")


if __name__ == "__main__":
    main()
