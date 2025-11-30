#!/usr/bin/env python3
"""
Ejecuta run_opt para uno o varios PID y resume la brecha entre XGBoost y
Regresión (base y remodelada). Guarda un CSV con cada corrida y muestra
promedios al final.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

MONEY_RE = re.compile(r"\$?\s*([+-]?[0-9][0-9,\.]*)")


def _to_float(txt: str) -> Optional[float]:
    m = MONEY_RE.search(txt)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


@dataclass
class RunResult:
    pid: int
    budget: float
    price_base_reported: Optional[float]
    base_real: Optional[float]
    base_xgb: Optional[float]
    base_reg: Optional[float]
    diff_base_abs: Optional[float]
    diff_base_pct: Optional[float]
    opt_xgb: Optional[float]
    opt_reg: Optional[float]
    diff_opt_abs: Optional[float]
    diff_opt_pct: Optional[float]
    total_cost_model: Optional[float]
    roi_xgb_pct: Optional[float]
    roi_reg_pct: Optional[float]


def parse_run_opt_output(text: str, pid: int, budget: float) -> RunResult:
    lines = text.splitlines()

    price_base_reported = None
    base_real = None
    base_xgb = None
    base_reg = None
    opt_xgb = None
    opt_reg = None
    total_cost_model = None

    for ln in lines:
        if "Precio real (dataset)" in ln:
            base_real = _to_float(ln)
        elif "Precio casa base:" in ln:
            price_base_reported = _to_float(ln)
        elif "Costos totales (modelo):" in ln:
            total_cost_model = _to_float(ln)
        elif "Casa base" in ln and "$" in ln and base_xgb is None and base_reg is None:
            # solo ancla de bloque; los valores vienen debajo
            continue
        elif "XGBoost:" in ln and base_xgb is None:
            base_xgb = _to_float(ln)
        elif "Regresión:" in ln and base_reg is None:
            base_reg = _to_float(ln)
        elif "Casa remodelada" in ln:
            # reseteo ancla (los próximos XGBoost/Regresión son remodelados)
            pass
        elif "Casa remodelada" in ln:
            pass

    # Segunda pasada para capturar remodelado (después de la línea "Casa remodelada")
    seen_remodel = False
    for ln in lines:
        if "Casa remodelada" in ln:
            seen_remodel = True
            continue
        if not seen_remodel:
            continue
        if "XGBoost:" in ln and opt_xgb is None:
            opt_xgb = _to_float(ln)
        elif "Regresión:" in ln and opt_reg is None:
            opt_reg = _to_float(ln)

    diff_base_abs = None
    diff_base_pct = None
    if base_xgb is not None and base_reg is not None and base_xgb != 0:
        diff_base_abs = base_reg - base_xgb
        diff_base_pct = diff_base_abs / base_xgb * 100

    diff_opt_abs = None
    diff_opt_pct = None
    if opt_xgb is not None and opt_reg is not None and opt_xgb != 0:
        diff_opt_abs = opt_reg - opt_xgb
        diff_opt_pct = diff_opt_abs / opt_xgb * 100

    def _calc_roi_pct(opt_price: Optional[float]) -> Optional[float]:
        if (
            opt_price is None
            or total_cost_model is None
            or total_cost_model == 0
            or price_base_reported is None
        ):
            return None
        utilidad = (opt_price - total_cost_model) - price_base_reported
        return utilidad / total_cost_model * 100.0

    roi_xgb_pct = _calc_roi_pct(opt_xgb)
    roi_reg_pct = _calc_roi_pct(opt_reg)

    return RunResult(
        pid=pid,
        budget=budget,
        price_base_reported=price_base_reported,
        base_real=base_real,
        base_xgb=base_xgb,
        base_reg=base_reg,
        diff_base_abs=diff_base_abs,
        diff_base_pct=diff_base_pct,
        opt_xgb=opt_xgb,
        opt_reg=opt_reg,
        diff_opt_abs=diff_opt_abs,
        diff_opt_pct=diff_opt_pct,
        total_cost_model=total_cost_model,
        roi_xgb_pct=roi_xgb_pct,
        roi_reg_pct=roi_reg_pct,
    )


def run_for_pid(pid: int, budget: float) -> RunResult:
    cmd = ["python3", "-m", "optimization.remodel.run_opt", "--pid", str(pid), "--budget", str(budget)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run_opt fallo para PID {pid}: {proc.stderr}")
    out_lower = proc.stdout.lower()
    if "infeas" in out_lower or "no hay solucion" in out_lower or "modelo no optimo" in out_lower:
        raise RuntimeError(f"run_opt infeasible/unbounded para PID {pid}")
    return parse_run_opt_output(proc.stdout, pid, budget)


def main():
    ap = argparse.ArgumentParser(description="Batch de comparación XGBoost vs regresión")
    ap.add_argument("--pids", type=str, default=None,
                    help="Lista de PID separados por coma, ej: 527145080,527127150. Si se omite, se muestrean al azar del CSV.")
    ap.add_argument("--sample-size", type=int, default=10, help="Cantidad de casas a muestrear del CSV cuando no se pasa --pids")
    ap.add_argument("--csv", type=str, default="data/raw/df_final_regresion.csv", help="CSV de donde sacar los PID si no se pasan explícitos")
    ap.add_argument("--budget", type=float, default=80000.0, help="Presupuesto a usar en run_opt")
    ap.add_argument("--seed", type=int, default=42, help="Seed para el muestreo aleatorio (None = diferente cada vez)")
    ap.add_argument("--out", type=str, default=None, help="Ruta de salida CSV (si no se entrega, se usa analysis/model_gap_runs_budget_<budget>.csv)")
    args = ap.parse_args()

    if args.pids:
        pids = [int(p.strip()) for p in args.pids.split(",") if p.strip()]
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        shuffled = df["PID"].sample(frac=1.0, random_state=args.seed).tolist()
        pids = []
        for pid in shuffled:
            if len(pids) >= args.sample_size:
                break
            pids.append(int(pid))
        print(f"Usando muestra aleatoria inicial de {len(pids)} PID desde {args.csv}: {pids}")

    # Ruta de salida: si no se entrega, usa budget en el nombre
    if args.out:
        out_path = Path(args.out)
    else:
        budget_tag = (
            str(int(args.budget))
            if float(args.budget).is_integer()
            else str(args.budget).replace(".", "p")
        )
        out_path = Path(f"analysis/model_gap_runs_budget_{budget_tag}.csv")

    print(f"ℹ️  Corriendo {len(pids)} casas con presupuesto ${args.budget:,.0f}")
    results: List[RunResult] = []
    tried = set()
    import pandas as pd
    all_pool = [int(x) for x in pd.read_csv(args.csv)["PID"].tolist()]

    idx = 0
    max_tries = max(len(pids), len(all_pool)) * 3
    attempts = 0
    while len(results) < args.sample_size and attempts < max_tries:
        pid = pids[idx] if idx < len(pids) else None
        if pid is None:
            # buscar reemplazo
            import random
            random.seed(args.seed)
            random.shuffle(all_pool)
            pid = next((p for p in all_pool if p not in tried), None)
            if pid is None:
                print("⚠️  No quedan PID disponibles para reemplazo.")
                break
            pids.append(pid)
        idx += 1
        tried.add(pid)
        print(f"➡️  Corriendo PID {pid} (budget {args.budget:,.0f}) ...", flush=True)
        try:
            res = run_for_pid(pid, args.budget)
            results.append(res)
        except Exception as exc:
            print(f"  ⚠️  PID {pid} falló ({exc}); reemplazando...")
            continue
        attempts += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"✓ Guardado en {out_path}")

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    base_diffs = [r.diff_base_abs for r in results if r.diff_base_abs is not None]
    base_pcts = [r.diff_base_pct for r in results if r.diff_base_pct is not None]
    opt_diffs = [r.diff_opt_abs for r in results if r.diff_opt_abs is not None]
    opt_pcts = [r.diff_opt_pct for r in results if r.diff_opt_pct is not None]

    avg_base_diff = _avg(base_diffs)
    avg_base_pct = _avg(base_pcts)
    avg_opt_diff = _avg(opt_diffs)
    avg_opt_pct = _avg(opt_pcts)
    avg_xgb_base = _avg([r.base_xgb for r in results])
    avg_reg_base = _avg([r.base_reg for r in results])
    avg_xgb_opt = _avg([r.opt_xgb for r in results])
    avg_reg_opt = _avg([r.opt_reg for r in results])

    print("\n======= Resumen de diferencias XGBoost vs Regresión ========")
    print(f"ℹ️  Corriendo {len(pids)} casas con presupuesto ${args.budget:,.0f}")
    print("\n📊 Promedios:")
    if avg_xgb_base is not None and avg_reg_base is not None:
        print(f"  Base (prom):   XGBoost ${avg_xgb_base:,.0f} | Regresión ${avg_reg_base:,.0f}")
    if avg_base_diff is not None:
        print(f"  Base:   Δ={avg_base_diff:+,.0f} ({avg_base_pct:+.1f}%)")
    if base_diffs:
        print(f"           min Δ={min(base_diffs):+,.0f}, max Δ={max(base_diffs):+,.0f}")
    if base_pcts:
        print(f"           min Δ%={min(base_pcts):+,.1f}%, max Δ%={max(base_pcts):+,.1f}%")
    if avg_xgb_opt is not None and avg_reg_opt is not None:
        print(f"  Remodelada (prom): XGBoost ${avg_xgb_opt:,.0f} | Regresión ${avg_reg_opt:,.0f}")
    if avg_opt_diff is not None:
        print(f"  Remodelada: Δ={avg_opt_diff:+,.0f} ({avg_opt_pct:+.1f}%)")
    if opt_diffs:
        print(f"              min Δ={min(opt_diffs):+,.0f}, max Δ={max(opt_diffs):+,.0f}")
    if opt_pcts:
        print(f"              min Δ%={min(opt_pcts):+,.1f}%, max Δ%={max(opt_pcts):+,.1f}%")


if __name__ == "__main__":
    main()
