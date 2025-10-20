#!/usr/bin/env python3
import argparse
import csv
import random
import re
import statistics as stats
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ----------------------------
# Regex robustos para parseo
# ----------------------------
# Aceptan números con separadores de miles, signo y decimales
_MONEY = r'[-+]?\$?[\d,]+(?:\.\d+)?'
_NUM   = r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)'

RE_OBJ   = re.compile(r'Valor objetivo\s*\(MIP\)\s*:\s*\$?\s*([-\d,\.]+)', re.I)
RE_ROI   = re.compile(r'ROI\s*:\s*\$?\s*([-\d,\.]+)', re.I)
# En tus prints aparece a veces con un "$" antes del porcentaje; lo ignoramos
RE_PCT   = re.compile(r'Porcentaje\s+Neto\s+de\s+Mejoras\s*:\s*\$?\s*([-\d,\.]+)\s*%', re.I)
RE_BUD   = re.compile(r'Presupuesto:\s*\$?\s*([-\d,\.]+)', re.I)
RE_RT    = re.compile(r'Tiempo\s+total:\s*([-\d,\.]+)s', re.I)

def _to_float(txt: str) -> float | None:
    try:
        return float(str(txt).replace(",", "").replace("$", "").strip())
    except Exception:
        return None

def run_once(pid: int, budget: float, py_exe: str = sys.executable) -> dict:
    """
    Ejecuta el optimizador 1 vez con el presupuesto dado y devuelve un dict con métricas parseadas.
    """
    cmd = [
        py_exe, "-m", "optimization.remodel.run_opt",
        "--pid", str(pid),
        "--budget", str(float(budget)),
    ]
    # Capturamos stdout/stderr (tu script imprime en stdout)
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    out = cp.stdout or ""

    # Parseo:
    obj  = None
    roi  = None
    pct  = None
    rt   = None

    m = RE_OBJ.search(out)
    if m: obj = _to_float(m.group(1))

    m = RE_ROI.search(out)
    if m: roi = _to_float(m.group(1))

    m = RE_PCT.search(out)
    if m: pct = _to_float(m.group(1))

    m = RE_RT.search(out)
    if m: rt = _to_float(m.group(1))

    return {
        "pid": pid,
        "budget": float(budget),
        "objective_mip": obj,
        "roi": roi,
        "pct_net_improve": pct,
        "runtime_s": rt,
        "raw_ok": obj is not None and roi is not None and pct is not None,
        "raw_output": out,  # útil si alguna corrida no parsea
    }

def summarize(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "p50": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": stats.mean(vals),
        "std": stats.pstdev(vals) if len(vals) > 1 else 0.0,
        "p50": stats.median(vals),
        "min": min(vals),
        "max": max(vals),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=528344070, help="PID de la casa (default: 528344070)")
    ap.add_argument("--trials", type=int, default=50, help="Corridas por tier (default: 50)")
    ap.add_argument("--py", type=str, default=sys.executable, help="Python a usar (default: actual)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    ap.add_argument("--outdir", type=str, default="bench_out", help="Carpeta de salida (CSV + figuras)")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Tiers de presupuesto
    tiers = {
        "low":  (15000, 40000),
        "mid":  (40000, 75000),
        "high": (75000, 200000),
    }

    all_rows = []
    for tier, (lo, hi) in tiers.items():
        print(f"\n=== Tier {tier.upper()} | rango ${lo:,} – ${hi:,} | {args.trials} corridas ===")
        for i in range(args.trials):
            # Toma presupuesto aleatorio uniforme en el rango
            budget = random.uniform(lo, hi)
            res = run_once(args.pid, budget, py_exe=args.py)
            res["tier"] = tier
            all_rows.append(res)
            ok = "OK" if res["raw_ok"] else "MISS"
            print(f"[{tier:>4}] #{i+1:02d} budget=${budget:,.0f} → obj={res['objective_mip']} roi={res['roi']} pct={res['pct_net_improve']} [{ok}]")

    # Guardar CSV
    csv_path = outdir / "remodel_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tier","pid","budget","objective_mip","roi","pct_net_improve","runtime_s","raw_ok"
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\n✔ Resultados guardados en: {csv_path}")

    # Resúmenes por tier
    print("\n=== RESUMEN POR TIER ===")
    for tier in tiers.keys():
        rows = [r for r in all_rows if r["tier"] == tier and r["raw_ok"]]
        s_obj = summarize(rows, "objective_mip")
        s_roi = summarize(rows, "roi")
        s_pct = summarize(rows, "pct_net_improve")

        def fmt(s):
            return f"n={s['n']}, mean={s['mean']:.2f} (std={s['std']:.2f}), p50={s['p50']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}" if s["n"] else "n=0"

        print(f"\n[{tier}]")
        print("  Obj (MIP):     ", fmt(s_obj))
        print("  ROI:           ", fmt(s_roi))
        print("  % Net mejora:  ", fmt(s_pct))

    # ----------------------------
    # Gráficas para presentar
    # ----------------------------
    # Boxplots por tier (obj y ROI)
    def boxplot_metric(metric, ylabel, fname):
        fig = plt.figure()
        data = [ [r[metric] for r in all_rows if r["tier"]==tier and r["raw_ok"]] for tier in tiers.keys() ]
        labels = [t.upper() for t in tiers.keys()]
        plt.boxplot(data, labels=labels)
        plt.title(f"{ylabel} por tier")
        plt.ylabel(ylabel)
        plt.xlabel("Tier de presupuesto")
        fig.tight_layout()
        out = outdir / fname
        plt.savefig(out, dpi=160)
        plt.close(fig)
        print(f"✔ Figura: {out}")

    boxplot_metric("objective_mip", "Valor objetivo (MIP)", "box_obj_mip.png")
    boxplot_metric("roi", "ROI", "box_roi.png")

    # Barras: promedio % neto de mejoras por tier
    fig = plt.figure()
    means = []
    labels = []
    for t in tiers.keys():
        vals = [r["pct_net_improve"] for r in all_rows if r["tier"] == t and r["raw_ok"]]
        if vals:
            means.append(sum(vals)/len(vals))
            labels.append(t.upper())
    plt.bar(labels, means)
    plt.title("Promedio % neto de mejoras por tier")
    plt.ylabel("% neto de mejoras")
    fig.tight_layout()
    out = outdir / "bar_pct_net.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"✔ Figura: {out}")

    # Scatter: presupuesto vs valor objetivo (todas las corridas, color por tier)
    tier_colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}
    fig = plt.figure()
    for t in tiers.keys():
        xs = [r["budget"] for r in all_rows if r["tier"]==t and r["raw_ok"]]
        ys = [r["objective_mip"] for r in all_rows if r["tier"]==t and r["raw_ok"]]
        plt.scatter(xs, ys, label=t.upper())
    plt.legend()
    plt.title("Presupuesto vs Valor objetivo (MIP)")
    plt.xlabel("Presupuesto ($)")
    plt.ylabel("Valor objetivo (MIP)")
    fig.tight_layout()
    out = outdir / "scatter_budget_obj.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"✔ Figura: {out}")

if __name__ == "__main__":
    main()
