# optimization/remodel/benchmark_remodel.py
#!/usr/bin/env python3
import argparse
import csv
import random
import re
import statistics as stats
import subprocess
import sys
from pathlib import Path
import os


import matplotlib.pyplot as plt

# ============== Limpieza / normalización de salida ==============
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = ANSI_RE.sub("", s)             # quita escapes ANSI
    s = s.replace("\u00A0", " ")       # NBSP -> espacio normal
    s = s.replace("\u202F", " ")       # narrow NBSP -> espacio
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def _to_float(txt: str) -> float | None:
    try:
        return float(str(txt).replace(",", "").replace("$", "").strip())
    except Exception:
        return None

def _search_last(pat: re.Pattern, text: str):
    last = None
    for m in pat.finditer(text):
        last = m
    return last

# ============== Patrones (muy permisivos) ==============
RE_OBJ = re.compile(r'Valor\s*objetivo\s*\(MIP\)\s*:\s*\$?\s*([-\d,\.]+)', re.I | re.U)
RE_ROI = re.compile(r'\bROI\s*:\s*\$?\s*([-\d,\.]+)', re.I | re.U)
# En tu print a veces pones un "$" antes del porcentaje; capturamos número + '%'
RE_PCT_LINE = re.compile(r'Porcentaje\s+Neto\s+de\s+Mejoras\s*:\s*(.+)', re.I | re.U)
RE_FIRST_NUM = re.compile(r'[-+]?\$?[\d,]+(?:\.\d+)?')     # primer número con comas/decimales
RE_PERCENT   = re.compile(r'([-+]?\$?[\d,]+(?:\.\d+)?)\s*%')  # número seguido de %

RE_RT  = re.compile(r'Tiempo\s*total\s*:\s*([-\d,\.]+)s', re.I | re.U)

def _parse_pct(line_tail: str) -> float | None:
    # intenta capturar "29%" o "$29%"; si no, primer número
    m = RE_PERCENT.search(line_tail)
    if m:
        return _to_float(m.group(1))
    m = RE_FIRST_NUM.search(line_tail)
    if m:
        return _to_float(m.group(1))
    return None

def _fallback_line_value(out: str, starts_with: str) -> float | None:
    # Busca la última línea que contenga la etiqueta y extrae el primer número
    cand = None
    for ln in out.splitlines():
        if starts_with.lower() in ln.lower():
            cand = ln
    if not cand:
        return None
    if "Porcentaje Neto de Mejoras" in starts_with:
        # partir después de ':'
        tail = cand.split(":", 1)[-1] if ":" in cand else cand
        return _parse_pct(tail)
    # genérico: primer número
    m = RE_FIRST_NUM.search(cand)
    return _to_float(m.group(1)) if m else None

def run_once(pid: int, budget: float, py_exe: str, logdir: Path, tier: str, idx: int) -> dict:
    cmd = [
        py_exe, "-m", "optimization.remodel.run_opt",
        "--pid", str(pid),
        "--budget", str(float(budget)),
    ]
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",         # <-- fuerza decodificación UTF-8
        errors="replace",         # <-- no muere si aparece algo raro
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}  # <-- pide al hijo emitir UTF-8
    )
    out = cp.stdout or ""

    raw = cp.stdout or ""
    out = _clean_text(raw)

    # 1) Regex directos (última coincidencia)
    obj = roi = pct = rt = None

    m = _search_last(RE_OBJ, out)
    if m: obj = _to_float(m.group(1))

    m = _search_last(RE_ROI, out)
    if m: roi = _to_float(m.group(1))

    # porcentaje: primero capturamos la línea y luego extraemos número
    m = _search_last(RE_PCT_LINE, out)
    if m:
        pct = _parse_pct(m.group(1))

    m = _search_last(RE_RT, out)
    if m: rt = _to_float(m.group(1))

    # 2) Fallback por línea si algo faltó
    if obj is None:
        obj = _fallback_line_value(out, "Valor objetivo (MIP)")
    if roi is None:
        roi = _fallback_line_value(out, "ROI")
    if pct is None:
        pct = _fallback_line_value(out, "Porcentaje Neto de Mejoras")
    if rt is None:
        rt = _fallback_line_value(out, "Tiempo total")

    ok = (obj is not None and roi is not None and pct is not None)

    # 3) Si MISS, guardar log para inspección
    if not ok:
        logdir.mkdir(parents=True, exist_ok=True)
        log_path = logdir / f"run_{tier}_{idx:02d}.log"
        try:
            log_path.write_text(raw, encoding="utf-8")
        except Exception:
            pass

    return {
        "pid": pid,
        "budget": float(budget),
        "objective_mip": obj,
        "roi": roi,
        "pct_net_improve": pct,
        "runtime_s": rt,
        "raw_ok": ok,
        "raw_output": out,
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
    logdir = outdir / "logs"

    tiers = {
        "low":  (15000, 40000),
        "mid":  (40000, 75000),
        "high": (75000, 200000),
    }

    all_rows = []
    for tier, (lo, hi) in tiers.items():
        print(f"\n=== Tier {tier.upper()} | rango ${lo:,} – ${hi:,} | {args.trials} corridas ===")
        for i in range(args.trials):
            budget = random.uniform(lo, hi)
            res = run_once(args.pid, budget, py_exe=args.py, logdir=logdir, tier=tier, idx=i+1)
            res["tier"] = tier
            all_rows.append(res)
            ok = "OK" if res["raw_ok"] else "MISS"
            print(f"[{tier:>4}] #{i+1:02d} budget=${budget:,.0f} → obj={res['objective_mip']} roi={res['roi']} pct={res['pct_net_improve']} [{ok}]")

    # CSV
    csv_path = outdir / "remodel_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tier","pid","budget","objective_mip","roi","pct_net_improve","runtime_s","raw_ok"
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\n✔ Resultados guardados en: {csv_path}")

    # Resumen
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

    # ===== Gráficas (solo si hay datos válidos) =====
    def boxplot_metric(metric, ylabel, fname):
        data = []
        labels = []
        for t in tiers.keys():
            vals = [r[metric] for r in all_rows if r["tier"]==t and r["raw_ok"] and r.get(metric) is not None]
            if vals:
                data.append(vals)
                labels.append(t.upper())
        if not data:
            print(f"⚠ No hay datos válidos para {ylabel}; se omite figura.")
            return
        fig = plt.figure()
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

    # Barras: % neto de mejoras
    means, lbls = [], []
    for t in tiers.keys():
        vals = [r["pct_net_improve"] for r in all_rows if r["tier"] == t and r["raw_ok"] and r.get("pct_net_improve") is not None]
        if vals:
            means.append(sum(vals)/len(vals))
            lbls.append(t.upper())
    if means:
        fig = plt.figure()
        plt.bar(lbls, means)
        plt.title("Promedio % neto de mejoras por tier")
        plt.ylabel("% neto de mejoras")
        fig.tight_layout()
        out = outdir / "bar_pct_net.png"
        plt.savefig(out, dpi=160)
        plt.close(fig)
        print(f"✔ Figura: {out}")
    else:
        print("⚠ No hay datos válidos para % neto de mejoras; se omite figura.")

    # Scatter presupuesto vs objetivo
    any_ok = any(r["raw_ok"] and r.get("objective_mip") is not None for r in all_rows)
    if any_ok:
        tier_colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}
        fig = plt.figure()
        for t in tiers.keys():
            xs = [r["budget"] for r in all_rows if r["tier"]==t and r["raw_ok"] and r.get("objective_mip") is not None]
            ys = [r["objective_mip"] for r in all_rows if r["tier"]==t and r["raw_ok"] and r.get("objective_mip") is not None]
            if xs and ys:
                plt.scatter(xs, ys, label=t.upper(), color=tier_colors.get(t))
        plt.legend()
        plt.title("Presupuesto vs Valor objetivo (MIP)")
        plt.xlabel("Presupuesto ($)")
        plt.ylabel("Valor objetivo (MIP)")
        fig.tight_layout()
        out = outdir / "scatter_budget_obj.png"
        plt.savefig(out, dpi=160)
        plt.close(fig)
        print(f"✔ Figura: {out}")
    else:
        print("⚠ No hay datos válidos para scatter; se omite figura.")

if __name__ == "__main__":
    main()
