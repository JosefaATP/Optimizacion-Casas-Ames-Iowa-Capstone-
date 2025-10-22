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
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# ============================ Helpers de limpieza ============================
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = ANSI_RE.sub("", s)
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
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

# ============================ Patrones robustos =============================
RE_OBJ       = re.compile(r'Valor\s*objetivo\s*\(MIP\)\s*:\s*\$?\s*([-\d,\.]+)', re.I | re.U)
RE_ROI       = re.compile(r'\bROI\s*:\s*\$?\s*([-\d,\.]+)', re.I | re.U)
RE_PCT_LINE  = re.compile(r'Porcentaje\s+Neto\s+de\s+Mejoras\s*:\s*(.+)', re.I | re.U)
RE_FIRST_NUM = re.compile(r'[-+]?\$?[\d,]+(?:\.\d+)?')
RE_PERCENT   = re.compile(r'([-+]?\$?[\d,]+(?:\.\d+)?)\s*%')
RE_RT        = re.compile(r'Tiempo\s*total\s*:\s*([-\d,\.]+)s', re.I | re.U)
RE_COST      = re.compile(r'Costos\s+totales\s*:\s*\$?\s*([-\d,\.]+)', re.I | re.U)
RE_SLACK     = re.compile(r'Slack\s+presupuesto\s*:\s*\$?\s*([-\d,\.]+)', re.I | re.U)

# Línea con PID y barrio (tiene un guion largo entre PID y Neighborhood)
RE_PID_NEI   = re.compile(r'PID\s*:\s*(\d+)\s*[–-]\s*([^|]+?)\s*\|\s*Presupuesto', re.I | re.U)

# Líneas de cambios: "- Nombre: base → nuevo (costo $...)"
RE_CHANGE    = re.compile(r'^\s*-\s*(.+?):\s*.+?→\s*.+?(?:\s*\(costo.*\))?\s*$', re.I | re.M)

def _parse_pct(line_tail: str) -> float | None:
    m = RE_PERCENT.search(line_tail)
    if m:
        return _to_float(m.group(1))
    m = RE_FIRST_NUM.search(line_tail)
    if m:
        return _to_float(m.group(1))
    return None

def _fallback_line_value(out: str, starts_with: str) -> float | None:
    cand = None
    for ln in out.splitlines():
        if starts_with.lower() in ln.lower():
            cand = ln
    if not cand:
        return None
    if "Porcentaje Neto de Mejoras" in starts_with:
        tail = cand.split(":", 1)[-1] if ":" in cand else cand
        return _parse_pct(tail)
    m = RE_FIRST_NUM.search(cand)
    return _to_float(m.group(1)) if m else None

RE_MONEY     = r'([-\d,\.]+)'

RE_OBJ       = re.compile(r'Valor\s*objetivo\s*\(MIP\)\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)

# ROI en dólares y ROI % (ambos soportan texto intermedio entre "ROI" y ":")
RE_ROI_USD   = re.compile(r'\bROI(?:\s*\(.*?\))?\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)
RE_ROI_PCT   = re.compile(r'\bROI\s*%\s*:\s*' + RE_MONEY + r'\s*%', re.I | re.U)

# tiempos y gap
RE_RT        = re.compile(r'Tiempo\s*total\s*:\s*' + RE_MONEY + r'\s*s', re.I | re.U)
RE_MIPGAP    = re.compile(r'MIP\s*Gap\s*:\s*' + RE_MONEY + r'\s*%', re.I | re.U)

# costos / slack (permite "(modelo)")
RE_COST      = re.compile(r'Costos\s+totales(?:\s*\(.*?\))?\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)
RE_SLACK     = re.compile(r'Slack\s+presupuesto\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)

# precios
RE_PRICE_BASE  = re.compile(r'Precio\s*casa\s*base\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)
RE_PRICE_OPT   = re.compile(r'Precio\s*casa\s*remodelada\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)
RE_DELTA_PRICE = re.compile(r'Δ\s*Precio\s*:\s*\$?\s*' + RE_MONEY, re.I | re.U)

# uplifts/porciones
RE_UPLIFT    = re.compile(r'Uplift\s*vs\s*base\s*:\s*' + RE_MONEY + r'\s*%', re.I | re.U)
RE_SHARE     = re.compile(r'%\s*del\s*precio\s*final\s*por\s*mejoras\s*:\s*' + RE_MONEY + r'\s*%', re.I | re.U)

# Línea con PID y barrio
RE_PID_NEI   = re.compile(r'PID\s*:\s*(\d+)\s*[–-]\s*([^|]+?)\s*\|\s*Presupuesto', re.I | re.U)

# Binarias activas
RE_BINS      = re.compile(r'Binarias\s*activas\s*:\s*(\d+)', re.I | re.U)

# Cambios (frecuencias): "- Nombre: base → nuevo (costo ...)"
RE_CHANGE    = re.compile(r'^\s*-\s*(.+?):\s*.+?→\s*.+?(?:\s*\(costo.*\))?\s*$', re.I | re.M)

def _parse_pct(line_tail: str) -> float | None:
    m = re.search(r'([-+]?\$?[\d,]+(?:\.\d+)?)', line_tail)
    return _to_float(m.group(1)) if m else None

def _fallback_line_value(out: str, starts_with: str) -> float | None:
    cand = None
    for ln in out.splitlines():
        if starts_with.lower() in ln.lower():
            cand = ln
    if not cand:
        return None
    m = re.search(r'([-+]?\$?[\d,]+(?:\.\d+)?)', cand)
    return _to_float(m.group(1)) if m else None

# ============================ Runner una corrida ============================
def run_once(pid: int, budget: float, py_exe: str, logdir: Path, tier: str, idx: int,
             basecsv: str | None) -> dict:
    cmd = [py_exe, "-m", "optimization.remodel.run_opt",
           "--pid", str(pid), "--budget", str(float(budget))]
    if basecsv:
        cmd += ["--basecsv", basecsv]

    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    raw = cp.stdout or ""
    out = _clean_text(raw)

    # Parse métricas principales
    obj = roi = pct = rt = total_cost = slack = None
    neighborhood = None
    changes = []

    m = _search_last(RE_OBJ, out);      obj = _to_float(m.group(1)) if m else None
    m = _search_last(RE_ROI, out);      roi = _to_float(m.group(1)) if m else None
    m = _search_last(RE_PCT_LINE, out); pct = _parse_pct(m.group(1)) if m else None
    m = _search_last(RE_RT, out);       rt  = _to_float(m.group(1)) if m else None
    m = _search_last(RE_COST, out);     total_cost = _to_float(m.group(1)) if m else None
    m = _search_last(RE_SLACK, out);    slack = _to_float(m.group(1)) if m else None
    m = _search_last(RE_PID_NEI, out)
    if m:
        # pid_str = m.group(1)  # no lo usamos, ya lo tenemos
        neighborhood = m.group(2).strip()

    # Cambios (frecuencias)
    for mc in RE_CHANGE.finditer(out):
        changes.append(mc.group(1).strip())

    # Fallbacks por si algo faltó
    if obj is None:  obj = _fallback_line_value(out, "Valor objetivo (MIP)")
    if roi is None:  roi = _fallback_line_value(out, "ROI")
    if pct is None:  pct = _fallback_line_value(out, "Porcentaje Neto de Mejoras")
    if rt is None:   rt  = _fallback_line_value(out, "Tiempo total")

    ok = (obj is not None and roi is not None and pct is not None)

    if not ok:
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / f"run_{tier}_{idx:02d}_pid{pid}.log").write_text(raw, encoding="utf-8")

    budget_used = None
    if slack is not None:
        budget_used = float(budget) - slack
        if budget_used < 0:
            budget_used = 0.0

    return {
        "pid": pid,
        "budget": float(budget),
        "objective_mip": obj,
        "roi": roi,
        "pct_net_improve": pct,
        "runtime_s": rt,
        "total_cost": total_cost,
        "slack": slack,
        "budget_used": budget_used,
        "neighborhood": neighborhood,
        "changes": changes,
        "raw_ok": ok,
        "raw_output": out,
    }

# ============================ Utilidades ============================
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

def _load_random_pids(basecsv: str, n: int, seed: int) -> list[int]:
    df = pd.read_csv(basecsv)
    if "PID" not in df.columns:
        raise ValueError(f"El CSV '{basecsv}' no tiene columna 'PID'.")
    pids = [int(x) for x in df["PID"].dropna().unique().tolist()]
    random.seed(seed)
    random.shuffle(pids)
    return pids[:min(n, len(pids))]

# ============================ Main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basecsv", type=str, required=True,
                    help="Ruta al CSV base (de donde se toman los PIDs al azar)")
    ap.add_argument("--n_houses", type=int, default=50,
                    help="Número de casas distintas a evaluar (default: 50)")
    ap.add_argument("--py", type=str, default=sys.executable, help="Python a usar (default: actual)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    ap.add_argument("--outdir", type=str, default="bench_out", help="Carpeta de salida (CSV + figuras)")
    ap.add_argument("--topn_neigh", type=int, default=10, help="Top-N neighborhoods para gráficos")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / "logs"

    # Tiers con presupuestos fijos
    tiers = {"low": 18_000.0, "mid": 50_000.0, "high": 120_000.0}

    # PIDs
    pids = _load_random_pids(args.basecsv, args.n_houses, args.seed)
    if not pids:
        print("⚠ No se encontraron PIDs en el CSV base.")
        return
    print(f"Se evaluarán {len(pids)} casas distintas con 3 presupuestos fijos por casa.\n")

    # Corridas
    all_rows = []
    idx_global = 0
    for j, pid in enumerate(pids, start=1):
        print(f"=== CASA #{j:02d} | PID={pid} ===")
        for tier, budget in tiers.items():
            idx_global += 1
            res = run_once(pid, budget, py_exe=args.py, logdir=logdir,
                           tier=tier, idx=idx_global, basecsv=args.basecsv)
            res["tier"] = tier
            all_rows.append(res)
            ok = "OK" if res["raw_ok"] else "MISS"
            print(f"[{tier:>4}] budget=${budget:,.0f} → obj={res['objective_mip']} "
                  f"roi={res['roi']} pct={res['pct_net_improve']} [{ok}]")
        print()

    # ============================ Guardar CSV ============================
    csv_path = outdir / "remodel_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tier","pid","neighborhood","budget","budget_used","total_cost","slack",
            "objective_mip","roi","pct_net_improve","runtime_s","raw_ok"
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "tier": r.get("tier"), "pid": r.get("pid"),
                "neighborhood": r.get("neighborhood"),
                "budget": r.get("budget"),
                "budget_used": r.get("budget_used"),
                "total_cost": r.get("total_cost"),
                "slack": r.get("slack"),
                "objective_mip": r.get("objective_mip"),
                "roi": r.get("roi"),
                "pct_net_improve": r.get("pct_net_improve"),
                "runtime_s": r.get("runtime_s"),
                "raw_ok": r.get("raw_ok"),
            })
    print(f"\n✔ Resultados guardados en: {csv_path}")

    # ============================ Resumen por tier ============================
    print("\n=== RESUMEN POR TIER ===")
    for tier in tiers.keys():
        rows_ok = [r for r in all_rows if r["tier"] == tier and r["raw_ok"]]
        s_obj = summarize(rows_ok, "objective_mip")
        s_roi = summarize(rows_ok, "roi")
        s_pct = summarize(rows_ok, "pct_net_improve")
        s_used = summarize(rows_ok, "budget_used")

        def fmt(s):
            return (f"n={s['n']}, mean={s['mean']:.2f} (std={s['std']:.2f}), "
                    f"p50={s['p50']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}") if s["n"] else "n=0"

        print(f"\n[{tier}]")
        print("  Obj (MIP):     ", fmt(s_obj))
        print("  ROI:           ", fmt(s_roi))
        print("  % Net mejora:  ", fmt(s_pct))
        print("  Budget usado:  ", fmt(s_used))

    # ============================ Gráficas por tier ============================
    def _vals(metric, tier_key):
        return [r[metric] for r in all_rows
                if r["tier"] == tier_key and r["raw_ok"] and r.get(metric) is not None]

    def boxplot_metric(metric, ylabel, fname):
        data, labels = [], []
        for t in tiers.keys():
            vals = _vals(metric, t)
            if vals:
                data.append(vals); labels.append(t.upper())
        if not data:
            print(f"⚠ No hay datos válidos para {ylabel}; se omite figura.")
            return
        fig = plt.figure()
        plt.boxplot(data, tick_labels=labels)
        plt.title(f"{ylabel} por tier (presupuestos fijos)")
        plt.ylabel(ylabel); plt.xlabel("Tier de presupuesto")
        fig.tight_layout(); out = outdir / fname
        plt.savefig(out, dpi=160); plt.close(fig)
        print(f"✔ Figura: {out}")

    boxplot_metric("objective_mip", "Valor objetivo (MIP)", "box_obj_mip.png")
    boxplot_metric("roi", "ROI", "box_roi.png")

    # Barras: % neto de mejoras
    means, lbls = [], []
    for t in tiers.keys():
        vals = _vals("pct_net_improve", t)
        if vals:
            means.append(sum(vals)/len(vals)); lbls.append(t.upper())
    if means:
        fig = plt.figure()
        plt.bar(lbls, means)
        plt.title("Promedio % neto de mejoras por tier (presupuestos fijos)")
        plt.ylabel("% neto de mejoras")
        fig.tight_layout(); out = outdir / "bar_pct_net.png"
        plt.savefig(out, dpi=160); plt.close(fig)
        print(f"✔ Figura: {out}")

    # Violin: budget usado por tier
    used_data, used_labels = [], []
    for t in tiers.keys():
        vals = _vals("budget_used", t)
        if vals:
            used_data.append(vals); used_labels.append(t.upper())
    if used_data:
        fig = plt.figure()
        plt.violinplot(used_data, showmedians=True)
        plt.xticks(range(1, len(used_labels)+1), used_labels)
        plt.title("Budget usado por tier")
        plt.ylabel("USD usados")
        fig.tight_layout(); out = outdir / "violin_budget_used.png"
        plt.savefig(out, dpi=160); plt.close(fig)
        print(f"✔ Figura: {out}")

    # Scatter: budget usado vs objetivo
    any_ok = any(r["raw_ok"] and r.get("objective_mip") is not None for r in all_rows)
    if any_ok:
        tier_colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}
        fig = plt.figure()
        for t in tiers.keys():
            xs = [r["budget_used"] for r in all_rows if r["tier"]==t and r["raw_ok"] and r.get("budget_used") is not None]
            ys = [r["objective_mip"] for r in all_rows if r["tier"]==t and r["raw_ok"] and r.get("objective_mip") is not None]
            if xs and ys:
                plt.scatter(xs, ys, label=t.upper(), color=tier_colors.get(t))
        plt.legend(); plt.title("Budget usado vs Valor objetivo (MIP)")
        plt.xlabel("Budget usado ($)"); plt.ylabel("Valor objetivo (MIP)")
        fig.tight_layout(); out = outdir / "scatter_used_obj.png"
        plt.savefig(out, dpi=160); plt.close(fig)
        print(f"✔ Figura: {out}")

        # Correlación (Pearson) global entre used y obj
        df_corr = pd.DataFrame([
            {"used": r["budget_used"], "obj": r["objective_mip"]}
            for r in all_rows if r["raw_ok"] and r.get("budget_used") is not None and r.get("objective_mip") is not None
        ])
        if len(df_corr) >= 3:
            corr = df_corr["used"].corr(df_corr["obj"])
            print(f"\n📎 Correlación Pearson (budget usado vs objetivo) = {corr:.3f}")

    # ============================ Top cambios por tier ============================
    tier_changes: dict[str, Counter] = {t: Counter() for t in tiers.keys()}
    for r in all_rows:
        if r["raw_ok"] and r.get("changes"):
            tier_changes[r["tier"]].update(r["changes"])

    topch_rows = []
    for t in tiers.keys():
        top5 = tier_changes[t].most_common(10)
        if not top5: continue
        for name, cnt in top5:
            topch_rows.append({"tier": t, "change": name, "count": cnt})

    if topch_rows:
        df_top = pd.DataFrame(topch_rows)
        df_top.to_csv(outdir / "top_changes_by_tier.csv", index=False, encoding="utf-8")
        print(f"✔ Top cambios por tier → {outdir/'top_changes_by_tier.csv'}")

        # Bar charts por tier (top-10)
        for t in tiers.keys():
            top10 = tier_changes[t].most_common(10)
            if not top10: continue
            labels = [k for k,_ in top10]; counts = [v for _,v in top10]
            fig = plt.figure(figsize=(8, 4.5))
            plt.barh(labels[::-1], counts[::-1])
            plt.title(f"Top 10 cambios – {t.upper()}")
            plt.xlabel("Frecuencia"); plt.tight_layout()
            out = outdir / f"top_changes_{t}.png"
            plt.savefig(out, dpi=160); plt.close(fig)
            print(f"✔ Figura: {out}")

    # ============================ Resumen por Neighborhood ============================
    rows_ok = [r for r in all_rows if r["raw_ok"]]
    df = pd.DataFrame(rows_ok)
    if "neighborhood" in df.columns and df["neighborhood"].notna().any():
        agg = (df.groupby("neighborhood")[["objective_mip","roi","pct_net_improve","budget_used"]]
                 .agg(["count","mean","median","std","min","max"]))
        agg.to_csv(outdir / "summary_by_neighborhood.csv", encoding="utf-8")
        print(f"✔ Resumen por neighborhood → {outdir/'summary_by_neighborhood.csv'}")

        # Top-N neighborhoods por objetivo medio (solo si hay suficientes)
        means = (df.groupby("neighborhood")["objective_mip"].mean()
                   .sort_values(ascending=False).head(args.topn_neigh))
        if not means.empty:
            fig = plt.figure(figsize=(9, 5))
            means.plot(kind="bar")
            plt.ylabel("Objetivo (MIP) promedio")
            plt.title(f"Top {args.topn_neigh} neighborhoods por objetivo medio")
            plt.tight_layout()
            out = outdir / "bar_neigh_topN_obj.png"
            plt.savefig(out, dpi=160); plt.close(fig)
            print(f"✔ Figura: {out}")
    else:
        print("⚠ No se pudo extraer 'Neighborhood' del output; se omiten gráficos por barrio.")

if __name__ == "__main__":
    main()
