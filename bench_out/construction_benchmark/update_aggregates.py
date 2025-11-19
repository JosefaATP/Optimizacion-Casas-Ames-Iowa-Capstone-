#!/usr/bin/env python3
"""
Recalcula tablas agregadas a partir de bench_out/benchmark_runs.csv.
"""
import csv
import json
import math
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "benchmark_runs.csv")
OUTDIR = SCRIPT_DIR
BASE_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "df_final_regresion.csv")
if not os.path.exists(BASE_CSV):
    BASE_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "base_completa_sin_nulos.csv")

FEATURES = [
    ("gr_liv_area", "Gr Liv Area", "gr_liv_area"),
    ("bsmt", "Total Bsmt SF", "bsmt"),
    ("garage_area", "Garage Area", "garage_area"),
    ("beds", "Bedroom AbvGr", "beds"),
    ("fullbath", "Full Bath", "fullbath"),
    ("overall", "Overall Qual", "overall_qual"),
]
SELECT_NEIGHS = ["SawyerW", "Veenker", "MeadowV", "Blueste", "Blmngtn", "NoRidge"]


def to_float(val):
    try:
        return float(val)
    except Exception:
        return math.nan


def read_rows():
    rows = []
    with open(SOURCE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit("No rows read from benchmark_runs.csv")
    return rows


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summarize(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {}
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0

    def pct(p):
        if n == 1:
            return vals[0]
        k = p * (n - 1)
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return vals[lo]
        frac = k - lo
        return vals[lo] + frac * (vals[hi] - vals[lo])

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": vals[0],
        "p25": pct(0.25),
        "median": pct(0.5),
        "p75": pct(0.75),
        "max": vals[-1],
    }


def main():
    rows = read_rows()
    per_neigh = defaultdict(lambda: {"sum_delta": 0.0, "sum_pct": 0.0, "sum_price": 0.0, "sum_roi": 0.0, "count": 0})
    per_budget = defaultdict(lambda: {
        "sum_price": 0.0, "sum_reg": 0.0, "sum_delta": 0.0,
        "sum_delta_pct": 0.0, "sum_slack": 0.0, "sum_obj": 0.0,
        "sum_roi": 0.0, "sum_budget_pct": 0.0,
        "count": 0, "neg": 0
    })
    per_lot = defaultdict(lambda: {
        "sum_price": 0.0, "sum_reg": 0.0, "sum_delta": 0.0,
        "sum_delta_pct": 0.0, "sum_roi": 0.0, "count": 0
    })
    bench_features = defaultdict(lambda: defaultdict(float))
    bench_counts = defaultdict(int)

    summary = defaultdict(list)

    for row in rows:
        neigh = row.get("neigh", "N/A")
        budget = to_float(row.get("budget"))
        lot = to_float(row.get("lot"))
        yp = to_float(row.get("y_price"))
        rp = to_float(row.get("reg_price"))
        da = to_float(row.get("delta_reg_abs"))
        dp = to_float(row.get("delta_reg_pct"))
        sl = to_float(row.get("slack"))
        cost = to_float(row.get("cost"))
        runtime = to_float(row.get("runtime_s"))
        obj = to_float(row.get("obj"))
        price_out = to_float(row.get("y_price_out"))

        for key, val in (("y_price", yp), ("reg_price", rp), ("delta_abs", da),
                         ("delta_pct", dp), ("slack", sl), ("cost", cost),
                         ("runtime", runtime), ("obj", obj)):
            if math.isfinite(val):
                summary[key].append(val)

        roi_val = math.nan
        if math.isfinite(yp) and math.isfinite(cost) and cost > 0:
            roi_val = (yp - cost) / cost
            summary["roi"].append(roi_val)
        value_improve_pct = math.nan
        if math.isfinite(price_out) and math.isfinite(yp) and yp not in (0.0, None):
            value_improve_pct = ((price_out - yp) / yp) * 100.0
            if math.isfinite(value_improve_pct):
                summary["value_improve_pct"].append(value_improve_pct)
        budget_used_pct = math.nan
        if math.isfinite(cost) and math.isfinite(budget) and budget > 0:
            budget_used_pct = (cost / budget) * 100.0
            summary["budget_used_pct"].append(budget_used_pct)

        sc_neigh = per_neigh[neigh]
        if math.isfinite(da):
            sc_neigh["sum_delta"] += da
        if math.isfinite(dp):
            sc_neigh["sum_pct"] += dp
        if math.isfinite(yp):
            sc_neigh["sum_price"] += yp
        if math.isfinite(roi_val):
            sc_neigh["sum_roi"] += roi_val
        sc_neigh["count"] += 1

        if math.isfinite(budget) and math.isfinite(yp):
            sc = per_budget[budget]
            sc["sum_price"] += yp
            if math.isfinite(rp):
                sc["sum_reg"] += rp
            if math.isfinite(da):
                sc["sum_delta"] += da
            if math.isfinite(dp):
                sc["sum_delta_pct"] += dp
            if math.isfinite(sl):
                sc["sum_slack"] += sl
            if math.isfinite(obj):
                sc["sum_obj"] += obj
                if obj < 0:
                    sc["neg"] += 1
            if math.isfinite(roi_val):
                sc["sum_roi"] += roi_val
            if math.isfinite(budget_used_pct):
                sc["sum_budget_pct"] += budget_used_pct
            sc["count"] += 1

        if math.isfinite(lot) and math.isfinite(yp):
            sc = per_lot[lot]
            sc["sum_price"] += yp
            if math.isfinite(rp):
                sc["sum_reg"] += rp
            if math.isfinite(da):
                sc["sum_delta"] += da
            if math.isfinite(dp):
                sc["sum_delta_pct"] += dp
            if math.isfinite(roi_val):
                sc["sum_roi"] += roi_val
            sc["count"] += 1
        for feat_key, _, row_field in FEATURES:
            val = to_float(row.get(row_field))
            if math.isfinite(val):
                bench_features[neigh][f"sum_{feat_key}"] += val
        bench_counts[neigh] += 1

    delta_rows = []
    for neigh, sc in per_neigh.items():
        if sc["count"] == 0:
            continue
        delta_rows.append([
            neigh,
            sc["sum_delta"] / sc["count"],
            sc["sum_pct"] / sc["count"],
            sc["sum_price"] / sc["count"],
            sc["count"],
        ])
    delta_rows_sorted = sorted(delta_rows, key=lambda r: r[1], reverse=True)
    write_csv(os.path.join(OUTDIR, "delta_by_neighborhood.csv"),
              ["neighborhood", "avg_delta_abs", "avg_delta_pct", "avg_price", "n_runs"],
              delta_rows_sorted)
    top10 = delta_rows_sorted[:10]
    write_csv(os.path.join(OUTDIR, "delta_by_neighborhood_top.csv"),
              ["neighborhood", "avg_delta_abs", "avg_delta_pct", "avg_price", "n_runs"],
              top10)

    budget_rows = []
    for budget, sc in sorted(per_budget.items()):
        if sc["count"] == 0:
            continue
        budget_rows.append([
            budget,
            sc["sum_price"] / sc["count"],
            sc["sum_reg"] / sc["count"],
            sc["sum_delta"] / sc["count"],
            sc["sum_delta_pct"] / sc["count"] if sc["count"] else math.nan,
            sc["sum_slack"] / sc["count"],
            sc["sum_obj"] / sc["count"],
            (sc["sum_roi"] / sc["count"]) if sc["count"] else math.nan,
            (sc["sum_budget_pct"] / sc["count"]) if sc["count"] else math.nan,
            sc["neg"],
            sc["count"],
        ])
    write_csv(os.path.join(OUTDIR, "price_by_budget.csv"),
              ["budget", "avg_price_xgb", "avg_price_reg", "avg_delta_abs", "avg_delta_pct", "avg_slack", "avg_obj", "avg_roi", "avg_budget_used_pct", "neg_runs", "n_runs"],
              budget_rows)

    lot_rows = []
    for lot, sc in sorted(per_lot.items()):
        if sc["count"] == 0:
            continue
        lot_rows.append([
            lot,
            sc["sum_price"] / sc["count"],
            sc["sum_reg"] / sc["count"],
            sc["sum_delta"] / sc["count"],
            sc["sum_delta_pct"] / sc["count"] if sc["count"] else math.nan,
            (sc["sum_roi"] / sc["count"]) if sc["count"] else math.nan,
            sc["count"],
        ])
    write_csv(os.path.join(OUTDIR, "price_by_lot.csv"),
              ["lot_area", "avg_price_xgb", "avg_price_reg", "avg_delta_abs", "avg_delta_pct", "avg_roi", "n_runs"],
              lot_rows)

    summary_out = {k: summarize(v) for k, v in summary.items()}
    with open(os.path.join(OUTDIR, "summary_overview.json"), "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    # ROI histogram
    roi_vals = summary.get("roi", [])
    bins = []
    lower = -0.2
    step = 0.05
    for i in range(12):
        bins.append((lower + i * step, lower + (i + 1) * step))
    hist_counts = []
    for low, high in bins:
        count = sum(1 for v in roi_vals if math.isfinite(v) and low <= v < high)
        label = f"{low:.2f}--{high:.2f}"
        hist_counts.append([label, count])
    write_csv(os.path.join(OUTDIR, "roi_hist.csv"), ["bucket", "count"], hist_counts)

    roi_rows = []
    for neigh, sc in per_neigh.items():
        if sc["count"] == 0:
            continue
        roi_rows.append([
            neigh,
            sc["sum_roi"] / sc["count"],
            sc["sum_price"] / sc["count"],
            sc["sum_delta"] / sc["count"],
            sc["count"],
        ])
    write_csv(os.path.join(OUTDIR, "roi_by_neighborhood.csv"),
              ["neighborhood", "avg_roi", "avg_price", "avg_delta_abs", "n_runs"],
              sorted(roi_rows, key=lambda r: r[1], reverse=True))

    # Baseline features from dataset
    base_features = defaultdict(lambda: defaultdict(float))
    base_counts = defaultdict(int)
    with open(BASE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.replace("\ufeff", "").strip() for h in next(reader)]
        neigh_idx = header.index("Neighborhood")
        col_idx = {}
        for feat_key, col_name, _ in FEATURES:
            try:
                col_idx[feat_key] = header.index(col_name)
            except ValueError:
                col_idx[feat_key] = None
        for row in reader:
            neigh = row[neigh_idx].strip()
            if not neigh:
                continue
            for feat_key, _, _ in FEATURES:
                idx = col_idx.get(feat_key)
                if idx is None:
                    continue
                val = to_float(row[idx])
                if math.isfinite(val):
                    base_features[neigh][f"sum_{feat_key}"] += val
            base_counts[neigh] += 1

    all_neighs = sorted(set(list(bench_counts.keys()) + list(base_counts.keys())))
    header = ["neighborhood"]
    for feat_key, _, _ in FEATURES:
        header += [f"bench_{feat_key}", f"base_{feat_key}"]
    wide_rows = []
    for neigh in all_neighs:
        row = [neigh]
        b_count = bench_counts.get(neigh, 0)
        base_count = base_counts.get(neigh, 0)
        for feat_key, _, _ in FEATURES:
            bench_avg = (bench_features[neigh].get(f"sum_{feat_key}", 0.0) / b_count) if b_count else math.nan
            base_avg = (base_features[neigh].get(f"sum_{feat_key}", 0.0) / base_count) if base_count else math.nan
            row += [bench_avg, base_avg]
        wide_rows.append(row)
    write_csv(os.path.join(OUTDIR, "features_vs_baseline.csv"), header, wide_rows)

    selected_rows = []
    for neigh in SELECT_NEIGHS:
        for row in wide_rows:
            if row[0] == neigh:
                selected_rows.append(row)
                break
    write_csv(os.path.join(OUTDIR, "features_vs_baseline_top.csv"), header, selected_rows)

    print("Aggregates updated under", OUTDIR)


if __name__ == "__main__":
    main()
