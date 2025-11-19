#!/usr/bin/env python3
import csv
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT = BASE / "mip_scenarios.csv"
PROCESSED = BASE / "mip_scenarios_processed.csv"
SUMMARY = BASE / "mip_scenarios_summary.csv"
BY_BLDG = BASE / "mip_scenarios_by_bldg.csv"
ROI_BY_BUDGET = BASE / "mip_scenarios_roi_by_budget.csv"

if not INPUT.exists():
    raise SystemExit(f"No existe {INPUT}")

rows = []
with INPUT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tag = row.get("tag", "")
        info = {"scenario": "", "profile": "", "bldg": "", "budget_tag": ""}
        for part in tag.split("|"):
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            info[key.strip()] = val.strip()
        scenario = info.get("scenario", "")
        profile = info.get("profile", row.get("profile", ""))
        bldg = info.get("bldg", row.get("bldg_type", ""))
        try:
            budget_tag = float(info.get("budget", row.get("budget", 0.0)))
        except ValueError:
            budget_tag = float(row.get("budget", 0.0))
        cost = float(row.get("cost", 0.0))
        obj = float(row.get("obj", 0.0))
        roi = obj / cost if cost not in (0.0, math.nan) else float("nan")
        slack = float(row.get("slack", 0.0))
        gr_liv = float(row.get("gr_liv_area", row.get("gr_liv_area"))) if row.get("gr_liv_area") else float(row.get("gr_liv_area", 0.0))
        beds = float(row.get("beds", 0.0))
        fullbath = float(row.get("fullbath", 0.0))
        halfbath = float(row.get("halfbath", 0.0))
        overall = float(row.get("overall_qual", 0.0))
        data = {
            **row,
            "scenario": scenario,
            "profile_tag": profile,
            "bldg_tag": bldg,
            "budget_value": budget_tag,
            "roi": roi,
            "roi_pct": roi * 100.0,
            "slack": slack,
            "gr_liv_area": gr_liv,
            "beds": beds,
            "fullbath": fullbath,
            "halfbath": halfbath,
            "overall_qual": overall,
        }
        rows.append(data)

if not rows:
    raise SystemExit("mip_scenarios.csv está vacío")

# guardar processed
fieldnames = list(rows[0].keys())
with PROCESSED.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# agrupaciones
agg_scenario = defaultdict(list)
agg_bldg = defaultdict(list)
for r in rows:
    agg_scenario[r["scenario"]].append(r)
    agg_bldg[r["bldg_tag"]].append(r)

summary_rows = []
for scen, items in agg_scenario.items():
    if not scen:
        scen = "(sin_nombre)"
    budgets = [float(it["budget_value"]) for it in items]
    rois = [it["roi"] for it in items if math.isfinite(it["roi"])]
    roi_max = max(rois) if rois else float("nan")
    roi_avg = sum(rois) / len(rois) if rois else float("nan")
    best = max(items, key=lambda it: it["roi"])
    summary_rows.append({
        "scenario": scen,
        "rows": len(items),
        "profile": best.get("profile_tag", ""),
        "bldg": best.get("bldg_tag", ""),
        "budget_min": min(budgets),
        "budget_max": max(budgets),
        "avg_roi_pct": roi_avg * 100.0 if math.isfinite(roi_avg) else float("nan"),
        "max_roi_pct": roi_max * 100.0 if math.isfinite(roi_max) else float("nan"),
        "best_budget": best.get("budget_value", 0.0),
        "avg_slack": sum(it.get("slack", 0.0) for it in items) / len(items),
        "avg_gr_liv": sum(it.get("gr_liv_area", 0.0) for it in items) / len(items),
        "avg_beds": sum(it.get("beds", 0.0) for it in items) / len(items),
        "avg_fullbath": sum(it.get("fullbath", 0.0) for it in items) / len(items),
        "avg_overall": sum(it.get("overall_qual", 0.0) for it in items) / len(items),
    })

summary_rows.sort(key=lambda r: r["max_roi_pct"], reverse=True)
with SUMMARY.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

# Guardar agrupación por tipo de edificio (solo para análisis, no se usa en LaTeX)
if agg_bldg:
    bldg_rows = []
    for bldg, items in agg_bldg.items():
        rois = [it["roi"] for it in items if math.isfinite(it["roi"])]
        roi_avg = sum(rois) / len(rois) if rois else float("nan")
        bldg_rows.append({
            "bldg": bldg,
            "rows": len(items),
            "avg_roi_pct": roi_avg * 100.0 if math.isfinite(roi_avg) else float("nan"),
            "avg_slack": sum(it.get("slack", 0.0) for it in items) / len(items),
            "avg_gr_liv": sum(it.get("gr_liv_area", 0.0) for it in items) / len(items),
            "avg_cost": sum(float(it.get("cost", 0.0)) for it in items) / len(items),
        })
    with BY_BLDG.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bldg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bldg_rows)

roi_rows = []
for r in rows:
    roi_rows.append({
        "scenario": r["scenario"],
        "bldg": r.get("bldg_tag", ""),
        "budget": float(r.get("budget_value", 0.0)),
        "roi_pct": r.get("roi_pct", float("nan")),
        "slack": r.get("slack", 0.0),
        "gr_liv_area": r.get("gr_liv_area", 0.0),
    })
with ROI_BY_BUDGET.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
    writer.writeheader()
    writer.writerows(roi_rows)

print(f"Procesado {len(rows)} registros; salidas en {PROCESSED.name}, {SUMMARY.name}, {BY_BLDG.name}, {ROI_BY_BUDGET.name}")
