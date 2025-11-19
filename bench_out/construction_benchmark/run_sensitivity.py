#!/usr/bin/env python3
"""
Genera análisis de sensibilidad para construcción:
 - Base profile
 - Sensibilidad univariada
 - Sensibilidad bivariada
 - Escenarios de diseño
 - Tornado de costos

Uso típico:
PYTHONPATH=. python bench_out/construction_benchmark/run_sensitivity.py \
    --reg-model models/reg/base_reg_vale.joblib \
    --basecsv data/raw/df_final_regresion.csv \
    --base-neigh GrnHill --base-lot 7000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from optimization.construction import costs
from optimization.construction.xgb_predictor import XGBBundle, PATHS as CONST_PATHS
from optimization.construction.io import load_base_df
from optimization.construction.preprocess_regresion import load_regression_reference_df, prepare_regression_input

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_BASECSV = os.path.join(ROOT, "data", "raw", "df_final_regresion.csv")
if not os.path.exists(DEFAULT_BASECSV):
    DEFAULT_BASECSV = os.path.join(ROOT, "data", "processed", "base_completa_sin_nulos.csv")
OUTDIR = os.path.join(os.path.dirname(__file__), "sensitivity")
os.makedirs(OUTDIR, exist_ok=True)

DROP_COLS = {"SalePrice", "SalePrice_Present"}


def load_base_row(basecsv: str, pid: int | None, neigh: str | None) -> tuple[pd.Series, pd.DataFrame]:
    df = load_base_df(basecsv)
    if pid is not None and "PID" in df.columns:
        rows = df[df["PID"] == pid]
        if rows.empty:
            raise SystemExit(f"PID {pid} no encontrado en {basecsv}")
        return rows.iloc[0].copy()
    if neigh is not None:
        canon = str(neigh).strip().lower()
        values = df["Neighborhood"].astype(str)
        mask = values.str.lower().str.strip() == canon
        if mask.any():
            return df[mask].iloc[0].copy(), df
        import difflib
        uniq = sorted(values.unique())
        suggestion = difflib.get_close_matches(neigh, uniq, n=1)
        msg = f"Neighborhood {neigh} no encontrado en {basecsv}"
        if suggestion:
            msg += f". ¿Quisiste decir {suggestion[0]}?"
        raise SystemExit(msg)
    return df.iloc[0].copy(), df


def fallback_from_df(col: str, df: pd.DataFrame | None):
    if df is None or col not in df.columns:
        return 0.0
    series = df[col].dropna()
    if series.empty:
        return 0.0
    try:
        return float(series.mode(dropna=True).iloc[0])
    except Exception:
        try:
            return float(series.iloc[0])
        except Exception:
            return 0.0


def build_xgb_features(row: pd.Series, bundle: XGBBundle, base_df: pd.DataFrame | None) -> pd.DataFrame:
    feats = bundle.feature_names_in()
    values = []
    for col in feats:
        if col in row:
            values.append(row[col])
        elif base_df is not None and col in base_df.columns:
            values.append(fallback_from_df(col, base_df))
        else:
            values.append(0.0)
    return pd.DataFrame([values], columns=feats)


def build_reg_features(row: pd.Series, reg_model, ref_df) -> pd.DataFrame:
    feat_names = getattr(reg_model, "feature_names_in_", None)
    feat_list = list(feat_names) if feat_names is not None else None
    return prepare_regression_input(None, row, feat_list, ref_df)


def predict_prices(row: pd.Series, reg_model, bundle: XGBBundle, *, reg_ref_df, base_df) -> Tuple[float, float]:
    feats_reg = build_reg_features(row, reg_model, reg_ref_df)
    price_reg = float(reg_model.predict(feats_reg)[0])
    feats_xgb = build_xgb_features(row, bundle, base_df)
    price_xgb = float(bundle.predict(feats_xgb).iloc[0])
    return price_reg, price_xgb


def map_quality_level(val: float) -> str:
    if val >= 9:
        return "Ex"
    if val >= 7:
        return "Gd"
    if val >= 5:
        return "TA"
    if val >= 3:
        return "Fa"
    return "Po"


def estimate_cost(row: pd.Series, ct: costs.CostTables) -> float:
    def f(name: str) -> float:
        return float(row.get(name, 0.0) or 0.0)

    area_1st = f("1st Flr SF")
    area_2nd = f("2nd Flr SF")
    total_bsmt = f("Total Bsmt SF")
    garage_area = f("Garage Area")
    gr_liv_area = f("Gr Liv Area")
    wood_deck = f("Wood Deck SF")
    open_porch = f("Open Porch SF")
    enclosed_porch = f("Enclosed Porch")
    screen_porch = f("Screen Porch")
    pool_area = f("Pool Area")

    bedrooms = f("Bedroom AbvGr")
    full_bath = f("Full Bath")
    half_bath = f("Half Bath")
    kitchens = f("Kitchen AbvGr") or 1.0

    kitchen_area = kitchens * 300.0
    full_bath_area = full_bath * 40.0
    half_bath_area = half_bath * 25.0
    bedroom_area = bedrooms * 180.0

    cost_val = 0.0
    cost_val += ct.construction_cost * (area_1st + area_2nd)
    cost_val += ct.finish_basement_per_f2 * total_bsmt
    cost_val += ct.kitchen_area_cost * kitchen_area
    cost_val += ct.fullbath_area_cost * full_bath_area
    cost_val += ct.halfbath_area_cost * half_bath_area
    cost_val += ct.bedroom_area_cost * bedroom_area
    cost_val += ct.garage_area_cost * garage_area
    cost_val += ct.pool_area_cost * pool_area
    cost_val += ct.wooddeck_cost * wood_deck
    cost_val += ct.openporch_cost * open_porch
    cost_val += ct.enclosedporch_cost * enclosed_porch
    cost_val += ct.screenporch_cost * screen_porch

    util = row.get("Utilities", "AllPub")
    cost_val += ct.util_cost(util)

    ext_qual = row.get("Exter Qual", None)
    if isinstance(ext_qual, str):
        cost_val += ct.exter_qual_cost(ext_qual.replace(" ", ""))
    ext_cond = row.get("Exter Cond", None)
    if isinstance(ext_cond, str):
        cost_val += ct.exter_cond_cost(ext_cond.replace(" ", ""))

    overall = map_quality_level(f("Overall Qual"))
    cost_val += ct.exter_qual_cost(overall)

    garage_finish = row.get("Garage Finish", "")
    if isinstance(garage_finish, str):
        cost_val += ct.garage_finish_cost(garage_finish)
    garage_qual = row.get("Garage Qual", "")
    if isinstance(garage_qual, str):
        cost_val += ct.garage_qc_costs.get(garage_qual, 0.0)

    return float(cost_val)


def evaluate(row: pd.Series, reg_model, bundle: XGBBundle, ct: costs.CostTables, *, reg_ref_df, base_df) -> Dict[str, float]:
    price_reg, price_xgb = predict_prices(row, reg_model, bundle, reg_ref_df=reg_ref_df, base_df=base_df)
    cost_val = estimate_cost(row, ct)
    util_reg = price_reg - cost_val
    util_xgb = price_xgb - cost_val
    roi_reg = util_reg / cost_val if cost_val else math.nan
    roi_xgb = util_xgb / cost_val if cost_val else math.nan
    return {
        "price_reg": price_reg,
        "price_xgb": price_xgb,
        "cost": cost_val,
        "utility_reg": util_reg,
        "utility_xgb": util_xgb,
        "roi_reg": roi_reg,
        "roi_xgb": roi_xgb,
    }


def apply_overrides(base_row: pd.Series, overrides: Dict[str, float | int | str]) -> pd.Series:
    row = base_row.copy()
    for key, val in overrides.items():
        row[key] = val
    return row


# Mutators for univariate/bivariate
def mutate_area(row: pd.Series, val: float):
    row["Gr Liv Area"] = float(val)
    row["1st Flr SF"] = float(val)
    row["2nd Flr SF"] = 0.0


def mutate_bsmt(row: pd.Series, val: float):
    row["Total Bsmt SF"] = float(val)


def mutate_rooms(col: str) -> Callable[[pd.Series, float], None]:
    def _mut(row: pd.Series, val: float):
        row[col] = float(val)
    return _mut


def mutate_quality(row: pd.Series, val: float):
    row["Overall Qual"] = float(val)


UNIVARIATE_CONFIG = [
    {"name": "Gr Liv Area", "values": [900, 1100, 1300, 1500, 1800, 2100, 2400], "mutator": mutate_area},
    {"name": "Total Bsmt SF", "values": [0, 400, 600, 800, 1000, 1200], "mutator": mutate_bsmt},
    {"name": "Bedroom AbvGr", "values": [2, 3, 4, 5], "mutator": mutate_rooms("Bedroom AbvGr")},
    {"name": "Full Bath", "values": [1, 2, 3], "mutator": mutate_rooms("Full Bath")},
    {"name": "Half Bath", "values": [0, 1, 2], "mutator": mutate_rooms("Half Bath")},
    {"name": "Kitchen AbvGr", "values": [1, 2, 3], "mutator": mutate_rooms("Kitchen AbvGr")},
    {"name": "Overall Qual", "values": [5, 6, 7, 8, 9, 10], "mutator": mutate_quality},
    {"name": "Garage Area", "values": [0, 200, 400, 600, 800, 1000], "mutator": mutate_rooms("Garage Area")},
]

BIVARIATE_CONFIG = [
    ("Gr Liv Area", mutate_area, [1000, 1300, 1600, 1900, 2200],
     "Overall Qual", mutate_quality, [5, 6, 7, 8, 9]),
    ("Gr Liv Area", mutate_area, [1000, 1300, 1600, 1900, 2200],
     "Full Bath", mutate_rooms("Full Bath"), [1, 2, 3]),
    ("Bedroom AbvGr", mutate_rooms("Bedroom AbvGr"), [2, 3, 4, 5],
     "Full Bath", mutate_rooms("Full Bath"), [1, 2, 3]),
]

SCENARIOS = [
    ("economica", {
        "Gr Liv Area": 950, "1st Flr SF": 950, "2nd Flr SF": 0,
        "Total Bsmt SF": 600, "Bedroom AbvGr": 2, "Full Bath": 1,
        "Half Bath": 0, "Kitchen AbvGr": 1, "Overall Qual": 6,
        "Garage Area": 250
    }),
    ("intermedia", {
        "Gr Liv Area": 1500, "1st Flr SF": 1500, "2nd Flr SF": 0,
        "Total Bsmt SF": 900, "Bedroom AbvGr": 3, "Full Bath": 2,
        "Half Bath": 1, "Kitchen AbvGr": 1, "Overall Qual": 8,
        "Garage Area": 400
    }),
    ("premium", {
        "Gr Liv Area": 2200, "1st Flr SF": 2200, "2nd Flr SF": 0,
        "Total Bsmt SF": 1200, "Bedroom AbvGr": 4, "Full Bath": 3,
        "Half Bath": 1, "Kitchen AbvGr": 2, "Overall Qual": 10,
        "Garage Area": 650
    }),
]

TORNADO_ATTRIBUTES = [
    "construction_cost",
    "finish_basement_per_f2",
    "kitchen_area_cost",
    "fullbath_area_cost",
    "bedroom_area_cost",
    "garage_area_cost",
]


def run_univariate(base_row, reg_model, bundle, ct, *, reg_ref_df, base_df):
    rows = []
    for cfg in UNIVARIATE_CONFIG:
        for val in cfg["values"]:
            row = base_row.copy()
            cfg["mutator"](row, val)
            metrics = evaluate(row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
            rows.append({
                "feature": cfg["name"],
                "value": val,
                **metrics,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "univariate.csv"), index=False)


def run_bivariate(base_row, reg_model, bundle, ct, *, reg_ref_df, base_df):
    for cfg in BIVARIATE_CONFIG:
        name_x, mut_x, vals_x, name_y, mut_y, vals_y = cfg
        rows = []
        for vx in vals_x:
            for vy in vals_y:
                row = base_row.copy()
                mut_x(row, vx)
                mut_y(row, vy)
                metrics = evaluate(row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
                rows.append({
                    "feature_x": name_x,
                    "value_x": vx,
                    "feature_y": name_y,
                    "value_y": vy,
                    **metrics,
                })
        fname = f"bivariate_{name_x.replace(' ', '_')}__{name_y.replace(' ', '_')}.csv"
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, fname), index=False)


def run_scenarios(base_row, reg_model, bundle, ct, *, reg_ref_df, base_df):
    rows = []
    for name, overrides in SCENARIOS:
        row = apply_overrides(base_row, overrides)
        metrics = evaluate(row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
        rows.append({
            "scenario": name,
            **overrides,
            **metrics,
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "scenarios.csv"), index=False)


def run_tornado(base_row, reg_model, bundle, ct, scenario_name: str, *, reg_ref_df, base_df):
    scenario = dict(SCENARIOS[-1][1])
    for name, overrides in SCENARIOS:
        if name == scenario_name:
            scenario = overrides
            break
    row = apply_overrides(base_row, scenario)
    base_metrics = evaluate(row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
    entries = []
    for attr in TORNADO_ATTRIBUTES:
        base_value = getattr(ct, attr)
        for factor in (0.8, 0.9, 1.1, 1.2):
            new_ct = copy.deepcopy(ct)
            setattr(new_ct, attr, base_value * factor)
            metrics = evaluate(row, reg_model, bundle, new_ct, reg_ref_df=reg_ref_df, base_df=base_df)
            entries.append({
                "cost_component": attr,
                "multiplier": factor,
                "delta_utility_xgb": metrics["utility_xgb"] - base_metrics["utility_xgb"],
                "delta_utility_reg": metrics["utility_reg"] - base_metrics["utility_reg"],
                **metrics,
            })
    pd.DataFrame(entries).to_csv(os.path.join(OUTDIR, "tornado.csv"), index=False)


def save_base_profile(row: pd.Series):
    path = os.path.join(OUTDIR, "base_profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row.to_dict(), f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Genera sensibilidades de construcción.")
    ap.add_argument("--reg-model", required=True, help="Ruta al modelo de regresión base (joblib)")
    ap.add_argument("--basecsv", default=DEFAULT_BASECSV, help="CSV base para construir perfiles")
    ap.add_argument("--base-pid", type=int, default=None, help="PID base (opcional)")
    ap.add_argument("--base-neigh", type=str, default=None, help="Barrio base si no hay PID")
    ap.add_argument("--base-lot", type=float, default=None, help="LotArea opcional para ajustar el perfil base")
    ap.add_argument("--xgbdir", type=str, default=None, help="Directorio alternativo que tenga model_xgb.joblib")
    ap.add_argument("--tornado-scenario", type=str, default="premium", help="Escenario a usar en el tornado")
    args = ap.parse_args()

    if args.xgbdir:
        from pathlib import Path
        CONST_PATHS.model_dir = Path(args.xgbdir)
        CONST_PATHS.xgb_model_file = Path(args.xgbdir) / "model_xgb.joblib"

    base_row, base_df = load_base_row(args.basecsv, args.base_pid, args.base_neigh)
    if args.base_lot is not None:
        base_row["LotArea"] = float(args.base_lot)
        if "Lot Frontage" in base_row:
            base_row["Lot Frontage"] = float(args.base_lot) ** 0.5
    save_base_profile(base_row)

    reg_model = joblib.load(args.reg_model)
    bundle = XGBBundle()
    ct = costs.CostTables()
    reg_ref_df = load_regression_reference_df()

    run_univariate(base_row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
    run_bivariate(base_row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
    run_scenarios(base_row, reg_model, bundle, ct, reg_ref_df=reg_ref_df, base_df=base_df)
    run_tornado(base_row, reg_model, bundle, ct, args.tornado_scenario, reg_ref_df=reg_ref_df, base_df=base_df)
    print(f"Sensibilidades guardadas en {OUTDIR}")


if __name__ == "__main__":
    main()
