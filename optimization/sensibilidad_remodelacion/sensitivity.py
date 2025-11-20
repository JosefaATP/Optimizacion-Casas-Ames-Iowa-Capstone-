"""
Análisis de sensibilidad para el modelo de remodelación.

Genera combinaciones (barrio x percentil de precio x presupuesto x piso de calidad)
ejecutando la optimización y guardando resultados en `optimization/sensibilidad_remodelacion`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# Asegura que el repo esté en sys.path cuando se ejecuta como script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
import gurobipy as gp

from optimization.remodel.costs import CostTables
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.io import get_base_house, load_base_df
from optimization.remodel.xgb_predictor import (
    XGBBundle,
    QUALITY_CANDIDATE_NAMES,
    UTIL_ORDER,
    UTIL_TO_ORD,
    ROOF_STYLE_TO_ORD,
    ROOF_MATL_TO_ORD,
    _coerce_quality_ordinals_inplace,
    _coerce_utilities_ordinal_inplace,
)
from optimization.remodel.run_opt import _row_with_dummies


# Categóricos definidos en remodel/features.py (copiados aquí para no depender de internals)
EXT_MATS = [
    "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc",
    "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShngl",
]
BSMT_TYPES = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "No aplica"]


ORD_TO_QUAL = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex", -1: "No aplica"}
QUAL_TO_ORD = {v: k for k, v in ORD_TO_QUAL.items()}
ORD_TO_UTIL = {v: k for k, v in UTIL_TO_ORD.items()}
ORD_TO_ROOF_STYLE = {v: k for k, v in ROOF_STYLE_TO_ORD.items()}
ORD_TO_ROOF_MATL = {v: k for k, v in ROOF_MATL_TO_ORD.items()}


def _ord_from_label(label: str | int | float | None, default: int = 2) -> int:
    try:
        val = int(pd.to_numeric(label, errors="coerce"))
        if val in (-1, 0, 1, 2, 3, 4):
            return val
    except Exception:
        pass
    return QUAL_TO_ORD.get(str(label).strip(), default)


def _pick_onehot(m: gp.Model, prefix: str, categories: Iterable[str], fallback: str | None) -> str | None:
    """Devuelve la categoría activa según vars x_{prefix}{cat} > 0.5."""
    for cat in categories:
        v = m.getVarByName(f"x_{prefix}{cat}")
        if v is not None and v.X > 0.5:
            return cat
    return fallback


def _apply_quality_floor(m: gp.Model, floor_ord: int, cols: Iterable[str]) -> None:
    for col in cols:
        v = m.getVarByName(f"x_{col}")
        if v is not None:
            m.addConstr(v >= floor_ord, name=f"floor_{col}_{floor_ord}")


def _json_default(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _pd.Timestamp):
        return o.isoformat()
    return str(o)


def _build_opt_row(base_row: pd.Series, m: gp.Model) -> pd.Series:
    """Reconstruye los valores 'raw' (antes de OHE) de la casa óptima."""
    opt_row = base_row.copy()

    # numéricas/ordinales directas
    direct_cols = [
        "Bedroom AbvGr", "Full Bath", "Garage Cars",
        "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
        "Gr Liv Area", "1st Flr SF", "Low Qual Fin SF",
        "Half Bath", "Kitchen AbvGr",
        "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
        "3Ssn Porch", "Screen Porch", "Pool Area",
        "Garage Area",
        "Kitchen Qual", "Exter Qual", "Exter Cond", "Heating QC",
        "Fireplace Qu", "Bsmt Qual", "Bsmt Cond", "Garage Qual",
        "Garage Cond", "Pool QC",
        "Utilities",
    ]
    for col in direct_cols:
        v = m.getVarByName(f"x_{col}")
        if v is not None and col in opt_row.index:
            opt_row[col] = v.X

    # Utilidades (ordinal -> etiqueta)
    if "Utilities" in opt_row.index:
        u_var = m.getVarByName("x_Utilities")
        if u_var is not None:
            opt_row["Utilities"] = int(round(u_var.X))

    # Roof ordinal -> string
    rs_var = m.getVarByName("x_Roof Style")
    if rs_var is not None and "Roof Style" in opt_row.index:
        opt_row["Roof Style"] = ORD_TO_ROOF_STYLE.get(int(round(rs_var.X)), opt_row["Roof Style"])

    rm_var = m.getVarByName("x_Roof Matl")
    if rm_var is not None and "Roof Matl" in opt_row.index:
        opt_row["Roof Matl"] = ORD_TO_ROOF_MATL.get(int(round(rm_var.X)), opt_row["Roof Matl"])

    # One-hot categóricas controladas por binarios
    opt_row["Mas Vnr Type"] = _pick_onehot(m, "mvt_is_", ["BrkCmn", "BrkFace", "CBlock", "Stone", "No aplica"], opt_row.get("Mas Vnr Type"))
    opt_row["Electrical"] = _pick_onehot(m, "elect_is_", ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"], opt_row.get("Electrical"))
    opt_row["Garage Finish"] = _pick_onehot(m, "garage_finish_is_", ["Fin", "RFn", "Unf", "No aplica"], opt_row.get("Garage Finish"))
    opt_row["Paved Drive"] = _pick_onehot(m, "paved_drive_is_", ["Y", "P", "N"], opt_row.get("Paved Drive"))
    opt_row["Fence"] = _pick_onehot(m, "fence_is_", ["GdPrv", "MnPrv", "GdWo", "MnWw", "No aplica"], opt_row.get("Fence"))

    opt_row["Exterior 1st"] = _pick_onehot(m, "ex1_is_", EXT_MATS, opt_row.get("Exterior 1st"))
    opt_row["Exterior 2nd"] = _pick_onehot(m, "ex2_is_", EXT_MATS, opt_row.get("Exterior 2nd"))

    opt_row["Heating"] = _pick_onehot(m, "heat_is_", ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"], opt_row.get("Heating"))
    opt_row["BsmtFin Type 1"] = _pick_onehot(m, "b1_is_", BSMT_TYPES, opt_row.get("BsmtFin Type 1"))
    opt_row["BsmtFin Type 2"] = _pick_onehot(m, "b2_is_", BSMT_TYPES, opt_row.get("BsmtFin Type 2"))
    opt_row["Pool QC"] = _pick_onehot(m, "poolqc_is_", ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"], opt_row.get("Pool QC"))

    return opt_row


@dataclass
class CaseResult:
    pid: int
    neighborhood: str
    percentile_label: str
    percentile_value: float
    budget: float
    quality_floor: str | None
    status: str
    base_price: float
    opt_price: float | None
    cost: float | None
    net_gain: float | None  # (opt_price - cost) - base_price
    slack: float | None
    mip_gap: float | None
    obj_val: float | None
    changes: list[dict]
    xgb_features_path: str
    model_path: str


def _summarize_changes(base_row: pd.Series, opt_row: pd.Series) -> list[dict]:
    changes = []
    for col in opt_row.index:
        if col not in base_row.index:
            continue
        b = base_row[col]
        n = opt_row[col]
        try:
            bnum = float(pd.to_numeric(b, errors="coerce"))
            nnum = float(pd.to_numeric(n, errors="coerce"))
            if pd.isna(bnum) and pd.isna(nnum):
                continue
            if abs(bnum - nnum) < 1e-6:
                continue
        except Exception:
            if str(b) == str(n):
                continue
        changes.append({"col": col, "base": b, "new": n})
    return changes


def run_case(
    pid: int,
    budget: float,
    bundle: XGBBundle,
    quality_floor: str | None,
    base_csv: Optional[Path] = None,
    out_dir: Path = Path("optimization/sensibilidad_remodelacion"),
) -> tuple[CaseResult, dict]:
    base = get_base_house(pid, base_csv)
    ct = CostTables()

    feat_order = bundle.feature_names_in()
    X_base = pd.DataFrame([_row_with_dummies(base.row, feat_order)], columns=feat_order)
    _coerce_quality_ordinals_inplace(X_base, bundle.quality_cols)
    _coerce_utilities_ordinal_inplace(X_base)
    precio_base = float(bundle.predict(X_base).iloc[0])

    m = build_mip_embed(base.row, budget, ct, bundle, base_price=precio_base)

    floor_ord = None
    if quality_floor is not None:
        floor_ord = _ord_from_label(quality_floor, default=2)
        _apply_quality_floor(m, floor_ord, QUALITY_CANDIDATE_NAMES)

    # Silenciar solver para grandes barridos
    try:
        m.Params.OutputFlag = 0
    except Exception:
        pass

    m.optimize()

    status_map = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    status_code = getattr(m, "Status", None)
    status = status_map.get(status_code, str(status_code))

    has_solution = bool(getattr(m, "SolCount", 0) and getattr(m, "SolCount") > 0)

    cost_var = m.getVarByName("cost_model")
    y_price_var = m.getVarByName("y_price")
    obj_val = m.objVal if has_solution and m is not None and hasattr(m, "objVal") else None

    opt_price = float(y_price_var.X) if has_solution and y_price_var is not None else None
    cost = float(cost_var.X) if has_solution and cost_var is not None else None
    slack = None
    if cost is not None and hasattr(m, "_budget_usd"):
        slack = float(m._budget_usd) - cost
    mip_gap = float(getattr(m, "MIPGap", np.nan)) if getattr(m, "MIPGap", None) is not None else None

    opt_row = _build_opt_row(base.row, m) if has_solution and opt_price is not None else base.row.copy()
    opt_feat = pd.DataFrame([_row_with_dummies(opt_row, feat_order)], columns=feat_order)
    _coerce_quality_ordinals_inplace(opt_feat, bundle.quality_cols)
    _coerce_utilities_ordinal_inplace(opt_feat)

    # recalcular y_price por pipeline solo como sanity
    opt_price_recalc = None
    try:
        opt_price_recalc = float(bundle.predict(opt_feat).iloc[0])
        if opt_price is None and has_solution:
            opt_price = opt_price_recalc
    except Exception:
        opt_price_recalc = None

    net_gain = None
    if opt_price is not None and cost is not None:
        net_gain = (opt_price - cost) - precio_base

    changes = _summarize_changes(base.row, opt_row) if has_solution else []

    # Guardar features listos para XGB
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"xgb_features_pid{pid}_b{int(budget)}{'_q'+str(floor_ord) if floor_ord is not None else ''}.csv"
    opt_feat.to_csv(feat_path, index=False)

    result = CaseResult(
        pid=pid,
        neighborhood=str(base.row.get("Neighborhood", "")),
        percentile_label="",
        percentile_value=float(base.row.get("SalePrice_Present", np.nan)),
        budget=budget,
        quality_floor=quality_floor,
        status=status,
        base_price=precio_base,
        opt_price=opt_price,
        cost=cost,
        net_gain=net_gain,
        slack=slack,
        mip_gap=mip_gap,
        obj_val=obj_val,
        changes=changes,
        xgb_features_path=str(feat_path),
        model_path=str(bundle.model_path),
    )

    extra = {
        "opt_price_recalc": opt_price_recalc,
        "opt_row": opt_row.to_dict(),
        "base_row": base.row.to_dict(),
    }
    return result, extra


def _select_representative_pids(df: pd.DataFrame, neighborhood: str, percentiles: Iterable[float]) -> list[dict]:
    sub = df.loc[df["Neighborhood"] == neighborhood].copy()
    if sub.empty:
        return []
    price_col = "SalePrice_Present"
    percentile_list = list(percentiles)
    qs = sub[price_col].quantile(percentile_list)
    labels = [f"p{int(p*100)}" for p in percentile_list]
    picks = []
    for label, qv in zip(labels, qs.values):
        sub["dist"] = (sub[price_col] - qv).abs()
        row = sub.sort_values("dist").iloc[0]
        picks.append({"label": label, "pid": int(row["PID"]), "price_val": float(row[price_col])})
    return picks


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Análisis de sensibilidad remodelación")
    ap.add_argument("--neighborhood", type=str, default="CollgCr",
                    help="Barrio específico o 'all' para todos con n>=10")
    ap.add_argument("--basecsv", type=str, default=None, help="CSV base (opcional)")
    ap.add_argument("--budgets", type=float, nargs="+", default=[20000, 50000, 100000])
    ap.add_argument("--percentiles", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--outdir", type=str, default="optimization/sensibilidad_remodelacion",
                    help="Carpeta de salida (resumen.csv, detalles.jsonl, features)")
    ap.add_argument("--model-path", type=str, default=None,
                    help="Ruta al model_xgb.joblib a usar (por defecto PATHS.xgb_model_file)")
    ap.add_argument("--max-base-bedrooms", type=float, default=None,
                    help="Si se define, solo usa pids con Bedroom AbvGr base <= este valor.")
    ap.add_argument("--min-neighborhood-count", type=int, default=10,
                    help="Para --neighborhood all, mínimo de casas necesarias por barrio (default 10).")
    return ap.parse_args()


def main():
    args = parse_args()

    df = load_base_df(args.basecsv)
    if args.max_base_bedrooms is not None and "Bedroom AbvGr" in df.columns:
        df = df.loc[pd.to_numeric(df["Bedroom AbvGr"], errors="coerce") <= args.max_base_bedrooms]
    nb_counts = df["Neighborhood"].value_counts()
    neighborhoods = []
    if args.neighborhood.lower() == "all":
        neighborhoods = list(nb_counts[nb_counts >= args.min_neighborhood_count].index)
    else:
        neighborhoods = [args.neighborhood]

    quality_levels = [None]  # calidades fijas deshabilitadas

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[CaseResult] = []
    details_path = out_dir / "detalles.jsonl"
    if details_path.exists():
        details_path.unlink()

    bundle = XGBBundle(Path(args.model_path)) if args.model_path else XGBBundle()

    # contador de progreso
    total_cases = 0
    for nb in neighborhoods:
        picks = _select_representative_pids(df, nb, args.percentiles)
        total_cases += len(picks) * len(args.budgets) * len(quality_levels)
    processed = 0

    for nb in neighborhoods:
        print(f"== Barrio {nb} ==")
        picks = _select_representative_pids(df, nb, args.percentiles)
        for pick in picks:
            for budget in args.budgets:
                for q_floor in quality_levels:
                    res, extra = run_case(
                        pick["pid"],
                        budget,
                        bundle,
                        q_floor,
                        args.basecsv,
                        out_dir,
                    )
                    res.percentile_label = pick["label"]
                    res.percentile_value = pick["price_val"]
                    summary_rows.append(res)
                    with details_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "meta": asdict(res),
                            "extra": extra,
                        }, default=_json_default) + "\n")
                    processed += 1
                    print(f"[{processed}/{total_cases}] PID {res.pid} ({res.percentile_label}) budget {budget} -> {res.status}, Δ=${(res.net_gain or 0):,.0f}")

    # guardar resumen CSV
    summary_df = pd.DataFrame([asdict(r) for r in summary_rows])
    summary_csv = out_dir / "resumen.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nListo. Resumen en {summary_csv}")
    print(f"Detalles JSONL en {details_path}")


if __name__ == "__main__":
    main()
