from typing import Dict, Any
import pandas as pd
import gurobipy as gp
import numpy as np

from sklearn.compose import ColumnTransformer

# >>> SHIMS antes de gurobi_ml
from .compat_sklearn import ensure_check_feature_names
ensure_check_feature_names()
from .compat_xgboost import patch_get_booster
patch_get_booster()

try:
    from gurobi_ml.sklearn.pipeline import add_pipeline_constr as _add_sklearn
except Exception:
    from gurobi_ml.sklearn import add_predictor_constr as _add_sklearn

from .features import MODIFIABLE, IMMUTABLE
from .costs import CostTables
from .xgb_predictor import XGBBundle

def _vtype(code: str):
    return gp.GRB.CONTINUOUS if code == "C" else (gp.GRB.BINARY if code == "B" else gp.GRB.INTEGER)

def build_mip_embed(base_row: pd.Series, budget: float, ct: CostTables, bundle: XGBBundle) -> gp.Model:
    m = gp.Model("remodel_embed")

    # -------------------
    # 1) Variables decisión
    # -------------------
    x: dict[str, gp.Var] = {}
    for f in MODIFIABLE:
        x[f.name] = m.addVar(lb=f.lb, ub=f.ub, vtype=_vtype(f.vartype), name=f"x_{f.name}")

    # -------------------
    # 2) Input al pipeline (orden correcto)
    # -------------------
    feature_order = bundle.feature_names_in()
    modifiable_names = {f.name for f in MODIFIABLE}
    row_vals: Dict[str, Any] = {}
    for fname in feature_order:
        if fname in modifiable_names:
            row_vals[fname] = x[fname]
        else:
            row_vals[fname] = base_row[fname]

    X_input = pd.DataFrame([row_vals], columns=feature_order)

    # forzar str en columnas que irán a OHE (para evitar isnan sobre object mixto)
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline as SKPipeline
    def _align_ohe_dtypes(X: pd.DataFrame, pre: ColumnTransformer) -> pd.DataFrame:
        trs = pre.transformers_ if hasattr(pre, "transformers_") else pre.transformers
        X2 = X.copy()
        for item in trs:
            name, transformer, cols = item[0], item[1], item[2]
            est = transformer.steps[-1][1] if isinstance(transformer, SKPipeline) else transformer
            if isinstance(est, OneHotEncoder):
                for c in cols:
                    if c in X2.columns:
                        X2[c] = X2[c].astype(str)
        return X2
    X_input = _align_ohe_dtypes(X_input, bundle.pre)

    # -------------------
    # 3) Restricción de Kitchen Qual (paquetes TA/EX) — como ya lo tenías
    # -------------------
    def _q_to_ord(v):
        mapping = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return mapping[str(v)]
    kq_base = _q_to_ord(base_row.get("Kitchen Qual", "TA"))
    dTA = x["delta_KitchenQual_TA"]
    dEX = x["delta_KitchenQual_EX"]
    m.addConstr(dTA + dEX <= 1, name="R9_kitchen_at_most_one_pkg")
    if kq_base >= 2: dTA.UB = 0
    if kq_base >= 4: dEX.UB = 0
    q_TA = max(kq_base, 2)
    q_EX = max(kq_base, 4)
    q_new = x["Kitchen Qual"]
    m.addConstr(q_new == kq_base + (q_TA - kq_base) * dTA + (q_EX - kq_base) * dEX,
                name="R9_kitchen_upgrade_link")

    # -------------------
    # 4) Predictor (pre -> XGB), salida en log
    # -------------------
    y_log = m.addVar(lb=-gp.GRB.INFINITY, name="y_log")
    _add_sklearn(m, bundle.pipe_for_gurobi(), X_input, [y_log])

    # convertir log→precio
    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        z_min, z_max = 10.0, 14.0
        grid = np.linspace(z_min, z_max, 81)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    # -------------------
    # 5) Costos (lineales)
    # -------------------
    def _num_base(name: str) -> float:
        try:
            return float(pd.to_numeric(base_row.get(name), errors="coerce"))
        except Exception:
            return 0.0

    base_vals = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
        "Wood Deck SF": _num_base("Wood Deck SF"),
        "Garage Cars": _num_base("Garage Cars"),
        "Total Bsmt SF": _num_base("Total Bsmt SF"),
    }

    lin_cost = gp.LinExpr(ct.project_fixed)

    def pos(expr):
        v = m.addVar(lb=0.0, name=f"pos_{len(m.getVars())}")
        m.addConstr(v >= expr)
        return v

    if "Bedroom AbvGr" in x:
        lin_cost += pos(x["Bedroom AbvGr"] - base_vals["Bedroom AbvGr"]) * ct.add_bedroom
    if "Full Bath" in x:
        lin_cost += pos(x["Full Bath"] - base_vals["Full Bath"]) * ct.add_bathroom
    if "Wood Deck SF" in x:
        lin_cost += pos(x["Wood Deck SF"] - base_vals["Wood Deck SF"]) * ct.deck_per_m2
    if "Garage Cars" in x:
        lin_cost += pos(x["Garage Cars"] - base_vals["Garage Cars"]) * ct.garage_per_car
    if "Total Bsmt SF" in x:
        lin_cost += pos(x["Total Bsmt SF"] - base_vals["Total Bsmt SF"]) * ct.finish_basement_per_f2

    # costos de paquetes Kitchen
    lin_cost += dTA * ct.kitchenQual_upgrade_TA
    lin_cost += dEX * ct.kitchenQual_upgrade_EX

    # -------------------
    # 6) Utilities (upgrade-only + costo si cambia + link al entero)
    # -------------------
    if "Utilities" in x:
        # nombres y ordinales consistentes con el entrenamiento
        util_names = {0: "ELO", 1: "NoSeWa", 2: "NoSewr", 3: "AllPub"}
        util_to_ord = {"ELO":0, "NoSeWa":1, "NoSewr":2, "AllPub":3}

        # entero de decisión (entra al predictor)
        u_new = x["Utilities"]

        # base como nombre y ordinal
        u_base_name = str(base_row.get("Utilities"))
        try:
            u_base_ord = int(pd.to_numeric(base_row.get("Utilities"), errors="coerce"))
            if u_base_ord not in (0,1,2,3):
                u_base_ord = util_to_ord.get(u_base_name, 0)
        except Exception:
            u_base_ord = util_to_ord.get(u_base_name, 0)

        # one-hot internos (exactamente uno)
        u_bin = {
            0: m.addVar(vtype=gp.GRB.BINARY, name="util_ELO"),
            1: m.addVar(vtype=gp.GRB.BINARY, name="util_NoSeWa"),
            2: m.addVar(vtype=gp.GRB.BINARY, name="util_NoSewr"),
            3: m.addVar(vtype=gp.GRB.BINARY, name="util_AllPub"),
        }
        m.addConstr(gp.quicksum(u_bin.values()) == 1, name="UTIL_one_hot")
        m.addConstr(u_new == gp.quicksum(k * u_bin[k] for k in u_bin), name="UTIL_link")

        # upgrade only
        m.addConstr(u_new >= u_base_ord, name="UTIL_upgrade_only")

        # costo: 0 si te quedas, costo categoría si cambias
        for k in u_bin:
            name_k = util_names[k]
            if k == u_base_ord:
                continue
            lin_cost += ct.util_cost(name_k) * u_bin[k]

    # -------------------
    # 7) Presupuesto y objetivo
    # -------------------
    total_cost = lin_cost
    m.addConstr(total_cost <= budget, name="budget")
    m.setObjective(y_price - total_cost, gp.GRB.MAXIMIZE)

    # -------------------
    # 8) Resto de restricciones (R1..R8) — igual que las que ya tenías
    # -------------------
    # (R1) 1stFlrSF ≥ 2ndFlrSF
    if "1st Flr SF" in x and "2nd Flr SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["2nd Flr SF"], name="R1_floor1_ge_floor2")

    # (R2) GrLivArea ≤ LotArea
    if "Gr Liv Area" in x and "Lot Area" in base_row:
        m.addConstr(x["Gr Liv Area"] <= float(base_row["Lot Area"]), name="R2_grliv_le_lot")

    # (R3) 1stFlrSF ≥ TotalBsmtSF
    if "1st Flr SF" in x and "Total Bsmt SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["Total Bsmt SF"], name="R3_floor1_ge_bsmt")

    # (R4) FullBath + HalfBath ≤ Bedroom
    def _val_or_var(col):
        return x[col] if col in x else float(base_row[col])
    need = all(c in base_row for c in ["Full Bath", "Bedroom AbvGr"])
    if need and ("Half Bath" in base_row or "Half Bath" in x):
        fullb = _val_or_var("Full Bath")
        halfb = _val_or_var("Half Bath") if ("Half Bath" in x or "Half Bath" in base_row) else 0.0
        beds  = _val_or_var("Bedroom AbvGr")
        m.addConstr(fullb + halfb <= beds, name="R4_baths_le_bedrooms")

    # (R5) mínimos
    if ("Full Bath" in x) or ("Full Bath" in base_row):
        m.addConstr(_val_or_var("Full Bath") >= 1, name="R5_min_fullbath")
    if ("Bedroom AbvGr" in x) or ("Bedroom AbvGr" in base_row):
        m.addConstr(_val_or_var("Bedroom AbvGr") >= 1, name="R5_min_bedrooms")
    if ("Kitchen AbvGr" in x) or ("Kitchen AbvGr" in base_row):
        m.addConstr(_val_or_var("Kitchen AbvGr") >= 1, name="R5_min_kitchen")

    # (R7) Gr Liv Area = 1st + 2nd (+ LowQual si existe)
    lowqual_names = ["Low Qual Fin SF", "LowQualFinSF"]
    lowqual_col = next((c for c in lowqual_names if c in X_input.columns), None)
    cols_needed = ["Gr Liv Area", "1st Flr SF", "2nd Flr SF"]
    if all(c in X_input.columns for c in cols_needed):
        def _v(name: str):
            return x[name] if name in x else float(base_row[name])
        lhs = _v("Gr Liv Area")
        rhs = _v("1st Flr SF") + _v("2nd Flr SF")
        if lowqual_col is not None:
            rhs += _v(lowqual_col)
        m.addConstr(lhs == rhs, name="R7_gr_liv_equality")

    # (R8) TotRms AbvGrd = Bedroom + Kitchen + Other
    r8_ok = all(c in X_input.columns for c in ["TotRms AbvGrd", "Bedroom AbvGr", "Kitchen AbvGr"])
    if r8_ok:
        other = m.addVar(lb=0, ub=15, vtype=gp.GRB.INTEGER, name="R8_other_rooms")
        m.addConstr(_val_or_var("TotRms AbvGrd") == _val_or_var("Bedroom AbvGr") + _val_or_var("Kitchen AbvGr") + other,
                    name="R8_rooms_balance")

    return m
