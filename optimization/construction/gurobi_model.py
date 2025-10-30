# -*- coding: utf-8 -*-
from __future__ import annotations
import math, re
from typing import Dict, Iterable, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from .xgb_predictor import XGBBundle
from .summary_and_costs_hooks import build_cost_expr, summarize_solution  # re-export summarize_solution

# ============================== helpers basicos ===================================

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _find_x_key(x: Dict[str, gp.Var], *tokens: str) -> str | None:
    want = [_norm(t) for t in tokens]
    for k in x.keys():
        nk = _norm(k)
        if all(t in nk for t in want):
            return k
    return None


def _link_if_exists(m: gp.Model, x: Dict[str, gp.Var], key: str, v: gp.Var):
    if key in x:
        m.addConstr(x[key] == v, name=f"link__{_norm(key)}")


def _safe_get(x: Dict[str, gp.Var], name: str) -> gp.Var | None:
    return x.get(name) or x.get(_find_x_key(x, name))


def _one_hot(m: gp.Model, name: str, options: Iterable[str]) -> Dict[str, gp.Var]:
    B: Dict[str, gp.Var] = {opt: m.addVar(vtype=GRB.BINARY, name=f"{name}__{opt}") for opt in options}
    m.addConstr(gp.quicksum(B.values()) == 1, name=f"ex__{name}")
    return B


def _tie_one_hot_to_x(m: gp.Model, x: Dict[str, gp.Var], feat_name: str, B: Dict[str, gp.Var]):
    for opt, b in B.items():
        key = _find_x_key(x, feat_name, opt)
        if key is not None:
            m.addConstr(x[key] == b, name=f"link__{_norm(feat_name)}__{_norm(opt)}")


# ===================== inputs en el orden del XGB (features) ======================

def _build_x_inputs(m: gp.Model, bundle: XGBBundle, X_base_row) -> tuple[List[str], Dict[str, gp.Var]]:
    feat_order: List[str] = list(bundle.feature_names_in())
    x_vars: Dict[str, gp.Var] = {}

    INT_NAMES = {"Full Bath","Half Bath","Bedroom AbvGr","Kitchen AbvGr","Garage Cars","Fireplaces"}
    NONNEG_LB0 = set(INT_NAMES) | {
        "1st Flr SF","2nd Flr SF","Gr Liv Area","Total Bsmt SF","Garage Area",
        "Wood Deck SF","Open Porch SF","Enclosed Porch","3Ssn Porch","Screen Porch",
        "Pool Area","Lot Area","Mas Vnr Area",
    }
    ORD_CANDIDATES = {"Roof Style", "Roof Matl", "Utilities"}

    for col in feat_order:
        val = X_base_row[col] if col in X_base_row.index else None
        lb, ub, vtype = 0.0, GRB.INFINITY, GRB.CONTINUOUS
        if col in INT_NAMES:
            vtype = GRB.INTEGER
        if col in NONNEG_LB0:
            lb = 0.0
        if col in ORD_CANDIDATES:
            vtype, lb, ub = GRB.INTEGER, -1, 8  # por si el pipe usa -1 para NA

        # dummies OHE: forzar [0,1]
        if "_" in col and col not in NONNEG_LB0:
            lb, ub = 0.0, 1.0

        if col == "Lot Area" and val is not None and not math.isnan(val):
            v = m.addVar(lb=float(val), ub=float(val), name=f"x_const__{col}")
        else:
            v = m.addVar(lb=lb, ub=ub, vtype=vtype, name=f"x_{col}")
        x_vars[col] = v
    

    m.update()
    return feat_order, x_vars


# ========================= embed del XGB (y_log, y_price) =========================

def _attach_xgb_embed(m: gp.Model, bundle: XGBBundle,
                      feat_order: List[str], x: Dict[str, gp.Var]) -> tuple[gp.Var, gp.Var]:
    # y_log que usa el XGB
    y_log = m.addVar(lb=-1e6, ub=1e6, name="y_log")

    # IMPORTANT: pasar la lista de variables en el MISMO orden que espera el XGB
    x_list_vars: List[gp.Var] = []
    for col in feat_order:
        if col not in x:
            # si por alguna razon falta, crea dummy continua 0
            x_list_vars.append(m.addVar(lb=0.0, ub=0.0, name=f"x_missing__{col}"))
        else:
            x_list_vars.append(x[col])

    # embebe arboles (la implementacion en xgb_predictor evita 1 - z con binaria complemento)
    bundle.attach_to_gurobi(m, x_list_vars, y_log)

    # map log -> price via PWL exp (si tu target fue log, usa exp; si fue log1p, usa expm1)
    y_price = m.addVar(lb=0.0, name="y_price")

    import numpy as np
    # rango razonable de log-precio, ajusta si quieres: p in [30k, 1.0M] -> log p en [10.3, 13.8]
    xs = np.linspace(10.3, 13.8, 40)
    ys = np.exp(xs)  # si usaste log1p, cambia por: np.expm1(xs)

    m.addGenConstrPWL(y_log, y_price, xs.tolist(), ys.tolist(), name="PWL_exp")

    return y_log, y_price


# =============================== costos (LinExpr) =================================

def _build_cost_expr(m: gp.Model, x: dict[str, gp.Var], ct) -> gp.LinExpr:
    cost = gp.LinExpr(0.0)

    # construccion principal por area
    c_base = getattr(ct, "construction_cost", 0.0)
    for nm in ("1st Flr SF","2nd Flr SF","Total Bsmt SF"):
        v = _safe_get(x, nm)
        if v is not None and c_base:
            cost += float(c_base) * v

    # terminaciones de sotano
    c_bsmt_fin = getattr(ct, "basement_finish_cost", 0.0)
    for nm in ("BsmtFin SF 1","BsmtFin SF 2"):
        v = _safe_get(x, nm)
        if v is not None and c_bsmt_fin:
            cost += float(c_bsmt_fin) * v

    def term_if(name: str, coef_attr: str):
        v = _safe_get(x, name)
        unit = getattr(ct, coef_attr, 0.0)
        return (float(unit) * v) if (v is not None and unit) else 0.0

    # porches, deck, piscina
    cost += term_if("Wood Deck SF",  "wooddeck_cost")
    cost += term_if("Open Porch SF", "openporch_cost")
    cost += term_if("Enclosed Porch","enclosedporch_cost")
    cost += term_if("3Ssn Porch",    "threessnporch_cost")
    cost += term_if("Screen Porch",  "screenporch_cost")
    cost += term_if("Pool Area",     "pool_area_cost")

    # garage por area (acepta dos posibles nombres en ct)
    xgar = _safe_get(x, "Garage Area")
    if xgar is not None:
        unit_gar = getattr(ct, "garage_area_cost", getattr(ct, "garage_cost_per_sf", 0.0))
        if unit_gar:
            cost += float(unit_gar) * xgar

    # fundacion por SF si ct lo define
    if hasattr(ct, "foundation_cost_per_sf"):
        for tag, unit in getattr(ct, "foundation_cost_per_sf").items():
            v = m.getVarByName(f"FA__{tag}")
            if v is not None:
                cost += float(unit) * v

    # techo por material * gamma(style, mat)
    if hasattr(ct, "roof_cost_by_material"):
        gamma = getattr(ct, "gamma", {})
        for tag, unit in getattr(ct, "roof_cost_by_material").items():
            for s in ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]:
                z = m.getVarByName(f"Z__{s}__{tag}")
                if z is not None:
                    g = float(gamma.get((s, tag), 1.10))
                    cost += float(unit) * g * z

    # areas de recintos
    for nm, attr in (("AreaKitchen","kitchen_area_cost"),
                     ("AreaFullBath","fullbath_area_cost"),
                     ("AreaHalfBath","halfbath_area_cost"),
                     ("AreaBedroom","bedroom_area_cost")):
        v = m.getVarByName(nm)
        if v is not None:
            cost += float(getattr(ct, attr, 0.0)) * v

    # reja perimetral (opcional)
    has_reja = m.getVarByName("HasReja")
    if has_reja is not None:
        cost += float(getattr(ct, "fence_cost_per_ft", 0.0)) * float(getattr(ct, "lot_frontage_ft", 60.0)) * has_reja

    return cost


# =============================== modelo principal ================================

def build_mip_embed(*, base_row, budget: float, ct, bundle: XGBBundle) -> gp.Model:
    m = gp.Model("construction_embed")

    lot_area = float(base_row.get("Lot Area", getattr(ct, "lot_area", 7000)))
    lot_frontage = float(base_row.get("Lot Frontage", getattr(ct, "lot_frontage_ft", 60)))

    # features X en el orden del XGB
    feat_order, x = _build_x_inputs(m, bundle, base_row)

        # fijar LOT AREA al valor de entrada
    key_lot = _find_x_key(x, "Lot Area")
    if key_lot is not None:
        m.addConstr(x[key_lot] == lot_area, name="link__lot_area_fix")

    # fijar NEIGHBORHOOD one-hots segun parametro/base_row
    # busca todas las columnas Neighborhood_*
    neigh_token = str(base_row.get("neigh_arg", "") or getattr(ct, "neigh_arg", "")).strip()
    # normaliza ejemplos: "NAmes" -> "NWAmes" si ya llega mapeado en base_row, se usara eso
    # si en base_row viene set con 1 una de ellas, usamos esa prioridad
    picked = None
    for col in feat_order:
        if col.startswith("Neighborhood_") and col in x:
            if (col in base_row.index) and (float(base_row[col]) == 1.0):
                picked = col
                break
    if picked is None and neigh_token:
        # intenta emparejar por contains, case-insensitive
        for col in feat_order:
            if col.startswith("Neighborhood_") and col in x:
                if neigh_token.lower() in col.lower():
                    picked = col
                    break
    # aplica fijacion: la elegida =1, las demas =0 (solo si existen en el modelo)
    if picked is not None:
        for col in feat_order:
            if col.startswith("Neighborhood_") and col in x:
                m.addConstr(x[col] == (1.0 if col == picked else 0.0),
                            name=f"link__{col}__fix")


    # === variables de niveles por piso ===
    Floor1 = m.addVar(vtype=GRB.BINARY, name="Floor1")
    Floor2 = m.addVar(vtype=GRB.BINARY, name="Floor2")
    m.addConstr(Floor1 == 1, name="7.4__first_floor_always")
    m.addConstr(Floor2 <= Floor1, name="7.4__second_floor_if_first")

    # conteos por piso
    FullBath1 = m.addVar(vtype=GRB.INTEGER, lb=0, name="FullBath1")
    FullBath2 = m.addVar(vtype=GRB.INTEGER, lb=0, name="FullBath2")
    HalfBath1 = m.addVar(vtype=GRB.INTEGER, lb=0, name="HalfBath1")
    HalfBath2 = m.addVar(vtype=GRB.INTEGER, lb=0, name="HalfBath2")
    Kitchen1  = m.addVar(vtype=GRB.INTEGER, lb=0, name="Kitchen1")
    Kitchen2  = m.addVar(vtype=GRB.INTEGER, lb=0, name="Kitchen2")
    Bedroom1  = m.addVar(vtype=GRB.INTEGER, lb=0, name="Bedroom1")
    Bedroom2  = m.addVar(vtype=GRB.INTEGER, lb=0, name="Bedroom2")

    # areas por piso
    AreaFullBath1 = m.addVar(lb=0.0, name="AreaFullBath1")
    AreaFullBath2 = m.addVar(lb=0.0, name="AreaFullBath2")
    AreaHalfBath1 = m.addVar(lb=0.0, name="AreaHalfBath1")
    AreaHalfBath2 = m.addVar(lb=0.0, name="AreaHalfBath2")
    AreaKitchen1  = m.addVar(lb=0.0, name="AreaKitchen1")
    AreaKitchen2  = m.addVar(lb=0.0, name="AreaKitchen2")
    AreaBedroom1  = m.addVar(lb=0.0, name="AreaBedroom1")
    AreaBedroom2  = m.addVar(lb=0.0, name="AreaBedroom2")
    AreaOther1    = m.addVar(lb=0.0, name="AreaOther1")
    AreaOther2    = m.addVar(lb=0.0, name="AreaOther2")
    OtherRooms1   = m.addVar(vtype=GRB.INTEGER, lb=0, name="OtherRooms1")
    OtherRooms2   = m.addVar(vtype=GRB.INTEGER, lb=0, name="OtherRooms2")

    # agregados
    AreaKitchen   = m.addVar(lb=0.0, name="AreaKitchen")
    AreaFullBath  = m.addVar(lb=0.0, name="AreaFullBath")
    AreaHalfBath  = m.addVar(lb=0.0, name="AreaHalfBath")
    AreaBedroom   = m.addVar(lb=0.0, name="AreaBedroom")
    OtherRooms    = m.addVar(vtype=GRB.INTEGER, lb=0, name="OtherRooms")

    # banderas de features exteriores
    HasOpenPorch    = m.addVar(vtype=GRB.BINARY, name="HasOpenPorch")
    HasEnclosedPorch= m.addVar(vtype=GRB.BINARY, name="HasEnclosedPorch")
    HasScreenPorch  = m.addVar(vtype=GRB.BINARY, name="HasScreenPorch")
    Has3SsnPorch    = m.addVar(vtype=GRB.BINARY, name="Has3SsnPorch")
    HasWoodDeck     = m.addVar(vtype=GRB.BINARY, name="HasWoodDeck")
    HasPool         = m.addVar(vtype=GRB.BINARY, name="HasPool")
    HasReja         = m.addVar(vtype=GRB.BINARY, name="HasReja")

    # link a features X cuando existen
    _link_if_exists(m, x, "Full Bath", FullBath1 + FullBath2)
    _link_if_exists(m, x, "Half Bath", HalfBath1 + HalfBath2)
    _link_if_exists(m, x, "Kitchen AbvGr", Kitchen1 + Kitchen2)
    _link_if_exists(m, x, "Bedroom AbvGr", Bedroom1 + Bedroom2)

    # alias a columnas X
    x1 = _safe_get(x, "1st Flr SF")
    x2 = _safe_get(x, "2nd Flr SF")
    xpool = _safe_get(x, "Pool Area")
    xdeck = _safe_get(x, "Wood Deck SF")
    xopen = _safe_get(x, "Open Porch SF")
    xencl = _safe_get(x, "Enclosed Porch")
    x3ssn = _safe_get(x, "3Ssn Porch")
    xscreen = _safe_get(x, "Screen Porch")
    xgar = _safe_get(x, "Garage Area")
    xgr = _safe_get(x, "Gr Liv Area")

    # Total Porch SF puede NO existir en X, crea var interna si falta
    xpor_tot = _safe_get(x, "Total Porch SF")
    if xpor_tot is None:
        xpor_tot = m.addVar(lb=0.0, name="TotalPorchSF")
        x["Total Porch SF"] = xpor_tot

    # 7.1.x basicas de lot y pisos
    if x1 is not None and xpor_tot is not None and xpool is not None:
        m.addConstr(x1 + xpor_tot + xpool <= lot_area, name="7.1.1__lot_cap")
    if x1 is not None and x2 is not None:
        m.addConstr(x2 <= x1, name="7.1.2__2nd_leq_1st")
    if xgr is not None and x1 is not None and x2 is not None:
        m.addConstr(xgr == x1 + x2, name="7.1.3__GrLivArea")

    m.addConstr(AreaFullBath == AreaFullBath1 + AreaFullBath2, name="7.1.4__AFullBath")
    m.addConstr(AreaHalfBath == AreaHalfBath1 + AreaHalfBath2, name="7.1.5__AHalfBath")
    m.addConstr(FullBath1 >= 1, name="7.1.6__fullbath_1min")
    m.addConstr(Kitchen1 >= 1, name="7.1.7__kitchen_1min")

    eps1 = getattr(ct, "eps_floor_min_first", 450)
    eps2 = getattr(ct, "eps_floor_min_second", 350)
    if x1 is not None and x2 is not None:
        m.addConstr(x2 <= getattr(ct, "M2ndFlrSF_max", 0.5 * lot_area) * Floor2, name="7.1.8__2nd_max")
        m.addConstr(x2 >= eps2 * Floor2, name="7.1.8__2nd_min_if_on")
        m.addConstr(x1 >= eps1 * Floor1, name="7.1.8__1st_min_if_on")

    # 7.2 conteos totales
    FullBath  = m.addVar(vtype=GRB.INTEGER, lb=0, name="FullBath")
    HalfBath  = m.addVar(vtype=GRB.INTEGER, lb=0, name="HalfBath")
    Kitchen   = m.addVar(vtype=GRB.INTEGER, lb=0, name="Kitchen")
    m.addConstr(FullBath == FullBath1 + FullBath2, name="7.2.1__FullBath_count")
    m.addConstr(HalfBath == HalfBath1 + HalfBath2, name="7.2.2__HalfBath_count")
    m.addConstr(Kitchen  == Kitchen1  + Kitchen2,  name="7.2.3__Kitchen_count")
    _link_if_exists(m, x, "Full Bath", FullBath)
    _link_if_exists(m, x, "Half Bath", HalfBath)
    _link_if_exists(m, x, "Kitchen AbvGr", Kitchen)

    # 7.3 caps por tipo de edificio
    bldg_opts = ["1Fam","TwnhsE","TwnhsI","Duplex","2FmCon"]
    Bldg = _one_hot(m, "BldgType", bldg_opts)
    _tie_one_hot_to_x(m, x, "Bldg Type", Bldg)
    Bed_max = {"1Fam":6,"TwnhsE":4,"TwnhsI":4,"Duplex":5,"2FmCon":8}
    F_max   = {"1Fam":4,"TwnhsE":3,"TwnhsI":3,"Duplex":4,"2FmCon":6}
    H_max   = {"1Fam":2,"TwnhsE":2,"TwnhsI":2,"Duplex":2,"2FmCon":3}
    K_max   = {"1Fam":1,"TwnhsE":1,"TwnhsI":1,"Duplex":2,"2FmCon":2}
    Ch_max  = {"1Fam":1,"TwnhsE":1,"TwnhsI":1,"Duplex":1,"2FmCon":2}
    Bedrooms = m.addVar(vtype=GRB.INTEGER, lb=0, name="Bedrooms")
    m.addConstr(Bedrooms == Bedroom1 + Bedroom2, name="7.3.1__bedrooms_sum")
    m.addConstr(Bedrooms <= gp.quicksum(Bed_max[b]*Bldg[b] for b in bldg_opts), name="7.3.1__bedrooms_cap")
    m.addConstr(FullBath <= gp.quicksum(F_max[b]*Bldg[b] for b in bldg_opts), name="7.3.2__fullbath_cap")
    m.addConstr(HalfBath <= gp.quicksum(H_max[b]*Bldg[b] for b in bldg_opts), name="7.3.3__halfbath_cap")
    m.addConstr(Kitchen  <= gp.quicksum(K_max[b]*Bldg[b] for b in bldg_opts), name="7.3.4__kitchen_cap")
    Fireplaces = _safe_get(x, "Fireplaces") or m.addVar(vtype=GRB.INTEGER, lb=0, name="Fireplaces")
    m.addConstr(Fireplaces <= gp.quicksum(Ch_max[b]*Bldg[b] for b in bldg_opts), name="7.3.5__fireplaces_cap")
    _link_if_exists(m, x, "Bedroom AbvGr", Bedrooms)

    # 7.5 garage
    GarageCars = _safe_get(x, "Garage Cars") or m.addVar(vtype=GRB.INTEGER, lb=0, name="GarageCars")
    if xgar is not None:
        m.addConstr(150 * GarageCars <= xgar, name="7.5.1__gar_area_lb")
        m.addConstr(xgar <= 250 * GarageCars, name="7.5.1__gar_area_ub")
    gar_types = ["NA","Attchd","Detchd","BuiltIn","Basment","CarPort","2Types"]
    GT = _one_hot(m, "GarageType", gar_types)
    _tie_one_hot_to_x(m, x, "Garage Type", GT)
    gf_opts = ["NA","Fin","RFn","Unf"]
    GF = _one_hot(m, "GarageFinish", gf_opts)
    _tie_one_hot_to_x(m, x, "Garage Finish", GF)
    m.addConstr(GF["NA"] == GT["NA"], name="7.5.3__finish_na_eq_type_na")
    m.addConstr(GF["Fin"] + GF["RFn"] + GF["Unf"] == 1 - GT["NA"], name="7.5.3__finish_if_has")
    m.addConstr(GarageCars >= 1 - GT["NA"], name="7.5.4__min_cars_if_has")
    m.addConstr(GarageCars <= int(getattr(ct, "max_garage_cars", 4)) * (1 - GT["NA"]), name="7.5.2__cap_cars_if_has")
    if xgar is not None:
        m.addConstr(xgar <= (0.2 * lot_area) * (1 - GT["NA"]), name="7.5.2__cap_area_if_has")

    # 7.7 techos (plan vs actual, estilo y material)
    PR1 = m.addVar(lb=0.0, name="PR1")
    PR2 = m.addVar(lb=0.0, name="PR2")
    PlanRoofArea = m.addVar(lb=0.0, name="PlanRoofArea")
    m.addConstr(PlanRoofArea == PR1 + PR2, name="7.7.2__PlanRoof_sum")
    U1 = getattr(ct, "alpha1", 0.6) * lot_area
    U2 = getattr(ct, "alpha2", 0.5) * lot_area
    Uplan = getattr(ct, "alpha_plan", 0.6) * lot_area
    if x1 is not None:
        m.addConstr(PR1 <= x1, name="7.7.1__pr1_leq_1st")
        m.addConstr(PR1 <= U1 * Floor1, name="7.7.1__pr1_if_floor1")
        m.addConstr(PR1 >= x1 - U1 * (1 - Floor1), name="7.7.1__pr1_linear")
    if x2 is not None:
        m.addConstr(PR2 <= x2, name="7.7.1__pr2_leq_2nd")
        m.addConstr(PR2 <= U2 * Floor2, name="7.7.1__pr2_if_floor2")
        m.addConstr(PR2 >= x2 - U2 * (1 - Floor2), name="7.7.1__pr2_linear")
    m.addConstr(PlanRoofArea <= Uplan, name="7.7.1__planroof_ub")

    S = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
    M = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
    RS = _one_hot(m, "RoofStyle", S)
    RM = _one_hot(m, "RoofMatl", M)
    _tie_one_hot_to_x(m, x, "Roof Style", RS)
    _tie_one_hot_to_x(m, x, "Roof Matl", RM)

    Y, Z = {}, {}
    for s in S:
        for mm in M:
            y = m.addVar(vtype=GRB.BINARY, name=f"Y__{s}__{mm}")
            z = m.addVar(lb=0.0, name=f"Z__{s}__{mm}")
            Y[(s, mm)] = y
            Z[(s, mm)] = z
            m.addConstr(y <= RS[s], name=f"7.7.4__y_leq_rs__{s}_{mm}")
            m.addConstr(y <= RM[mm], name=f"7.7.4__y_leq_rm__{s}_{mm}")
            m.addConstr(y >= RS[s] + RM[mm] - 1, name=f"7.7.4__y_ge_and__{s}_{mm}")
            m.addConstr(z <= PlanRoofArea, name=f"7.7.5__z_leq_plan__{s}_{mm}")
            m.addConstr(z <= Uplan * y, name=f"7.7.5__z_leq_u_y__{s}_{mm}")
            m.addConstr(z >= PlanRoofArea - Uplan * (1 - y), name=f"7.7.5__z_ge_plan_u__{s}_{mm}")
    m.addConstr(gp.quicksum(Y.values()) == 1, name="7.7.4__one_y")

    gamma = getattr(ct, "gamma", {})
    ActualRoofArea = m.addVar(lb=0.0, name="ActualRoofArea")
    m.addConstr(
        ActualRoofArea == gp.quicksum(float(gamma.get((s, mm), 1.10)) * Z[(s, mm)] for s in S for mm in M),
        name="7.7.3__actual_roof_area",
    )

    # 7.8, 7.9 caps de areas total y por parte
    tbsmt = _safe_get(x, "Total Bsmt SF")
    if tbsmt is not None and x1 is not None and x2 is not None:
        TotArea = m.addVar(lb=0.0, name="TotalArea")
        m.addConstr(TotArea == x1 + x2 + tbsmt, name="7.8__total_area")
        m.addConstr(x1 <= 0.6 * lot_area, name="7.9__cap_1st")
        m.addConstr(x2 <= 0.5 * lot_area, name="7.9__cap_2nd")
        m.addConstr(tbsmt <= 0.5 * lot_area, name="7.9__cap_bsmt")
        if xgr is not None:
            m.addConstr(xgr <= 0.8 * lot_area, name="11.10__GrLiv_leq_0.8lot")
        if xgar is not None:
            m.addConstr(xgar <= 0.2 * lot_area, name="7.9__cap_garage")

    # 7.10 relacion banos vs dormitorios
    m.addConstr(3 * FullBath >= 2 * Bedrooms, name="7.10__baths_vs_bed")

    # 7.11 piscina
    if all(v is not None for v in [x1,xgar,xdeck,xopen,xencl,xscreen,x3ssn,xpool]):
        m.addConstr(xpool <= (lot_area - x1 - xgar - xdeck - xopen - xencl - xscreen - x3ssn) * HasPool, name="7.11.1__pool_space")
        m.addConstr(xpool <= 0.1 * lot_area * HasPool, name="7.11.1__pool_max")
        m.addConstr(xpool >= 160 * HasPool, name="7.11.1__pool_min")
        m.addConstr(xpool >= 0, name="7.11.1__pool_nonneg")

    # 7.12 porches (forzamos identidad TotalPorch = suma componentes)
    if all(v is not None for v in [xopen,xencl,xscreen,x3ssn]):
        m.addConstr(xpor_tot == xopen + xencl + xscreen + x3ssn, name="7.12.1__porch_sum")
        m.addConstr(xpor_tot <= 0.25 * lot_area, name="7.12.1__porch_cap_total")
        if x1 is not None:
            m.addConstr(xpor_tot <= x1, name="7.12.1__porch_leq_1st")
        m.addConstr(xopen  >= 40 * HasOpenPorch, name="7.12.2__open_min")
        m.addConstr(xencl  >= 60 * HasEnclosedPorch, name="7.12.2__encl_min")
        m.addConstr(xscreen>= 40 * HasScreenPorch, name="7.12.2__screen_min")
        m.addConstr(x3ssn  >= 80 * Has3SsnPorch, name="7.12.2__3ssn_min")
        m.addConstr(xdeck + xpor_tot + (xpool if xpool is not None else 0) <= 0.35 * lot_area, name="7.12.3__ext_cap")
        m.addConstr(xdeck + xopen <= 0.20 * lot_area, name="7.12.3__deck_open_cap")
        m.addConstr(xdeck >= 40 * HasWoodDeck, name="7.13__deck_min")
        m.addConstr(xdeck <= 0.15 * lot_area * HasWoodDeck, name="7.13__deck_max")

    # 7.18 sotano: exposicion y tipos, areas y banos
    exp_opts = ["Gd","Av","Mn","No","NA"]
    EXP = _one_hot(m, "BsmtExposure", exp_opts)
    _tie_one_hot_to_x(m, x, "Bsmt Exposure", EXP)
    B1 = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"]
    B2 = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"]
    T1 = _one_hot(m, "BsmtFinType1", B1)
    T2 = _one_hot(m, "BsmtFinType2", B2)
    _tie_one_hot_to_x(m, x, "BsmtFin Type 1", T1)
    _tie_one_hot_to_x(m, x, "BsmtFin Type 2", T2)

    real_keys = ["GLQ","ALQ","BLQ","Rec","LwQ"]
    phi1 = gp.quicksum(T1[k] for k in B1 if k != "NA")
    phi2 = gp.quicksum(T2[k] for k in B2 if k != "NA")
    psi1 = gp.quicksum(T1[k] for k in real_keys)
    psi2 = gp.quicksum(T2[k] for k in real_keys)

    U_bsmt = 0.5 * lot_area
    Af_min = getattr(ct, "bsmt_finish_min_area", 100.0)
    UbF = getattr(ct, "bsmt_fullbath_cap", 2)
    UbH = getattr(ct, "bsmt_halfbath_cap", 1)

    BsmtFinSF1 = m.addVar(lb=0.0, name="BsmtFinSF1")
    BsmtFinSF2 = m.addVar(lb=0.0, name="BsmtFinSF2")
    BsmtFullBath = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtFullBath")
    BsmtHalfBath = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtHalfBath")

    if tbsmt is not None:
        m.addConstr(tbsmt == BsmtFinSF1 + BsmtFinSF2, name="7.8__bsmt_parts_sum")
        m.addConstr(tbsmt <= U_bsmt * (1 - EXP["NA"]), name="7.18__bsmt_cap_if_not_na")
    m.addConstr(BsmtFinSF1 <= U_bsmt * phi1, name="7.18__sf1_cap_by_type")
    m.addConstr(BsmtFinSF2 <= U_bsmt * phi2, name="7.18__sf2_cap_by_type")
    m.addConstr(BsmtFinSF1 >= Af_min * psi1, name="7.18__sf1_min_if_real")
    m.addConstr(BsmtFinSF2 >= Af_min * psi2, name="7.18__sf2_min_if_real")
    m.addConstr(BsmtFullBath <= UbF * (psi1 + psi2), name="7.18__bsmt_fullbath_if_real")
    m.addConstr(BsmtHalfBath <= UbH * (psi1 + psi2), name="7.18__bsmt_halfbath_if_real")
    m.addConstr(BsmtFinSF1 <= U_bsmt * (1 - EXP["NA"]), name="7.18__sf1_off_if_na")
    m.addConstr(BsmtFinSF2 <= U_bsmt * (1 - EXP["NA"]), name="7.18__sf2_off_if_na")
    m.addConstr(BsmtFullBath <= UbF * (1 - EXP["NA"]), name="7.18__bf_off_if_na")
    m.addConstr(BsmtHalfBath <= UbH * (1 - EXP["NA"]), name="7.18__bh_off_if_na")
    _link_if_exists(m, x, "Bsmt Full Bath", BsmtFullBath)
    _link_if_exists(m, x, "Bsmt Half Bath", BsmtHalfBath)

    # 7.15 exteriores 1st vs 2nd: flag mismo material
    SameMaterial = m.addVar(vtype=GRB.BINARY, name="SameMaterial")
    ext1_opts = ["VinylSd","MetalSd","Wd Sdng","HdBoard","Stucco","Plywood","CemntBd","BrkFace","BrkComm","WdShing","AsbShng","Stone","ImStucc","AsphShn","CBlock"]
    ext2_opts = ext1_opts
    EXT1 = _one_hot(m, "Exterior1st", ext1_opts)
    EXT2 = _one_hot(m, "Exterior2nd", ext2_opts)
    _tie_one_hot_to_x(m, x, "Exterior 1st", EXT1)
    _tie_one_hot_to_x(m, x, "Exterior 2nd", EXT2)
    for e1 in ext1_opts:
        if e1 in EXT2:
            m.addConstr(SameMaterial >= EXT1[e1] + EXT2[e1] - 1, name=f"7.15__same_mat__{e1}")

    # 7.16 enchapados
    mas_opts = ["BrkCmn","BrkFace","CBlock","None","Stone"]
    MV = _one_hot(m, "MasVnrType", mas_opts)
    _tie_one_hot_to_x(m, x, "Mas Vnr Type", MV)
    MasVnrArea = _safe_get(x, "Mas Vnr Area") or m.addVar(lb=0.0, name="MasVnrArea")
    Umas = 0.4 * ((x1 + x2) if (x1 is not None and x2 is not None) else lot_area)
    m.addConstr(MasVnrArea <= Umas, name="7.16.1__mas_cap")
    m.addConstr(MasVnrArea >= 20.0 * (1 - MV["None"]), name="7.16__mas_min_if_used")

    # 7.20 fundacion
    f_opts = ["BrkTil","CBlock","PConc","Slab","Stone","Wood"]
    FND = _one_hot(m, "Foundation", f_opts)
    _tie_one_hot_to_x(m, x, "Foundation", FND)
    AreaFoundation = m.addVar(lb=0.0, name="AreaFoundation")
    if x1 is not None:
        m.addConstr(AreaFoundation == x1, name="7.20__AreaFoundation_eq_1st")
    Ufound = 0.6 * lot_area
    for f in f_opts:
        FA = m.addVar(lb=0.0, name=f"FA__{f}")
        m.addConstr(FA <= AreaFoundation, name=f"7.20__fa_leq_area__{f}")
        m.addConstr(FA <= Ufound * FND[f], name=f"7.20__fa_leq_u_f__{f}")
        m.addConstr(FA >= AreaFoundation - Ufound * (1 - FND[f]), name=f"7.20__fa_ge_lin__{f}")
    # if Slab o Wood entonces NA en BsmtExposure (permite NA)
    m.addConstr(FND["Slab"] <= EXP["NA"], name="7.20__slab_implies_na")
    m.addConstr(FND["Wood"] <= EXP["NA"], name="7.20__wood_implies_na")

    # 7.21 caps de areas por piso
    UBed1, UFullB1, UHalfB1, UKitch1, UOther1 = 0.5 * lot_area, 200, 80, 300, 0.5 * lot_area
    UBed2, UFullB2, UHalfB2, UKitch2, UOther2 = 200, 60, 20, 200, 0.5 * lot_area
    m.addConstr(AreaBedroom1 <= UBed1 * Floor1, name="7.21.1__abed1_cap")
    m.addConstr(AreaFullBath1 <= UFullB1 * Floor1, name="7.21.1__afb1_cap")
    m.addConstr(AreaHalfBath1 <= UHalfB1 * Floor1, name="7.21.1__ahb1_cap")
    m.addConstr(AreaKitchen1  <= UKitch1 * Floor1, name="7.21.1__ak1_cap")
    m.addConstr(AreaOther1    <= UOther1 * Floor1, name="7.21.1__aoth1_cap")
    m.addConstr(AreaBedroom2 <= UBed2 * Floor2, name="7.21.2__abed2_cap")
    m.addConstr(AreaFullBath2 <= UFullB2 * Floor2, name="7.21.2__afb2_cap")
    m.addConstr(AreaHalfBath2 <= UHalfB2 * Floor2, name="7.21.2__ahb2_cap")
    m.addConstr(AreaKitchen2  <= UKitch2 * Floor2, name="7.21.2__ak2_cap")
    m.addConstr(AreaOther2    <= UOther2 * Floor2, name="7.21.2__aoth2_cap")
    if x2 is not None:
        m.addConstr(AreaBedroom2 + AreaKitchen2 + AreaHalfBath2 + AreaFullBath2 + AreaOther2 <= x2, name="7.21__sum_leq_2nd")

    # 7.22 otros recintos
    m.addConstr(OtherRooms == OtherRooms1 + OtherRooms2, name="7.22.1__other_cnt_sum")
    m.addConstr(AreaOther1 >= 100 * OtherRooms1, name="7.22.4__other1_min")
    m.addConstr(AreaOther2 >= 100 * OtherRooms2, name="7.22.4__other2_min")
    m.addConstr(OtherRooms1 >= 1, name="7.22.2__other_min_on_1st")
    m.addConstr(OtherRooms2 <= 8 * Floor2, name="7.22.3__other_cnt_2nd_if_on")

    # 7.23 area exterior porimetro (aprox simple)
    smin, smax = 20.0, 70.0

    P1 = m.addVar(lb=0.0, name="P1")
    P2 = m.addVar(lb=0.0, name="P2")
    AreaExterior = m.addVar(lb=0.0, name="AreaExterior1st")

    # UB seguro: usa smin, crece con area y no cruza con el LB constante
    if x1 is not None:
        # P1 <= 2 * (x1/smin + smin) cuando hay primer piso
        m.addConstr(P1 <= 2 * ((x1 / smin) + smin) * Floor1, name="7.23__p1_ub_safe")
        # LB constante seguro: si hay primer piso, a lo menos un rectangulo 2*smin por smin
        m.addConstr(P1 >= 4 * smin * Floor1, name="7.23__p1_lb_safe")

    if x2 is not None:
        # P2 <= 2 * (x2/smin + smin) cuando hay segundo piso
        m.addConstr(P2 <= 2 * ((x2 / smin) + smin) * Floor2, name="7.23__p2_ub_safe")
        # LB constante seguro en el segundo piso
        m.addConstr(P2 >= 4 * smin * Floor2, name="7.23__p2_lb_safe")

    # si hay segundo piso, exige P1 >= P2. si no hay, no impone nada
    m.addConstr(P1 >= P2 * Floor2, name="7.23__p1_ge_p2_if_2nd")

    # area de fachada exterior
    Hext = getattr(ct, "Hext", 7.0)
    m.addConstr(AreaExterior == Hext * (P1 + P2), name="7.23.1__area_ext")


    ext1_opts = ["VinylSd","MetalSd","Wd Sdng","HdBoard","Stucco","Plywood","CemntBd","BrkFace","BrkComm","WdShing","AsbShng","Stone","ImStucc","AsphShn","CBlock"]
    W = {}
    Uext = getattr(ct, "Uext", 4500.0)
    for e1 in ext1_opts:
        w = m.addVar(lb=0.0, name=f"W__{e1}")
        W[e1] = w
        b = EXT1[e1]  # usa el one-hot ya creado, evita getVarByName(None)
        m.addConstr(w <= AreaExterior, name=f"7.23.1__w_leq_area__{e1}")
        m.addConstr(w <= Uext * b, name=f"7.23.1__w_leq_u__{e1}")
        m.addConstr(w >= AreaExterior - Uext * (1 - b), name=f"7.23.1__w_ge_lin__{e1}")
    m.addConstr(gp.quicksum(W.values()) == AreaExterior, name="7.23.1__w_sum")

 ##REVISAR. NO DBERIA NECESITAR ESTO SI ESTÁN BIEN PUESTAS LAAS RESTRICCIONES
    # === UBs seguros para variables X existentes ===

    safe_ubs = {
        "1st Flr SF": 0.60 * lot_area,
        "2nd Flr SF": 0.50 * lot_area,
        "Total Bsmt SF": 0.50 * lot_area,
        "Gr Liv Area": 0.80 * lot_area,
        "Garage Area": 0.20 * lot_area,
        "Wood Deck SF": 0.15 * lot_area,
        "Open Porch SF": 0.25 * lot_area,
        "Enclosed Porch": 0.25 * lot_area,
        "3Ssn Porch": 0.25 * lot_area,
        "Screen Porch": 0.25 * lot_area,
        "Pool Area": 0.10 * lot_area,
        "Lot Area": lot_area,
        "Full Bath": 6,
        "Half Bath": 6,
        "Bedroom AbvGr": 10,
        "Kitchen AbvGr": 3,
        "Garage Cars": int(getattr(ct, "max_garage_cars", 4)),
        "Fireplaces": 3,
    }

    import math
    for nm, ub in safe_ubs.items():
        v = x.get(nm) or x.get(_find_x_key(x, nm))
        if v is not None:
            try:
                if (not math.isfinite(float(v.UB))) or (float(v.UB) > float(ub)):
                    v.UB = float(ub)
            except Exception:
                pass


    # === COSTO total y OBJETIVO ===
    cost_expr = _build_cost_expr(m, x, ct)
    cost_var = m.addVar(lb=0.0, name="cost_model")
    m.addConstr(cost_var == cost_expr, name="def_cost_model")

    y_log, y_price = _attach_xgb_embed(m, bundle, feat_order, x)

    # objetivo = precio - costo, con presupuesto
    m.setObjective(y_price - cost_var, GRB.MAXIMIZE)
    m.addConstr(cost_var <= budget, name="budget")

    # Auditoria y etiquetas para post-proceso
    m._y_price_var = y_price
    m._y_log_var   = y_log
    m._budget_usd  = float(budget)
    m._x = x
    m._report_vars = {
        "Floor1": Floor1, "Floor2": Floor2,
        "1st Flr SF": x1, "2nd Flr SF": x2,
        "FullBath": FullBath, "HalfBath": HalfBath, "Kitchen": Kitchen,
        "Bedrooms": Bedrooms, "Garage Area": xgar, "Garage Cars": GarageCars,
        "PlanRoofArea": PlanRoofArea, "ActualRoofArea": ActualRoofArea,
        "Total Bsmt SF": tbsmt,
        "Total Porch SF": xpor_tot,
        "Wood Deck SF": xdeck, "Pool Area": xpool,
        "Cost": cost_var, "y_price": y_price,
    }

    # Tolerancias numericas estrictas
    m.setParam("MIPGap", getattr(ct, "mip_gap", 1e-3))
    m.setParam("TimeLimit", getattr(ct, "time_limit", 300))
    m.setParam("FeasibilityTol", 1e-7)
    m.setParam("IntFeasTol", 1e-7)
    m.setParam("OptimalityTol", 1e-7)
    m.setParam("NumericFocus", 3)

    return m

# ================= DEBUG IIS / CONFLICTS =================

import collections
import math

def dump_infeasibility_report(m: gp.Model, tag: str = "construction") -> None:
    """
    - fuerza DualReductions=0 y re-optimiza para distinguir INF vs UNBD
    - si es INF, corre computeIIS y escribe archivos .lp y .ilp
    - imprime resumen agrupado por prefijo (antes de "__") para ubicar rapido el bloque conflictivo
    """
    try:
        # 1) distinguir infeasible vs unbounded
        m.setParam("DualReductions", 0)
        m.optimize()
    except Exception:
        pass

    status = int(getattr(m, "Status", -1))
    print(f"[DEBUG] status tras DualReductions=0: {status}")

    # escribe el modelo completo por si ayuda
    try:
        m.write(f"{tag}_model.lp")
    except Exception:
        pass

    if status not in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
        print("[DEBUG] el modelo no quedo inelegible para IIS (no es INFEASIBLE/INF_OR_UNBD).")
        return

    # 2) IIS
    print("[DEBUG] corriendo computeIIS() ...")
    m.computeIIS()
    try:
        m.write(f"{tag}_conflict.ilp")
        print(f"[DEBUG] escrito IIS en {tag}_conflict.ilp y modelo en {tag}_model.lp")
    except Exception as e:
        print(f"[DEBUG] no pude escribir .ilp: {e}")

    # 3) listar conflictos por tipo
    con_flags = [c for c in m.getConstrs() if getattr(c, "IISConstr", 0)]
    qcon_flags = [qc for qc in m.getQConstrs() if getattr(qc, "IISQConstr", 0)]
    gen_flags = [gc for gc in m.getGenConstrs() if getattr(gc, "IISGenConstr", 0)]
    sos_flags = [s for s in m.getSOSs() if getattr(s, "IISSOS", 0)]
    v_lb = [v for v in m.getVars() if getattr(v, "IISLB", 0)]
    v_ub = [v for v in m.getVars() if getattr(v, "IISUB", 0)]

    print(f"[IIS] lin-constr: {len(con_flags)}  q-constr: {len(qcon_flags)}  gen-constr: {len(gen_flags)}  sos: {len(sos_flags)}  varLB: {len(v_lb)}  varUB: {len(v_ub)}")

    # 4) agrupar por prefijo antes de "__" para ver bloques (ej: 7.12.1, 7.7.4, etc)
    def pref(nm: str) -> str:
        return nm.split("__")[0] if "__" in nm else nm

    bucket = collections.Counter(pref(c.ConstrName) for c in con_flags)
    if bucket:
        print("[IIS] top prefijos conflictivos:")
        for k, cnt in bucket.most_common(20):
            print(f"   {k:>20s} -> {cnt}")

    # 5) sample de restricciones exactas
    if con_flags:
        print("[IIS] primeras 30 restricciones lineales en el IIS:")
        for c in con_flags[:30]:
            print("   -", c.ConstrName)

    # 6) bounds problemáticos
    if v_lb or v_ub:
        print("[IIS] variables con bounds en el IIS (muestra hasta 20):")
        for v in (v_lb + v_ub)[:20]:
            try:
                print(f"   - {v.VarName}  LB={float(v.LB)}  UB={float(v.UB)}")
            except Exception:
                print(f"   - {v.VarName}  (LB/UB no legibles)")

    # 7) gen y q constr
    if gen_flags:
        print("[IIS] gen-constr en IIS (ej, PWL_exp):")
        for gc in gen_flags[:20]:
            print("   -", gc.ConstrName)
    if qcon_flags:
        print("[IIS] q-constr en IIS:")
        for qc in qcon_flags[:20]:
            print("   -", qc.QCName)

def build_and_tag(m: gp.Model, tag: str):
    """
    util chico por si quieres setear un sufijo de archivos para distintos escenarios
    """
    setattr(m, "_debug_tag", str(tag or "construction"))
