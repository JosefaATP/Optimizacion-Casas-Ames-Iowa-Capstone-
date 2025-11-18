# -*- coding: utf-8 -*-
from __future__ import annotations
import math, re, json
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
        # intenta match directo; si falla, usa alias comunes (None/NA <-> No aplica)
        key = _find_x_key(x, feat_name, opt)
        if key is None:
            opt_norm = _norm(opt)
            aliases = []
            if opt_norm in ("none",):
                aliases = ["No aplica", "None"]
            elif opt_norm in ("na",):
                aliases = ["No aplica", "NA"]
            for alt in aliases:
                key = _find_x_key(x, feat_name, alt)
                if key is not None:
                    break
        if key is not None:
            m.addConstr(x[key] == b, name=f"link__{_norm(feat_name)}__{_norm(opt)}")


def _fix_ohe_group(m: gp.Model, x: Dict[str, gp.Var], feat_order: list[str], group: str, chosen: str):
    """Fija un grupo one-hot en X (solo para columnas OHE del XGB) con nombre tipo 'Group_Label'.
    - Establece la columna del label elegido a 1 y el resto del grupo a 0.
    - Robusto a espacios vs guiones bajos en los nombres de columnas del XGB.
    - Tolerante a alias del nombre base (p. ej., 'Functional' vs 'Functiono aplical').
    """
    def base_of(col: str) -> str:
        return col.split("_", 1)[0].strip()
    group_norm = _norm(group)
    # Aliases específicos que hemos visto en datos
    ALIASES = {
        "functional": ["functional", "functionoaplical", "funcional", "FunctioNo aplical"],
    }
    alias_norms = set(ALIASES.get(group_norm, [group_norm]))

    # 1) Colecciona todas las columnas del grupo buscando por prefijo y por base normalizada
    cols: list[str] = []
    for c in feat_order:
        if c not in x:
            continue
        if c.startswith(f"{group}_") or c.startswith(f"{group} "):
            cols.append(c)
            continue
        b = base_of(c)
        if _norm(b) in alias_norms:
            cols.append(c)
    if not cols:
        return
    # Busca la columna elegida
    pick = None
    for c in cols:
        # etiqueta = token después del último '_'
        tail = c.rsplit("_", 1)[1] if "_" in c else c
        if tail == chosen:
            pick = c
            break
    # Si no encontramos el nombre exacto, intenta match lax normalizado
    if pick is None:
        want = _norm(chosen)
        for c in cols:
            tail = c.rsplit("_", 1)[1] if "_" in c else c
            if _norm(tail) == want:
                pick = c
                break
    # Fija elegido=1 y resto=0
    for c in cols:
        try:
            m.addConstr(x[c] == (1.0 if c == pick else 0.0), name=f"link__fix_ohe__{_norm(group)}__{_norm(c)}")
        except Exception:
            pass


def _enforce_exclusive_ohe(m: gp.Model, x: Dict[str, gp.Var], feat_order: list[str], group: str):
    """Agrega sum-to-one para todas las columnas OHE del grupo en X."""
    def base_of(col: str) -> str:
        return col.split("_", 1)[0].strip()
    group_norm = _norm(group)
    cols: list[gp.Var] = []
    for c in feat_order:
        if c not in x:
            continue
        if c.startswith(f"{group}_") or c.startswith(f"{group} "):
            cols.append(x[c]); continue
        if _norm(base_of(c)) == group_norm:
            cols.append(x[c])
    if len(cols) > 1:
        try:
            m.addConstr(gp.quicksum(cols) == 1.0, name=f"ex__x__{_norm(group)}__onehot")
        except Exception:
            pass


# ===================== inputs en el orden del XGB (features) ======================

def _build_x_inputs(m: gp.Model, bundle: XGBBundle, X_base_row, lot_area) -> tuple[List[str], Dict[str, gp.Var]]:
    # Importante: usar el orden que ve el Booster (ColumnTransformer numeric cols)
    try:
        feat_order: List[str] = list(bundle.booster_feature_order())
    except Exception:
        feat_order = list(bundle.feature_names_in())
    x_vars: Dict[str, gp.Var] = {}

    INT_NAMES = {"Full Bath","Half Bath","Bedroom AbvGr","Kitchen AbvGr","Garage Cars","Fireplaces"}
    NONNEG_LB0 = set(INT_NAMES) | {
        "1st Flr SF","2nd Flr SF","Gr Liv Area","Total Bsmt SF","Garage Area",
        "Wood Deck SF","Open Porch SF","Enclosed Porch","3Ssn Porch","Screen Porch",
        "Pool Area","Lot Area","Mas Vnr Area",
    }
    # Trata como ordinales enteros [-1..4] varias calidades y utilidades
    ORD_CANDIDATES = {
        "Roof Style", "Roof Matl", "Utilities",
        "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
        "Heating QC", "Kitchen Qual", "Garage Qual", "Garage Cond",
        "Pool QC", "Fireplace Qu",
    }

    for col in feat_order:
        val = X_base_row[col] if col in X_base_row.index else None
        lb, ub, vtype = 0.0, GRB.INFINITY, GRB.CONTINUOUS
        if col in INT_NAMES:
            vtype = GRB.INTEGER
        if col in NONNEG_LB0:
            lb = 0.0
        if col in ORD_CANDIDATES:
            # Ordinales: -1 (No aplica) .. 4 (Ex). Para Roof Style/Matl usamos enteros más amplios
            if col in ("Roof Style", "Roof Matl"):
                vtype, lb, ub = GRB.INTEGER, -1, 8
            else:
                vtype, lb, ub = GRB.INTEGER, -1, 4

        # dummies OHE: forzar [0,1]
        if "_" in col and col not in NONNEG_LB0:
            lb, ub = 0.0, 1.0
        
        if col == "Lot Area":
            lb = ub = float(lot_area)  # fijo
        elif col == "1st Flr SF":
            ub = min(ub, 0.60 * lot_area)
        elif col == "2nd Flr SF":
            ub = min(ub, 0.50 * lot_area)
        elif col == "Total Bsmt SF":
            ub = min(ub, 0.50 * lot_area)
        elif col == "Gr Liv Area":
            ub = min(ub, 0.80 * lot_area)
        elif col == "Garage Area":
            ub = min(ub, 0.20 * lot_area)
        elif col == "Total Porch SF":
            ub = min(ub, 0.25 * lot_area)
        elif col in ("Wood Deck SF","Open Porch SF","Enclosed Porch","3Ssn Porch","Screen Porch"):
            ub = min(ub, 0.25 * lot_area)
        elif col == "Pool Area":
            ub = min(ub, 0.10 * lot_area)

        if col == "Lot Area" and val is not None and not math.isnan(val):
            v = m.addVar(lb=float(val), ub=float(val), name=f"x_const__{col}")
        else:
            v = m.addVar(lb=lb, ub=ub, vtype=vtype, name=f"x_{col}")
        x_vars[col] = v
    

    m.update()
    return feat_order, x_vars


# ===== bound tightening from Booster thresholds (reduce big-M) =====
def _tighten_bounds_from_booster(m: gp.Model, bundle: XGBBundle,
                                 feat_order: List[str], x: Dict[str, gp.Var],
                                 tol: float = 1e-6) -> None:
    try:
        bst = bundle.reg.get_booster()
    except Exception:
        bst = getattr(bundle.reg, "_Booster", None)
    if bst is None:
        return

    try:
        dumps = bst.get_dump(with_stats=False, dump_format="json")
        # Si el bundle trae early stopping, usa solo esos arboles
        try:
            n_use = int(getattr(bundle, "n_trees_use", 0) or 0)
        except Exception:
            n_use = 0
        if n_use > 0 and n_use < len(dumps):
            dumps = dumps[:n_use]
    except Exception:
        return

    thr_map: Dict[int, List[float]] = {}
    for js in dumps:
        try:
            node = json.loads(js)
        except Exception:
            continue

        def walk(nd):
            if "leaf" in nd:
                return
            try:
                f_idx = int(str(nd.get("split", "")).replace("f", ""))
                thr = float(nd.get("split_condition", float("nan")))
            except Exception:
                f_idx, thr = None, float("nan")
            if f_idx is not None and math.isfinite(thr):
                thr_map.setdefault(f_idx, []).append(thr)
            for ch in nd.get("children", []):
                walk(ch)
        walk(node)

    # columnas que fijamos por proyecto: NO estrechar (evita choques con valores fijos)
    skip_cols = {
        "Year Built", "Year Remod/Add", "Year Sold", "Month Sold",
        "Yr Sold", "Mo Sold", "Garage Yr Blt", "Garage Yr Built",
        "Overall Qual", "Overall Cond",
        "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual",
        "Utilities", "Low Qual Fin SF",
    }

    for f_idx, thrs in thr_map.items():
        if not thrs:
            continue
        col = feat_order[f_idx] if f_idx < len(feat_order) else None
        if col is None or col not in x:
            continue
        if col in skip_cols:
            # mantén bounds originales en columnas fijas por diseño
            continue
        v = x[col]
        try:
            vtype = getattr(v, "VType", GRB.CONTINUOUS)
        except Exception:
            vtype = GRB.CONTINUOUS
        # Detectar dummies/OHE (trátalas como binarias 0..1; NO subas LB)
        try:
            cur_lb = float(getattr(v, "LB", 0.0))
            cur_ub = float(getattr(v, "UB", float("inf")))
        except Exception:
            cur_lb, cur_ub = 0.0, float("inf")
        is_binary_like = (cur_lb >= -tol) and (cur_ub <= 1.0 + tol)
        tmin, tmax = min(thrs), max(thrs)

        # margen pequeño (más grande si el rango es minúsculo)
        pad = max(tol, 0.01 * (abs(tmax - tmin) + 1.0))
        new_lb = tmin - pad
        new_ub = tmax + pad

        # Si binaria o dummy: respeta [0,1]
        if vtype == GRB.BINARY or is_binary_like:
            new_lb, new_ub = 0.0, 1.0
        # Si entera, expande un poco
        elif vtype == GRB.INTEGER:
            new_lb = math.floor(new_lb - 2)
            new_ub = math.ceil(new_ub + 2)

        # No subir LB: mantener la libertad del diseño (x puede ser 0 si el modelo lo decide)
        # Solo reducimos UBs para bajar big-M.
        try:
            if (not math.isfinite(float(v.UB))) or float(v.UB) > new_ub:
                v.UB = float(new_ub)
        except Exception:
            pass
    m.update()


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
    # Usa versión estricta del embed: emula "x < thr" para ramas izquierdas
    try:
        bundle.attach_to_gurobi_strict(m, x_list_vars, y_log)
    except Exception:
        bundle.attach_to_gurobi(m, x_list_vars, y_log)

    # Calibración: y_log_cal = y_log + b0_offset
    try:
        b0 = float(getattr(bundle, 'b0_offset', 0.0) or 0.0)
    except Exception:
        b0 = 0.0
    y_log_cal = m.addVar(lb=-1e6, ub=1e6, name="y_log_cal")
    m.addConstr(y_log_cal == y_log + b0, name="YLOG_CAL")

    # map log -> price via PWL expm1 (tu entrenamiento usa log1p)
    y_price = m.addVar(lb=0.0, name="y_price")

    import numpy as np
    # rango razonable de log-precio, ajusta si quieres: p in [30k, 1.0M] -> log p en [10.3, 13.8]
    # Densificamos la PWL para reducir error (convexo): más puntos en [10.3,13.8]
    xs = np.linspace(10.3, 13.8, 401)
    ys = np.expm1(xs)

    y_log.LB = float(min(xs)); y_log.UB = float(max(xs))
    y_price.LB = float(min(ys)); y_price.UB = float(max(ys))
    m.update()


    m.addGenConstrPWL(y_log_cal, y_price, xs.tolist(), ys.tolist(), name="PWL_exp")

    return y_log_cal, y_price


# =============================== costos (LinExpr) =================================

def _build_cost_expr(m, x, ct):
    # Asegura que getVarByName vea variables recién creadas
    try:
        m.update()
    except Exception:
        pass
    cost = gp.LinExpr(0.0)
    m._cost_terms = []
    dbg_counts = {"roof":0,"ext1":0,"ext2":0,"heat":0,"elec":0,"paved":0,"found":0,"areas":0}

    def add(label, coef, var):
        if var is None: 
            return
        coef = float(coef)
        if abs(coef) < 1e-12:
            return
        m._cost_terms.append((label, coef, var))
        nonlocal cost
        cost += coef * var

    def add_if_missing(label, coef, var):
        if var is None:
            return
        try:
            if any(lab == label for (lab, _, _) in m._cost_terms):
                return
        except Exception:
            pass
        add(label, coef, var)

    # Construcción base por ft2
    # Cobrar SOLO la parte no cubierta por costos específicos. Aquí consideramos:
    # - Remainders (partición por piso)
    # - Área "Other" (recintos genéricos) por piso
    # - Sótano sin terminar
    c_base = float(getattr(ct, "construction_cost", 230.0))
    rem1 = m.getVarByName("Remainder1")
    rem2 = m.getVarByName("Remainder2")
    if rem1 is not None:
        add("Remainder1 @construction", c_base, rem1)
    if rem2 is not None:
        add("Remainder2 @construction", c_base, rem2)
    aoth1 = m.getVarByName("AreaOther1")
    aoth2 = m.getVarByName("AreaOther2")
    if aoth1 is not None:
        add("AreaOther1 @construction", c_base, aoth1)
    if aoth2 is not None:
        add("AreaOther2 @construction", c_base, aoth2)
    bsmt_unf = m.getVarByName("BsmtUnfSF") or _safe_get(x, "Bsmt Unf SF")
    if bsmt_unf is not None:
        add("Bsmt Unf SF @construction", c_base, bsmt_unf)

    # Terminación de sótano acabado (usa tu clave exacta)
    c_fin = float(getattr(ct, "finish_basement_per_f2", 0.0))
    add("BsmtFin SF 1 @finish", c_fin, m.getVarByName("BsmtFinSF1"))
    add("BsmtFin SF 2 @finish", c_fin, m.getVarByName("BsmtFinSF2"))

    add("Wood Deck SF", float(getattr(ct, "wooddeck_cost", 0.0)), _safe_get(x, "Wood Deck SF"))
    add("Open Porch SF", float(getattr(ct, "openporch_cost", 0.0)), _safe_get(x, "Open Porch SF"))
    add("Enclosed Porch", float(getattr(ct, "enclosedporch_cost", 0.0)), _safe_get(x, "Enclosed Porch"))
    add("3Ssn Porch", float(getattr(ct, "threessnporch_cost", 0.0)), _safe_get(x, "3Ssn Porch"))
    add("Screen Porch", float(getattr(ct, "screenporch_cost", 0.0)), _safe_get(x, "Screen Porch"))
    add("Pool Area", float(getattr(ct, "pool_area_cost", 0.0)), _safe_get(x, "Pool Area"))

    # Garage: si tienes costo por ft2, usa esa clave; si no, se costea por acabado (abajo)
    if hasattr(ct, "garage_area_cost"):
        add("Garage Area", float(getattr(ct, "garage_area_cost", 0.0)), _safe_get(x, "Garage Area"))

    # Foundation por tipo si defines tabla per-ft2 (opcional)
    if hasattr(ct, "foundation_cost_per_sf"):
        for tag, unit in ct.foundation_cost_per_sf.items():
            add(f"FA {tag}", float(unit), m.getVarByName(f"FA__{tag}"))
            dbg_counts["found"] += 1

    # Techo: costo fijo por material (tu tabla roof_matl_fixed)
    if hasattr(ct, "roof_matl_fixed"):
        for mm, lump in ct.roof_matl_fixed.items():
            v = m.getVarByName(f"RoofMatl__{mm}")
            add(f"RoofMatl {mm}", float(lump), v)
            dbg_counts["roof"] += 1
    # Techo: costo por estilo (opcional, si defines ct.roof_style_costs)
    if hasattr(ct, "roof_style_costs"):
        for s, unit in getattr(ct, "roof_style_costs", {}).items():
            v = m.getVarByName(f"RoofStyle__{s}")
            add(f"RoofStyle {s}", float(unit), v)

    # Heating: costo por tipo seleccionado (lumpsum)
    if hasattr(ct, "heating_type_costs"):
        for h, unit in ct.heating_type_costs.items():
            v = m.getVarByName(f"Heating__{h}")
            if v is not None:
                add(f"Heating {h}", float(unit), v)
                dbg_counts["heat"] += 1

    # Central Air: costo de instalación si 'Y'
    if hasattr(ct, "central_air_install"):
        v = m.getVarByName("CentralAir__Y")
        if v is not None:
            add("CentralAir Install", float(getattr(ct, "central_air_install", 0.0)), v)

    # Electrical: costo por tipo
    if hasattr(ct, "electrical_type_costs"):
        for e, unit in ct.electrical_type_costs.items():
            v = m.getVarByName(f"Electrical__{e}")
            if v is not None:
                add(f"Electrical {e}", float(unit), v)
                dbg_counts["elec"] += 1

    # Paved Drive: costo por categoría
    if hasattr(ct, "paved_drive_costs"):
        for d, unit in ct.paved_drive_costs.items():
            v = m.getVarByName(f"PavedDrive__{d}")
            if v is not None:
                add(f"PavedDrive {d}", float(unit), v)
                dbg_counts["paved"] += 1

    # Mampostería (Mas Vnr): costo por ft2 según tipo, multiplicado por área con MvProd
    if hasattr(ct, "mas_vnr_costs_sqft"):
        for t, unit in ct.mas_vnr_costs_sqft.items():
            mv = m.getVarByName(f"MvProd__{t}")
            if mv is not None:
                add(f"MasVnr {t}", float(unit), mv)

    # Exterior 1st: costo fijo por material seleccionado (lumpsum por frente en tu tabla)
    if hasattr(ct, "exterior_matl_lumpsum"):
        for e1, lump in ct.exterior_matl_lumpsum.items():
            v = m.getVarByName(f"Exterior1st__{e1}")
            add(f"Exterior1st {e1}", float(lump), v)
            dbg_counts["ext1"] += 1

    # Exterior 2nd: costo fijo por material seleccionado
    if hasattr(ct, "exterior_matl_lumpsum"):
        for e2, lump in ct.exterior_matl_lumpsum.items():
            v2 = m.getVarByName(f"Exterior2nd__{e2}")
            add(f"Exterior2nd {e2}", float(lump), v2)
            dbg_counts["ext2"] += 1

    # Garage finish: costo fijo por acabado
    if hasattr(ct, "garage_finish_costs_sqft"):
        name_map = {"NA": "No aplica", "Fin": "Fin", "RFn": "RFn", "Unf": "Unf"}
        for gf_code, label in name_map.items():
            v = m.getVarByName(f"GarageFinish__{gf_code}")
            if v is not None:
                add(f"GarageFinish {gf_code}", float(ct.garage_finish_costs_sqft.get(label, 0.0)), v)

    # Constante 1 por si falta Floor1 (fallback seguro para lumpsum)
    ONE = m.getVarByName("CONST_ONE") or m.addVar(lb=1.0, ub=1.0, name="CONST_ONE")
    gate_floor1 = m.getVarByName("Floor1") or ONE
    # Utilities (lumpsum por categoría) – usamos AllPub (fijo a 3)
    if hasattr(ct, "utilities_costs"):
        add("Utilities AllPub", float(ct.util_cost("AllPub")), gate_floor1)

    # Kitchen Qual (paquete): desactivado por ahora (se costea via area de cocina)
    # if hasattr(ct, "kitchenQual_upgrade_EX"):
    #     add("KitchenQual EX", float(getattr(ct, "kitchenQual_upgrade_EX", 0.0)), gate_floor1)

    # Calidad/Condición exterior (fijas a Ex) – costos lumpsum anclados a Floor1
    # Exterior Qual/Cond (lumpsum): desactivado para evitar doble conteo; el exterior se costea por material
    # if hasattr(ct, "exter_qual_costs"):
    #     add("ExterQual EX", float(ct.exter_qual_cost("Ex")), gate_floor1)
    # if hasattr(ct, "exter_cond_costs"):
    #     add("ExterCond EX", float(ct.exter_cond_cost("Ex")), gate_floor1)

    has_reja = m.getVarByName("HasReja")
    if has_reja is not None:
        add("Reja lineal",
            float(getattr(ct, "fence_build_cost_per_ft", 0.0)) * float(getattr(ct, "lot_frontage_ft", 0.0)),
            has_reja)

    # Fireplace: costo por chimenea con calidad excelente (si así lo decides)
    if hasattr(ct, "fireplace_costs"):
        fp_cnt = m.getVarByName("Fireplaces") or _safe_get(x, "Fireplaces")
        try:
            unit_fp = float(ct.fireplace_cost("Ex"))
            if fp_cnt is not None and unit_fp:
                add("Fireplace EX", unit_fp, fp_cnt)
        except Exception:
            pass

    # 11.xx premiums y costos por área adicionales
    try:
        gate_floor1 = m.getVarByName("Floor1") or m.addVar(lb=1.0, ub=1.0, name="CONST_ONE_PREM")
    except Exception:
        gate_floor1 = None

    # [CONSTRUCCION] Premiums de Kitchen/Exterior apagados (se usan en REMODEL)
    # if hasattr(ct, "kitchenQual_upgrade_EX"):
    #     kcnt = m.getVarByName("Kitchen") or gate_floor1
    #     if kcnt is not None:
    #         add("KitchenQual EX", float(getattr(ct, "kitchenQual_upgrade_EX", 0.0)), kcnt)

    # Exterior premium (calidad y condición)
    # if hasattr(ct, "exter_qual_costs") and gate_floor1 is not None:
    #     add("ExterQual EX", float(ct.exter_qual_cost("Ex")), gate_floor1)
    # if hasattr(ct, "exter_cond_costs") and gate_floor1 is not None:
    #     add("ExterCond EX", float(ct.exter_cond_cost("Ex")), gate_floor1)

    # Costos por área de ambientes
    for nm, attr in (("AreaKitchen", "kitchen_area_cost"),
                     ("AreaFullBath", "fullbath_area_cost"),
                     ("AreaHalfBath", "halfbath_area_cost"),
                     ("AreaBedroom",  "bedroom_area_cost")):
        v = m.getVarByName(nm)
        if v is not None:
            add(nm, float(getattr(ct, attr, 0.0)), v)
            dbg_counts["areas"] += 1

    # MiscFeature (lumpsum por categoría)
    if hasattr(ct, "misc_feature_costs"):
        for k, unit in ct.misc_feature_costs.items():
            v = m.getVarByName(f"MiscFeature__{k}")
            if v is not None:
                add(f"MiscFeature {k}", float(unit), v)

    # Redundancy guard: si por cualquier motivo algún bloque anterior no anexó términos,
    # refuerza aquí con add_if_missing. (Evita duplicar si ya están.)
    try:
        # Heating / Electrical / PavedDrive
        for h, unit in getattr(ct, "heating_type_costs", {}).items():
            add_if_missing(f"Heating {h}", float(unit), m.getVarByName(f"Heating__{h}"))
        for e, unit in getattr(ct, "electrical_type_costs", {}).items():
            add_if_missing(f"Electrical {e}", float(unit), m.getVarByName(f"Electrical__{e}"))
        for d, unit in getattr(ct, "paved_drive_costs", {}).items():
            add_if_missing(f"PavedDrive {d}", float(unit), m.getVarByName(f"PavedDrive__{d}"))
        # Roof / Exterior
        for mm, lump in getattr(ct, "roof_matl_fixed", {}).items():
            add_if_missing(f"RoofMatl {mm}", float(lump), m.getVarByName(f"RoofMatl__{mm}"))
        for e1, lump in getattr(ct, "exterior_matl_lumpsum", {}).items():
            add_if_missing(f"Exterior1st {e1}", float(lump), m.getVarByName(f"Exterior1st__{e1}"))
            add_if_missing(f"Exterior2nd {e1}", float(lump), m.getVarByName(f"Exterior2nd__{e1}"))
        # Foundation per tipo (por área)
        for f, unit in getattr(ct, "foundation_cost_per_sf", {}).items():
            add_if_missing(f"FA {f}", float(unit), m.getVarByName(f"FA__{f}"))
        # Áreas de ambientes
        for nm, attr in (("AreaKitchen", "kitchen_area_cost"),
                         ("AreaFullBath", "fullbath_area_cost"),
                         ("AreaHalfBath", "halfbath_area_cost"),
                         ("AreaBedroom",  "bedroom_area_cost")):
            v = m.getVarByName(nm)
            add_if_missing(nm, float(getattr(ct, attr, 0.0)), v)
        # Basement finished areas
        c_fin = float(getattr(ct, "finish_basement_per_f2", 0.0))
        add_if_missing("BsmtFin SF 1 @finish", c_fin, m.getVarByName("BsmtFinSF1"))
        add_if_missing("BsmtFin SF 2 @finish", c_fin, m.getVarByName("BsmtFinSF2"))
    except Exception:
        pass

    # debug opcional desactivado

    return cost


# =============================== modelo principal ================================

def build_mip_embed(*, base_row, budget: float, ct, bundle: XGBBundle) -> gp.Model:
    m = gp.Model("construction_embed")

    lot_area = float(base_row.get("Lot Area", getattr(ct, "lot_area", 7000)))
    lot_frontage = float(base_row.get("Lot Frontage", getattr(ct, "lot_frontage_ft", 60)))

    # features X en el orden del XGB
    feat_order, x = _build_x_inputs(m, bundle, base_row, lot_area)

        # fijar LOT AREA al valor de entrada
    key_lot = _find_x_key(x, "Lot Area")
    if key_lot is not None:
        m.addConstr(x[key_lot] == lot_area, name="link__lot_area_fix")

    # fijar NEIGHBORHOOD one-hots segun parametro/base_row
    # busca todas las columnas Neighborhood_*
    neigh_token = str(base_row.get("neigh_arg", "") or getattr(ct, "neigh_arg", "")).strip()
    # Si no llega neigh_arg, usa el nombre en base_row["Neighborhood"] como token
    if not neigh_token:
        try:
            neigh_token = str(base_row.get("Neighborhood", "")).strip()
        except Exception:
            neigh_token = ""
    # normaliza ejemplos: "NAmes" -> "NWAmes" si ya llega mapeado en base_row, se usara eso
    # si en base_row viene set con 1 una de ellas, usamos esa prioridad
    picked = None
    for col in feat_order:
        if col.startswith("Neighborhood_") and col in x:
            if (col in base_row.index) and (float(base_row[col]) == 1.0):
                picked = col
                break
    if picked is None and neigh_token:
        # 1) intento exacto Neighborhood_<token>
        exact = f"Neighborhood_{neigh_token}"
        if exact in x:
            picked = exact
        else:
            # 2) intento por contains, case-insensitive
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

    # === fijar OHE de MS SubClass (dato base, no decisión) ===
    try:
        ms_cols = [c for c in feat_order if c.startswith("MS SubClass_") and c in x]
        if ms_cols:
            _enforce_exclusive_ohe(m, x, feat_order, "MS SubClass")
    except Exception:
        pass

    # === fijar OHE de Fence (por defecto: No aplica) para coherencia del XGB ===
    try:
        fence_cols = [c for c in feat_order if c.startswith("Fence_") and c in x]
        if fence_cols:
            for c in fence_cols:
                m.addConstr(x[c] == (1.0 if c.endswith("No aplica") else 0.0), name=f"link__fence__fix__{c}")
    except Exception:
        pass

    # === fijar columnas que son parametros del proyecto (nueva construcción) ===
    def _fix_const(col_name: str, value: float):
        # Fija SOLO si existe variable exacta; evita emparejar OHE por contiene
        if col_name in x:
            try:
                m.addConstr(x[col_name] == float(value), name=f"fix__{_norm(col_name)}")
            except Exception:
                pass

    def _fix_either_numeric_or_ohe(col_name: str, value: float | int | str):
        # Si existe columna numérica exacta, fíjala; si no, intenta fijar grupo OHE col_name_* al valor
        if col_name in x:
            _fix_const(col_name, value)  # exacto
        else:
            try:
                _fix_ohe_group(m, x, feat_order, col_name, str(value))
            except Exception:
                pass

    # Años y meses (construcción y venta en 2025, sin remodel previo)
    _fix_const("Year Built", getattr(ct, "build_year", 2025))
    _fix_const("Year Remod/Add", getattr(ct, "remod_year", 2025))
    _fix_either_numeric_or_ohe("Year Sold", getattr(ct, "sale_year", 2025))
    _fix_either_numeric_or_ohe("Month Sold", getattr(ct, "sale_month", 6))
    _fix_either_numeric_or_ohe("Yr Sold", getattr(ct, "sale_year", 2025))
    _fix_either_numeric_or_ohe("Mo Sold", getattr(ct, "sale_month", 6))

    # Fijar categorías del terreno y entorno a valores típicos (no decisiones de construcción)
    try:
        _fix_ohe_group(m, x, feat_order, "MS Zoning", "RL")
        _fix_ohe_group(m, x, feat_order, "Street", "Pave")
        _fix_ohe_group(m, x, feat_order, "Alley", "No aplica")
        _fix_ohe_group(m, x, feat_order, "Lot Shape", "Reg")
        _fix_ohe_group(m, x, feat_order, "LandContour", "Lvl")
        _fix_ohe_group(m, x, feat_order, "LotConfig", "Inside")
        _fix_ohe_group(m, x, feat_order, "LandSlope", "Gtl")
        _fix_ohe_group(m, x, feat_order, "Condition 1", "Norm")
        _fix_ohe_group(m, x, feat_order, "Condition 2", "Norm")
        _fix_ohe_group(m, x, feat_order, "Functional", "Typ")
        _fix_ohe_group(m, x, feat_order, "Sale Type", "WD")
        _fix_ohe_group(m, x, feat_order, "Sale Condition", "Normal")
        # Si el proyecto es vivienda unifamiliar por defecto, fija Bldg Type
        try:
            bldg_pick = str(base_row.get("BldgType", "1Fam"))
            _fix_ohe_group(m, x, feat_order, "Bldg Type", bldg_pick)
        except Exception:
            pass
        # Exclusividades en X (sum-to-one)
        for g in [
            "MS Zoning","Street","Alley","Lot Shape","LandContour","LotConfig","LandSlope",
            "Condition 1","Condition 2","Functional","Sale Type","Sale Condition","Neighborhood",
        ]:
            _enforce_exclusive_ohe(m, x, feat_order, g)
    except Exception:
        pass

    # Calidades: sólo ajustamos límite inferior (Average hacia arriba) en las que tienen costo asociado
    # búsqueda de columnas con costos declarados en ct
    qual_cost_cols = []
    if hasattr(ct, "heating_qc_costs"):
        qual_cost_cols.append("Heating QC")
    if hasattr(ct, "fireplace_costs"):
        qual_cost_cols.append("Fireplace Qu")
    if hasattr(ct, "poolqc_costs"):
        qual_cost_cols.append("Pool QC")
    if hasattr(ct, "bsmt_cond_upgrade_costs"):
        qual_cost_cols.append("Bsmt Cond")
    MIN_Q = float(getattr(ct, "min_quality_level", 2.0))  # TA en escala -1..4
    for qcol in qual_cost_cols:
        v = _safe_get(x, qcol)
        if v is not None:
            try:
                v.LB = max(getattr(v, "LB", -1.0), MIN_Q)
                v.UB = max(getattr(v, "UB", 4.0), 4.0)
            except Exception:
                pass

    # Utilities: AllPub (ordinal 3 según entrenamiento)
    _fix_const("Utilities", 3)

    # Sin terminación de baja calidad
    _fix_const("Low Qual Fin SF", 0.0)


    # === variables de niveles por piso ===
    Floor1 = m.addVar(vtype=GRB.BINARY, name="Floor1")
    Floor2 = m.addVar(vtype=GRB.BINARY, name="Floor2")
    IsOneStory = m.addVar(vtype=GRB.BINARY, name="IsOneStory")
    IsTwoStory = m.addVar(vtype=GRB.BINARY, name="IsTwoStory")
    m.addConstr(Floor1 == 1, name="7.4__first_floor_always")
    m.addConstr(Floor2 <= Floor1, name="7.4__second_floor_if_first")
    m.addConstr(IsOneStory + IsTwoStory == 1, name="7.4__one_or_two_story")
    m.addConstr(Floor2 == IsTwoStory, name="7.4__two_story_link")

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

    floor2_caps = getattr(ct, "floor2_count_caps", {})
    bed2_cap   = float(floor2_caps.get("bedrooms", getattr(ct, "bedroom2_cap", 8)))
    fb2_cap    = float(floor2_caps.get("fullbath", getattr(ct, "fullbath2_cap", 4)))
    hb2_cap    = float(floor2_caps.get("halfbath", getattr(ct, "halfbath2_cap", 3)))
    kit2_cap   = float(floor2_caps.get("kitchen", getattr(ct, "kitchen2_cap", 2)))
    othrm2_cap = float(floor2_caps.get("otherrooms", getattr(ct, "otherrooms2_cap", 8)))
    m.addConstr(Bedroom2  <= bed2_cap   * Floor2, name="7.21.2__bed2_onoff")
    m.addConstr(FullBath2 <= fb2_cap    * Floor2, name="7.21.2__fullbath2_onoff")
    m.addConstr(HalfBath2 <= hb2_cap    * Floor2, name="7.21.2__halfbath2_onoff")
    m.addConstr(Kitchen2  <= kit2_cap   * Floor2, name="7.21.2__kitchen2_onoff")
    m.addConstr(OtherRooms2 <= othrm2_cap * Floor2, name="7.22.3__other_cnt_2nd_if_on")

    # agregados
    AreaKitchen   = m.addVar(lb=0.0, name="AreaKitchen")
    AreaFullBath  = m.addVar(lb=0.0, name="AreaFullBath")
    AreaHalfBath  = m.addVar(lb=0.0, name="AreaHalfBath")
    AreaBedroom   = m.addVar(lb=0.0, name="AreaBedroom")
    AreaOther     = m.addVar(lb=0.0, name="AreaOther")
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
    porch_caps = {
        "open":   float(getattr(ct, "u_open_ratio", 0.10)) * lot_area,
        "encl":   float(getattr(ct, "u_encl_ratio", 0.10)) * lot_area,
        "three":  float(getattr(ct, "u_3ssn_ratio", 0.10)) * lot_area,
        "screen": float(getattr(ct, "u_screen_ratio", 0.05)) * lot_area,
    }

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
    # 11.3 Consistencias por ambiente (áreas agregadas)
    m.addConstr(AreaKitchen  == AreaKitchen1  + AreaKitchen2,  name="11.3__AKitchen")
    m.addConstr(AreaBedroom  == AreaBedroom1  + AreaBedroom2,  name="11.3__ABedroom")
    m.addConstr(AreaOther    == AreaOther1    + AreaOther2,    name="7.22.1__AreaOther_sum")
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
    # 11.11 relaciones de baños: HalfBath <= FullBath
    m.addConstr(HalfBath <= FullBath, name="11.11__half_leq_full")
    # Binaria de presencia de cocina para gatear costo de calidad
    HasKitchen = m.addVar(vtype=GRB.BINARY, name="HasKitchen")
    UK = 2  # cota segura de cocinas totales (según K_max)
    m.addConstr(HasKitchen <= Kitchen, name="11.xx__has_kitch_leq_cnt")
    m.addConstr(HasKitchen - (Kitchen / UK) >= 0, name="11.xx__has_kitch_ge_frac")

    # 7.3 caps por tipo de edificio (opcional, solo si ct.bldg_caps existe)
    bldg_opts = ["1Fam","TwnhsE","TwnhsI","Duplex","2FmCon"]
    Bldg = _one_hot(m, "BldgType", bldg_opts)
    _tie_one_hot_to_x(m, x, "Bldg Type", Bldg)
    caps = getattr(ct, "bldg_caps", None)
    if caps is None:
        Bed_max = F_max = H_max = K_max = Ch_max = None
    else:
        Bed_max = caps.get("bedrooms_max", {})
        F_max   = caps.get("fullbath_max", {})
        H_max   = caps.get("halfbath_max", {})
        K_max   = caps.get("kitchen_max", {})
        Ch_max  = caps.get("fireplaces_max", {})
    Bedrooms = m.addVar(vtype=GRB.INTEGER, lb=0, name="Bedrooms")
    m.addConstr(Bedrooms == Bedroom1 + Bedroom2, name="7.3.1__bedrooms_sum")
    if Bed_max:
        m.addConstr(Bedrooms <= gp.quicksum(float(Bed_max.get(b, 1e6))*Bldg[b] for b in bldg_opts), name="7.3.1__bedrooms_cap")
    if F_max:
        m.addConstr(FullBath <= gp.quicksum(float(F_max.get(b, 1e6))*Bldg[b] for b in bldg_opts), name="7.3.2__fullbath_cap")
    if H_max:
        m.addConstr(HalfBath <= gp.quicksum(float(H_max.get(b, 1e6))*Bldg[b] for b in bldg_opts), name="7.3.3__halfbath_cap")
    if K_max:
        m.addConstr(Kitchen  <= gp.quicksum(float(K_max.get(b, 1e6))*Bldg[b] for b in bldg_opts), name="7.3.4__kitchen_cap")
    else:
        # Regla opcional por tamaño: permitir más cocinas en 1Fam si la casa es muy grande
        thr = getattr(ct, 'kitchen_by_area_thresholds', None)  # p.ej. [3200, 5000]
        if thr and xgr is not None and '1Fam' in Bldg:
            z = []
            Ugr = 0.8 * float(lot_area)
            for i, t in enumerate(thr):
                zi = m.addVar(vtype=GRB.BINARY, name=f"zKitchen_{i+2}")
                # xgr >= t - M*(1-zi)
                m.addConstr(xgr >= float(t) - Ugr * (1 - zi), name=f"11.xx__kitch_allow_if_area_{i+2}")
                z.append(zi)
            # Kitchen <= 1 + sum(z) si 1Fam; libre (<=10) si no 1Fam
            m.addConstr(Kitchen <= 1 + gp.quicksum(z) + 10.0 * (1 - Bldg['1Fam']), name="7.3.4__kitchen_by_area_1fam")
        else:
            # Guard-rail por defecto: Kitchen ≤ 1 en 1Fam (se puede desactivar en ct)
            if bool(getattr(ct, 'enforce_single_kitchen_1fam', True)):
                try:
                    m.addConstr(Kitchen <= 1.0 * Bldg['1Fam'] + 10.0 * (1 - Bldg['1Fam']), name="7.3.4__kitchen_guard_1fam")
                except Exception:
                    pass
    Fireplaces = _safe_get(x, "Fireplaces") or m.addVar(vtype=GRB.INTEGER, lb=0, name="Fireplaces")
    if Ch_max:
        m.addConstr(Fireplaces <= gp.quicksum(float(Ch_max.get(b, 1e6))*Bldg[b] for b in bldg_opts), name="7.3.5__fireplaces_cap")
    _link_if_exists(m, x, "Bedroom AbvGr", Bedrooms)

    # Ajusta mínimo de 1er piso por tipo de edificio si hay mapa en ct
    try:
        eps_map = getattr(ct, "eps1_by_bldg", None)
    except Exception:
        eps_map = None
    if eps_map and x1 is not None:
        try:
            m.addConstr(
                x1 >= gp.quicksum(float(eps_map.get(b, 450.0)) * Bldg[b] for b in bldg_opts) * Floor1,
                name="7.1.8__1st_min_by_bldg",
            )
        except Exception:
            pass

    # Fireplace quality: -1 si no hay chimenea
    v_fq = _safe_get(x, "Fireplace Qu")
    if v_fq is not None:
        HasFire = m.addVar(vtype=GRB.BINARY, name="HasFireplace")
        Ufire = 4.0
        m.addConstr(HasFire >= Fireplaces / 4.0, name="11.xx__has_fire_ge_frac")
        m.addConstr(Fireplaces <= 4.0 * HasFire, name="11.xx__fire_leq_cap")
        try:
            m.addGenConstrIndicator(HasFire, False, v_fq == -1.0, name="11.xx__fireplace_qu_eq_na_if_none")
        except Exception:
            pass

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

    # Garage quality/condition: 4 si hay garage, -1 si NO aplica
    for qcol in ["Garage Qual", "Garage Cond"]:
        vq = _safe_get(x, qcol)
        if vq is not None:
            try:
                m.addConstr(vq == 4.0 - 5.0 * GT["NA"], name=f"11.xx__{_norm(qcol)}_eq_4_or_na")
            except Exception:
                pass

    # Amarrar OHE del XGB a one-hot interno para Garage Type/Finish (evita múltiple 1s en CSV)
    for lab, var in {
        "Garage Type_2Types": GT["2Types"],
        "Garage Type_Attchd": GT["Attchd"],
        "Garage Type_Basment": GT["Basment"],
        "Garage Type_BuiltIn": GT["BuiltIn"],
        "Garage Type_CarPort": GT["CarPort"],
        "Garage Type_Detchd": GT["Detchd"],
        "Garage Type_No aplica": GT["NA"],
    }.items():
        if lab in x:
            m.addConstr(x[lab] == var, name=f"link__x__{_norm(lab)}")

    for lab, var in {
        "Garage Finish_Fin": GF["Fin"],
        "Garage Finish_RFn": GF["RFn"],
        "Garage Finish_Unf": GF["Unf"],
        "Garage Finish_No aplica": GF["NA"],
    }.items():
        if lab in x:
            m.addConstr(x[lab] == var, name=f"link__x__{_norm(lab)}")

    # Garage Yr Built: igual al Year Built si hay garage; 0 si GarageType=NA
    try:
        gyb = _safe_get(x, "Garage Yr Built") or _safe_get(x, "Garage Yr Blt") or _safe_get(x, "GarageYrBlt")
        yb  = _safe_get(x, "Year Built")
        if gyb is not None and yb is not None:
            Uyr = 2100.0
            # Si no hay garage (GT['NA']=1) ⇒ gyb = 0
            m.addConstr(gyb <= Uyr * (1 - GT["NA"]), name="11.xx__gar_yr_zero_if_na")
            # Si hay garage ⇒ gyb = yb (con big-M)
            m.addConstr(gyb >= yb - Uyr * GT["NA"], name="11.xx__gar_yr_ge_yb_if_has")
            m.addConstr(gyb <= yb + Uyr * GT["NA"], name="11.xx__gar_yr_le_yb_if_has")
    except Exception:
        pass

    # 11.1 HouseStyle exclusividad + relación con pisos
    hs_opts = ["1Story","2Story"]
    HS = _one_hot(m, "HouseStyle", hs_opts)
    m.addConstr(Floor2 == HS["2Story"], name="11.1__floor2_only_if_2story")
    m.addConstr(IsOneStory == HS["1Story"], name="7.4__hs_one_story")
    m.addConstr(IsTwoStory == HS["2Story"], name="7.4__hs_two_story")

    # MS SubClass coherente con HouseStyle (1Story->20, 2Story->60)
    try:
        ms20 = x.get("MS SubClass_20")
        ms60 = x.get("MS SubClass_60")
        if ms20 is not None:
            m.addConstr(ms20 == HS.get("1Story", 0), name="link__mssubclass__20_if_1story")
        if ms60 is not None:
            m.addConstr(ms60 == HS.get("2Story", 0), name="link__mssubclass__60_if_2story")
        # apaga el resto de subclasses si existen
        for c in [c for c in x.keys() if c.startswith("MS SubClass_") and c not in ("MS SubClass_20","MS SubClass_60")]:
            try:
                m.addConstr(x[c] == 0.0, name=f"link__mssubclass__off__{c}")
            except Exception:
                pass
    except Exception:
        pass

    # 7.xx Exclusividades adicionales (Heating / Central Air / Electrical / PavedDrive)
    heat_opts = ["Floor","GasA","GasW","Grav","OthW","Wall"]
    HEAT = _one_hot(m, "Heating", heat_opts)
    _tie_one_hot_to_x(m, x, "Heating", HEAT)

    air_opts = ["N","Y"]
    CA = _one_hot(m, "CentralAir", air_opts)
    _tie_one_hot_to_x(m, x, "Central Air", CA)

    elect_opts = ["SBrkr","FuseA","FuseF","FuseP","Mix"]
    EL = _one_hot(m, "Electrical", elect_opts)
    _tie_one_hot_to_x(m, x, "Electrical", EL)

    pd_opts = ["Y","P","N"]
    PD = _one_hot(m, "PavedDrive", pd_opts)
    _tie_one_hot_to_x(m, x, "Paved Drive", PD)

    # MiscFeature: asegurar exclusividad y ligar Misc Val
    misc_opts = ["Elev","Gar2","Othr","Shed","TenC","No aplica"]
    MF = _one_hot(m, "MiscFeature", misc_opts)
    _tie_one_hot_to_x(m, x, "Misc Feature", MF)
    misc_val = _safe_get(x, "Misc Val")
    if misc_val is not None and MF.get("No aplica") is not None:
        try:
            m.addGenConstrIndicator(MF["No aplica"], True, misc_val == 0.0, name="11.xx__miscval_zero_if_na")
        except Exception:
            # fallback: misc_val <= 0 si No aplica (y LB>=0) ⇒ 0
            m.addConstr(misc_val <= 1e-6 * (1 - MF["No aplica"]), name="11.xx__miscval_leq_zero_if_na")

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
            # Indicadores más fuertes: y=1 => z=PlanRoofArea; y=0 => z=0
            try:
                m.addGenConstrIndicator(y, True,  z == PlanRoofArea, name=f"7.7.5__z_eq_plan_if_y__{s}_{mm}")
                m.addGenConstrIndicator(y, False, z == 0.0,          name=f"7.7.5__z_eq_0_if_noty__{s}_{mm}")
            except Exception:
                pass
    m.addConstr(gp.quicksum(Y.values()) == 1, name="7.7.4__one_y")

    gamma = getattr(ct, "gamma", {})
    ActualRoofArea = m.addVar(lb=0.0, name="ActualRoofArea")
    m.addConstr(
        ActualRoofArea == gp.quicksum(float(gamma.get((s, mm), 1.10)) * Z[(s, mm)] for s in S for mm in M),
        name="7.7.3__actual_roof_area",
    )

    # 7.8, 7.9 caps de areas total y por parte
    tbsmt = _safe_get(x, "Total Bsmt SF")
    TotArea = None
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

    # 7.10 relacion banos vs dormitorios (tope razonable) y mínimos básicos
    m.addConstr(3 * FullBath <= 2 * Bedrooms, name="7.10__baths_vs_bed")
    m.addConstr(FullBath >= 1, name="11.xx__fullbath_min1")
    m.addConstr(Bedrooms >= 1, name="11.xx__bedrooms_min1")

    # 7.11 piscina
    if all(v is not None for v in [x1,xgar,xdeck,xopen,xencl,xscreen,x3ssn,xpool]):
        m.addConstr(xpool <= (lot_area - x1 - xgar - xdeck - xopen - xencl - xscreen - x3ssn) * HasPool, name="7.11.1__pool_space")
        m.addConstr(xpool <= 0.1 * lot_area * HasPool, name="7.11.1__pool_max")
        m.addConstr(xpool >= 160 * HasPool, name="7.11.1__pool_min")
        m.addConstr(xpool >= 0, name="7.11.1__pool_nonneg")
        # Pool QC = -1 si no hay piscina; si hay, libre dentro de sus bounds (-1..4)
        v_pqc = _safe_get(x, "Pool QC")
        if v_pqc is not None:
            try:
                m.addGenConstrIndicator(HasPool, False, v_pqc == -1.0, name="11.xx__poolqc_eq_na_if_no_pool")
            except Exception:
                pass

    # 7.12 porches (forzamos identidad TotalPorch = suma componentes)
    if all(v is not None for v in [xopen,xencl,xscreen,x3ssn]):
        m.addConstr(xpor_tot == xopen + xencl + xscreen + x3ssn, name="7.12.1__porch_sum")
        m.addConstr(xpor_tot <= 0.25 * lot_area, name="7.12.1__porch_cap_total")
        if x1 is not None:
            m.addConstr(xpor_tot <= x1, name="7.12.1__porch_leq_1st")
        m.addConstr(xopen  >= 40 * HasOpenPorch, name="7.12.2__open_min")
        m.addConstr(xopen  <= porch_caps["open"]   * HasOpenPorch, name="7.12.2__open_max")
        m.addConstr(xencl  >= 60 * HasEnclosedPorch, name="7.12.2__encl_min")
        m.addConstr(xencl  <= porch_caps["encl"]   * HasEnclosedPorch, name="7.12.2__encl_max")
        m.addConstr(xscreen>= 40 * HasScreenPorch, name="7.12.2__screen_min")
        m.addConstr(xscreen<= porch_caps["screen"] * HasScreenPorch, name="7.12.2__screen_max")
        m.addConstr(x3ssn  >= 80 * Has3SsnPorch, name="7.12.2__3ssn_min")
        m.addConstr(x3ssn  <= porch_caps["three"] * Has3SsnPorch, name="7.12.2__3ssn_max")
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
        # ===== PATCH B start: basement links =====
        # crea variable para Bsmt Unf si no existe en x
        BsmtUnfSF = _safe_get(x, "Bsmt Unf SF") or m.addVar(lb=0.0, name="BsmtUnfSF")
        m.addConstr(BsmtUnfSF == 0.0, name="7.0__bsmt_unf_zero")

        # enlaza features del XGB a tus variables
        _link_if_exists(m, x, "Bsmt Unf SF", BsmtUnfSF)
        _link_if_exists(m, x, "BsmtFin SF 1", BsmtFinSF1)
        _link_if_exists(m, x, "BsmtFin SF 2", BsmtFinSF2)

        # reemplaza la suma anterior por esta con Unf
        if tbsmt is not None:
            m.addConstr(tbsmt == BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF, name="7.8__bsmt_parts_sum")
            _link_if_exists(m, x, "Total Bsmt SF", tbsmt)
        # ===== PATCH B end =====

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

    # ===== PATCH B extra: TotRms y min areas sobre rasante =====
    # TotRms AbvGrd = dormitorios + cocinas + otras piezas (ajusta OtherRooms si ya lo tienes)
    TotRms = _safe_get(x, "TotRms AbvGrd") or m.addVar(lb=0.0, name="TotRms")
    m.addConstr(TotRms == Bedrooms + FullBath + HalfBath + OtherRooms, name="7.22.1__totrms_def")
    _link_if_exists(m, x, "TotRms AbvGrd", TotRms)

    # min areas por cuenta para evitar cuartos fantasma
    # 11.20 Áreas mínimas por ambiente (alineado con PDF)
    m.addConstr(AreaFullBath >= 40 * FullBath,  name="11.20__min_area_full_bath")
    m.addConstr(AreaHalfBath >= 20 * HalfBath,  name="11.20__min_area_half_bath")
    m.addConstr(AreaKitchen  >= 75 * Kitchen,   name="11.20__min_area_kitchen")
    m.addConstr(AreaBedroom  >= 70 * Bedrooms,  name="11.20__min_area_bedroom")

    # la suma de areas funcionales no puede pasar el total de 1ro+2do piso
    if x1 is not None and x2 is not None:
        m.addConstr(
            AreaBedroom + AreaKitchen + AreaFullBath + AreaHalfBath + AreaOther1 + AreaOther2
            <= x1 + x2,
            name="7.xx__areas_leq_floors"
        )
    # ===== fin PATCH B extra =====


    # 7.15 exteriores 1st vs 2nd: flag mismo material
    SameMaterial = m.addVar(vtype=GRB.BINARY, name="SameMaterial")
    ext1_opts = ["VinylSd","MetalSd","Wd Sdng","HdBoard","Stucco","Plywood","CemntBd","BrkFace","BrkComm","WdShngl","AsbShng","Stone","ImStucc","AsphShn","CBlock"]
    ext2_opts = ext1_opts
    EXT1 = _one_hot(m, "Exterior1st", ext1_opts)
    EXT2 = _one_hot(m, "Exterior2nd", ext2_opts)
    _tie_one_hot_to_x(m, x, "Exterior 1st", EXT1)
    _tie_one_hot_to_x(m, x, "Exterior 2nd", EXT2)
    for e1 in ext1_opts:
        if e1 in EXT2:
            m.addConstr(SameMaterial >= EXT1[e1] + EXT2[e1] - 1, name=f"7.15__same_mat__{e1}")
            m.addConstr(EXT1[e1] == EXT2[e1], name=f"7.19__exterior_match__{e1}")

    # 7.16 enchapados
    mas_opts = ["BrkCmn","BrkFace","CBlock","None","Stone"]
    MV = _one_hot(m, "MasVnrType", mas_opts)
    _tie_one_hot_to_x(m, x, "Mas Vnr Type", MV)
    MasVnrArea = _safe_get(x, "Mas Vnr Area") or m.addVar(lb=0.0, name="MasVnrArea")
    # Umas = fmas * AreaExterior (usamos la misma variable 'AreaExterior1st')
    AreaExterior_var = m.getVarByName("AreaExterior1st") or m.addVar(lb=0.0, name="AreaExterior1st")
    fmas = float(getattr(ct, "fmas", 0.4))
    Umas = fmas * AreaExterior_var
    m.addConstr(MasVnrArea <= Umas, name="7.16.1__mas_cap")
    # Si MasVnrType == None → Ã¡rea 0
    if "None" in MV:
        m.addConstr(MasVnrArea <= Umas * (1 - MV["None"]), name="7.16.1__mas_zero_if_none")
    # Mínimo si se usa algún tipo distinto de None
    m.addConstr(MasVnrArea >= 20.0 * (1 - MV["None"]), name="7.16__mas_min_if_used")
    # Variables auxiliares MvProd_t = MasVnrArea si se usa tipo t; 0 si no
    MvProd = {}
    for t in mas_opts:
        v = m.addVar(lb=0.0, name=f"MvProd__{t}")
        MvProd[t] = v
        m.addConstr(v <= MasVnrArea, name=f"7.16__mv_leq_area__{t}")
        m.addConstr(v <= Umas * MV[t], name=f"7.16__mv_leq_u_type__{t}")
        m.addConstr(v >= MasVnrArea - Umas * (1 - MV[t]), name=f"7.16__mv_ge_area_u__{t}")
    if TotArea is not None:
        m.addConstr(MasVnrArea <= TotArea, name="7.16__mas_leq_total_area")
    elif any(v is not None for v in (x1, x2, tbsmt)):
        total_expr = gp.LinExpr()
        if x1 is not None:
            total_expr += x1
        if x2 is not None:
            total_expr += x2
        if tbsmt is not None:
            total_expr += tbsmt
        m.addConstr(MasVnrArea <= total_expr, name="7.16__mas_leq_total_area_fallback")

    # 7.20 fundacion
    f_opts = ["BrkTil","CBlock","PConc","Slab","Stone","Wood"]
    FND = _one_hot(m, "Foundation", f_opts)
    _tie_one_hot_to_x(m, x, "Foundation", FND)
    AreaFoundation = m.addVar(lb=0.0, name="AreaFoundation")
    if x1 is not None:
        m.addConstr(AreaFoundation == x1, name="7.20__AreaFoundation_eq_1st")
        # Realismo: sótano no excede la huella (o pequeña tolerancia),
        # y si hay exposición (Gd/Av/Mn) puede permitir un poco más.
        if tbsmt is not None:
            try:
                rho_base = float(getattr(ct, "basement_to_foundation_ratio_max", 1.0))
            except Exception:
                rho_base = 1.0
            try:
                rho_exp = float(getattr(ct, "basement_ratio_if_exposed", rho_base))
            except Exception:
                rho_exp = rho_base
            # Cota base SIEMPRE activa
            m.addConstr(tbsmt <= rho_base * AreaFoundation, name="7.18__bsmt_leq_found_base")
            # Cota adicional si hay exposición (relaja a rho_exp)
            EXP_Gd = m.getVarByName("BsmtExposure__Gd")
            EXP_Av = m.getVarByName("BsmtExposure__Av")
            EXP_Mn = m.getVarByName("BsmtExposure__Mn")
            if all(v is not None for v in [EXP_Gd, EXP_Av, EXP_Mn]):
                is_exposed = EXP_Gd + EXP_Av + EXP_Mn
                Ubig = 1e6
                m.addConstr(tbsmt <= rho_exp * AreaFoundation + Ubig * (1 - is_exposed), name="7.18__bsmt_leq_found_if_exposed")
    Ufound = 0.6 * lot_area
    for f in f_opts:
        FA = m.addVar(lb=0.0, name=f"FA__{f}")
        m.addConstr(FA <= AreaFoundation, name=f"7.20__fa_leq_area__{f}")
        m.addConstr(FA <= Ufound * FND[f], name=f"7.20__fa_leq_u_f__{f}")
        m.addConstr(FA >= AreaFoundation - Ufound * (1 - FND[f]), name=f"7.20__fa_ge_lin__{f}")
    # if Slab o Wood entonces NA en BsmtExposure (permite NA)
    m.addConstr(FND["Slab"] <= EXP["NA"], name="7.20__slab_implies_na")
    m.addConstr(FND["Wood"] <= EXP["NA"], name="7.20__wood_implies_na")
    m.addConstr(EXP["NA"] <= FND["Slab"] + FND["Wood"], name="7.20__na_implies_slab_or_wood")

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
    # Partición exacta de áreas por piso con remanentes no negativos
    if x1 is not None:
        Remainder1 = m.addVar(lb=0.0, name="Remainder1")
        m.addConstr(
            AreaBedroom1 + AreaKitchen1 + AreaHalfBath1 + AreaFullBath1 + AreaOther1 + Remainder1 == x1,
            name="7.21.1__partition_1st",
        )
    if x2 is not None:
        Remainder2 = m.addVar(lb=0.0, name="Remainder2")
        m.addConstr(
            AreaBedroom2 + AreaKitchen2 + AreaHalfBath2 + AreaFullBath2 + AreaOther2 + Remainder2 == x2,
            name="7.21.2__partition_2nd",
        )

    # 7.22 otros recintos
    m.addConstr(OtherRooms == OtherRooms1 + OtherRooms2, name="7.22.1__other_cnt_sum")
    m.addConstr(AreaOther1 >= 100 * OtherRooms1, name="7.22.4__other1_min")
    m.addConstr(AreaOther2 >= 100 * OtherRooms2, name="7.22.4__other2_min")
    m.addConstr(OtherRooms1 >= 1, name="7.22.2__other_min_on_1st")

    # 7.23 area exterior y perimetros (lineal con big-M)
    smin, smax = 20.0, 70.0

    P1 = m.addVar(lb=0.0, name="P1")
    P2 = m.addVar(lb=0.0, name="P2")
    AreaExterior = m.getVarByName("AreaExterior1st") or m.addVar(lb=0.0, name="AreaExterior1st")

    # bounds max de area por piso que ya tienes
    U1 = getattr(ct, "alpha1", 0.6) * lot_area   # ej 0.6 * lot_area
    U2 = getattr(ct, "alpha2", 0.5) * lot_area   # ej 0.5 * lot_area

    # upper bound seguro de perimetro por piso (cuando el piso existe)
    Uperim1 = 2 * ((U1 / smin) + smin)  # lineal en x1, constante en big-M
    Uperim2 = 2 * ((U2 / smin) + smin)

    if x1 is not None:
        # si hay 1er piso: P1 <= 2*(x1/smin + smin); si no hay: P1 <= 0
        m.addConstr(P1 <= 2*((x1 / smin) + smin) + Uperim1*(1 - Floor1), name="7.23__p1_ub_linear")
        m.addConstr(P1 <= Uperim1 * Floor1, name="7.23__p1_onoff")
        # LB seguro cuando hay 1er piso
        m.addConstr(P1 >= 4*smin * Floor1, name="7.23__p1_lb_linear")

    if x2 is not None:
        # si hay 2do piso: P2 <= 2*(x2/smin + smin); si no hay: P2 <= 0
        m.addConstr(P2 <= 2*((x2 / smin) + smin) + Uperim2*(1 - Floor2), name="7.23__p2_ub_linear")
        m.addConstr(P2 <= Uperim2 * Floor2, name="7.23__p2_onoff")
        # LB seguro cuando hay 2do piso
        m.addConstr(P2 >= 4*smin * Floor2, name="7.23__p2_lb_linear")

    # si hay segundo piso, exige P1 >= P2
    m.addConstr(P1 >= P2 - Uperim2*(1 - Floor2), name="7.23__p1_ge_p2_if_2nd")

    # area de fachada exterior
    Hext = getattr(ct, "Hext", 7.0)
    m.addConstr(AreaExterior == Hext * (P1 + P2), name="7.23.1__area_ext")

    # materiales de Exterior1st (usar el one-hot existente)
    ext1_opts = ["VinylSd","MetalSd","Wd Sdng","HdBoard","Stucco","Plywood","CemntBd","BrkFace","BrkComm","WdShngl","AsbShng","Stone","ImStucc","AsphShn","CBlock"]
    W = {}
    # Cota más apretada para W__e1: AreaExterior <= Hext*(Uperim1+Uperim2)
    Uext = Hext * (Uperim1 + Uperim2)

    # IMPORTANTE: necesitas tener creado antes:
    #   EXT1 = _one_hot(m, "Exterior1st", ext1_opts)
    #   _tie_one_hot_to_x(m, x, "Exterior 1st", EXT1)

    for e1 in ext1_opts:
        w = m.addVar(lb=0.0, name=f"W__{e1}")
        W[e1] = w
        b = EXT1[e1]  # no uses getVarByName, usa el one-hot
        m.addConstr(w <= AreaExterior, name=f"7.23.1__w_leq_area__{e1}")
        m.addConstr(w <= Uext * b, name=f"7.23.1__w_leq_u__{e1}")
        m.addConstr(w >= AreaExterior - Uext * (1 - b), name=f"7.23.1__w_ge_lin__{e1}")
        # Indicadores: b=1 => w=AreaExterior; b=0 => w=0
        try:
            m.addGenConstrIndicator(b, True,  w == AreaExterior, name=f"7.23.1__w_eq_area_if_b__{e1}")
            m.addGenConstrIndicator(b, False, w == 0.0,         name=f"7.23.1__w_eq_0_if_notb__{e1}")
        except Exception:
            pass
    m.addConstr(gp.quicksum(W.values()) == AreaExterior, name="7.23.1__w_sum")

    # 11.xx Fence (selección de categoría) y HasReja
    fence_opts = ["GdPrv","MnPrv","GdWo","MnWw","No aplica"]
    FENCE = _one_hot(m, "Fence", fence_opts)
    _tie_one_hot_to_x(m, x, "Fence", FENCE)
    if "No aplica" in FENCE:
        m.addConstr(HasReja == 1 - FENCE["No aplica"], name="11.xx__hasreja_from_fence")
    # MiscFeature (decisión) y costo opcional
    misc_opts = ["Elev","Gar2","Othr","Shed","TenC","No aplica"]
    MISC = _one_hot(m, "MiscFeature", misc_opts)
    _tie_one_hot_to_x(m, x, "Misc Feature", MISC)



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

    # handlers que lee run_opt
    m._lin_cost_expr  = cost_expr
    m._cost_terms     = getattr(m, "_cost_terms", [])  # para [COST-BREAKDOWN]
    m._X_input        = {"order": feat_order, "x": x} # para [AUDIT]
    m._ct             = ct  # acceso a tablas de costos desde auditorías

    # estrechar bounds antes de embebido (reduce big-M)
    _tighten_bounds_from_booster(m, bundle, feat_order, x)

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
