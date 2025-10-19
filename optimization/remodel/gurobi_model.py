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
    # 1) Variables de decisión
    # -------------------
    x: dict[str, gp.Var] = {}
    for f in MODIFIABLE:
        x[f.name] = m.addVar(lb=f.lb, ub=f.ub, vtype=_vtype(f.vartype), name=f"x_{f.name}")

    lin_cost = gp.LinExpr(ct.project_fixed)

    def pos(expr):
        v = m.addVar(lb=0.0, name=f"pos_{len(m.getVars())}")
        m.addConstr(v >= expr)
        return v

    # -------------------
    # 2) Armar X_input (fila 1xN en el ORDEN que espera el modelo)
    # -------------------
    feature_order = bundle.feature_names_in()
    modif = {f.name for f in MODIFIABLE}

    # Base: var si es modificable, si no el valor de la casa base
    row_vals: Dict[str, Any] = {}
    for fname in feature_order:
        row_vals[fname] = x[fname] if fname in modif else base_row.get(fname, 0)

    # ... arriba: X_input = pd.DataFrame([row_vals], columns=feature_order)
    X_input = pd.DataFrame([row_vals], columns=feature_order, dtype=object)
    # (mantén el _align_ohe_dtypes(...) tal como ya lo tienes)


    # -------------------
    # 3) Kitchen Qual (paquetes TA/EX)  [solo constraints aquí]
    # -------------------
    def _q_to_ord(v):
        mapping = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return mapping.get(str(v), 2)

    kq_base = _q_to_ord(base_row.get("Kitchen Qual", "TA"))
    dTA = x["delta_KitchenQual_TA"]
    dEX = x["delta_KitchenQual_EX"]
    m.addConstr(dTA + dEX <= 1, name="R9_kitchen_at_most_one_pkg")
    if kq_base >= 2: dTA.UB = 0
    if kq_base >= 4: dEX.UB = 0
    q_TA = max(kq_base, 2)
    q_EX = max(kq_base, 4)
    q_new = x["Kitchen Qual"]
    m.addConstr(
        q_new == kq_base + (q_TA - kq_base) * dTA + (q_EX - kq_base) * dEX,
        name="R9_kitchen_upgrade_link"
    )

    # ================== (EXTERIOR) ==================

    # === Exter Qual / Exter Cond → upgrade-only + costo por nivel ===
    def _q_to_ord(v):
        mapping = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return mapping.get(str(v), 2)

    if "Exter Qual" in x:
        exq_base = _q_to_ord(base_row.get("Exter Qual", "TA"))
        m.addConstr(x["Exter Qual"] >= exq_base, name="R_EXQ_upgrade_only")
        lin_cost += pos(x["Exter Qual"] - exq_base) * ct.exter_qual_upgrade_per_level

    if "Exter Cond" in x:
        exc_base = _q_to_ord(base_row.get("Exter Cond", "TA"))
        m.addConstr(x["Exter Cond"] >= exc_base, name="R_EXC_upgrade_only")
        lin_cost += pos(x["Exter Cond"] - exc_base) * ct.exter_cond_upgrade_per_level

    # Materiales disponibles (usa mismos nombres que en costs.py y en tu CSV)
    EXT_MATS = [
        "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
        "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
    ]

    # Binarios de decisión (de features.py)
    ex1 = {nm: x[f"ex1_is_{nm}"] for nm in EXT_MATS if f"ex1_is_{nm}" in x}
    ex2 = {nm: x[f"ex2_is_{nm}"] for nm in EXT_MATS if f"ex2_is_{nm}" in x}

    # ¿Existe Exterior2nd en la casa base?
    ex2_base_name = str(base_row.get("Exterior 2nd", "None"))
    Ilas2 = 0 if (ex2_base_name in ["None", "nan", "NaN", "NoneType", "0"] or pd.isna(base_row.get("Exterior 2nd"))) else 1

    # (E1) selección única / activación por Ilas2
    if ex1:
        m.addConstr(gp.quicksum(ex1.values()) == 1, name="EXT_ex1_pick_one")
    if ex2:
        m.addConstr(gp.quicksum(ex2.values()) == Ilas2, name="EXT_ex2_pick_ilas2")

    # (E2) inyectar dummies a X_input (tu modelo trae OHE de exteriores)
    for nm in EXT_MATS:
        col1 = f"Exterior 1st_{nm}"
        if col1 in X_input.columns and nm in ex1:
            X_input.loc[0, col1] = ex1[nm]
        col2 = f"Exterior 2nd_{nm}"
        if col2 in X_input.columns and nm in ex2:
            X_input.loc[0, col2] = ex2[nm]

    # (E3) ELEGIBILIDAD (solo si calidad o condición ≤ TA(=2))
    def _q_to_ord(v):
        M = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        try:
            return int(v)
        except Exception:
            return M.get(str(v), 2)

    exq_base = _q_to_ord(base_row.get("Exter Qual", 2))
    exc_base = _q_to_ord(base_row.get("Exter Cond", 2))
    eligible = 1 if (exq_base <= 2 or exc_base <= 2) else 0

    # material base (nombres CSV)
    ex1_base_name = str(base_row.get("Exterior 1st"))
    ex2_base_name = str(base_row.get("Exterior 2nd"))

    # Si NO elegible -> obligar a mantener materiales base
    if not eligible:
        # frente 1
        for nm in EXT_MATS:
            if nm in ex1:
                ex1[nm].UB = 1 if nm == ex1_base_name else 0
        # frente 2 (solo si existe)
        if Ilas2 == 1:
            for nm in EXT_MATS:
                if nm in ex2:
                    ex2[nm].UB = 1 if nm == ex2_base_name else 0

    # (E4) Costos de cambio de material (solo si cambias)
    area_ext = ct.exterior_area_proxy(base_row)

    # frente 1: suma de “material distinto al base”
    for nm in EXT_MATS:
        if nm in ex1 and nm != ex1_base_name:
            # demolición + reconstrucción con material nm
            lin_cost += (ct.exterior_demo_face1 * area_ext) * ex1[nm]
            lin_cost += (ct.ext_mat_cost(nm) * area_ext) * ex1[nm]

    # frente 2 (si existe)
    if Ilas2 == 1:
        for nm in EXT_MATS:
            if nm in ex2 and nm != ex2_base_name:
                lin_cost += (ct.exterior_demo_face2 * area_ext) * ex2[nm]
                lin_cost += (ct.ext_mat_cost(nm) * area_ext) * ex2[nm]

    
    # ================== FIN (EXTERIOR) ==================

    # ================== (MAS VNR: tipo + área) ==================
    MV_TYPES = ["BrkCmn", "BrkFace", "CBlock", "No aplica", "Stone"]  # <-- sin "None"

    mvt = {nm: x[f"mvt_is_{nm}"] for nm in MV_TYPES if f"mvt_is_{nm}" in x}

    if mvt:
        m.addConstr(gp.quicksum(mvt.values()) == 1, name="MVT_pick_one")

    def _put_var(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    for nm in MV_TYPES:
        col = f"Mas Vnr Type_{nm}"
        if col in X_input.columns and nm in mvt:
            _put_var(X_input, col, mvt[nm])

    mvt_base = str(base_row.get("Mas Vnr Type", "No aplica")).strip()
    try:
        mv_area_base = float(pd.to_numeric(base_row.get("Mas Vnr Area"), errors="coerce") or 0.0)
    except Exception:
        mv_area_base = 0.0

    base_cost = ct.mas_vnr_cost(mvt_base)
    mv_area = x["Mas Vnr Area"] if "Mas Vnr Area" in x else mv_area_base

    # política: si base es "No aplica" o área=0 → no se puede agregar
    no_base_veneer = (mv_area_base <= 1e-9) or (mvt_base == "No aplica")
    if no_base_veneer:
        if "No aplica" in mvt:
            m.addConstr(mvt["No aplica"] == 1, name="MVT_stay_noaplica")
        for nm in MV_TYPES:
            if nm != "No aplica" and nm in mvt:
                mvt[nm].UB = 0
        if isinstance(mv_area, gp.Var):
            mv_area.LB = 0.0
            mv_area.UB = 0.0
    else:
        for nm in MV_TYPES:
            if nm in mvt and ct.mas_vnr_cost(nm) < base_cost:
                mvt[nm].UB = 0
        if isinstance(mv_area, gp.Var):
            m.addConstr(mv_area >= mv_area_base, name="MVT_area_no_decrease")

    area_term = mv_area if isinstance(mv_area, gp.Var) else float(mv_area)
    for nm in MV_TYPES:
        if nm in mvt and nm != mvt_base:
            lin_cost += ct.mas_vnr_cost(nm) * area_term * mvt[nm]
    # ================== FIN (MAS VNR) ==================



    # -------------------
    # 4) ROOF: estilo/material con dummies "horneadas" y compatibilidad
    # -------------------
    style_names = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
    matl_names  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]

    # binarios provenientes de features.py
    s_bin = {nm: x[f"roof_style_is_{nm}"] for nm in style_names if f"roof_style_is_{nm}" in x}
    m_bin = {nm: x[f"roof_matl_is_{nm}"]  for nm in matl_names  if f"roof_matl_is_{nm}"  in x}

    # (exactamente uno)
    if s_bin:
        m.addConstr(gp.quicksum(s_bin.values()) == 1, name="ROOF_pick_one_style")
    if m_bin:
        m.addConstr(gp.quicksum(m_bin.values()) == 1, name="ROOF_pick_one_matl")

    # helper para asignar gp.Var en DataFrame sin warning
    def _put_var(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":  # 'O' = object
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    # Inyectar dummies en X_input si el modelo las tiene
    for nm in style_names:
        col = f"Roof Style_{nm}"
        if col in X_input.columns and nm in s_bin:
            _put_var(X_input, col, s_bin[nm])

    for nm in matl_names:
        col = f"Roof Matl_{nm}"
        if col in X_input.columns and nm in m_bin:
            _put_var(X_input, col, m_bin[nm])


    # Compatibilidad (0 = prohibido) usando labels del dataset
    for sn, forbids in {
        "Gable":   ["Membran"],
        "Hip":     ["Membran"],
        "Flat":    ["WdShngl", "ClyTile", "CompShg"],   # techo plano: sin tejas ni asfalto tradicional
        "Mansard": ["Membran"],
        "Shed":    ["ClyTile"],                         # ej: evitar teja pesada en shed muy inclinado
    }.items():
        for mn in forbids:
            if (f"roof_style_is_{sn}" in x) and (f"roof_matl_is_{mn}" in x):
                m.addConstr(x[f"roof_style_is_{sn}"] + x[f"roof_matl_is_{mn}"] <= 1,
                            name=f"ROOF_compat_{sn}_{mn}")


# ================== FIN (ROOF) ==================

    # ========== GARAGE FINISH (solo mejora, no crea garage) ==========
    GF_CATS = ["Fin", "RFn", "Unf", "No aplica"]

    # --- Variables de decisión ---
    gf = {g: x.get(f"garage_finish_is_{g}") for g in GF_CATS if f"garage_finish_is_{g}" in x}

    # --- Datos base ---
    def safe_val(colname: str) -> float:
        return float(base_row.get(colname, 0.0) or 0.0)

    baseGF = {g: safe_val(f"Garage Finish_{g}") for g in GF_CATS}
    maskGF = {g: 1 - baseGF[g] for g in GF_CATS}

    # ================== CASO 1: SIN GARAGE ==================
    if baseGF["No aplica"] == 1.0:
        for g, v in gf.items():
            if v is not None:
                m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GF_fix_NA_{g}")
        lin_cost += 0.0  # sin costo

    # ================== CASO 2: HAY GARAGE ==================
    else:
        # Selección única
        if gf:
            m.addConstr(gp.quicksum(v for v in gf.values() if v is not None) == 1, name="GF_pick_one")

        # Bloquear empeoramientos (no se puede pasar de Fin → Unf, por ejemplo)
        if "Fin" in gf and "RFn" in gf and "Unf" in gf:
            base_idx = next((i for i, g in enumerate(GF_CATS) if baseGF[g] == 1), None)
            if base_idx is not None:
                for i, g in enumerate(GF_CATS):
                    if i < base_idx and g in gf:
                        gf[g].UB = 0  # prohíbe categorías peores que la actual

        # Costos: solo si hay cambio real
        def _cost(name): return ct.garage_finish_costs.get(name, 0.0)
        lin_cost += gp.quicksum(_cost(g) * maskGF[g] * gf[g] for g in gf)
    # ================== FIN GARAGE FINISH ==================


    # ================== (CENTRAL AIR) ==================
    base_air_raw = str(base_row.get("Central Air", "N")).strip()
    base_is_Y = base_air_raw in {"Y", "Yes", "1", "True"}

    col_Y = "Central Air_Y"
    col_N = "Central Air_N"

    has_Y = col_Y in X_input.columns
    has_N = col_N in X_input.columns

    if has_Y or has_N:
        # 1 binaria de decisión
        air_yes = m.addVar(vtype=gp.GRB.BINARY, name="central_air_yes")

        # Si la base ya tiene aire, no se puede quitar
        if base_is_Y:
            air_yes.LB = 1.0
            air_yes.UB = 1.0
        else:
            # costo al agregar
            lin_cost += ct.central_air_install * air_yes

        # Proxy para el dummy "_N" (en vez de meter 1 - air_yes directo en el DF)
        air_no = None
        if has_N:
            air_no = m.addVar(vtype=gp.GRB.BINARY, name="central_air_no")
            m.addConstr(air_yes + air_no == 1, name="CentralAir_onehot")

        # Inyectar a X_input
        if has_Y:
            if X_input[col_Y].dtype != "O":
                X_input[col_Y] = X_input[col_Y].astype("object")
            X_input.loc[0, col_Y] = air_yes

        if has_N:
            if X_input[col_N].dtype != "O":
                X_input[col_N] = X_input[col_N].astype("object")
            X_input.loc[0, col_N] = air_no


    # ================== FIN (CENTRAL AIR) ==================


    # ========== POOL QC (solo mejora, no crea piscina) ==========
    PQC_CATS = ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"]

    # --- Variables de decisión ---
    pq = {g: x.get(f"pool_qc_is_{g}") for g in PQC_CATS if f"pool_qc_is_{g}" in x}

    # --- Datos base ---
    def safe_val(colname: str) -> float:
        return float(base_row.get(colname, 0.0) or 0.0)

    basePQ = {g: safe_val(f"Pool QC_{g}") for g in PQC_CATS}
    maskPQ = {g: 1 - basePQ[g] for g in PQC_CATS}

    # ================== CASO 1: SIN PISCINA ==================
    if basePQ["No aplica"] == 1.0:
        for g, v in pq.items():
            if v is not None:
                m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"PoolQC_fix_NA_{g}")
        lin_cost += 0.0

    # ================== CASO 2: HAY PISCINA ==================
    else:
        if pq:
            m.addConstr(gp.quicksum(v for v in pq.values() if v is not None) == 1, name="PoolQC_pick_one")

        # Bloquear empeoramientos
        ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]
        base_idx = next((i for i, g in enumerate(ORDER) if basePQ[g] == 1), None)
        if base_idx is not None:
            for i, g in enumerate(ORDER):
                if i < base_idx and g in pq:
                    pq[g].UB = 0  # prohíbe bajar calidad

        # Costos (solo si hay cambio)
        def _cost(name): return ct.pool_qc_costs.get(name, 0.0)
        lin_cost += gp.quicksum(_cost(g) * maskPQ[g] * pq[g] for g in pq)
    # ================== FIN POOL QC ==================


# ========== ÁREA LIBRE Y DECISIONES DE AMPLIACIÓN / AGREGADO ==========
    A_Full, A_Half, A_Kitch, A_Bed = 40.0, 20.0, 75.0, 70.0  # ft²

    AddFull, AddHalf, AddKitch, AddBed = x["AddFull"], x["AddHalf"], x["AddKitch"], x["AddBed"]

    COMPONENTES = ["Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
                "3Ssn Porch", "Screen Porch", "Pool Area"]
    z = {c: {s: x[f"z{s}_{c.replace(' ', '')}"] for s in [10, 20, 30]} for c in COMPONENTES}

    def _val(col):
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce") or 0.0)
        except Exception:
            return 0.0

    lot_area = _val("Lot Area")
    first_flr = _val("1st Flr SF")
    garage = _val("Garage Area")
    wooddeck = _val("Wood Deck SF")
    openporch = _val("Open Porch SF")
    enclosed = _val("Enclosed Porch")
    ssn3 = _val("3Ssn Porch")
    screen = _val("Screen Porch")
    pool = _val("Pool Area")

    area_libre_base = lot_area - (first_flr + garage + wooddeck + openporch +
                                enclosed + ssn3 + screen + pool)
    if area_libre_base < 0:
        area_libre_base = 0.0

    for c in COMPONENTES:
        m.addConstr(sum(z[c][s] for s in [10, 20, 30]) <= 1, name=f"AMPL_one_scale_{c.replace(' ', '')}")

    delta = {}
    for c in COMPONENTES:
        base_val = _val(c)
        if pd.isna(base_val) or np.isinf(base_val):
            base_val = 0.0
        delta[c] = {s: round(base_val * s / 100, 3) for s in [10, 20, 30]}

    m.addConstr(
        (area_libre_base
        - (A_Full * AddFull + A_Half * AddHalf + A_Kitch * AddKitch + A_Bed * AddBed)
        - gp.quicksum(delta[c][s] * z[c][s] for c in COMPONENTES for s in [10, 20, 30]))
        >= 0, name="AREA_libre_no_negativa"
    )

    lin_cost += ct.construction_cost * (A_Full * AddFull + A_Half * AddHalf + A_Kitch * AddKitch + A_Bed * AddBed)

    for c in COMPONENTES:
        lin_cost += (ct.ampl10_cost * delta[c][10] * z[c][10] +
                    ct.ampl20_cost * delta[c][20] * z[c][20] +
                    ct.ampl30_cost * delta[c][30] * z[c][30])
    # ================== FIN (ÁREA LIBRE Y AMPLIACIONES) ==================

    # ========== GARAGE QUAL / COND (solo mejora calidad, no construye garage) ==========
    G_CATS = ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"]
    G_LEQ_AV = {"TA", "Fa", "Po"}

    # --- Variables de decisión ---
    gq = {g: x.get(f"garage_qual_is_{g}") for g in G_CATS if f"garage_qual_is_{g}" in x}
    gc = {g: x.get(f"garage_cond_is_{g}") for g in G_CATS if f"garage_cond_is_{g}" in x}
    upgG = x.get("UpgGarage")

    # --- Datos base (usa nombres consistentes con el meta.json) ---
    def safe_val(colname: str) -> float:
        """Devuelve 1.0 si la columna existe y vale 1, sino 0.0."""
        return float(base_row.get(colname, 0.0) or 0.0)

    baseGQ = {g: safe_val(f"Garage Qual_{g}") for g in G_CATS}
    baseGC = {g: safe_val(f"Garage Cond_{g}") for g in G_CATS}
    maskQ = {g: 1 - baseGQ[g] for g in G_CATS}
    maskC = {g: 1 - baseGC[g] for g in G_CATS}

    # ================== CASO 1: SIN GARAGE ==================
    if baseGQ["No aplica"] == 1.0 or baseGC["No aplica"] == 1.0:
        # 🔒 No hay garage → fijar todo en “No aplica” y desactivar mejoras
        for g, v in gq.items():
            if v is not None:
                m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GQ_fix_NA_{g}")
        for g, v in gc.items():
            if v is not None:
                m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GC_fix_NA_{g}")
        if upgG is not None:
            m.addConstr(upgG == 0, name="UpgGarage_disabled_for_NA")

        lin_cost += 0.0  # sin costo

    # ================== CASO 2: HAY GARAGE ==================
    else:
        # Selección única
        if gq:
            m.addConstr(gp.quicksum(v for v in gq.values() if v is not None) == 1, name="GQ_pick_one")
        if gc:
            m.addConstr(gp.quicksum(v for v in gc.values() if v is not None) == 1, name="GC_pick_one")

        # Solo permite mejoras de calidad
        if upgG is not None:
            # upgG = 1 solo si hay alguna mejora en Q o C
            m.addConstr(
                gp.quicksum(maskQ[g] * gq[g] for g in gq if g != "No aplica")
                + gp.quicksum(maskC[g] * gc[g] for g in gc if g != "No aplica")
                <= len(G_LEQ_AV) * upgG,
                name="GARAGE_mask_sum_le_upg"
            )

        # Costos (solo si hay cambio real)
        def _cost(name): return ct.garage_qc_costs.get(name, 0.0)
        lin_cost += gp.quicksum(_cost(g) * maskQ[g] * gq[g] for g in gq)
        lin_cost += gp.quicksum(_cost(g) * maskC[g] * gc[g] for g in gc)



# ========== PAVED DRIVE ==========
    PAVED_CATS = ["Y", "P", "N"]

    # Variables binarias
    paved = {d: x[f"paved_drive_is_{d}"] for d in PAVED_CATS if f"paved_drive_is_{d}" in x}

    # Restricción: selección única
    if paved:
        m.addConstr(gp.quicksum(paved.values()) == 1, name="PAVED_pick_one")

    # Determinar categoría base
    base_pd = str(base_row.get("Paved Drive", "N")).strip()
    if base_pd not in PAVED_CATS:
        base_pd = "N"

    # Conjuntos permitidos según base
    if base_pd == "Y":
        allowed = ["Y"]
    elif base_pd == "P":
        allowed = ["P", "Y"]
    else:  # base N
        allowed = ["N", "P", "Y"]

    # Bloquear categorías no permitidas
    for d in PAVED_CATS:
        if d not in allowed and d in paved:
            paved[d].UB = 0

    # Agregar costo solo si hay cambio
    # Agregar costo solo si hay cambio
    lin_cost += gp.quicksum(
        ct.paved_drive_costs[d] * paved[d]
        for d in PAVED_CATS
        if d != base_pd and d in paved
    )

# ================== FIN (PAVED DRIVE) ==================


# ========== FENCE ==========
    FENCE_CATS = ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]

    fn = {f: x[f"fence_is_{f}"] for f in FENCE_CATS if f"fence_is_{f}" in x}

    if fn:
        m.addConstr(gp.quicksum(fn.values()) == 1, name="FENCE_pick_one")

    base_f = str(base_row.get("Fence", "NA")).strip()
    if base_f not in FENCE_CATS:
        base_f = "NA"

    # conjuntos permitidos
    if base_f == "NA":
        allowed = ["NA", "MnPrv", "GdPrv"]
    elif base_f in ["GdWo", "MnWw"]:
        allowed = [base_f, "MnPrv", "GdPrv"]
    else:
        allowed = [base_f]

    for f in FENCE_CATS:
        if f not in allowed and f in fn:
            fn[f].UB = 0

    # costo por cambio de categoría
    lin_cost += gp.quicksum(
        ct.fence_category_cost(f) * fn[f]
        for f in FENCE_CATS if f != base_f and f in fn
    )

    # costo por construcción nueva
    lot_front = float(pd.to_numeric(base_row.get("Lot Frontage"), errors="coerce") or 0.0)
    if base_f == "NA":
        for f in ["MnPrv", "GdPrv"]:
            if f in fn:
                lin_cost += ct.fence_build_cost_per_ft * lot_front * fn[f]
# ================== FIN (FENCE) ==================

    # ================== (ELECTRICAL) ==================
    ELECT_TYPES = ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]

    # binarios de decisión (de features.py)
    e_bin = {nm: x[f"elect_is_{nm}"] for nm in ELECT_TYPES if f"elect_is_{nm}" in x}

    # (El1) elegir exactamente uno
    if e_bin:
        m.addConstr(gp.quicksum(e_bin.values()) == 1, name="ELEC_pick_one")

    # (El2) inyectar dummies a X_input (pipeline entrenado con OHE de Electrical)
    for nm, vb in e_bin.items():
        col = f"Electrical_{nm}"
        if col in X_input.columns:
            _put_var(X_input, col, vb)

    # (El3) upgrade-only por costo: solo permitir tipos con costo >= costo del base
    elec_base_name = str(base_row.get("Electrical", "SBrkr"))
    base_cost = ct.electrical_cost(elec_base_name)
    for nm, vb in e_bin.items():
        if ct.electrical_cost(nm) < base_cost:
            vb.UB = 0  # no puedes elegir algo "más barato/peor" que el base

    # (El4) costo: si cambias, pagas demolición pequeña + costo del nuevo tipo
    # Detectar la base para evitar cobrar si te quedas igual
    for nm, vb in e_bin.items():
        if nm != elec_base_name:
            lin_cost += ct.electrical_demo_small * vb
            lin_cost += ct.electrical_cost(nm) * vb
    # ================== FIN (ELECTRICAL) ==================


    # -------------------
    # 5) Enlazar predictor (pre -> XGB) y pasar de log a precio
    # -------------------
    y_log = m.addVar(lb=-gp.GRB.INFINITY, name="y_log")
    _add_sklearn(m, bundle.pipe_for_gurobi(), X_input, [y_log])

    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        grid = np.linspace(10.0, 14.0, 81)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    # -------------------
    # 6) Costos lineales (numéricos + cocina + utilities + roof)
    # -------------------
    def _num_base(col: str) -> float:
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce"))
        except Exception:
            return 0.0

    base_vals = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
        "Wood Deck SF": _num_base("Wood Deck SF"),
        "Garage Cars": _num_base("Garage Cars"),
        "Total Bsmt SF": _num_base("Total Bsmt SF"),
    }



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

    # Cocina (paquetes) — ahora sí sumamos costo
    lin_cost += dTA * ct.kitchenQual_upgrade_TA
    lin_cost += dEX * ct.kitchenQual_upgrade_EX

    # Utilities (upgrade-only; costo solo si cambias)
    if "Utilities" in x:
        util_names = {0: "ELO", 1: "NoSeWa", 2: "NoSewr", 3: "AllPub"}
        util_to_ord = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}

        u_new = x["Utilities"]
        u_base_name = str(base_row.get("Utilities"))
        try:
            u_base_ord = int(pd.to_numeric(base_row.get("Utilities"), errors="coerce"))
            if u_base_ord not in (0, 1, 2, 3):
                u_base_ord = util_to_ord.get(u_base_name, 0)
        except Exception:
            u_base_ord = util_to_ord.get(u_base_name, 0)

        u_bin = {
            0: m.addVar(vtype=gp.GRB.BINARY, name="util_ELO"),
            1: m.addVar(vtype=gp.GRB.BINARY, name="util_NoSeWa"),
            2: m.addVar(vtype=gp.GRB.BINARY, name="util_NoSewr"),
            3: m.addVar(vtype=gp.GRB.BINARY, name="util_AllPub"),
        }
        m.addConstr(gp.quicksum(u_bin.values()) == 1, name="UTIL_one_hot")
        m.addConstr(u_new == gp.quicksum(k * u_bin[k] for k in u_bin), name="UTIL_link")
        m.addConstr(u_new >= u_base_ord, name="UTIL_upgrade_only")  # solo mejorar

        for k, vb in u_bin.items():
            if k != u_base_ord:
                lin_cost += ct.util_cost(util_names[k]) * vb

    # Roof: costos (fuera del bloque de utilities)
    if s_bin or m_bin:
        base_style = str(base_row.get("Roof Style", "Gable"))
        base_matl  = str(base_row.get("Roof Matl",  "CompShg"))

        for sn, vb in s_bin.items():
            if sn != base_style:
                lin_cost += ct.roof_style_cost(sn) * vb  # costo fijo por cambiar estilo

        roof_area = float(pd.to_numeric(base_row.get("Gr Liv Area"), errors="coerce") or 0.0)
        for mn, vb in m_bin.items():
            if mn != base_matl:
                lin_cost += ct.roof_matl_cost(mn) * roof_area * vb  # $/ft2 * área

    # -------------------
    # 7) Presupuesto y objetivo
    # -------------------
    total_cost = lin_cost
    m.addConstr(total_cost <= budget, name="budget")
    m.setObjective(y_price - total_cost, gp.GRB.MAXIMIZE)

    # -------------------
    # 8) Resto de restricciones (R1..R8)
    # -------------------
    if "1st Flr SF" in x and "2nd Flr SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["2nd Flr SF"], name="R1_floor1_ge_floor2")

    if "Gr Liv Area" in x and "Lot Area" in base_row:
        m.addConstr(x["Gr Liv Area"] <= float(base_row["Lot Area"]), name="R2_grliv_le_lot")

    if "1st Flr SF" in x and "Total Bsmt SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["Total Bsmt SF"], name="R3_floor1_ge_bsmt")

    def _val_or_var(col):
        return x[col] if col in x else float(base_row[col])

    need = all(c in base_row for c in ["Full Bath", "Bedroom AbvGr"])
    if need and ("Half Bath" in base_row or "Half Bath" in x):
        fullb = _val_or_var("Full Bath")
        halfb = _val_or_var("Half Bath") if ("Half Bath" in x or "Half Bath" in base_row) else 0.0
        beds  = _val_or_var("Bedroom AbvGr")
        m.addConstr(fullb + halfb <= beds, name="R4_baths_le_bedrooms")

    if ("Full Bath" in x) or ("Full Bath" in base_row):
        m.addConstr(_val_or_var("Full Bath") >= 1, name="R5_min_fullbath")
    if ("Bedroom AbvGr" in x) or ("Bedroom AbvGr" in base_row):
        m.addConstr(_val_or_var("Bedroom AbvGr") >= 1, name="R5_min_bedrooms")
    if ("Kitchen AbvGr" in x) or ("Kitchen AbvGr" in base_row):
        m.addConstr(_val_or_var("Kitchen AbvGr") >= 1, name="R5_min_kitchen")

    lowqual_names = ["Low Qual Fin SF", "LowQualFinSF"]
    lowqual_col = next((c for c in lowqual_names if c in X_input.columns), None)
    cols_needed = ["Gr Liv Area", "1st Flr SF", "2nd Flr SF"]
    if all(c in X_input.columns for c in cols_needed):
        lhs = _val_or_var("Gr Liv Area")
        rhs = _val_or_var("1st Flr SF") + _val_or_var("2nd Flr SF")
        if lowqual_col is not None:
            rhs += _val_or_var(lowqual_col)
        m.addConstr(lhs == rhs, name="R7_gr_liv_equality")

    r8_ok = all(c in X_input.columns for c in ["TotRms AbvGrd", "Bedroom AbvGr", "Kitchen AbvGr"])
    if r8_ok:
        other = m.addVar(lb=0, ub=15, vtype=gp.GRB.INTEGER, name="R8_other_rooms")
        m.addConstr(_val_or_var("TotRms AbvGrd") == _val_or_var("Bedroom AbvGr") + _val_or_var("Kitchen AbvGr") + other,
                    name="R8_rooms_balance")

    return m
