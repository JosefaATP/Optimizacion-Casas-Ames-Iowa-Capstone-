# optimization/remodel/run_opt.py
import argparse
import pandas as pd
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import (
    XGBBundle,
    _coerce_quality_ordinals_inplace,
    _coerce_utilities_ordinal_inplace,
)
from .gurobi_model import build_mip_embed
from .features import MODIFIABLE

# -----------------------------
# Mapeos Utilities (ordinales)
# -----------------------------
UTIL_TO_ORD = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}
ORD_TO_UTIL = {v: k for k, v in UTIL_TO_ORD.items()}

def _safe_util_ord(val) -> int:
    """Convierte val a 0..3 de forma robusta."""
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.notna(v) and int(v) in (0, 1, 2, 3):
            return int(v)
    except Exception:
        pass
    return UTIL_TO_ORD.get(str(val), 0)

def _qual_to_ord(val, default=2):
    MAP = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.notna(v):
            iv = int(v)
            if iv in (0,1,2,3,4):
                return iv
    except Exception:
        pass
    return MAP.get(str(val).strip(), default)

# -----------------------------
# Helpers de impresión
# -----------------------------
def money(v: float) -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return str(v)
    if pd.isna(f):
        return "-"
    return f"${f:,.0f}"

def frmt_num(x) -> str:
    """Formatea numeros, si viene string/NaN, devuelve '-' o el string."""
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return str(x)
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}" if abs(v - round(v)) > 1e-6 else f"{int(round(v))}"

def _getv(m, *names):
    """Devuelve la primera variable que exista entre varios alias."""
    for nm in names:
        v = m.getVarByName(nm)
        if v is not None:
            return v
    # plan C: buscar ignorando espacios/underscores y case
    target_keys = [nm.replace(" ", "").replace("_", "").lower() for nm in names]
    for v in m.getVars():
        key = v.VarName.replace(" ", "").replace("_", "").lower()
        if key in target_keys:
            return v
    return None

# -----------------------------
# Construir fila base con dummies
# -----------------------------
def _row_with_dummies(base_row: pd.Series, feat_order: list[str]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for col in feat_order:
        if col in base_row.index:
            vals[col] = base_row[col]
        else:
            if "_" in col:
                base_col, dummy_val = col.split("_", 1)
                if base_col in base_row.index:
                    vals[col] = 1.0 if str(base_row[base_col]) == dummy_val else 0.0
                else:
                    vals[col] = 0.0
            else:
                vals[col] = 0.0
    return vals

# =============================
# main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--basecsv", type=str, default=None, help="ruta alternativa al CSV base")
    args = ap.parse_args()

    # Datos base, costos y modelo
    base = get_base_house(args.pid, base_csv=args.basecsv)
    ct = costs.CostTables()
    bundle = XGBBundle()

    # ===== precio_base (ValorInicial) con el pipeline COMPLETO =====
    feat_order = bundle.feature_names_in()
    X_base = pd.DataFrame([_row_with_dummies(base.row, feat_order)], columns=feat_order)

    # normalizaciones defensivas
    _coerce_quality_ordinals_inplace(X_base, getattr(bundle, "quality_cols", []))
    _coerce_utilities_ordinal_inplace(X_base)

    precio_base = float(bundle.predict(X_base).iloc[0])

    # ============== DEBUGS (opcionales) ==============
    # Kitchen Qual
    if "Kitchen Qual" in feat_order:
        X_dbg = X_base.copy()
        vals = []
        for q in [0, 1, 2, 3, 4]:
            X_dbg.loc[:, "Kitchen Qual"] = q
            vals.append((q, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Kitchen Qual -> precio:", vals)

    # Utilities
    if "Utilities" in feat_order:
        vals = []
        for k, name in enumerate(["ELO", "NoSeWa", "NoSewr", "AllPub"]):
            X_dbg = X_base.copy()
            X_dbg.loc[:, "Utilities"] = k
            vals.append((k, name, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Utilities -> precio:", [(k, name, round(p, 2)) for (k, name, p) in vals])

    # Roof Style
    try:
        if any(c.startswith("Roof Style_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Roof Style_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]:
                col = f"Roof Style_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Roof Style -> precio:", vals)
    except Exception:
        pass

    # Roof Matl
    try:
        if any(c.startswith("Roof Matl_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Roof Matl_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]:
                col = f"Roof Matl_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Roof Matl -> precio:", vals)
    except Exception:
        pass

    # Exterior 1st
    if any(c.startswith("Exterior 1st_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("Exterior 1st_")]:
            Xd[c] = 0.0
        vals = []
        for nm in ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
                   "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl"]:
            col = f"Exterior 1st_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG Exterior 1st -> precio:", vals)

    # Exter Qual / Exter Cond
    if "Exter Qual" in X_base.columns:
        Xd = X_base.copy()
        vals = []
        for q in [0,1,2,3,4]:
            Xd.loc[:, "Exter Qual"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Exter Qual -> precio:", vals)

    if "Exter Cond" in X_base.columns:
        Xd = X_base.copy()
        vals = []
        for q in [0,1,2,3,4]:
            Xd.loc[:, "Exter Cond"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Exter Cond -> precio:", vals)

    # Mas Vnr Type
    try:
        if any(c.startswith("Mas Vnr Type_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Mas Vnr Type_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["BrkCmn","BrkFace","CBlock","Stone","No aplica"]:
                col = f"Mas Vnr Type_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Mas Vnr Type -> precio:", vals)
    except Exception:
        pass

    # Electrical
    if any(c.startswith("Electrical_") for c in X_base.columns):
        X_dbg = X_base.copy()
        for c in [c for c in X_dbg.columns if c.startswith("Electrical_")]:
            X_dbg[c] = 0.0
        vals = []
        for nm in ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]:
            col = f"Electrical_{nm}"
            if col in X_dbg.columns:
                X_dbg[col] = 1.0
                vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                X_dbg[col] = 0.0
        print("DEBUG Electrical -> precio:", vals)

    # Central Air
    try:
        if any(c.startswith("Central Air_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Central Air_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["N", "Y"]:
                col = f"Central Air_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Central Air -> precio:", vals)
        else:
            print("DEBUG Central Air -> precio: []")
    except Exception:
        pass

    # --- DEBUG: Heating (TIPO) -> precio ---
    if any(c.startswith("Heating_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("Heating_")]:
            Xd[c] = 0.0
        vals = []
        for nm in ["Floor","GasA","GasW","Grav","OthW","Wall"]:
            col = f"Heating_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG Heating (TIPO) -> precio:", vals)

    # --- DEBUG: Heating QC (NIVEL) -> precio ---
    if "Heating QC" in X_base.columns:
        vals = []
        for q in [0,1,2,3,4]:
            Xd = X_base.copy()
            Xd.loc[:, "Heating QC"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Heating QC (nivel) -> precio:", vals)

    # --- DEBUG: Basement Cond (NIVEL) -> precio ---
    if "Bsmt Cond" in feat_order:
        Xd = X_base.copy()
        vals = []
        for q in [0,1,2,3,4]:
            Xd.loc[:, "Bsmt Cond"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Bsmt Cond -> precio:", vals)

    # DEBUG: terminar 100 ft² vs no terminar (manteniendo total constante)
    if all(c in feat_order for c in ["BsmtFin SF 1","Bsmt Unf SF","Total Bsmt SF"]):
        Xd = X_base.copy()
        def price(df): return float(bundle.predict(df).iloc[0])
        b1 = float(pd.to_numeric(base.row.get("BsmtFin SF 1"), errors="coerce") or 0.0)
        bu = float(pd.to_numeric(base.row.get("Bsmt Unf SF"),   errors="coerce") or 0.0)
        tb = float(pd.to_numeric(base.row.get("Total Bsmt SF"), errors="coerce") or (b1+bu))
        if bu >= 100:
            Xd.loc[:, "BsmtFin SF 1"] = b1 + 100
            Xd.loc[:, "Bsmt Unf SF"]  = bu - 100
            Xd.loc[:, "Total Bsmt SF"] = tb
            p_up = price(Xd)
            print(f"DEBUG Bsmt: +100 ft² terminados → Δprecio = {p_up - precio_base:,.2f}")
        
        # --- DEBUG: BsmtFin Type 1 -> precio ---
    B1_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]
    if any(c.startswith("BsmtFin Type 1_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("BsmtFin Type 1_")]:
            Xd[c] = 0.0
        vals = []
        for nm in B1_TYPES:
            col = f"BsmtFin Type 1_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG BsmtFin Type 1 -> precio:", [(nm, round(p, 2)) for nm, p in vals])

    # --- DEBUG: BsmtFin Type 2 -> precio ---
    B2_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]
    if any(c.startswith("BsmtFin Type 2_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("BsmtFin Type 2_")]:
            Xd[c] = 0.0
        vals = []
        for nm in B2_TYPES:
            col = f"BsmtFin Type 2_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG BsmtFin Type 2 -> precio:", [(nm, round(p, 2)) for nm, p in vals])

    # --- DEBUG: Fireplace Qu -> precio ---
    if "Fireplace Qu" in X_base.columns:
        vals = []
        for q in [0,1,2,3,4]:
            Xd = X_base.copy()
            Xd.loc[:, "Fireplace Qu"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Fireplace Qu -> precio:", vals)

    # --- DEBUG helper para columnas de calidad (ordinal u OHE) ---
    def _debug_quality_feature(col_name: str):
        """
        Si el modelo trae la calidad como ordinal (0..4), prueba esos niveles.
        Si viene como OHE (col_name_Po/Fa/TA/Gd/Ex), activa cada dummy.
        Si no está en el modelo, lo indica.
        """
        if col_name in X_base.columns:
            # ordinal
            vals = []
            for q in [0, 1, 2, 3, 4]:
                Xd = X_base.copy()
                Xd.loc[:, col_name] = q
                vals.append((q, float(bundle.predict(Xd).iloc[0])))
            print(f"DEBUG {col_name} -> precio:", [(q, round(p, 2)) for (q, p) in vals])
            return

        # OHE fallback
        onehots = [c for c in X_base.columns if c.startswith(f"{col_name}_")]
        if onehots:
            Xd = X_base.copy()
            for c in onehots:
                Xd[c] = 0.0
            order = ["Po","Fa","TA","Gd","Ex"]
            vals = []
            for nm in order:
                col = f"{col_name}_{nm}"
                if col in Xd.columns:
                    Xd[col] = 1.0
                    vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                    Xd[col] = 0.0
            print(f"DEBUG {col_name} (OHE) -> precio:", [(nm, round(p, 2)) for (nm, p) in vals])
        else:
            print(f"DEBUG {col_name} -> precio: [no está en el modelo]")

    # --- DEBUG: Fireplace / Bsmt Qual / Bsmt Cond ---
    _debug_quality_feature("Fireplace Qu")
    _debug_quality_feature("Bsmt Qual")
    _debug_quality_feature("Bsmt Cond")



    # ============ FIN DEBUGS ============

    # ===== construir MIP =====
    m: gp.Model = build_mip_embed(base.row, args.budget, ct, bundle)

    # Ajustes de resolución
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = PARAMS.time_limit
    m.Params.LogToConsole = PARAMS.log_to_console

    # ===== objetivo como (Final - Inicial - Costos) =====
    m.ObjCon = -precio_base

    # Optimizar
    m.optimize()

    # ===== chequear factibilidad =====
    if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        print("no se encontro solucion factible, status:", m.Status)
        try:
            if m.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                print("Intentando IIS (conjunto mínimo de restricciones en conflicto)...")
                m.computeIIS()
                viols = []
                try:
                    for c in m.getConstrs():
                        if c.getAttr(gp.GRB.Attr.IISConstr):
                            viols.append(c.ConstrName)
                except Exception:
                    pass
                try:
                    for gc in m.getGenConstrs():
                        try:
                            if gc.getAttr(gp.GRB.Attr.IISConstr):
                                viols.append(gc.GenConstrName)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    for v in m.getVars():
                        if v.getAttr(gp.GRB.Attr.IISLB):
                            viols.append(f"LB({v.VarName})")
                        if v.getAttr(gp.GRB.Attr.IISUB):
                            viols.append(f"UB({v.VarName})")
                except Exception:
                    pass
                if viols:
                    print("IIS (restricciones/bounds conflictivos):")
                    for nm in sorted(set(viols)):
                        print(" -", nm)
                try:
                    m.write("model_iis.ilp")
                    m.write("model_full.lp")
                    print("Archivos escritos: model_iis.ilp (IIS) y model_full.lp (modelo).")
                except Exception:
                    pass
        except Exception as e:
            print("Fallo al obtener IIS:", e)
        return

    # ===== leer precios =====
    precio_remodelada = float(m.getVarByName("y_price").X)
    y_log = float(m.getVarByName("y_log").X)  # por si quieres verlo

    # ===== reconstruir costos de remodelación (para reporte) =====
    def _pos(v: float) -> float:
        return v if v > 0 else 0.0

    def _num_base(name: str) -> float:
        try:
            return float(pd.to_numeric(base.row.get(name), errors="coerce"))
        except Exception:
            return 0.0

    base_vals = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
        "Garage Cars": _num_base("Garage Cars"),
    }

    def _to_float_safe(val) -> float:
        try:
            v = float(pd.to_numeric(val, errors="coerce"))
            return 0.0 if pd.isna(v) else v
        except Exception:
            return 0.0

    # decisiones óptimas (x_*)
    opt = {f.name: m.getVarByName(f"x_{f.name}").X for f in MODIFIABLE if m.getVarByName(f"x_{f.name}") is not None}

    cambios_costos = []

    # costos numéricos mapeados (dorms, baños, garage)
    def costo_var(nombre: str, base_v: float, nuevo_v: float) -> float:
        delta = nuevo_v - base_v
        if nombre == "Bedroom AbvGr":
            return _pos(delta) * ct.add_bedroom
        if nombre == "Full Bath":
            return _pos(delta) * ct.add_bathroom
        if nombre == "Garage Cars":
            return _pos(delta) * ct.garage_per_car
        return 0.0

    total_cost_vars = 0.0
    for nombre, base_v in base_vals.items():
        nuevo_v = _to_float_safe(opt.get(nombre, base_v))
        base_vf = _to_float_safe(base_v)
        c = costo_var(nombre, base_vf, nuevo_v)
        if abs(nuevo_v - base_vf) > 1e-9:
            cambios_costos.append((nombre, base_vf, nuevo_v, c))
        total_cost_vars += c

    # ---- Cocina (paquetes) ----
    def _q_to_ord(txt) -> int:
        MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(txt)
        except Exception:
            return MAP.get(str(txt), -1)

    kq_base = _q_to_ord(base.row.get("Kitchen Qual", "TA"))
    dTA = dEX = 0
    kq_new = kq_base
    try:
        v_ta = m.getVarByName("x_delta_KitchenQual_TA")
        v_ex = m.getVarByName("x_delta_KitchenQual_EX")
        if v_ta is not None: dTA = int(round(v_ta.X))
        if v_ex is not None: dEX = int(round(v_ex.X))
    except Exception:
        pass
    try:
        v_kq = m.getVarByName("x_Kitchen Qual")
        if v_kq is not None:
            kq_new = int(round(v_kq.X))
    except Exception:
        if dTA:
            kq_new = max(kq_base, 2)
        if dEX:
            kq_new = max(kq_new, 4)

    kitchen_cost = dTA * ct.kitchenQual_upgrade_TA + dEX * ct.kitchenQual_upgrade_EX
    if (dTA or dEX) and (kq_new != kq_base):
        cambios_costos.append(("Kitchen Qual", kq_base, kq_new, float(kitchen_cost)))

    # ---- Utilities (solo si cambias) ----
    base_util_name = str(base.row.get("Utilities", "ELO"))
    if base_util_name not in UTIL_TO_ORD:
        base_util_name = ORD_TO_UTIL.get(_safe_util_ord(base.row.get("Utilities")), "ELO")

    util_pick = None
    for nm in ["ELO", "NoSeWa", "NoSewr", "AllPub"]:
        v = m.getVarByName(f"util_{nm}")
        if v is not None and v.X > 0.5:
            util_pick = nm
            break

    util_cost_add = 0.0
    if util_pick is not None and util_pick != base_util_name:
        util_cost_add = ct.util_cost(util_pick)
        cambios_costos.append(("Utilities", base_util_name, util_pick, float(util_cost_add)))

    # ---- Roof elegido + costos ----
    style_names = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
    matl_names  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
    style_new = None
    matl_new  = None
    for nm in style_names:
        v = m.getVarByName(f"roof_style_is_{nm}")
        if v is not None and v.X > 0.5:
            style_new = nm
            break
    for nm in matl_names:
        v = m.getVarByName(f"roof_matl_is_{nm}")
        if v is not None and v.X > 0.5:
            matl_new = nm
            break

    style_base = str(base.row.get("Roof Style", "Gable"))
    matl_base  = str(base.row.get("Roof Matl",  "CompShg"))

    roof_area = float(pd.to_numeric(base.row.get("Gr Liv Area"), errors="coerce") or 0.0)
    style_cost = 0.0
    matl_cost  = 0.0
    if style_new and style_new != style_base:
        style_cost = ct.roof_style_cost(style_new)
        cambios_costos.append(("Roof Style", style_base, style_new, style_cost))
    if matl_new and matl_new != matl_base:
        matl_cost = ct.roof_matl_cost(matl_new) * roof_area
        cambios_costos.append(("Roof Matl", matl_base, matl_new, matl_cost))

    # ---- Exterior 1st/2nd ----
    ext_names = ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard",
                 "ImStucc","MetalSd","Plywood","PreCast","Stone","Stucco","VinylSd",
                 "Wd Sdng","WdShngl"]

    def _pick_ext(prefix: str, names: list[str]) -> str | None:
        for nm in names:
            v = m.getVarByName(f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    ex1_new = _pick_ext("ex1_", ext_names)
    ex2_new = _pick_ext("ex2_", ext_names)
    ex1_base = str(base.row.get("Exterior 1st", "VinylSd"))
    ex2_base = str(base.row.get("Exterior 2nd", ex1_base))
    area_ext = ct.exterior_area_proxy(base.row)

    ext_cost_sum = 0.0
    if ex1_new and ex1_new != ex1_base:
        c_demo = (ct.exterior_demo_face1 * area_ext)
        c_mat  = (ct.ext_mat_cost(ex1_new) * area_ext)
        ext_cost_sum += (c_demo + c_mat)
        cambios_costos.append(("Exterior 1st", ex1_base, ex1_new, c_demo + c_mat))
    if ex2_new and ex2_new != ex2_base:
        c_demo = (ct.exterior_demo_face2 * area_ext)
        c_mat  = (ct.ext_mat_cost(ex2_new) * area_ext)
        ext_cost_sum += (c_demo + c_mat)
        cambios_costos.append(("Exterior 2nd", ex2_base, ex2_new, c_demo + c_mat))

    # ---- Exter Qual / Exter Cond (reporte + costo) ----
    def _q_to_ord_report(v):
        MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return MAP.get(str(v), 2)

    exter_qual_cost = 0.0
    exter_cond_cost = 0.0
    try:
        # Exter Qual
        v_exq = m.getVarByName("x_Exter Qual")
        if v_exq is not None:
            exq_new = int(round(v_exq.X))
            exq_base = _q_to_ord_report(base.row.get("Exter Qual", "TA"))
            if exq_new > exq_base:
                exter_qual_cost = (exq_new - exq_base) * ct.exter_qual_upgrade_per_level
                cambios_costos.append(("Exter Qual", exq_base, exq_new, exter_qual_cost))

        # Exter Cond
        v_exc = m.getVarByName("x_Exter Cond")
        if v_exc is not None:
            exc_new = int(round(v_exc.X))
            exc_base = _q_to_ord_report(base.row.get("Exter Cond", "TA"))
            if exc_new > exc_base:
                exter_cond_cost = (exc_new - exc_base) * ct.exter_cond_upgrade_per_level
                cambios_costos.append(("Exter Cond", exc_base, exc_new, exter_cond_cost))
    except Exception:
        pass

    # ---- Electrical ----
    elec_extra = 0.0
    try:
        picked = None
        for nm in ["SBrkr","FuseA","FuseF","FuseP","Mix"]:
            v = m.getVarByName(f"x_elect_is_{nm}")
            if v is not None and v.X > 0.5:
                picked = nm
                break
        base_elec = str(base.row.get("Electrical", "SBrkr"))
        if picked and picked != base_elec:
            elec_extra = ct.electrical_demo_small + ct.electrical_cost(picked)
            cambios_costos.append(("Electrical", base_elec, picked, elec_extra))
    except Exception:
        pass

    # ---- Mas Vnr Type ----
    MVT_NAMES = ["BrkCmn", "BrkFace", "CBlock", "Stone", "No aplica", "None"]
    def _pick_mvt_from_model() -> str | None:
        for nm in MVT_NAMES:
            v = m.getVarByName(f"x_mvt_is_{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    mvt_extra = 0.0
    try:
        mvt_pick = _pick_mvt_from_model()
        mvt_base = str(base.row.get("Mas Vnr Type", "No aplica")).strip()

        v_mv_area = m.getVarByName("x_Mas Vnr Area")
        if v_mv_area is not None:
            mv_area = float(v_mv_area.X)
        else:
            try:
                mv_area = float(pd.to_numeric(base.row.get("Mas Vnr Area"), errors="coerce") or 0.0)
            except Exception:
                mv_area = 0.0

        if (mvt_pick is not None) and (mvt_pick != mvt_base):
            mvt_extra = ct.mas_vnr_cost(mvt_pick) * mv_area
            cambios_costos.append(("Mas Vnr Type", mvt_base, mvt_pick, mvt_extra))
    except Exception:
        pass

    # ---- Central Air (reporte + costo) ----
    central_air_cost_add = 0.0
    try:
        base_air = "Y" if str(base.row.get("Central Air", "N")).strip() in {"Y","Yes","1","True"} else "N"
        v_yes = m.getVarByName("central_air_yes")
        if v_yes is not None:
            pick = "Y" if v_yes.X > 0.5 else "N"
            print(f"Central Air (base -> óptimo): {base_air} -> {pick}")
            if pick != base_air:
                central_air_cost_add = float(ct.central_air_install)
                cambios_costos.append(("Central Air", base_air, pick, central_air_cost_add))
    except Exception:
        pass

    # ---- Heating (reporte + costo) ----
    heating_cost_report = 0.0
    try:
        heat_types = ["Floor","GasA","GasW","Grav","OthW","Wall"]
        heat_base = str(base.row.get("Heating","GasA")).strip()

        # tipo final (admite 'x_heat_is_*' o 'heat_is_*')
        heat_new = None
        for nm in heat_types:
            v = _getv(m, f"x_heat_is_{nm}", f"heat_is_{nm}")
            if v is not None and v.X > 0.5:
                heat_new = nm
                break

        # upg_type / change_type (admite con/sin x_)
        v_change = _getv(m, "x_heat_change_type", "heat_change_type")
        v_upg    = _getv(m, "x_heat_upg_type",   "heat_upg_type")

        change_type = int(round(v_change.X)) if v_change is not None else 0
        upg_type    = int(round(v_upg.X))    if v_upg    is not None else 0

        # 1) Reconstruir mismo tipo
        if upg_type == 1 and change_type == 0:
            c_same = ct.heating_type_cost(heat_base)
            heating_cost_report += c_same
            cambios_costos.append(("Heating (reconstruir tipo)", heat_base, heat_base, c_same))

        # 2) Cambiar a tipo más caro
        if heat_new and heat_new != heat_base:
            c_new = ct.heating_type_cost(heat_new)
            heating_cost_report += c_new
            cambios_costos.append(("Heating (tipo)", heat_base, heat_new, c_new))

        # 3) Cambiar calidad
        qc_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
        qc_base_val = _qual_to_ord(base.row.get("Heating QC"), default=2)
        qc_base_txt = qc_map[qc_base_val]
        vq = _getv(m, "x_Heating QC", "x_HeatingQC")
        if vq is not None:
            qc_new_val = int(round(vq.X))
            if qc_new_val != qc_base_val:
                qc_new_txt = qc_map.get(qc_new_val, str(qc_new_val))
                c_qc = ct.heating_qc_cost(qc_new_txt)
                heating_cost_report += c_qc
                cambios_costos.append(("Heating QC", qc_base_txt, qc_new_txt, c_qc))
    except Exception as e:
        print("[HEAT-DEBUG] Error en reporte:", e)

    # ---- BsmtFin: reconstrucción de valores + costo ----
    bsmt_finish_cost_report = 0.0
    try:
        v_fin = m.getVarByName("bsmt_finish")
        v_tr1 = m.getVarByName("bsmt_tr1")
        v_tr2 = m.getVarByName("bsmt_tr2")
        b1_var = m.getVarByName("bsmt_fin1")
        b2_var = m.getVarByName("bsmt_fin2")
        bu_var = m.getVarByName("bsmt_unf")

        if all(v is not None for v in [v_fin, v_tr1, v_tr2, b1_var, b2_var, bu_var]):
            tr1 = float(v_tr1.X)
            tr2 = float(v_tr2.X)

            # bases
            b1b = float(pd.to_numeric(base.row.get("BsmtFin SF 1"), errors="coerce") or 0.0)
            b2b = float(pd.to_numeric(base.row.get("BsmtFin SF 2"), errors="coerce") or 0.0)
            bub = float(pd.to_numeric(base.row.get("Bsmt Unf SF"),   errors="coerce") or 0.0)
            tbb = float(pd.to_numeric(base.row.get("Total Bsmt SF"), errors="coerce") or (b1b+b2b+bub))

            # nuevos desde variables (conservación)
            b1n = float(b1_var.X)
            b2n = float(b2_var.X)
            bun = float(bu_var.X)
            tbn = b1n + b2n + bun

            # costo: solo lo trasladado a “fin”
            added = tr1 + tr2
            if added > 1e-9:
                bsmt_finish_cost_report = ct.finish_basement_per_f2 * added

            # cambios para imprimir (cobrando solo lo que corresponde)
            if abs(b1n - b1b) > 1e-9:
                cambios_costos.append(("BsmtFin SF 1", b1b, b1n, ct.finish_basement_per_f2 * tr1))
            if abs(b2n - b2b) > 1e-9:
                cambios_costos.append(("BsmtFin SF 2", b2b, b2n, ct.finish_basement_per_f2 * tr2))
            if abs(bun - bub) > 1e-9:
                cambios_costos.append(("Bsmt Unf SF",  bub, bun, 0.0))
            if abs(tbn - tbb) > 1e-9:
                cambios_costos.append(("Total Bsmt SF", tbb, tbn, 0.0))
    except Exception as e:
        print("[BSMT-DEBUG] error construyendo reporte:", e)

    # Fallback (si no existían tr1/tr2/vars de conservación): usar solo delta positivo
    if bsmt_finish_cost_report == 0.0:
        try:
            b1_base = float(pd.to_numeric(base.row.get("BsmtFin SF 1"), errors="coerce") or 0.0)
            b2_base = float(pd.to_numeric(base.row.get("BsmtFin SF 2"), errors="coerce") or 0.0)
            b1_new  = _to_float_safe(opt.get("BsmtFin SF 1", b1_base))
            b2_new  = _to_float_safe(opt.get("BsmtFin SF 2", b2_base))
            c_b1 = max(0.0, b1_new - b1_base) * ct.finish_basement_per_f2
            c_b2 = max(0.0, b2_new - b2_base) * ct.finish_basement_per_f2
            if c_b1 > 0:
                cambios_costos.append(("BsmtFin SF 1", b1_base, b1_new, c_b1))
            if c_b2 > 0:
                cambios_costos.append(("BsmtFin SF 2", b2_base, b2_new, c_b2))
            bsmt_finish_cost_report = c_b1 + c_b2
        except Exception:
            pass


    # ---- Bsmt Cond (reporte + costo) ----
    bsmt_cond_cost_report = 0.0
    try:
        v_bc = m.getVarByName("x_Bsmt Cond")
        if v_bc is not None:
            bc_new = int(round(v_bc.X))
            inv_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
            def _ord(v):
                M={"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
                try: return int(v)
                except: return M.get(str(v),2)
            bc_base = _ord(base.row.get("Bsmt Cond","TA"))
            if bc_new > bc_base:
                bc_new_txt  = inv_map[bc_new]
                bc_base_txt = inv_map[bc_base]
                # si definiste ct.bsmt_cond_cost(nivel), úsalo; si no, costo por nivel:
                try:
                    c_bc = ct.bsmt_cond_cost(bc_new_txt)
                except Exception:
                    c_bc = (bc_new - bc_base) * 2000.0
                bsmt_cond_cost_report += c_bc
                cambios_costos.append(("Bsmt Cond", bc_base_txt, bc_new_txt, c_bc))
    except Exception:
        pass

    # ---- BsmtFin Type 1/2 (reporte + costo) ----
    bsmt_type_cost_report = 0.0
    TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]

    def _pick_cat(prefix: str) -> str | None:
        # intenta con y sin 'x_' por si difieren los nombres
        for nm in TYPES:
            v = _getv(m, f"x_{prefix}{nm}", f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    # Base (desde la casa)
    b1_base = str(base.row.get("BsmtFin Type 1", "No aplica")).strip()
    b2_base = str(base.row.get("BsmtFin Type 2", "No aplica")).strip()

    # Nuevo (desde el modelo)
    b1_new = _pick_cat("b1_is_") or b1_base
    b2_new = _pick_cat("b2_is_") or b2_base

    # Costo por tipología (ajusta tu CostTables.bsmt_type_cost)
    def _bsmt_type_cost(nm: str) -> float:
        try:
            return float(ct.bsmt_type_cost(nm))
        except Exception:
            return 0.0

    if b1_new != b1_base:
        c = _bsmt_type_cost(b1_new)
        cambios_costos.append(("BsmtFin Type 1", b1_base, b1_new, c))
        bsmt_type_cost_report += c

    # En Ames, Type 2 puede no existir; cobra solo si la base tenía algo
    if b2_base != "No aplica" and b2_new != b2_base:
        c = _bsmt_type_cost(b2_new)
        cambios_costos.append(("BsmtFin Type 2", b2_base, b2_new, c))
        bsmt_type_cost_report += c

    # reflejar en snapshot SOLO si opt_dict ya existe en ese punto
    if 'opt_dict' in locals():
        opt_dict["BsmtFin Type 1"] = b1_new
        opt_dict["BsmtFin Type 2"] = b2_new

    # ---- Fireplace Qu (reporte + costo) ----
    fp_cost_report = 0.0
    try:
        fp_base = str(base.row.get("Fireplace Qu", "No aplica")).strip()
        has_fp_base = 0 if fp_base in ["No aplica","NA"] else 1

        def _pick_fp() -> str | None:
            for nm in ["Po","Fa","TA","Gd","Ex"]:
                v = _getv(m, f"x_fp_is_{nm}", f"fp_is_{nm}")
                if v is not None and v.X > 0.5:
                    return nm
            # si no hay binarios (NA), mantenemos base
            return "No aplica" if has_fp_base == 0 else fp_base

        fp_new = _pick_fp()

        if has_fp_base == 1 and (fp_new != fp_base):
            c = ct.fireplace_cost(fp_new)
            cambios_costos.append(("Fireplace Qu", fp_base, fp_new, c))
            fp_cost_report += c

        if 'opt_dict' in locals():
            opt_dict["Fireplace Qu"] = fp_new
    except Exception:
        pass



    # === total final de costos para reporte ===
    total_cost = (
        float(ct.project_fixed)
        + total_cost_vars
        + float(kitchen_cost)
        + float(util_cost_add)
        + float(elec_extra)
        + float(mvt_extra)
        + float(style_cost)
        + float(matl_cost)
        + float(ext_cost_sum)
        + float(central_air_cost_add)
        + float(exter_qual_cost)
        + float(exter_cond_cost)
        + float(heating_cost_report)
        + float(bsmt_cond_cost_report)
        + float(bsmt_finish_cost_report)
        + float(bsmt_type_cost_report) 
        + float(fp_cost_report)
    )

    # ===== métricas =====
    aumento_utilidad = (precio_remodelada - precio_base) - total_cost

    print("\n===== RESULTADOS DE LA OPTIMIZACIÓN =====")

    # ===== impresión =====
    if aumento_utilidad <= 0:
        print("tu casa ya esta en su punto optimo para tu presupuesto")
        print(f"precio casa base: {money(precio_base)}")
        print(f"precio casa remodelada (optimo hallado): {money(precio_remodelada)}")
        print(f"costos totales de remodelacion: {money(total_cost)}")
    else:
        print(f"Aumento de Utilidad: {money(aumento_utilidad)}")
        print(f"precio casa base: {money(precio_base)}")
        print(f"precio casa remodelada: {money(precio_remodelada)}")
        print(f"costos totales de remodelacion: {money(total_cost)}\n")

        print("Cambios hechos en la casa")
        CAT_FIELDS = {
            "Mas Vnr Type","Roof Style","Roof Matl","Utilities",
            "Electrical","Exterior 1st","Exterior 2nd","Central Air","Heating (tipo)",
            "Heating (reconstruir tipo)", "Heating QC"
        }

        for nombre, b, n, c in cambios_costos:
            if nombre in CAT_FIELDS:
                b_show, n_show = (str(b), str(n))
            else:
                b_show, n_show = (frmt_num(b), frmt_num(n))
            suf = f" (costo {money(c)})" if c > 0 else " (sin costo mapeado)"
            print(f"- {nombre}: en casa base -> {b_show}  ,  en casa nueva -> {n_show}{suf}")

        # ================== SNAPSHOT: Atributos base vs óptimo ==================
        def _qual_ord_to_txt(v: int) -> str:
            MAP = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}
            try:
                iv = int(round(float(v)))
                return MAP.get(iv, str(v))
            except Exception:
                return str(v)

        def _util_ord_to_txt(v: int | float | str) -> str:
            try:
                iv = int(round(float(v)))
            except Exception:
                return str(v)
            return {0:"ELO", 1:"NoSeWa", 2:"NoSewr", 3:"AllPub"}.get(iv, str(v))

        # 1) partimos de la fila original (dict) y clonamos
        base_dict = dict(base.row.items())
        opt_dict  = dict(base.row.items())  # iremos sobre-escribiendo con lo óptimo

        # 2) numéricos que sí modificamos (si existen en el modelo/instancia)
        for nm in ["Bedroom AbvGr","Full Bath","Garage Cars","Total Bsmt SF",
                   "Gr Liv Area","1st Flr SF","2nd Flr SF","Low Qual Fin SF",
                   "Mas Vnr Area","TotRms AbvGrd","Kitchen AbvGr","Half Bath","Wood Deck SF",
                   "BsmtFin SF 1","BsmtFin SF 2","Bsmt Unf SF"]:
            if nm in base_dict and nm in opt:
                try:
                    opt_dict[nm] = float(opt[nm])
                except Exception:
                    pass

        # 3) calidades (si existen)
        for nm in ["Kitchen Qual","Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond",
                   "Heating QC","Fireplace Qu","Garage Qual","Garage Cond","Pool QC"]:
            if nm in base_dict:
                if f"x_{nm}" in [v.VarName for v in m.getVars()]:
                    try:
                        val = m.getVarByName(f"x_{nm}").X
                        opt_dict[nm] = _qual_ord_to_txt(val)
                    except Exception:
                        pass

        # 4) Utilities (ordinal→texto)
        if "Utilities" in base_dict:
            if util_pick is not None:
                opt_dict["Utilities"] = util_pick
            elif "Utilities" in opt:
                opt_dict["Utilities"] = _util_ord_to_txt(opt["Utilities"])

        # 5) Central Air (binaria Y/N)
        try:
            v_yes = m.getVarByName("central_air_yes")
            if v_yes is not None:
                opt_dict["Central Air"] = "Y" if v_yes.X > 0.5 else "N"
        except Exception:
            pass

        # 7) Roof resultado final
        if style_new is not None:
            opt_dict["Roof Style"] = style_new
        if matl_new is not None:
            opt_dict["Roof Matl"] = matl_new

        # 8) Exterior 1st/2nd
        if ex1_new is not None:
            opt_dict["Exterior 1st"] = ex1_new
        if ex2_new is not None:
            opt_dict["Exterior 2nd"] = ex2_new

        # 10) Imprimir snapshot
        keys = sorted(set(base_dict.keys()) | set(opt_dict.keys()))
        rows = []
        for k in keys:
            b = base_dict.get(k, "")
            n = opt_dict.get(k, "")
            def _fmt(v):
                try:
                    fv = float(pd.to_numeric(v, errors="coerce"))
                    if pd.isna(fv):
                        return str(v)
                    return f"{fv:,.2f}" if abs(fv - round(fv)) > 1e-6 else f"{int(round(fv))}"
                except Exception:
                    return str(v)
            rows.append((k, _fmt(b), _fmt(n)))

        df_snapshot = pd.DataFrame(rows, columns=["Atributo", "Base", "Óptimo"])
        pd.set_option("display.max_rows", 9999)
        print("\n=== SNAPSHOT: atributos de la casa (Base vs Óptimo) ===")
        print(df_snapshot.to_string(index=False))

    print("\n===== FIN RESULTADOS DE LA OPTIMIZACIÓN =====")


if __name__ == "__main__":
    main()
