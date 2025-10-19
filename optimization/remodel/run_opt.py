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


        # === DEBUG Garage Finish → precio: barrido Fin, RFn, Unf, No aplica ===
    try:
        cols_gf = ["Garage Finish_Fin", "Garage Finish_RFn", "Garage Finish_Unf", "Garage Finish_No aplica"]

        if any(col in X_base.columns for col in cols_gf):
            print("\nDEBUG Garage Finish → precio:")
            for gf in cols_gf:
                # copio la base
                df_test = X_base.copy()

                # apago todas las dummies de Garage Finish
                for c in cols_gf:
                    if c in df_test.columns:
                        df_test.loc[:, c] = 0

                # activo solo una
                if gf in df_test.columns:
                    df_test.loc[:, gf] = 1

                # predigo con el bundle del pipeline embebido
                y_pred = bundle.predict(df_test)[0]
                print(f"   {gf.replace('Garage Finish_', ''):>10}: {y_pred:,.0f}")
    except Exception as e:
        print(f"(debug garage finish omitido: {e})")


    # === DEBUG Pool QC → precio: barrido ordinal (0–4 o Po–Ex) ===
    try:
        if "Pool QC" in X_base.columns:
            print("\nDEBUG Pool QC → precio (ordinal):")
            Xd = X_base.copy()

            # probar distintos niveles de calidad
            niveles = [0, 1, 2, 3, 4]
            etiquetas = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}

            vals = []
            for q in niveles:
                Xd.loc[:, "Pool QC"] = q
                y_pred = float(bundle.predict(Xd).iloc[0])
                vals.append((q, etiquetas[q], y_pred))
                print(f"   {etiquetas[q]:>10}: {y_pred:,.0f}")
        else:
            print("(debug pool qc omitido: columna Pool QC no encontrada en X_base)")
    except Exception as e:
        print(f"(debug pool qc omitido: {e})")


            # === DEBUG Ampliaciones → precio: barrido +10%, +20%, +30% ===
    try:
        AMPL_COMPONENTES = [
            "Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
            "3Ssn Porch", "Screen Porch", "Pool Area"
        ]

        print("\nDEBUG Ampliaciones → precio:")
        for comp in AMPL_COMPONENTES:
            if comp not in X_base.columns:
                continue
            base_val = float(pd.to_numeric(X_base.loc[0, comp], errors="coerce") or 0.0)
            if base_val <= 0:
                continue

            X_dbg = X_base.copy()
            vals = []
            for pct in [10, 20, 30]:
                X_dbg.loc[0, comp] = base_val * (1 + pct / 100)
                y_pred = float(bundle.predict(X_dbg).iloc[0])
                vals.append((pct, y_pred))
                X_dbg.loc[0, comp] = base_val  # restaurar

            print(f"  {comp:>15}: base={base_val:,.0f} → "
                  + ", ".join([f"+{p}%={v:,.0f}" for p, v in vals]))
    except Exception as e:
        print(f"(debug ampliaciones omitido: {e})")

    
# === DEBUG Garage Qual / Garage Cond → precio (ordinal 0–4 o Po–Ex) ===
    try:
        if "Garage Qual" in X_base.columns and "Garage Cond" in X_base.columns:
            print("\nDEBUG GarageQual / GarageCond → precio:")

            # Copia base
            Xd = X_base.copy()

            # Definición de niveles
            niveles = [0, 1, 2, 3, 4]
            etiquetas = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}

            # Resultados individuales
            print("   → Garage Qual:")
            for q in niveles:
                Xd.loc[:, "Garage Qual"] = q
                y_pred = float(bundle.predict(Xd).iloc[0])
                print(f"      {etiquetas[q]:>8}: {y_pred:,.0f}")

            print("   → Garage Cond:")
            for q in niveles:
                Xd.loc[:, "Garage Cond"] = q
                y_pred = float(bundle.predict(Xd).iloc[0])
                print(f"      {etiquetas[q]:>8}: {y_pred:,.0f}")
        else:
            print("(debug garage qual/cond omitido: columnas no encontradas en X_base)")
    except Exception as e:
        print(f"(debug garage qual/cond omitido: {e})")



    # ============ FIN DEBUGS ============


    # ===== construir MIP =====
    m: gp.Model = build_mip_embed(base.row, args.budget, ct, bundle)

    # Ajustes de resolución
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = PARAMS.time_limit
    m.Params.LogToConsole = PARAMS.log_to_console

    # ===== objetivo como (Final - Inicial - Costos) =====
    m.ObjCon = -precio_base


    '''# 1) Caso libre (como ahora)
    m.optimize()
    obj_free = m.ObjVal if m.Status == gp.GRB.OPTIMAL else None
    y_free = m.getVarByName("central_air_yes").X if m.getVarByName("central_air_yes") else None

    # 2) Caso forzado Y
    m2 = m.copy()
    v_yes2 = m2.getVarByName("central_air_yes")
    if v_yes2 is not None:
        m2.addConstr(v_yes2 == 1, name="DEBUG_force_CA_Y")
    m2.optimize()
    obj_y = m2.ObjVal if m2.Status == gp.GRB.OPTIMAL else None
    y_forced = v_yes2.X if v_yes2 is not None else None

    print(f"[DEBUG] Obj libre={obj_free}, CA_yes={y_free} || Obj forzado Y={obj_y}, CA_yes={y_forced}")'''

    # Optimizar
    m.optimize()

    # ===== chequear factibilidad =====
    if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        print("no se encontro solucion factible, status:", m.Status)
        # Intento de IIS (opcional)
        try:
            if m.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                print("Intentando IIS (conjunto mínimo de restricciones en conflicto)...")
                m.computeIIS()

                viols = []

                # Lineales
                try:
                    for c in m.getConstrs():
                        if c.getAttr(gp.GRB.Attr.IISConstr):
                            viols.append(c.ConstrName)
                except Exception:
                    pass

                # GenConstr (si tu versión lo soporta)
                try:
                    for gc in m.getGenConstrs():
                        try:
                            if gc.getAttr(gp.GRB.Attr.IISConstr):
                                viols.append(gc.GenConstrName)
                        except Exception:
                            # Algunas builds no exponen IIS en GenConstr
                            pass
                except Exception:
                    pass

                # Bounds
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

                # Guardar modelo para depurar
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
        "Total Bsmt SF": _num_base("Total Bsmt SF"),
    }

    def costo_var(nombre: str, base_v: float, nuevo_v: float) -> float:
        delta = nuevo_v - base_v
        if nombre == "Bedroom AbvGr":
            return _pos(delta) * ct.add_bedroom
        if nombre == "Full Bath":
            return _pos(delta) * ct.add_bathroom
        if nombre == "Garage Cars":
            return _pos(delta) * ct.garage_per_car
        if nombre == "Total Bsmt SF":
            return _pos(delta) * ct.finish_basement_per_f2
        return 0.0

    def _to_float_safe(val) -> float:
        try:
            v = float(pd.to_numeric(val, errors="coerce"))
            return 0.0 if pd.isna(v) else v
        except Exception:
            return 0.0

    # decisiones óptimas (x_*)
    opt = {f.name: m.getVarByName(f"x_{f.name}").X for f in MODIFIABLE}

    cambios_costos = []

    # costos numéricos mapeados
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
        dTA = int(round(m.getVarByName("x_delta_KitchenQual_TA").X))
        dEX = int(round(m.getVarByName("x_delta_KitchenQual_EX").X))
    except Exception:
        pass
    try:
        kq_new = int(round(m.getVarByName("x_Kitchen Qual").X))
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

    # ---- Garage Finish (reporte + costo + debug) ----
    try:
        base_gf = {nm: float(base.row.get(f"Garage Finish_{nm}", 0.0)) for nm in ["Fin", "RFn", "Unf", "No aplica"]}
        sol_gf = {nm: m.getVarByName(f"x_garage_finish_is_{nm}").X if m.getVarByName(f"x_garage_finish_is_{nm}") else 0.0
                for nm in ["Fin", "RFn", "Unf", "No aplica"]}
        upg_val = m.getVarByName("x_UpgGarageFinish").X if m.getVarByName("x_UpgGarageFinish") else 0.0

        # detectar categorías (si no hay ninguna activa, usar "No aplica")
        gf_before = next((k for k, v in base_gf.items() if v == 1), "No aplica")
        gf_after = max(sol_gf, key=sol_gf.get) if sol_gf else "No aplica"

        if gf_before != gf_after:
            # si hay cambio, obtener costo
            costo_gf = ct.garage_finish_cost(gf_after)
            cambios_costos.append(("Garage Finish", gf_before, gf_after, costo_gf))
            print(f"Cambio en acabado de garage: {gf_before} → {gf_after} ({money(costo_gf)})")
        else:
            print(f"Sin cambio en acabado de garage (sigue en {gf_before})")

    except Exception as e:
        print("⚠️ (aviso, no crítico) error leyendo resultado de GarageFinish:", e)


        # ---- Pool QC (reporte + costo + debug) ----
    try:
        pool_area = float(pd.to_numeric(base.row.get("Pool Area"), errors="coerce") or 0.0)

        # si es ordinal (0–4) en lugar de dummies:
        if "Pool QC" in base.row.index:
            MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
            revMAP = {v: k for k, v in MAP.items()}

            pq_before_val = MAP.get(str(base.row.get("Pool QC", "No aplica")), None)
            v_pool = m.getVarByName("x_Pool QC")

            if v_pool is not None:
                pq_after_val = int(round(v_pool.X))
                pq_before = revMAP.get(pq_before_val, "No aplica")
                pq_after = revMAP.get(pq_after_val, "No aplica")

                if pq_before != pq_after:
                    costo_pq = ct.poolqc_costs.get(pq_after, 0.0) + ct.pool_area_cost * pool_area
                    cambios_costos.append(("Pool QC", pq_before, pq_after, costo_pq))
                    print(f"Cambio en calidad de piscina: {pq_before} → {pq_after} ({money(costo_pq)})")
                else:
                    print(f"Sin cambio en calidad de piscina (sigue en {pq_before})")
            else:
                print("(info) variable x_Pool QC no presente en modelo")
        else:
            print("(info) columna Pool QC no está en base.row")
    except Exception as e:
        print(f"⚠️ (aviso, no crítico) error leyendo resultado de Pool QC:", e)


        # ========== AMPLIACIONES Y AGREGADOS (post-solve) ==========
    try:
        # --- Parámetros fijos ---
        A_Full, A_Half, A_Kitch, A_Bed = 40.0, 20.0, 75.0, 70.0

        # --- Binarios de agregados ---
        agregados = {
            "AddFull": ("Full Bath", A_Full, ct.construction_cost),
            "AddHalf": ("Half Bath", A_Half, ct.construction_cost),
            "AddKitch": ("Kitchen", A_Kitch, ct.construction_cost),
            "AddBed": ("Bedroom", A_Bed, ct.construction_cost),
        }

        for key, (nombre, area, costo_unit) in agregados.items():
            var = m.getVarByName(f"x_{key}")
            if var and var.X > 0.5:
                costo_total = costo_unit * area
                cambios_costos.append((nombre, "sin", "agregado", costo_total))
                print(f"agregado: {nombre} (+{area:.0f} ft²) → costo {money(costo_total)}")

        # --- Ampliaciones ---
        AMPL_COMPONENTES = [
            "Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
            "3Ssn Porch", "Screen Porch", "Pool Area"
        ]
        COSTOS = {10: ct.ampl10_cost, 20: ct.ampl20_cost, 30: ct.ampl30_cost}

        for comp in AMPL_COMPONENTES:
            base_val = float(pd.to_numeric(base.row.get(comp), errors="coerce") or 0.0)
            if base_val <= 0:
                continue

            for pct in [10, 20, 30]:
                v = m.getVarByName(f"x_z{pct}_{comp.replace(' ', '')}")
                if v and v.X > 0.5:
                    delta = base_val * pct / 100
                    costo = COSTOS[pct] * delta
                    cambios_costos.append((f"{comp} (+{pct}%)", base_val, base_val + delta, costo))
                    print(f"ampliación: {comp} +{pct}% (+{delta:.1f} ft²) → costo {money(costo)}")
                    break  # sólo una por componente

    except Exception as e:
        print(f"⚠️ error leyendo ampliaciones/agregados: {e}")

    
        # ========== GARAGE QUAL / COND (post-solve) ==========
    try:
        G_CATS = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]

        # --- funciones auxiliares ---
        def _get_base(attr, g):
            return float(base_row.get(f"{attr}_{g}", 0.0) or 0.0)

        def _get_new_val(varname):
            v = m.getVarByName(varname)
            return v.X if v else 0.0

        def _find_selected(attr_prefix):
            for g in G_CATS:
                v = m.getVarByName(f"x_{attr_prefix}_is_{g}")
                if v and v.X > 0.5:
                    return g
            return "No aplica"

        # --- leer categorías base ---
        base_qual = next((g for g in G_CATS if _get_base("Garage Qual", g) == 1), "No aplica")
        base_cond = next((g for g in G_CATS if _get_base("Garage Cond", g) == 1), "No aplica")

        # --- categorías nuevas ---
        new_qual = _find_selected("garage_qual")
        new_cond = _find_selected("garage_cond")

        # --- costos ---
        def _cost(g):
            return ct.garage_qc_costs.get(g, 0.0)

        cost_qual = _cost(new_qual) if new_qual != base_qual else 0.0
        cost_cond = _cost(new_cond) if new_cond != base_cond else 0.0

        # --- impresión por consola ---
        print("\nCambios en GarageQual / GarageCond:")
        print(f"  GarageQual: {base_qual} → {new_qual}  "
              f"({'sin cambio' if base_qual == new_qual else f'costo ${cost_qual:,.0f}'})")
        print(f"  GarageCond: {base_cond} → {new_cond}  "
              f"({'sin cambio' if base_cond == new_cond else f'costo ${cost_cond:,.0f}'})")

        # --- registrar en resumen final ---
        if base_qual != new_qual:
            cambios_costos.append(
                ("GarageQual", base_qual, new_qual, cost_qual)
            )
        if base_cond != new_cond:
            cambios_costos.append(
                ("GarageCond", base_cond, new_cond, cost_cond)
            )

    except Exception as e:
        print(f"(post-solve garage qual/cond omitido: {e})")
    # ================== FIN AMPLIACIONES Y AGREGADOS ==========




    # ---- Mas Vnr Type ----
    MVT_NAMES = ["BrkCmn", "BrkFace", "CBlock", "Stone", "No aplica", "None"]  # soporta ambos labels
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
                # SIEMPRE lo reportamos como cambio (aunque cueste 0)
                cambios_costos.append(("Central Air", base_air, pick, central_air_cost_add))
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
            "Electrical","Exterior 1st","Exterior 2nd","Central Air"
        }

        for nombre, b, n, c in cambios_costos:
            if nombre in CAT_FIELDS:
                b_show, n_show = (str(b), str(n))
            else:
                b_show, n_show = (frmt_num(b), frmt_num(n))
            suf = f" (costo {money(c)})" if c > 0 else " (sin costo mapeado)"
            print(f"- {nombre}: en casa base -> {b_show}  ,  en casa nueva -> {n_show}{suf}")

        '''# Resumen roof elegido
        if style_new:
            print(f"Roof Style base->nuevo: {style_base} -> {style_new}")
        if matl_new:
            print(f"Roof Matl  base->nuevo: {matl_base} -> {matl_new}")

        # Resumen electrical (si quieres ver estado final aunque no cambie)
        try:
            picked = None
            for nm in ["SBrkr","FuseA","FuseF","FuseP","Mix"]:
                v = m.getVarByName(f"x_elect_is_{nm}")
                if v is not None and v.X > 0.5:
                    picked = nm
                    break
            base_elec = str(base.row.get("Electrical", "SBrkr"))
            if picked:
                print(f"Electrical base->nuevo: {base_elec} -> {picked}")
        except Exception:
            pass'''


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
                "Mas Vnr Area","TotRms AbvGrd","Kitchen AbvGr","Half Bath","Wood Deck SF"]:
            if nm in base_dict:
                if nm in opt:  # variable de decisión
                    try:
                        opt_dict[nm] = float(opt[nm])
                    except Exception:
                        pass

        # 3) calidades (si existen)
        for nm in ["Kitchen Qual","Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond",
                "Heating QC","Fireplace Qu","Garage Qual","Garage Cond","Pool QC"]:
            if nm in base_dict:
                # si está como x_var, úsalo; si no, mantenemos base
                if f"x_{nm}" in [v.VarName for v in m.getVars()]:
                    try:
                        val = m.getVarByName(f"x_{nm}").X
                        opt_dict[nm] = _qual_ord_to_txt(val)
                    except Exception:
                        pass
                else:
                    # en tu flujo ya calculaste kq_new para Kitchen Qual
                    if nm == "Kitchen Qual":
                        opt_dict[nm] = _qual_ord_to_txt(kq_new) if 'kq_new' in locals() else base_dict.get(nm)

        # 4) Utilities (ordinal→texto)
        if "Utilities" in base_dict:
            # preferimos lo que detectamos por binarios util_*
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

        # 6) Electrical (categoría final)
        try:
            elect_final = None
            for nm in ["SBrkr","FuseA","FuseF","FuseP","Mix"]:
                v = m.getVarByName(f"x_elect_is_{nm}")
                if v is not None and v.X > 0.5:
                    elect_final = nm
                    break
            if elect_final:
                opt_dict["Electrical"] = elect_final
        except Exception:
            pass

        # 7) Techos (roof)
        if style_new is not None:
            opt_dict["Roof Style"] = style_new
        if matl_new is not None:
            opt_dict["Roof Matl"] = matl_new

        # 8) Exterior 1st/2nd
        if ex1_new is not None:
            opt_dict["Exterior 1st"] = ex1_new
        if ex2_new is not None:
            opt_dict["Exterior 2nd"] = ex2_new

        # 9) Masonry veneer (tipo + área)
        if mvt_pick is not None:
            # Homogeneiza label None/No aplica si aplica
            if mvt_pick == "None":
                mvt_pick = "No aplica"
            opt_dict["Mas Vnr Type"] = mvt_pick
        if "mv_area" in locals():
            opt_dict["Mas Vnr Area"] = mv_area

        # 10) Armar DataFrame de comparación (todas las llaves visibles)
        keys = sorted(set(base_dict.keys()) | set(opt_dict.keys()))
        rows = []
        for k in keys:
            b = base_dict.get(k, "")
            n = opt_dict.get(k, "")
            # formateo liviano para números
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

        '''# 11) (Opcional) Guardar a CSV
        try:
            out_csv = f"snapshot_pid_{args.pid}.csv"
            df_snapshot.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"(Snapshot guardado en {out_csv})")
        except Exception:
            pass'''

    print("\n===== FIN RESULTADOS DE LA OPTIMIZACIÓN =====")


if __name__ == "__main__":
    main()
