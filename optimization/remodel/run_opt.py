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

from shutil import get_terminal_size

def _termw(default=100):
    try:
        return max(default, get_terminal_size().columns)
    except Exception:
        return default

def box_title(title: str, width: int | None = None):
    w = width or _termw()
    t = f" {title} "
    line = "═" * max(0, w - 2)
    print(f"╔{line}╗")
    mid = (w - 2 - len(t)) // 2
    print(f"║{(' ' * max(0, mid))}{t}{(' ' * max(0, w-2-len(t)-mid))}║")

def box_end(width: int | None = None):
    w = width or _termw()
    print(f"╚{'═' * (w - 2)}╝")

def print_kv(rows: list[tuple[str, str | float | int | None]], pad_left=2, key_w=26, width=None):
    """Imprime pares clave-valor alineados."""
    w = width or _termw()
    for k, v in rows:
        if v is None:
            vtxt = "N/A"
        elif isinstance(v, (int, float)):
            vtxt = f"{v:,.2f}" if abs(v - round(v)) > 1e-6 else f"{int(round(v)):,}"
        else:
            vtxt = str(v)
        print("║ " + " " * pad_left + f"{k:<{key_w}} : {vtxt}".ljust(w - 4) + " ║")

def print_changes_table(cambios: list[tuple[str, object, object, float | None]], width=None):
    """Tabla compacta: Nombre | Base → Nuevo | Costo"""
    if not cambios:
        print("║   (No se detectaron cambios)".ljust((width or _termw()) - 2) + "║")
        return
    w = width or _termw()
    name_w = 28
    val_w  = 22
    head = f"{'Cambio':<{name_w}} | {'Base → Nuevo':<{val_w}} | {'Costo':>12}"
    print("║   " + head.ljust(w - 6) + " ║")
    print("║   " + ("-" * len(head)).ljust(w - 6) + " ║")
    for nombre, base_val, new_val, cost_val in cambios:
        base_txt = f"{base_val}"
        new_txt  = f"{new_val}"
        costo    = "-" if (cost_val is None or abs(cost_val) < 1e-9) else f"${cost_val:,.0f}"
        line = f"{nombre:<{name_w}} | {base_txt} → {new_txt:<{val_w - len(base_txt) - 3}} | {costo:>12}"
        print("║   " + line.ljust(w - 6) + " ║")

def print_snapshot_table(base_row: dict, opt_row: dict | None = None, width=None, max_rows=60):
    """Snapshot base vs óptimo en tabla compacta (limita filas para no inundar la consola)."""
    w = width or _termw()
    keys = sorted(set(base_row.keys()) | set((opt_row or {}).keys()))
    name_w = 28
    val_w  = (w - 8 - name_w) // 2  # espacio para dos columnas de valores
    head = f"{'Atributo':<{name_w}} | {'Base':<{val_w}} | {'Óptimo':<{val_w}}"
    print("║   " + head.ljust(w - 6) + " ║")
    print("║   " + ("-" * len(head)).ljust(w - 6) + " ║")

    def _fmt(v):
        import pandas as _pd
        try:
            fv = float(_pd.to_numeric(v, errors="coerce"))
            if _pd.isna(fv):
                return str(v)
            return f"{fv:,.0f}"
        except Exception:
            s = str(v)
            return s if len(s) <= val_w else s[:val_w-1] + "…"

    shown = 0
    for k in keys:
        if shown >= max_rows:
            print("║   … (más filas ocultas)".ljust(w - 4) + " ║")
            break
        b = _fmt(base_row.get(k, ""))
        n = _fmt((opt_row or {}).get(k, ""))
        line = f"{k:<{name_w}} | {b:<{val_w}} | {n:<{val_w}}"
        print("║   " + line.ljust(w - 6) + " ║")
        shown += 1

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

    # --- base_row robusto ---
    try:
        base_row = base.row
    except AttributeError:
        base_row = base if isinstance(base, pd.Series) else pd.Series(base)

    # ===== precio_base (ValorInicial) con el pipeline COMPLETO =====
    feat_order = bundle.feature_names_in()
    X_base = pd.DataFrame([_row_with_dummies(base_row, feat_order)], columns=feat_order)

    # normalizaciones defensivas
    _coerce_quality_ordinals_inplace(X_base, getattr(bundle, "quality_cols", []))
    _coerce_utilities_ordinal_inplace(X_base)

    precio_base = float(bundle.predict(X_base).iloc[0])

    # --- (debugs opcionales recortados para brevedad) ---
    if "Kitchen Qual" in feat_order:
        X_dbg = X_base.copy()
        vals = []
        for q in [0, 1, 2, 3, 4]:
            X_dbg.loc[:, "Kitchen Qual"] = q
            vals.append((q, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Kitchen Qual -> precio:", vals)

    if "Utilities" in feat_order:
        vals = []
        for k, name in enumerate(["ELO", "NoSeWa", "NoSewr", "AllPub"]):
            X_dbg = X_base.copy()
            X_dbg.loc[:, "Utilities"] = k
            vals.append((k, name, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Utilities -> precio:", [(k, name, round(p, 2)) for (k, name, p) in vals])

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

    # ===== construir MIP =====
    m: gp.Model = build_mip_embed(base_row, args.budget, ct, bundle, base_price=precio_base)

    # Ajustes de resolución
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = PARAMS.time_limit
    m.Params.LogToConsole = PARAMS.log_to_console
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol     = 1e-7
    m.Params.OptimalityTol  = 1e-7
    m.Params.NumericFocus   = 3

    # Lock "Garage Cars" al valor base si existe como decisión
    v_gc = m.getVarByName("x_Garage Cars")
    if v_gc is not None:
        base_gc = int(pd.to_numeric(base_row.get("Garage Cars"), errors="coerce") or 0)
        v_gc.LB = base_gc
        v_gc.UB = base_gc

    # Optimizar
    m.optimize()

    # ===== leer precios =====
    precio_remodelada = float(m.getVarByName("y_price").X)
    y_log = float(m.getVarByName("y_log").X)

    # ===== reconstruir costos de remodelación (para reporte) =====
    # Inicialización de TODOS los acumuladores y listas (evita NameError)  
    exter_qual_cost = 0.0
    exter_cond_cost = 0.0
    garage_finish_cost_report = 0.0
    garage_qc_cost_report = 0.0
    fence_cost_report = 0.0
    pool_qc_cost_report = 0.0
    kitchen_cost = 0.0
    util_cost_add = 0.0
    elec_extra = 0.0
    mvt_extra = 0.0
    style_cost = 0.0
    matl_cost = 0.0
    central_air_cost_add = 0.0
    heating_cost_report = 0.0
    bsmt_cond_cost_report = 0.0
    bsmt_finish_cost_report = 0.0
    bsmt_type_cost_report = 0.0
    fp_cost_report = 0.0
    ampl_cost_report = 0.0
    agregados_cost_report = 0.0

    cambios_costos = []
    cambios_categoricos = []

    try:
        cambios_categoricos
    except NameError:
        cambios_categoricos = []

    def _pos(v: float) -> float:
        return v if v > 0 else 0.0

    def _num_base(name: str) -> float:
        try:
            return float(pd.to_numeric(base_row.get(name), errors="coerce"))
        except Exception:
            return 0.0

    base_vals = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
    }

    def _to_float_safe(val) -> float:
        try:
            v = float(pd.to_numeric(val, errors="coerce"))
            return 0.0 if pd.isna(v) else v
        except Exception:
            return 0.0

    # decisiones óptimas (x_*)
    opt = {f.name: m.getVarByName(f"x_{f.name}").X for f in MODIFIABLE if m.getVarByName(f"x_{f.name}") is not None}

    # flags de agregados
    def _bx(nm):
        v = m.getVarByName(f"x_{nm}")
        return (float(v.X) if v is not None else 0.0)

    add_flags = {
        "AddFull":  _bx("AddFull"),
        "AddHalf":  _bx("AddHalf"),
        "AddKitch": _bx("AddKitch"),
        "AddBed":   _bx("AddBed"),
    }

    # costos numéricos (protección por si cambias dormitorio sin usar AddBed)
    def costo_var(nombre: str, base_v: float, nuevo_v: float) -> float:
        delta = nuevo_v - base_v
        if nombre == "Full Bath":
            return 0.0
        if nombre == "Bedroom AbvGr":
            if add_flags.get("AddBed", 0.0) >= 0.5:
                return 0.0
            return _pos(delta) * ct.add_bedroom
        return 0.0

    total_cost_vars = 0.0
    for nombre, base_v in base_vals.items():
        nuevo_v = _to_float_safe(opt.get(nombre, base_v))
        base_vf = _to_float_safe(base_v)
        c = costo_var(nombre, base_vf, nuevo_v)
        if abs(nuevo_v - base_vf) > 1e-9:
            cambios_costos.append((nombre, base_vf, nuevo_v, c))
        total_cost_vars += c

    # ---- Kitchen Qual (one-hot coherente con el MIP) ----
    def _k_ord(txt) -> int:
        M = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        try:
            return int(txt)
        except Exception:
            return M.get(str(txt).strip(), 2)

    kq_base = _k_ord(base_row.get("Kitchen Qual", "TA"))
    kq_pick = None
    for nm in ["Po","Fa","TA","Gd","Ex"]:
        v = m.getVarByName(f"kit_is_{nm}")
        if v is not None and v.X > 0.5:
            kq_pick = nm
            break

    if kq_pick is not None:
        ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        if ORD[kq_pick] > kq_base:
            try:
                kitchen_cost = float(ct.kitchen_level_cost(kq_pick))
            except Exception:
                kitchen_cost = 0.0
            cambios_costos.append(("Kitchen Qual", kq_base, ORD[kq_pick], float(kitchen_cost)))

    # ---- Utilities (solo si cambias) ----
    base_util_name = str(base_row.get("Utilities", "ELO"))
    if base_util_name not in UTIL_TO_ORD:
        base_util_name = ORD_TO_UTIL.get(_safe_util_ord(base_row.get("Utilities")), "ELO")

    util_pick = None
    for nm in ["ELO", "NoSeWa", "NoSewr", "AllPub"]:
        v = m.getVarByName(f"util_{nm}")
        if v is not None and v.X > 0.5:
            util_pick = nm
            break

    if util_pick is not None and util_pick != base_util_name:
        util_cost_add = ct.util_cost(util_pick)
        cambios_costos.append(("Utilities", base_util_name, util_pick, float(util_cost_add)))

    # ---- Roof elegido + costos (lumpsum para material) ----
    style_names = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
    matl_names  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]

    def _pick_active(prefix, names):
        for nm in names:
            v = _getv(m, f"x_{prefix}{nm}", f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    def _norm_lbl(s: str) -> str:
        s = str(s).strip()
        return "CemntBd" if s == "CmentBd" else s

    style_new = _pick_active("roof_style_is_", style_names) or _norm_lbl(base_row.get("Roof Style", "Gable"))
    matl_new  = _pick_active("roof_matl_is_", matl_names)  or _norm_lbl(base_row.get("Roof Matl",  "CompShg"))

    style_base = _norm_lbl(base_row.get("Roof Style", "Gable"))
    matl_base  = _norm_lbl(base_row.get("Roof Matl",  "CompShg"))

    if style_new and style_new != style_base:
        try:
            style_cost = float(ct.roof_style_cost(style_new))
        except Exception:
            style_cost = 0.0
        cambios_costos.append(("Roof Style", style_base, style_new, style_cost))
    if matl_new and matl_new != matl_base:
        try:
            matl_cost = float(ct.get_roof_matl_cost(matl_new))
        except Exception:
            try:
                matl_cost = float(ct.roof_matl_cost(matl_new))
            except Exception:
                matl_cost = 0.0
        cambios_costos.append(("Roof Matl", matl_base, matl_new, matl_cost))

    # ---- Exterior 1st/2nd (LUMPSUM: sin área ni demolición aparte) ----
    ext_names = [
        "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard",
        "ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco",
        "VinylSd","Wd Sdng","WdShngl"
    ]

    def _norm_ext(v):
        s = str(v).strip()
        fixes = {"CmentBd": "CemntBd"}
        return fixes.get(s, s)

    def _pick_ext(prefix: str, names: list[str]) -> str | None:
        for nm in names:
            v = _getv(
                m,
                f"x_{prefix}is_{nm}",   # x_ex1_is_AsbShng
                f"{prefix}is_{nm}",     # ex1_is_AsbShng
                f"x_{prefix}{nm}",      # x_ex1_AsbShng (fallback)
                f"{prefix}{nm}"         # ex1_AsbShng  (fallback)
            )
            if v is not None:
                try:
                    if float(v.X) > 0.5:
                        return nm
                except Exception:
                    pass
        return None

    ex1_new  = _pick_ext("ex1_", ext_names)
    ex2_new  = _pick_ext("ex2_", ext_names)
    ex1_base = _norm_ext(base_row.get("Exterior 1st", "VinylSd"))
    ex2_base = _norm_ext(base_row.get("Exterior 2nd", ex1_base))

    ext_cost_sum = 0.0
    if ex1_new and ex1_new != ex1_base:
        c1 = float(ct.ext_mat_cost(ex1_new))
        ext_cost_sum += c1
        cambios_costos.append(("Exterior 1st", ex1_base, ex1_new, c1))
    if ex2_new and ex2_new != ex2_base:
        c2 = float(ct.ext_mat_cost(ex2_new))
        ext_cost_sum += c2
        cambios_costos.append(("Exterior 2nd", ex2_base, ex2_new, c2))

    # ---- Exter Qual / Exter Cond (costos fijos por nivel final) ----
    def _ord_ex(v, default=2):
        MAP = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
        try:
            iv = int(pd.to_numeric(v, errors="coerce"))
            return iv if iv in (0,1,2,3,4) else default
        except Exception:
            return MAP.get(str(v).strip(), default)

    try:
        ord2txt = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}

        # Exter Qual
        v_exq = m.getVarByName("x_Exter Qual")
        if v_exq is not None:
            exq_new  = int(round(float(v_exq.X)))
            exq_base = _ord_ex(base_row.get("Exter Qual", "TA"))
            if exq_new > exq_base:
                exter_qual_cost = float(ct.exter_qual_cost(ord2txt[exq_new]))
                cambios_costos.append(("Exter Qual", ord2txt[exq_base], ord2txt[exq_new], exter_qual_cost))

        # Exter Cond
        v_exc = m.getVarByName("x_Exter Cond")
        if v_exc is not None:
            exc_new  = int(round(float(v_exc.X)))
            exc_base = _ord_ex(base_row.get("Exter Cond", "TA"))
            if exc_new > exc_base:
                exter_cond_cost = float(ct.exter_cond_cost(ord2txt[exc_new]))
                cambios_costos.append(("Exter Cond", ord2txt[exc_base], ord2txt[exc_new], exter_cond_cost))
    except Exception:
        pass


    # ---- Electrical ----
    try:
        picked = None
        for nm in ["SBrkr","FuseA","FuseF","FuseP","Mix"]:
            v = m.getVarByName(f"x_elect_is_{nm}")
            if v is not None and v.X > 0.5:
                picked = nm
                break
        base_elec = _norm_lbl(base_row.get("Electrical", "SBrkr"))
        if picked and picked != base_elec:
            elec_extra = ct.electrical_demo_small + ct.electrical_cost(picked)
            cambios_costos.append(("Electrical", base_elec, picked, elec_extra))
    except Exception:
        pass

    # ========== AMPLIACIONES Y AGREGADOS (post-solve) ==========
    try:
        def _to_float_safe_local(val):
            try:
                v = float(pd.to_numeric(val, errors="coerce"))
                return 0.0 if pd.isna(v) else v
            except Exception:
                return 0.0

        A_Full  = float(getattr(ct, "area_full_bath_std", 40.0))
        A_Half  = float(getattr(ct, "area_half_bath_std", 20.0))
        A_Kitch = float(getattr(ct, "area_kitchen_std",  75.0))
        A_Bed   = float(getattr(ct, "area_bedroom_std",  70.0))

        C_COST  = float(getattr(ct, "construction_cost", 0.0))  # $/ft² genérico

        agregados = {
            "AddFull":  ("Full Bath",     A_Full,  C_COST),
            "AddHalf":  ("Half Bath",     A_Half,  C_COST),
            "AddKitch": ("Kitchen AbvGr", A_Kitch, C_COST),
            "AddBed":   ("Bedroom AbvGr", A_Bed,   C_COST),
        }
        FULL_FIX  = float(getattr(ct, "bath_full_fixture_fixed", 0.0))
        HALF_FIX  = float(getattr(ct, "bath_half_fixture_fixed", 0.0))
        BED_FIX   = float(getattr(ct, "bedroom_finish_fixed", 0.0))
        KITCH_FIX = float(getattr(ct, "add_kitchen_fixed", 0.0))

        for key, (nombre, area, costo_unit) in agregados.items():
            var = m.getVarByName(f"x_{key}")
            if var and var.X > 0.5:
                c_obra = float(costo_unit) * float(area)
                c_fix  = 0.0
                if key == "AddFull":
                    c_fix = FULL_FIX
                elif key == "AddHalf":
                    c_fix = HALF_FIX
                elif key == "AddBed":
                    c_fix = BED_FIX
                elif key == "AddKitch":
                    c_fix = KITCH_FIX

                costo_total = c_obra + c_fix
                agregados_cost_report += costo_total
                cambios_costos.append(("{}" .format(nombre), "sin", "agregado", costo_total))

        AMPL_COMPONENTES = [
            "Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
            "3Ssn Porch", "Screen Porch", "Pool Area"
        ]
        COSTOS = {
            10: float(getattr(ct, "ampl10_cost", 0.0)),
            20: float(getattr(ct, "ampl20_cost", 0.0)),
            30: float(getattr(ct, "ampl30_cost", 0.0)),
        }

        for comp in AMPL_COMPONENTES:
            base_val = float(pd.to_numeric(base_row.get(comp), errors="coerce") or 0.0)
            if base_val <= 0:
                continue
            for pct in [10, 20, 30]:
                v = m.getVarByName(f"x_z{pct}_{comp.replace(' ', '')}")
                if v and v.X > 0.5:
                    delta = base_val * pct / 100.0
                    costo = COSTOS[pct] * delta
                    ampl_cost_report += costo
                    cambios_costos.append((f"{comp} (+{pct}%)", base_val, base_val + delta, costo))
                    print(f"ampliación: {comp} +{pct}% (+{delta:.1f} ft²) → costo {money(costo)}")
                    break  # una sola por componente

        # ===== AMPLIACIONES DIRECTAS 1st/2nd FLOOR (si subieron pies²) =====
        for flr in ["1st Flr SF", "2nd Flr SF"]:
            base_v = _to_float_safe_local(base_row.get(flr))
            new_v  = _to_float_safe_local(opt.get(flr, base_v))
            delta = new_v - base_v
            if delta <= 1e-6:
                continue

            A_Full  = float(getattr(ct, "area_full_bath_std", 40.0))
            A_Half  = float(getattr(ct, "area_half_bath_std", 20.0))
            A_Kitch = float(getattr(ct, "area_kitchen_std",  75.0))
            A_Bed   = float(getattr(ct, "area_bedroom_std",  70.0))
            C_COST  = float(getattr(ct, "construction_cost", 230.0))

            addfull  = (m.getVarByName("x_AddFull")  or 0)
            addhalf  = (m.getVarByName("x_AddHalf")  or 0)
            addkitch = (m.getVarByName("x_AddKitch") or 0)
            addbed   = (m.getVarByName("x_AddBed")   or 0)

            try:
                addfull  = float(addfull.X)  if hasattr(addfull, "X")  else 0.0
                addhalf  = float(addhalf.X)  if hasattr(addhalf, "X")  else 0.0
                addkitch = float(addkitch.X) if hasattr(addkitch, "X") else 0.0
                addbed   = float(addbed.X)   if hasattr(addbed, "X")   else 0.0
            except Exception:
                pass

            delta_explicado_adds = 0.0
            if flr == "1st Flr SF":
                delta_explicado_adds = (
                    A_Full  * addfull +
                    A_Half  * addhalf +
                    A_Kitch * addkitch +
                    A_Bed   * addbed
                )

            delta_directo = max(0.0, delta - delta_explicado_adds)
            if delta_directo > 1e-6:
                c = C_COST * delta_directo
                ampl_cost_report += c
                cambios_costos.append((flr, base_v, base_v + delta_directo, c))

    except Exception as e:
        print(f"⚠️ error leyendo ampliaciones/agregados: {e}")

    # ---- Garage Finish (reporte robusto) ----
    GF_CATS = ["Fin", "RFn", "Unf", "No aplica"]
    garage_finish_cost_report = 0.0  # siempre inicializado

    # ¿La casa tiene garage? (según la base)
    def _has_garage() -> bool:
        try:
            gt = str(base_row.get("Garage Type", "No aplica")).strip()
            if gt in {"NA", "No aplica"}:
                return False
            area = float(pd.to_numeric(base_row.get("Garage Area"), errors="coerce") or 0.0)
            cars = float(pd.to_numeric(base_row.get("Garage Cars"), errors="coerce") or 0.0)
            return (area > 0 or cars > 0)
        except Exception:
            return False

    has_garage = _has_garage()

    def _pick_one_gf(allow_na: bool):
        best_nm, best_x = None, -1.0
        for nm in GF_CATS:
            if not allow_na and nm == "No aplica":
                continue
            v = _getv(m, f"x_garage_finish_is_{nm}", f"garage_finish_is_{nm}")
            if v is not None:
                try:
                    xv = float(v.X)
                except Exception:
                    xv = 0.0
                if xv > best_x:
                    best_x, best_nm = xv, nm
        return best_nm, best_x

    # base normalizado
    gf_base = str(base_row.get("Garage Finish", "No aplica")).strip()
    if gf_base in {"NA", "N/A"}:
        gf_base = "No aplica"

    # Elegir categoría del modelo, pero si hay garage, ignorar "No aplica"
    gf_new, gf_x = _pick_one_gf(allow_na=not has_garage)
    if gf_new is None:
        gf_new = gf_base

    # Si por cualquier razón llegó "No aplica" pero la casa tiene garage, forzar base
    if has_garage and gf_new == "No aplica":
        gf_new = gf_base
        gf_x = 0.0

    # Reportar solo si realmente cambió
    if gf_new != gf_base:
        try:
            gf_cost = float(ct.garage_finish_cost(gf_new))
        except Exception:
            gf_cost = 0.0
        garage_finish_cost_report = gf_cost
        cambios_costos.append(("Garage Finish", gf_base, gf_new, gf_cost))
        cambios_categoricos.append(("Garage Finish", gf_base, gf_new))


    # ---- Pool QC (reporte + costo) ----
    try:
        pool_area = float(pd.to_numeric(base_row.get("Pool Area"), errors="coerce") or 0.0)
        ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        INV = {v: k for k, v in ORD.items()}

        def _base_pq_cat() -> str:
            for tag in ["Po","Fa","TA","Gd","Ex","No aplica","NA"]:
                col = f"Pool QC_{tag}"
                if col in base_row and float(pd.to_numeric(base_row.get(col), errors="coerce") or 0.0) == 1.0:
                    return "No aplica" if tag == "NA" else tag
            if "Pool QC" in base_row:
                raw = base_row.get("Pool QC")
                try:
                    iv = int(pd.to_numeric(raw, errors="coerce"))
                    return INV.get(iv, "No aplica")
                except Exception:
                    s = str(raw).strip()
                    return "No aplica" if s in {"NA", "No aplica"} else (s if s in ORD else "No aplica")
            return "No aplica"

        pq_before = _base_pq_cat()

        v_ord = m.getVarByName("x_Pool QC")
        if v_ord is not None:
            pq_after = INV.get(int(round(v_ord.X)), "No aplica")
        else:
            pq_after = "No aplica"
            for tag in ["Po","Fa","TA","Gd","Ex","No aplica","NA"]:
                v = m.getVarByName(f"x_pool_qc_is_{tag}")
                if v is not None and v.X > 0.5:
                    pq_after = "No aplica" if tag == "NA" else tag
                    break

        if pq_after != pq_before:
            cat_cost = float(getattr(ct, "poolqc_costs", {}).get(pq_after, 0.0))
            area_cost = float(getattr(ct, "pool_area_cost", 0.0)) * pool_area
            costo_pq = cat_cost + area_cost
            pool_qc_cost_report = costo_pq
            cambios_costos.append(("Pool QC", pq_before, pq_after, costo_pq))
            print(f"Cambio en calidad de piscina: {pq_before} → {pq_after} ({money(costo_pq)})")

    except Exception as e:
        print(f"⚠️ (aviso, no crítico) error leyendo resultado de Pool QC: {e}")

    # ========== GARAGE QUAL / COND (post-solve) ==========
    try:
        G_CATS = ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"]

        def _na2noaplica(v):
            s = str(v).strip()
            return "No aplica" if s in {"NA", "No aplica"} else s

        def _qtxt(v):
            M = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}
            try:
                iv = int(pd.to_numeric(v, errors="coerce"))
                if iv in M:
                    return M[iv]
            except Exception:
                pass
            return _na2noaplica(v)

        base_qual = _qtxt(base_row.get("Garage Qual", "No aplica"))
        base_cond = _qtxt(base_row.get("Garage Cond", "No aplica"))

        def _find_selected(prefix: str) -> str:
            for g in G_CATS:
                v = m.getVarByName(f"x_{prefix}_is_{g}")
                if v and v.X > 0.5:
                    return g
            return "No aplica"

        new_qual = _find_selected("garage_qual")
        new_cond = _find_selected("garage_cond")

        def _cost(name: str) -> float:
            return ct.garage_qc_costs.get(name, 0.0)

        cost_qual = _cost(new_qual) if new_qual != base_qual else 0.0
        cost_cond = _cost(new_cond) if new_cond != base_cond else 0.0
        garage_qc_cost_report = cost_qual + cost_cond

        if base_qual != new_qual:
            cambios_costos.append(("GarageQual", base_qual, new_qual, cost_qual))
        if base_cond != new_cond:
            cambios_costos.append(("GarageCond", base_cond, new_cond, cost_cond))

    except Exception as e:
        print(f"(post-solve garage qual/cond omitido: {e})")

    # ========== PAVED DRIVE (post-solve) ==========
    try:
        PAVED_CATS = ["Y", "P", "N"]

        def _find_selected_pd():
            for d in PAVED_CATS:
                v = m.getVarByName(f"x_paved_drive_is_{d}")
                if v and v.X > 0.5:
                    return d
            return "N"

        base_pd = str(base_row.get("Paved Drive", "N")).strip()
        new_pd = _find_selected_pd()

        cost_pd = ct.paved_drive_cost(new_pd) if new_pd != base_pd else 0.0
        if base_pd != new_pd:
            cambios_costos.append(("Paved Drive", base_pd, new_pd, cost_pd))

    except Exception as e:
        print(f"(post-solve paved drive omitido: {e})")

    # ========== FENCE (post-solve) ==========
    try:
        FENCE_CATS = ["GdPrv", "MnPrv", "GdWo", "MnWw", "No aplica"]

        def _na2noaplica(v):
            s = str(v).strip()
            return "No aplica" if s in {"NA", "No aplica"} else s

        def _find_selected_fence():
            for f in FENCE_CATS:
                v = m.getVarByName(f"x_fence_is_{f}")
                if v and v.X > 0.5:
                    return f
            return "No aplica"

        base_f = _na2noaplica(base_row.get("Fence", "No aplica"))
        new_f = _find_selected_fence()
        lot_front = float(pd.to_numeric(base_row.get("Lot Frontage"), errors="coerce") or 0.0)

        cost_f = 0.0
        if base_f == "No aplica" and new_f in ["MnPrv", "GdPrv"]:
            cost_f = ct.fence_build_cost_per_ft * lot_front
        elif new_f != base_f:
            cost_f = ct.fence_category_cost(new_f)

        fence_cost_report = cost_f
        if new_f != base_f:
            cambios_costos.append(("Fence", base_f, new_f, cost_f))

    except Exception as e:
        print(f"(post-solve fence omitido: {e})")

    # ---- Mas Vnr Type ----
    MVT_NAMES = ["BrkCmn", "BrkFace", "CBlock", "Stone", "No aplica", "None"]
    def _pick_mvt_from_model() -> str | None:
        for nm in MVT_NAMES:
            v = m.getVarByName(f"x_mvt_is_{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    try:
        mvt_pick = _pick_mvt_from_model()
        mvt_base = str(base_row.get("Mas Vnr Type", "No aplica")).strip()

        v_mv_area = m.getVarByName("x_Mas Vnr Area")
        if v_mv_area is not None:
            mv_area = float(v_mv_area.X)
        else:
            try:
                mv_area = float(pd.to_numeric(base_row.get("Mas Vnr Area"), errors="coerce") or 0.0)
            except Exception:
                mv_area = 0.0

        if (mvt_pick is not None) and (mvt_pick != mvt_base):
            mvt_extra = ct.mas_vnr_cost(mvt_pick) * mv_area
            cambios_costos.append(("Mas Vnr Type", mvt_base, mvt_pick, mvt_extra))
    except Exception:
        pass

    # ---- Central Air (reporte + costo) ----
    try:
        base_air = "Y" if str(base_row.get("Central Air", "N")).strip() in {"Y","Yes","1","True"} else "N"
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
    try:
        heat_types = ["Floor","GasA","GasW","Grav","OthW","Wall"]
        heat_base = str(base_row.get("Heating","GasA")).strip()

        heat_new = None
        for nm in heat_types:
            v = _getv(m, f"x_heat_is_{nm}", f"heat_is_{nm}")
            if v is not None and v.X > 0.5:
                heat_new = nm
                break

        v_change = _getv(m, "x_heat_change_type", "heat_change_type")
        v_upg    = _getv(m, "x_heat_upg_type",   "heat_upg_type")

        change_type = int(round(v_change.X)) if v_change is not None else 0
        upg_type    = int(round(v_upg.X))    if v_upg    is not None else 0

        if upg_type == 1 and change_type == 0:
            c_same = ct.heating_type_cost(heat_base)
            heating_cost_report += c_same
            cambios_costos.append(("Heating (reconstruir tipo)", heat_base, heat_base, c_same))

        if heat_new and heat_new != heat_base:
            c_new = ct.heating_type_cost(heat_new)
            heating_cost_report += c_new
            cambios_costos.append(("Heating (tipo)", heat_base, heat_new, c_new))

        qc_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
        qc_base_val = _qual_to_ord(base_row.get("Heating QC"), default=2)
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

    # ---- BsmtFin: costo alineado al modelo (tr1/tr2) ----
    try:
        def _num(x, default=0.0):
            try:
                v = float(pd.to_numeric(x, errors="coerce"))
                return default if pd.isna(v) else v
            except Exception:
                return default

        v_tr1 = m.getVarByName("bsmt_tr1")
        v_tr2 = m.getVarByName("bsmt_tr2")

        if v_tr1 is not None or v_tr2 is not None:
            tr1 = float(v_tr1.X) if v_tr1 is not None else 0.0
            tr2 = float(v_tr2.X) if v_tr2 is not None else 0.0
            added = tr1 + tr2
            if added > 1e-9:
                bsmt_finish_cost_report = float(ct.finish_basement_per_f2) * added
                total_base = _num(base_row.get("BsmtFin SF 1"), 0.0) + _num(base_row.get("BsmtFin SF 2"), 0.0)
                total_new  = total_base + added
                if bsmt_finish_cost_report > 1e-9:
                    cambios_costos.append((
                        "Basement finish (ft²)",
                        int(round(total_base)),
                        int(round(total_new)),
                        float(bsmt_finish_cost_report),
                        ))

            b1_var = m.getVarByName("bsmt_fin1")
            b2_var = m.getVarByName("bsmt_fin2")
            bu_var = m.getVarByName("bsmt_unf")

            b1b = _num(base_row.get("BsmtFin SF 1"), 0.0)
            b2b = _num(base_row.get("BsmtFin SF 2"), 0.0)
            bub = _num(base_row.get("Bsmt Unf SF"),   0.0)

            b1n = float(b1_var.X) if b1_var is not None else b1b
            b2n = float(b2_var.X) if b2_var is not None else b2b
            bun = float(bu_var.X) if bu_var is not None else bub

            if abs(b1n - b1b) > 1e-9: cambios_costos.append(("BsmtFin SF 1", b1b, b1n, 0.0))
            if abs(b2n - b2b) > 1e-9: cambios_costos.append(("BsmtFin SF 2", b2b, b2n, 0.0))
            if abs(bun - bub) > 1e-9: cambios_costos.append(("Bsmt Unf SF",  bub, bun, 0.0))

            total_base = int(round(b1b + b2b))
            total_new  = int(round(b1n + b2n))
            if bsmt_finish_cost_report > 1e-9 and total_new != total_base:
                cambios_costos.append((
                    "Basement finish (ft²)",
                    total_base,
                    total_new,
                    float(bsmt_finish_cost_report)
                ))

        else:
            b1_base = _num(base_row.get("BsmtFin SF 1"), 0.0)
            b2_base = _num(base_row.get("BsmtFin SF 2"), 0.0)

            v_b1 = m.getVarByName("bsmt_fin1")
            v_b2 = m.getVarByName("bsmt_fin2")

            b1_new = float(v_b1.X) if v_b1 is not None else b1_base
            b2_new = float(v_b2.X) if v_b2 is not None else b2_base

            c_b1 = max(0.0, b1_new - b1_base) * float(ct.finish_basement_per_f2)
            c_b2 = max(0.0, b2_new - b2_base) * float(ct.finish_basement_per_f2)

            if c_b1 > 0.0: cambios_costos.append(("BsmtFin SF 1", b1_base, b1_new, c_b1))
            if c_b2 > 0.0: cambios_costos.append(("BsmtFin SF 2", b2_base, b2_new, c_b2))

            bsmt_finish_cost_report = c_b1 + c_b2

    except Exception as e:
        print("[BSMT-DEBUG] error construyendo reporte:", e)

    # ---- Bsmt Cond (reporte + costo) ----
    try:
        v_bc = m.getVarByName("x_Bsmt Cond")
        if v_bc is not None:
            bc_new = int(round(v_bc.X))
            inv_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
            def _ord(v):
                M={"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
                try: return int(v)
                except: return M.get(str(v),2)
            bc_base = _ord(base_row.get("Bsmt Cond","TA"))
            if bc_new > bc_base:
                bc_new_txt  = inv_map[bc_new]
                bc_base_txt = inv_map[bc_base]
                try:
                    c_bc = ct.bsmt_cond_cost(bc_new_txt)
                except Exception:
                    c_bc = (bc_new - bc_base) * 2000.0
                bsmt_cond_cost_report += c_bc
                cambios_costos.append(("Bsmt Cond", bc_base_txt, bc_new_txt, c_bc))
    except Exception:
        pass

    # ---- BsmtFin Type 1/2 (reporte + costo) ----
    TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]

    def _pick_cat(prefix: str) -> str | None:
        for nm in TYPES:
            v = _getv(m, f"x_{prefix}{nm}", f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    b1_base = str(base_row.get("BsmtFin Type 1", "No aplica")).strip()
    b2_base = str(base_row.get("BsmtFin Type 2", "No aplica")).strip()

    b1_new = _pick_cat("b1_is_",) or b1_base
    b2_new = _pick_cat("b2_is_",) or b2_base

    def _bsmt_type_cost(nm: str) -> float:
        try:
            return float(ct.bsmt_type_cost(nm))
        except Exception:
            return 0.0

    if b1_new != b1_base:
        c = _bsmt_type_cost(b1_new)
        cambios_costos.append(("BsmtFin Type 1", b1_base, b1_new, c))
        bsmt_type_cost_report += c

    if b2_base != "No aplica" and b2_new != b2_base:
        c = _bsmt_type_cost(b2_new)
        cambios_costos.append(("BsmtFin Type 2", b2_base, b2_new, c))
        bsmt_type_cost_report += c

    # ---- Fireplace Qu (reporte + costo) ----
    try:
        v_fp = m.getVarByName("x_Fireplace Qu")
        if v_fp is not None:
            new_val = int(round(v_fp.X))
            MAPI = {-1:"No aplica", 0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
            base_val = _qual_to_ord(base_row.get("Fireplace Qu"), default=-1)
            base_txt = MAPI.get(base_val, "No aplica")
            new_txt  = MAPI.get(new_val,  "No aplica")
            if base_val >= 0 and new_val != base_val:
                c = ct.fireplace_cost(new_txt)
                fp_cost_report += c
                cambios_costos.append(("Fireplace Qu", base_txt, new_txt, c))
    except Exception:
        pass

    # === total final de costos (DESGLOSE) para diagnóstico ===
    total_cost_reporte = (
        float(ct.project_fixed)
        + float(total_cost_vars)
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
        + float(fence_cost_report)
        + float(garage_qc_cost_report)
        + float(ampl_cost_report)
        + float(agregados_cost_report)
        + float(pool_qc_cost_report)
        + float(garage_finish_cost_report)   # <— FIX: esta es la buena
    )

    # --- Lecturas seguras desde el modelo ---
    if hasattr(m, "_lin_cost_expr"):
        lin_cost_expr = m._lin_cost_expr
    else:
        lin_cost_expr = None

    print("\n" + "="*60)
    print("               RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60 + "\n")

    print("\n[DBG] Costo usado por término (var * coef):")
    if lin_cost_expr is not None:
        used = []
        for i in range(lin_cost_expr.size()):
            v = lin_cost_expr.getVar(i)
            c = float(lin_cost_expr.getCoeff(i))
            x = float(getattr(v, "X", 0.0))
            if x * c > 1e-6:
                used.append((v.VarName, x * c, c, x))
        used.sort(key=lambda t: -t[1])
        for name, val, coef, x in used[:80]:
            print(f"   {name:30s} → {val:,.0f}   (coef={coef:,.0f}, x={x:.2f})")

    # --- helpers seguros para % ---
    def _pct(num, den):
        import pandas as _pd
        try:
            num = float(_pd.to_numeric(num, errors="coerce"))
            den = float(_pd.to_numeric(den, errors="coerce"))
            if abs(den) < 1e-9:
                return None
            return 100.0 * num / den
        except Exception:
            return None

    # Métricas y valores del modelo
    precio_base_val = getattr(m, "_base_price_val", None)
    precio_opt_var  = getattr(m, "_y_price_var", None)
    precio_opt      = precio_opt_var.X if precio_opt_var is not None else None
    total_cost_var  = m.getVarByName("cost_model")
    total_cost_model = float(total_cost_var.X) if total_cost_var is not None else 0.0
    budget_usd      = float(getattr(m, "_budget_usd", 0.0))

    delta_precio = utilidad_incremental = None
    share_final_pct = uplift_base_pct = roi_pct = None

    if (precio_base_val is not None) and (precio_opt is not None):
        delta_precio = precio_opt - precio_base_val
        utilidad_incremental = (precio_opt - total_cost_model) - precio_base_val  # = ObjVal con ObjCon=-y_base
        share_final_pct = _pct((precio_opt - precio_base_val), precio_opt)
        uplift_base_pct = _pct((precio_opt - precio_base_val), precio_base_val)
        roi_pct = _pct(utilidad_incremental, total_cost_model)

    # Slack de presupuesto
    budget_slack = budget_usd - total_cost_model

    print("\n" + "="*60)
    print("               RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60)
    print(f"📍 PID: {base_row.get('PID', 'N/A')} – {base_row.get('Neighborhood', 'N/A')} | Presupuesto: ${args.budget:,.0f}")
    print(f"🧮 Modelo: {m.ModelName if hasattr(m, 'ModelName') else 'Gurobi MIP'}")
    print(f"⏱️ Tiempo total: {getattr(m, 'Runtime', 0.0):.2f}s | MIP Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    print("💰 **Resumen Económico**")
    print(f"  Precio casa base:        ${precio_base_val:,.0f}" if precio_base_val is not None else "  Precio casa base:        N/A")
    print(f"  Precio casa remodelada:  ${precio_opt:,.0f}"      if precio_opt is not None      else "  Precio casa remodelada:  N/A")
    print(f"  Δ Precio:                ${delta_precio:,.0f}"    if delta_precio is not None    else "  Δ Precio:                N/A")
    print(f"  Costos totales (modelo): ${total_cost_model:,.0f}")
    if abs(total_cost_reporte - total_cost_model) > 1e-6:
        print(f"  (nota) desglose reporte: ${total_cost_reporte:,.0f}  → Δ vs modelo = ${total_cost_model - total_cost_reporte:,.0f}")

    obj_val = getattr(m, "ObjVal", None)
    if obj_val is not None:
        print(f"  Valor objetivo (MIP):    ${obj_val:,.2f}   (≡ y_price - total_cost - y_base)")
    else:
        obj_recalc = (precio_opt or 0.0) - total_cost_model - (precio_base_val or 0.0)
        print(f"  Valor objetivo (MIP):    ${obj_recalc:,.2f}   (recalculado)")

    if uplift_base_pct is not None:
        print(f"  Uplift vs base:          {uplift_base_pct:.0f}%   (=(y_price - y_base)/y_base)")
    if share_final_pct is not None:
        print(f"  % del precio final por mejoras: {share_final_pct:.0f}%   (=(y_price - y_base)/y_price)")
    if utilidad_incremental is not None:
        print(f"  ROI (Δ neto $):          ${utilidad_incremental:,.0f}   (=(y_price - cost) - y_base)")
    if roi_pct is not None:
        print(f"  ROI %:                   {roi_pct:.0f}%   (= ROI$/cost)")
    print(f"  Slack presupuesto:       ${budget_slack:,.2f}")

    # ------------------ Diagnóstico ------------------
    print("\n🔍 **Diagnóstico del modelo**")
    changed_bin_vars = []
    for v in m.getVars():
        try:
            if abs(v.LB) <= 1e-9 and abs(v.UB - 1.0) <= 1e-9 and v.X > 0.5:
                changed_bin_vars.append(v.VarName)
        except Exception:
            pass
    print(f"  🔸 Binarias activas: {len(changed_bin_vars)} (ejemplos: {changed_bin_vars[:10]})")

    # ------------------ Cambios (resumen) ------------------
    print("\n🏠 **Cambios hechos en la casa**")
    if cambios_costos:
        def _is_dup_line(name):
            return ("Full Bath" in name and "agregado" in name.lower())
        for nombre, base_val, new_val, cost_val in cambios_costos:
            if _is_dup_line(nombre):
                continue
            suf = f" (costo ${cost_val:,.0f})" if (cost_val is not None and cost_val > 0) else ""
            print(f"  - {nombre}: {base_val} → {new_val}{suf}")
    else:
        print("  (No se detectaron cambios)")

    # ------------------ Snapshot completo Base vs Óptimo ------------------
    print("\n🧾 **Snapshot: atributos Base vs Óptimo (completo)**")
    try:
        base_dict = dict(base_row.items())
        opt_dict = dict(base_row.items())
        if isinstance(opt, dict):
            opt_dict.update(opt)

        # Sobrescribir etiquetas de categorías escogidas
        def _pick_active(prefix, choices):
            for nm in choices:
                v = m.getVarByName(f"x_{prefix}{nm}")
                if v is not None and v.X > 0.5:
                    return nm
            return None
        style_names = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
        matl_names  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
        elec_names  = ["SBrkr","FuseA","FuseF","FuseP","Mix"]
        style_new = _pick_active("roof_style_is_", style_names) or _norm_lbl(base_row.get("Roof Style","Gable"))
        matl_new  = _pick_active("roof_matl_is_", matl_names)  or _norm_lbl(base_row.get("Roof Matl","CompShg"))
        elec_new  = _pick_active("elect_is_",     elec_names)  or _norm_lbl(base_row.get("Electrical","SBrkr"))
        opt_dict["Roof Style"] = style_new
        opt_dict["Roof Matl"]  = matl_new
        opt_dict["Electrical"] = elec_new

        # Garage Finish en snapshot (usa el pick del bloque de garage)
        if 'gf_new' in locals() and gf_new:
            opt_dict["Garage Finish"] = gf_new

        # Exterior 1st / Exterior 2nd en snapshot (usa picks del bloque exterior)
        if ex1_new:
            opt_dict["Exterior 1st"] = ex1_new
        if ex2_new:
            opt_dict["Exterior 2nd"] = ex2_new

        # Central Air
        v_yes = m.getVarByName("central_air_yes")
        if v_yes is not None:
            opt_dict["Central Air"] = "Y" if v_yes.X > 0.5 else "N"

        # Paved Drive (con x_)
        sel_pd = None
        for d in ["Y", "P", "N"]:
            v = m.getVarByName(f"x_paved_drive_is_{d}")
            if v is not None and v.X > 0.5:
                sel_pd = d
                break
        if sel_pd:
            opt_dict["Paved Drive"] = sel_pd
        

        keys = sorted(set(base_dict.keys()) | set(opt_dict.keys()))
        rows = []
        for k in keys:
            b = base_dict.get(k, "")
            n = opt_dict.get(k, "")
            def _fmt(v):
                try:
                    fv = float(pd.to_numeric(v, errors="coerce"))
                    if pd.isna(fv): return str(v)
                    return f"{fv:,.0f}"
                except Exception:
                    return str(v)
            rows.append((k, _fmt(b), _fmt(n)))
        df_snapshot = pd.DataFrame(rows, columns=["Atributo", "Base", "Óptimo"])
        pd.set_option("display.max_rows", 9999)
        print(df_snapshot.to_string(index=False))
    except Exception as e:
        print(f"⚠️  Error al generar snapshot: {e}")

    # ------------------ Métricas del optimizador ------------------
    print("\n📈 **Métricas del optimizador**")
    try:
        tiempo_total = getattr(m, "Runtime", 0.0)
        gap_final = getattr(m, "MIPGap", None)
        print(f"  ⏱️  Tiempo total: {tiempo_total:,.2f} segundos ({tiempo_total/60:.2f} min)")
        if gap_final is not None and gap_final < gp.GRB.INFINITY:
            print(f"  📉 MIP Gap final: {gap_final*100:.2f}%")
        else:
            print("  📉 MIP Gap final: N/D")
    except Exception as e:
        print(f"  ⚠️  No se pudieron calcular métricas: {e}")

    print("\n" + "="*60)
    print("            FIN RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
