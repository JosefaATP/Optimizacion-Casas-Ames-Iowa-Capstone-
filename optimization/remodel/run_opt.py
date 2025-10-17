## optimization/remodel/run_opt.py
import argparse
import pandas as pd
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import XGBBundle, _coerce_quality_ordinals_inplace, _coerce_utilities_ordinal_inplace
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
    return f"${v:,.0f}"

def frmt_num(x: float) -> str:
    return f"{x:,.2f}" if abs(x - round(x)) > 1e-6 else f"{int(round(x))}"

# -----------------------------
# Construir fila base con dummies
# -----------------------------
def _row_with_dummies(base_row: pd.Series, feat_order: list[str]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for col in feat_order:
        if col in base_row.index:
            vals[col] = base_row[col]
        else:
            # dummy estilo "<col>_<val>"
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

    # --- DEBUG: Kitchen Qual → precio ---
    if "Kitchen Qual" in feat_order:
        X_dbg = X_base.copy()
        vals = []
        for q in [0, 1, 2, 3, 4]:
            X_dbg.loc[:, "Kitchen Qual"] = q
            vals.append((q, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Kitchen Qual → precio:", vals)

    # --- DEBUG: Utilities → precio ---
    if "Utilities" in feat_order:
        vals = []
        for k, name in enumerate(["ELO", "NoSeWa", "NoSewr", "AllPub"]):
            X_dbg = X_base.copy()
            X_dbg.loc[:, "Utilities"] = k
            vals.append((k, name, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Utilities → precio:", [(k, name, round(p, 2)) for (k, name, p) in vals])

    # --- DEBUG: Roof Style/Matl → precio (si el modelo tiene dummies de techo) ---
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
            print("DEBUG Roof Style → precio:", vals)
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
            print("DEBUG Roof Matl → precio:", vals)
    except Exception:
        pass

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

    # Si no hay solución, salir elegante
    if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        print("no se encontro solucion factible, status:", m.Status)
        return

    # ===== leer precios =====
    precio_remodelada = float(m.getVarByName("y_price").X)
    y_log = float(m.getVarByName("y_log").X)  # por si quieres verlo

    # ===== reconstruir costos de remodelación =====
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
        "Wood Deck SF": _num_base("Wood Deck SF"),
        "Garage Cars": _num_base("Garage Cars"),
        "Total Bsmt SF": _num_base("Total Bsmt SF"),
    }

    def costo_var(nombre: str, base_v: float, nuevo_v: float) -> float:
        delta = nuevo_v - base_v
        if nombre == "Bedroom AbvGr":
            return _pos(delta) * ct.add_bedroom
        if nombre == "Full Bath":
            return _pos(delta) * ct.add_bathroom
        if nombre == "Wood Deck SF":
            return _pos(delta) * ct.deck_per_m2
        if nombre == "Garage Cars":
            return _pos(delta) * ct.garage_per_car
        if nombre == "Total Bsmt SF":
            return _pos(delta) * ct.finish_basement_per_f2
        return 0.0

    # decisiones óptimas (variables “x_*”)
    opt = {f.name: m.getVarByName(f"x_{f.name}").X for f in MODIFIABLE}

    cambios_costos = []
    total_cost_vars = 0.0
    for nombre, base_v in base_vals.items():
        nuevo_v = float(opt.get(nombre, base_v))
        c = costo_var(nombre, base_v, nuevo_v)
        if abs(nuevo_v - base_v) > 1e-9:
            cambios_costos.append((nombre, base_v, nuevo_v, c))
        total_cost_vars += c

    # === costos de cocina por paquetes ===
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

    # === Utilities elegido (binarios util_*) y costo aplicado sólo si cambió ===
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

    # === total final de costos ===
    total_cost = float(ct.project_fixed) + total_cost_vars + float(kitchen_cost) + float(util_cost_add)

    # ===== métricas =====
    aumento_utilidad = (precio_remodelada - precio_base) - total_cost

    # ===== impresión =====
    if aumento_utilidad <= 0:
        print("tu casa ya está en su punto optimo para tu presupuesto")
        print(f"precio casa base: {money(precio_base)}")
        print(f"precio casa remodelada (óptimo hallado): {money(precio_remodelada)}")
        print(f"costos totales de remodelación: {money(total_cost)}")
    else:
        print(f"Aumento de Utilidad: {money(aumento_utilidad)}")
        print(f"precio casa base: {money(precio_base)}")
        print(f"precio casa remodelada: {money(precio_remodelada)}")
        print(f"costos totales de remodelación: {money(total_cost)}\n")

        print("Cambios hechos en la casa")
        for nombre, b, n, c in cambios_costos:
            suf = f" (costo {money(c)})" if c > 0 else " (sin costo mapeado)"
            print(f"- {nombre}: en casa base -> {frmt_num(b)}  ;  en casa nueva -> {frmt_num(n)}{suf}")
        print(f"\nprecio cambios totales: {money(total_cost)}")

        if dTA or dEX:
            print(f"Costos cocina: TA={money(ct.kitchenQual_upgrade_TA)} (x{dTA}), "
                  f"EX={money(ct.kitchenQual_upgrade_EX)} (x{dEX})")

        # Roof elegido (si está en el MIP con códigos)
        try:
            style_idx = int(round(m.getVarByName("style_code").X))
            matl_idx  = int(round(m.getVarByName("matl_code").X))
            style_names = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
            matl_names  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]
            style_new = style_names[style_idx]
            matl_new  = matl_names[matl_idx]
            style_base = str(base.row.get("Roof Style", "Gable"))
            matl_base  = str(base.row.get("Roof Matl",  "CompShg"))
            print(f"Roof Style base→nuevo: {style_base} → {style_new}")
            print(f"Roof Matl  base→nuevo: {matl_base} → {matl_new}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
