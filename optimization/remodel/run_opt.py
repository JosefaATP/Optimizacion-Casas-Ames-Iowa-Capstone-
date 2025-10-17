# optimization/remodel/run_opt.py
import argparse
import pandas as pd
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import XGBBundle, _coerce_quality_ordinals_inplace
from .gurobi_model import build_mip_embed
from .features import MODIFIABLE

# ----- utilidades (mapeos) -----
UTIL_TO_ORD = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}
ORD_TO_UTIL = {v: k for k, v in UTIL_TO_ORD.items()}

def _safe_util_ord(val) -> int:
    """Convierte 'val' a ordinal 0..3; default 0 (ELO) si no se puede."""
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.notna(v) and int(v) in (0, 1, 2, 3):
            return int(v)
    except Exception:
        pass
    return UTIL_TO_ORD.get(str(val), 0)

# ----- helpers de impresión -----
def money(v: float) -> str:
    return f"${v:,.0f}"

def frmt_num(x: float) -> str:
    return f"{x:,.2f}" if abs(x - round(x)) > 1e-6 else f"{int(round(x))}"

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
    X_base = pd.DataFrame([{c: base.row[c] for c in feat_order}], columns=feat_order)

    # Normaliza calidades a 0..4
    _coerce_quality_ordinals_inplace(X_base, getattr(bundle, "quality_cols", []))

    # Normaliza Utilities a 0..3 si existe
    if "Utilities" in X_base.columns:
        X_base.loc[0, "Utilities"] = _safe_util_ord(X_base.loc[0, "Utilities"])

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
        for k in [0, 1, 2, 3]:
            X_dbg = X_base.copy()
            X_dbg.loc[:, "Utilities"] = k
            vals.append((ORD_TO_UTIL[k], float(bundle.predict(X_dbg).iloc[0])))
        pretty = [(k, name, round(p, 2)) for k, (name, p) in enumerate([(u, v) for u, v in vals])]
        print("DEBUG Utilities → precio:", pretty)

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
    y_log = float(m.getVarByName("y_log").X)

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
    for nombre, base_v in base_vals.items():  # sólo numéricas mapeadas a costo
        nuevo_v = float(opt.get(nombre, base_v))
        c = costo_var(nombre, base_v, nuevo_v)
        if abs(nuevo_v - base_v) > 1e-9:
            cambios_costos.append((nombre, base_v, nuevo_v, c))
        total_cost_vars += c

    # === costos de cocina por paquetes (leer del modelo) ===
    def _q_to_ord(txt) -> int:
        MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(txt)
        except Exception:
            return MAP.get(str(txt), -1)

    kq_base = _q_to_ord(base.row.get("Kitchen Qual", "TA"))

    dTA = 0
    dEX = 0
    kq_new = kq_base
    try:
        dTA = int(round(m.getVarByName("x_delta_KitchenQual_TA").X))
        dEX = int(round(m.getVarByName("x_delta_KitchenQual_EX").X))
    except Exception:
        pass

    try:
        kq_new = int(round(m.getVarByName("x_Kitchen Qual").X))
    except Exception:
        kq_new = max(kq_base, 2) if dTA else kq_base
        kq_new = max(kq_new, 4) if dEX else kq_new

    kitchen_cost = dTA * ct.kitchenQual_upgrade_TA + dEX * ct.kitchenQual_upgrade_EX
    if (dTA or dEX) and (kq_new != kq_base):
        cambios_costos.append(("Kitchen Qual", kq_base, kq_new, float(kitchen_cost)))

    # === Utilities elegido (leer binarios util_*) y costo aplicado sólo si cambió ===
    util_vars = [v for v in ["util_ELO", "util_NoSeWa", "util_NoSewr", "util_AllPub"]]
    util_pick = None
    for vname in util_vars:
        try:
            v = m.getVarByName(vname)
            if v is not None and v.X > 0.5:
                util_pick = vname.split("util_")[1]
                break
        except Exception:
            pass

    base_util_name = str(base.row.get("Utilities", "ELO"))
    # Normaliza base a etiqueta conocida
    if base_util_name not in UTIL_TO_ORD:
        try:
            base_util_name = ORD_TO_UTIL[_safe_util_ord(base.row.get("Utilities"))]
        except Exception:
            base_util_name = "ELO"

    util_cost_add = 0.0
    if util_pick is not None and util_pick != base_util_name:
        util_cost_add = ct.util_cost(util_pick)
        cambios_costos.append(("Utilities", base_util_name, util_pick, float(util_cost_add)))

    # === total final de costos ===
    total_cost = float(ct.project_fixed) + total_cost_vars + float(kitchen_cost) + float(util_cost_add)

    # ===== métricas solicitadas =====
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

if __name__ == "__main__":
    main()
