import argparse
import pandas as pd
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from .costs import CostTables
from .xgb_predictor import XGBBundle
from .gurobi_model import build_mip_embed
from .features import MODIFIABLE


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
    ct = CostTables()
    bundle = XGBBundle()

    # ===== precio_base (ValorInicial) con el pipeline COMPLETO =====
    feat_order = bundle.feature_names_in()
    X_base = pd.DataFrame([{c: base.row[c] for c in feat_order}], columns=feat_order)
    precio_base = float(bundle.predict(X_base).iloc[0])

    # ===== construir MIP =====
    m: gp.Model = build_mip_embed(base.row, args.budget, ct, bundle)

    # Ajustes de resolución
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = PARAMS.time_limit
    m.Params.LogToConsole = PARAMS.log_to_console

    # ===== hacer que el valor del objetivo sea (Final - Inicial - Costos) =====
    # El objetivo actual en el modelo es (y_price - total_cost).
    # Restamos el constante 'precio_base' para que ObjVal = y_price - total_cost - precio_base.
    m.ObjCon = -precio_base

    # Optimizar
    m.optimize()

    # Si no hay solución, salir elegante
    if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        print("no se encontro solucion factible, status:", m.Status)
        return

    # ===== leer precios =====
    precio_remodelada = float(m.getVarByName("y_price").X)   # ValorFinal
    y_log = float(m.getVarByName("y_log").X)                 # por si quieres verlo

    # ===== reconstruir costos de remodelación tal como en el MIP =====
    def _pos(v: float) -> float:
        return v if v > 0 else 0.0

    base_vals = {f.name: float(base.row.get(f.name, 0.0)) for f in MODIFIABLE}

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
            return _pos(delta) * ct.finish_basement_per_m2
        return 0.0

    # decisiones óptimas
    opt = {f.name: m.getVarByName(f"x_{f.name}").X for f in MODIFIABLE}

    cambios_costos = []
    total_cost_vars = 0.0
    for nombre, nuevo_v in opt.items():
        b = base_vals.get(nombre, 0.0)
        c = costo_var(nombre, b, float(nuevo_v))
        if abs(nuevo_v - b) > 1e-9:
            cambios_costos.append((nombre, b, float(nuevo_v), c))
        total_cost_vars += c

    total_cost = float(ct.project_fixed) + total_cost_vars  # C_total

    # ===== métricas solicitadas =====
    # Valor objetivo EXACTO (como en tu PDF): ΔP - C_total
    # ΔP = precio_remodelada - precio_base
    aumento_utilidad = (precio_remodelada - precio_base) - total_cost

    # Por construcción, debe coincidir con m.ObjVal (salvo tolerancias numéricas)
    # print("debug | m.ObjVal:", m.ObjVal, " | aumento_utilidad:", aumento_utilidad)

    # ===== impresión bonita =====
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


if __name__ == "__main__":
    main()
