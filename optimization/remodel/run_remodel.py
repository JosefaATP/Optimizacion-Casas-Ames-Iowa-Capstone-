# optimization/remodel/run_remodel.py
import yaml
import pandas as pd
from pathlib import Path
from .model_spec import BaseHouse, CostTables, PoolRules, Compat
from .gurobi_model import RemodelMILP
from .xgb_predictor import XGBPricePredictor
from .evaluate import score_plans


# -------------------------------
# Desglose de costo (Δ vs BASE)
# -------------------------------
# Contenido CORREGIDO de la función explain_cost_delta en run_remodel.py

def explain_cost_delta(plan, base_features, costs):
    """
    Devuelve (total, [(item, costo_delta), ...]) sumando SOLO si el valor NUEVO
    es distinto al de la BASE (mismo criterio que el MILP).
    """
    parts = []
    total = 0.0

    # If the MILP produced a Cost_breakdown, prefer that (it was used to compute plan['Cost'])
    if isinstance(plan, dict) and plan.get("Cost_breakdown"):
        try:
            items = plan.get("Cost_breakdown")
            total = float(plan.get("Cost", sum(v for _, v in items if isinstance(v, (int, float)))))
            return total, items
        except Exception:
            # if any problem reading the MILP breakdown, continue to recompute
            pass

    def add(label, val):
        nonlocal total
        try:
            v = float(val or 0.0)
        except Exception:
            v = 0.0
        if abs(v) > 1e-6:
            parts.append((label, v))
            total += v

    # --- Utilities ---
    # (Mantenemos el código anterior para Utilities, RoofStyle, RoofMatl, Exterior, etc.)
    new = (plan.get("Utilities") or [None])[0]
    base_u = str(base_features.get("Utilities"))
    if new and new != base_u:
        add(f"Utilities: {base_u} -> {new}", costs.utilities.get(new, 0.0))

    # --- Roof Style ---
    new = (plan.get("RoofStyle") or [None])[0]
    base_rs = str(base_features.get("Roof Style"))
    if new and new != base_rs:
        add(f"RoofStyle: {base_rs} -> {new}", costs.roof_style.get(new, 0.0))

    # --- Roof Matl ---
    new = (plan.get("RoofMatl") or [None])[0]
    base_rm = str(base_features.get("Roof Matl"))
    if new and new != base_rm:
        add(f"RoofMatl: {base_rm} -> {new}", costs.roof_matl.get(new, 0.0))

    # --- Exterior 1st ---
    new = (plan.get("Exterior1st") or [None])[0]
    base_e1 = str(base_features.get("Exterior 1st"))
    if new and new != base_e1:
        add(f"Exterior1st: {base_e1} -> {new}", costs.exterior1st.get(new, 0.0))

    # --- Exterior 2nd (si existe en base) ---
    base_e2 = str(base_features.get("Exterior 2nd") or "")
    tmp = (plan.get("Exterior2nd") or [])
    new = tmp[0] if tmp else ""
    if base_e2 != "" and new != base_e2:
        add(f"Exterior2nd: {base_e2} -> {new}", costs.exterior2nd.get(new, 0.0))
    # ... (Otras variables categóricas, como MasVnrType, Electrical, Heating, KitchenQual) ...

    # --- Central Air (SOLUCIÓN AL PROBLEMA DE $65) ---
    # --- Fragmento CRÍTICO para Central Air en explain_cost_delta ---
    # (Asume que base_features es canonicalizada)

    base_ca_raw = str(base_features.get("Central Air")).upper().strip()
    new_ca_raw = str(plan.get("CentralAir", base_ca_raw)).upper().strip()
    costs_central_air_install = costs.central_air_install # Cargar el costo de YAML

    if base_ca_raw in ("N", "NO") and new_ca_raw == "YES":
        # El modelo de Gurobi cobra esto si es binario. El valor viene de YAML.
        add("CentralAir: Instalar (N/No -> Yes)", costs_central_air_install)
        
    # --- Piscina (Si se añade, se cobra, pero PoolArea = 0.0 en tu output, por eso no cobra) ---
    area = float(plan.get("PoolArea", 0.0) or 0.0)
    if area > 1e-6: # Solo cobra si el área es significativamente mayor a cero
        # Usamos la lógica de Gurobi, que cobra el área total
        add(f"PoolArea: Construcción ({area:.1f} ft2)", costs.cost_pool_ft2 * area)

    # --- Sótano (Si se termina, se cobra, pero x_b1/x_b2 = 0.0 en tu output, por eso no cobra) ---
    x_b1 = plan.get("x_to_BsmtFin1", 0.0)
    x_b2 = plan.get("x_to_BsmtFin2", 0.0)
    total_bsmt_added = x_b1 + x_b2

    if total_bsmt_added > 1e-6:
         add(f"Bsmt: Terminar {total_bsmt_added:.1f} ft2", costs.cost_finish_bsmt_ft2 * total_bsmt_added)


    # Devolvemos el total del desglose y los ítems
    return total, parts

# -------------------------------
# Rutas de proyecto
# -------------------------------
BASE   = Path(__file__).resolve().parents[2]
MODELS = BASE / "models"
COSTS  = BASE / "costs"


# -------------------------------
# Loaders
# -------------------------------
def load_costs():
    mats_path = COSTS / "materials.yaml"
    print("[DEBUG] Cargando costos desde:", mats_path)
    return CostTables.from_yaml(mats_path)


def load_compat():
    df = pd.read_csv(COSTS / "compat_roof.csv")  # columnas: style, material, allowed(0/1)
    forbidden = set((r.style, r.material) for _, r in df.iterrows() if getattr(r, "allowed", 1) == 0)
    return Compat(roof_style_by_matl_forbidden=forbidden)


def load_pool_rules():
    rules = yaml.safe_load(open(COSTS / "pool.yaml", "r", encoding="utf-8"))
    return PoolRules(
        max_share=rules.get("max_share", 0.20),
        min_area_per_quality=rules.get("min_area_per_quality", 100),
    )


# -------------------------------
# Main
# -------------------------------
def main():
    # 1) Casa base (una fila ya procesada)
    base_row = pd.read_csv(BASE / "data/processed/one_house_bad.csv").iloc[0].to_dict()
    base_house = BaseHouse(features=base_row)
    base_features = base_house.features

    # 2) Costos y reglas
    costs = load_costs()
    compat = load_compat()
    pool_rules = load_pool_rules()

    # 3) MILP generador (maximizar utilidad = pred_new - pred_base - costo)
    predictor = XGBPricePredictor()
    milp = RemodelMILP(base_house, costs, pool_rules, compat, budget=200000, predictor=predictor)
    milp.build()
    plans = milp.solve_pool(k=30, time_limit=60)

# añade plan base que compite explícitamente
    plans.insert(0, {
        "__baseline__": True,
        "Utilities":   [base_features.get("Utilities")],
        "RoofStyle":   [base_features.get("Roof Style")],
        "RoofMatl":    [base_features.get("Roof Matl")],
        "Exterior1st": [base_features.get("Exterior 1st")],
        "Exterior2nd": [base_features.get("Exterior 2nd","")],
        "MasVnrType":  [base_features.get("Mas Vnr Type","None")],
        "Electrical":  [base_features.get("Electrical")],
        "CentralAir":  base_features.get("Central Air"),
        "Heating":     [base_features.get("Heating")],
        "KitchenQual": [base_features.get("Kitchen Qual")],
        "AddPool":     0,
        "PoolQC":      0,
        "PoolArea":    0.0,
        "Cost":        0.0
    })

    # 4) Evaluación con XGBoost (utilidad = pred_new - pred_base - costo)
    predictor = XGBPricePredictor()  # usa tus modelos por defecto
    ranked = score_plans(predictor, base_features, plans)  # base_pred se calcula dentro

    # 5) Top-K bonito
    top_k = 5
    for i, r in enumerate(ranked[:top_k], 1):
        base_price = r["base_pred"]
        new_price  = r["pred_final"]
        cost       = r["cost"]
        profit     = r["profit"]
        print(f"[{i}] UTILIDAD (USD): {profit:,.0f}")
        print(f"    Precio base pred.: {base_price:,.0f}")
        print(f"    Precio nuevo pred.: {new_price:,.0f}")
        print(f"    Costo remodelación: {cost:,.0f}")
        print("    Cambios:")
        for k, (old, new) in r["changes"].items():
            print(f"      - {k}: {old}  ->  {new}")
        print()

        tot, items = explain_cost_delta(r["plan"], base_features, costs)
        print("    >>> Desglose de costo (Δ vs base):")
        for name, val in items:
            print(f"       - {name}: {val:,.0f}")
        print(f"       = TOTAL: {tot:,.0f}\n")

        # Budget diagnostics (if MILP added debug flags)
        try:
            pb = r.get("plan", {})
            if pb.get("_violates_budget"):
                b = pb.get("_budget_value")
                m = pb.get("_total_cost_from_model")
                reported = pb.get("Cost")
                print(f"    !!! VIOLACIÓN DE PRESUPUESTO: budget={b:,.0f}, model_total={m if m is not None else 'N/A'}, reported Cost={reported:,.0f}\n")
            elif pb.get("_budget_value") is not None:
                b = pb.get("_budget_value")
                m = pb.get("_total_cost_from_model")
                reported = pb.get("Cost")
                print(f"    Presupuesto MILP: {b:,.0f}; model_total={m if m is not None else 'N/A'}; reported Cost={reported:,.0f}\n")
        except Exception:
            pass

        best = ranked[0]
        if best["profit"] <= 0:
            print("👉 Recomendación: NO remodelar (utilidad ≤ 0).")
            print(f"    Precio base pred.: {best['base_pred']:,.0f}")
            print("    Costo remodelación: 0")
            print("    Cambios: ninguno\n")
            return



if __name__ == "__main__":
    main()
