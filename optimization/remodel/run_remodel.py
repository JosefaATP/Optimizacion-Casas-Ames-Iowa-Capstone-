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
def explain_cost_delta(plan, base_features, costs):
    """
    Devuelve (total, [(item, costo_delta), ...]) sumando SOLO si el valor NUEVO
    es distinto al de la BASE (mismo criterio que el MILP).
    """
    parts = []
    total = 0.0

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

    # --- Mas Vnr Type ---
    base_mvt = str(base_features.get("Mas Vnr Type"))
    new = (plan.get("MasVnrType") or [None])[0]
    if new:
        base_mvt_n = "None" if base_mvt in ("No aplica", "NA", "", "None") else base_mvt
        new_n = "None" if str(new).strip() in ("No aplica", "NA", "", "None") else str(new).strip()
        if new_n != base_mvt_n:
            add(f"MasVnrType: {base_mvt} -> {new}", costs.mas_vnr_type.get(new_n, 0.0))

    # --- Electrical ---
    base_el = str(base_features.get("Electrical"))
    new = (plan.get("Electrical") or [None])[0]
    if new and new != base_el:
        add(f"Electrical: {base_el} -> {new}", costs.electrical.get(new, 0.0))

    # --- Central Air (solo cobra si pasamos de No->Yes) ---
    base_ca = str(base_features.get("Central Air"))
    new = plan.get("CentralAir", base_ca)
    if base_ca in ("N", "No") and new == "Yes":
        add("CentralAirInstall", costs.central_air_install)

    # --- Heating ---
    base_h = str(base_features.get("Heating"))
    new = (plan.get("Heating") or [None])[0]
    if new and new != base_h:
        add(f"Heating: {base_h} -> {new}", costs.heating.get(new, 0.0))

    # --- Kitchen Qual ---
    base_kq = str(base_features.get("Kitchen Qual"))
    new = (plan.get("KitchenQual") or [None])[0]
    if new and new != base_kq:
        kcost = costs.kitchen_remodel.get(new, costs.kitchen_qual.get(new, 0.0))
        add(f"Kitchen: {base_kq} -> {new}", kcost)

    # --- Pool ---
    area = float(plan.get("PoolArea", 0.0) or 0.0)
    if area > 0:
        add(f"PoolArea: +{area:.1f} ft2", costs.cost_pool_ft2 * area)

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
    mats = yaml.safe_load(open(COSTS / "materials.yaml", "r", encoding="utf-8"))
    print("[DEBUG] Cargando costos desde:", COSTS / "materials.yaml")
    return CostTables(
        utilities=mats["Utilities"],
        roof_style=mats["RoofStyle"],
        roof_matl=mats["RoofMatl"],
        exterior1st=mats["Exterior1st"],
        exterior2nd=mats["Exterior2nd"],
        mas_vnr_type=mats["MasVnrType"],
        electrical=mats["Electrical"],
        heating=mats["Heating"],
        kitchen_qual=mats["KitchenQual"],
        central_air_install=mats.get("CentralAirInstall", 6000.0),
        cost_finish_bsmt_ft2=mats.get("CBsmt", 20.0),
        cost_pool_ft2=mats.get("PoolCostPerFt2", 70.0),
        cost_addition_ft2=mats.get("Cstr_floor", 110.0),
        cost_demolition_ft2=mats.get("Cdemolition", 2.0),
        kitchen_remodel=mats.get("KitchenRemodel", {}),
    )


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
    base_row = pd.read_csv(BASE / "data/processed/one_house.csv").iloc[0].to_dict()
    base_house = BaseHouse(features=base_row)
    base_features = base_house.features

    # 2) Costos y reglas
    costs = load_costs()
    compat = load_compat()
    pool_rules = load_pool_rules()

    # 3) MILP generador (solo factible) + soluciones
    milp = RemodelMILP(base_house, costs, pool_rules, compat, budget=60000.0)  # usa None si no quieres tope
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

        best = ranked[0]
        if best["profit"] <= 0:
            print("👉 Recomendación: NO remodelar (utilidad ≤ 0).")
            print(f"    Precio base pred.: {best['base_pred']:,.0f}")
            print("    Costo remodelación: 0")
            print("    Cambios: ninguno\n")
            return



if __name__ == "__main__":
    main()
