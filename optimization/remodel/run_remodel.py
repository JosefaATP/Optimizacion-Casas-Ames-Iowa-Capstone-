import yaml
import pandas as pd
from pathlib import Path
from .model_spec import BaseHouse, CostTables, PoolRules, Compat
from .gurobi_model import RemodelMILP
from .xgb_predictor import XGBPricePredictor
from .evaluate import score_plans

# --- rutas (ajusta a tu repo)
BASE = Path(__file__).resolve().parents[2]
MODELS = BASE / "models"
COSTS  = BASE / "costs"

def load_costs():
    mats = yaml.safe_load(open(COSTS/"materials.yaml","r", encoding="utf-8"))
    return CostTables(
        utilities=mats["Utilities"],
        roof_style=mats["RoofStyle"],
        roof_matl=mats["RoofMatl"],
        exterior1st=mats["Exterior1st"],            # <- NUEVO
        exterior2nd=mats["Exterior2nd"], 
        mas_vnr_type=mats["MasVnrType"],
        electrical=mats["Electrical"],
        central_air_install=mats["CentralAirInstall"],
        heating=mats["Heating"],
        kitchen_qual=mats["KitchenQual"],
    )

def load_compat():
    df = pd.read_csv(COSTS/"compat_roof.csv")  # columnas: style, material, allowed(0/1)
    forbidden = set((r.style, r.material) for _,r in df.iterrows() if r.allowed==0)
    return Compat(roof_style_by_matl_forbidden=forbidden)

def load_pool_rules():
    rules = yaml.safe_load(open(COSTS/"pool.yaml","r", encoding="utf-8"))
    return PoolRules(max_share=rules["max_share"], min_area_per_quality=rules["min_area_per_quality"])

def main():
    # 1) lee una casa base (ej. de un CSV ya preparado)
    base_row = pd.read_csv(BASE/"data/processed/one_house.csv").iloc[0].to_dict()
    base = BaseHouse(features=base_row)

    # 2) carga costos y reglas
    costs = load_costs()
    compat = load_compat()
    pool_rules = load_pool_rules()

    # 3) construye y resuelve el MILP generador (K planes)
    milp = RemodelMILP(base, costs, pool_rules, compat, budget=20000.0)
    milp.build()
    plans = milp.solve_pool(k=30, time_limit=60)

    # 4) evalua utilidades con XGBoost
    predictor = XGBPricePredictor()
    ranked = score_plans(predictor, base.features, plans)

    # 5) muestra top-5
    for i, r in enumerate(ranked[:5], 1):
        print(f"[{i}] profit={r['profit']:.0f}  pred_final={r['pred_final']:.0f}  cost={r['plan']['Cost']:.0f}")
        print("    changes:", {k:v for k,v in r["plan"].items() if k not in ("Cost",)})
        print()

    # ... luego de obtener 'ranked'
    top_k = 5
    for i, r in enumerate(ranked[:top_k], 1):
        base = r["base_pred"]
        newv = r["pred_final"]
        cost = r["cost"]
        prof = r["profit"]
        print(f"[{i}] UTILIDAD (USD): {prof:,.0f}")
        print(f"    Precio base pred.: {base:,.0f}")
        print(f"    Precio nuevo pred.: {newv:,.0f}")
        print(f"    Costo remodelación: {cost:,.0f}")
        print("    Cambios:")
        for k,(old,new) in r["changes"].items():
            print(f"      - {k}: {old}  ->  {new}")
        print()

if __name__ == "__main__":
    main()
