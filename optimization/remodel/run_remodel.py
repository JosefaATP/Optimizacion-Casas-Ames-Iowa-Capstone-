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
    milp = RemodelMILP(base, costs, pool_rules, compat, budget=None)
    milp.build()
    plans = milp.solve_pool(k=30, time_limit=60)

    # 4) evalua utilidades con XGBoost
    predictor = XGBPricePredictor(str(MODELS/"xgb_remodel.pkl"), str(MODELS/"encoders.pkl"))
    ranked = score_plans(predictor, base.features, plans)

    # 5) muestra top-5
    for i, r in enumerate(ranked[:5], 1):
        print(f"[{i}] profit={r['profit']:.0f}  pred_final={r['pred_final']:.0f}  cost={r['plan']['Cost']:.0f}")
        print("    changes:", {k:v for k,v in r["plan"].items() if k not in ("Cost",)})
        print()

if __name__ == "__main__":
    main()
