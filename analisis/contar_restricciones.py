import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import gurobipy as gp

from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.config import PARAMS
from optimization.remodel.io import get_base_house
from optimization.remodel import costs
from optimization.remodel.xgb_predictor import XGBBundle

# ==============================
# CONFIGURACIÓN
# ==============================
PID_EJEMPLO = 526301100   # puedes cambiar por otro PID de tu base
BUDGET = 40000            # o cualquier otro presupuesto
CSV_BASE = "data/processed/base_completa_sin_nulos.csv"

# ==============================
# CONSTRUCCIÓN DEL MODELO
# ==============================
base = get_base_house(PID_EJEMPLO, base_csv=CSV_BASE)
ct = costs.CostTables()
bundle = XGBBundle()

m: gp.Model = build_mip_embed(base.row, BUDGET, ct, bundle)

# ==============================
# INFORMACIÓN DEL MODELO
# ==============================
print(f"🔹 Variables: {m.NumVars}")
print(f"🔹 Restricciones lineales: {m.NumConstrs}")
print(f"🔹 Restricciones cuadráticas: {m.NumQConstrs}")
print(f"🔹 Restricciones generales: {m.NumGenConstrs}")
print(f"🔹 Total aproximado de restricciones: {m.NumConstrs + m.NumQConstrs + m.NumGenConstrs}")

# Si quieres guardar el modelo para inspeccionarlo:
m.write("modelo_restricciones.lp")
print("📄 Archivo 'modelo_restricciones.lp' generado con todas las ecuaciones.")
