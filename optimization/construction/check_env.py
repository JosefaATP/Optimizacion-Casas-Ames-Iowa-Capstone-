import importlib
mods = ["gurobipy", "gurobi_ml", "xgboost", "sklearn", "pandas", "numpy", "joblib"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"ok: {m}")
    except Exception as e:
        print(f"FALTA: {m} -> {e}")

# chequeo licencia gurobi
try:
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    print("Gurobi OK: version", gp.gurobi.version())
except Exception as e:
    print("Problema con licencia/instalacion de Gurobi:", e)