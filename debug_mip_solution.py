"""Debug: Verificar solución inconsistente del MIP."""
import gurobipy as gp
import pickle
import sys

pid = 528328100

# Cargar el modelo guardado
try:
    with open(f"debug_model_{pid}.pkl", "rb") as f:
        m = pickle.load(f)
    print(f"[LOADED] Modelo from pickle")
except:
    print("[ERROR] No se pudo cargar debug_model_{pid}.pkl")
    print("Ejecutando optimización primero...")
    import subprocess
    subprocess.run([sys.executable, "-m", "optimization.remodel.run_opt", 
                   f"--pid", str(pid), "--budget", "500000"], check=False)
    sys.exit(1)

# Buscar variables Electrical
m.update()
print("\n[DEBUG] Variables Electrical en el modelo:")
for v in m.getVars():
    if "Electrical" in v.VarName or "elect" in v.VarName.lower():
        try:
            print(f"  {v.VarName:40s} = {v.X:10.6f}  (LB={v.LB}, UB={v.UB})")
        except:
            pass

# Buscar variables Open Porch
print("\n[DEBUG] Variables Open Porch:")
for v in m.getVars():
    if "Open Porch" in v.VarName or "OpenPorch" in v.VarName:
        try:
            print(f"  {v.VarName:40s} = {v.X:10.6f}  (LB={v.LB}, UB={v.UB})")
        except:
            pass

# Buscar variables que comiencen con "Exterior" y fueron cambiadas
print("\n[DEBUG] Variables Exterior en la solución:")
for v in m.getVars():
    if "Exterior" in v.VarName:
        try:
            x_val = v.X
            if abs(x_val) > 1e-6:
                print(f"  {v.VarName:40s} = {x_val:10.6f}")
        except:
            pass

print("\n[CHECK] Verificar constraints one-hot para Electrical:")
for c in m.getConstrs():
    cname = c.ConstrName
    if "Electrical" in cname or "elect" in cname.lower():
        try:
            slack = c.Slack
            print(f"  {cname:50s} slack={slack:12.6f}")
        except:
            pass
