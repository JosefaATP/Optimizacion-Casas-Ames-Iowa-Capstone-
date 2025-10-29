import argparse
import pandas as pd
import gurobipy as gp
import numpy as np

from optimization.construction.gurobi_model import build_base_input_row, build_mip_embed
from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import XGBBundle

from shutil import get_terminal_size

# ============== helpers simples de impresion ==============
def _termw(default=100):
    try:
        return max(default, get_terminal_size().columns)
    except Exception:
        return default

def money(v: float) -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return str(v)
    if pd.isna(f):
        return "-"
    return f"${f:,.0f}"

# ========= debug infeas =========
def debug_infeas(m: gp.Model, tag="construction"):
    m.Params.Presolve = 0
    m.Params.InfUnbdInfo = 1
    m.computeIIS()
    m.write(f"{tag}.ilp")
    m.write(f"{tag}.lp")

    bad_cons = [c for c in m.getConstrs() if c.IISConstr]
    bad_q    = [q for q in m.getQConstrs() if q.IISQConstr]
    bad_gen  = [g for g in m.getGenConstrs() if g.IISGenConstr]
    bad_lb   = [v for v in m.getVars() if v.IISLB]
    bad_ub   = [v for v in m.getVars() if v.IISUB]

    print("\n=== IIS: restricciones en conflicto ===")
    for c in bad_cons: print(" CONSTR:", c.ConstrName)
    for q in bad_q:    print(" QCONSTR:", q.QCName)
    for g in bad_gen:  print(" GENCONSTR:", g.GenConstrName)

    print("\n=== IIS: limites en conflicto ===")
    for v in bad_lb: print(f" LB  : {v.VarName} = {v.LB}")
    for v in bad_ub: print(f" UB  : {v.VarName} = {v.UB}")

# ======== reconstruir X que vio el embed ========
def rebuild_embed_input_df(m, base_X: pd.DataFrame) -> pd.DataFrame:
    Xi = getattr(m, "_X_input", None)
    if Xi is None or Xi.empty:
        return base_X.copy()

    X_in = Xi.copy()
    for c in X_in.columns:
        v = X_in.loc[0, c]
        if hasattr(v, "X"):
            try:
                X_in.loc[0, c] = float(v.X)
            except Exception:
                X_in.loc[0, c] = 0.0
    # respeta orden guardado por el builder
    feat_order = list(getattr(m, "_feat_order", []))
    if feat_order:
        X_in = X_in[feat_order]
    return X_in

# ============== auditorias utiles ==============
def audit_cost_breakdown_vars(m, top=20):
    expr = getattr(m, "_lin_cost_expr", None)
    if expr is None:
        print("[COST-BREAKDOWN] no _lin_cost_expr")
        return
    try:
        vs = expr.getVars(); cs = expr.getCoeffs(); X = m.getAttr("X", vs)
    except Exception:
        print("[COST-BREAKDOWN] no se pudo leer terminos")
        return
    rows = []
    for v, c, x in zip(vs, cs, X):
        try:
            contrib = float(c) * float(x)
        except Exception:
            continue
        if abs(contrib) > 1e-6:
            rows.append((v.VarName, float(c), float(x), contrib))
    rows.sort(key=lambda t: abs(t[3]), reverse=True)
    print("\n[COST-BREAKDOWN] top terminos de costo:")
    for name, c, x, contr in rows[:top]:
        print(f"  {name:<35s} coef={c:>10.4f} * X={x:>10.4f}  => {contr:>10.2f}")

def audit_predict_outside(m, bundle, base_X):
    X_in = rebuild_embed_input_df(m, base_X)
    try:
        y_hat = float(bundle.predict(X_in).iloc[0])
        y_log = float(np.log1p(y_hat))
        print("\n[AUDIT] Outside predict usando EXACTO el X del embed:")
        print(f"  y_hat(outside) = {y_hat:,.2f}  |  y_log(outside) = {y_log:.6f}")
        if hasattr(m, "_y_log_var"):
            print(f"  y_log(MIP)     = {float(m._y_log_var.X):.6f}")
    except Exception as e:
        print(f"[AUDIT] fallo predict outside: {e}")

# ============================== Main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--basecsv", type=str, default=None, help="ruta alternativa al CSV base")
    args = ap.parse_args()

    # datos base, costos y modelo ML
    base = get_base_house(args.pid, base_csv=args.basecsv)
    ct = costs.CostTables()
    bundle = XGBBundle()

    # base_row robusto
    try:
        base_row = base.row
    except AttributeError:
        base_row = base if isinstance(base, pd.Series) else pd.Series(base)

    # X base: solo para sanity y auditorias, no entra a la FO
    X_base = build_base_input_row(bundle, base_row)
    try:
        y_full = float(bundle.predict(X_base).iloc[0])
        y_log_embed = float(bundle.pipe_for_gurobi().predict(X_base)[0])
        y_from_embed = float(np.expm1(y_log_embed))
        print(f"[SANITY] full.predict -> price = {y_full:,.2f}")
        print(f"[SANITY] embed(pre+reg) -> log = {y_log_embed:.6f} | price≈ {y_from_embed:,.2f}")
    except Exception:
        pass

    # construir y resolver el MIP de construccion
    m: gp.Model = build_mip_embed(
        base_row=base_row,
        budget_usd=args.budget,
        ct=ct,
        bundle=bundle,
        base_price=0.0  # no aplica baseline en construccion
    )

    # parametros de resolucion
    m.Params.MIPGap         = PARAMS.mip_gap
    m.Params.TimeLimit      = PARAMS.time_limit
    m.Params.LogToConsole   = PARAMS.log_to_console
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol     = 1e-7
    m.Params.OptimalityTol  = 1e-7
    m.Params.NumericFocus   = 3

    m.optimize()
    audit_cost_breakdown_vars(m, top=30)

    # auditorias de prediccion usando el mismo X_in que vio el embed
    try:
        base_X_for_embed = getattr(m, "_X_base_numeric", X_base)
        audit_predict_outside(m, bundle, base_X_for_embed)
    except Exception:
        pass

    # estado
    st = m.Status
    if st in (gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED):
        print("\n❌ Modelo infeasible o unbounded. Generando IIS…")
        try:
            debug_infeas(m, tag="construction_debug")
        except Exception as e:
            print(f"[WARN] computeIIS fallo: {e}")
        return

    if st not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) or m.SolCount == 0:
        print("\n⚠️ No hay solucion valida, se omite impresion.")
        return

    # lectura segura de solucion
    y_price_var = getattr(m, "_y_price_var", None)
    precio_opt  = float(y_price_var.X) if y_price_var is not None else None
    total_cost_var = m.getVarByName("cost_model")
    if total_cost_var is not None:
        total_cost_model = float(total_cost_var.X)
    else:
        total_cost_model = float(getattr(m, "_lin_cost_expr", gp.LinExpr(0.0)).getValue())

    budget_usd   = float(getattr(m, "_budget_usd", args.budget))
    budget_slack = budget_usd - total_cost_model

    utilidad_neta = None
    roi_pct = None
    if precio_opt is not None:
        utilidad_neta = precio_opt - total_cost_model
        roi_pct = (utilidad_neta / total_cost_model * 100.0) if total_cost_model else None

    # ===== salida compacta =====
    print("\n" + "="*60)
    print("               RESULTADOS DE LA OPTIMIZACION")
    print("="*60 + "\n")

    print(f"📍 PID: {base_row.get('PID', 'N/A')} – {base_row.get('Neighborhood', 'N/A')} | Presupuesto: ${args.budget:,.0f}")
    print(f"🧮 Modelo: {m.ModelName if hasattr(m, 'ModelName') else 'Gurobi MIP'}")
    print(f"⏱️ Tiempo total: {getattr(m, 'Runtime', 0.0):.2f}s | MIP Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    print("💰 **Resumen economico**")
    if precio_opt is not None:
        print(f"  Precio post construccion: ${precio_opt:,.0f}")
    print(f"  Costos totales (modelo):  ${total_cost_model:,.0f}")

    obj_val = getattr(m, "ObjVal", None)
    if obj_val is not None:
        print(f"  Valor objetivo (MIP):     ${obj_val:,.2f}   (≡ y_price - cost)")

    if utilidad_neta is not None:
        print(f"  Utilidad neta:            ${utilidad_neta:,.0f}")
    if roi_pct is not None:
        print(f"  ROI %:                    {roi_pct:.0f}%")
    print(f"  Slack presupuesto:        ${budget_slack:,.2f}")

    # checks rapidos
    cB = m.getConstrByName("BUDGET")
    if cB:
        print(f"\n[CHECK] BUDGET slack = {cB.Slack:.6f}")

if __name__ == "__main__":
    main()
