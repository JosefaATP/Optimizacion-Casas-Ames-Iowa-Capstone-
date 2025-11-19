# optimization/remodel/run_opt.py
import argparse
import pandas as pd
import gurobipy as gp
import numpy as np
import joblib
import os
import sys

# Asegurar que estamos en el directorio correcto del proyecto
from pathlib import Path
project_dir = Path(__file__).parent.parent.parent
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

from optimization.remodel.gurobi_model import build_base_input_row
from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import XGBBundle
from .gurobi_model import build_mip_embed
from .features import MODIFIABLE
from .quality_calculator import QualityCalculator, calculate_overall_qual_from_improvements

from shutil import get_terminal_size


import pandas as _pd
import numpy as _np

def _to_num(x):
    """Convierte a float cuando se puede; deja strings/constantes si no."""
    try:
        return float(_pd.to_numeric(x, errors="coerce"))
    except Exception:
        return x

def rebuild_embed_input_df(m, base_X):
    """
    Toma m._X_input (con objetos gp.Var o constantes),
    reemplaza Vars por su .X, y respeta el orden de features.
    """
    Xi = getattr(m, "_X_input", None)
    if Xi is None or Xi.empty:
        # Fallback sensato: devuelve la base numérica
        return base_X.copy()

    X_in = Xi.copy()
    for c in X_in.columns:
        v = X_in.loc[0, c]
        if hasattr(v, "X"):           # es una gp.Var
            try:
                X_in.loc[0, c] = float(v.X)
            except Exception:
                X_in.loc[0, c] = 0.0
        else:
            X_in.loc[0, c] = _to_num(v)

    # Fuerza el orden exacto que espera el bundle si lo tenemos guardado
    try:
        feat_order = list(getattr(m, "_feat_order", []))
        if feat_order:
            X_in = X_in[feat_order]
    except Exception:
        pass

    return X_in
# ==============================
# Utilidades de impresión
# ==============================
def _termw(default=100):
    try:
        return max(default, get_terminal_size().columns)
    except Exception:
        return default

def box_title(title: str, width: int | None = None):
    w = width or _termw()
    t = f" {title} "
    line = "═" * max(0, w - 2)
    print(f"╔{line}╗")
    mid = (w - 2 - len(t)) // 2
    print(f"║{(' ' * max(0, mid))}{t}{(' ' * max(0, w-2-len(t)-mid))}║")

def box_end(width: int | None = None):
    w = width or _termw()
    print(f"╚{'═' * (w - 2)}╝")

def money(v: float) -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return str(v)
    if pd.isna(f):
        return "-"
    return f"${f:,.0f}"

def frmt_num(x) -> str:
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return str(x)
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}" if abs(v - round(v)) > 1e-6 else f"{int(round(v))}"

def _getv(m: gp.Model, *names):
    """Devuelve la primera variable que exista entre varios alias."""
    for nm in names:
        v = m.getVarByName(nm)
        if v is not None:
            return v
    # fallback: buscar ignorando espacios/underscores y case
    targets = [nm.replace(" ", "").replace("_", "").lower() for nm in names]
    for v in m.getVars():
        key = v.VarName.replace(" ", "").replace("_", "").lower()
        if key in targets:
            return v
    return None


# ==============================
# Helpers snapshot
# ==============================
def print_snapshot_table(base_row: dict, opt_row: dict | None = None, width=None, max_rows=90):
    w = width or _termw()
    keys = sorted(set(base_row.keys()) | set((opt_row or {}).keys()))
    name_w = 28
    val_w  = (w - 8 - name_w) // 2
    head = f"{'Atributo':<{name_w}} | {'Base':<{val_w}} | {'Óptimo':<{val_w}}"
    print("║   " + head.ljust(w - 6) + " ║")
    print("║   " + ("-" * len(head)).ljust(w - 6) + " ║")

    def _fmt(v):
        try:
            fv = float(pd.to_numeric(v, errors="coerce"))
            if pd.isna(fv):
                return str(v)
            return f"{fv:,.0f}"
        except Exception:
            s = str(v)
            return s if len(s) <= val_w else s[:val_w-1] + "…"

    shown = 0
    for k in keys:
        if shown >= max_rows:
            print("║   … (más filas ocultas)".ljust(w - 4) + " ║")
            break
        b = _fmt(base_row.get(k, ""))
        n = _fmt((opt_row or {}).get(k, ""))
        line = f"{k:<{name_w}} | {b:<{val_w}} | {n:<{val_w}}"
        print("║   " + line.ljust(w - 6) + " ║")
        shown += 1


# ==============================
# Diagnóstico de infactibilidad
# ==============================
def debug_infeas(m: gp.Model, tag="remodel"):
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
    for c in bad_cons:
        print(" CONSTR:", c.ConstrName)
    for q in bad_q:
        print(" QCONSTR:", q.QCName)
    for g in bad_gen:
        print(" GENCONSTR:", g.GenConstrName)

    print("\n=== IIS: límites en conflicto ===")
    for v in bad_lb:
        print(f" LB  : {v.VarName} = {v.LB}")
    for v in bad_ub:
        print(f" UB  : {v.VarName} = {v.UB}")


def _materialize_solution_X(m) -> pd.DataFrame | None:
    Xi = getattr(m, "_X_input", None)
    if Xi is None or Xi.empty:
        print("[DBG] No hay m._X_input")
        return None
    Z = Xi.copy()
    for c in Z.columns:
        v = Z.iloc[0][c]
        if hasattr(v, "X"):  # es una gp.Var
            try:
                Z.iloc[0, Z.columns.get_loc(c)] = float(v.X)
            except Exception:
                Z.iloc[0, Z.columns.get_loc(c)] = 0.0
        else:
            # mantener constantes tal cual
            pass
    # fuerza dtype float si se puede
    try:
        Z = Z.astype(float)
    except Exception:
        pass
    return Z

def debug_free_upgrades(m: gp.Model, eps=1e-8):
    """Lista columnas de X que cambiaron vs base y NO tienen costo directo (lin_cost),
    marcando posibles costos indirectos vía 'drivers'."""
    Xi = getattr(m, "_X_input", None)
    Xb = getattr(m, "_X_base_numeric", None)
    if Xi is None or Xb is None or Xi.empty or Xb.empty:
        print("[FREE-UPGR] Falta _X_input / _X_base_numeric en el modelo.")
        return []

    # 1) variables con costo directo (aparecen en lin_cost)
    cost_vars = set()
    expr = getattr(m, "_lin_cost_expr", None)
    if expr is not None:
        # lineales
        try:
            vs = expr.getVars(); cs = expr.getCoeffs()
            for v, c in zip(vs, cs):
                if abs(float(c)) > 0:
                    cost_vars.add(v.VarName)
        except Exception:
            pass
        # por si acaso, términos cuadráticos
        try:
            for i in range(expr.size()):
                v1 = expr.getVar1(i); v2 = expr.getVar2(i); c = float(expr.getCoeff(i))
                if abs(c) > 0:
                    cost_vars.add(v1.VarName); cost_vars.add(v2.VarName)
        except Exception:
            pass

    # 1.b) heurística de "drivers" de costo indirecto
    driver_tokens = ("Add", "z10_", "z20_", "z30_", "finish", "_upg_", "roof_change", "bsmt_to_")
    drivers = {v.VarName for v in m.getVars() if any(t in v.VarName for t in driver_tokens)}

    # 2) detectar cambios en X
    free = []
    changed = []
    for col in Xi.columns:
        base_val = float(Xb.iloc[0][col]) if col in Xb.columns else None
        cur = Xi.iloc[0][col]
        if hasattr(cur, "X"):
            try:
                xval = float(cur.X)
            except Exception:
                continue
            if (base_val is None) or (abs(xval - base_val) > eps):
                vname = getattr(cur, "VarName", None)
                changed.append((col, base_val, xval, vname))
                if vname not in cost_vars:
                    # Señaliza posible costo indirecto si luce como métrica dependiente de drivers
                    looks_dependent = any(tok in (vname or "") for tok in ("_1st","_Half","_Full","_Deck","Bsmt","Garage","Gr Liv"))
                    note = " (posible costo indirecto vía driver)" if looks_dependent and drivers else ""
                    free.append((col, base_val, xval, vname, note))

    print(f"[FREE-UPGR] Cambios en X: {len(changed)}  |  Cambios SIN costo: {len(free)}")
    for col, bv, xv, vname, note in free[:60]:
        print(f"   - {col:<30s} base={bv} -> sol={xv}   var={vname}{note}")

    return free


def check_predict_consistency(m: gp.Model, bundle):
    """Compara y_price del MIP vs bundle.predict(X_sol) y (si hay log_target) también y_log."""
    Z = _materialize_solution_X(m)
    if Z is None:
        return
    try:
        y_hat = float(bundle.predict(Z).iloc[0])
    except Exception as e:
        print(f"[PRED] Error en bundle.predict(Z): {e}")
        return

    v_price = m.getVarByName("y_price")
    v_log   = m.getVarByName("y_log")

    if v_price is not None:
        print(f"[PRED] bundle.predict(X_sol) = {y_hat:,.2f}  |  y_price (MIP) = {float(v_price.X):,.2f}")
    if v_log is not None:
        import numpy as _np
        try:
            print(f"[PRED] log(y_hat) = {float(_np.log(max(y_hat, 1e-9))):.6f}  |  y_log (MIP) = {float(v_log.X):.6f}")
        except Exception:
            pass
def audit_cost_breakdown_vars(m, top=20):
    expr = getattr(m, "_lin_cost_expr", None)
    if expr is None:
        print("[COST-BREAKDOWN] no _lin_cost_expr")
        return
    try:
        vs = expr.getVars(); cs = expr.getCoeffs()
        X  = m.getAttr("X", vs)
    except Exception:
        print("[COST-BREAKDOWN] no se pudo leer términos")
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
    print("\n[COST-BREAKDOWN] top términos de costo:")
    for name, c, x, contr in rows[:top]:
        print(f"  {name:<35s} coef={c:>10.4f} * X={x:>10.4f}  => {contr:>10.2f}")

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--basecsv", type=str, default=None, help="ruta alternativa al CSV base")
    ap.add_argument("--debug-xgb", action="store_true", help="imprime sensibilidades del XGB (rápido)")
    ap.add_argument("--time-limit", type=float, default=None, help="sobrescribe TimeLimit del solver (segundos)")
    ap.add_argument("--reg-model", type=str, default="models/regression_model.joblib", help="ruta al modelo de regresión serializado")
    args = ap.parse_args()

    # Datos base, costos y modelo ML
    base = get_base_house(args.pid, base_csv=args.basecsv)
    ct = costs.CostTables()
    bundle = XGBBundle()

    # base_row robusto
    try:
        base_row = base.row
    except AttributeError:
        base_row = base if isinstance(base, pd.Series) else pd.Series(base)

    # ===== precio base en el espacio del pipeline =====
    X_base = build_base_input_row(bundle, base_row)
    precio_base = float(bundle.predict(X_base).iloc[0])
    #print("log1p(y_base) =", np.log1p(y_base))

    if args.debug_xgb:
        try:
            if "Kitchen Qual" in X_base.columns:
                Xd = X_base.copy()
                vals = []
                for q in [0, 1, 2, 3, 4]:
                    Xd.loc[:, "Kitchen Qual"] = q
                    vals.append((q, float(bundle.predict(Xd).iloc[0])))
                print("DEBUG Kitchen Qual -> precio:", vals)
        except Exception:
            pass

        try:
            if "Utilities" in X_base.columns:
                vals = []
                for k, name in enumerate(["ELO", "NoSeWa", "NoSewr", "AllPub"]):
                    Xd = X_base.copy()
                    Xd.loc[:, "Utilities"] = k
                    vals.append((k, name, float(bundle.predict(Xd).iloc[0])))
                print("DEBUG Utilities -> precio:", [(k, name, round(p, 2)) for (k, name, p) in vals])
        except Exception:
            pass

    # --- Sanity: pipeline full (TTR) vs pipeline embed (pre + reg) ---
    y_full = float(bundle.predict(X_base).iloc[0])              # en escala de precio
    # Usar el predictor alineado del bundle (evita fallos del ColumnTransformer pickled)
    y_log_embed = float(bundle.predict(X_base).iloc[0])  # en escala log1p
    y_from_embed = float(np.expm1(y_log_embed))

    print(f"[SANITY] full.predict -> price = {y_full:,.2f}")
    print(f"[SANITY] embed (pre+reg) -> log = {y_log_embed:.6f} | price≈ {y_from_embed:,.2f}")

    # ===== construir y resolver el MIP =====
    m: gp.Model = build_mip_embed(base_row, args.budget, ct, bundle, base_price=precio_base)

    # Parámetros de resolución
    time_limit = args.time_limit if args.time_limit is not None else PARAMS.time_limit
    m.Params.MIPGap         = PARAMS.mip_gap
    m.Params.TimeLimit      = time_limit
    m.Params.LogToConsole   = PARAMS.log_to_console
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol     = 1e-7
    m.Params.OptimalityTol  = 1e-7
    m.Params.NumericFocus   = 3

    # Lock a Garage Cars si existe como decisión
    v_gc = m.getVarByName("x_Garage Cars")
    if v_gc is not None:
        base_gc = int(pd.to_numeric(base_row.get("Garage Cars"), errors="coerce") or 0)
        v_gc.LB = base_gc
        v_gc.UB = base_gc


    m.optimize()
    audit_cost_breakdown_vars(m, top=30)
    # tras optimize(), usa el mismo X que vio el embed:
    X_in = rebuild_embed_input_df(m, m._X_base_numeric)
    try:
        # Guardar X_in para diagnósticos posteriores (formato largo idx/feature/value)
        long_rows = []
        for col in X_in.columns:
            try:
                val = float(pd.to_numeric(X_in.loc[0, col], errors="coerce") or 0.0)
            except Exception:
                val = 0.0
            long_rows.append({"idx": 0, "feature": col, "value": val, "LB": "", "UB": ""})
        pd.DataFrame(long_rows).to_csv("X_input_after_opt.csv", index=False)
        print("[AUDIT] guardado X_input_after_opt.csv")
    except Exception as e:
        print(f"[WARN] no pude guardar X_input_after_opt.csv: {e}")
    try:
        y_hat_out = float(bundle.predict(X_in).iloc[0])
        ylog_out  = float(np.log1p(y_hat_out))  # si tu target original es log1p
        delta = float(m._y_log_var.X) - ylog_out
        scale = float(np.exp(delta))
        print(f"[CALIB] y_log(MIP) - log1p(predict_outside) = {delta:.6f} (≈×{scale:.3f})")
    except Exception as e:
        print(f"[CALIB] no pude calcular delta: {e}")

    # ==== AUDIT PACK ====
    import math

    def _eval_linexpr(expr, m):
        """Evalúa un gp.LinExpr en la solución."""
        try:
            vs  = expr.getVars()
            cs  = expr.getCoeffs()
            val = expr.getConstant() if hasattr(expr, "getConstant") else 0.0
            vx  = m.getAttr("X", vs)
            return float(val + sum(c*xi for c, xi in zip(cs, vx)))
        except Exception:
            # fallback brutal: intenta .getValue() si existe, o 0
            try:
                return float(expr.getValue())
            except Exception:
                return float("nan")

    def _to_num(x):
        try:
            z = float(pd.to_numeric(x, errors="coerce"))
            if math.isfinite(z):
                return z
        except Exception:
            pass
        return x  # deja strings si no se puede


    def audit_embed_inputs(m, base_X):
        """
        Lista TODAS las columnas del embed que son Vars y cómo quedaron,
        con LB/UB y valor base.
        """
        rows = []
        for c in m._X_input.columns:
            v = m._X_input.loc[0, c]
            if hasattr(v, "X"):  # sólo Vars
                base_val = None
                if c in base_X.columns:
                    try:
                        base_val = float(base_X.iloc[0][c])
                    except Exception:
                        base_val = str(base_X.iloc[0][c])
                rows.append({
                    "col": c,
                    "X": float(v.X),
                    "LB": getattr(v, "LB", None),
                    "UB": getattr(v, "UB", None),
                    "base": base_val
                })
        if rows:
            df = pd.DataFrame(rows)
            print("\n[AUDIT] Vars inyectadas al embed (que pueden mover predicción):")
            print(df.sort_values("col").to_string(index=False))
        else:
            print("\n[AUDIT] No hay Vars inyectadas al embed (todo sería constante/base).")

    def audit_all_onehots(m):
        """
        Busca patrones de dummies activas (col que contiene '_') y muestra top diferencias.
        Útil para detectar dummies que cambiaron sin que las listaras en FREE-UPGR.
        """
        diffs = []
        for c in m._X_input.columns:
            if "_" not in c:
                continue
            v = m._X_input.loc[0, c]
            if hasattr(v, "X"):
                try:
                    x = float(v.X)
                except Exception:
                    continue
                # reporta dummies activas
                if x > 1e-6:
                    diffs.append((c, x))
        if diffs:
            diffs.sort()
            print("\n[AUDIT] Dummies activas en la solución (valor ~1):")
            for nm, x in diffs:
                print(f"  - {nm}: {x:.6f}")
        else:
            print("\n[AUDIT] No quedaron dummies activas distintas de base (o no son Vars).")

    def audit_costs(m, base_price, bundle):
        """
        Compara y_price, y_base, objetivo, y el costo usado por el modelo.
        Si tu reporter imprime 0 pero aquí sale >0, el problema está en el reporter.
        """
        try:
            y_price = float(m._y_price_var.X)
        except Exception:
            y_price = float("nan")
        uplift = y_price - float(base_price)
        obj    = float(m.objVal)
        # Lee el costo directamente del modelo (fuente de verdad)
        cost_var = m.getVarByName("cost_model") or getattr(m, "_cost_var", None)
        if cost_var is not None:
            used_cost = float(cost_var.X)
        else:
            # Fallback: evalúa el LinExpr guardado
            used_cost = _eval_linexpr(m._lin_cost_expr, m) if hasattr(m, "_lin_cost_expr") else float("nan")


        print("\n=== AUDIT COSTS ===")
        print(f"  y_base        = {base_price:,.2f}")
        print(f"  y_price (MIP) = {y_price:,.2f}")
        print(f"  uplift        = {uplift:,.2f}  = y_price - y_base")
        print(f"  obj           = {obj:,.2f}      = y_price - cost - y_base")
        print(f"  ==> cost(model) ≈ {used_cost:,.2f}")

        # Evalúa directamente el LinExpr del costo si lo guardaste
        if hasattr(m, "_lin_cost_expr"):
            lc_val = _eval_linexpr(m._lin_cost_expr, m)
            print(f"  lin_cost(eval LinExpr) = {lc_val:,.2f}")
            if math.isfinite(used_cost) and abs(lc_val - used_cost) > 1e-4:
                print("  [WARN] used_cost vs lin_cost(eval) no coinciden: revisa si el reporter usa _lin_cost_expr.X correctamente.")
        else:
            print("  [WARN] m._lin_cost_expr no está seteado.")

        # slack presupuesto (útil para confirmar que el costo sí pegó en el modelo)
        try:
            cB = m.getConstrByName("BUDGET")
            if cB is not None:
                print(f"  BUDGET slack  = {cB.Slack:,.6f}")
        except Exception:
            pass

    def audit_predict_outside(m, bundle, base_X):
        """
        Predice con el bundle sobre
        (a) X_rebuilt_from_embed_inputs  y
        (b) tu X_sol (si lo reconstruyes aparte),
        para ver el origen de la brecha.
        """
        X_in = rebuild_embed_input_df(m, base_X)
        try:
            y_hat = float(bundle.predict(X_in).iloc[0])
            if bundle.is_log_target():
                import numpy as np
                y_log = float(np.log1p(y_hat))  # si tu y era log1p
            else:
                y_log = float("nan")
            print("\n[AUDIT] Outside predict usando EXACTO el X que vio el embed:")
            print(f"  y_hat(outside from embed X_in) = {y_hat:,.2f}")
            print(f"  y_log(outside)≈ {y_log:.6f} | y_log(MIP)≈ {float(m._y_log_var.X):.6f}")
        except Exception as e:
            print(f"[AUDIT] Falló predict outside con X_in: {e}")

    # === Ejecuta los audits:
    try:
        base_X = m._X_base_numeric if hasattr(m, "_X_base_numeric") else build_base_input_row(bundle, base_row)
    except Exception:
        base_X = build_base_input_row(bundle, base_row)

    audit_embed_inputs(m, base_X)         # 1) Qué Vars entraron al embed y cómo quedaron
    audit_all_onehots(m)                  # 2) Dummies activas
    audit_predict_outside(m, bundle, base_X)  # 3) Reproduce la predicción externa sobre el mismo X_in
    audit_costs(m, m._base_price_val, bundle)    # 4) Costos: uplift vs obj vs LinExpr

    print("\n[AUDIT] Done.\n")

    # ========= RECONSTRUCCIÓN EXACTA DESDE lo que entró al embed =========
    import numpy as _np
    import pandas as _pd

    def _to_scalar_after_solve(obj):
        """Convierte obj en float leyendo Var.X si aplica. Soporta Series, listas, numpy, etc."""
        # Si viene una Series de 1 elemento, baja al elemento
        if isinstance(obj, _pd.Series):
            if len(obj) == 0:
                return 0.0
            obj = obj.iloc[0]
        # Si viene array/lista/tupla de 1, baja al elemento
        if isinstance(obj, (list, tuple, _np.ndarray)) and len(obj) == 1:
            obj = obj[0]
        # Si es Var de gurobi
        try:
            # algunos wrappers tienen .X, otros no; si falla, sigue
            return float(obj.X)
        except Exception:
            pass
        # Si es numpy escalar / float / int
        try:
            return float(obj)
        except Exception:
            try:
                return float(_np.asarray(obj).item())
            except Exception:
                return 0.0

    try:
        feat_order = list(getattr(m, "_feat_order", []))
        x_cols     = getattr(m, "_x_cols", None)
        x_vars     = getattr(m, "_x_vars", None)

        if not feat_order or x_cols is None or x_vars is None:
            print("[RECON2] pc/x_vars/feat_order no disponible.")
        else:
            # Construye el vector de entrada EXACTO que se usó en el embed
            vals = []
            for c in feat_order:
                v = x_vars[c] if isinstance(x_vars, dict) else x_vars[x_cols.index(c)]
                vals.append(_to_scalar_after_solve(v))
            X_from_embed = _pd.DataFrame([vals], columns=feat_order)

            # Predicción "fuera de Gurobi" pero con esas mismas features
            ylog_out = float(bundle.pipe_for_gurobi().predict(X_from_embed)[0])
            ylog_mip = float(m._y_log_var.X)
            print(f"[RECON2] y_log(outside from embed inputs)= {ylog_out:.6f} | y_log(MIP)= {ylog_mip:.6f} | Δ={ylog_out - ylog_mip:+.3e}")

            # Detección de features que cambiaron vs base numérica
            base_num = getattr(m, "_X_base_numeric", None)
            diffs = []
            if base_num is not None and not base_num.empty:
                base_s = base_num.iloc[0]
                for c in feat_order:
                    b = float(base_s[c]) if c in base_s.index else 0.0
                    a = float(X_from_embed.loc[0, c])
                    if abs(a - b) > 1e-9:
                        diffs.append((c, b, a))
            print(f"[RECON2] #features distintas vs base: {len(diffs)}")
            if diffs:
                # imprime las 15 más “movidas” por magnitud
                diffs_sorted = sorted(diffs, key=lambda t: abs(t[2]-t[1]), reverse=True)[:15]
                for (c, b, a) in diffs_sorted:
                    print(f"  - {c}: base={b:.4g} -> mip={a:.4g} (Δ={a-b:+.4g})")
    except Exception as e:
        print(f"[RECON2] fallo: {e}")



    try:
        free = debug_free_upgrades(m)
        check_predict_consistency(m, bundle)
    except Exception as e:
        print(f"[DBG] Falló diagnóstico de upgrades gratis / consistencia: {e}")
    
    try:
        if hasattr(m, "_y_log_embed_varname") and m._y_log_embed_varname:
            v_emb = m.getVarByName(m._y_log_embed_varname)
            if v_emb is not None:
                print(f"[EMBED] y_log_embed_var = {v_emb.X:.6f}  |  y_log(MIP) = {m.getVarByName('y_log').X:.6f}")
    except Exception as e:
        print(f"[EMBED] No pude leer y_log_embed_var: {e}")


    # Estado
    st = m.Status
    if st in (gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED):
        print("\n❌ Modelo infeasible/unbounded. Generando IIS…")
        try:
            debug_infeas(m, tag="remodel_debug")
        except Exception as e:
            print(f"[WARN] computeIIS falló: {e}")
        # Feasibility relaxation informativa
        try:
            print("\n[INFO] Ejecutando feasibility relaxation…")
            r = m.copy()
            r.feasRelaxS(relaxobjtype=0, minrelax=True, vrelax=False, crelax=True)
            r.optimize()
        except Exception:
            pass
        return

    if st not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) or m.SolCount == 0:
        print("\n⚠️ No hay solución válida; se omite impresión de variables.")
        return

    # ========= Lectura segura de solución =========
    precio_opt_var  = getattr(m, "_y_price_var", None)
    precio_opt      = float(precio_opt_var.X) if precio_opt_var is not None else None
    total_cost_var  = m.getVarByName("cost_model")
    if total_cost_var is not None:
        total_cost_model = float(total_cost_var.X)
    else:
        # fallback seguro: evalúa el LinExpr que usa el modelo
        def _eval_linexpr(expr, m):
            vs  = expr.getVars(); cs = expr.getCoeffs()
            vx  = m.getAttr("X", vs)
            const = expr.getConstant() if hasattr(expr, "getConstant") else 0.0
            return float(const + sum(c*xi for c, xi in zip(cs, vx)))
        total_cost_model = _eval_linexpr(getattr(m, "_lin_cost_expr", gp.LinExpr(0.0)), m)

    budget_usd      = float(getattr(m, "_budget_usd", args.budget))
    budget_slack    = budget_usd - total_cost_model

    # Métricas
    delta_precio = utilidad_incremental = None
    share_final_pct = uplift_base_pct = roi_pct = None
    if (precio_base is not None) and (precio_opt is not None):
        delta_precio = precio_opt - precio_base
        utilidad_incremental = (precio_opt - total_cost_model) - precio_base
        def _pct(num, den):
            try:
                num = float(pd.to_numeric(num, errors="coerce"))
                den = float(pd.to_numeric(den, errors="coerce"))
                if abs(den) < 1e-9:
                    return None
                return 100.0 * num / den
            except Exception:
                return None
        share_final_pct = _pct((precio_opt - precio_base), precio_opt)
        uplift_base_pct = _pct((precio_opt - precio_base), precio_base)
        roi_pct         = _pct(utilidad_incremental, total_cost_model)

    # ========= Reporte de cambios (resumen compacto) =========
    cambios_costos: list[tuple[str, object, object, float | None]] = []

    # (1) Central Air
    try:
        base_air = "Y" if str(base_row.get("Central Air", "N")).strip().upper() in {"Y","YES","1","TRUE"} else "N"
        v_yes = m.getVarByName("central_air_yes")
        if v_yes is not None:
            pick = "Y" if v_yes.X > 0.5 else "N"
            if pick != base_air:
                cambios_costos.append(("Central Air", base_air, pick, float(ct.central_air_install)))
    except Exception:
        pass

    # (2) Kitchen Qual (one-hot kit_is_*)
    try:
        ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        inv = {v:k for k,v in ORD.items()}
        kq_base_txt = str(base_row.get("Kitchen Qual","TA")).strip()
        kq_base = ORD.get(kq_base_txt, 2)

        pick = None
        for nm in ["Po","Fa","TA","Gd","Ex"]:
            v = m.getVarByName(f"kit_is_{nm}")
            if v is not None and v.X > 0.5:
                pick = nm; break
        if pick is not None and ORD[pick] > kq_base:
            cambios_costos.append(("Kitchen Qual", inv[kq_base], pick, float(ct.kitchen_level_cost(pick))))
    except Exception:
        pass

    # (3) Utilities (util_* one-hot si existe)
    try:
        util_names = ["ELO","NoSeWa","NoSewr","AllPub"]
        base_util = str(base_row.get("Utilities","ELO"))
        pick = None
        for nm in util_names:
            v = m.getVarByName(f"util_{nm}")
            if v is not None and v.X > 0.5:
                pick = nm; break
        if pick and pick != base_util:
            cambios_costos.append(("Utilities", base_util, pick, float(ct.util_cost(pick))))
    except Exception:
        pass

    # (4) Roof Matl / Style
    def _norm_lbl(s: str) -> str:
        s = str(s).strip()
        return {"CmentBd":"CemntBd"}.get(s, s)

    def _pick_active(prefix, names):
        for nm in names:
            v = _getv(m, f"x_{prefix}{nm}", f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    try:
        style_names = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
        matl_names  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
        style_new = _pick_active("roof_style_is_", style_names) or _norm_lbl(base_row.get("Roof Style","Gable"))
        matl_new  = _pick_active("roof_matl_is_",  matl_names)  or _norm_lbl(base_row.get("Roof Matl", "CompShg"))
        style_base = _norm_lbl(base_row.get("Roof Style","Gable"))
        matl_base  = _norm_lbl(base_row.get("Roof Matl", "CompShg"))
        if style_new != style_base:
            cambios_costos.append(("Roof Style", style_base, style_new, float(ct.roof_style_cost(style_new))))
        if matl_new != matl_base:
            try:
                c_mat = float(ct.get_roof_matl_cost(matl_new))
            except Exception:
                c_mat = float(ct.roof_matl_cost(matl_new))
            cambios_costos.append(("Roof Matl",  matl_base,  matl_new,  c_mat))
    except Exception:
        pass

    # (5) Electrical (elect_is_*)
    try:
        elec_names = ["SBrkr","FuseA","FuseF","FuseP","Mix"]
        base_elec  = _norm_lbl(base_row.get("Electrical","SBrkr"))
        pick = None
        for nm in elec_names:
            v = _getv(m, f"x_elect_is_{nm}", f"elect_is_{nm}")
            if v is not None and v.X > 0.5:
                pick = nm; break
        if pick and pick != base_elec:
            cambios_costos.append(("Electrical", base_elec, pick,
                                   float(ct.electrical_demo_small) + float(ct.electrical_cost(pick))))
    except Exception:
        pass

    # (6) Exteriors 1st/2nd (ex1_is_*, ex2_is_*)
    try:
        ext_names = [
            "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard",
            "ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco",
            "VinylSd","Wd Sdng","WdShngl"
        ]
        def _pick_ext(prefix: str):
            for nm in ext_names:
                v = _getv(m, f"x_{prefix}is_{nm}", f"{prefix}is_{nm}")
                if v is not None and v.X > 0.5: return nm
            return None

        ex1_base = _norm_lbl(base_row.get("Exterior 1st", "VinylSd"))
        ex2_base = _norm_lbl(base_row.get("Exterior 2nd", ex1_base))
        ex1_new = _pick_ext("ex1_") or ex1_base
        ex2_new = _pick_ext("ex2_") or ex2_base

        if ex1_new != ex1_base:
            cambios_costos.append(("Exterior 1st", ex1_base, ex1_new, float(ct.ext_mat_cost(ex1_new))))
        if ex2_new != ex2_base:
            cambios_costos.append(("Exterior 2nd", ex2_base, ex2_new, float(ct.ext_mat_cost(ex2_new))))
    except Exception:
        pass

    # (7) Mas Vnr Type (+ área si aplica)
    try:
        mvt_names = ["BrkCmn","BrkFace","CBlock","Stone","No aplica","None"]
        mvt_base = str(base_row.get("Mas Vnr Type","No aplica")).strip()
        mvt_pick = None
        for nm in mvt_names:
            v = m.getVarByName(f"x_mvt_is_{nm}")
            if v is not None and v.X > 0.5:
                mvt_pick = nm; break
        mv_area = float(pd.to_numeric(base_row.get("Mas Vnr Area"), errors="coerce") or 0.0)
        v_mv = m.getVarByName("x_Mas Vnr Area")
        if v_mv is not None:
            mv_area = float(v_mv.X)
        if mvt_pick and mvt_pick != mvt_base and mv_area > 0:
            cambios_costos.append(("Mas Vnr Type", mvt_base, mvt_pick, float(ct.mas_vnr_cost(mvt_pick) * mv_area)))
    except Exception:
        pass

    # (8) Heating + Heating QC
    try:
        heat_types = ["Floor","GasA","GasW","Grav","OthW","Wall"]
        heat_base = str(base_row.get("Heating","GasA")).strip()
        heat_new = None
        for nm in heat_types:
            v = _getv(m, f"x_heat_is_{nm}", f"heat_is_{nm}")
            if v is not None and v.X > 0.5:
                heat_new = nm; break

        if heat_new and heat_new != heat_base:
            cambios_costos.append(("Heating (tipo)", heat_base, heat_new, float(ct.heating_type_cost(heat_new))))

        # QC
        q_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
        def _ord(v, default=2):
            M={"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
            try: return int(pd.to_numeric(v, errors="coerce"))
            except Exception: return M.get(str(v).strip(), default)
        qc_base_val = _ord(base_row.get("Heating QC"), default=2)
        vq = _getv(m, "x_Heating QC", "x_HeatingQC")
        if vq is not None:
            qc_new_val = int(round(vq.X))
            if qc_new_val > qc_base_val:
                cambios_costos.append(("Heating QC", q_map[qc_base_val], q_map[qc_new_val],
                                       float(ct.heating_qc_cost(q_map[qc_new_val]))))
    except Exception:
        pass

    # (8.5) Fireplace Qual – reporter
    try:
        ORD = {"No aplica": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        INV = {v: k for k, v in ORD.items()}

        def _fq_txt(v):
            try:
                n = int(pd.to_numeric(v, errors="coerce"))
                return INV.get(n, "No aplica")
            except Exception:
                s = str(v).strip()
                return "No aplica" if s in {"", "NA", "N/A", "NoAplica"} else s

        base_fq = _fq_txt(base_row.get("Fireplace Qu", "No aplica"))

        # preferimos las dummies, si existen
        pick = None
        for nm in ["Po", "Fa", "TA", "Gd", "Ex", "No aplica"]:
            v = _getv(m, f"fireplace_is_{nm}")
            if v is not None and v.X > 0.5:
                pick = nm
                break
        if pick is None:
            # fallback: leer la var numérica
            vnum = _getv(m, "x_Fireplace Qu", "Fireplace Qu", "x_FireplaceQu")
            if vnum is not None:
                try:
                    pick = INV.get(int(round(float(vnum.X))), base_fq)
                except Exception:
                    pick = base_fq

        def _fq_cost(name):
            if hasattr(ct, "fireplace_qc_costs"):
                return float(ct.fireplace_qc_costs.get(name, 0.0))
            if hasattr(ct, "fireplace_costs"):
                return float(ct.fireplace_costs.get(name, 0.0))
            if hasattr(ct, "fireplace_cost"):
                try:
                    return float(ct.fireplace_cost(name))
                except Exception:
                    return 0.0
            return 0.0

        if pick and pick != base_fq and pick != "No aplica":
            cambios_costos.append(("Fireplace Qual", base_fq, pick, _fq_cost(pick)))
    except Exception as e:
        print(f"[TRACE] Fireplace reporter falló: {e}")


    # (9) Basement finish (usa variables del modelo si existen)
    try:
        # Si el modelo creó finish all: bsmt_finish_all + bsmt_to_fin1/2
        v_all = m.getVarByName("bsmt_finish_all")
        v_tr1 = m.getVarByName("bsmt_to_fin1")
        v_tr2 = m.getVarByName("bsmt_to_fin2")
        if v_all is not None and (v_tr1 is not None or v_tr2 is not None):
            added = float((v_tr1.X if v_tr1 else 0.0) + (v_tr2.X if v_tr2 else 0.0))
            if added > 1e-9:
                cambios_costos.append(("Basement finish (ft²)",
                                       "base", f"+{int(round(added))} ft²",
                                       float(ct.finish_basement_per_f2) * added))
    except Exception:
        pass

    # (10) Garage Finish (garage_finish_is_*)
    try:
        gf_cats = ["Fin","RFn","Unf","No aplica"]
        def _has_garage():
            try:
                gt = str(base_row.get("Garage Type","No aplica")).strip()
                if gt in {"NA","No aplica"}:
                    return False
                area = float(pd.to_numeric(base_row.get("Garage Area"), errors="coerce") or 0.0)
                cars = float(pd.to_numeric(base_row.get("Garage Cars"), errors="coerce") or 0.0)
                return (area > 0 or cars > 0)
            except Exception:
                return False
        has_g = _has_garage()
        gf_base = str(base_row.get("Garage Finish","No aplica")).strip()
        if gf_base in {"NA","N/A"}:
            gf_base = "No aplica"
        gf_new = None
        for nm in gf_cats:
            v = _getv(m, f"x_garage_finish_is_{nm}", f"garage_finish_is_{nm}")
            if v is not None and v.X > 0.5:
                gf_new = nm; break
        if has_g and gf_new and gf_new != "No aplica" and gf_new != gf_base:
            cambios_costos.append(("Garage Finish", gf_base, gf_new, float(ct.garage_finish_cost(gf_new))))
    except Exception:
        pass

    # (11) Cambios numéricos sin costo directo (áreas básicas)
    def _n(x): 
        try: return float(pd.to_numeric(x, errors="coerce") or 0.0)
        except Exception: return 0.0

    def _opt_val_chg(col: str):
        v = _getv(m, f"x_{col}")
        if v is not None:
            try:
                return float(v.X)
            except Exception:
                return None
        if hasattr(m, "_X_input") and (col in m._X_input.columns):
            obj = m._X_input.loc[0, col]
            try:
                return float(obj.X) if hasattr(obj, "X") else float(pd.to_numeric(obj, errors="coerce") or 0.0)
            except Exception:
                return None
        return None

    for nm in ["1st Flr SF", "Gr Liv Area", "Half Bath", "Wood Deck SF"]:
        b = _n(base_row.get(nm))
        a = _opt_val_chg(nm)
        if a is not None and abs(a - b) > 1e-9:
            cambios_costos.append((nm, b, a, None))  # sin costo explícito
            

    # (12) Costos explícitos de ampliaciones (explican 1st/GrLiv/Half/Bed)
    A_Full, A_Half, A_Kitch, A_Bed = 40.0, 20.0, 75.0, 70.0

    def _v(name):
        return _getv(m, f"x_{name}", name)

    # Si reportaste cambios "genéricos" de estas columnas, elimínalos (evita doble conteo)
    def _drop_generic(names):
        nonlocal cambios_costos
        cambios_costos = [t for t in cambios_costos if t[0] not in names]

    # Add*
    for nm, A in [("AddFull", A_Full), ("AddHalf", A_Half), ("AddKitch", A_Kitch), ("AddBed", A_Bed)]:
        v = _v(nm)
        if v is not None and float(v.X) > 1e-6:
            # si AddHalf estaba listado como Half Bath/1st/GrLiv, quítalos y reemplaza por la línea con costo
            if nm == "AddHalf":
                _drop_generic(["1st Flr SF", "Gr Liv Area", "Half Bath"])
            cambios_costos.append((
                f"{nm} (+{int(A)} ft²)", "-", f"+{int(A*float(v.X))} ft²",
                float(ct.construction_cost) * A * float(v.X)
            ))

    # Gatillo extra 1st floor (si aplica)
    v_add1 = _v("Add1stFlr")
    if v_add1 is not None and float(v_add1.X) > 0.5:
        delta_1flr = 40.0
        _drop_generic(["1st Flr SF", "Gr Liv Area"])  # lo explica este gatillo
        cambios_costos.append((
            "Extra 1st Flr (gatillo 40 ft²)", "-", "+40 ft²",
            float(ct.construction_cost) * delta_1flr
        ))

    # (13) Ampliaciones porcentuales (z10/z20/z30) para superficies (Deck, Porches, Pool, Garage)
    try:
        def _num(x):
            try: 
                return float(pd.to_numeric(x, errors="coerce") or 0.0)
            except Exception: 
                return 0.0

        # Columna → posibles alias para buscar z-flags con nombres distintos
        COMPONENTES = [
            ("Wood Deck SF",   ["WoodDeckSF","WoodDeck","DeckSF","Deck"]),
            ("Open Porch SF",  ["OpenPorchSF","OpenPorch"]),
            ("Enclosed Porch", ["EnclosedPorch"]),
            ("3Ssn Porch",     ["3SsnPorch","ThreeSsnPorch","ThreeSeasonPorch"]),
            ("Screen Porch",   ["ScreenPorch"]),
            ("Pool Area",      ["PoolArea","Pool"]),
            ("Garage Area",    ["GarageArea","Garage"])
        ]

        def _find_zflag(m, s, col, aliases):
            # prueba varios nombres razonables
            candidates = []
            base = col
            # variantes con/ sin espacios / SF
            candidates += [f"z{s}_{base}",
                        f"z{s}_{base.replace(' ','')}",
                        f"z{s}_{base.replace(' ','_')}",
                        f"z{s}_{base.replace(' ','').replace('SF','')}",
                        f"z{s}_{base.replace(' ','_').replace('SF','')}"]
            # añade alias
            for a in aliases:
                candidates += [f"z{s}_{a}", f"z{s}{a}", f"z_{s}_{a}"]
            # busca
            for nm in candidates:
                v = _getv(m, nm)
                if v is not None:
                    return v
            # fallback: búsqueda “contiene” (muy laxa pero útil)
            tgt = base.replace(" ", "").replace("_", "").replace("%","").lower()
            for v in m.getVars():
                vn = v.VarName.replace(" ", "").replace("_","").replace("%","").lower()
                if vn.startswith(f"z{s}") and tgt in vn:
                    return v
            return None

        # helper: valor óptimo de una columna (lee var x_... o X_in)
        def _opt_val_chg(col: str):
            v = _getv(m, f"x_{col}")
            if v is not None:
                try:
                    return float(v.X)
                except Exception:
                    return None
            if hasattr(m, "_X_input") and (col in m._X_input.columns):
                obj = m._X_input.loc[0, col]
                try:
                    return float(obj.X) if hasattr(obj, "X") else float(pd.to_numeric(obj, errors="coerce") or 0.0)
                except Exception:
                    return None
            return None

        for col, aliases in COMPONENTES:
            base_c = _num(base_row.get(col))
            new_c  = _opt_val_chg(col)
            if new_c is None:
                continue
            delta = new_c - base_c

            # intenta z10/z20/z30; si no encuentra z*, infiere por delta≈%base
            picked = False
            for s in (10, 20, 30):
                vflag = _find_zflag(m, s, col, aliases)
                flag_on = (vflag is not None and float(vflag.X) > 0.5)
                # tolerancia: 1 ft² o 1% del base, lo que sea mayor
                tol = max(1.0, 0.01 * base_c)
                by_ratio = (base_c > 0 and abs(delta - base_c * s / 100.0) <= tol)

                if flag_on or by_ratio:
                    unit_cost = {10: ct.ampl10_cost, 20: ct.ampl20_cost, 30: ct.ampl30_cost}[s]
                    # evita doble conteo: borra la línea genérica del mismo atributo si ya estaba
                    cambios_costos = [t for t in cambios_costos if t[0] != col]
                    cambios_costos.append((
                        f"{col} +{s}%", base_c, new_c,
                        float(unit_cost) * (base_c * s / 100.0)
                    ))
                    picked = True
                    break

            # si no “pickeó” nada pero hubo cambio, deja (o conserva) la línea genérica sin costo
            if (not picked) and abs(delta) > 1e-9:
                # Si ya hay una genérica, no duplicar
                if not any(t[0] == col for t in cambios_costos):
                    cambios_costos.append((col, base_c, new_c, None))

    except Exception:
        pass

    # (X) Garage Qual / Cond – reporter robusto
    try:
        G_LIST = ["Po", "Fa", "TA", "Gd", "Ex", "No aplica"]

        def _norm_noap(s):
            s = str(s).strip()
            return "No aplica" if s in {"", "NA", "N/A", "NoAplica", "No aplica", "None", "nan"} else s

        def _g_txt(v):
            # base puede venir como número u texto
            M = {0:"Po", 1:"Fa", 2:"TA", 3:"Gd", 4:"Ex"}
            try:
                vv = int(pd.to_numeric(v, errors="coerce"))
                return M.get(vv, "No aplica")
            except Exception:
                return _norm_noap(v)

        gq_base = _g_txt(base_row.get("Garage Qual", "No aplica"))
        gc_base = _g_txt(base_row.get("Garage Cond", "No aplica"))

        # pick dummies o fallback numérico
        def _pick_active_any(prefix_dummy: str, numeric_var_names=("x_Garage Qual","x_GarageQual")):
            # prueba con o sin prefijo x_
            for nm in G_LIST:
                v = _getv(m, f"x_{prefix_dummy}{nm}", f"{prefix_dummy}{nm}")
                if v is not None and v.X > 0.5:
                    return nm
            # fallback: variable numérica (ordinal 0..4)
            for nv in numeric_var_names:
                vnum = _getv(m, nv)
                if vnum is not None:
                    M = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
                    try:
                        return M.get(int(round(float(vnum.X))), "No aplica")
                    except Exception:
                        pass
            return None

        q_new = _pick_active_any("garage_qual_is_")
        c_new = _pick_active_any("garage_cond_is_")

        # normaliza
        q_new = _norm_noap(q_new) if q_new is not None else None
        c_new = _norm_noap(c_new) if c_new is not None else None
        gq_base = _norm_noap(gq_base)
        gc_base = _norm_noap(gc_base)

        # costo de destino (mismo esquema que usas en otros qual/cond)
        def _cost(name):
            try:
                return float(ct.garage_qc_costs.get(name, 0.0))
            except Exception:
                return 0.0

        if q_new and q_new != gq_base and q_new != "No aplica":
            cambios_costos.append(("Garage Qual", gq_base, q_new, _cost(q_new)))
        if c_new and c_new != gc_base and c_new != "No aplica":
            cambios_costos.append(("Garage Cond", gc_base, c_new, _cost(c_new)))

        # TRAZA opcional para depurar por qué a veces hay costo y no aparece cambio
        try:
            cq = _cost(q_new) if (q_new and q_new != gq_base and q_new != "No aplica") else 0.0
            cc = _cost(c_new) if (c_new and c_new != gc_base and c_new != "No aplica") else 0.0
            print(f"[TRACE] GQ: base={gq_base} -> pick={q_new} ; cost={cq}")
            print(f"[TRACE] GC: base={gc_base} -> pick={c_new} ; cost={cc}")
        except Exception:
            pass

    except Exception as e:
        print(f"[TRACE] Reporter Garage falló: {e}")


    # (14) Chequeo: suma de costos reportados vs lin_cost del modelo
    try:
        rep_cost = sum(float(c or 0.0) for (_, _, _, c) in cambios_costos)
        lc_val = _eval_linexpr(m._lin_cost_expr, m) if hasattr(m, "_lin_cost_expr") else float("nan")
        print(f"\n[CHECK COSTS] Reportados={money(rep_cost)} | lin_cost={money(lc_val)} | Δ={money(rep_cost - lc_val)}")
    except Exception:
        pass


    # ========= Salida =========
    print("\n" + "="*60)
    print("               RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60 + "\n")

    print(f"📍 PID: {base_row.get('PID', 'N/A')} – {base_row.get('Neighborhood', 'N/A')} | Presupuesto: ${args.budget:,.0f}")
    print(f"🧮 Modelo: {m.ModelName if hasattr(m, 'ModelName') else 'Gurobi MIP'}")
    print(f"⏱️ Tiempo total: {getattr(m, 'Runtime', 0.0):.2f}s | MIP Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    print("💰 **Resumen Económico**")
    print(f"  Precio casa base:        ${precio_base:,.0f}")
    print(f"  Precio casa remodelada:  ${precio_opt:,.0f}"      if precio_opt is not None else "  Precio casa remodelada:  N/A")
    print(f"  Δ Precio:                ${delta_precio:,.0f}"    if delta_precio is not None else "  Δ Precio:                N/A")
    print(f"  Costos totales (modelo): ${total_cost_model:,.0f}")

    obj_val = getattr(m, "ObjVal", None)
    if obj_val is not None:
        print(f"  Valor objetivo (MIP):    ${obj_val:,.2f}   (≡ y_price - cost - y_base)")
    else:
        obj_recalc = (precio_opt or 0.0) - total_cost_model - (precio_base or 0.0)
        print(f"  Valor objetivo (MIP):    ${obj_recalc:,.2f}   (recalculado)")

    if uplift_base_pct is not None:
        print(f"  Uplift vs base:          {uplift_base_pct:.0f}%")
    if share_final_pct is not None:
        print(f"  % del precio final por mejoras: {share_final_pct:.0f}%")
    if utilidad_incremental is not None:
        print(f"  ROI (Δ neto $):          ${utilidad_incremental:,.0f}")
    if roi_pct is not None:
        print(f"  ROI %:                   {roi_pct:.0f}%")
    print(f"  Slack presupuesto:       ${budget_slack:,.2f}")

    # ========= NUEVA SECCIÓN: COMPARACIÓN XGBoost vs REGRESIÓN =========
    try:
        from optimization.remodel.regression_predictor import RegressionPredictor
        # Cargar wrapper de regresión
        reg_pred = RegressionPredictor()
        
        # Datos base (sin cambios)
        X_base = build_base_input_row(bundle, base_row)
        
        # Reconstruir datos optimizados desde el modelo
        try:
            X_opt = rebuild_embed_input_df(m, X_base)
        except Exception:
            # Fallback: usar datos base si hay error
            X_opt = X_base
        
        # Predicciones con regresión
        try:
            reg_base = reg_pred.predict(X_base)
            reg_opt = reg_pred.predict(X_opt)
            
            print("\n📊 **Comparación de Modelos (XGBoost vs Regresión)**")
            print(f"  Casa base (sin mejoras):")
            print(f"    - XGBoost:   ${precio_base:,.0f}")
            print(f"    - Regresión: ${reg_base:,.0f}")
            if precio_base is not None:
                diff_base = reg_base - precio_base
                pct_base = (diff_base / precio_base * 100) if precio_base > 0 else 0
                print(f"    - Diferencia: ${diff_base:+,.0f} ({pct_base:+.1f}%)")
            
            print(f"\n  Casa remodelada (con mejoras):")
            print(f"    - XGBoost:   ${precio_opt:,.0f}" if precio_opt is not None else f"    - XGBoost:   N/A")
            print(f"    - Regresión: ${reg_opt:,.0f}")
            if precio_opt is not None:
                diff_opt = reg_opt - precio_opt
                pct_opt = (diff_opt / precio_opt * 100) if precio_opt > 0 else 0
                print(f"    - Diferencia: ${diff_opt:+,.0f} ({pct_opt:+.1f}%)")
            
            # Delta de mejora
            xgb_delta = precio_opt - precio_base if (precio_opt is not None and precio_base is not None) else None
            reg_delta = reg_opt - reg_base
            
            print(f"\n  Uplift (mejora por remodelación):")
            if xgb_delta is not None:
                print(f"    - XGBoost:   ${xgb_delta:,.0f} ({xgb_delta/precio_base*100:.1f}%)")
            print(f"    - Regresión: ${reg_delta:,.0f} ({reg_delta/reg_base*100:.1f}%)")
            
            if xgb_delta is not None:
                gap = abs(xgb_delta - reg_delta)
                gap_pct = (gap / max(abs(xgb_delta), abs(reg_delta)) * 100) if max(abs(xgb_delta), abs(reg_delta)) > 0 else 0
                print(f"    - Gap:       ${gap:,.0f} ({gap_pct:.1f}%)")
                if gap_pct < 10:
                    print(f"      ✅ Modelos convergen (gap < 10%)")
                elif gap_pct < 20:
                    print(f"      ⚠️  Brecha moderada (gap 10-20%)")
                else:
                    print(f"      ⚠️  Brecha significativa (gap > 20%)")
            
        except Exception as e:
            print(f"\n[INFO] No se pudo calcular comparación con regresión: {e}")
    
    except ImportError:
        pass  # Regresión no disponible
    except Exception as e:
        print(f"\n[TRACE] Error en sección de comparación XGBoost vs Regresión: {e}")

    # Calidad global y calidades clave
    try:
        def _qual_txt(v):
            M = {-1: "No aplica", 0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}
            try:
                iv = int(round(float(pd.to_numeric(v, errors="coerce"))))
                return M.get(iv, str(v))
            except Exception:
                return str(v)

        def _qual_opt(col: str, extra_alias: str | None = None):
            aliases = [f"x_{col}", col]
            if extra_alias:
                aliases.insert(0, extra_alias)
            v = _getv(m, *aliases)
            if v is not None:
                try:
                    return v.X
                except Exception:
                    pass
            if hasattr(m, "_X_input") and col in getattr(m, "_X_input").columns:
                try:
                    return float(pd.to_numeric(m._X_input.loc[0, col], errors="coerce"))
                except Exception:
                    return m._X_input.loc[0, col]
            return base_row.get(col)

        QUAL_COLS = [
            ("Overall Qual", "Overall_Qual_calc"),
            ("Kitchen Qual", None),
            ("Exter Qual", None),
            ("Exter Cond", None),
            ("Heating QC", None),
            ("Fireplace Qu", None),
            ("Bsmt Cond", None),
            ("Garage Qual", None),
            ("Garage Cond", None),
            ("Pool QC", None),
        ]

        # ===== NUEVO: Calcula mejora sofisticada de calidad =====
        try:
            # Reconstruye la fila óptima
            opt_row_dict = dict(base_row.items())
            
            for col, alias in QUAL_COLS:
                if col == "Overall Qual":
                    continue  # Lo calcularemos, no lo leemos
                opt_val = _qual_opt(col, extra_alias=alias)
                if opt_val is not None:
                    opt_row_dict[col] = opt_val
            
            opt_row_series = pd.Series(opt_row_dict)
            
            # Usa el QualityCalculator para obtener el análisis desglosado
            calc = QualityCalculator(max_boost=2.0)
            quality_result = calc.calculate_boost(base_row, opt_row_series)
            
            # Imprime el reporte desglosado
            print("\n" + calc.format_changes_report(quality_result))
            
        except Exception as e:
            print(f"\n[TRACE] Cálculo sofisticado de calidad falló: {e}")

        print("\n🌟 **Calidad general y calidades clave (detalle)**")
        for col, alias in QUAL_COLS:
            base_val = base_row.get(col, "N/A")
            opt_val  = _qual_opt(col, extra_alias=alias)
            base_txt = _qual_txt(base_val)
            opt_txt  = _qual_txt(opt_val)
            if opt_val is None:
                opt_txt = "N/A"
            if base_txt == opt_txt:
                print(f"  - {col}: {base_txt}")
            else:
                try:
                    delta = float(opt_val) - float(pd.to_numeric(base_val, errors="coerce") or 0.0)
                    print(f"  - {col}: {base_txt} → {opt_txt} (Δ {delta:+.1f})")
                except Exception:
                    print(f"  - {col}: {base_txt} → {opt_txt}")
    except Exception as e:
        print(f"[TRACE] Resumen de calidades falló: {e}")

    # Cambios resumidos
    print("\n🏠 **Cambios hechos en la casa**")
    if cambios_costos:
        for nombre, base_val, new_val, cost_val in cambios_costos:
            suf = f" (costo {money(cost_val)})" if (cost_val is not None and cost_val > 0) else ""
            print(f"  - {nombre}: {base_val} → {new_val}{suf}")
    else:
        print("  (No se detectaron cambios)")

    # Snapshot Base vs Óptimo (exacto con X_in del embed)
    print("\n🧾 **Snapshot: atributos Base vs Óptimo (compacto)**")
    try:
        # 1) Reconstruye EXACTAMENTE lo que vio el embed
        X_in = rebuild_embed_input_df(m, m._X_base_numeric)

        def _opt_val(col: str):
            """1) lee var x_<col> si existe; 2) usa X_in[col]; 3) cae a base."""
            v = _getv(m, f"x_{col}")
            if v is not None:
                try:
                    return float(v.X)
                except Exception:
                    pass
            if hasattr(X_in, "columns") and (col in X_in.columns):
                try:
                    return float(pd.to_numeric(X_in.loc[0, col], errors="coerce") or 0.0)
                except Exception:
                    pass
            return base_row.get(col)

        base_dict = dict(base_row.items())
        opt_dict  = dict(base_row.items())

        # 2) Sobrescribe las numéricas que sí cambian en tu MIP
        for col in [
            "1st Flr SF", "Gr Liv Area", "Half Bath", "Wood Deck SF",
            "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch",
            "Pool Area", "Garage Area", "Bsmt Unf SF", "BsmtFin SF 1", "BsmtFin SF 2", "Total Bsmt SF"
        ]:
            opt_dict[col] = _opt_val(col)

        # Fireplace Qu (mostrar etiqueta)
        try:
            M = {-1:"No aplica", 0:"Po", 1:"Fa", 2:"TA", 3:"Gd", 4:"Ex"}
            v_fp = _getv(m, "x_Fireplace Qu")
            if v_fp is not None:
                opt_dict["Fireplace Qu"] = M.get(int(round(float(v_fp.X))), opt_dict.get("Fireplace Qu"))
        except Exception:
            pass

        # 3) Inyecta categóricas como ya hacías
        v_yes = m.getVarByName("central_air_yes")
        if v_yes is not None:
            opt_dict["Central Air"] = "Y" if v_yes.X > 0.5 else "N"

        style_names = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
        matl_names  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
        elec_names  = ["SBrkr","FuseA","FuseF","FuseP","Mix"]

        # estas helpers ya las tienes definidas más arriba en el archivo
        style_new = _pick_active("roof_style_is_", style_names) or _norm_lbl(base_row.get("Roof Style","Gable"))
        matl_new  = _pick_active("roof_matl_is_",  matl_names)  or _norm_lbl(base_row.get("Roof Matl", "CompShg"))
        elec_new  = _pick_active("elect_is_",      elec_names)  or _norm_lbl(base_row.get("Electrical","SBrkr"))

        opt_dict["Roof Style"] = style_new
        opt_dict["Roof Matl"]  = matl_new
        opt_dict["Electrical"] = elec_new

        print_snapshot_table(base_dict, opt_dict)

    except Exception as e:
        print(f"⚠️  Error al generar snapshot: {e}")


    print("\n" + "="*60)
    print("            FIN RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60 + "\n")

    # ============================================================
    # COMPARACIÓN DE PREDICTORES: XGBoost vs Regresión Lineal
    # ============================================================
    try:
        from optimization.remodel.regression_predictor import RegressionPredictor
        
        print("\n" + "="*70)
        print("  COMPARACIÓN: Predicción con XGBoost vs Regresión Lineal")
        print("="*70)
        
        # 1. Obtener precio base (sin mejoras)
        X_base = build_base_input_row(bundle, base_row)
        precio_base_xgb = float(bundle.predict(X_base).iloc[0])
        
        # 2. Obtener predicción XGBoost de casa REMODELADA
        X_opt = rebuild_embed_input_df(m, m._X_base_numeric)
        precio_opt_xgb = float(bundle.predict(X_opt).iloc[0])
        
        # 3. Cargar regresión y predecir
        try:
            # Preferimos el modelo re-procesado (mismas columnas que build_base_input_row)
            reg_model_path = "models/regression_model_reprocesed.pkl"
            if not Path(reg_model_path).exists():
                # fallback al CLI --reg-model si existe, o al modelo previo
                reg_model_path = args.reg_model if Path(args.reg_model).exists() else "models/regression_model_final.pkl"
            reg_predictor = RegressionPredictor(reg_model_path)
            
            # Predecir base con regresión (usando mismo X_base)
            precio_base_reg = reg_predictor.predict(X_base)
            
            # Predecir optimizado con regresión (usando X_opt)
            precio_opt_reg = reg_predictor.predict(X_opt)
            
            # Cálculos
            mejora_xgb = precio_opt_xgb - precio_base_xgb
            mejora_xgb_pct = (mejora_xgb / precio_base_xgb * 100) if precio_base_xgb > 0 else 0
            
            mejora_reg = precio_opt_reg - precio_base_reg
            mejora_reg_pct = (mejora_reg / precio_base_reg * 100) if precio_base_reg > 0 else 0
            
            diff_predicciones = precio_opt_xgb - precio_opt_reg
            diff_pct = (diff_predicciones / precio_opt_reg * 100) if precio_opt_reg > 0 else 0
            
            # Mostrar resultados
            print(f"\n� PREDICCIONES DEL PRECIO ACTUAL (sin mejoras):")
            print(f"   XGBoost:   ${precio_base_xgb:,.0f}")
            print(f"   Regresión: ${precio_base_reg:,.0f}")
            
            print(f"\n📊 PREDICCIONES DEL PRECIO REMODELADO (con mejoras):")
            print(f"   XGBoost:   ${precio_opt_xgb:,.0f}  (+{mejora_xgb_pct:.1f}%)")
            print(f"   Regresión: ${precio_opt_reg:,.0f}  (+{mejora_reg_pct:.1f}%)")
            
            print(f"\n📊 DIFERENCIA ENTRE MODELOS (para casa remodelada):")
            print(f"   XGBoost - Regresión: ${diff_predicciones:+,.0f} ({diff_pct:+.1f}%)")
            
            if abs(diff_pct) < 10:
                print(f"   ✅ Modelos convergen: diferencia < 10%")
            elif abs(diff_pct) < 20:
                print(f"   ⚠️  Modelos divergen moderadamente: diferencia 10-20%")
            else:
                print(f"   ❌ Modelos divergen significativamente: diferencia > 20%")
            
            print(f"\n💡 INTERPRETACIÓN:")
            if mejora_xgb_pct > mejora_reg_pct:
                print(f"   XGBoost predice mayor impacto (+{mejora_xgb_pct:.1f}% vs +{mejora_reg_pct:.1f}%)")
            else:
                print(f"   Regresión predice mayor impacto (+{mejora_reg_pct:.1f}% vs +{mejora_xgb_pct:.1f}%)")

            # Resumen compacto en una línea (fácil de leer/copiar)
            print("\n--- RESUMEN COMPACTO ---")
            gap_base_abs = precio_base_xgb - precio_base_reg
            gap_base_pct = (gap_base_abs / precio_base_reg * 100) if precio_base_reg else 0
            who_base = "XGBoost supera a Regresión" if gap_base_abs > 0 else "Regresión supera a XGBoost"
            print(f"  Base:       Regresión ${precio_base_reg:,.0f} | XGBoost ${precio_base_xgb:,.0f}  ({who_base} {gap_base_pct:+.2f}%)")

            gap_opt_abs = diff_predicciones
            gap_opt_pct = diff_pct
            who_opt = "XGBoost supera a Regresión" if gap_opt_abs > 0 else "Regresión supera a XGBoost"
            print(f"  Remodelada: Regresión ${precio_opt_reg:,.0f} | XGBoost ${precio_opt_xgb:,.0f}  ({who_opt} {gap_opt_pct:+.2f}%)")
            print(f"  Gap remodelada: ${diff_predicciones:+,.0f} ({diff_pct:+.2f}% vs Regresión)")
                    
        except FileNotFoundError as e:
            print(f"\n⚠️  Modelo de regresión no encontrado:")
            print(f"   {e}")
            print(f"\n   Para entrenar el modelo, ejecuta:")
            print(f"   python3 training/train_regression_final.py")
        except Exception as e:
            print(f"\n⚠️  Error al usar modelo de regresión:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            print(f"   python3 training/train_regression_model.py")
    
    except Exception as e:
        print(f"\n⚠️  Error general en comparación de predictores: {e}")

    def _max_viol(m, name_prefix):
        viol = 0.0
        for c in m.getConstrs():
            if c.ConstrName.startswith(name_prefix):
                viol = max(viol, abs(c.Slack))
        return viol

    print("\n[CHECKS]")
    for tag in ["KIT_", "EXT_", "MVT_", "ROOF_", "GaFin_", "HEAT_", "BSMT_", "UTIL_", "R", "AREA_", "PoolQC_"]:
        mv = _max_viol(m, tag)
        print(f"  max |slack| {tag:<8} = {mv:.3e}")

    cB = m.getConstrByName("BUDGET")
    if cB:
        print(f"  BUDGET slack = {cB.Slack:.6f}")

if __name__ == "__main__":
    main()
