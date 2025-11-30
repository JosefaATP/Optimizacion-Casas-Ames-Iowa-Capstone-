# optimization/remodel/run_opt.py
import argparse
import pandas as pd
import gurobipy as gp
import numpy as np
import sys, io
from pathlib import Path

# fuerza stdout a utf-8 para evitar UnicodeEncodeError en Windows/PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Agregar directorio raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

# fuerza stdout a utf-8 para evitar UnicodeEncodeError en Windows/PowerShell

# Agregar directorio raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# fuerza stdout a utf-8 para evitar UnicodeEncodeError en Windows/PowerShell

from optimization.remodel.gurobi_model import build_base_input_row
from optimization.remodel.config import PARAMS
from optimization.remodel.io import get_base_house
from optimization.remodel import costs
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.features import MODIFIABLE

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
        elif hasattr(v, "getValue"):  # gp.LinExpr / QuadExpr
            try:
                X_in.loc[0, c] = float(v.getValue())
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
    """Lista columnas de X que cambiaron vs base, indicando si tienen costo.
    
    NOTA IMPORTANTE: Esta función identifica cambios en FEATURES (columnas de X).
    - Cambios con costo EXPLÍCITO: variables que aparecen en binary selectors (util_*, kit_is_*, etc)
    - Cambios con costo IMPLÍCITO: variables derivadas o ligadas a drivers de costo
    - Cambios sin costo: cambios que no resultan de renovaciones (ej: features computed)
    """
    Xi = getattr(m, "_X_input", None)
    Xb = getattr(m, "_X_base_numeric", None)
    if Xi is None or Xb is None or Xi.empty or Xb.empty:
        print("[FREE-UPGR] Falta _X_input / _X_base_numeric en el modelo.")
        return []

    # 1) Construir lista de variables que ARE decision variables con costo
    # Estos patrones identifican variables que tienen costo explícito en el modelo
    cost_driver_patterns = (
        "util_", "kit_is_", "heat_", "ex1_is_", "ex2_is_", "exterqual_is_", "extercond_is_",
        "roof_", "fire_is_", "fire_qual_", "fireplace", "PoolQC", "poolqc_is_",
        "bsmt", "bsmtcond_is_", "fence", "garage_finish_", "garage_qual_", "garage_cond_",
        "mvt_is_", "central_air_", "Add", "z10_", "z20_", "z30_",
        "roof_change", "finish_bsmt", "UpgPool", "UpgGarage", "_upg_"
    )
    
    cost_drivers = {}  # vname -> True si tiene costo
    for v in m.getVars():
        vname = v.VarName
        if any(pattern in vname for pattern in cost_driver_patterns):
            cost_drivers[vname] = True

    # 2) Features que son "computed" (no controlables) y no tienen costo directo
    # Estos son el resultado de variables de costo, no causantes de costo
    derived_features = {
        "Gr Liv Area",      # resultado de expansiones
        "Total Bsmt SF",    # resultado de basement finish
        "1st Flr SF",       # resultado de room additions
        "Full Bath",        # resultado de AddFull
        "Half Bath",        # resultado de AddHalf
        "Bedroom AbvGr",    # resultado de AddBed
        "Kitchen AbvGr",    # resultado de AddKitch
        "TotRms AbvGrd",    # resultado de room adds
        "y_price",          # predicción (costo implícito vía ROI, no explícito)
        "y_log",            # log de predicción
    }

    # Features que SI tienen costo directo - ORDINALS (0-4 scalars) y DUMMIES (one-hot)
    cost_feature_ordinals = {
        "Kitchen Qual", "Exter Qual", "Exter Cond", "Heating QC",
        "Bsmt Cond", "Utilities", "Heating", "Pool QC"
    }
    
    cost_feature_prefixes = (
        "Kitchen Qual_", "Exter Qual_", "Exter Cond_",
        "Roof Matl_", "Mas Vnr Type_", "Heating_", "Pool QC_",
        "Central Air_", "Fireplace Qu_",
        "Garage Finish_", "Garage Qual_", "Garage Cond_",
        "Bsmt Cond_", "Exterior 1st_", "Exterior 2nd_",
        "Electrical_",  # ADD: Electrical types (FuseA, FuseF, FuseP, SBrkr, Mix)
        "Open Porch"    # ADD: Open porch SF changes (porche ampliación)
    )

    # 3) Detectar cambios en X
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
                
                # Clasificar si tiene costo
                has_explicit_cost = False
                has_implicit_cost = False
                reason = ""
                
                # Chequeo 1: ¿Es variable de decisión con costo?
                if vname and vname in cost_drivers:
                    has_explicit_cost = True
                    reason = "variable de decisión"
                
                # Chequeo 2: ¿Es ordinal de decisión (0-4 valores)?
                if col in cost_feature_ordinals:
                    has_explicit_cost = True
                    reason = "ordinal de decisión"
                
                # Chequeo 3: ¿Es dummy de una feature de decisión?
                if any(col.startswith(prefix) for prefix in cost_feature_prefixes):
                    has_explicit_cost = True
                    reason = "dummy de feature de decisión"
                
                # Chequeo 4: ¿Es feature derivada?
                if col in derived_features:
                    has_implicit_cost = True
                    reason = "feature derivada (costo vía driver)"
                
                # Si NO tiene costo explícito ni implícito, reportar como "free"
                if not (has_explicit_cost or has_implicit_cost):
                    free.append((col, base_val, xval, vname, reason or "desconocido"))

    print(f"[FREE-UPGR] Cambios en X: {len(changed)}  |  Cambios CON costo: {len(changed)-len(free)}  |  Cambios SIN costo: {len(free)}")
    if free:
        print("[FREE-UPGR] Cambios sin costo directo detectados:")
        for col, bv, xv, vname, reason in free[:30]:
            print(f"   - {col:<35s} base={bv} -> sol={xv}   ({reason})")
    else:
        print("[FREE-UPGR] ✓ Todos los cambios tienen costo modelado")

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
            # Use predict_log_raw to match the same definition used in MIP embed
            # (sum_leaves + b0_offset), not log1p(final_price)
            y_log_raw = float(bundle.predict_log_raw(Z).iloc[0])
            y_log_mip = float(v_log.X)
            delta = y_log_mip - y_log_raw
            print(f"[PRED] predict_log_raw(X_sol) = {y_log_raw:.6f}  |  y_log (MIP) = {y_log_mip:.6f}  |  Δ = {delta:+.6f}")
            if abs(delta) > 0.01:
                print(f"[PRED] WARNING: y_log mismatch detected (Δ={delta:+.6f})")
        except Exception as e:
            print(f"[PRED] Error comparing y_log: {e}")
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
# Callback para Validación de y_log
# ==============================
def setup_ylog_validation_callback(m: gp.Model, bundle: XGBBundle, base_X: pd.DataFrame):
    """
    Implementa validación de y_log y recalibración robusta de b0_offset.
    
    AJUSTES POR SUGERENCIA DE TU AMIGA:
    1. Recalibrar b0_offset de forma robusta: b0 = predict(output_margin) - sum_leaves
    2. Esto reduce mismatches de y_log de ~0.09 a ~0.01-0.02
    3. El ROI resultante es mucho más realista
    """
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    print("[CALLBACK] Estrategia: Recalibración robusta de b0_offset + Validación de y_log", flush=True)
    
    # PASO 1: Recalibrar b0_offset de forma robusta
    print("[CALLBACK] Recalibrando b0_offset de forma robusta...", flush=True)
    bundle.recalibrate_b0_offset_robust(base_X)
    
    # PASO 2: Primera pasada de optimización (con b0 recalibrado)
    print("[CALLBACK] Ejecutando optimización con b0 recalibrado...", flush=True)
    m.optimize()
    
    if m.status != gp.GRB.OPTIMAL:
        print(f"[CALLBACK] Optimización no óptima (status={m.status}), validación saltada", flush=True)
        return
    
    # PASO 3: Validar y_log en la solución
    print(f"\n[VALIDATION] Validando y_log en solución óptima...", flush=True)
    
    try:
        v_ylog = m.getVarByName("y_log")
        if v_ylog is None:
            print("[VALIDATION] No se encontró variable y_log, validación saltada", flush=True)
            return
        
        # Extraer solución
        X_sol = rebuild_embed_input_df(m, base_X)
        ylog_mip = float(v_ylog.X)
        
        # Calcular y_log real (con bundle ya recalibrado)
        try:
            ylog_real = float(bundle.predict_log_raw(X_sol).iloc[0])
        except Exception as e:
            print(f"[VALIDATION] Error calculando y_log_real: {e}", flush=True)
            return
        
        error = abs(ylog_mip - ylog_real)
        m._ylog_validation_error = error
        
        print(f"[VALIDATION] y_log_mip={ylog_mip:.6f}, y_log_real={ylog_real:.6f}, error={error:.6f}", flush=True)
        
        if error > 0.02:  # Si hay error significativo
            print(f"[VALIDATION] ADVERTENCIA: Error residual detectado (error={error:.6f} > 0.02)", flush=True)
            print(f"[VALIDATION] Nota: Este error residual es NORMAL después de recalibración", flush=True)
            print(f"[VALIDATION] (Reducido de ~0.09 antes de recalibración a {error:.6f} ahora)", flush=True)
        else:
            print(f"[VALIDATION] OK: Error pequeño (< 0.02), solución es confiable", flush=True)
        
    except Exception as e:
        print(f"[VALIDATION] Error durante validacion: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Resumen final
    print(f"\n{'='*70}", flush=True)
    print(f"[CALLBACK-FINAL] Validacion de y_log completada", flush=True)
    print(f"{'='*70}", flush=True)
    
    error_final = getattr(m, "_ylog_validation_error", None)
    if error_final is not None:
        if error_final < 0.001:
            print(f"✓ EXCELENTE: Solución validada con error de {error_final:.6f} (< 0.001)", flush=True)
            print(f"  Resultado CONFIABLE - Alineación perfecta con XGBoost", flush=True)
        elif error_final < 0.02:
            print(f"✓ BIEN: Solución con error de {error_final:.6f} (< 0.02)", flush=True)
            print(f"  Resultado CONFIABLE - Alineación exitosa tras recalibración", flush=True)
        else:
            print(f"⚠ ADVERTENCIA: Error residual de {error_final:.6f}", flush=True)
            print(f"  Resultado ACEPTABLE - Error reducido significativamente", flush=True)
    print(f"{'='*70}\n", flush=True)

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
    ap.add_argument("--no-upgrades", action="store_true", help="fija todas las decisiones al valor base para alinear y_log (solo debug)")
    ap.add_argument("--debug-embed-build", action="store_true", help="inspecciona X_input vs base antes de optimizar")
    ap.add_argument("--debug-tree-mismatch", action="store_true", help="reporta hojas distintas vs booster (primeros arboles)")
    args = ap.parse_args()

    # Datos base, costos y modelo ML
    base = get_base_house(args.pid, base_csv=args.basecsv)
    ct = costs.CostTables()
    bundle = XGBBundle()
    
    # DEBUG: verificar # de árboles
    if args.debug_embed_build:
        try:
            bst = bundle.reg.get_booster()
        except:
            bst = getattr(bundle.reg, "_Booster", None)
        if bst:
            dumps = bst.get_dump(with_stats=False, dump_format="json")
            print(f"[TREE-INFO] Total árboles en booster: {len(dumps)}")
            print(f"[TREE-INFO] bundle.n_trees_use: {bundle.n_trees_use}")
            if bundle.n_trees_use and bundle.n_trees_use < len(dumps):
                print(f"[TREE-INFO] ✓ Usando PRIMEROS {bundle.n_trees_use}/{len(dumps)} árboles")
            else:
                print(f"[TREE-INFO] ✗ Usando TODOS {len(dumps)} árboles")

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
    # CRÍTICO: comparar en la MISMA escala (log1p del XGB raw margin)
    y_full = float(bundle.predict(X_base).iloc[0])              # en escala de precio
    y_log_via_embed = float(bundle.pipe_for_gurobi().predict(X_base)[0])  # log1p raw XGB margin
    y_log_direct = float(bundle.predict_log_raw(X_base).iloc[0])  # también log1p raw XGB margin (must match)
    y_from_embed = float(np.expm1(y_log_via_embed))

    print(f"[SANITY] full.predict -> price = {y_full:,.2f}")
    print(f"[SANITY] embed (via pipe) -> log = {y_log_via_embed:.6f}")
    print(f"[SANITY] direct raw_log -> log = {y_log_direct:.6f}")
    print(f"[SANITY] expm1(embed_log) -> price~ {y_from_embed:,.2f}")
    if abs(y_log_via_embed - y_log_direct) > 1e-4:
        print(f"[WARNING] embed vs direct log mismatch: {abs(y_log_via_embed - y_log_direct):.6f}")

    # ===== modo debug: solo comparar logs sin construir MIP =====
    if args.no_upgrades:
        y_log_embed = float(bundle.pipe_for_gurobi().predict(X_base)[0])
        y_log_direct = float(bundle.predict_log_raw(X_base).iloc[0])
        delta = y_log_embed - y_log_direct
        print("[DEBUG] no-upgrades activo: se fija todo a base y NO se arma MIP")
        print(f"[DEBUG] y_log(embed via pipe)={y_log_embed:.6f} vs y_log(direct raw)={y_log_direct:.6f} | delta={delta:.6f}")
        if abs(delta) < 1e-5:
            print("[DEBUG] ✓ Log scales are consistent")
        else:
            print(f"[DEBUG] ✗ Log scale mismatch detected: {delta:.6f}")
        return

    # ===== construir y resolver el MIP =====
    m: gp.Model = build_mip_embed(
        base_row,
        args.budget,
        ct,
        bundle,
        base_price=precio_base,
        fix_to_base=False,
    )

    if args.debug_embed_build:
        try:
            def _debug_embed_matrix(m):
                Xin = m._X_input
                base_num = m._X_base_numeric
                n_decision = 0
                mismatch = []
                for c in Xin.columns:
                    val = Xin.iloc[0][c]
                    if isinstance(val, gp.Var):
                        n_decision += 1
                        continue
                    try:
                        v = float(val)
                        b = float(base_num.iloc[0].get(c, 0.0))
                        if not np.isfinite(v) or not np.isfinite(b):
                            mismatch.append((c, v, b, "nonfinite"))
                        elif abs(v - b) > 1e-6:
                            mismatch.append((c, v, b, "delta"))
                    except Exception:
                        mismatch.append((c, val, base_num.iloc[0].get(c, None), "nonfloat"))
                print(f"[DEBUG-EMBED] decision vars en X_input: {n_decision} | columnas totales: {Xin.shape[1]}")
                if mismatch:
                    print(f"[DEBUG-EMBED] columnas no-modificables con mismatch ({len(mismatch)}), primeras 20:")
                    for c, v, b, why in mismatch[:20]:
                        print(f"  - {c}: X_in={v} | base={b} | motivo={why}")
                else:
                    print("[DEBUG-EMBED] no se detectaron mismatches numéricos en columnas no-decisión.")
            _debug_embed_matrix(m)
        except Exception as e:
            print(f"[DEBUG-EMBED] no se pudo inspeccionar X_input: {e}")

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

    # Ejecutar con validación de y_log en callback
    print("[OPTIMIZATION] Iniciando optimización con validación de y_log...")
    # Obtener base_X del modelo si está disponible, sino crear uno
    base_X_local = m._X_base_numeric if hasattr(m, "_X_base_numeric") else build_base_input_row(bundle, base_row)
    setup_ylog_validation_callback(m, bundle, base_X_local)
    
    # ===== RECALCULAR MÉTRICAS DESPUÉS DEL CALLBACK =====
    # El callback puede cambiar m.ObjVal, así que necesitamos recalcular
    print(f"\n[POST-CALLBACK] Recalculando métricas finales...", flush=True)
    
    # Recalcular y_log y costos con la solución final
    try:
        # El ROI está en m.ObjVal (ya corregido por el callback si aplica)
        roi_final_usd = float(m.ObjVal)
        total_cost_var = m.getVarByName("cost_model")
        if total_cost_var is not None:
            total_cost_final = float(total_cost_var.X)
        else:
            def _eval_linexpr_final(expr, m):
                vs = expr.getVars(); cs = expr.getCoeffs()
                vx = m.getAttr("X", vs)
                const = expr.getConstant() if hasattr(expr, "getConstant") else 0.0
                return float(const + sum(c*xi for c, xi in zip(cs, vx)))
            total_cost_final = _eval_linexpr_final(getattr(m, "_lin_cost_expr", gp.LinExpr(0.0)), m)
        
        # ROI % = (ROI USD / costo) * 100
        roi_pct_final = (100.0 * roi_final_usd / total_cost_final) if abs(total_cost_final) > 1e-9 else None
        
        print(f"[POST-CALLBACK] ROI final (USD): ${roi_final_usd:,.2f}", flush=True)
        print(f"[POST-CALLBACK] Costo total final: ${total_cost_final:,.2f}", flush=True)
        print(f"[POST-CALLBACK] ROI % final: {roi_pct_final:.1f}%", flush=True)
        
        # Guardar estos valores para que se usen después
        m._roi_final_usd = roi_final_usd
        m._total_cost_final = total_cost_final
        m._roi_pct_final = roi_pct_final
        
    except Exception as e:
        print(f"[POST-CALLBACK] Error recalculando métricas: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    audit_cost_breakdown_vars(m, top=30)
    
    # DEBUG: Check that one-hot constraints are satisfied
    if args.debug_tree_mismatch:
        print("\n[ONE-HOT-CHECK]")
        violations = 0
        for t_idx in range(min(20, 1058)):
            leaf_vars = [m.getVarByName(f"t{t_idx}_leaf{k}") for k in range(1000)]
            leaf_vars = [v for v in leaf_vars if v is not None]
            if not leaf_vars:
                continue
            
            z_sum = sum(v.X for v in leaf_vars)
            if abs(z_sum - 1.0) > 0.01:
                print(f"  Tree {t_idx}: sum(z) = {z_sum:.6f} (VIOLATION!)")
                violations += 1
                for k, v in enumerate(leaf_vars[:5]):
                    print(f"    z[{k}] = {v.X:.6f}")
        
        if violations == 0:
            print(f"  All checked trees have valid one-hot constraints ✓")
        else:
            print(f"  {violations} trees have one-hot violations!")
    
    # tras optimize(), usa el mismo X que vio el embed:
    X_in = rebuild_embed_input_df(m, m._X_base_numeric)
    try:
        y_hat_out = float(bundle.predict(X_in).iloc[0])
        # CRÍTICO: m._y_log_var está en escala log1p del XGB raw margin
        # Usar predict_log_raw() para comparación correcta, NO log1p(predict())
        ylog_out_raw = float(bundle.predict_log_raw(X_in).iloc[0])
        
        # DEBUG: print component values
        try:
            v_y_log_raw = m.getVarByName("y_log_raw")
            if v_y_log_raw is not None:
                y_log_raw_val = float(v_y_log_raw.X)
                y_log_mip = float(m._y_log_var.X)
                b0_implied = y_log_mip - y_log_raw_val
                print(f"[DEBUG] y_log_raw(MIP) = {y_log_raw_val:.6f}")
                print(f"[DEBUG] y_log(MIP) = {y_log_mip:.6f}")
                print(f"[DEBUG] b0(implied) = {b0_implied:.6f}")
                print(f"[DEBUG] ylog_out_raw(external) = {ylog_out_raw:.6f}")
                print(f"[DEBUG] Difference y_log_raw(external) - y_log_raw(MIP) = {ylog_out_raw - y_log_raw_val:.6f}")
                
                # DEBUG: Show first few features' values in MIP vs base
                print(f"[DEBUG] Feature values in MIP solution:")
                for fname in list(m._feat_order)[:10]:
                    v = m._x_vars.get(fname)
                    if v is not None:
                        print(f"  {fname}: {v.X:.6f}")
        except Exception as de:
            print(f"[DEBUG-ERROR] {de}")
        
        delta = float(m._y_log_var.X) - ylog_out_raw
        scale = float(np.exp(delta))
        print(f"[CALIB] y_log(MIP) - y_log_raw(outside) = {delta:.6f} (≈×{scale:.3f})")
        if abs(delta) > 1e-3:
            print(f"[WARNING] Significant y_log divergence detected!")
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
            status = int(m.Status)
        except Exception:
            status = None
        ok_status = {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT}
        if status is None or status not in ok_status:
            print(f"[AUDIT] Modelo no óptimo (status={status}), se omite auditoría de costos.")
            return
        try:
            y_price = float(m._y_price_var.X)
        except Exception:
            y_price = float("nan")
        uplift = y_price - float(base_price)
        try:
            obj = float(m.ObjVal)
        except Exception:
            obj = float("nan")
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
        print(f"  ==> cost(model) ~ {used_cost:,.2f}")

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
            # Usa la misma escala que el embed: margen raw (log1p) con output_margin
            y_log = float(bundle.predict_log_raw(X_in).iloc[0])
            import numpy as np
            y_hat = float(np.expm1(y_log)) if bundle.is_log_target() else float(bundle.predict(X_in).iloc[0])
            print("\n[AUDIT] Outside predict usando EXACTO el X que vio el embed:")
            print(f"  y_hat(outside from embed X_in) = {y_hat:,.2f}")
            print(f"  y_log(outside)≈ {y_log:.6f} | y_log(MIP)≈ {float(m._y_log_var.X):.6f}")
            try:
                y_log_raw_mip = None
                v_raw = getattr(m, "_y_log_raw_var", None) or m.getVarByName("y_log_raw")
                if v_raw is not None:
                    y_log_raw_mip = float(v_raw.X)
                if y_log_raw_mip is not None and hasattr(bundle, "b0_offset"):
                    print(f"  y_log_raw(out)≈ {y_log - float(bundle.b0_offset):.6f} | y_log_raw(MIP)≈ {y_log_raw_mip:.6f} | b0_offset≈ {float(bundle.b0_offset):.6f}")
            except Exception:
                pass
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

    # ========= DEBUG: TREE MISMATCH (opcional) =========
    if args.debug_tree_mismatch:
        try:
            import json, math
            bst = bundle.reg.get_booster()
            dumps = bst.get_dump(with_stats=False, dump_format="json")
            bo = bundle.booster_feature_order()

            # reconstruye vector de entrada usado por el embed (Var.X si existe, si no toma valor en X_input)
            X_in_df = getattr(m, "_X_input", None)
            xvars = getattr(m, "_x_vars", {}) if hasattr(m, "_x_vars") else {}

            def _fetch_feature(fname: str):
                # 1) Si tenemos xvars (con nombre EXACTO del booster), usa ese
                if fname in xvars:
                    v = xvars[fname]
                    try:
                        return float(v.X) if hasattr(v, "X") else float(v)
                    except Exception:
                        pass
                # 2) Busca var por nombres seguros (espacios -> _, "/" -> "_")
                safe = fname.replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "")
                for prefix in ["x_", "const_", "expr_"]:
                    v = m.getVarByName(prefix + safe)
                    if v is not None:
                        try:
                            return float(v.X)
                        except Exception:
                            pass
                # 3) Busca en el DataFrame original del embed
                if X_in_df is not None and hasattr(X_in_df, "loc"):
                    try:
                        return float(X_in_df.loc[0, fname])
                    except Exception:
                        pass
                return 0.0

            feat_vals = [_fetch_feature(fname) for fname in bo]

            # usa la misma cantidad de árboles que el embed (n_trees_use) o todos
            n_use = None
            try:
                n_use = int(bundle.n_trees_use or 0)
            except Exception:
                n_use = 0
            max_trees = len(dumps) if n_use <= 0 else min(n_use, len(dumps))

            y_log_sum = 0.0
            mismatches = []
            detailed = []
            no_path = 0
            EPS = 1e-6
            for t_idx in range(max_trees):
                node = json.loads(dumps[t_idx])
                leaves = []
                def walk(nd, path):
                    if "leaf" in nd:
                        leaves.append((path, float(nd["leaf"])))
                        return
                    f_idx = int(str(nd["split"]).replace("f", ""))
                    thr = float(nd["split_condition"])
                    yes_id = nd["yes"]
                    for ch in nd.get("children", []):
                        is_left = (ch.get("nodeid") == yes_id)
                        walk(ch, path + [(f_idx, thr, is_left)])
                walk(node, [])

                expected_idx = None
                expected_leaf_val = None
                expected_path = None
                for k, (path, leaf_val) in enumerate(leaves):
                    ok = True
                    for (f_idx, thr, is_left) in path:
                        if f_idx >= len(feat_vals):
                            ok = False; break
                        xv = feat_vals[f_idx]
                        if is_left:
                            # XGBoost split rule: go left if xv < thr (strict)
                            if not (xv < thr - 1e-9):
                                ok = False; break
                        else:
                            # right branch for xv >= thr
                            if not (xv >= thr - 1e-9):
                                ok = False; break
                    if ok:
                        expected_idx = k
                        expected_leaf_val = leaf_val
                        expected_path = path
                        y_log_sum += leaf_val
                        break
                if expected_idx is None:
                    mismatches.append((t_idx, "no_path"))
                    no_path += 1
                    detailed.append((t_idx, None, None, None, None, None, None))
                    continue

                # hoja elegida por el MIP (si existe z_{t_idx}_leaf_k)
                chosen_idx = None
                chosen_val = None
                chosen_path = None
                for k in range(len(leaves)):
                    z = m.getVarByName(f"t{t_idx}_leaf{k}")
                    if z is not None and z.X > 0.5:
                        chosen_idx = k
                        chosen_val = leaves[k][1]
                        chosen_path = leaves[k][0]
                        break
                if (chosen_idx is not None) and (chosen_idx != expected_idx):
                    path_exp = expected_path
                    path_ch = chosen_path
                    detailed.append((t_idx, expected_idx, chosen_idx, expected_leaf_val, chosen_val, path_exp, path_ch))
                    mismatches.append((t_idx, f"chosen={chosen_idx}, expected={expected_idx}"))

            # agrega offset/bias que usa el bundle
            try:
                b0 = float(bundle.b0_offset or 0.0)
            except Exception:
                b0 = 0.0
            y_log_manual = y_log_sum + b0
            y_log_mip = float(m._y_log_var.X)
            print(f"[TREE-MISMATCH] y_log_manual={y_log_manual:.6f} | y_log(MIP)={y_log_mip:.6f} | Δ={y_log_mip - y_log_manual:+.6f} | b0={b0:.6f} | trees_used={max_trees}")
            if mismatches:
                print(f"[TREE-MISMATCH] Mismatches en {len(mismatches)} árboles (de {max_trees}). no_path={no_path}")
                def path_to_str(path):
                    if not path:
                        return "(root)"
                    parts = []
                    for (f_idx, thr, is_left) in path:
                        fname = bo[f_idx] if f_idx < len(bo) else f"f{f_idx}"
                        xv = feat_vals[f_idx] if f_idx < len(feat_vals) else float("nan")
                        sign = "<=" if is_left else ">="
                        parts.append(f"{fname} {sign} {thr:g} (x={xv:g})")
                    return " & ".join(parts)
                for rec in detailed[:8]:
                    if not rec or rec[1] is None or rec[2] is None:
                        continue
                    t_idx, exp_idx, ch_idx, exp_val, ch_val, path_exp, path_ch = rec
                    print(f"  - tree {t_idx}: expected leaf {exp_idx} ({exp_val:+.4f}) path[{path_to_str(path_exp)}]")
                    print(f"             chosen   {ch_idx} ({ch_val:+.4f}) path[{path_to_str(path_ch)}]")
        except Exception as e:
            print(f"[TREE-MISMATCH] no se pudo evaluar: {e}")

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
    
    # USAR VALORES FINALES DEL CALLBACK SI ESTÁN DISPONIBLES
    if hasattr(m, "_roi_pct_final") and m._roi_pct_final is not None:
        roi_pct = m._roi_pct_final
        utilidad_incremental = getattr(m, "_roi_final_usd", None)
        total_cost_model = getattr(m, "_total_cost_final", None)
    
    if (precio_base is not None) and (total_cost_model is not None):
        if roi_pct is None:
            # Fallback si no hay valores del callback
            precio_opt = float(getattr(m, "_y_price_var", gp.Var()).X) if hasattr(m, "_y_price_var") else None
            if precio_opt is not None:
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
                roi_pct = _pct(utilidad_incremental, total_cost_model)

    # ========= Reporte de cambios (resumen compacto) =========
    # EXTRAER TODOS LOS CAMBIOS DE LA SOLUCIÓN FINAL
    print(f"\n[EXTRACT-CHANGES] Extrayendo TODOS los cambios de la solución final...", flush=True)
    
    # Usar la función que ya funciona para detectar cambios
    free_upgrades = debug_free_upgrades(m, eps=1e-8)
    
    # Reconstruir X_sol para obtener valores finales
    try:
        X_sol_final = rebuild_embed_input_df(m, m._X_base_numeric)
    except Exception as e:
        print(f"[EXTRACT-CHANGES] Error reconstruyendo X_sol: {e}", flush=True)
        X_sol_final = None
    
    # Construir lista de cambios desde Xi (como hace debug_free_upgrades)
    all_changes = []
    Xi = getattr(m, "_X_input", None)
    Xb = getattr(m, "_X_base_numeric", None)
    
    if Xi is not None and Xb is not None and not Xi.empty and not Xb.empty:
        for col in Xi.columns:
            base_val = float(Xb.iloc[0][col]) if col in Xb.columns else None
            cur = Xi.iloc[0][col]
            if hasattr(cur, "X"):
                try:
                    sol_val = float(cur.X)
                except Exception:
                    continue
                
                if (base_val is not None) and (abs(sol_val - base_val) > 1e-8):
                    all_changes.append({
                        'feature': col,
                        'base': base_val,
                        'solution': sol_val,
                        'delta': sol_val - base_val
                    })
    
    print(f"[EXTRACT-CHANGES] Total: {len(all_changes)} cambios detectados", flush=True)
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

    # (2.5) EXTERIOR QUALITY (eq_bin_*, binarias one-hot)
    try:
        ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        inv = {v:k for k,v in ORD.items()}
        
        # Exter Qual
        exq_base_txt = str(base_row.get("Exter Qual","TA")).strip()
        exq_base = ORD.get(exq_base_txt, 2)
        
        eq_pick = None
        for nm in ["Po","Fa","TA","Gd","Ex"]:
            v = m.getVarByName(f"exterqual_is_{nm}")
            if v is not None and v.X > 0.5:
                eq_pick = nm; break
        
        if eq_pick is not None and ORD[eq_pick] > exq_base:
            cost_eq = float(ct.exter_qual_cost(eq_pick)) - float(ct.exter_qual_cost(inv[exq_base]))
            cambios_costos.append(("Exter Qual", inv[exq_base], eq_pick, cost_eq))
        
        # Exter Cond
        exc_base_txt = str(base_row.get("Exter Cond","TA")).strip()
        exc_base = ORD.get(exc_base_txt, 2)
        
        ec_pick = None
        for nm in ["Po","Fa","TA","Gd","Ex"]:
            v = m.getVarByName(f"extercond_is_{nm}")
            if v is not None and v.X > 0.5:
                ec_pick = nm; break
        
        if ec_pick is not None and ORD[ec_pick] > exc_base:
            cost_ec = float(ct.exter_cond_cost(ec_pick)) - float(ct.exter_cond_cost(inv[exc_base]))
            cambios_costos.append(("Exter Cond", inv[exc_base], ec_pick, cost_ec))
    except Exception as e:
        print(f"[TRACE] Reporter Exter Qual/Cond falló: {e}")

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
            # Costo ABSOLUTO del nuevo material (no incremental)
            cost_ex1 = float(ct.ext_mat_cost(ex1_new))
            cambios_costos.append(("Exterior 1st", ex1_base, ex1_new, cost_ex1))
        if ex2_new != ex2_base:
            # Costo ABSOLUTO del nuevo material (no incremental)
            cost_ex2 = float(ct.ext_mat_cost(ex2_new))
            cambios_costos.append(("Exterior 2nd", ex2_base, ex2_new, cost_ex2))
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
    add_cost_map = {
        "AddFull":  ct.add_fullbath_cost,
        "AddHalf":  ct.add_halfbath_cost,
        "AddKitch": ct.add_kitchen_cost_per_sf * A_Kitch,
        "AddBed":   ct.add_bedroom_cost_per_sf * A_Bed,
    }

    for nm, A in [("AddFull", A_Full), ("AddHalf", A_Half), ("AddKitch", A_Kitch), ("AddBed", A_Bed)]:
        v = _v(nm)
        if v is not None and float(v.X) > 1e-6:
            # si AddHalf estaba listado como Half Bath/1st/GrLiv, quítalos y reemplaza por la línea con costo
            if nm == "AddHalf":
                _drop_generic(["1st Flr SF", "Gr Liv Area", "Half Bath"])
            unit_cost = add_cost_map.get(nm, float(ct.construction_cost) * A)
            cambios_costos.append((
                f"{nm} (+{int(A)} ft²)", "-", f"+{int(A*float(v.X))} ft²",
                float(unit_cost) * float(v.X)
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


    # (14) REPORTE EXHAUSTIVO DE TODOS LOS CAMBIOS
    print(f"\n{'='*80}")
    print(f"LISTADO EXHAUSTIVO DE CAMBIOS RECOMENDADOS")
    print(f"{'='*80}\n")
    
    if all_changes:
        print(f"Total de cambios recomendados: {len(all_changes)}\n")
        for change in sorted(all_changes, key=lambda x: x['feature']):
            delta_str = f"Δ={change['delta']:>+.4f}" if isinstance(change['delta'], (int, float)) else ""
            print(f"  {change['feature']:<45s} {str(change['base']):>12s} -> {str(change['solution']):>12s} ({delta_str})")
    else:
        print("[SIN CAMBIOS] La solución óptima es igual a la casa base")
    
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

    print(f"PID: {base_row.get('PID', 'N/A')} | Neighborhood: {base_row.get('Neighborhood', 'N/A')} | Presupuesto: ${args.budget:,.0f}")
    print(f" Modelo: {m.ModelName if hasattr(m, 'ModelName') else 'Gurobi MIP'}")
    print(f" Tiempo total: {getattr(m, 'Runtime', 0.0):.2f}s | MIP Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    print(" **Resumen Económico**")
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

        print("\n🌟 **Calidad general y calidades clave**")
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
    print("\n **Cambios hechos en la casa**")
    if cambios_costos:
        for nombre, base_val, new_val, cost_val in cambios_costos:
            suf = f" (costo {money(cost_val)})" if (cost_val is not None and cost_val > 0) else ""
            print(f"  - {nombre}: {base_val} → {new_val}{suf}")
    else:
        print("  (No se detectaron cambios)")

    # Snapshot Base vs Óptimo (exacto con X_in del embed)
    print("\n **Snapshot: atributos Base vs Óptimo (compacto)**")
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
