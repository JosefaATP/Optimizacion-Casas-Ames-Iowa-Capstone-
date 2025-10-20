# optimization/remodel/run_opt.py
import argparse
import pandas as pd
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import (
    XGBBundle,
    _coerce_quality_ordinals_inplace,
    _coerce_utilities_ordinal_inplace,
)
from .gurobi_model import build_mip_embed
from .features import MODIFIABLE

from shutil import get_terminal_size

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

def print_kv(rows: list[tuple[str, str | float | int | None]], pad_left=2, key_w=26, width=None):
    """Imprime pares clave-valor alineados."""
    w = width or _termw()
    for k, v in rows:
        if v is None:
            vtxt = "N/A"
        elif isinstance(v, (int, float)):
            vtxt = f"{v:,.2f}" if abs(v - round(v)) > 1e-6 else f"{int(round(v)):,}"
        else:
            vtxt = str(v)
        print("║ " + " " * pad_left + f"{k:<{key_w}} : {vtxt}".ljust(w - 4) + " ║")

def print_changes_table(cambios: list[tuple[str, object, object, float | None]], width=None):
    """Tabla compacta: Nombre | Base → Nuevo | Costo"""
    if not cambios:
        print("║   (No se detectaron cambios)".ljust((width or _termw()) - 2) + "║")
        return
    w = width or _termw()
    name_w = 28
    val_w  = 22
    head = f"{'Cambio':<{name_w}} | {'Base → Nuevo':<{val_w}} | {'Costo':>12}"
    print("║   " + head.ljust(w - 6) + " ║")
    print("║   " + ("-" * len(head)).ljust(w - 6) + " ║")
    for nombre, base_val, new_val, cost_val in cambios:
        base_txt = f"{base_val}"
        new_txt  = f"{new_val}"
        costo    = "-" if (cost_val is None or abs(cost_val) < 1e-9) else f"${cost_val:,.0f}"
        line = f"{nombre:<{name_w}} | {base_txt} → {new_txt:<{val_w - len(base_txt) - 3}} | {costo:>12}"
        print("║   " + line.ljust(w - 6) + " ║")

def print_snapshot_table(base_row: dict, opt_row: dict | None = None, width=None, max_rows=60):
    """Snapshot base vs óptimo en tabla compacta (limita filas para no inundar la consola)."""
    w = width or _termw()
    keys = sorted(set(base_row.keys()) | set((opt_row or {}).keys()))
    name_w = 28
    val_w  = (w - 8 - name_w) // 2  # espacio para dos columnas de valores
    head = f"{'Atributo':<{name_w}} | {'Base':<{val_w}} | {'Óptimo':<{val_w}}"
    print("║   " + head.ljust(w - 6) + " ║")
    print("║   " + ("-" * len(head)).ljust(w - 6) + " ║")

    def _fmt(v):
        import pandas as _pd
        try:
            fv = float(_pd.to_numeric(v, errors="coerce"))
            if _pd.isna(fv):
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

# -----------------------------
# Mapeos Utilities (ordinales)
# -----------------------------
UTIL_TO_ORD = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}
ORD_TO_UTIL = {v: k for k, v in UTIL_TO_ORD.items()}

def _safe_util_ord(val) -> int:
    """Convierte val a 0..3 de forma robusta."""
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.notna(v) and int(v) in (0, 1, 2, 3):
            return int(v)
    except Exception:
        pass
    return UTIL_TO_ORD.get(str(val), 0)

def _qual_to_ord(val, default=2):
    MAP = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.notna(v):
            iv = int(v)
            if iv in (0,1,2,3,4):
                return iv
    except Exception:
        pass
    return MAP.get(str(val).strip(), default)

# -----------------------------
# Helpers de impresión
# -----------------------------
def money(v: float) -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return str(v)
    if pd.isna(f):
        return "-"
    return f"${f:,.0f}"

def frmt_num(x) -> str:
    """Formatea numeros, si viene string/NaN, devuelve '-' o el string."""
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return str(x)
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}" if abs(v - round(v)) > 1e-6 else f"{int(round(v))}"

def _getv(m, *names):
    """Devuelve la primera variable que exista entre varios alias."""
    for nm in names:
        v = m.getVarByName(nm)
        if v is not None:
            return v
    # plan C: buscar ignorando espacios/underscores y case
    target_keys = [nm.replace(" ", "").replace("_", "").lower() for nm in names]
    for v in m.getVars():
        key = v.VarName.replace(" ", "").replace("_", "").lower()
        if key in target_keys:
            return v
    return None

# -----------------------------
# Construir fila base con dummies
# -----------------------------
def _row_with_dummies(base_row: pd.Series, feat_order: list[str]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for col in feat_order:
        if col in base_row.index:
            vals[col] = base_row[col]
        else:
            if "_" in col:
                base_col, dummy_val = col.split("_", 1)
                if base_col in base_row.index:
                    vals[col] = 1.0 if str(base_row[base_col]) == dummy_val else 0.0
                else:
                    vals[col] = 0.0
            else:
                vals[col] = 0.0
    return vals

# =============================
# main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--basecsv", type=str, default=None, help="ruta alternativa al CSV base")
    args = ap.parse_args()

    # Datos base, costos y modelo
    base = get_base_house(args.pid, base_csv=args.basecsv)
    ct = costs.CostTables()
    bundle = XGBBundle()

    # ===== precio_base (ValorInicial) con el pipeline COMPLETO =====
    feat_order = bundle.feature_names_in()
    X_base = pd.DataFrame([_row_with_dummies(base.row, feat_order)], columns=feat_order)

    # normalizaciones defensivas
    _coerce_quality_ordinals_inplace(X_base, getattr(bundle, "quality_cols", []))
    _coerce_utilities_ordinal_inplace(X_base)

    precio_base = float(bundle.predict(X_base).iloc[0])

    # ============== DEBUGS (opcionales) ==============
    # Kitchen Qual
    if "Kitchen Qual" in feat_order:
        X_dbg = X_base.copy()
        vals = []
        for q in [0, 1, 2, 3, 4]:
            X_dbg.loc[:, "Kitchen Qual"] = q
            vals.append((q, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Kitchen Qual -> precio:", vals)

    # Utilities
    if "Utilities" in feat_order:
        vals = []
        for k, name in enumerate(["ELO", "NoSeWa", "NoSewr", "AllPub"]):
            X_dbg = X_base.copy()
            X_dbg.loc[:, "Utilities"] = k
            vals.append((k, name, float(bundle.predict(X_dbg).iloc[0])))
        print("DEBUG Utilities -> precio:", [(k, name, round(p, 2)) for (k, name, p) in vals])

    # Roof Style
    try:
        if any(c.startswith("Roof Style_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Roof Style_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]:
                col = f"Roof Style_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Roof Style -> precio:", vals)
    except Exception:
        pass

    # Roof Matl
    try:
        if any(c.startswith("Roof Matl_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Roof Matl_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]:
                col = f"Roof Matl_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Roof Matl -> precio:", vals)
    except Exception:
        pass

    # Exterior 1st
    if any(c.startswith("Exterior 1st_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("Exterior 1st_")]:
            Xd[c] = 0.0
        vals = []
        for nm in ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
                   "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl"]:
            col = f"Exterior 1st_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG Exterior 1st -> precio:", vals)

    # Exter Qual / Exter Cond
    if "Exter Qual" in X_base.columns:
        Xd = X_base.copy()
        vals = []
        for q in [0,1,2,3,4]:
            Xd.loc[:, "Exter Qual"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Exter Qual -> precio:", vals)

    if "Exter Cond" in X_base.columns:
        Xd = X_base.copy()
        vals = []
        for q in [0,1,2,3,4]:
            Xd.loc[:, "Exter Cond"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Exter Cond -> precio:", vals)

    # Mas Vnr Type
    try:
        if any(c.startswith("Mas Vnr Type_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Mas Vnr Type_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["BrkCmn","BrkFace","CBlock","Stone","No aplica"]:
                col = f"Mas Vnr Type_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Mas Vnr Type -> precio:", vals)
    except Exception:
        pass

    # Electrical
    if any(c.startswith("Electrical_") for c in X_base.columns):
        X_dbg = X_base.copy()
        for c in [c for c in X_dbg.columns if c.startswith("Electrical_")]:
            X_dbg[c] = 0.0
        vals = []
        for nm in ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]:
            col = f"Electrical_{nm}"
            if col in X_dbg.columns:
                X_dbg[col] = 1.0
                vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                X_dbg[col] = 0.0
        print("DEBUG Electrical -> precio:", vals)

    # Central Air
    try:
        if any(c.startswith("Central Air_") for c in X_base.columns):
            X_dbg = X_base.copy()
            for c in [c for c in X_dbg.columns if c.startswith("Central Air_")]:
                X_dbg[c] = 0.0
            vals = []
            for nm in ["N", "Y"]:
                col = f"Central Air_{nm}"
                if col in X_dbg.columns:
                    X_dbg[col] = 1.0
                    vals.append((nm, float(bundle.predict(X_dbg).iloc[0])))
                    X_dbg[col] = 0.0
            print("DEBUG Central Air -> precio:", vals)
        else:
            print("DEBUG Central Air -> precio: []")
    except Exception:
        pass


        # === DEBUG Garage Finish → precio: barrido Fin, RFn, Unf, No aplica ===
    try:
        cols_gf = ["Garage Finish_Fin", "Garage Finish_RFn", "Garage Finish_Unf", "Garage Finish_No aplica"]

        if any(col in X_base.columns for col in cols_gf):
            print("\nDEBUG Garage Finish → precio:")
            for gf in cols_gf:
                # copio la base
                df_test = X_base.copy()

                # apago todas las dummies de Garage Finish
                for c in cols_gf:
                    if c in df_test.columns:
                        df_test.loc[:, c] = 0

                # activo solo una
                if gf in df_test.columns:
                    df_test.loc[:, gf] = 1

                # predigo con el bundle del pipeline embebido
                y_pred = bundle.predict(df_test)[0]
                print(f"   {gf.replace('Garage Finish_', ''):>10}: {y_pred:,.0f}")
    except Exception as e:
        print(f"(debug garage finish omitido: {e})")


    # === DEBUG Pool QC → precio: barrido ordinal (0–4 o Po–Ex) ===
    try:
        if "Pool QC" in X_base.columns:
            print("\nDEBUG Pool QC → precio (ordinal):")
            Xd = X_base.copy()

            # probar distintos niveles de calidad
            niveles = [0, 1, 2, 3, 4]
            etiquetas = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}

            vals = []
            for q in niveles:
                Xd.loc[:, "Pool QC"] = q
                y_pred = float(bundle.predict(Xd).iloc[0])
                vals.append((q, etiquetas[q], y_pred))
                print(f"   {etiquetas[q]:>10}: {y_pred:,.0f}")
        else:
            print("(debug pool qc omitido: columna Pool QC no encontrada en X_base)")
    except Exception as e:
        print(f"(debug pool qc omitido: {e})")


            # === DEBUG Ampliaciones → precio: barrido +10%, +20%, +30% ===

    try:
        AMPL_COMPONENTES = [
            "Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
            "3Ssn Porch", "Screen Porch", "Pool Area"
        ]

        print("\nDEBUG Ampliaciones → precio:")
        for comp in AMPL_COMPONENTES:
            if comp not in X_base.columns:
                continue
            base_val = float(pd.to_numeric(X_base.loc[0, comp], errors="coerce") or 0.0)
            if base_val <= 0:
                continue

            X_dbg = X_base.copy()
            vals = []
            for pct in [10, 20, 30]:
                X_dbg.loc[:, comp] = pd.to_numeric(X_dbg[comp], errors="coerce").astype(float)
                X_dbg.loc[0, comp] = float(base_val) * (1 + pct/100.0)


                y_pred = float(bundle.predict(X_dbg).iloc[0])
                vals.append((pct, y_pred))
                X_dbg.loc[0, comp] = base_val  # restaurar

            print(f"  {comp:>15}: base={base_val:,.0f} → "
                  + ", ".join([f"+{p}%={v:,.0f}" for p, v in vals]))
    except Exception as e:
        print(f"(debug ampliaciones omitido: {e})")

    
# === DEBUG Garage Qual / Garage Cond → precio (ordinal 0–4 o Po–Ex) ===
    try:
        if "Garage Qual" in X_base.columns and "Garage Cond" in X_base.columns:
            print("\nDEBUG GarageQual / GarageCond → precio:")

            # Copia base
            Xd = X_base.copy()

            # Definición de niveles
            niveles = [0, 1, 2, 3, 4]
            etiquetas = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}

            # Resultados individuales
            print("   → Garage Qual:")
            for q in niveles:
                Xd.loc[:, "Garage Qual"] = q
                y_pred = float(bundle.predict(Xd).iloc[0])
                print(f"      {etiquetas[q]:>8}: {y_pred:,.0f}")

            print("   → Garage Cond:")
            for q in niveles:
                Xd.loc[:, "Garage Cond"] = q
                y_pred = float(bundle.predict(Xd).iloc[0])
                print(f"      {etiquetas[q]:>8}: {y_pred:,.0f}")
        else:
            print("(debug garage qual/cond omitido: columnas no encontradas en X_base)")
    except Exception as e:
        print(f"(debug garage qual/cond omitido: {e})")

        # === DEBUG PavedDrive → precio: barrido N, P, Y ===
    try:
        cols_pd = ["Paved Drive_N", "Paved Drive_P", "Paved Drive_Y"]

        if any(col in X_base.columns for col in cols_pd):
            print("\nDEBUG PavedDrive → precio:")
            for pd_cat in ["N", "P", "Y"]:
                df_test = X_base.copy()

                # apagar todas las dummies
                for c in cols_pd:
                    if c in df_test.columns:
                        df_test.loc[:, c] = 0

                # activar solo una categoría
                col = f"Paved Drive_{pd_cat}"
                if col in df_test.columns:
                    df_test.loc[:, col] = 1

                y_pred = float(bundle.predict(df_test).iloc[0])
                print(f"   {pd_cat:>2}: {y_pred:,.0f}")
    except Exception as e:
        print(f"(debug paved drive omitido: {e})")

        # === DEBUG Fence → precio: barrido de categorías ===
    try:
        cols_fence = [f"Fence_{f}" for f in ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]]
        if any(col in X_base.columns for col in cols_fence):
            print("\nDEBUG Fence → precio:")
            for f in ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]:
                df_test = X_base.copy()
                for c in cols_fence:
                    if c in df_test.columns:
                        df_test.loc[:, c] = 0
                col = f"Fence_{f}"
                if col in df_test.columns:
                    df_test.loc[:, col] = 1
                y_pred = float(bundle.predict(df_test).iloc[0])
                print(f"   {f:>5}: {y_pred:,.0f}")
    except Exception as e:
        print(f"(debug fence omitido: {e})")




    # ============ FIN DEBUGS ============
    # --- DEBUG: Heating (TIPO) -> precio ---
    if any(c.startswith("Heating_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("Heating_")]:
            Xd[c] = 0.0
        vals = []
        for nm in ["Floor","GasA","GasW","Grav","OthW","Wall"]:
            col = f"Heating_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG Heating (TIPO) -> precio:", vals)

    # --- DEBUG: Heating QC (NIVEL) -> precio ---
    if "Heating QC" in X_base.columns:
        vals = []
        for q in [0,1,2,3,4]:
            Xd = X_base.copy()
            Xd.loc[:, "Heating QC"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Heating QC (nivel) -> precio:", vals)

    # --- DEBUG: Basement Cond (NIVEL) -> precio ---
    if "Bsmt Cond" in feat_order:
        Xd = X_base.copy()
        vals = []
        for q in [-1,0,1,2,3,4]:
            Xd.loc[:, "Bsmt Cond"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Bsmt Cond -> precio:", vals)

    # DEBUG: terminar 100 ft² vs no terminar (manteniendo total constante)
    if all(c in feat_order for c in ["BsmtFin SF 1","Bsmt Unf SF","Total Bsmt SF"]):
        Xd = X_base.copy()
        def price(df): return float(bundle.predict(df).iloc[0])
        b1 = float(pd.to_numeric(base.row.get("BsmtFin SF 1"), errors="coerce") or 0.0)
        bu = float(pd.to_numeric(base.row.get("Bsmt Unf SF"),   errors="coerce") or 0.0)
        tb = float(pd.to_numeric(base.row.get("Total Bsmt SF"), errors="coerce") or (b1+bu))
        if bu >= 100:
            Xd.loc[:, "BsmtFin SF 1"] = b1 + 100
            Xd.loc[:, "Bsmt Unf SF"]  = bu - 100
            Xd.loc[:, "Total Bsmt SF"] = tb
            p_up = price(Xd)
            print(f"DEBUG Bsmt: +100 ft² terminados → Δprecio = {p_up - precio_base:,.2f}")
        
        # --- DEBUG: BsmtFin Type 1 -> precio ---
    B1_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]
    if any(c.startswith("BsmtFin Type 1_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("BsmtFin Type 1_")]:
            Xd[c] = 0.0
        vals = []
        for nm in B1_TYPES:
            col = f"BsmtFin Type 1_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG BsmtFin Type 1 -> precio:", [(nm, round(p, 2)) for nm, p in vals])

    # --- DEBUG: BsmtFin Type 2 -> precio ---
    B2_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]
    if any(c.startswith("BsmtFin Type 2_") for c in X_base.columns):
        Xd = X_base.copy()
        for c in [c for c in Xd.columns if c.startswith("BsmtFin Type 2_")]:
            Xd[c] = 0.0
        vals = []
        for nm in B2_TYPES:
            col = f"BsmtFin Type 2_{nm}"
            if col in Xd.columns:
                Xd[col] = 1.0
                vals.append((nm, float(bundle.predict(Xd).iloc[0])))
                Xd[col] = 0.0
        print("DEBUG BsmtFin Type 2 -> precio:", [(nm, round(p, 2)) for nm, p in vals])

    # --- DEBUG: Fireplace Qu -> precio ---
    if "Fireplace Qu" in X_base.columns:
        vals = []
        for q in [-1,0,1,2,3,4]:
            Xd = X_base.copy()
            Xd.loc[:, "Fireplace Qu"] = q
            vals.append((q, float(bundle.predict(Xd).iloc[0])))
        print("DEBUG Fireplace Qu -> precio:", vals)



    # ============ FIN DEBUGS ============

    # ===== construir MIP =====
    m: gp.Model = build_mip_embed(base.row, args.budget, ct, bundle, base_price=precio_base)



    # Ajustes de resolución
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = PARAMS.time_limit
    m.Params.LogToConsole = PARAMS.log_to_console

    # ===== objetivo como (Final - Inicial - Costos) =====
    m.ObjCon = -precio_base

    # Optimizar
    m.optimize()

    for c in m.getConstrs():
        if "BUDGET" in c.ConstrName:
            print(f"[DBG] Presupuesto usado efectivamente: RHS={c.RHS:.2f}, Slack={c.Slack:.2f}")

    print("\n===== DEBUG PRESUPUESTO =====")
    print(f"BUDGET declarado en modelo: {getattr(m, '_budget', 'N/D')}")
    bud = next((c for c in m.getConstrs() if c.ConstrName == "BUDGET"), None)
    if bud:
        try:
            rhs = bud.RHS
            slack = bud.Slack
            lhs = rhs - slack
            print(f"LHS (costos totales): {lhs:,.2f}")
            print(f"RHS (presupuesto):    {rhs:,.2f}")
            print(f"Slack:                {slack:,.2f}")
        except Exception as e:
            print(f"⚠️ No se pudo leer restricción: {e}")
    else:
        print("⚠️ No se encontró restricción 'BUDGET'")
    
    # --- Debug: leer precios y costos directamente del modelo ---
    if hasattr(m, "_y_price_var"):
        precio_opt = m._y_price_var.X
        print(f"[DBG] y_price (predicho): {precio_opt:,.2f}")

    if hasattr(m, "_base_price_val"):
        print(f"[DBG] base_price_val (desde modelo): {m._base_price_val:,.2f}")

    if hasattr(m, "_lin_cost_expr"):
        try:
            lin_val = m._lin_cost_expr.getValue()
            print(f"[DBG] lin_cost.getValue(): {lin_val:,.2f}")
        except Exception as e:
            print(f"[DBG] lin_cost.getValue() error: {e}")



    # ===== chequear factibilidad =====
    if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        print("no se encontro solucion factible, status:", m.Status)
        try:
            if m.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                print("Intentando IIS (conjunto mínimo de restricciones en conflicto)...")
                m.computeIIS()
                viols = []
                try:
                    for c in m.getConstrs():
                        if c.getAttr(gp.GRB.Attr.IISConstr):
                            viols.append(c.ConstrName)
                except Exception:
                    pass
                try:
                    for gc in m.getGenConstrs():
                        try:
                            if gc.getAttr(gp.GRB.Attr.IISConstr):
                                viols.append(gc.GenConstrName)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    for v in m.getVars():
                        if v.getAttr(gp.GRB.Attr.IISLB):
                            viols.append(f"LB({v.VarName})")
                        if v.getAttr(gp.GRB.Attr.IISUB):
                            viols.append(f"UB({v.VarName})")
                except Exception:
                    pass
                if viols:
                    print("IIS (restricciones/bounds conflictivos):")
                    for nm in sorted(set(viols)):
                        print(" -", nm)
                try:
                    m.write("model_iis.ilp")
                    m.write("model_full.lp")
                    print("Archivos escritos: model_iis.ilp (IIS) y model_full.lp (modelo).")
                except Exception:
                    pass
        except Exception as e:
            print("Fallo al obtener IIS:", e)
        return

    # ===== leer precios =====
    precio_remodelada = float(m.getVarByName("y_price").X)
    y_log = float(m.getVarByName("y_log").X)  # por si quieres verlo

    # ===== reconstruir costos de remodelación (para reporte) =====
    def _pos(v: float) -> float:
        return v if v > 0 else 0.0

    def _num_base(name: str) -> float:
        try:
            return float(pd.to_numeric(base.row.get(name), errors="coerce"))
        except Exception:
            return 0.0

    base_vals = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
        "Garage Cars": _num_base("Garage Cars"),
    }

    def _to_float_safe(val) -> float:
        try:
            v = float(pd.to_numeric(val, errors="coerce"))
            return 0.0 if pd.isna(v) else v
        except Exception:
            return 0.0

    # decisiones óptimas (x_*)
    opt = {f.name: m.getVarByName(f"x_{f.name}").X for f in MODIFIABLE if m.getVarByName(f"x_{f.name}") is not None}

    cambios_costos = []

    # flags de agregados (para evitar doble conteo en delta de contadores)
    def _bx(nm):
        v = m.getVarByName(f"x_{nm}")
        return (float(v.X) if v is not None else 0.0)

    add_flags = {
        "AddFull":  _bx("AddFull"),
        "AddHalf":  _bx("AddHalf"),
        "AddKitch": _bx("AddKitch"),
        "AddBed":   _bx("AddBed"),
    }

    # costos numéricos mapeados (dorms, baños, garage)
    def costo_var(nombre: str, base_v: float, nuevo_v: float) -> float:
        delta = nuevo_v - base_v
        # Los baños completos adicionales se cobran por AddFull (obra + terminaciones)
        if nombre == "Full Bath":
            return 0.0
        # Si se agregó dormitorio vía AddBed, no cobrar delta de "Bedroom AbvGr"
        if nombre == "Bedroom AbvGr":
            if add_flags.get("AddBed", 0.0) >= 0.5:
                return 0.0
            return _pos(delta) * ct.add_bedroom
        return 0.0


    total_cost_vars = 0.0
    for nombre, base_v in base_vals.items():
        nuevo_v = _to_float_safe(opt.get(nombre, base_v))
        base_vf = _to_float_safe(base_v)
        c = costo_var(nombre, base_vf, nuevo_v)
        if abs(nuevo_v - base_vf) > 1e-9:
            cambios_costos.append((nombre, base_vf, nuevo_v, c))
        total_cost_vars += c

    # ---- Cocina (paquetes) ----
    def _q_to_ord(txt) -> int:
        MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(txt)
        except Exception:
            return MAP.get(str(txt), -1)

    kq_base = _q_to_ord(base.row.get("Kitchen Qual", "TA"))
    dTA = dEX = 0
    kq_new = kq_base
    try:
        v_ta = m.getVarByName("x_delta_KitchenQual_TA")
        v_ex = m.getVarByName("x_delta_KitchenQual_EX")
        if v_ta is not None: dTA = int(round(v_ta.X))
        if v_ex is not None: dEX = int(round(v_ex.X))
    except Exception:
        pass
    try:
        v_kq = m.getVarByName("x_Kitchen Qual")
        if v_kq is not None:
            kq_new = int(round(v_kq.X))
    except Exception:
        if dTA:
            kq_new = max(kq_base, 2)
        if dEX:
            kq_new = max(kq_new, 4)

    kitchen_cost = dTA * ct.kitchenQual_upgrade_TA + dEX * ct.kitchenQual_upgrade_EX
    if (dTA or dEX) and (kq_new != kq_base):
        cambios_costos.append(("Kitchen Qual", kq_base, kq_new, float(kitchen_cost)))

    # ---- Utilities (solo si cambias) ----
    base_util_name = str(base.row.get("Utilities", "ELO"))
    if base_util_name not in UTIL_TO_ORD:
        base_util_name = ORD_TO_UTIL.get(_safe_util_ord(base.row.get("Utilities")), "ELO")

    util_pick = None
    for nm in ["ELO", "NoSeWa", "NoSewr", "AllPub"]:
        v = m.getVarByName(f"util_{nm}")
        if v is not None and v.X > 0.5:
            util_pick = nm
            break

    util_cost_add = 0.0
    if util_pick is not None and util_pick != base_util_name:
        util_cost_add = ct.util_cost(util_pick)
        cambios_costos.append(("Utilities", base_util_name, util_pick, float(util_cost_add)))

    # ---- Roof elegido + costos ----
    style_names = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
    matl_names  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
    style_new = None
    matl_new  = None
    for nm in style_names:
        v = m.getVarByName(f"roof_style_is_{nm}")
        if v is not None and v.X > 0.5:
            style_new = nm
            break
    for nm in matl_names:
        v = m.getVarByName(f"roof_matl_is_{nm}")
        if v is not None and v.X > 0.5:
            matl_new = nm
            break

    style_base = str(base.row.get("Roof Style", "Gable"))
    matl_base  = str(base.row.get("Roof Matl",  "CompShg"))

    roof_area = float(pd.to_numeric(base.row.get("Gr Liv Area"), errors="coerce") or 0.0)
    style_cost = 0.0
    matl_cost  = 0.0
    if style_new and style_new != style_base:
        style_cost = ct.roof_style_cost(style_new)
        cambios_costos.append(("Roof Style", style_base, style_new, style_cost))
    if matl_new and matl_new != matl_base:
        matl_cost = ct.roof_matl_cost(matl_new) * roof_area
        cambios_costos.append(("Roof Matl", matl_base, matl_new, matl_cost))

    # ---- Exterior 1st/2nd ----
    ext_names = ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard",
                 "ImStucc","MetalSd","Plywood","PreCast","Stone","Stucco","VinylSd",
                 "Wd Sdng","WdShngl"]

    def _pick_ext(prefix: str, names: list[str]) -> str | None:
        for nm in names:
            v = m.getVarByName(f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    ex1_new = _pick_ext("ex1_", ext_names)
    ex2_new = _pick_ext("ex2_", ext_names)
    ex1_base = str(base.row.get("Exterior 1st", "VinylSd"))
    ex2_base = str(base.row.get("Exterior 2nd", ex1_base))
    area_ext = ct.exterior_area_proxy(base.row)

    ext_cost_sum = 0.0
    if ex1_new and ex1_new != ex1_base:
        c_demo = (ct.exterior_demo_face1 * area_ext)
        c_mat  = (ct.ext_mat_cost(ex1_new) * area_ext)
        ext_cost_sum += (c_demo + c_mat)
        cambios_costos.append(("Exterior 1st", ex1_base, ex1_new, c_demo + c_mat))
    if ex2_new and ex2_new != ex2_base:
        c_demo = (ct.exterior_demo_face2 * area_ext)
        c_mat  = (ct.ext_mat_cost(ex2_new) * area_ext)
        ext_cost_sum += (c_demo + c_mat)
        cambios_costos.append(("Exterior 2nd", ex2_base, ex2_new, c_demo + c_mat))

    # ---- Exter Qual / Exter Cond (reporte + costo) ----
    def _q_to_ord_report(v):
        MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return MAP.get(str(v), 2)

    exter_qual_cost = 0.0
    exter_cond_cost = 0.0
    try:
        # Exter Qual
        v_exq = m.getVarByName("x_Exter Qual")
        if v_exq is not None:
            exq_new = int(round(v_exq.X))
            exq_base = _q_to_ord_report(base.row.get("Exter Qual", "TA"))
            if exq_new > exq_base:
                exter_qual_cost = (exq_new - exq_base) * ct.exter_qual_upgrade_per_level
                cambios_costos.append(("Exter Qual", exq_base, exq_new, exter_qual_cost))

        # Exter Cond
        v_exc = m.getVarByName("x_Exter Cond")
        if v_exc is not None:
            exc_new = int(round(v_exc.X))
            exc_base = _q_to_ord_report(base.row.get("Exter Cond", "TA"))
            if exc_new > exc_base:
                exter_cond_cost = (exc_new - exc_base) * ct.exter_cond_upgrade_per_level
                cambios_costos.append(("Exter Cond", exc_base, exc_new, exter_cond_cost))
    except Exception:
        pass

    # ---- Electrical ----
    elec_extra = 0.0
    try:
        picked = None
        for nm in ["SBrkr","FuseA","FuseF","FuseP","Mix"]:
            v = m.getVarByName(f"x_elect_is_{nm}")
            if v is not None and v.X > 0.5:
                picked = nm
                break
        base_elec = str(base.row.get("Electrical", "SBrkr"))
        if picked and picked != base_elec:
            elec_extra = ct.electrical_demo_small + ct.electrical_cost(picked)
            cambios_costos.append(("Electrical", base_elec, picked, elec_extra))
    except Exception:
        pass

    # ========== AMPLIACIONES Y AGREGADOS (post-solve) ==========
    ampl_cost_report = 0.0
    agregados_cost_report = 0.0
    try:
        # --- Helpers locales seguros ---
        def _to_float_safe(val):
            import pandas as _pd
            try:
                v = float(_pd.to_numeric(val, errors="coerce"))
                return 0.0 if _pd.isna(v) else v
            except Exception:
                return 0.0

        # ===== Parámetros base =====
        A_Full  = float(getattr(ct, "area_full_bath_std", 40.0))
        A_Half  = float(getattr(ct, "area_half_bath_std", 20.0))
        A_Kitch = float(getattr(ct, "area_kitchen_std",  75.0))
        A_Bed   = float(getattr(ct, "area_bedroom_std",  70.0))

        C_COST  = float(getattr(ct, "construction_cost", 0.0))  # $/ft² genérico

        # ===== AGREGADOS BINARIOS =====
        # Construcción (si la variable binaria está activa)
        agregados = {
            "AddFull":  ("Full Bath",     A_Full,  C_COST),
            "AddHalf":  ("Half Bath",     A_Half,  C_COST),
            "AddKitch": ("Kitchen AbvGr", A_Kitch, C_COST),
            "AddBed":   ("Bedroom AbvGr", A_Bed,   C_COST),
        }
        # Costos extra de terminaciones/equipamiento (si los has definido en CostTables)
        FULL_FIX  = float(getattr(ct, "bath_full_fixture_fixed", 0.0))   # e.g., artefactos/terminaciones
        HALF_FIX  = float(getattr(ct, "bath_half_fixture_fixed", 0.0))
        BED_FIX   = float(getattr(ct, "bedroom_finish_fixed", 0.0))      # opcional
        KITCH_FIX = float(getattr(ct, "add_kitchen_fixed", 0.0))         # fijo adicional a obra

        for key, (nombre, area, costo_unit) in agregados.items():
            var = m.getVarByName(f"x_{key}")
            if var and var.X > 0.5:
                c_obra = float(costo_unit) * float(area)
                c_fix  = 0.0
                if key == "AddFull":
                    c_fix = FULL_FIX
                elif key == "AddHalf":
                    c_fix = HALF_FIX
                elif key == "AddBed":
                    c_fix = BED_FIX
                elif key == "AddKitch":
                    c_fix = KITCH_FIX

                costo_total = c_obra + c_fix
                agregados_cost_report += costo_total
                cambios_costos.append((nombre, "sin", "agregado", costo_total))
                print(f"agregado: {nombre} (+{area:.0f} ft²) → costo {money(costo_total)}")

        # ===== AMPLIACIONES PORCENTUALES =====
        AMPL_COMPONENTES = [
            "Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
            "3Ssn Porch", "Screen Porch", "Pool Area"
        ]
        COSTOS = {
            10: float(getattr(ct, "ampl10_cost", 0.0)),
            20: float(getattr(ct, "ampl20_cost", 0.0)),
            30: float(getattr(ct, "ampl30_cost", 0.0)),
        }

        for comp in AMPL_COMPONENTES:
            base_val = float(pd.to_numeric(base.row.get(comp), errors="coerce") or 0.0)
            if base_val <= 0:
                continue
            for pct in [10, 20, 30]:
                v = m.getVarByName(f"x_z{pct}_{comp.replace(' ', '')}")
                if v and v.X > 0.5:
                    delta = base_val * pct / 100.0
                    costo = COSTOS[pct] * delta
                    ampl_cost_report += costo
                    cambios_costos.append((f"{comp} (+{pct}%)", base_val, base_val + delta, costo))
                    print(f"ampliación: {comp} +{pct}% (+{delta:.1f} ft²) → costo {money(costo)}")
                    break  # una sola por componente
        # ===== AMPLIACIONES DIRECTAS 1st/2nd FLOOR (si subieron pies²) =====
        for flr in ["1st Flr SF", "2nd Flr SF"]:
            base_v = _to_float_safe(base.row.get(flr))
            new_v  = _to_float_safe(opt.get(flr, base_v))

            delta = new_v - base_v
            if delta <= 1e-6:
                continue

            # Descontar la parte explicada por agregados discretos (Add*)
            A_Full  = float(getattr(ct, "area_full_bath_std", 40.0))
            A_Half  = float(getattr(ct, "area_half_bath_std", 20.0))
            A_Kitch = float(getattr(ct, "area_kitchen_std",  75.0))
            A_Bed   = float(getattr(ct, "area_bedroom_std",  70.0))
            C_COST  = float(getattr(ct, "construction_cost", 230.0))

            addfull  = (m.getVarByName("x_AddFull")  or 0)
            addhalf  = (m.getVarByName("x_AddHalf")  or 0)
            addkitch = (m.getVarByName("x_AddKitch") or 0)
            addbed   = (m.getVarByName("x_AddBed")   or 0)

            try:
                addfull  = float(addfull.X)  if hasattr(addfull, "X")  else 0.0
                addhalf  = float(addhalf.X)  if hasattr(addhalf, "X")  else 0.0
                addkitch = float(addkitch.X) if hasattr(addkitch, "X") else 0.0
                addbed   = float(addbed.X)   if hasattr(addbed, "X")   else 0.0
            except Exception:
                pass

            delta_explicado_adds = 0.0
            if flr == "1st Flr SF":
                delta_explicado_adds = (
                    A_Full  * addfull +
                    A_Half  * addhalf +
                    A_Kitch * addkitch +
                    A_Bed   * addbed
                )

            delta_directo = max(0.0, delta - delta_explicado_adds)
            if delta_directo > 1e-6:
                c = C_COST * delta_directo
                ampl_cost_report += c
                cambios_costos.append((flr, base_v, base_v + delta_directo, c))
                print(f"ampliación directa: {flr} +{delta_directo:.1f} ft² → costo {money(c)}")
            else:
                # Todo el aumento del 1st Flr viene de Add*, ya reportados aparte → no cobrar de nuevo
                pass


        # ===== PAQUETE DE COCINA – calidad mínima Gd =====
        # Detecta aumento de cocinas y/o baja calidad y cobra paquete/upgrade
        try:
            # 1) ¿se añadió una cocina?
            k_base = _to_float_safe(base.row.get("Kitchen AbvGr"))
            k_new  = _to_float_safe(opt.get("Kitchen AbvGr", k_base))
            addk_var  = m.getVarByName("x_AddKitch")
            addk_bin  = (addk_var is not None and addk_var.X > 0.5)

            # 2) Lectura de calidad antes/después
            def _kq_to_ord(v, default=2):
                MP = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
                try:
                    iv = int(pd.to_numeric(v, errors="coerce"))
                    return iv if iv in (0,1,2,3,4) else default
                except Exception:
                    return MP.get(str(v).strip(), default)

            kq_base = _kq_to_ord(base.row.get("Kitchen Qual", "TA"), default=2)
            v_kq = m.getVarByName("x_Kitchen Qual")
            kq_new = int(round(v_kq.X)) if v_kq is not None else kq_base

            # 3) Política: mínimo Gd (3) si hay una cocina nueva
            MIN_KQ = 3  # Gd
            def _kitchen_finish_cost(level_txt: str) -> float:
                """
                Costo de terminaciones/paquete de cocina para el nivel destino.
                'Gd' hereda el costo de TA si no hay tabla específica.
                """
                level_txt = str(level_txt).strip()
                if level_txt == "Ex":
                    return float(getattr(ct, "kitchenQual_upgrade_EX", 0.0))
                if level_txt in {"Gd", "TA"}:
                    return float(getattr(ct, "kitchenQual_upgrade_TA", 0.0))
                return 0.0

            
            # 3.a) Si se añadió cocina sin x_AddKitch (fallback), cobra obra+fijo
            if (k_new - k_base) > 0.5 and not addk_bin:
                c_obra = C_COST * A_Kitch
                c_fijo = KITCH_FIX
                c_k = c_obra + c_fijo
                agregados_cost_report += c_k
                cambios_costos.append(("Kitchen AbvGr", k_base, k_new, c_k))
                print(f"agregado (fallback): Kitchen AbvGr +1 → costo {money(c_k)}")

            # 3.b) Si hay cocina nueva o calidad final < Gd, cobra paquete/upgrade hasta Gd
            needs_package = ((k_new - k_base) > 0.5) or (kq_new < MIN_KQ)
            if needs_package:
                target_txt = "Gd"
                # cobra el paquete/terminaciones de cocina para Gd
                c_pkg = _kitchen_finish_cost(target_txt)
                if c_pkg > 0:
                    agregados_cost_report += c_pkg
                    cambios_costos.append(("Kitchen Package", "mínimo Gd", target_txt, c_pkg))
                    print(f"paquete de cocina (mín. {target_txt}) → costo {money(c_pkg)}")

                # además, si kq_new<MIN_KQ explícitamente, cobra upgrade incremental
                if kq_new < MIN_KQ:
                    # si necesitas detallar upgrades por escalones, añade aquí
                    pass

        except Exception as e:
            print(f"(paquete de cocina omitido: {e})")

    except Exception as e:
        print(f"⚠️ error leyendo ampliaciones/agregados: {e}")
# ================== FIN AMPLIACIONES Y AGREGADOS ==========

    # ---- Garage Finish (reporte + costo) ----
    garage_finish_cost_report = 0.0
    try:
        gf_names = ["Fin", "RFn", "Unf", "No aplica"]

        # base: dummies y fallback textual
        def _base_gf(tag: str) -> float:
            # intenta dummies con "No aplica" o "NA"
            for lab in ["No aplica", "NA"]:
                col = f"Garage Finish_{lab if tag=='No aplica' else tag}"
                if col in base.row:
                    return float(pd.to_numeric(base.row.get(col), errors="coerce") or 0.0)
            # fallback a columna textual
            if "Garage Finish" in base.row:
                txt = str(base.row.get("Garage Finish")).strip()
                if txt in {"NA", "No aplica"} and tag == "No aplica": return 1.0
                if txt == tag: return 1.0
            return 0.0

        base_gf = {nm: _base_gf(nm) for nm in gf_names}

        # óptimo: leo las binarias si existen
        sol_gf = {}
        for nm in gf_names:
            v = m.getVarByName(f"x_garage_finish_is_{nm}")
            sol_gf[nm] = float(v.X) if v is not None else 0.0

        v_upg = m.getVarByName("x_UpgGarage")
        upg_val = float(v_upg.X) if v_upg is not None else 1.0  # si no existe, no condiciona

        gf_before = next((k for k, v in base_gf.items() if v == 1.0), "No aplica")
        gf_after = max(sol_gf, key=sol_gf.get) if sol_gf else "No aplica"

        costo_gf = 0.0
        if gf_before != gf_after:
            costo_gf = float(ct.garage_finish_cost(gf_after)) * (1.0 if upg_val >= 0.5 else 0.0)
            cambios_costos.append(("Garage Finish", gf_before, gf_after, costo_gf))
            print(f"Cambio en Garage Finish: {gf_before} -> {gf_after} ({money(costo_gf)})")
        else:
            print(f"Sin cambio en Garage Finish, sigue en {gf_before}")

        garage_finish_cost_report = float(costo_gf)

    except Exception as e:
        print("⚠️ aviso no crítico, Garage Finish:", e)


    # ---- Pool QC (reporte + costo) ----
    pool_qc_cost_report = 0.0
    try:
        pool_area = float(pd.to_numeric(base.row.get("Pool Area"), errors="coerce") or 0.0)

        # mapeos
        ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        INV = {v: k for k, v in ORD.items()}
        CATS = ["Po", "Fa", "TA", "Gd", "Ex", "No aplica"]

        # ---- base (acepta dummy o textual/ordinal) ----
        def _base_pq_cat() -> str:
            # dummies primero
            for tag in ["Po","Fa","TA","Gd","Ex","No aplica","NA"]:
                col = f"Pool QC_{tag}"
                if col in base.row and float(pd.to_numeric(base.row.get(col), errors="coerce") or 0.0) == 1.0:
                    return "No aplica" if tag == "NA" else tag
            # textual/ordinal
            if "Pool QC" in base.row:
                raw = base.row.get("Pool QC")
                try:
                    iv = int(pd.to_numeric(raw, errors="coerce"))
                    return INV.get(iv, "No aplica")
                except Exception:
                    s = str(raw).strip()
                    return "No aplica" if s in {"NA", "No aplica"} else (s if s in ORD else "No aplica")
            return "No aplica"

        pq_before = _base_pq_cat()

        # ---- óptimo (prefiere ordinal x_Pool QC; si no, binarias x_pool_qc_is_*) ----
        v_ord = m.getVarByName("x_Pool QC")
        if v_ord is not None:
            pq_after = INV.get(int(round(v_ord.X)), "No aplica")
        else:
            # busca alguna de las binarias
            pq_after = "No aplica"
            for tag in ["Po","Fa","TA","Gd","Ex","No aplica","NA"]:
                v = m.getVarByName(f"x_pool_qc_is_{tag}")
                if v is not None and v.X > 0.5:
                    pq_after = "No aplica" if tag == "NA" else tag
                    break

        # ---- costo ----
        costo_pq = 0.0
        if pq_after != pq_before:
            # costo por categoría + costo por área (si tu política así lo define)
            cat_cost = float(getattr(ct, "poolqc_costs", {}).get(pq_after, 0.0))
            area_cost = float(getattr(ct, "pool_area_cost", 0.0)) * pool_area
            costo_pq = cat_cost + area_cost
            cambios_costos.append(("Pool QC", pq_before, pq_after, costo_pq))
            print(f"Cambio en calidad de piscina: {pq_before} → {pq_after} ({money(costo_pq)})")
        else:
            print(f"Sin cambio en calidad de piscina (sigue en {pq_before})")

        pool_qc_cost_report = float(costo_pq)

    except Exception as e:
        print(f"⚠️ (aviso, no crítico) error leyendo resultado de Pool QC: {e}")



  
    # ========== GARAGE QUAL / COND (post-solve) ==========
    try:
        G_CATS = ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"]
        garage_qc_cost_report = 0.0

        def _na2noaplica(v):
            s = str(v).strip()
            return "No aplica" if s in {"NA", "No aplica"} else s

        def _qtxt(v):
            """Devuelve etiqueta ('Po'..'Ex' o 'No aplica') aceptando texto o 0..4."""
            M = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}
            try:
                iv = int(pd.to_numeric(v, errors="coerce"))
                if iv in M:
                    return M[iv]
            except Exception:
                pass
            return _na2noaplica(v)

        base_row = base.row
        base_qual = _qtxt(base_row.get("Garage Qual", "No aplica"))
        base_cond = _qtxt(base_row.get("Garage Cond", "No aplica"))

        def _find_selected(prefix: str) -> str:
            for g in G_CATS:
                v = m.getVarByName(f"x_{prefix}_is_{g}")
                if v and v.X > 0.5:
                    return g
            return "No aplica"

        new_qual = _find_selected("garage_qual")
        new_cond = _find_selected("garage_cond")

        def _cost(name: str) -> float:
            return ct.garage_qc_costs.get(name, 0.0)

        cost_qual = _cost(new_qual) if new_qual != base_qual else 0.0
        cost_cond = _cost(new_cond) if new_cond != base_cond else 0.0
        garage_qc_cost_report = cost_qual + cost_cond

        print("\nCambios en GarageQual / GarageCond:")
        print(f"  GarageQual: {base_qual} → {new_qual}  "
            f"({'sin cambio' if cost_qual == 0.0 else f'costo ${cost_qual:,.0f}'})")
        print(f"  GarageCond: {base_cond} → {new_cond}  "
            f"({'sin cambio' if cost_cond == 0.0 else f'costo ${cost_cond:,.0f}'})")

        if base_qual != new_qual:
            cambios_costos.append(("GarageQual", base_qual, new_qual, cost_qual))
        if base_cond != new_cond:
            cambios_costos.append(("GarageCond", base_cond, new_cond, cost_cond))

    except Exception as e:
        print(f"(post-solve garage qual/cond omitido: {e})")
    # ========== FIN GARAGE QUAL / COND ==========

        # ========== PAVED DRIVE (post-solve) ==========

    try:
        PAVED_CATS = ["Y", "P", "N"]

        def _find_selected_pd():
            for d in PAVED_CATS:
                v = m.getVarByName(f"x_paved_drive_is_{d}")
                if v and v.X > 0.5:
                    return d
            return "N"

        base_pd = str(base.row.get("Paved Drive", "N")).strip()
        new_pd = _find_selected_pd()

        cost_pd = ct.paved_drive_cost(new_pd) if new_pd != base_pd else 0.0

        print("\nCambio en PavedDrive:")
        print(f"  {base_pd} → {new_pd}  "
              f"({'sin cambio' if base_pd == new_pd else f'costo ${cost_pd:,.0f}'})")

        if base_pd != new_pd:
            cambios_costos.append(("Paved Drive", base_pd, new_pd, cost_pd))

    except Exception as e:
        print(f"(post-solve paved drive omitido: {e})")

    # ================== FIN PAVED DRIVE ==========

  # ========== FENCE (post-solve) ==========
    fence_cost_report = 0.0   # <--- añade esta línea ANTES del try
    try:
        FENCE_CATS = ["GdPrv", "MnPrv", "GdWo", "MnWw", "No aplica"]

        def _na2noaplica(v):
            s = str(v).strip()
            return "No aplica" if s in {"NA", "No aplica"} else s

        def _find_selected_fence():
            for f in FENCE_CATS:
                v = m.getVarByName(f"x_fence_is_{f}")
                if v and v.X > 0.5:
                    return f
            return "No aplica"

        base_f = _na2noaplica(base.row.get("Fence", "No aplica"))
        new_f = _find_selected_fence()
        lot_front = float(pd.to_numeric(base.row.get("Lot Frontage"), errors="coerce") or 0.0)

        cost_f = 0.0
        if base_f == "No aplica" and new_f in ["MnPrv", "GdPrv"]:
            cost_f = ct.fence_build_cost_per_ft * lot_front
        elif new_f != base_f:
            cost_f = ct.fence_category_cost(new_f)

        fence_cost_report = cost_f  # <--- importante

        print("\nCambio en Fence:")
        print(f"  {base_f} → {new_f}  "
            f"({'sin cambio' if cost_f == 0.0 else f'costo ${cost_f:,.0f}'})")

        if new_f != base_f:
            cambios_costos.append(("Fence", base_f, new_f, cost_f))

    except Exception as e:
        print(f"(post-solve fence omitido: {e})")
    # ================== FIN FENCE ==================



    # ---- Mas Vnr Type ----
    MVT_NAMES = ["BrkCmn", "BrkFace", "CBlock", "Stone", "No aplica", "None"]
    def _pick_mvt_from_model() -> str | None:
        for nm in MVT_NAMES:
            v = m.getVarByName(f"x_mvt_is_{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    mvt_extra = 0.0
    try:
        mvt_pick = _pick_mvt_from_model()
        mvt_base = str(base.row.get("Mas Vnr Type", "No aplica")).strip()

        v_mv_area = m.getVarByName("x_Mas Vnr Area")
        if v_mv_area is not None:
            mv_area = float(v_mv_area.X)
        else:
            try:
                mv_area = float(pd.to_numeric(base.row.get("Mas Vnr Area"), errors="coerce") or 0.0)
            except Exception:
                mv_area = 0.0

        if (mvt_pick is not None) and (mvt_pick != mvt_base):
            mvt_extra = ct.mas_vnr_cost(mvt_pick) * mv_area
            cambios_costos.append(("Mas Vnr Type", mvt_base, mvt_pick, mvt_extra))
    except Exception:
        pass

    # ---- Central Air (reporte + costo) ----
    central_air_cost_add = 0.0
    try:
        base_air = "Y" if str(base.row.get("Central Air", "N")).strip() in {"Y","Yes","1","True"} else "N"
        v_yes = m.getVarByName("central_air_yes")
        if v_yes is not None:
            pick = "Y" if v_yes.X > 0.5 else "N"
            print(f"Central Air (base -> óptimo): {base_air} -> {pick}")
            if pick != base_air:
                central_air_cost_add = float(ct.central_air_install)
                cambios_costos.append(("Central Air", base_air, pick, central_air_cost_add))
    except Exception:
        pass

    # ---- Heating (reporte + costo) ----
    heating_cost_report = 0.0
    try:
        heat_types = ["Floor","GasA","GasW","Grav","OthW","Wall"]
        heat_base = str(base.row.get("Heating","GasA")).strip()

        # tipo final (admite 'x_heat_is_*' o 'heat_is_*')
        heat_new = None
        for nm in heat_types:
            v = _getv(m, f"x_heat_is_{nm}", f"heat_is_{nm}")
            if v is not None and v.X > 0.5:
                heat_new = nm
                break

        # upg_type / change_type (admite con/sin x_)
        v_change = _getv(m, "x_heat_change_type", "heat_change_type")
        v_upg    = _getv(m, "x_heat_upg_type",   "heat_upg_type")

        change_type = int(round(v_change.X)) if v_change is not None else 0
        upg_type    = int(round(v_upg.X))    if v_upg    is not None else 0

        # 1) Reconstruir mismo tipo
        if upg_type == 1 and change_type == 0:
            c_same = ct.heating_type_cost(heat_base)
            heating_cost_report += c_same
            cambios_costos.append(("Heating (reconstruir tipo)", heat_base, heat_base, c_same))

        # 2) Cambiar a tipo más caro
        if heat_new and heat_new != heat_base:
            c_new = ct.heating_type_cost(heat_new)
            heating_cost_report += c_new
            cambios_costos.append(("Heating (tipo)", heat_base, heat_new, c_new))

        # 3) Cambiar calidad
        qc_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
        qc_base_val = _qual_to_ord(base.row.get("Heating QC"), default=2)
        qc_base_txt = qc_map[qc_base_val]
        vq = _getv(m, "x_Heating QC", "x_HeatingQC")
        if vq is not None:
            qc_new_val = int(round(vq.X))
            if qc_new_val != qc_base_val:
                qc_new_txt = qc_map.get(qc_new_val, str(qc_new_val))
                c_qc = ct.heating_qc_cost(qc_new_txt)
                heating_cost_report += c_qc
                cambios_costos.append(("Heating QC", qc_base_txt, qc_new_txt, c_qc))
    except Exception as e:
        print("[HEAT-DEBUG] Error en reporte:", e)

 # ---- BsmtFin: reconstrucción de valores + costo ----
    bsmt_finish_cost_report = 0.0
    try:
        v_fin = m.getVarByName("bsmt_finish")
        v_tr1 = m.getVarByName("bsmt_tr1")
        v_tr2 = m.getVarByName("bsmt_tr2")
        b1_var = m.getVarByName("bsmt_fin1")
        b2_var = m.getVarByName("bsmt_fin2")
        bu_var = m.getVarByName("bsmt_unf")

        if all(v is not None for v in [v_fin, v_tr1, v_tr2, b1_var, b2_var, bu_var]):
            tr1 = float(v_tr1.X)
            tr2 = float(v_tr2.X)

            # bases
            b1b = float(pd.to_numeric(base.row.get("BsmtFin SF 1"), errors="coerce") or 0.0)
            b2b = float(pd.to_numeric(base.row.get("BsmtFin SF 2"), errors="coerce") or 0.0)
            bub = float(pd.to_numeric(base.row.get("Bsmt Unf SF"),   errors="coerce") or 0.0)
            tbb = float(pd.to_numeric(base.row.get("Total Bsmt SF"), errors="coerce") or (b1b+b2b+bub))

            # nuevos desde variables (conservación)
            b1n = float(b1_var.X)
            b2n = float(b2_var.X)
            bun = float(bu_var.X)
            tbn = b1n + b2n + bun

            # costo: solo lo trasladado a “fin”
            added = tr1 + tr2
            if added > 1e-9:
                bsmt_finish_cost_report = ct.finish_basement_per_f2 * added

            # cambios para imprimir (cobrando solo lo que corresponde)
            if abs(b1n - b1b) > 1e-9:
                cambios_costos.append(("BsmtFin SF 1", b1b, b1n, ct.finish_basement_per_f2 * tr1))
            if abs(b2n - b2b) > 1e-9:
                cambios_costos.append(("BsmtFin SF 2", b2b, b2n, ct.finish_basement_per_f2 * tr2))
            if abs(bun - bub) > 1e-9:
                cambios_costos.append(("Bsmt Unf SF",  bub, bun, 0.0))
            if abs(tbn - tbb) > 1e-9:
                cambios_costos.append(("Total Bsmt SF", tbb, tbn, 0.0))
    except Exception as e:
        print("[BSMT-DEBUG] error construyendo reporte:", e)

    # Fallback (si no existían tr1/tr2/vars de conservación): usar solo delta positivo
    if bsmt_finish_cost_report == 0.0:
        try:
            b1_base = float(pd.to_numeric(base.row.get("BsmtFin SF 1"), errors="coerce") or 0.0)
            b2_base = float(pd.to_numeric(base.row.get("BsmtFin SF 2"), errors="coerce") or 0.0)
            b1_new  = _to_float_safe(opt.get("BsmtFin SF 1", b1_base))
            b2_new  = _to_float_safe(opt.get("BsmtFin SF 2", b2_base))
            c_b1 = max(0.0, b1_new - b1_base) * ct.finish_basement_per_f2
            c_b2 = max(0.0, b2_new - b2_base) * ct.finish_basement_per_f2
            if c_b1 > 0:
                cambios_costos.append(("BsmtFin SF 1", b1_base, b1_new, c_b1))
            if c_b2 > 0:
                cambios_costos.append(("BsmtFin SF 2", b2_base, b2_new, c_b2))
            bsmt_finish_cost_report = c_b1 + c_b2
        except Exception:
            pass


    # ---- Bsmt Cond (reporte + costo) ----
    bsmt_cond_cost_report = 0.0
    try:
        v_bc = m.getVarByName("x_Bsmt Cond")
        if v_bc is not None:
            bc_new = int(round(v_bc.X))
            inv_map = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
            def _ord(v):
                M={"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
                try: return int(v)
                except: return M.get(str(v),2)
            bc_base = _ord(base.row.get("Bsmt Cond","TA"))
            if bc_new > bc_base:
                bc_new_txt  = inv_map[bc_new]
                bc_base_txt = inv_map[bc_base]
                # si definiste ct.bsmt_cond_cost(nivel), úsalo; si no, costo por nivel:
                try:
                    c_bc = ct.bsmt_cond_cost(bc_new_txt)
                except Exception:
                    c_bc = (bc_new - bc_base) * 2000.0
                bsmt_cond_cost_report += c_bc
                cambios_costos.append(("Bsmt Cond", bc_base_txt, bc_new_txt, c_bc))
    except Exception:
        pass

    # ---- BsmtFin Type 1/2 (reporte + costo) ----
    bsmt_type_cost_report = 0.0
    TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]

    def _pick_cat(prefix: str) -> str | None:
        # intenta con y sin 'x_' por si difieren los nombres
        for nm in TYPES:
            v = _getv(m, f"x_{prefix}{nm}", f"{prefix}{nm}")
            if v is not None and v.X > 0.5:
                return nm
        return None

    # Base (desde la casa)
    b1_base = str(base.row.get("BsmtFin Type 1", "No aplica")).strip()
    b2_base = str(base.row.get("BsmtFin Type 2", "No aplica")).strip()

    # Nuevo (desde el modelo)
    b1_new = _pick_cat("b1_is_") or b1_base
    b2_new = _pick_cat("b2_is_") or b2_base

    # Costo por tipología (ajusta tu CostTables.bsmt_type_cost)
    def _bsmt_type_cost(nm: str) -> float:
        try:
            return float(ct.bsmt_type_cost(nm))
        except Exception:
            return 0.0

    if b1_new != b1_base:
        c = _bsmt_type_cost(b1_new)
        cambios_costos.append(("BsmtFin Type 1", b1_base, b1_new, c))
        bsmt_type_cost_report += c

    # En Ames, Type 2 puede no existir; cobra solo si la base tenía algo
    if b2_base != "No aplica" and b2_new != b2_base:
        c = _bsmt_type_cost(b2_new)
        cambios_costos.append(("BsmtFin Type 2", b2_base, b2_new, c))
        bsmt_type_cost_report += c

    # reflejar en snapshot SOLO si opt_dict ya existe en ese punto
    if 'opt_dict' in locals():
        opt_dict["BsmtFin Type 1"] = b1_new
        opt_dict["BsmtFin Type 2"] = b2_new

    # ---- Fireplace Qu (reporte + costo) ----
    fp_cost_report = 0.0
    try:
        v_fp = m.getVarByName("x_Fireplace Qu")
        if v_fp is not None:
            new_val = int(round(v_fp.X))
            MAPI = {-1:"No aplica", 0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
            base_val = _qual_to_ord(base.row.get("Fireplace Qu"), default=-1)  # usa tu helper actual
            base_txt = MAPI.get(base_val, "No aplica")
            new_txt  = MAPI.get(new_val,  "No aplica")

            # Política: si base = No aplica, no debería cambiar porque lo fijamos en el MIP
            # Si base >=0 y cambia a otro nivel, cobra costo del nivel nuevo (ajusta si tu tabla es incremental)
            if base_val >= 0 and new_val != base_val:
                c = ct.fireplace_cost(new_txt)  # o ct.fireplace_cost_level(new_txt) según tu tabla
                fp_cost_report += c
                cambios_costos.append(("Fireplace Qu", base_txt, new_txt, c))
    except Exception:
        pass



    # === total final de costos para reporte ===
    total_cost = (
        float(ct.project_fixed)
        + total_cost_vars
        + float(kitchen_cost)
        + float(util_cost_add)
        + float(elec_extra)
        + float(mvt_extra)
        + float(style_cost)
        + float(matl_cost)
        + float(ext_cost_sum)
        + float(central_air_cost_add)
        + float(exter_qual_cost)
        + float(exter_cond_cost)
        + float(heating_cost_report)
        + float(bsmt_cond_cost_report)
        + float(bsmt_finish_cost_report)
        + float(bsmt_type_cost_report) 
        + float(fp_cost_report)
        + float(fence_cost_report)
        + float(garage_qc_cost_report)     
        + float(ampl_cost_report)          
        + float(agregados_cost_report)     
        + float(pool_qc_cost_report)  
        + float(garage_finish_cost_report)

    )

    # ===== métricas =====
    aumento_utilidad = (precio_remodelada - precio_base) - total_cost
# Evitar que el reporte se imprima más de una vez
    if globals().get("_REPORTE_IMPRESO", False):
        return
    globals()["_REPORTE_IMPRESO"] = True


    # --- Inicializar variables si no existen todavía ---
    # --- Inicializar variables si no existen todavía ---
    y_price = locals().get("y_price", None)
    y_base = locals().get("y_base", None)
    lin_cost = locals().get("lin_cost", 0.0)
    obj_val = getattr(m, "ObjVal", None)  
    opt_row = locals().get("opt_row", {})
    cambios_costos = locals().get("cambios_costos", [])
    budget_slack = locals().get("budget_slack", 0.0)



    results = {
        "y_price": y_price,
        "y_base": y_base,
        "lin_cost": lin_cost,
        "cost_report": lin_cost,
        "opt_row": opt_row,
        "cambios_costos": cambios_costos,
        "budget_slack": budget_slack,
    }

    print("\n" + "="*60)
    print("               RESULTADOS DE LA OPTIMIZACIÓN")
    ...
    print("="*60 + "\n")

    # ============================================================
    #              RESULTADOS DE LA OPTIMIZACIÓN
    # ============================================================

    print("\n[DBG] Comparación de costos línea a línea:")
    if hasattr(m, "_lin_cost_expr"):
        for i in range(m._lin_cost_expr.size()):
            v = m._lin_cost_expr.getVar(i)
            c = m._lin_cost_expr.getCoeff(i)
            if abs(c) > 0:
                print(f"   {v.VarName:30s} → {c:,.0f}")

    # --- Lecturas seguras de precios y costos desde el modelo ---
    precio_base = getattr(m, "_base_price_val", None)
    precio_opt_var = getattr(m, "_y_price_var", None)
    precio_opt = precio_opt_var.X if precio_opt_var is not None else None
    total_cost_var = m.getVarByName("cost_model")
    total_cost_model = float(total_cost_var.X) if total_cost_var is not None else 0.0

    # --- Recalcular costo reportado (de la lista de cambios) ---
    total_cost_recalc = sum(c for _, _, _, c in cambios_costos if c is not None)

    # --- Comparar costos modelo vs reporte ---
    print("\n===== DEBUG PRESUPUESTO =====")
    print(f"BUDGET declarado en modelo: {getattr(m, '_budget_usd', 0.0):,.2f}")
    print(f"LHS (costos totales): {total_cost_model:,.2f}")
    print(f"RHS (presupuesto):    {getattr(m, '_budget_usd', 0.0):,.2f}")
    


    if precio_opt is not None:
        print(f"[DBG] y_price (predicho): {precio_opt:,.2f}")
    if precio_base is not None:
        print(f"[DBG] base_price_val (desde modelo): {precio_base:,.2f}")
    if hasattr(m, "_lin_cost_expr"):
        try:
            print(f"[DBG] lin_cost.getValue(): {m._lin_cost_expr.getValue():,.2f}")
        except Exception as e:
            print(f"[DBG] lin_cost.getValue() error: {e}")

    print(f"[DBG] Costo según modelo Gurobi: {total_cost_model:,.0f}")
    print(f"[DBG] Costo según reporte de cambios: {total_cost_recalc:,.0f}")
    if abs(total_cost_model - total_cost_recalc) > 1e-6:
        print("⚠️  Diferencia detectada entre costo modelado y costo reportado.")
        total_cost = total_cost_model
    else:
        total_cost = total_cost_recalc

    # --- Calcular delta precio y utilidad incremental ---
    delta_precio = None
    utilidad_incremental = None
    if precio_base is not None and precio_opt is not None:
        delta_precio = precio_opt - precio_base
        utilidad_incremental = (precio_opt - total_cost_model) - precio_base
        margen = ((precio_opt - precio_base)/precio_opt)*100

    # ============================================================
    #                RESULTADOS IMPRESOS
    # ============================================================

    print("\n" + "="*60)
    print("               RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60)
    print(f"📍 PID: {base.row.get('PID', 'N/A')} – {base.row.get('Neighborhood', 'N/A')} | Presupuesto: ${args.budget:,.0f}")
    print(f"🧮 Modelo: {m.ModelName if hasattr(m, 'ModelName') else 'Gurobi MIP'}")
    print(f"⏱️ Tiempo total: {getattr(m, 'Runtime', 0.0):.2f}s | MIP Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    # ------------------ 1) Resumen Económico ------------------
    print("💰 **Resumen Económico**")
    if precio_base is not None:
        print(f"  Precio casa base:        ${precio_base:,.0f}")
    else:
        print("  Precio casa base:        N/A")
    if precio_opt is not None:
        print(f"  Precio casa remodelada:  ${precio_opt:,.0f}")
    else:
        print("  Precio casa remodelada:  N/A")
    if delta_precio is not None:
        print(f"  Δ Precio:                ${delta_precio:,.0f}")
    else:
        print("  Δ Precio:                N/A")

    print(f"  Costos totales:          ${total_cost:,.0f}")
    if obj_val is not None:
        print(f"  Valor objetivo (MIP):    ${obj_val:,.2f}   (≡ y_price - total_cost)")
    else:
        obj_recalc = (precio_opt or 0) - total_cost
        print(f"  Valor objetivo (MIP):    ${obj_recalc:,.2f}   (recalculado)")
    if utilidad_incremental is not None:
        print(f"ROI:       ${utilidad_incremental:,.0f}   (=(y_price - cost) - y_base)")
    if margen is not None:
        print(f"Porcentaje Neto de Mejoras:       ${margen:,.0f}%   (=(y_price - y_base)/y_price")
    if budget_slack is not None:
        print(f"  Slack presupuesto:       ${budget_slack:,.2f}")

    # ------------------ 2) Diagnóstico ------------------
    print("\n🔍 **Diagnóstico del modelo**")
    changed_bin_vars = []
    for v in m.getVars():
        try:
            if abs(v.LB) <= 1e-9 and abs(v.UB - 1.0) <= 1e-9 and v.X > 0.5:
                changed_bin_vars.append(v.VarName)
        except Exception:
            pass
    print(f"  🔸 Binarias activas: {len(changed_bin_vars)} (ejemplos: {changed_bin_vars[:10]})")
    if budget_slack is None:
        print("  ⚠️  No se encontró o no se pudo leer la restricción 'BUDGET'.")

    # ------------------ 3) Cambios (resumen) ------------------
    print("\n🏠 **Cambios hechos en la casa**")
    if cambios_costos:
        def _is_dup_line(name):
            return ("Full Bath" in name and "agregado" in name.lower())
        for nombre, base_val, new_val, cost_val in cambios_costos:
            if _is_dup_line(nombre):
                continue
            suf = f" (costo ${cost_val:,.0f})" if (cost_val is not None and cost_val > 0) else ""
            print(f"  - {nombre}: {base_val} → {new_val}{suf}")
    else:
        print("  (No se detectaron cambios)")

    # ------------------ 4) Snapshot completo Base vs Óptimo ------------------
    print("\n🧾 **Snapshot: atributos Base vs Óptimo (completo)**")
    try:
        base_dict = dict(base.row.items())
        opt_dict = dict(base.row.items())
        if 'opt' in locals() and isinstance(opt, dict):
            opt_dict.update(opt)

        # Central Air
        v_yes = m.getVarByName("central_air_yes")
        if v_yes is not None:
            opt_dict["Central Air"] = "Y" if v_yes.X > 0.5 else "N"
        # Paved Drive
        sel_pd = None
        for d in ["Y", "P", "N"]:
            v = m.getVarByName(f"paved_drive_is_{d}")
            if v is not None and v.X > 0.5:
                sel_pd = d
                break
        if sel_pd:
            opt_dict["Paved Drive"] = sel_pd

        # Comparativo DataFrame
        keys = sorted(set(base_dict.keys()) | set(opt_dict.keys()))
        rows = []
        for k in keys:
            b = base_dict.get(k, "")
            n = opt_dict.get(k, "")
            def _fmt(v):
                try:
                    fv = float(pd.to_numeric(v, errors="coerce"))
                    if pd.isna(fv): return str(v)
                    return f"{fv:,.0f}"
                except Exception:
                    return str(v)
            rows.append((k, _fmt(b), _fmt(n)))
        df_snapshot = pd.DataFrame(rows, columns=["Atributo", "Base", "Óptimo"])
        pd.set_option("display.max_rows", 9999)
        print(df_snapshot.to_string(index=False))
    except Exception as e:
        print(f"⚠️  Error al generar snapshot: {e}")

    # ------------------ 5) Métricas del optimizador ------------------
    print("\n📈 **Métricas del optimizador**")
    try:
        tiempo_total = getattr(m, "Runtime", 0.0)
        gap_final = getattr(m, "MIPGap", None)
        print(f"  ⏱️  Tiempo total: {tiempo_total:,.2f} segundos ({tiempo_total/60:.2f} min)")
        if gap_final is not None and gap_final < gp.GRB.INFINITY:
            print(f"  📉 MIP Gap final: {gap_final*100:.2f}%")
        else:
            print("  📉 MIP Gap final: N/D")
    except Exception as e:
        print(f"  ⚠️  No se pudieron calcular métricas: {e}")

    print("\n" + "="*60)
    print("            FIN RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
