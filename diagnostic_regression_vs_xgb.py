#!/usr/bin/env python3
"""
Diagnóstico actualizado: comparar Regresión re-procesada vs XGBoost
usando exactamente los mismos features de run_opt (build_base_input_row).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--pid", type=int, default=526301100, help="PID a diagnosticar")
ap.add_argument("--opt-csv", type=str, default="X_input_after_opt.csv",
                help="CSV con features de la casa optimizada (salida run_opt)")
args = ap.parse_args()

print("\n" + "="*70)
print("  DIAGNÓSTICO: REGRESIÓN (reprocesada) vs XGBoost")
print("="*70 + "\n")

# ============================================================================
# CARGAR MODELOS
# ============================================================================
try:
    from optimization.remodel.regression_predictor import RegressionPredictor
    reg = RegressionPredictor()  # usa models/regression_model_reprocesed.pkl
    print(f"✓ Regresión cargada ({len(reg.feature_names)} features)")
except Exception as e:
    print(f"❌ Error cargando regresión reprocesada: {e}")
    sys.exit(1)

try:
    from optimization.remodel.xgb_predictor import XGBBundle
    from optimization.remodel.gurobi_model import build_base_input_row
    bundle = XGBBundle()
    print("✓ XGBoost cargado")
except Exception as e:
    print(f"❌ Error cargando XGBoost/bundle: {e}")
    sys.exit(1)

# ============================================================================
# CARGAR DATOS DE TRAINING
# ============================================================================
df = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"✓ Datos cargados: {df.shape}")

pid = args.pid
if pid not in df["PID"].values:
    print(f"❌ PID {pid} no encontrado en datos")
    sys.exit(1)

row = df[df["PID"] == pid].iloc[0]
print(f"✓ PID {pid} encontrado")
print(f"  - Precio real: ${row['SalePrice_Present']:,.0f}")

# ============================================================================
# PROCESAR CON EL MISMO PIPELINE DE run_opt
# ============================================================================
X_row = build_base_input_row(bundle, row)

# Predicción con regresión reprocesada (alineada automáticamente)
pred_reg = reg.predict(X_row)
err_reg = (pred_reg - row["SalePrice_Present"]) / row["SalePrice_Present"] * 100

# Predicción con XGBoost (usando las mismas columnas numéricas one-hot)
X_row_aligned = X_row[bundle.feature_names_in()]
pred_xgb = float(bundle.reg.predict(X_row_aligned)[0])
if bundle.is_log_target():
    pred_xgb = float(np.exp(pred_xgb))
err_xgb = (pred_xgb - row["SalePrice_Present"]) / row["SalePrice_Present"] * 100

# Diferencia entre modelos sobre la misma casa base
diff_abs = pred_xgb - pred_reg
diff_pct = (diff_abs / pred_reg * 100) if pred_reg != 0 else float("nan")

# ============================================================================
# MOSTRAR RESULTADOS
# ============================================================================
print("\n📊 RESULTADOS")
print(f"  Precio real base:        ${row['SalePrice_Present']:,.0f}")
print(f"  Regresión reprocesada: ${pred_reg:,.0f} ({err_reg:+.2f}%)")
print(f"  XGBoost:               ${pred_xgb:,.0f} ({err_xgb:+.2f}%)")
print(f"  Diferencia absoluta:   ${abs(diff_abs):,.0f} ({diff_pct:+.2f}% vs Regresión)")

# ============================================================================
# SI VIENE CSV CON X_opt (CASA REMODELADA), COMPARAR TAMBIÉN
# ============================================================================
opt_path = Path(args.opt_csv)
if opt_path.exists():
    try:
        raw_opt = pd.read_csv(opt_path)
        X_opt = raw_opt.pivot_table(index='idx', columns='feature', values='value').fillna(0).iloc[[0]]
        
        # Regresión (alinear columnas)
        X_reg_opt = pd.DataFrame(columns=reg.feature_names)
        for c in reg.feature_names:
            if c in X_opt.columns:
                X_reg_opt[c] = [float(pd.to_numeric(X_opt[c].iloc[0], errors="coerce") or 0.0)]
            else:
                X_reg_opt[c] = [0.0]
        pred_log_reg_opt = reg.model.predict(X_reg_opt.values)[0]
        pred_reg_opt = float(np.exp(pred_log_reg_opt))
        
        # XGBoost (alinear columnas)
        X_xgb_opt = pd.DataFrame(columns=bundle.feature_names_in())
        for c in bundle.feature_names_in():
            if c in X_opt.columns:
                X_xgb_opt[c] = [float(pd.to_numeric(X_opt[c].iloc[0], errors="coerce") or 0.0)]
            else:
                X_xgb_opt[c] = [0.0]
        from optimization.remodel.xgb_predictor import _coerce_quality_ordinals_inplace, _coerce_utilities_ordinal_inplace
        X_xgb_opt = _coerce_quality_ordinals_inplace(X_xgb_opt, bundle.quality_cols)
        X_xgb_opt = _coerce_utilities_ordinal_inplace(X_xgb_opt)
        pred_xgb_opt = float(bundle.reg.predict(X_xgb_opt)[0])
        if bundle.is_log_target():
            pred_xgb_opt = float(np.exp(pred_xgb_opt))
        
        # Métricas remodelada
        err_reg_opt = (pred_reg_opt - row["SalePrice_Present"]) / row["SalePrice_Present"] * 100
        err_xgb_opt = (pred_xgb_opt - row["SalePrice_Present"]) / row["SalePrice_Present"] * 100
        diff_abs_opt = pred_xgb_opt - pred_reg_opt
        diff_pct_opt = (diff_abs_opt / pred_reg_opt * 100) if pred_reg_opt != 0 else float("nan")
        
        print("\n🏠 RESULTADOS CASA REMODELADA (desde X_input_after_opt.csv)")
        print(f"  Regresión reprocesada: ${pred_reg_opt:,.0f} ({err_reg_opt:+.2f}%)")
        print(f"  XGBoost:               ${pred_xgb_opt:,.0f} ({err_xgb_opt:+.2f}%)")
        print(f"  Diferencia absoluta:   ${abs(diff_abs_opt):,.0f} ({diff_pct_opt:+.2f}% vs Regresión)")
    except Exception as e:
        print(f"\n⚠️  No se pudo calcular la casa remodelada desde {opt_path}: {e}")
else:
    print(f"\nℹ️  No se encontró {opt_path}; solo se muestra la casa base.")

print("\n" + "="*70 + "\n")
