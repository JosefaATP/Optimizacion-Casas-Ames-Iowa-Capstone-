"""
Test si el y_log mismatch es consistente en múltiples propiedades
Usa directamente run_opt para cada propiedad
"""
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Disable Gurobi output
import gurobipy as gp

from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.run_opt import check_predict_consistency, run_optimization_for_pid

# Load PIDs to test
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
np.random.seed(42)
pids = np.random.choice(df_all['PID'].unique(), size=min(20, len(df_all)), replace=False)

results = []

for i, pid in enumerate(pids):
    try:
        print(f"[{i+1:2d}/20] PID {pid}...", end=" ")
        sys.stdout.flush()
        
        # Redirect stdout to capture output
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            m = run_optimization_for_pid(
                pid=int(pid),
                budget=100000,
                base_csv='data/processed/base_completa_sin_nulos.csv',
                verbose=False,
                debug=False
            )
            
            # Capture any prediction consistency output
            captured = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            if m is None:
                print("✗ No model")
                continue
            
            # Extract y_log values
            v_log = m.getVarByName("y_log")
            v_log_raw = m.getVarByName("y_log_raw")
            
            if v_log is None:
                print("✗ No y_log var")
                continue
            
            y_log_mip = float(v_log.X)
            y_log_raw_mip = float(v_log_raw.X) if v_log_raw else np.nan
            
            # Try to get external prediction
            bundle = XGBBundle()
            Z = getattr(m, '_X_input', None)
            if Z is None or Z.empty:
                print("✗ No solution X")
                continue
            
            try:
                y_log_raw_ext = float(bundle.predict_log_raw(Z).iloc[0])
                delta = y_log_mip - y_log_raw_ext
                pct = abs(delta) / abs(y_log_mip) * 100 if y_log_mip != 0 else 0
                
                results.append({
                    'PID': int(pid),
                    'y_log_ext': y_log_raw_ext,
                    'y_log_mip': y_log_mip,
                    'y_log_raw_mip': y_log_raw_mip,
                    'delta': delta,
                    'pct_diff': pct
                })
                print(f"Δ={delta:+.6f} ({pct:.2f}%)")
            except Exception as e:
                print(f"✗ Pred error: {str(e)[:30]}")
        except Exception as e:
            sys.stdout = old_stdout
            print(f"✗ {str(e)[:40]}")
            
    except Exception as e:
        print(f"✗ Error: {str(e)[:40]}")

# Print summary
if results:
    df_results = pd.DataFrame(results)
    print("\n" + "="*90)
    print("SUMMARY - y_log MISMATCH ACROSS 20 PROPERTIES")
    print("="*90)
    for _, row in df_results.iterrows():
        print(f"  PID {row['PID']:12.0f} | ext={row['y_log_ext']:10.6f} | mip={row['y_log_mip']:10.6f} | Δ={row['delta']:+.6f} ({row['pct_diff']:5.2f}%)")
    
    print("\n" + "-"*90)
    print(f"Mean delta:    {df_results['delta'].mean():+.6f}")
    print(f"Std delta:     {df_results['delta'].std():.6f}")
    print(f"Mean pct:      {df_results['pct_diff'].mean():.2f}%")
    print(f"Min delta:     {df_results['delta'].min():+.6f}")
    print(f"Max delta:     {df_results['delta'].max():+.6f}")
    print(f"Variation:     {df_results['delta'].max() - df_results['delta'].min():.6f}")
    
    # Check if consistent
    if df_results['delta'].std() < 0.01:
        print("\n✓ MISMATCH IS HIGHLY CONSISTENT (std < 0.01)")
        print(f"  Can be corrected by adding constant offset: {df_results['delta'].mean():.6f}")
    elif df_results['delta'].std() < 0.05:
        print("\n⚠ MISMATCH IS SOMEWHAT CONSISTENT (std < 0.05)")
        print(f"  Might be correctable, but with some variance")
    else:
        print("\n✗ MISMATCH VARIES SIGNIFICANTLY (std > 0.05)")
        print(f"  Likely a deeper issue, not a simple offset")
else:
    print("\n✗ No valid results collected")
