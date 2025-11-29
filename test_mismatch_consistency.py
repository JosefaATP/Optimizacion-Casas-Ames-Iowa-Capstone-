"""
Test si el y_log mismatch es consistente en múltiples propiedades
"""
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.run_opt import get_base_house
from optimization.remodel import costs
import pandas as pd
import numpy as np
import gurobipy as gp
import warnings
warnings.filterwarnings('ignore')

bundle = XGBBundle()
ct = costs.CostTables()

# Load base data
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')

# Get 20 random properties
np.random.seed(42)
pids = np.random.choice(df_all['PID'].unique(), size=min(20, len(df_all)), replace=False)

results = []

for i, pid in enumerate(pids):
    try:
        print(f"\n[{i+1}/20] Testing PID {pid}...")
        
        # Get base house
        base_x = get_base_house(pid, base_csv='data/processed/base_completa_sin_nulos.csv')
        if base_x is None:
            print(f"  ✗ Could not get base_x")
            continue
        
        # Build model
        m = build_mip_embed(
            base_row=base_x.iloc[0],
            budget=100000,
            ct=ct,
            bundle=bundle,
            verbose=0,
        )
        
        # Optimize silently
        m.Params.OutputFlag = 0
        m.optimize()
        
        if m.status != gp.GRB.OPTIMAL:
            print(f"  ✗ MIP status: {m.status}")
            continue
        
        # Extract solution
        Z = base_x.copy()
        for c in Z.columns:
            v = Z.iloc[0][c]
            if hasattr(v, "X"):
                Z.iloc[0, Z.columns.get_loc(c)] = float(v.X)
        
        # Get predictions
        try:
            y_log_raw_ext = float(bundle.predict_log_raw(Z).iloc[0])
        except:
            y_log_raw_ext = np.nan
        
        try:
            v_log = m.getVarByName("y_log")
            y_log_mip = float(v_log.X) if v_log is not None else np.nan
        except:
            y_log_mip = np.nan
        
        if not np.isnan(y_log_raw_ext) and not np.isnan(y_log_mip):
            delta = y_log_mip - y_log_raw_ext
            pct_diff = abs(delta) / abs(y_log_mip) * 100 if y_log_mip != 0 else 0
            results.append({
                'PID': pid,
                'y_log_ext': y_log_raw_ext,
                'y_log_mip': y_log_mip,
                'delta': delta,
                'pct_diff': pct_diff
            })
            print(f"  ✓ ext={y_log_raw_ext:.6f}  mip={y_log_mip:.6f}  Δ={delta:+.6f} ({pct_diff:.2f}%)")
        else:
            print(f"  ✗ NaN values")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

if results:
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SUMMARY - y_log MISMATCH ACROSS 20 PROPERTIES")
    print("="*80)
    print(df_results.to_string(index=False))
    print(f"\nMean delta: {df_results['delta'].mean():.6f}")
    print(f"Std delta:  {df_results['delta'].std():.6f}")
    print(f"Mean pct:   {df_results['pct_diff'].mean():.2f}%")
    print(f"Min delta:  {df_results['delta'].min():.6f}")
    print(f"Max delta:  {df_results['delta'].max():.6f}")
    
    # Check if consistent
    if df_results['delta'].std() < 0.01:
        print("\n✓ MISMATCH IS HIGHLY CONSISTENT (can be manually corrected)")
    else:
        print("\n✗ MISMATCH VARIES (likely a deeper issue)")
