"""
Test mismatch consistency by running optimization 20 times
"""
import subprocess
import re
import pandas as pd
import numpy as np

df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
np.random.seed(42)
pids = np.random.choice(df_all['PID'].unique(), size=min(20, len(df_all)), replace=False)

results = []

for i, pid in enumerate(pids):
    pid_int = int(pid)
    print(f"[{i+1:2d}/20] PID {pid_int}...", end=" ", flush=True)
    
    # Run optimization
    cmd = f'.venv311\\Scripts\\python.exe -m optimization.remodel.run_opt --pid {pid_int} --budget 100000'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        
        # Extract y_log values from output
        pred_match = re.search(r'\[PRED\] predict_log_raw\(X_sol\) = ([\d.+-]+).*y_log \(MIP\) = ([\d.+-]+).*Δ = ([\d.+-]+)', output)
        
        if pred_match:
            y_log_ext = float(pred_match.group(1))
            y_log_mip = float(pred_match.group(2))
            delta = float(pred_match.group(3))
            pct = abs(delta) / abs(y_log_mip) * 100 if y_log_mip != 0 else 0
            
            results.append({
                'PID': pid_int,
                'y_log_ext': y_log_ext,
                'y_log_mip': y_log_mip,
                'delta': delta,
                'pct_diff': pct
            })
            print(f"✓ Δ={delta:+.6f} ({pct:.2f}%)")
        else:
            print(f"✗ No match")
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout")
    except Exception as e:
        print(f"✗ {str(e)[:30]}")

# Print summary
if results:
    df_results = pd.DataFrame(results)
    print("\n" + "="*100)
    print("SUMMARY - y_log MISMATCH ACROSS 20 PROPERTIES")
    print("="*100)
    for _, row in df_results.iterrows():
        print(f"  PID {row['PID']:12.0f} | ext={row['y_log_ext']:10.6f} | mip={row['y_log_mip']:10.6f} | Δ={row['delta']:+.6f} ({row['pct_diff']:5.2f}%)")
    
    print("\n" + "-"*100)
    print(f"Mean delta:    {df_results['delta'].mean():+.6f}")
    print(f"Std delta:     {df_results['delta'].std():.6f}")
    print(f"Mean pct:      {df_results['pct_diff'].mean():.2f}%")
    print(f"Min delta:     {df_results['delta'].min():+.6f}")
    print(f"Max delta:     {df_results['delta'].max():+.6f}")
    print(f"Variation:     {df_results['delta'].max() - df_results['delta'].min():.6f}")
    
    # Check if consistent
    variation = df_results['delta'].max() - df_results['delta'].min()
    if df_results['delta'].std() < 0.001:
        print(f"\n✓✓✓ MISMATCH IS EXTREMELY CONSISTENT (std={df_results['delta'].std():.6f})")
        print(f"    Can be EASILY corrected with constant offset: {df_results['delta'].mean():.6f}")
    elif df_results['delta'].std() < 0.01:
        print(f"\n✓ MISMATCH IS CONSISTENT (std={df_results['delta'].std():.6f})")
        print(f"  Can be corrected with constant offset: {df_results['delta'].mean():.6f}")
    elif df_results['delta'].std() < 0.05:
        print(f"\n⚠ MISMATCH IS SOMEWHAT CONSISTENT (std={df_results['delta'].std():.6f})")
        print(f"  Might be correctable with small variance")
    else:
        print(f"\n✗ MISMATCH VARIES SIGNIFICANTLY (std={df_results['delta'].std():.6f})")
        print(f"  Likely a deeper issue")
else:
    print("\n✗ No valid results")
