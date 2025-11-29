#!/usr/bin/env python3
"""Quick ROI check - test 2 properties"""

import subprocess

pids = [526351010, 528344070]

print("=" * 70)
print("QUICK ROI CHECK WITH ALL FIXES")
print("=" * 70)

for pid in pids:
    print(f"\n[Testing PID {pid}]")
    
    cmd = [
        ".venv311/Scripts/python.exe",
        "-m", "optimization.remodel.run_opt",
        "--pid", str(pid),
        "--budget", "500000"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=120)
    output = result.stdout
    
    # Extract ROI
    for line in output.split('\n'):
        if 'ROI %:' in line:
            print(f"  {line.strip()}")
        if 'Costos totales' in line:
            print(f"  {line.strip()}")
        if 'y_base' in line and '=' in line:
            print(f"  {line.strip()}")
        if 'y_price (MIP)' in line:
            print(f"  {line.strip()}")

print("\nDone!")
