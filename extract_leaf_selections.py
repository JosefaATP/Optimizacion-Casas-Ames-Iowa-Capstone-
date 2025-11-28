#!/usr/bin/env python3
"""
Extract which tree leaves were actually selected by Gurobi in the last MIP run.
"""
import sys
sys.path.insert(0, 'optimization/remodel')

# Read the last MIP solution and extract leaf selections
# This requires reading Gurobi's output or the saved solution

import subprocess
import re

# Run the optimization again with debug output
result = subprocess.run([
    sys.executable, "-m", "optimization.remodel.run_opt",
    "--pid", "526351010",
    "--budget", "500000",
    "--time-limit", "5"
], capture_output=True, text=True)

output = result.stdout + result.stderr

# Extract the tree selection audit output
print("=== TREE SELECTION AUDIT FROM LAST RUN ===\n")

# Look for lines with tree selection
for line in output.split('\n'):
    if 'Tree' in line and ('selected leaf' in line or 'y_log_raw' in line or 'Difference' in line):
        print(line)

print("\n=== KEY METRICS ===\n")

# Extract key debug metrics
for line in output.split('\n'):
    if '[DEBUG]' in line or '[CALIB]' in line or '[WARNING]' in line:
        print(line)
