#!/usr/bin/env python3
"""
Check which area expansion variables (z10_*, z20_*, z30_*) are active in the MIP solution.
"""

import subprocess
import json
import re

pid = 528344070
budget = 500000

# Run optimization and capture full output
cmd = [
    ".venv311/Scripts/python.exe",
    "-m", "optimization.remodel.run_opt",
    "--pid", str(pid),
    "--budget", str(budget)
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

# Look for area expansion variables in audit section
output = result.stdout

print("=" * 70)
print("AREA EXPANSION VARIABLES CHECK")
print("=" * 70)

# Find the audit section
if "[AUDIT] Vars inyectadas" in output:
    audit_start = output.find("[AUDIT] Vars inyectadas")
    audit_section = output[audit_start:audit_start+10000]
    
    # Look for z-variables
    z_vars = {}
    for line in audit_section.split('\n'):
        if 'z10_' in line or 'z20_' in line or 'z30_' in line:
            print(line)
            
            # Extract variable name and value
            if 'z10_' in line:
                match = re.search(r'(z10_\w+)\s+([\d.]+)', line)
                if match:
                    z_vars[match.group(1)] = match.group(2)
            elif 'z20_' in line:
                match = re.search(r'(z20_\w+)\s+([\d.]+)', line)
                if match:
                    z_vars[match.group(1)] = match.group(2)
            elif 'z30_' in line:
                match = re.search(r'(z30_\w+)\s+([\d.]+)', line)
                if match:
                    z_vars[match.group(1)] = match.group(2)

print("\n" + "=" * 70)
print("Z-VARIABLES ACTIVE IN SOLUTION:")
print("=" * 70)

if z_vars:
    for var, val in sorted(z_vars.items()):
        if float(val) > 0.5:  # Binary, so > 0.5 means activated
            print(f"✓ {var} = {val}")
else:
    print("No z-variables found in audit output")
    print("\nSearching for expansion keywords...")
    
    # Search for expansion-related lines
    for line in output.split('\n'):
        if ('Porch SF' in line or 'Wood Deck' in line or 'Garage Area' in line or 
            'Pool Area' in line) and ('→' in line or '->' in line or 'base=' in line):
            print(line.strip())

print("\nExpansion costs found:")
for line in output.split('\n'):
    if 'costo $' in line and ('Porch' in line or 'Wood Deck' in line or 'Garage' in line or 'Pool' in line):
        print("  " + line.strip())
