#!/usr/bin/env python3
"""Test path-based tree embedding on base case and mismatch case."""

import sys
from optimization.remodel.run_opt import run_opt

# Test base case
print("=" * 70)
print("TEST 1: Base case (526351010, budget $500k)")
print("=" * 70)
try:
    result = run_opt('526351010', 500000, verbose=True)
    print(f"\nResult:")
    print(f"  y_log_raw: {result.get('y_log_raw', 'N/A')}")
    print(f"  y_log: {result.get('y_log', 'N/A')}")
    print(f"  Expected y_log_raw: -0.048855")
    if 'y_log_raw' in result:
        err = abs(result['y_log_raw'] - (-0.048855))
        print(f"  Error: {err:.6f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST 2: Mismatch case (528328100, budget $500k)")
print("=" * 70)
try:
    result = run_opt('528328100', 500000, verbose=True)
    print(f"\nResult:")
    print(f"  y_log_raw: {result.get('y_log_raw', 'N/A')}")
    print(f"  y_log: {result.get('y_log', 'N/A')}")
    print(f"  Expected y_log_raw: ~13.238561 (external) or close")
    if 'y_log_raw' in result:
        external = 13.238561
        err = abs(result['y_log_raw'] - external)
        print(f"  Error vs external: {err:.6f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
