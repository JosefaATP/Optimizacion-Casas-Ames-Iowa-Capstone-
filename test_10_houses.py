#!/usr/bin/env python3
"""
Test the 3 critical fixes with 10 random houses.
Goals:
1. Verify area expansions are now working (z-variables)
2. Check ROI changes with the path exclusion constraints
3. See if any house recommends area expansions
"""

import subprocess
import time
import json

# PIDs to test (manually selected diverse houses)
test_pids = [
    526351010,  # Previous test case
    526353030,  # Random selection
    526355080,
    526356060,
    526360070,
    526365090,
    526370010,
    526375050,
    526380020,
    526385100,
]

budget = 500000
results = {}

print("=" * 70)
print(f"Testing 10 houses with area expansion fixes")
print(f"Budget: ${budget:,}")
print("=" * 70)

for idx, pid in enumerate(test_pids, 1):
    print(f"\n[{idx}/10] Testing PID {pid}...")
    
    try:
        # Run optimization
        cmd = [
            "python", "-m", "optimization.remodel.run_opt",
            "--pid", str(pid),
            "--budget", str(budget)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per house
        )
        
        if result.returncode == 0:
            # Parse output for ROI and expansion info
            output = result.stdout
            
            # Extract key metrics
            if "ROI:" in output:
                roi_line = [line for line in output.split('\n') if 'ROI:' in line][0]
                print(f"  OK {roi_line.strip()}")
                
                # Check for expansions
                if "z10_" in output or "z20_" in output or "z30_" in output:
                    print(f"  BUILD Area expansions detected!")
                    results[pid] = "SUCCESS_WITH_EXPANSIONS"
                else:
                    results[pid] = "SUCCESS_NO_EXPANSIONS"
            else:
                print(f"  OK Optimization completed (ROI not in stdout)")
                results[pid] = "SUCCESS_NO_EXPANSIONS"
        else:
            err_msg = result.stderr[:100] if result.stderr else "No stderr"
            print(f"  ERROR {err_msg}")
            results[pid] = "ERROR"
    
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>2 min)")
        results[pid] = "TIMEOUT"
    except Exception as e:
        print(f"  ERROR {str(e)[:100]}")
        results[pid] = "ERROR"
    
    time.sleep(1)  # Small delay between runs

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

expansion_count = sum(1 for v in results.values() if "EXPANSIONS" in v)
success_count = sum(1 for v in results.values() if "SUCCESS" in v)

print(f"SUCCESS: {success_count}/{len(test_pids)}")
print(f"WITH AREA EXPANSIONS: {expansion_count}/{len(test_pids)}")

for pid, status in results.items():
    print(f"  PID {pid}: {status}")

print("\nNext: Review the house with expansions in detail")
