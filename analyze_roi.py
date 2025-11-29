#!/usr/bin/env python3
"""
Analyze ROI variability across multiple properties.
Goal: Understand why ROI ranges from 56% to 215%
"""

import subprocess
import re

test_pids = [
    526351010,  # Previous: 215% ROI
    528344070,  # Previous: 56% ROI
    526353030,  # Previous: 56% ROI
    526355080,  # Previous: 160% ROI
    526356060,
]

print("=" * 80)
print("ROI ANALYSIS: Testing 5 properties with all fixes deployed")
print("=" * 80)

results = []

for idx, pid in enumerate(test_pids, 1):
    print(f"\n[{idx}/5] PID {pid}...")
    
    cmd = [
        ".venv311/Scripts/python.exe",
        "-m", "optimization.remodel.run_opt",
        "--pid", str(pid),
        "--budget", "500000"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=120)
        output = result.stdout
        
        # Extract key metrics
        roi_match = re.search(r"ROI %:\s+([\d.]+)%", output)
        roi_val = float(roi_match.group(1)) if roi_match else None
        
        cost_match = re.search(r"lin_cost\(eval LinExpr\)\s*=\s*([\d,]+\.?\d*)", output)
        cost_val = float(cost_match.group(1).replace(",", "")) if cost_match else None
        
        base_price_match = re.search(r"y_base\s*=\s*([\d,]+\.?\d*)", output)
        base_price = float(base_price_match.group(1).replace(",", "")) if base_price_match else None
        
        final_price_match = re.search(r"y_price \(MIP\)\s*=\s*([\d,]+\.?\d*)", output)
        final_price = float(final_price_match.group(1).replace(",", "")) if final_price_match else None
        
        uplift_match = re.search(r"Δ Precio:\s*\$([\d,]+\.?\d*)", output)
        uplift = float(uplift_match.group(1).replace(",", "")) if uplift_match else None
        
        # Extract expansion info
        expansions = []
        for line in output.split('\n'):
            if '+10%' in line or '+20%' in line or '+30%' in line:
                if 'Porch' in line or 'Wood Deck' in line or 'Garage' in line or 'Pool' in line:
                    expansions.append(line.strip())
        
        results.append({
            'pid': pid,
            'roi': roi_val,
            'cost': cost_val,
            'base_price': base_price,
            'final_price': final_price,
            'uplift': uplift,
            'expansions': len(expansions),
            'success': True
        })
        
        print(f"  ROI: {roi_val}% | Cost: ${cost_val:,.0f} | Uplift: ${uplift:,.0f}")
        if expansions:
            print(f"  Expansions: {len(expansions)}")
        
    except Exception as e:
        print(f"  ERROR: {str(e)[:100]}")
        results.append({
            'pid': pid,
            'success': False,
            'error': str(e)[:100]
        })

# Summary analysis
print("\n" + "=" * 80)
print("SUMMARY ANALYSIS")
print("=" * 80)

successful = [r for r in results if r['success']]

if successful:
    print("\nROI Distribution:")
    for r in sorted(successful, key=lambda x: x['roi'], reverse=True):
        exp_str = f" ({r['expansions']} expansions)" if r['expansions'] > 0 else ""
        print(f"  PID {r['pid']}: {r['roi']}% ROI | Cost: ${r['cost']:,.0f}{exp_str}")
    
    roi_values = [r['roi'] for r in successful]
    print(f"\nAverage ROI: {sum(roi_values)/len(roi_values):.1f}%")
    print(f"Min ROI: {min(roi_values):.1f}%")
    print(f"Max ROI: {max(roi_values):.1f}%")
    
    # Analyze cost vs ROI correlation
    print("\n\nCost-ROI Correlation Analysis:")
    print("Higher cost → Lower ROI? (Expected)")
    for r in sorted(successful, key=lambda x: x['cost']):
        print(f"  Cost: ${r['cost']:>9,.0f} → ROI: {r['roi']:>6.1f}%")

print("\n" + "=" * 80)
print("NEXT STEP: Investigate why some properties have 3-4x higher ROI")
print("  - Check if it's due to different base prices")
print("  - Check if it's due to 'free upgrades' (constraints that allow no-cost changes)")
print("  - Verify that cost accounting is correct")
print("=" * 80)
