#!/usr/bin/env python
"""
Check PWL approximation accuracy.
"""

import numpy as np

def check_pwl_accuracy():
    # Build the PWL grid
    grid = np.linspace(10.0, 14.0, 161)
    y_prices = np.expm1(grid)
    
    # Test values
    test_values = [
        12.389358,  # base value 
        12.507481,  # MIP value (after solve)
        12.394826,  # external predict
    ]
    
    for y_log_test in test_values:
        # Find bracketing indices
        if y_log_test < grid[0] or y_log_test > grid[-1]:
            print(f"y_log={y_log_test:.6f} is OUTSIDE grid range!")
            continue
        
        # Exact value
        y_exact = np.expm1(y_log_test)
        
        # PWL interpolation
        idx = np.searchsorted(grid, y_log_test)
        if idx == 0:
            y_pwl = y_prices[0]
        elif idx >= len(grid):
            y_pwl = y_prices[-1]
        else:
            # Linear interpolation between grid[idx-1] and grid[idx]
            x0, x1 = grid[idx-1], grid[idx]
            y0, y1 = y_prices[idx-1], y_prices[idx]
            # y = y0 + (y_log_test - x0) * (y1 - y0) / (x1 - x0)
            y_pwl = y0 + (y_log_test - x0) * (y1 - y0) / (x1 - x0)
        
        delta = y_pwl - y_exact
        pct = 100.0 * delta / y_exact if y_exact != 0 else 0
        
        print(f"\ny_log={y_log_test:.6f}")
        print(f"  exact:  {y_exact:,.2f}")
        print(f"  PWL:    {y_pwl:,.2f}")
        print(f"  Δ:      {delta:+,.2f} ({pct:+.4f}%)")

if __name__ == "__main__":
    check_pwl_accuracy()
