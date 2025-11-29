#!/usr/bin/env python3
"""
Check why all_fixed is False
"""
import gurobipy as gp
import math

m = gp.Model()
xv = m.addVar(lb=10.0, ub=10.0, name="test")
m.update()  # IMPORTANT: Need to update model before reading variable attributes

print(f"xv.LB = {xv.LB}")
print(f"xv.UB = {xv.UB}")
print(f"getattr(xv, 'LB', None) = {getattr(xv, 'LB', None)}")
print(f"getattr(xv, 'UB', None) = {getattr(xv, 'UB', None)}")

def fin(v):
    try:
        return math.isfinite(float(v))
    except Exception as e:
        print(f"fin() failed with: {e}")
        return False

print(f"fin(xv.LB) = {fin(xv.LB)}")
print(f"fin(xv.UB) = {fin(xv.UB)}")

all_fixed = (
    fin(getattr(xv, "LB", None)) and 
    fin(getattr(xv, "UB", None)) and
    float(xv.LB) == float(xv.UB)
)
print(f"all_fixed = {all_fixed}")
