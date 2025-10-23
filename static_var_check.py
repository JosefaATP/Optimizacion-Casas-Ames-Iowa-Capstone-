#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static check: compare feature names referenced in gurobi_model.py against MODIFIABLE
"""
import re
from pathlib import Path

repo_root = Path('.')
model_file = repo_root / 'optimization' / 'remodel' / 'gurobi_model.py'

# import modifiable names
from optimization.remodel.features import MODIFIABLE
modifiable_names = {f.name for f in MODIFIABLE}

text = model_file.read_text(encoding='utf-8')

# patterns to find:
# - getVarByName("x_<name>") or getVarByName('x_<name>')
# - x["<name>"] or x.get("<name>")
# - quoted literals that look like feature names (words, spaces, numbers, % & '-')

p_getvar = re.compile(r"getVarByName\(\s*[\"']x_([^\"']+)[\"']\s*\)")
refs = set(p_getvar.findall(text))

p_x_index = re.compile(r"x\[\s*[\"']([^\"']+)[\"']\s*\]")
refs |= set(p_x_index.findall(text))

p_x_get = re.compile(r"x\.get\(\s*[\"']([^\"']+)[\"']")
refs |= set(p_x_get.findall(text))

# find quoted literals that look like feature names (at least one space or underscore OR letters and digits)
p_quoted = re.compile(r"[\"']([A-Za-z0-9 _\-&\(\)]+)[\"']")
for m in p_quoted.findall(text):
    # filter very short tokens and python keywords
    if len(m) > 2 and not m.isnumeric():
        # heuristics: include ones with spaces (likely features) or underscores or capitalized words
        if ' ' in m or '_' in m or any(c.isupper() for c in m[:2]):
            refs.add(m)

# normalize some obvious variants used in code
# e.g. 'Kitchen Qual' vs 'Kitchen Qual' same; ensure stripping
refs = {r.strip() for r in refs if r.strip()}

# Now find which refs are not in modifiable_names
not_in_mod = sorted([r for r in refs if r not in modifiable_names])

print('Total modifiable features:', len(modifiable_names))
print('Total referenced feature-like tokens found:', len(refs))
print('\nSample referenced tokens (first 80):')
for i, r in enumerate(sorted(refs)[:80]):
    print(' ', r)

print('\nReferenced tokens NOT in MODIFIABLE (candidates):')
for r in not_in_mod:
    print(' ', r)

# write a CSV for inspection
import csv
out = 'static_var_check_report.csv'
with open(out, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['token','in_modifiable'])
    for r in sorted(refs):
        w.writerow([r, r in modifiable_names])
print('\nWrote', out)
