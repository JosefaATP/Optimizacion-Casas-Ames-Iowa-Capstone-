#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precise static check: find feature keys referenced with x["..."], x.get("..."), or getVarByName("x_<name>")
and compare them to MODIFIABLE.
"""
import re
from pathlib import Path
from optimization.remodel.features import MODIFIABLE

repo_root = Path('.')
model_file = repo_root / 'optimization' / 'remodel' / 'gurobi_model.py'
text = model_file.read_text(encoding='utf-8')

# patterns
p_x_bracket = re.compile(r'x\s*\[\s*["\']([^"\']+)["\']\s*\]')
p_x_get = re.compile(r'x\.get\(\s*["\']([^"\']+)["\']')
p_getvar = re.compile(r'getVarByName\(\s*["\']x_([^"\']+)["\']')

bracket = set(p_x_bracket.findall(text))
getcall = set(p_x_get.findall(text))
getvar = set(p_getvar.findall(text))

all_refs = set(list(bracket) + list(getcall) + list(getvar))
modifiable_names = {f.name for f in MODIFIABLE}

missing = sorted([r for r in all_refs if r not in modifiable_names])

print('Found', len(bracket), 'x[...] usages (unique):', sorted(bracket)[:20])
print('Found', len(getcall), 'x.get(...) usages (unique):', sorted(getcall)[:20])
print('Found', len(getvar), 'getVarByName(x_...) usages (unique):', sorted(getvar)[:20])
print('\nTotal distinct referenced keys:', len(all_refs))
print('MODIFIABLE count:', len(modifiable_names))
print('\nMissing keys (referenced in gurobi_model but not in MODIFIABLE):')
for m in missing:
    print(' ', m)

# write out CSV
import csv
with open('static_var_precise_report.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['key','in_modifiable'])
    for k in sorted(all_refs):
        w.writerow([k, k in modifiable_names])
print('\nWrote static_var_precise_report.csv')
