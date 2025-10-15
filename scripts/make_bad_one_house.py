"""Make a degraded test house (one_house_bad.csv) from the existing processed one_house.csv.

This script reads `data/processed/one_house.csv`, takes the first row, changes several
features to poor/low-quality values (so XGBoost is more likely to recommend upgrades)
and writes the result to `data/processed/one_house_bad.csv` without overwriting the original.

Usage (from repo root, with venv active):
  & ".venv/Scripts/python.exe" "scripts/make_bad_one_house.py"

After that you can either:
  - copy the bad file over the original (backup first), or
  - edit `run_remodel.py` to point to the bad file while testing.

The script is intentionally conservative and does not overwrite any file.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
PROC = ROOT.parent / "data" / "processed"
SRC = PROC / "one_house.csv"
OUT = PROC / "one_house_bad.csv"

if not SRC.exists():
    raise SystemExit(f"source file not found: {SRC}")

df = pd.read_csv(SRC)
if df.shape[0] == 0:
    raise SystemExit("source file empty")

# take first row and make a copy
row = df.iloc[0].copy()

# Set of degradations to force remodel recommendations
bad_changes = {
    # categorical low-quality choices
    'Kitchen Qual': 'Po',
    'Exter Qual': 'Po',
    'Exter Cond': 'Po',
    'Mas Vnr Type': 'None',
    'Electrical': 'FuseA',
    'Heating': 'Wall',
    'Central Air': 'No',
    'Utilities': 'NoSeWa',
    # cheap roof/exterior
    'Roof Style': 'Flat',
    'Roof Matl': 'Roll',
    'Exterior 1st': 'Plywood',
    'Exterior 2nd': '',
    # rooms/baths low but valid
    'Full Bath': 1,
    'Half Bath': 0,
    'Bedroom AbvGr': 1,
    'Kitchen AbvGr': 1,
    # basement unfin large so finishing is attractive
    'Bsmt Unf SF': max(400, int(float(row.get('Bsmt Unf SF', 0) or 0) + 400)),
    'BsmtFin SF 1': 0,
    'BsmtFin SF 2': 0,
    # pool none
    'Pool Area': 0,
    'Pool QC': 'None',
}

for k, v in bad_changes.items():
    if k in row.index:
        row[k] = v

# ensure numeric features are numeric
for col in ['Bsmt Unf SF', 'BsmtFin SF 1', 'BsmtFin SF 2', '1st Flr SF', '2nd Flr SF', 'Gr Liv Area']:
    if col in row.index:
        try:
            row[col] = float(row[col] or 0.0)
        except Exception:
            row[col] = 0.0

# write single-row bad file
bad_df = pd.DataFrame([row])
bad_df.to_csv(OUT, index=False, encoding='utf-8')
print(f"Wrote degraded house to: {OUT}")
print("To test: back up original and replace, or edit run_remodel to read this file.")
