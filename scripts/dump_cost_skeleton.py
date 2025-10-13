from pathlib import Path
import pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]  # raiz del repo
# ajusta el nombre del csv crudo que usaste para entrenar
csv_path = ROOT / "data" / "raw" / "casas_completas_con_present.csv"
df = pd.read_csv(csv_path)

def uniq(col):
    return sorted([str(x) for x in df[col].dropna().unique()])

costs = {
  "Utilities":           {u: 0 for u in uniq("Utilities")},
  "RoofStyle":           {u: 0 for u in uniq("Roof Style")},
  "RoofMatl":            {u: 0 for u in uniq("Roof Matl")},
  # union de Exterior 1st y 2nd para que no falte ninguno
  "Exterior":            {u: 0 for u in sorted(set(uniq("Exterior 1st")) | set(uniq("Exterior 2nd")))},
  "MasVnrType":          {u: 0 for u in uniq("Mas Vnr Type")},
  "Electrical":          {u: 0 for u in uniq("Electrical")},
  "Heating":             {u: 0 for u in uniq("Heating")},
  "KitchenQual":         {u: 0 for u in uniq("Kitchen Qual")},
  "CentralAirInstall":   0
}

out = ROOT / "costs" / "materials.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(costs, open(out, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print("Esqueleto escrito en", out)
