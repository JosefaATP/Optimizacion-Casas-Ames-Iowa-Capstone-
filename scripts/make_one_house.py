from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/raw/casas_completas_con_present.csv")

# si tu target se llama distinto, ajusta el nombre abajo
row = df.sample(1, random_state=7).drop(columns=["Sale Price"], errors="ignore")

out = ROOT / "data" / "processed" / "one_house.csv"
out.parent.mkdir(parents=True, exist_ok=True)
row.to_csv(out, index=False)
print("Casa base guardada en", out)
