import pandas as pd
from pathlib import Path


p_in  = Path("data/raw/casas_completas_sinnulos.csv")
p_out = Path("data/raw/casas_completas_con_present.csv")

cpi_dict = {
    2006: 201.558,
    2007: 207.344,
    2008: 215.254,
    2009: 214.565,
    2010: 218.,  
}

REF_CPI = 320.0   

df = pd.read_csv(p_in, sep=None, engine="python")
df.columns = [c.replace("\ufeff","").strip() for c in df.columns]

yr_candidates = ["Yr Sold","YrSold","Year Sold","YearSold","Yr_Sold","Year_Sold"]
yr_col = next((c for c in yr_candidates if c in df.columns), None)
if yr_col is None:
    raise ValueError(f"No encuentro la columna de año de venta. Columnas disponibles: {df.columns.tolist()}")


df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
df[yr_col]      = pd.to_numeric(df[yr_col], errors="coerce").astype("Int64")


def present(row):
    y = row[yr_col]
    p = row["SalePrice"]
    if pd.isna(y) or pd.isna(p): 
        return pd.NA
    y = int(y)
    if y not in cpi_dict:
        return pd.NA
    return p * (REF_CPI / cpi_dict[y])

df["SalePrice_Present"] = df.apply(present, axis=1)


print("[info] factores unicos (REF_CPI/CPI_año):",
      sorted(df.dropna(subset=[yr_col])[[yr_col]].assign(f=lambda x: REF_CPI / x[yr_col].map(cpi_dict)).dropna()["f"].unique())[:10])
print("[info] ejemplo:")
print(df[[yr_col,"SalePrice","SalePrice_Present"]].head())


removed = df["SalePrice_Present"].isna().sum()
if removed:
    print(f"[warn] filas sin SalePrice_Present: {removed} (año fuera de diccionario o datos faltantes)")

df = df.dropna(subset=["SalePrice_Present"])


df.to_csv(p_out, index=False)
print(f"[ok] guardado -> {p_out}")