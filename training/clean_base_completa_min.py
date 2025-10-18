# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/casas_completas_con_present.csv")
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "base_completa_sin_nulos.csv"

def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def main():
    df = pd.read_csv(RAW, sep=None, engine="python")
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]

    # 1) LotFrontage -> mediana por Neighborhood, fallback mediana global
    lotfront = first_existing(df, ["Lot Frontage","LotFrontage"])
    neigh    = first_existing(df, ["Neighborhood"])
    if lotfront:
        df[lotfront] = pd.to_numeric(df[lotfront], errors="coerce")
        if neigh:
            df[lotfront] = df[lotfront].fillna(df.groupby(neigh)[lotfront].transform("median"))
        df[lotfront] = df[lotfront].fillna(df[lotfront].median())

    # 2) Garage Yr Blt: 2207->2007, NaN->0 (int)
    gyb = first_existing(df, ["Garage Yr Blt","GarageYrBlt"])
    if gyb:
        df[gyb] = pd.to_numeric(df[gyb], errors="coerce")
        df.loc[df[gyb] == 2207, gyb] = 2007
        df[gyb] = df[gyb].fillna(0).astype(int)

    # 3) Mas Vnr Area / Type
    mva = first_existing(df, ["Mas Vnr Area","MasVnrArea"])
    mvt = first_existing(df, ["Mas Vnr Type","MasVnrType"])
    if mva:
        df[mva] = pd.to_numeric(df[mva], errors="coerce").fillna(0)   # numerica -> 0
    if mvt:
        df[mvt] = df[mvt].astype("string").fillna("NoAplica")         # categorica -> NoAplica

# reemplaza SOLO el bloque 4) y 5) por esto:

    # 4) eliminar filas con info clave de Garage faltante (regla estricta)
    garage_cars = first_existing(df, ["Garage Cars","GarageCars"])
    if garage_cars:
        before = len(df)
        bad_idx = df.index[df[garage_cars].isna()]
        df = df.loc[~df[garage_cars].isna()].copy()
        print(f"drop por {garage_cars} NaN: {before - len(df)} filas -> idx {list(bad_idx)}")

    # 5) eliminar filas con info clave de Bsmt faltante (regla estricta)
    total_bsmt = first_existing(df, ["Total Bsmt SF","TotalBsmtSF"])
    if total_bsmt:
        before = len(df)
        bad_idx = df.index[df[total_bsmt].isna()]
        df = df.loc[~df[total_bsmt].isna()].copy()
        print(f"drop por {total_bsmt} NaN: {before - len(df)} filas -> idx {list(bad_idx)}")


    # 6) eliminar filas con Electrical nulo (segun informe)
    elec = first_existing(df, ["Electrical"])
    if elec:
        before = len(df)
        df = df[~df[elec].isna()].copy()
        dropped = before - len(df)
        print(f"drop por Electrical NaN: {dropped} filas")
    
        # 6) claves extra: drop si falta cualquiera
    must_have_groups = [
        # Bsmt
        ["Bsmt Exposure","BsmtExposure"],
        ["Bsmt Full Bath","BsmtFullBath"],
        ["Bsmt Half Bath","BsmtHalfBath"],
        ["BsmtFin Type 2","BsmtFinType2"],
        # Garage
        ["Garage Finish","GarageFinish"],
    ]

    def drop_if_missing_any(df, names_list):
        # names_list es una lista con variantes del mismo campo
        col = first_existing(df, names_list)
        if not col:
            return 0
        # normaliza strings vacios a NaN
        df[col] = (df[col]
                    .replace(["", " ", "NA", "NaN", "None"], np.nan)
                    .where(~df[col].isna(), np.nan))
        before = len(df)
        bad_idx = df.index[df[col].isna()]
        df.drop(index=bad_idx, inplace=True)
        print(f"drop por {col} NaN/vacio: {before - len(df)} filas -> idx {list(bad_idx)}")
        return before - len(df)

    for variants in must_have_groups:
        drop_if_missing_any(df, variants)


    # 7) reporte rapido de NaN restantes (no imputamos mas)
    na_left = df.isna().sum().sort_values(ascending=False)
    print("top columnas con NaN remanente:\n", na_left[na_left>0].head(15))

    df.to_csv(OUT, index=False)
    print(f"ok -> {OUT} , filas: {len(df)}")

if __name__ == "__main__":
    main()
