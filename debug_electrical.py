"""Debug: Verificar variables eléctricas en features."""
import pandas as pd
import sys
sys.path.insert(0, '/home/valen')
from training.data_loader import load_data

# Cargar los datos
df, metadata = load_data()

# Ver columnas que contienen "Electrical"
elec_cols = [col for col in df.columns if "Electrical" in col]
print(f"Columnas con 'Electrical': {elec_cols}")

# También "elect"
elect_cols = [col for col in df.columns if "elect" in col.lower()]
print(f"Columnas con 'elect': {elect_cols}")

# Ver las primeras filas de una casa
print("\nPrimera casa (basura):")
print(df.iloc[0][['Electrical'] + elec_cols])

# Casa 528328100
casa_idx = df[df['PID'] == 528328100].index
if len(casa_idx) > 0:
    print(f"\nCasa 528328100:")
    print(df.iloc[casa_idx[0]][['Electrical'] + elec_cols])
