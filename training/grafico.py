import pandas as pd
import matplotlib.pyplot as plt

# Ruta del archivo
ruta = "models/xgb_bayes_search/all_iterations.csv"

# Cargar datos
df = pd.read_csv(ruta)
r2_mean = df["r2"].mean()

# Estilo tipo paper
plt.style.use("seaborn-v0_8-whitegrid")

# Crear figura
fig, ax = plt.subplots(figsize=(14, 6))

# Línea principal R²
ax.plot(
    df["iteration"], df["r2"],
    color="#4B6C9E", linewidth=1.8, marker="o", markersize=4, alpha=0.9,

)

# Línea promedio
ax.axhline(
    r2_mean, color="gray", linestyle="--", linewidth=1.4,

)

# Formato general
ax.set_title("Progreso del Coeficiente de Determinación (R²) por Iteración", fontsize=15, pad=14, weight="semibold")
ax.set_xlabel("Iteración", fontsize=12)
ax.set_ylabel("R²", fontsize=12)
ax.set_ylim(0.7, 1)
ax.set_xlim(1, 74)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=11, loc="lower right", frameon=False)

# Márgenes
plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1)
plt.show()
