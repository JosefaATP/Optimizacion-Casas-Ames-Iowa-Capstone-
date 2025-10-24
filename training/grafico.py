import pandas as pd
import matplotlib.pyplot as plt
import os


ruta = "models/xgb_bayes_search/all_iterations.csv"
df = pd.read_csv(ruta)


output_dir = "models/xgb_bayes_search/plots"
os.makedirs(output_dir, exist_ok=True)


df = df[df["iteration"] <= 75]


metricas = {
    "r2": {"titulo": "Progreso del Coeficiente de Determinación (R²)", "ylabel": "R²", "ylim": (0.7, 1)},
    "rmse": {"titulo": "Progreso del Error Cuadrático Medio (RMSE)", "ylabel": "RMSE", "ylim": (25000, 60000)},
    "mape": {"titulo": "Progreso del Error Porcentual Absoluto Medio (MAPE)", "ylabel": "MAPE (%)", "ylim": (5, 15)}
}

plt.style.use("seaborn-v0_8-whitegrid")

for col, meta in metricas.items():
    mean_val = df[col].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["iteration"], df[col],
            color="#4B6C9E", linewidth=1.8, marker="o", markersize=4, alpha=0.9, label=col.upper())
    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1.4, label=f"Media {col.upper()}")

    ax.set_title(meta["titulo"], fontsize=15, pad=14, weight="semibold")
    ax.set_xlabel("Iteración", fontsize=12)
    ax.set_ylabel(meta["ylabel"], fontsize=12)
    ax.set_ylim(meta["ylim"])
    ax.set_xlim(1, 75)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11, loc="best", frameon=False)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1)


    output_path = os.path.join(output_dir, f"{col}_iter_75.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Gráfico guardado: {output_path}")
