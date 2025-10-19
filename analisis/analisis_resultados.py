import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimization.remodel.run_opt import ejecutar_modelo

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
DATA_PATH = Path("data/processed/base_completa_sin_nulos.csv")  # Ajusta si tu dataset está en otro lugar
RESULTS_PATH = Path("analisis/resultados")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ==============================
# PARÁMETROS DEL EXPERIMENTO
# ==============================
N_CASAS = 5  # Número de casas a tomar al azar
PRESUPUESTOS = [15000, 40000, 200000]  # Presupuestos a evaluar

# ==============================
# CARGA Y VALIDACIÓN DE DATOS
# ==============================
df = pd.read_csv(DATA_PATH)

if "PID" not in df.columns:
    raise ValueError("El archivo no contiene una columna llamada 'PID'.")

# Filtrar casas válidas
df_validas = df[df["PID"].notna()].drop_duplicates(subset=["PID"])
casas_muestra = df_validas.sample(n=min(N_CASAS, len(df_validas)), random_state=42)

print(f"\n✅ Se seleccionaron {len(casas_muestra)} casas válidas de la base de datos.\n")

# ==============================
# EJECUCIÓN DEL MODELO
# ==============================
resultados = []

print(f"🔍 Ejecutando {len(casas_muestra)} casas con {len(PRESUPUESTOS)} presupuestos cada una...\n")

for _, casa in tqdm(casas_muestra.iterrows(), total=len(casas_muestra), desc="Procesando casas"):
    pid = int(casa["PID"])

    for presupuesto in PRESUPUESTOS:
        print(f"\n🏠 Ejecutando optimización para casa {pid} con presupuesto ${presupuesto:,}")
        try:
            res = ejecutar_modelo(pid=pid, budget=presupuesto)
            resultados.append({
                "pid": pid,
                "presupuesto": presupuesto,
                **res
            })
        except Exception as e:
            print(f"⚠️ Error con casa {pid} - presupuesto {presupuesto}: {e}")
            resultados.append({
                "pid": pid,
                "presupuesto": presupuesto,
                "error": str(e)
            })

# ==============================
# GUARDAR RESULTADOS
# ==============================
df_resultados = pd.DataFrame(resultados)
nombre_archivo = f"estudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
ruta_guardado = RESULTS_PATH / nombre_archivo
df_resultados.to_csv(ruta_guardado, index=False)
print(f"\n✅ Resultados guardados en: {ruta_guardado}")

# ==============================
# ANÁLISIS Y GRÁFICO
# ==============================
if not df_resultados.empty and "aumento_utilidad" in df_resultados.columns:
    df_resultados["rentabilidad_%"] = (
        100 * df_resultados["aumento_utilidad"] /
        df_resultados["total_cost"].replace(0, float("nan"))
    )

    resumen = (
        df_resultados.groupby("presupuesto")
        .agg(aumento_utilidad=("aumento_utilidad", "mean"),
             rentabilidad_prom=("rentabilidad_%", "mean"))
        .reset_index()
    )

    resumen["aumento_utilidad"] = resumen["aumento_utilidad"].round(0).astype(int)
    resumen["rentabilidad_%"] = resumen["rentabilidad_prom"].round(1).astype(str) + "%"

    print("\n📊 Tabla resumen:")
    print(resumen[["presupuesto", "aumento_utilidad", "rentabilidad_%"]].to_string(index=False))

    # ==============================
    # GRÁFICO + TABLA EMBEBIDA
    # ==============================
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Línea azul: aumento utilidad
    ax1.plot(
        resumen["presupuesto"],
        resumen["aumento_utilidad"],
        marker="o",
        color="tab:blue",
        label="Aumento utilidad ($)"
    )
    ax1.set_xlabel("Presupuesto ($)")
    ax1.set_ylabel("Aumento utilidad ($)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Barras verdes: rentabilidad %
    ax2 = ax1.twinx()
    ax2.bar(
        resumen["presupuesto"],
        resumen["rentabilidad_prom"],
        alpha=0.3,
        color="tab:green",
        width=40000,
        label="Rentabilidad (%)"
    )
    ax2.set_ylabel("Rentabilidad (%)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("Análisis de Remodelaciones por Presupuesto")
    fig.tight_layout()

    # Crear tabla en formato solicitado
    tabla = resumen[["presupuesto", "aumento_utilidad", "rentabilidad_%"]]
    cell_text = tabla.values.tolist()
    column_labels = tabla.columns

    tabla_plot = plt.table(
        cellText=cell_text,
        colLabels=column_labels,
        loc="bottom",
        cellLoc="center"
    )
    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(9)
    tabla_plot.scale(1.1, 1.3)

    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Guardar gráfico
    grafico_path = RESULTS_PATH / f"grafico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(grafico_path, dpi=300)
    plt.show()
    print(f"\n📈 Gráfico con tabla guardado en: {grafico_path}")

else:
    print("⚠️ No se generaron resultados válidos para calcular promedios.")
