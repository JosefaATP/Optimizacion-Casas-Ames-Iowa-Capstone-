import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
from optimization.remodel.run_opt import ejecutar_modelo

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
DATA_PATH = Path("data/processed/base_completa_sin_nulos.csv")
RESULTS_PATH = Path("analisis/resultados")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ==============================
# PARÁMETROS DEL EXPERIMENTO
# ==============================
N_CASAS = 100
PRESUPUESTOS = [15000, 40000, 75000, 200000]

# ==============================
# CARGA Y VALIDACIÓN DE DATOS
# ==============================
df = pd.read_csv(DATA_PATH)

if "PID" not in df.columns:
    raise ValueError("El archivo no contiene una columna llamada 'PID'.")

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
# ANÁLISIS Y TABLA FINAL
# ==============================
if not df_resultados.empty and "aumento_utilidad" in df_resultados.columns:
    df_resultados["ROI_%"] = (
        100 * df_resultados["aumento_utilidad"] /
        df_resultados["total_cost"].replace(0, float("nan"))
    )

    df_resultados["Margen_%"] = (
        100 * df_resultados["aumento_utilidad"] /
        df_resultados["precio_remodelada"].replace(0, float("nan"))
    )

    resumen = (
        df_resultados.groupby("presupuesto")
        .agg(
            aumento_utilidad=("aumento_utilidad", "mean"),
            ROI_prom=("ROI_%", "mean"),
            margen_prom=("Margen_%", "mean"),
            costos_totales=("total_cost", "mean"),
            valor_final_casa=("precio_remodelada", "mean")
        )
        .reset_index()
    )

    resumen["aumento_utilidad"] = resumen["aumento_utilidad"].round(0).astype(int)
    resumen["ROI_%"] = resumen["ROI_prom"].round(1).astype(str) + "%"
    resumen["margen_%"] = resumen["margen_prom"].round(1).astype(str) + "%"
    resumen["costos_totales"] = resumen["costos_totales"].round(0).astype(int)
    resumen["valor_final_casa"] = resumen["valor_final_casa"].round(0).astype(int)

    print("\n📊 Tabla resumen final:")
    print(
        resumen[
            ["presupuesto", "aumento_utilidad", "ROI_%", "margen_%", "costos_totales", "valor_final_casa"]
        ].to_string(index=False)
    )

else:
    print("⚠️ No se generaron resultados válidos para calcular promedios.")
