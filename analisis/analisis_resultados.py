import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
import json
import re
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
# CARGAR META.JSON (estructura esperada)
# ==============================
try:
    META_PATH = Path("models/xgb/completa_present_log_p2_1800_ELEGIDO10__roof/meta.json")
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        EXPECTED_COLS = meta.get("numeric_cols", []) + meta.get("dummy_cols", [])
        print(f"📘 meta.json cargado correctamente ({len(EXPECTED_COLS)} columnas esperadas).")
    else:
        print(f"⚠️ No se encontró meta.json en {META_PATH}")
        EXPECTED_COLS = []
except Exception as e:
    print(f"⚠️ No se pudo cargar meta.json: {e}")
    EXPECTED_COLS = []

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
            # Ejecutar el modelo
            res = ejecutar_modelo(pid=pid, budget=presupuesto)

        except Exception as e:
            msg = str(e)
            if "columns are missing" in msg and EXPECTED_COLS:
                # Extraer las columnas faltantes del mensaje de error
                faltantes = set(re.findall(r"'([^']+)'", msg))
                print(f"🧩 Corrigiendo columnas faltantes ({len(faltantes)}): {list(faltantes)[:5]} ...")

                # Crear DataFrame vacío con columnas esperadas (rellenadas en 0)
                df_fix = pd.DataFrame(columns=EXPECTED_COLS)
                for col in faltantes:
                    df_fix[col] = 0

                # Reintentar ejecutar el modelo
                try:
                    res = ejecutar_modelo(pid=pid, budget=presupuesto)
                except Exception as e2:
                    print(f"❌ No se pudo corregir casa {pid}: {e2}")
                    resultados.append({
                        "pid": pid,
                        "presupuesto": presupuesto,
                        "error": str(e2)
                    })
                    continue
            else:
                print(f"⚠️ Error con casa {pid} - presupuesto {presupuesto}: {e}")
                resultados.append({
                    "pid": pid,
                    "presupuesto": presupuesto,
                    "error": str(e)
                })
                continue

        # Validar formato del resultado
        if isinstance(res, dict) and "precio_base" in res:
            precio_base = res.get("precio_base", 0)
            precio_remodelada = res.get("precio_remodelada", 0)
            total_cost = res.get("total_cost", 0)
            aumento_utilidad = res.get("aumento_utilidad", 0)

            # Calcular métricas adicionales
            roi = (aumento_utilidad / total_cost * 100) if total_cost > 0 else 0
            margen = (aumento_utilidad / precio_base * 100) if precio_base > 0 else 0

            resultados.append({
                "pid": pid,
                "presupuesto": presupuesto,
                "precio_base": precio_base,
                "precio_remodelada": precio_remodelada,
                "aumento_utilidad": aumento_utilidad,
                "total_cost": total_cost,
                "roi_%": round(roi, 2),
                "margen_%": round(margen, 2)
            })

        else:
            print(f"⚠️ Resultado inesperado para casa {pid}")
            resultados.append({
                "pid": pid,
                "presupuesto": presupuesto,
                "error": "resultado_invalido"
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
    df_validos = df_resultados[df_resultados["aumento_utilidad"].notna() & ~df_resultados["aumento_utilidad"].astype(str).str.contains("error")]

    if not df_validos.empty:
        resumen = (
            df_validos.groupby("presupuesto")
            .agg(
                aumento_utilidad_prom=("aumento_utilidad", "mean"),
                costo_prom=("total_cost", "mean"),
                valor_final_prom=("precio_remodelada", "mean"),
                roi_prom=("roi_%", "mean"),
                margen_prom=("margen_%", "mean")
            )
            .reset_index()
        )

        resumen["aumento_utilidad_prom"] = resumen["aumento_utilidad_prom"].round(0).astype(int)
        resumen["costo_prom"] = resumen["costo_prom"].round(0).astype(int)
        resumen["valor_final_prom"] = resumen["valor_final_prom"].round(0).astype(int)
        resumen["roi_%"] = resumen["roi_prom"].round(2)
        resumen["margen_%"] = resumen["margen_prom"].round(2)

        print("\n📊 Tabla resumen general:")
        print(resumen[["presupuesto", "aumento_utilidad_prom", "costo_prom", "valor_final_prom", "roi_%", "margen_%"]].to_string(index=False))
    else:
        print("⚠️ No hubo resultados válidos para generar promedios.")
else:
    print("⚠️ No se generaron resultados válidos para calcular promedios.")
