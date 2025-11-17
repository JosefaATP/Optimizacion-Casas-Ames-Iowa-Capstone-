# Comparación XGBoost vs. Regresión Lineal (CV 10x10)

Este paquete documenta, paso a paso, cómo contrastar el desempeño del modelo de **XGBoost** ganador del `bayes_summary.json` con la **regresión lineal** usada como caso base (`df_final_regresion.csv`). El objetivo es cuantificar, fold por fold, en qué porcentaje de viviendas XGBoost supera al baseline al estimar `SalePrice_Present`, así como explicar cuándo la regresión gana.

La guía está pensada para alguien que jamás haya leído del tema: si sigues las instrucciones literalmente podrás recrear los resultados, interpretarlos y generar evidencia para reportes ejecutivos.

---

## 1. Requisitos previos

1. **Python 3.9 o superior** (se usaron 3.11 durante el desarrollo).
2. Librerías:
   ```bash
   pip install --upgrade xgboost scikit-learn pandas numpy
   ```
   > Si trabajas en un entorno aislado (por ejemplo, este repositorio dentro de Codespaces o Conda), recuerda activar el ambiente antes de instalar.
3. Archivos de datos ya presentes en el repo:
   - `data/processed/base_completa_sin_nulos.csv` → dataset completo para XGBoost.
   - `data/raw/df_final_regresion.csv` → dataset original de la regresión baseline.
4. Archivo con hiperparámetros ganadores: `models/xgb_bayes_search/bayes_summary.json`.

No se requiere ningún otro dato externo: el script alinea ambas fuentes mediante la columna `PID` y valida que `SalePrice_Present` sea idéntico.

---

## 2. Estructura de la carpeta

```
analysis/xgb_vs_regression/
├── docs/
│   └── README.md           ← este documento con toda la guía
├── reports/                ← aparecerán aquí los CSV/JSON generados
│   └── (se llenará al ejecutar el script)
└── scripts/
    └── cv10_compare_baselines.py
```

- **`scripts/cv10_compare_baselines.py`**: orquestador principal. Reproduce la validación 10×10, entrena ambos modelos por fold y escribe resultados con nivel de detalle por vivienda.
- **`reports/`**: carpeta de salida. Después de correr el script encontrarás:
  - `fold_scores.csv`: 100 filas (10 repeticiones × 10 folds) con métricas R2/MAPE/RMSE/MAE para XGBoost y la regresión, además del `% de victorias`.
  - `repeat_means.csv`: promedios por repetición (10 filas), útil para resumir la estabilidad de cada modelo.
  - `summary.json`: panorama global (medias, desvíos estándar y tasas de victoria ponderadas).
  - `fold_predictions/rep_##_fold_##.csv`: archivo por fold con cada vivienda, el valor real, las predicciones, errores absolutos y un flag si XGBoost ganó.

---

## 3. Cómo ejecutar la comparación

1. Abre una terminal en la raíz del repositorio.
2. Ejecuta:
   ```bash
   python analysis/xgb_vs_regression/scripts/cv10_compare_baselines.py \
       --csv-xgb data/processed/base_completa_sin_nulos.csv \
       --csv-reg data/raw/df_final_regresion.csv \
       --bayes-summary models/xgb_bayes_search/bayes_summary.json \
       --target SalePrice_Present \
       --outdir analysis/xgb_vs_regression \
       --n-splits 10 \
       --repeats 10
   ```

   - `--csv-xgb` y `--csv-reg` apuntan a las bases de cada modelo; puedes reemplazarlas si generas nuevas versiones.
   - `--bayes-summary` se usa para leer los hiperparámetros exactos con los que se reportó `R2 = 0.93699`.
   - `--n-splits`/`--repeats` permiten probar otros esquemas (por ejemplo, 5×20), aunque el estudio oficial se basa en 10×10.

3. El script imprimirá confirmaciones y mostrará la tasa global de victorias de XGBoost.

> **Tip:** El proceso entrena 100 modelos de XGBoost con `n_estimators = 3758`, por lo que puede tardar varios minutos. Ejecuta en un entorno con al menos 8 GB de RAM para estar cómodo.

---

## 4. ¿Qué hace exactamente el script?

1. **Alinea datasets (PID a PID)**: se asegura de que ambas fuentes contengan las mismas viviendas y que el `SalePrice_Present` coincida.
2. **Prepara features**:
   - XGBoost reutiliza la lógica de `training/cv10_simple.py`, respetando columnas categóricas nativas (`enable_categorical=True`).
   - La regresión convierte variables categóricas a variables ficticias con `OneHotEncoder` e imputa los numéricos con la mediana antes de ajustar una regresión lineal.
3. **Reproduce los 100 folds**: usa `KFold(n_splits=10, shuffle=True, random_state=SEED + repetición)` tal como en los experimentos previos.
4. **Entrena y evalúa**:
   - Calcula `R2`, `MAPE`, `RMSE`, `MAE` y el error absoluto medio para ambos modelos.
   - Compara predicción contra valor real por vivienda y marca quién gana (menor error absoluto).
5. **Genera reportes autodescriptivos** que puedes abrir en Excel, PowerBI, Sheets o usar en notebooks para hacer análisis adicionales.

---

## 5. Interpretación del % de victorias

- **`pct_xgb_wins`** en `fold_scores.csv`: porcentaje (0–100) de viviendas del fold donde `|error_xgb| < |error_regresión|`.
- **`xgb_wins`** en los archivos por fold: bandera booleana por fila. Filtrando `xgb_wins == False` obtienes los casos donde la regresión gana; allí podrás buscar patrones (e.g. viviendas con muy pocas habitaciones, propiedades históricas, etc.).
- El `summary.json` incluye dos indicadores:
  - `mean_per_fold`: promedio simple del porcentaje de victorias a nivel fold.
  - `overall_weighted`: victorias totales de XGBoost divididas por el total de propiedades evaluadas (ponderado por tamaño de fold).

Si observas que la regresión gana en ciertos segmentos, puedes usar los CSV para agrupar por barrio, año de construcción o cualquier característica y justificar por qué ocurre (por ejemplo, XGBoost puede sobre-penalizar categorías con pocos ejemplos).

---

## 6. Preguntas frecuentes

| Pregunta | Respuesta |
| --- | --- |
| **¿Puedo usar otro modelo base?** | Sí. Reemplaza la parte final del pipeline en `cv10_compare_baselines.py` (la sección `build_regression_template`) por el modelo que prefieras. |
| **¿Cómo cambio las métricas?** | Modifica la función `evaluate_models` para agregar/quitar métricas. Recuerda actualizar `summary["metrics"]`. |
| **¿Por qué hay tantas columnas con espacios?** | La base original conserva los nombres exactos del dataset Ames; no los alteres para mantener compatibilidad con scripts previos. |
| **El script tarda demasiado** | Reduce `n_estimators` en `models/xgb_bayes_search/bayes_summary.json` (solo para pruebas) o corre menos repeticiones (`--repeats 2`) mientras depuras. |

---

## 7. Próximos pasos sugeridos

1. Ejecutar el script y revisar `reports/summary.json`.
2. Abrir `reports/fold_predictions` para localizar viviendas donde la regresión supera a XGBoost y preparar explicaciones cualitativas.
3. (Opcional) Crear visualizaciones rápidas en un notebook/notebook Python usando los CSV como insumo para una presentación.

Con esto tendrás una traza completa —datos, métricas y documentación— para defender la comparación ante cualquier revisión técnica o de negocio.
