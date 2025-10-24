# README 

>1. Guia completa para correr y entender el modulo de **remodelacion** del proyecto *Optimizacion Casas Ames Iowa Capstone*.
>2. Guia completa para entender el modulo de **training** del XGBoost.
---
# Módulo `optimization/remodel`

## 1) Que hace este modulo

Este modulo construye y resuelve un MIP en Gurobi que decide **qué cambios de remodelacion conviene hacer** a una casa para **maximizar la utilidad**:
utilidad = (precio_remodelado - costo_total) - costo_inicial.

* El **precio** lo predice un modelo XGBoost ya entrenado y encapsulado en un pipeline de preprocesamiento.
* Los **costos** de cambiar atributos estan definidos en tablas y reglas de negocio.
* El MIP respeta politicas como no degradar calidades, no crear features prohibidas, limites de presupuesto, etc.

---

## 2) Requisitos rapidos

* Python 3.11 recomendado
* Gurobi 10+ con licencia activa
* Windows, macOS o Linux
* Paquetes del archivo `requirements.txt` en la raiz del repo

### Instalacion en una terminal limpia

```bash
# crear venv
python -m venv .venv311
# activar
# Windows
.venv311\Scripts\activate
# macOS/Linux
# source .venv311/bin/activate

# instalar dependencias
pip install -r requirements.txt
```
---

## 3) Estructura del modulo y para que sirve cada archivo

Ruta base: `optimization/remodel/`

* `__init__.py`
  inicializador del paquete, sin logica de negocio.

* `benchmark_remodel.py`
  corre *benchmarks* de punta a punta en varios casos, guarda resultados en `bench_out/`.
  util para pruebas rapidas y para validar tiempos y brechas.

* `check_env.py`
  chequea que el entorno este OK, por ejemplo version de Python, paquetes clave, y licencia Gurobi.
  primer comando que conviene correr.

* `compat_sklearn.py`
  helpers para compatibilidad scikit-learn, por ejemplo wrappers de pipeline y transformaciones.

* `compat_xgboost.py`
  utilidades para extraer arboles y hojas del XGBoost y conectarlos con el MIP.

* `config.py`
  **archivo de configuracion central**: rutas de datos y modelos, seeds, defaults de ejecucion, y flags.
  si algo no corre por rutas, revisa aqui primero.

* `costs.py`
  tablas de costos y funciones para mapear **costo por cambio** de cada categoria, por ejemplo:
  +1 dormitorio, +1 baño, metros agregados, garage por auto, deck m2, etc.
  tambien contiene politicas por categoria.

* `features.py`
  define el set de **features modificables**, dominios y codificaciones, ademas de meta-datos usados por el MIP
  (por ejemplo calidades ordinales, binarios de garage, piscina, etc.).

* `gurobi_model.py`
  **corazon del modulo**. Construye el **MIP embedido** con la prediccion del XGBoost, agrega restricciones de negocio,
  arma la funcion objetivo y retorna un `gp.Model` listo para resolver.
  tambien contiene politicas de "no empeorar", PWL para deshacer el log si aplica, y flags de debug.

* `io.py`
  carga/guarda datos, lee la casa base por `pid`, exporta resultados de corrida, snapshots, y reportes en texto.

* `run_opt.py`
  **driver principal** para correr un caso: lee config, carga una casa base, construye el MIP, resuelve,
  e imprime un **resumen legible**: aumento de utilidad, precio base, precio remodelado, costos y lista de cambios.

* `utils.py`
  funciones auxiliares: manejo de seeds, formatos, timers, trazas, checks.

* `xgb_predictor.py`
  encapsula el pipeline de precio: preprocesa features, ejecuta el XGBoost y expone funciones para el MIP.

---

## 4) Flujo completo 

1. Configura rutas en `optimization/remodel/config.py`

   * ruta del CSV base con casas
   * ruta del modelo XGBoost guardado
   * carpeta de salida `bench_out/`
   * defaults de presupuesto, tier, etc.

2. Verifica entorno

```bash
python -m optimization.remodel.check_env
```

3. Corre un caso simple por `pid` y presupuesto

```bash
python -m optimization.remodel.run_opt --pid 534402170 --budget 50000
```

4. Revisa la salida en consola y los archivos en `bench_out/`.

5. (Opcional) Corre un benchmark sobre una lista de casos

```bash
python -m optimization.remodel.benchmark_remodel --n 0 --budget 50000
```

> Nota: Cuidado con la cantidad de casas a correr en este último punto, pus se demora bastante (aprox, 1 minuto por casa x 3 budgets)
---

## 5) Datos y modelos

* **Datos**: el CSV base está en    `data/processed/base_completa_sin_nulo.csv`

* **Modelo XGB**: carpeta bajo `models/xgb/...` con `model_xgb.joblib` y `booster.json`.
  La ruta se configura en `config.py`.

---

## 6) Salidas esperadas

En consola se verá algo como:

```
Optimal solution found (tolerance 1.00e-03)
Aumento de Utilidad: $220,671
precio casa base: $103,700
precio casa remodelada: $324,371
costos totales de remodelacion: $0

Cambios hechos en la casa
- Total Bsmt SF: base -> 816  ;  nuevo -> 264
- ...
```

Métricas y gráficos se guardan en `bench_out/` (al correr el archivo `benchmark_remodel.py`):

* CSV con resultados agregado por corrida
* Snapshot de cambios propuestos por atributo
* Logs con gap, tiempo y restricciones activas si debug esta ON

---
# Módulo `training`

Bayes_tune_with_retrain

# `training/bayes_tune_with_retrain.py`

Script que realiza **búsqueda bayesiana** de hiperparámetros para **XGBoost** con `scikit-optimize` (`gp_minimize`).
En **cada iteración** re-entrena el modelo llamando a tu función oficial `retrain_xgb`, evalúa en test y guarda métricas/artefactos.

## ¿Qué hace el código?

1. **Fija semillas** para reproducibilidad (`numpy`, `random`, `PYTHONHASHSEED`).
2. Define un **espacio de búsqueda** (rango de hiperparámetros XGB).
3. Usa `gp_minimize` (adquisición **EI**) para proponer hiperparámetros.
4. En cada propuesta:

   * Combina los parámetros sugeridos con tu `Config().xgb_params`.
   * Llama a `retrain_xgb(csv_path, outdir, cfg, verbose=False)`.
   * Lee **métricas de test** (R², RMSE, MAE, MAPE, skew, kurtosis).
   * Registra resultados y actualiza récord de R².
   * Si R² ≥ umbral, guarda el trial como **finalista**.
5. Al final, genera un **resumen** con el mejor R² y los mejores parámetros.

## Requisitos

```bash
pip install xgboost scikit-learn scikit-optimize numpy
```

Estructura/funciones internas esperadas:

* `training.funcion.retrain_xgb(csv_path, outdir, cfg, verbose)`
* `training.config.Config` (con `xgb_params`)
* `training.metrics.regression_report` (solo import estático)

## Uso rápido

Ejecutar **desde la raíz del repo**:

```bash
python -m training.bayes_tune_with_retrain --csv RUTA_AL_CSV
```

Ejemplo:

```bash
python -m training.bayes_tune_with_retrain \
  --csv data/processed/base_completa_sin_nulos.csv
```

### Argumentos

| Flag                 | Tipo  | Default                     | Descripción                                   |
| -------------------- | ----- | --------------------------- | --------------------------------------------- |
| `--csv`              | str   | *(obligatorio)*             | Ruta al CSV limpio que consume `retrain_xgb`. |
| `--base-outdir`      | str   | `models/xgb_bayes_search_2` | Carpeta raíz para resultados.                 |
| `--n-calls`          | int   | `80`                        | Iteraciones totales de la búsqueda.           |
| `--n-initial-points` | int   | `25`                        | Puntos aleatorios iniciales (exploración).    |
| `--r2-threshold`     | float | `0.93`                      | Umbral de R² para marcar *finalistas*.        |

## Salidas

Dentro de `--base-outdir`:

* `trial_XXX/` → modelo y métricas **de cada iteración** (lo que devuelva `retrain_xgb`).
* `all_iterations.csv` → **todas** las iteraciones: `iteration,r2,rmse,mae,mape,residual_skew,residual_kurtosis`.
* `progressive_records.csv` / `.jsonl` → solo cuando se **supera el mejor R²** previo.
* `bayes_summary.json` → resumen final: mejor R², mejores hiperparámetros, iteración del mejor y finalistas.

## Cómo cambiar el espacio de búsqueda

Edita el bloque `SEARCH_SPACE` en el archivo:

```python
SEARCH_SPACE = [
    Integer(1200, 4000,  name="n_estimators"),
    Real(0.02, 0.07,     prior="log-uniform", name="learning_rate"),
    Integer(3, 7,        name="max_depth"),
    Integer(4, 14,       name="min_child_weight"),
    Real(0.0, 3.0,       name="gamma"),
    Real(0.65, 1.0,      name="subsample"),
    Real(0.4, 1.0,       name="colsample_bytree"),
    Real(0.5, 4.0,       name="reg_lambda"),
    Real(0.05, 1.2,      prior="log-uniform", name="reg_alpha"),
]
```

## Tips / Problemas comunes

* **“No encuentra módulos”** → Ejecuta con `-m` desde la raíz, asegúrate que `training/` tenga `__init__.py`.
* **No hay finalistas** → baja `--r2-threshold` (p. ej., `0.90`) o amplía rangos en `SEARCH_SPACE`.
* **Lento** → reduce `--n-calls`, acota `n_estimators` máx., o baja `--n-initial-points`.



Build_present_from_year_cpi


# Ajuste de Precio a Valor Presente

Este script toma un dataset de ventas de casas y **convierte el precio de venta original a “precio presente”**, usando un **factor de corrección basado en el CPI (Índice de Precios al Consumidor)**.

La idea es que una casa vendida en 2006 no vale lo mismo en términos reales que una casa vendida en 2010, incluso si el precio nominal es igual.
Por eso se recalcula:

```
Precio_Presente = Precio_Original * (REF_CPI / CPI_del_año)
```

Donde:

* `REF_CPI` es un valor de referencia común utilizado para igualar todas las ventas.
* `cpi_dict` contiene el CPI por año disponible.

---

## Entrada y salida

| Archivo                                    | Descripción                                                 |
| ------------------------------------------ | ----------------------------------------------------------- |
| `data/raw/casas_completas_sinnulos.csv`    | Dataset original con precios y año de venta                 |
| `data/raw/casas_completas_con_present.csv` | Dataset resultante con la nueva columna `SalePrice_Present` |

---

## ¿Qué hace el script?

1. **Lee el CSV** independientemente de si está separado por `,` o `;`.

2. **Detecta automáticamente la columna del año** (ej. `Yr Sold`, `YearSold`, etc.).

3. Convierte `SalePrice` y el año a valores numéricos.

4. Calcula **`SalePrice_Present`** usando la fórmula:

   ```python
   Precio_Presente = SalePrice * (REF_CPI / CPI[año])
   ```

5. Reporta si hay filas que no se pudieron calcular (por años no presentes en el CPI).

6. **Guarda el dataset resultante** en formato CSV estándar (separador coma).

---

## Ajusta el CPI si lo necesitas

El CPI usado está definido aquí:

```python
cpi_dict = {
    2006: 201.558,
    2007: 207.344,
    2008: 215.254,
    2009: 214.565,
    2010: 218.,  # referencia
}

REF_CPI = 320.0
```

* Si quieres usar otro año base → cambia `REF_CPI`.
* Si tu dataset incluye otros años → agrega valores al `cpi_dict`.

---

## Ejecución

Solo corre el script:

```bash
python ajustar_precio_presente.py
```

(o el nombre que tenga en tu proyecto)

Si termina correctamente mostrará algo como:

```
[ok] guardado -> data/raw/casas_completas_con_present.csv
```

---

## Resultado final

Tu dataset ahora incluye una nueva columna:

| Año  | SalePrice (original) | SalePrice_Present |
| ---- | -------------------- | ----------------- |
| 2007 | 150,000              | 231,785           |
| 2010 | 150,000              | 220,183           |
| 2006 | 150,000              | 238,210           |





cv10_simple.py 

Perfecto. Te dejo un **README breve y claro en español**, listo para copiar/pegar para `training/cv10_simple.py`:

---

# Validación Cruzada 10×10 para XGBoost

Script que ejecuta **validación cruzada repetida** (K-Fold con `K=10` y `M=10` repeticiones por defecto) sobre un **XGBRegressor** con hiperparámetros fijos (`BEST_PARAMS`). Calcula y guarda **R², RMSE, MAE y MAPE** por *fold*, promedios por *repetición* y un **resumen final** (media ± sd de los promedios por repetición).

> Archivo: `training/cv10_simple.py`

---

## ¿Qué hace?

1. **Carga datos** desde un CSV y elimina columnas no usadas (`PID`, `Order`, `SalePrice`, `\ufeffOrder` si existen).
2. Separa `X` (todas las columnas salvo la variable objetivo) y `y` (por defecto `SalePrice_Present`).
3. Soporta dos modos de categóricas:

   * `--one-hot` → aplica `pd.get_dummies` (one-hot).
   * (por defecto) **Soporte categórico nativo** de XGBoost (`enable_categorical=True`).
4. Ejecuta **KFold** con barajado y semilla controlada, repitiendo el K-Fold **M** veces (cambia la semilla base en cada repetición).
5. Entrena y predice por *fold*; calcula **R², RMSE, MAE, MAPE** y guarda resultados.
6. Genera:

   * `fold_scores.csv`: métricas por cada fold de cada repetición.
   * `repeat_means.csv`: promedios por repetición.
   * `summary.json`: **media y sd** (ddof=1) de los promedios por repetición + metadatos del CV.

---

## Requisitos

```bash
pip install xgboost scikit-learn numpy pandas
```

---

## Uso rápido

Ejecutar desde la raíz del repo:

```bash
python -m training.cv10_simple --csv RUTA_AL_CSV
```

Parámetros:

| Flag         | Tipo | Default                             | Descripción                              |
| ------------ | ---- | ----------------------------------- | ---------------------------------------- |
| `--csv`      | str  | *(obligatorio)*                     | Ruta al CSV con los datos.               |
| `--target`   | str  | `SalePrice_Present`                 | Columna objetivo.                        |
| `--outdir`   | str  | `models/xgb_bayes_search_2/cv10x10` | Carpeta de salida.                       |
| `--n-splits` | int  | `10`                                | Número de folds (K).                     |
| `--repeats`  | int  | `10`                                | Número de repeticiones (M).              |
| `--one-hot`  | flag | `False`                             | Usa one-hot en vez de categórico nativo. |

Ejemplos:

```bash
# Modo por defecto: categórico nativo XGBoost
python -m training.cv10_simple --csv data/raw/casas_completas_con_present.csv

# Con one-hot y menos repeticiones
python -m training.cv10_simple --csv data/raw/casas.csv --one-hot --repeats 5 --n-splits 5 --outdir models/xgb/cv5x5_onehot
```

---

## Salidas

En `--outdir`:

* `fold_scores.csv` → filas = `repeats * n_splits` (cada fold de cada repetición):
  columnas `rep, fold, R2, RMSE, MAE, MAPE`.
* `repeat_means.csv` → 1 fila por repetición con el **promedio** de cada métrica.
* `summary.json` →

  ```json
  {
    "R2":   {"mean": ..., "sd": ...},
    "RMSE": {"mean": ..., "sd": ...},
    "MAE":  {"mean": ..., "sd": ...},
    "MAPE": {"mean": ..., "sd": ...},
    "n_splits": 10,
    "repeats": 10,
    "seed_base": 42,
    "one_hot": false
  }
  ```

---

## Notas importantes

* Los hiperparámetros usados están en `BEST_PARAMS` (incluye `tree_method="hist"`, `random_state=42`, `n_jobs=-1`). Ajusta si lo necesitas.
* Si usas **categórico nativo** (default), el script convierte columnas no numéricas a `category` y activa `enable_categorical=True`.
  Con `--one-hot`, se aplica `pd.get_dummies(...)` y **se desactiva** `enable_categorical`.
* **RMSE** se calcula como `sqrt(MSE)` por compatibilidad con versiones antiguas de scikit-learn.
* **MAPE** está implementado manualmente (evita división por cero con `eps=1e-12`).

Funcion.py


# Re-entrenamiento XGBoost (mismo entorno / pipeline)

Script que **re-entrena un modelo XGBoost** usando **el mismo preprocesamiento y configuración** que el pipeline original, y guarda modelo, booster, métricas y metadatos de entrenamiento.

> Archivo: `training/retrain_xgb_same_env.py`
> Función principal que puedes importar: `retrain_xgb(...)`

---

## ¿Qué hace?

1. **Parche de compatibilidad XGBoost**
   `patch_get_booster()` asegura que el booster pueda serializarse/recargarse de forma robusta (se guarda `save_raw()`), evitando problemas de entorno.

2. **Carga y limpieza de datos**

   * Lee el CSV con separador autodetectado.
   * Limpia encabezados (BOM/espacios).
   * Convierte a categórica columnas puntuales si existen (`"MS SubClass"`, `"Mo Sold"`).
   * Asegura objetivo numérico y **elimina filas con NA en la `target`**.

3. **Codificaciones específicas**

   * **Calidad (QUAL_ORD)**: mapea `Po/Fa/TA/Gd/Ex` a **0..4**; valores raros o “No aplica” → **-1**.
   * **Utilities**: mapea a ordinal **0..3**; valores no mapeables → **-1**.
   * **One-hot “horneado”** para el resto de categóricas (todas las que no sean quality ni Utilities).
   * Guarda cuáles columnas quedaron dummificadas.

4. **Selección de features y preprocesamiento**

   * Infere **numéricas** y **categóricas** con `infer_feature_types` (respetando `cfg.drop_cols`).
   * Construye **preprocesador** para numéricas con `build_preprocessor(...)`.

5. **Modelo con transformación logarítmica del target**

   * Envuelve `XGBRegressor(**cfg.xgb_params)` en `TransformedTargetRegressor` (`log1p` / `expm1`).
   * Split train/test con `cfg.test_size` y `cfg.random_state`.

6. **Entrenamiento, métricas y guardado**

   * Calcula reporte de regresión (`regression_report`) para train y test.
   * Añade **skew/kurtosis de residuales**.
   * Guarda:

     * `model_xgb.joblib` (pipeline completo)
     * `booster.json` (si es posible)
     * `metrics.json`
     * `meta.json` (config, listas de columnas, params usados, etc.)

---

## Requisitos

```bash
pip install xgboost scikit-learn numpy pandas joblib
```

Además, el proyecto debe proveer:

* `training/config.py` → clase `Config` con:

  * `target`, `drop_cols`, `numeric_cols`, `categorical_cols`, `xgb_params`, `test_size`, `random_state`
* `training/preprocess.py` → `infer_feature_types`, `build_preprocessor`, `QUAL_ORD`, `UTIL_TO_ORD`
* `training/metrics.py` → `regression_report`
* `optimization/remodel/compat_xgboost.py` → `patch_get_booster`

---

## Uso como script

Desde la raíz del repo:

```bash
python -m training.retrain_xgb_same_env \
  --csv data/processed/base_completa_sin_nulos.csv \
  --outdir models/xgb/ordinal_p2_1800
```

### Argumentos

* `--csv` (obligatorio): ruta al CSV limpio.
* `--outdir` (opcional): carpeta de salida (default `models/xgb/ordinal_p2_1800`).

---

## Uso como función (importable)

```python
from training.retrain_xgb_same_env import retrain_xgb
from training.config import Config

cfg = Config()  # opcional: puedes modificar cfg.xgb_params antes
artefacts = retrain_xgb(
    csv_path="data/processed/base_completa_sin_nulos.csv",
    outdir="models/xgb/ordinal_p2_1800",
    cfg=cfg,
    test_size=None,       # usa cfg.test_size si None
    random_state=None,    # usa cfg.random_state si None
    verbose=True,
)

print(artefacts["paths"])
# {'model': '.../model_xgb.joblib', 'booster': '.../booster.json', 'metrics': '.../metrics.json', 'meta': '.../meta.json'}
```

---

## Salidas

En `--outdir`:

* `model_xgb.joblib` → **Pipeline** (`pre` + `TransformedTargetRegressor(XGB)`).
* `booster.json` → booster del XGB (si el entorno lo permite).
* `metrics.json` → métricas `train/test` + `residual_skew/kurtosis` + `log_target=True`.
* `meta.json` → target, columnas drop, listas de numéricas y categóricas, `xgb_params`, columnas dummificadas, etc.


Grafico.py

# Gráficos de Evolución de Métricas (Bayes Search)

Este script toma el archivo `all_iterations.csv` generado durante la búsqueda bayesiana de hiperparámetros y **genera gráficos de progreso** de las métricas por iteración.

## ¿Qué hace el script?

1. **Carga los resultados** desde:

   ```
   models/xgb_bayes_search/all_iterations.csv
   ```

   (cada fila representa una iteración de la búsqueda).

2. **Filtra** solo las primeras 75 iteraciones:

   ```python
   df = df[df["iteration"] <= 75]
   ```

3. **Genera tres gráficos**:

   * Evolución de **R²**
   * Evolución del **RMSE**
   * Evolución del **MAPE**

4. **Dibuja la media** de cada métrica como línea punteada para comparar.

5. **Guarda los gráficos** en:

   ```
   models/xgb_bayes_search/plots/
   ```

---

## Requisitos

```bash
pip install pandas matplotlib
```

---

## Ejecución

Solo corre el script:

```bash
python generar_graficos.py
```

(al nombre que le pongas)

Si todo funciona bien, verás mensajes como:

```
✅ Gráfico guardado: models/xgb_bayes_search/plots/r2_iter_75.png
✅ Gráfico guardado: models/xgb_bayes_search/plots/rmse_iter_75.png
✅ Gráfico guardado: models/xgb_bayes_search/plots/mape_iter_75.png
```

---

## Resultado final

En la carpeta `plots/` tendrás imágenes como:

| Archivo            | Métrica | Interpretación                                         |
| ------------------ | ------- | ------------------------------------------------------ |
| `r2_iter_75.png`   | R²      | Mide calidad del ajuste (más alto = mejor).            |
| `rmse_iter_75.png` | RMSE    | Error medio en unidades del precio (más bajo = mejor). |
| `mape_iter_75.png` | MAPE    | Error porcentual medio (más bajo = mejor).             |


--
##  Uso y referencias a la IA
El código fue echo con ayuda de la IA. A continuación links utilizados para la elaboración de partes del código:

* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68f15667-a570-832e-a6dd-232c3b7b75cb?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68fa7f1e-dc9c-8329-b102-4779c19ece59?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68e5109e-b2b4-832b-8a8a-24b882354dd1?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68f4a28b-bd48-832e-b6ec-421c75ceb765?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68e5109e-b2b4-832b-8a8a-24b882354dd1?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/share/68fc07e9-e768-8008-9e11-0dead2fd7fc9
* https://chatgpt.com/share/68fc084a-71dc-8008-8b9c-3c0ca6b18326
* https://chatgpt.com/share/68fc08b0-708c-8008-b337-4af0640732e0
