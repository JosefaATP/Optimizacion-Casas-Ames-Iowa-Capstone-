# Proyecto Capstone – Remodelación Óptima Ames (README detallado para colaboradora)

> Objetivo: maximizar la utilidad de una remodelación (`precio_predicho - costos`) bajo un modelo **XGBoost embebido en Gurobi**.  
> El modelo estima el precio de venta de una casa remodelada, y Gurobi elige qué cambios realizar (dormitorios, techos, materiales, etc.) bajo restricciones constructivas y presupuesto.

---

## 🧩 Flujo general del proyecto

1. **Carga de la casa base**  
   Archivo: `optimization/remodel/io.py`  
   Función: `get_base_house(pid)`  
   → Devuelve la fila (`pandas.Series`) con los datos originales de la casa.

2. **Predicción de precio con ML**  
   Archivo: `optimization/remodel/xgb_predictor.py`  
   Clase: `XGBBundle`  
   → Carga un `pipeline sklearn` (transformaciones + modelo XGBoost).  
   → Devuelve el precio predicho (`predict()`) y permite embebido con Gurobi (`pipe_for_gurobi()`).

3. **Optimización con Gurobi**  
   Archivo: `optimization/remodel/gurobi_model.py`  
   Función: `build_mip_embed(base_row, budget, ct, bundle)`  
   → Crea las variables de decisión (`x`), agrega restricciones y costos.  
   → Usa `gurobi_ml` para conectar el pipeline con Gurobi.  
   → Objetivo: `Maximize( precio_predicho - costos_totales )`.

4. **Ejecución del modelo**  
   Archivo: `optimization/remodel/run_opt.py`  
   → Llama a `build_mip_embed()`, corre la optimización y muestra:  
   - Sensibilidades “DEBUG … → precio”.  
   - Log de Gurobi.  
   - Resumen de cambios, costos y precio remodelado.

---

## 🗂️ Estructura de carpetas y qué puede tocar

| Carpeta / Archivo | Descripción | ¿Tocar? |
|--------------------|--------------|----------|
| `optimization/remodel/gurobi_model.py` | Donde se agregan restricciones y costos. | ✅ Sí |
| `optimization/remodel/costs.py` | Donde se definen los valores de costos unitarios y funciones auxiliares. | ✅ Sí |
| `optimization/remodel/features.py` | Donde se definen las variables modificables (`MODIFIABLE`). | ✅ Sí |
| `optimization/remodel/run_opt.py` | Donde se hacen los prints de debug y lectura de resultados. | ✅ Sí |
| `optimization/remodel/xgb_predictor.py` | Pipeline de ML. **No editar.** | ⚠️ No |
| `optimization/remodel/compat_sklearn.py` y `compat_xgboost.py` | Parchean el pipeline para `gurobi_ml`. | ⚠️ No |
| `models/xgb/...` | Archivos del modelo entrenado (`joblib`, `booster.json`). | ⚠️ No |

---

## 🧠 Lógica básica del modelo

1. **Variables de decisión**
   - Se definen en `features.py` → lista `MODIFIABLE`.
   - Ejemplo:
     ```python
     Feature("Full Bath", lb=1, ub=4, vartype="I"),
     Feature("Garage Cars", lb=0, ub=4, vartype="I"),
     Feature("Roof Style_is_Gable", lb=0, ub=1, vartype="B"),
     Feature("Exter Qual", lb=0, ub=4, vartype="I"),
     ```
   - Los nombres deben coincidir con los usados por el pipeline XGBoost.

2. **Inyección al pipeline**
   - En `build_mip_embed`, se arma un DataFrame `X_input` con los valores (constantes o variables) que el modelo espera:
     ```python
     X_input = pd.DataFrame([row_vals], columns=feature_order, dtype=object)
     ```
   - Si el modelo tiene variables *one-hot* (por ejemplo `"Roof Style_Gable"`), se inyectan directamente las binarias Gurobi en esas columnas.

3. **Predicción embebida**
   - Se usa `gurobi_ml` para agregar restricciones del modelo ML dentro del MILP:
     ```python
     _add_sklearn(m, bundle.pipe_for_gurobi(), X_input, [y_log])
     ```
   - Si el modelo está en log, se usa una restricción PWL para aplicar `exp()`.

4. **Función objetivo**
   ```python
   maximize: y_price - total_cost


💰 Cómo se calculan los costos

Archivo: optimization/remodel/costs.py → Clase CostTables

Cada tipo de cambio tiene un costo lineal asociado que se suma a lin_cost dentro del modelo:

lin_cost = gp.LinExpr(ct.project_fixed)
lin_cost += pos(x["Full Bath"] - base_vals["Full Bath"]) * ct.add_bathroom
lin_cost += pos(x["Bedroom AbvGr"] - base_vals["Bedroom AbvGr"]) * ct.add_bedroom
lin_cost += pos(x["Garage Cars"] - base_vals["Garage Cars"]) * ct.garage_per_car

Costos ya definidos:

Dormitorios, baños, garaje, deck, sótano, cocina.

Roof Style / Roof Matl: costos fijos por estilo o USD/ft² por material.

Exterior 1st / 2nd: costos por demolición + reconstrucción + mejoras de calidad.

Exter Qual / Exter Cond: costo lineal por nivel (upgrade-only).

Mas Vnr Type / Area: costo por ft² de mampostería según tipo.

🏗️ Restricciones incluidas por defecto
Código	Descripción
R1	1st Flr SF ≥ 2nd Flr SF
R2	Gr Liv Area ≤ Lot Area
R3	1st Flr SF ≥ Total Bsmt SF
R4	Full Bath + Half Bath ≤ Bedrooms
R5	Mínimos (Full Bath ≥1, Bedroom ≥1, Kitchen ≥1)
R7	Gr Liv Area = 1st + 2nd (+ LowQual)
R8	TotRms AbvGrd = Bedroom + Kitchen + Other

Además de las reglas especiales:

Kitchen Qual: upgrades TA o EX mediante binarios (delta_KitchenQual_TA, delta_KitchenQual_EX).

Utilities: upgrade-only (ELO → AllPub).

Roof: elegir un estilo y un material, con compatibilidad prohibida entre algunos.

Exterior: 1er y 2do frente, elegibilidad por calidad, costos de demo/material.

Masonry (Mas Vnr): tipo “elige uno”, upgrade-only y costo = ft² × tarifa.

🧱 Ejemplo de patrón (categoría tipo “elige 1”)
# Ejemplo: Roof Style
style_names = ["Flat", "Gable", "Hip", "Shed"]
s_bin = {nm: x[f"roof_style_is_{nm}"] for nm in style_names}

# elegir solo uno
m.addConstr(gp.quicksum(s_bin.values()) == 1)

# inyectar al modelo
for nm in style_names:
    col = f"Roof Style_{nm}"
    if col in X_input.columns:
        X_input.loc[0, col] = s_bin[nm]

# costos (solo si cambias)
base_style = str(base_row.get("Roof Style"))
for nm, vb in s_bin.items():
    if nm != base_style:
        lin_cost += ct.roof_style_cost(nm) * vb

🔢 Patrón (calidad ordinal “upgrade-only”)
base = _q_to_ord(base_row.get("Exter Qual", "TA"))  # TA = 2
m.addConstr(x["Exter Qual"] >= base)
lin_cost += ct.exter_qual_upgrade_per_level * (x["Exter Qual"] - base)


Usamos esta lógica para:

Kitchen Qual

Exter Qual

Exter Cond

(y cualquier otra calidad o condición ordinal)

🧮 Patrón general para costos y restricciones

Definir variable en features.py → MODIFIABLE.

Agregar lógica en gurobi_model.py (restricción + costo).

Actualizar prints en run_opt.py si quieres verla en la salida.

Agregar tarifas en costs.py.

Verificar impacto con --budget amplio para observar sensibilidad.

🧾 Ejecución típica
python -m optimization.remodel.run_opt --pid 528138060 --budget 40000000


Ejemplo de salida:

DEBUG Kitchen Qual → precio: [...]
DEBUG Roof Style → precio: [...]
DEBUG Exterior 1st → precio: [...]
DEBUG Mas Vnr Type → precio: [...]

Aumento de Utilidad: $233,703
precio casa base: $289,209
precio casa remodelada: $543,912
costos totales de remodelación: $21,000

Cambios hechos en la casa:
- Full Bath: 2 → 3 (costo $12,000)
- Garage Cars: 2 → 3 (costo $9,000)
Mas Vnr Type base→nuevo: BrkFace → BrkFace

⚙️ Qué debe evitar modificar la colaboradora

❌ No tocar:

bundle.pipe_for_gurobi() o su estructura.

Los nombres de columnas OHE del modelo (Roof Style_*, Exterior 1st_*, etc.).

La definición del pipeline en xgb_predictor.py.

Las funciones de compatibilidad (compat_sklearn.py, compat_xgboost.py).

⚠️ Sí puede tocar:

Agregar restricciones o costos nuevos en gurobi_model.py.

Agregar costos unitarios en costs.py.

Definir nuevas variables en features.py.

Agregar prints o debugs en run_opt.py.

✅ Checklist para agregar una restricción nueva

¿La variable existe en el pipeline?

Si es numérica → MODIFIABLE y costo lineal.

Si es ordinal → restricción >= base y costo por nivel.

Si es categórica → binarios, selección única e inyección en dummies.

Agregar costo a lin_cost.

Agregar prints de debug (opcional).

Correr con presupuesto amplio (--budget 40000000) para validar.

Revisar que:

No haya errores de KeyError en nombres de columnas.

El precio base y remodelado se impriman correctamente.

Los costos totales sean razonables.

🧩 Recordatorio sobre proxies de área

Roof → Gr Liv Area

Exterior → 0.8 * Gr Liv Area

Mas Vnr → Mas Vnr Area (ya viene en los datos)

Estos proxies se usan para convertir costos $/ft² a valores totales.

🧠 Consejos finales

Nombrar todas las restricciones (name="R_"...) para depurar.

Evitar loops grandes fuera del bloque principal (cada iteración ralentiza Gurobi).

Si una categoría no aparece en el modelo, no inyectarla (revisar con bundle.feature_names_in()).

Los warnings de Setting an item of incompatible dtype no son errores, solo conviene usar astype("object") antes.

📘 Ejemplo de bloque completo nuevo (para inspirarse)
# ================== (NEW BLOCK: ejemplo) ==================
NEW_TYPES = ["A", "B", "C"]
nt = {nm: x[f"new_is_{nm}"] for nm in NEW_TYPES if f"new_is_{nm}" in x}

# elegir 1
m.addConstr(gp.quicksum(nt.values()) == 1, name="NEW_pick_one")

# inyectar al modelo
for nm in NEW_TYPES:
    col = f"New Var_{nm}"
    if col in X_input.columns:
        X_input.loc[0, col] = nt[nm]

# costo si cambias
base_new = str(base_row.get("New Var", "A"))
for nm in NEW_TYPES:
    if nm != base_new:
        lin_cost += 5000 * nt[nm]
# ================== FIN (NEW BLOCK) ==================

🚀 Conclusión

El flujo completo es:

CASA BASE (csv) → pipeline sklearn (pre + XGB) → gurobi_model.build_mip_embed()
     ↳ crea variables
     ↳ aplica restricciones y costos
     ↳ predice precio dentro de Gurobi
 → Maximiza (precio_predicho - costo_total)
 → Devuelve plan óptimo + prints de cambios


Si se siguen estos pasos y se respeta la estructura actual, se pueden agregar todas las nuevas restricciones sin romper el modelo.