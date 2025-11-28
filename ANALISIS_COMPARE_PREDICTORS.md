# ANÁLISIS: `scripts/compare_predictors.py` - Integración con run_opt.py

**Documento de análisis y guía de integración**

---

## 📋 ¿QUÉ HACE `compare_predictors.py`?

El script compara **2 modelos de predicción de precio** sobre una casa remodelada:

1. **XGBoost** (modelo productivo)
2. **Regresión Linear** (modelo base del equipo anterior)

La idea es validar que XGBoost predice mejor que la regresión antigua.

---

## 🔍 ANÁLISIS LÍNEA POR LÍNEA

### Modos de Operación

```python
# Modo 1: Resolver el MIP + Comparar
python scripts/compare_predictors.py --pid 526301100 --budget 80000 --reg-model models/baseline.joblib

# Modo 2: Usar casa remodelada precalculada + Comparar (sin resolver MIP)
python scripts/compare_predictors.py --pid 526301100 --xin-csv X_input_after_opt.csv --reg-model models/baseline.joblib
```

**Diferencia:**
- Modo 1: Resuelve la optimización (lento) y luego compara
- Modo 2: Carga casa ya optimizada de un CSV (rápido)

---

### Step 1: Cargar Insumos (líneas 35-41)

```python
base = get_base_house(pid)           # Lee base_house.csv con PID
base_row = base.row
ct = costs.CostTables()              # Tablas de costos (cocina, exterior, etc.)
bundle = XGBBundle()                 # Carga modelo XGBoost productivo

# Precio base: intenta CSV de regresión, sino usa XGB
precio_base = _precio_base_from_csv(pid)
if precio_base is None:
    X_base = build_base_input_row(bundle, base_row)
    precio_base = float(bundle.predict(X_base).iloc[0])  # XGB predice
```

**Lo que pasa:**
- Lee la casa base del CSV
- Obtiene el precio original (SalePrice_Present o SalePrice del CSV)
- Si no lo encuentra, usa XGBoost para predecir el precio actual

---

### Step 2: Decisión - Resolver MIP o Cargar CSV (líneas 53-74)

```python
if xin_csv:
    # MODO 2: Cargar casa óptima del CSV (saltea el MIP)
    X_in = _load_x_input_from_csv(xin_csv)
    m = None
else:
    # MODO 1: Resolver el MIP (caro computacionalmente)
    m = build_mip_embed(base_row, budget, ct, bundle, base_price=precio_base)
    m.Params.TimeLimit = time_limit
    m.optimize()  # ← Aquí se resuelve la optimización
    X_in = rebuild_embed_input_df(m, m._X_base_numeric)  # Reconstruir fila óptima
```

**La lógica:**
- Si pasas `--xin-csv`: saltas resolver el MIP (rápido)
- Si pasas `--budget`: resuelves el MIP (completo pero lento)

---

### Step 3: Predicciones (líneas 76-140)

```python
# 1. Precio XGBoost
precio_xgb = float(bundle.predict(X_in).iloc[0])

# 2. Cargar modelo de regresión
reg_model = joblib.load(reg_model_path)

# 3. Preparar fila para regresión (alineación de columnas)
reg_cols = list(getattr(reg_model, "feature_names_in_", []))
# ... código complejo para alinear nombres de columnas ...
X_reg = pd.DataFrame([new_row], columns=reg_cols)

# 4. Precio Regresión
reg_pred = reg_model.predict(X_reg)
precio_reg = float(np.exp(reg_pred[0]))  # ← Deslogaritmo
```

**Lo que pasa:**

| Paso | Acción | Resultado |
|------|--------|-----------|
| 1 | XGBoost predice sobre casa remodelada | `precio_xgb` |
| 2 | Carga modelo regresión del joblib | `reg_model` |
| 3 | Alinea columnas (XGB != Regresión) | `X_reg` con columnas correctas |
| 4 | Regresión predice (modelo entrenado en log) | `precio_reg` (exponenciado) |

---

### Step 4: Chequeos de Validez (líneas 142-155)

```python
# Verifica que la solución sea factible
if m is not None:
    max_slack = max(abs(c.Slack) for c in m.getConstrs())
    if max_slack > 1e-3:
        raise RuntimeError(f"Solución infactible!")
    
    # Verifica que el precio no baje
    y_price_opt = float(m._y_price_var.X)
    if y_price_opt < precio_base - 1e-3:
        raise RuntimeError(f"Precio baja el original!")
```

**Seguridad:**
- Si la solución tiene restricciones violadas → ERROR
- Si la casa "baja de precio" → ERROR (no tiene sentido una remodelación que baja precio)

---

### Step 5: Comparación e Impresión (líneas 157-170)

```python
# Calcula diferencia porcentual
uplift_vs_reg = (precio_xgb - precio_reg) / precio_reg * 100

# Imprime resultados
print(f"Precio base (XGB):           ${precio_base:,.0f}")
print(f"Precio remodelado XGB:       ${precio_xgb:,.0f}")
print(f"Precio remodelado Regresión: ${precio_reg:,.0f}")
print(f"Diferencia % (XGB vs Reg):   {uplift_vs_reg:.2f}%")
```

**Output ejemplo:**
```
Precio base (XGB):           $195,000
Precio remodelado XGB:       $215,000
Precio remodelado Regresión: $208,500
Diferencia % (XGB vs Reg):   3.12%
```

---

## 🔗 FLUJO ACTUAL

```
run_opt.py (main)
    │
    ├─→ Resuelve MIP
    │   └─→ Obtiene casa_remodelada
    │
    └─→ FIN (no compara predictores)


compare_predictors.py (script separado)
    │
    ├─→ Resuelve MIP (o carga CSV)
    │   └─→ Obtiene casa_remodelada
    │
    ├─→ XGBoost(casa_remodelada) → precio_xgb
    ├─→ Regresión(casa_remodelada) → precio_reg
    │
    └─→ Compara e imprime diferencia
```

**Problema:** Son **2 scripts separados** que resuelven el MIP de forma independiente

---

## 💡 OPCIÓN 1: Integración Directa en run_opt.py (RECOMENDADO)

Agregar la comparación de predictores **al final de run_opt.py**, justo después de imprimir resultados.

### Modificación Propuesta

**En `run_opt.py` después de la línea 1387 (FIN RESULTADOS), agregar:**

```python
# ============================================
# COMPARAR PREDICTORES (XGBoost vs Regresión)
# ============================================

try:
    import joblib
    from optimization.remodel.run_opt import rebuild_embed_input_df
    
    print("\n" + "="*80)
    print("COMPARACIÓN: XGBoost vs Regresión Base")
    print("="*80)
    
    # 1. Precio base
    precio_base = float(m._y_base_var.X) if hasattr(m, '_y_base_var') else None
    if precio_base is None:
        X_base = build_base_input_row(bundle, base_row)
        precio_base = float(bundle.predict(X_base).iloc[0])
    
    # 2. Casa remodelada ya resuelta
    X_opt = rebuild_embed_input_df(m, m._X_base_numeric)
    
    # 3. XGBoost prediction
    precio_xgb = float(bundle.predict(X_opt).iloc[0])
    
    # 4. Regresión prediction (si existe el modelo)
    try:
        # Buscar modelo de regresión
        reg_paths = [
            "models/reg/regresion_base.joblib",
            "models/baseline.joblib",
            "models/regresion.joblib"
        ]
        reg_model = None
        for path in reg_paths:
            if os.path.exists(path):
                reg_model = joblib.load(path)
                break
        
        if reg_model:
            # Alinear columnas con regresión
            reg_cols = list(getattr(reg_model, "feature_names_in_", []))
            if reg_cols:
                try:
                    df_reg = pd.read_csv("data/raw/df_final_regresion.csv")
                    df_reg.columns = [c.replace("\ufeff", "").strip() for c in df_reg.columns]
                    row_reg = df_reg.loc[df_reg["PID"] == args.pid].iloc[0]
                except:
                    row_reg = pd.Series({c: base_row.get(c, np.nan) for c in reg_cols})
                
                # Construir X_reg alineada
                new_row = {}
                for c in reg_cols:
                    if c in X_opt.columns:
                        new_row[c] = float(X_opt[c].iloc[0])
                    else:
                        new_row[c] = row_reg.get(c, np.nan)
                X_reg = pd.DataFrame([new_row], columns=reg_cols)
                
                # Predecir con regresión
                reg_pred = float(reg_model.predict(X_reg)[0])
                precio_reg = np.exp(reg_pred) if reg_pred < 50 else reg_pred  # Deslogaritmo si necesario
                
                # Comparación
                diff_absoluta = precio_xgb - precio_reg
                diff_porcentaje = (diff_absoluta / precio_reg * 100) if precio_reg > 0 else np.nan
                uplift_xgb = (precio_xgb - precio_base) / precio_base * 100
                uplift_reg = (precio_reg - precio_base) / precio_base * 100
                
                print(f"\n💰 COMPARACIÓN DE PREDICTORES:")
                print(f"  Precio base (actual):        ${precio_base:,.0f}")
                print(f"  Precio remodelado (XGBoost): ${precio_xgb:,.0f}  (+{uplift_xgb:.1f}%)")
                print(f"  Precio remodelado (Regresión): ${precio_reg:,.0f}  (+{uplift_reg:.1f}%)")
                print(f"\n  📊 Diferencia XGBoost vs Regresión:")
                print(f"     Absoluta: ${diff_absoluta:,.0f}")
                print(f"     Porcentaje: {diff_porcentaje:.2f}%")
                
                if diff_porcentaje > 0:
                    print(f"\n  ✅ XGBoost SUPERA a Regresión por {diff_porcentaje:.2f}%")
                else:
                    print(f"\n  ⚠️  Regresión SUPERA a XGBoost por {abs(diff_porcentaje):.2f}%")
        else:
            print("\n⚠️  No se encontró modelo de regresión. Saltando comparación.")
    
    except Exception as e:
        print(f"\n⚠️  Error al comparar predictores: {e}")

except Exception as e:
    print(f"\n⚠️  Error en sección de comparación: {e}")
```

---

## 💡 OPCIÓN 2: Mantener Script Separado (Actual)

Ejecutar en 2 pasos:

```bash
# Paso 1: Optimizar
python3 optimization/remodel/run_opt.py --pid 526301100 --budget 80000

# Paso 2: Comparar predictores
python3 scripts/compare_predictors.py --pid 526301100 --budget 80000 --reg-model models/reg/regresion_base.joblib
```

**Ventajas:**
- Scripts modulares
- No aumenta tamaño de run_opt.py

**Desventajas:**
- Resuelve el MIP 2 veces (lento)
- El usuario debe ejecutar 2 comandos

---

## 💡 OPCIÓN 3: Usar compare_predictors con CSV (MEJOR SI QUIERES RAPIDEZ)

```bash
# Paso 1: Optimizar y guardar X_opt a CSV
python3 optimization/remodel/run_opt.py --pid 526301100 --budget 80000 --output-csv X_input_after_opt.csv

# Paso 2: Comparar (sin resolver MIP otra vez)
python3 scripts/compare_predictors.py --pid 526301100 --xin-csv X_input_after_opt.csv --reg-model models/reg/regresion_base.joblib
```

**Ventajas:**
- Resuelve MIP 1 sola vez
- Rápido (no replica cálculos)
- Modular

**Desventajas:**
- Requiere guardar CSV intermedio

---

## 🎯 MI RECOMENDACIÓN

**OPCIÓN 1: Integración Directa en run_opt.py**

**Razones:**
1. ✅ El usuario ve TODO en 1 ejecución
2. ✅ No replica cálculos (MIP se resuelve 1 sola vez)
3. ✅ Output integral: optimización + comparación de predictores
4. ✅ Listo para presentar en Capstone

**Código a agregar:**
- Aproximadamente 80-100 líneas al final de run_opt.py
- Reutiliza funciones ya existentes (no código nuevo)
- Maneja excepciones si modelo regresión no existe

---

## 📊 OUTPUT ESPERADO AL FINAL DE run_opt.py

```
================================================================================
            FIN RESULTADOS DE LA OPTIMIZACIÓN
================================================================================

================================================================================
COMPARACIÓN: XGBoost vs Regresión Base
================================================================================

💰 COMPARACIÓN DE PREDICTORES:
  Precio base (actual):        $195,000
  Precio remodelado (XGBoost): $215,000  (+10.3%)
  Precio remodelado (Regresión): $208,500  (+6.9%)

  📊 Diferencia XGBoost vs Regresión:
     Absoluta: $6,500
     Porcentaje: 3.12%

  ✅ XGBoost SUPERA a Regresión por 3.12%
```

---

## 🔧 ARCHIVOS A MODIFICAR

### 1. `optimization/remodel/run_opt.py`
- Agregar import al inicio: `import joblib, os`
- Agregar sección de comparación al final (antes del `if __name__ == "__main__"`)

### 2. Opcional: `optimization/remodel/run_opt.py` (parámetro)
- Agregar `--skip-compare` para saltear comparación si quieres rapidez

### 3. Opcional: `scripts/compare_predictors.py`
- Ahora sería "respaldo" para comparaciones puntuales

---

## 📋 CHECKLIST

- [ ] Revisar dónde está el modelo de regresión en tu estructura
- [ ] Confirmar path a `models/reg/regresion_base.joblib` (o similar)
- [ ] Agregar imports necesarios a run_opt.py
- [ ] Agregar sección de comparación
- [ ] Validar que `build_base_input_row` funciona correctamente
- [ ] Probar con 1 house (ej. PID 526301100)
- [ ] Verificar que output se ve claro y correcto

---

**¿Quieres que implemente la Opción 1 ahora?**

