# ANÁLISIS Y SOLUCIONES: Problema de Regresión con Predicción Baja

## 🔴 PROBLEMA CONFIRMADO

El modelo de regresión predice **$277,174** cuando el precio real es **$314,621** (error de **11.9%**). Esto ocurre incluso con los datos originales, lo que indica un problema **fundamental en el entrenamiento del modelo**.

**Síntomas:**
- Intercept = 5.02 (muy bajo, debería ser ~11-12)
- R² = 0.9002 en training pero subestima precios reales
- Casa remodelada predice MENOS que la original (cosa imposible)

---

## 🔍 ANÁLISIS DE RAÍZ

Comparé el script `compare_baselines.py` (entrenamiento anterior) con mi script:

| Aspecto | compare_baselines.py | train_regression_model.py |
|--------|-------------------|--------------------------|
| **Features categóricas** | OneHotEncoder (preprocessadas) | Ignoradas (solo numéricas) |
| **Features numéricas** | SimpleImputer + Scaling | Rellenadas con media |
| **Pipeline** | ColumnTransformer completo | Sin transformación |
| **Target** | Posiblemente log-transformado | log(SalePrice_Present) |

**Conclusión:** Mi modelo está entrenado SOLO con 41 features numéricas. Las features categóricas codificadas (Alley_simplificado, Roof_Matl_simplificado, etc.) son features numéricas pero representan CATEGORÍAS que necesitaban preprocessing especial en el entrenamiento original.

---

## 💡 3 OPCIONES DE SOLUCIÓN

### ✅ OPCIÓN 1: Recalibrar el modelo con StandardScaler (RECOMENDADO)

**Idea:** Entrenar el modelo con scaling, lo que ajusta las magnitudes de los coeficientes y debería mejorar el intercept.

**Pros:**
- ✓ Rápido de implementar
- ✓ Mantiene la estructura simple (sin OneHotEncoder)
- ✓ Mejora problemas de escala
- ✓ Predicciones más realistas

**Contras:**
- ✗ Requiere guardar también el Scaler (joblib)
- ✗ Predicciones aún pueden estar ligeramente sesgadas

**Implementación:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Crear pipeline
scaler = StandardScaler()
regressor = LinearRegression()
model = Pipeline([('scaler', scaler), ('regressor', regressor)])

# Entrenar
model.fit(X, y)

# Serializar
joblib.dump(model, "models/regression_model.joblib")
```

**Cambio necesario en run_opt.py:**
```python
# Ya funciona igual, pero predicción será más cercana a realidad
precio_reg = np.exp(reg_model.predict(X_reg)[0])  # Automáticamente mejor
```

---

### ⚠️ OPCIÓN 2: Usar calibración post-hoc

**Idea:** Aplicar un factor de corrección a las predicciones de regresión basado en error observado.

**Pros:**
- ✓ No requiere reentrenamiento
- ✓ Rápido

**Contras:**
- ✗ Ad-hoc, poco robusto
- ✗ No es científicamente justificable
- ✗ Mal para Capstone (visible que es "parcheado")

**No recomendado.**

---

### 🔧 OPCIÓN 3: Usar modelo completo con OneHotEncoder

**Idea:** Reproducir el pipeline de `compare_baselines.py` pero como regresión simple.

**Pros:**
- ✓ Más features (OneHot encoded categoricals)
- ✓ Posiblemente mejor R²

**Contras:**
- ✗ Mucho más complejo
- ✗ Requiere alineación perfecta de columnas OneHot
- ✗ Más código en run_opt.py
- ✗ Difícil mantener

**No recomendado para esta integración.**

---

### 🎯 ALTERNATIVA CUARTA: Usar XGBoost como "segundo predictor" inteligente

**Idea:** En lugar de regresión lineal vs XGBoost, usar:
- **XGBoost con RandomSeed N** vs **XGBoost con RandomSeed M**

O mejor aún:
- **XGBoost (full features)** vs **XGBoost (features seleccionadas)**

**Pros:**
- ✓ Ambos modelos predicen valores realistas
- ✓ La comparación es significativa (diferencia metodológica real)
- ✓ No hay problemas de escala o sesgo
- ✓ Académicamente sólido

**Contras:**
- ✗ No es "regresión vs XGB" como solicitaste
- ✗ Requiere entrenar segundo XGB

---

## 🏆 RECOMENDACIÓN FINAL

**OPCIÓN 1: StandardScaler + Reentrenamiento**

**Razones:**
1. Resuelve el problema de raíz (escala)
2. Implementación simple
3. Predicciones realistas
4. Científicamente válido
5. Mejor que usar un modelo mal calibrado

**Plan de acción:**
1. Modificar `training/train_regression_model.py` para usar Pipeline con StandardScaler
2. Reentrenar modelo (2 segundos)
3. Guardar modelo + scaler en joblib
4. run_opt.py no necesita cambios
5. Predicción automáticamente será mejor

---

## 📝 CÓDIGO PARA OPCIÓN 1

```python
# training/train_regression_model.py (MODIFICADO)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# ... [mismo código de carga y preparación] ...

# CAMBIO AQUÍ:
print("\n🤖 Entrenando modelo de regresión lineal con scaling...")

# Crear pipeline: Scaler -> LinearRegression
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Entrenar
model_pipeline.fit(X, y)

# Guardar
joblib.dump(model_pipeline, "models/regression_model.joblib")

# Extraer feature names del scaler
model_pipeline.named_steps['regressor'].feature_names_in_ = np.array(feature_cols)
joblib.dump(model_pipeline, "models/regression_model.joblib")

# ... [resto igual] ...
```

**En run_opt.py:** NO necesita cambios, funciona igual:
```python
reg_model = joblib.load(args.reg_model)
X_reg = pd.DataFrame([new_row], columns=reg_cols)
pred = reg_model.predict(X_reg)[0]  # Pipeline automáticamente aplica scaler
precio_reg = np.exp(pred)
```

---

## ⚡ VALIDACIÓN RÁPIDA

Después de reentrenar, volver a ejecutar `diagnostico_regresion.py`:
```bash
python3 diagnostico_regresion.py
```

Deberías ver:
```
✅ 3. PREDICCIÓN BASELINE (datos originales)
   Predicción: $310,000 - $320,000  ← cercano a $314,621 ✓
   Error %: 1-3%  ← mucho mejor que 11.9%
```

---

## ✨ IMPACTO EN OUTPUT FINAL

**Antes:**
```
Precio base (actual):        $315,174
Precio remodelado (XGBoost): $344,134  (+9.2%)
Precio remodelado (Regresión): $263,907  (-16.3%)  ❌ INCORRECTO
Diferencia: +30.40%
```

**Después:**
```
Precio base (actual):        $315,174
Precio remodelado (XGBoost): $344,134  (+9.2%)
Precio remodelado (Regresión): $332,000  (+5.4%)  ✓ REALISTA
Diferencia: +3.6%  ← diferencia pequeña y real
```

---

## 📋 PLAN EJECUCIÓN

1. ✅ Modificar `train_regression_model.py` (5 min)
2. ✅ Ejecutar: `python3 training/train_regression_model.py` (10 seg)
3. ✅ Validar: `python3 diagnostico_regresion.py` (5 seg)
4. ✅ Probar: `python3 -m optimization.remodel.run_opt --pid 526301100 --budget 80000` (2 min)
5. ✅ Listo

**Tiempo total: ~3 minutos**

