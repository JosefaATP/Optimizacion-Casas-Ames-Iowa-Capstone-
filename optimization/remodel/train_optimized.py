#!/usr/bin/env python3
"""
Script para entrenar XGBoost con los parámetros optimizados.
Estos parámetros fueron generados por un optimizador (Optuna/Hyperopt/similar).

Parámetros optimizados:
- n_estimators: 2843
- learning_rate: 0.042345759919321546
- max_depth: 3
- min_child_weight: 4.0
- gamma: 0.050035425161215466
- subsample: 0.5205384818739047
- colsample_bytree: 0.5693415844980262
- reg_lambda: 3.827629507754387
- reg_alpha: 0.05963017089026609
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

# Asegúrate de tener acceso a los módulos del proyecto
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.remodel.config import PATHS

def main():
    parser = argparse.ArgumentParser(
        description="Entrenar XGBoost con parámetros optimizados"
    )
    parser.add_argument(
        "--csv",
        default=str(PATHS.base_csv),
        help=f"CSV de entrada (default: {PATHS.base_csv})"
    )
    parser.add_argument(
        "--target",
        default="SalePrice_Present",
        help="Nombre de columna target"
    )
    parser.add_argument(
        "--outdir",
        default="models/xgb/optimized_params_2843",
        help="Directorio de salida"
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Aplicar log1p al target"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Paciencia para early stopping (rondas sin mejora)"
    )
    args = parser.parse_args()

    # Crear directorio de salida
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 70)
    print("  ENTRENAMIENTO XGBOOST CON PARÁMETROS OPTIMIZADOS")
    print("=" * 70)
    print()

    # Cargar datos
    print(f"📂 Cargando datos desde: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✓ Datos cargados: {df.shape}")
    print()

    # Preparar target
    print(f"🎯 Target: {args.target}")
    if args.target not in df.columns:
        print(f"❌ Error: '{args.target}' no está en las columnas")
        return
    
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.dropna(subset=[args.target])
    print(f"✓ Target cargado ({len(df)} filas válidas)")
    print()

    # Preparar features
    drop_cols = ["PID", "Order", "SalePrice", "\ufeffOrder"]
    X = df.drop(
        columns=[args.target] + [c for c in drop_cols if c in df.columns],
        errors="ignore"
    )
    y = df[args.target].values
    
    print(f"📊 Features: {X.shape[1]}")
    print(f"   Primeras 5 columnas: {list(X.columns[:5])}")
    
    # Convertir columnas object a numéricas (one-hot encoding simplificado)
    print("🔄 Codificando variables categóricas...")
    from sklearn.preprocessing import LabelEncoder
    
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        print(f"   Codificando {col}...")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    print(f"   ✓ Codificadas {len(object_cols)} columnas")
    print()

    # Split
    print(f"✂️  Splitting (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print()

    # Preparar target si se solicita log
    if args.log_target:
        print("📈 Aplicando log1p al target...")
        y_train_fit = np.log1p(y_train)
        y_test_fit = np.log1p(y_test)
    else:
        y_train_fit = y_train
        y_test_fit = y_test

    # Hiperparámetros optimizados
    xgb_params = {
        "n_estimators": 2843,
        "learning_rate": 0.042345759919321546,
        "max_depth": 3,
        "min_child_weight": 4.0,
        "gamma": 0.050035425161215466,
        "subsample": 0.5205384818739047,
        "colsample_bytree": 0.5693415844980262,
        "reg_lambda": 3.827629507754387,
        "reg_alpha": 0.05963017089026609,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }

    print("🔧 Hiperparámetros optimizados:")
    for k, v in xgb_params.items():
        if isinstance(v, float):
            print(f"   {k:<20} = {v:.10f}")
        else:
            print(f"   {k:<20} = {v}")
    print()

    # Entrenar modelo
    print("🚀 Entrenando modelo XGBoost...")
    print(f"   (n_estimators={xgb_params['n_estimators']})")
    print()

    model = XGBRegressor(**xgb_params)

    # Entrenar sin early stopping (compatible con todas las versiones)
    model.fit(X_train, y_train_fit)

    best_iter = getattr(model, "best_iteration", None)
    print(f"✓ Entrenamiento completado")
    if best_iter is not None:
        print(f"  Best iteration: {best_iter} (de {xgb_params['n_estimators']})")
    print()

    # Predicciones
    print("📊 Evaluando modelo...")
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    if args.log_target:
        yhat_train = np.expm1(yhat_train)
        yhat_test = np.expm1(yhat_test)

    # Métricas
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Residuos
        residuals = pd.Series(y_true - y_pred)
        skew = float(residuals.skew())
        kurtosis = float(residuals.kurtosis())
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE_pct": mape,
            "R2": r2,
            "residual_skew": skew,
            "residual_kurtosis": kurtosis
        }

    metrics_train = calc_metrics(y_train, yhat_train)
    metrics_test = calc_metrics(y_test, yhat_test)

    print()
    print("╔" + "=" * 68 + "╗")
    print("║  RESULTADOS DEL ENTRENAMIENTO")
    print("╠" + "=" * 68 + "╣")
    
    print("║ TRAIN SET                                                              ║")
    print(f"║   RMSE:  ${metrics_train['RMSE']:>15,.0f}                                    ║")
    print(f"║   MAE:   ${metrics_train['MAE']:>15,.0f}                                    ║")
    print(f"║   MAPE:  {metrics_train['MAPE_pct']:>16.2f}%                                   ║")
    print(f"║   R²:    {metrics_train['R2']:>17.4f}                                   ║")
    
    print("║                                                                        ║")
    print("║ TEST SET                                                               ║")
    print(f"║   RMSE:  ${metrics_test['RMSE']:>15,.0f}                                    ║")
    print(f"║   MAE:   ${metrics_test['MAE']:>15,.0f}                                    ║")
    print(f"║   MAPE:  {metrics_test['MAPE_pct']:>16.2f}%                                   ║")
    print(f"║   R²:    {metrics_test['R2']:>17.4f}                                   ║")
    
    ratio_mape = metrics_test['MAPE_pct'] / metrics_train['MAPE_pct']
    severity = "SEVERO" if ratio_mape > 2.5 else "MODERADO" if ratio_mape > 1.5 else "LEVE"
    
    print("║                                                                        ║")
    print(f"║ ANÁLISIS DE OVERFITTING                                                ║")
    print(f"║   Ratio MAPE (test/train): {ratio_mape:>5.2f}x  ({severity})                          ║")
    print(f"║   Gap R²: {(metrics_train['R2'] - metrics_test['R2']):>6.4f}                                     ║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Guardar artefactos
    print("💾 Guardando artefactos...")
    
    # Crear pipeline compatible con el código de remodelación
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    
    # Preprocessador dummy (el encoding ya lo hicimos)
    class DummyTransformer:
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X
        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)
    
    pre = DummyTransformer()
    pipe = Pipeline(steps=[("pre", pre), ("xgb", model)])
    pipe.feature_names_in_ = X_train.columns
    
    # Modelo path
    model_path = os.path.join(args.outdir, "model_xgb.joblib")
    joblib.dump(pipe, model_path)
    print(f"   ✓ Modelo: {model_path}")

    # Métricas
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "train": metrics_train,
            "test": metrics_test,
            "log_target": args.log_target,
            "best_iteration": int(best_iter) if best_iter is not None else -1
        }, f, indent=2)
    print(f"   ✓ Métricas: {metrics_path}")

    # Meta
    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "target": args.target,
            "drop_cols": drop_cols,
            "numeric_cols": list(X.columns),
            "categorical_cols": [],
            "xgb_params": xgb_params,
            "log_target": args.log_target,
            "training_rows": len(X_train),
            "test_rows": len(X_test)
        }, f, indent=2)
    print(f"   ✓ Metadata: {meta_path}")

    print()
    print("=" * 70)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print()
    print(f"📦 Resultados guardados en: {args.outdir}/")
    print()

if __name__ == "__main__":
    main()
