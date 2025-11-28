#!/usr/bin/env python3
"""
Wrapper para hacer predicciones con regresión en los datos de run_opt.py

El problema: run_opt.py genera datos en formato one-hot encoding (como XGBoost),
pero la regresión espera el mismo formato.

Solución: Este wrapper toma los datos del modelo optimizado y los transforma
al formato exacto que espera la regresión entrenada.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

class RegressionPredictor:
    """
    Predice el precio con regresión lineal entrenada
    
    Usa el modelo entrenado en: models/regression_model_reprocesed.pkl
    Datos transformados con build_base_input_row() (100% compatible con run_opt.py)
    """
    
    def __init__(self, model_path: str = "models/regression_model_reprocesed.pkl"):
        """Carga el modelo de regresión"""
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}. "
                                  "Ejecuta: python3 training/train_regression_reprocessed.py")
        with open(self.model_path, "rb") as f:
            self.model_pkg = pickle.load(f)
        
        self.model = self.model_pkg["model"]
        self.feature_names = self.model_pkg.get("feature_names", [])
        
        # Recolectar todos los metadatos disponibles
        self.metadata = {
            "n_features": len(self.feature_names),
            "r2": self.model_pkg.get("r2", None),
            "rmse_real": self.model_pkg.get("rmse_real", None),
            "mae_real": self.model_pkg.get("mae_real", None),
            "mape": self.model_pkg.get("mape", None),
            "n_training_samples": self.model_pkg.get("n_training_samples", None),
            "processed_with": self.model_pkg.get("processed_with", "unknown"),
        }
        
    def predict(self, X: pd.DataFrame) -> float:
        """
        Predice el precio a partir de features one-hot encoded
        
        Args:
            X: DataFrame con features (puede tener más o menos columnas)
            
        Returns:
            Precio predicho en $ (escala real, no log)
        """
        
        # Garantizar que es DataFrame
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        
        if len(X) == 0:
            raise ValueError("X vacío")
        
        # Asegurar que X es una fila
        if len(X) > 1:
            X = X.iloc[[0]]
        
        # Alinear features exactamente como el modelo espera
        X_aligned = pd.DataFrame(columns=self.feature_names)
        for col in self.feature_names:
            if col in X.columns:
                val = X[col].iloc[0] if len(X) > 0 else 0.0
                X_aligned.loc[0, col] = float(val)
            else:
                X_aligned.loc[0, col] = 0.0
        
        # Asegurar float
        X_aligned = X_aligned.astype(float)
        
        # Predecir en escala log
        try:
            if len(X_aligned) == 0:
                raise ValueError(f"X_aligned vacío después de alineación. Features esperadas: {len(self.feature_names)}, Features recibidas: {len(X.columns)}")
            
            pred_log = self.model.predict(X_aligned.values)[0]
            pred_price = float(np.exp(pred_log))
        except Exception as e:
            raise ValueError(f"Error en predicción: {e}")
        
        return pred_price


# ============================================================================
# SCRIPT DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    # Cambiar al directorio raíz del proyecto
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, '.')
    
    print("\n" + "="*70)
    print("  TEST: RegressionPredictor (regression_model_reprocesed.pkl)")
    print("="*70 + "\n")
    
    # Cargar regresión
    try:
        reg = RegressionPredictor()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Cargar datos de prueba
    df = pd.read_csv("data/raw/df_final_regresion.csv")
    
    # Encontrar PID 526301100
    if 526301100 not in df["PID"].values:
        print("❌ PID no encontrado en datos de entrenamiento")
        sys.exit(1)
    
    idx = df[df["PID"] == 526301100].index[0]
    row = df.iloc[idx]
    real_price = row["SalePrice_Present"]
    
    print(f"🏠 Casa PID 526301100:")
    print(f"  - Precio real: ${real_price:,.0f}\n")
    
    # Procesar con build_base_input_row (exactamente como run_opt.py)
    try:
        from optimization.remodel.xgb_predictor import XGBBundle
        from optimization.remodel.gurobi_model import build_base_input_row
        
        bundle = XGBBundle()  # Carga modelo por defecto
        X_test = build_base_input_row(bundle, row)
        
        # Predecir con regresión
        pred = reg.predict(X_test)
        error = pred - real_price
        error_pct = (error / real_price) * 100
        
        print(f"💰 PREDICCIÓN REGRESIÓN:")
        print(f"  - Predicción: ${pred:,.0f}")
        print(f"  - Real:       ${real_price:,.0f}")
        print(f"  - Error:      ${error:+,.0f} ({error_pct:+.2f}%)")
        print(f"\n  🎯 Metadata:")
        print(f"     - Features: {reg.metadata.get('n_features', 'N/A')}")
        
        r2 = reg.metadata.get('r2', 'N/A')
        if isinstance(r2, (int, float)):
            print(f"     - R² train: {r2:.4f}")
        else:
            print(f"     - R² train: {r2}")
            
        mape = reg.metadata.get('mape', 'N/A')
        if isinstance(mape, (int, float)):
            print(f"     - MAPE train: {mape:.2f}%")
        else:
            print(f"     - MAPE train: {mape}")
        
        if abs(error_pct) < 10:
            print(f"\n  ✅ Error excelente (<10%)")
        elif abs(error_pct) < 15:
            print(f"\n  ✅ Error bueno (<15%)")
        elif abs(error_pct) < 25:
            print(f"\n  ⚠️  Error moderado")
        else:
            print(f"\n  ❌ Error alto (>25%)")
            
    except Exception as e:
        print(f"❌ Error durante prueba: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")

