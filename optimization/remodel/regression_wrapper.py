"""
SOLUCIÓN: Wrapper para regresión que maneja nombres de features correctamente
Esto asegura que el Pipeline sklearn funciona con feature names seguros
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

class RegressionModelWrapper:
    """
    Wrapper que maneja el Pipeline de regresión y asegura pasar DataFrames con feature names correctos
    """
    def __init__(self, model_path="models/regression_model.joblib"):
        self.pipeline = joblib.load(model_path)
        # Obtener feature names del regressor
        regressor = self.pipeline.named_steps['regressor']
        self.feature_names = list(regressor.feature_names_in_) if hasattr(regressor, 'feature_names_in_') else []
        
    def predict(self, data_dict):
        """
        Predice el precio (en log space) dado un diccionario de features
        
        Args:
            data_dict: dict con keys que deben coincidir con feature_names
            
        Returns:
            float: predicción en log space (requiere np.exp para precio real)
        """
        if not self.feature_names:
            raise ValueError("No feature names found in regression model")
        
        # Construir DataFrame garantizando orden y nombres correctos
        row_data = {}
        for feat in self.feature_names:
            if feat in data_dict:
                row_data[feat] = data_dict[feat]
            else:
                # Si falta una feature, llenar con 0 como fallback
                row_data[feat] = 0.0
        
        # Crear DataFrame con feature names en orden exacto
        X = pd.DataFrame([row_data], columns=self.feature_names)
        
        # Predecir
        pred = self.pipeline.predict(X)[0]
        
        return pred
    
    def predict_precio(self, data_dict):
        """
        Predice el precio en dinero (exponenciado)
        """
        pred_log = self.predict(data_dict)
        precio = np.exp(pred_log)
        return precio

if __name__ == "__main__":
    # Test
    import sys
    
    print("Testing RegressionModelWrapper...")
    wrapper = RegressionModelWrapper()
    print(f"✓ Features: {len(wrapper.feature_names)}")
    print(f"✓ Primeros 5: {wrapper.feature_names[:5]}")
    
    # Cargar data de test
    df = pd.read_csv("data/raw/df_final_regresion.csv")
    test_row = df.iloc[0]
    
    # Convertir a dict
    test_dict = {col: float(test_row[col]) for col in wrapper.feature_names}
    
    # Predecir
    pred_log = wrapper.predict(test_dict)
    precio = wrapper.predict_precio(test_dict)
    
    print(f"\n✓ Predicción (log space): {pred_log:.6f}")
    print(f"✓ Predicción (precio): ${precio:,.0f}")
    print(f"✓ Valor real: ${test_row['SalePrice_Present']:,.0f}")
    print(f"✓ Error: {abs(precio - test_row['SalePrice_Present']) / test_row['SalePrice_Present'] * 100:.2f}%")
