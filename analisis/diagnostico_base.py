# optimization/remodel/diagnostico_base.py
import os
import pandas as pd

def diagnosticar_base_datos():
    """Diagnostica problemas con la base de datos"""
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("🔍 DIAGNÓSTICO DE BASE DE DATOS")
    print("="*50)
    
    # Intentar importar config
    try:
        from optimization.remodel.config import PARAMS
        data_file = PARAMS.data_file
        print(f"📁 Archivo de datos en config: {data_file}")
        
        # Verificar si existe
        posibles_rutas = [
            data_file,
            os.path.join(PROJECT_ROOT, data_file),
            os.path.join(PROJECT_ROOT, "data", data_file),
            os.path.join(PROJECT_ROOT, "data", "ames_housing.csv"),
            os.path.join(PROJECT_ROOT, "ames_housing.csv"),
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                print(f"✅ ENCONTRADO: {ruta}")
                df = pd.read_csv(ruta)
                print(f"📊 Dimensiones: {df.shape}")
                print(f"📝 Columnas: {list(df.columns)[:10]}...")  # Primeras 10 columnas
                
                # Verificar IDs
                if 'PID' in df.columns:
                    ids = df['PID'].dropna().unique()
                    print(f"🔢 IDs encontrados: {len(ids)}")
                    print(f"📋 Ejemplos: {ids[:5].tolist()}")
                else:
                    print("❌ No se encuentra columna 'PID'")
                    # Buscar columnas que puedan ser IDs
                    id_cols = [col for col in df.columns if 'id' in col.lower()]
                    print(f"🔍 Columnas que podrían ser IDs: {id_cols}")
                
                break
        else:
            print("❌ No se encontró el archivo de datos en ninguna ruta posible")
            
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")

if __name__ == "__main__":
    diagnosticar_base_datos()