import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_ROOT = BASE_DIR.parent 

# Ruta de entrada (la base generada en el paso anterior)
BASE_MODELADO_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_V6_C5_ST10_FINAL.csv'

# 🌟 CORRECCIÓN DE RUTA FINAL: Usando tu carpeta '04_Modelos_Entrenados'
OUTPUT_SUMMARY_PATH = PROYECTO_ROOT / '04_Modelos_Entrenados' / 'resumen_poisson_V6_FINAL.txt'

def entrenar_modelo_poisson(base_path, output_path):
    """Carga los datos, entrena el modelo de Regresión de Poisson y guarda el resumen."""
    
    try:
        df = pd.read_csv(base_path)
    except FileNotFoundError:
        print(f"\n🚨 ERROR: Archivo no encontrado en: {base_path}")
        print("Asegúrate de haber ejecutado el script de cálculo de datos ('calculo_datos_v6...') primero.")
        return

    # 1. Definir Variables (Modelo V6.0)
    X_cols = [
        'Local_CORNERS_AF_AVG', 'Local_CORNERS_EC_AVG', 'Visitante_CORNERS_AF_AVG', 'Visitante_CORNERS_EC_AVG',
        'Local_ST_AF_AVG', 'Local_ST_EC_AVG', 'Visitante_ST_AF_AVG', 'Visitante_ST_EC_AVG',
        'FACTOR_LOCAL' 
    ]
    Y = df['CORNERS_TOTAL_PARTIDO']
    X = df[X_cols]
    
    print(f"\nDatos cargados. Filas totales: {len(df)}")
    print(f"Columnas utilizadas (X): {len(X.columns)}")
    
    # 2. Entrenamiento del Modelo de Regresión de Poisson
    poisson_model = sm.GLM(Y, X, family=sm.families.Poisson())
    poisson_results = poisson_model.fit()
    
    # 3. Guardar Resumen
    
    # 🌟 CORRECCIÓN CRÍTICA: Crear el directorio si no existe 
    # (En tu caso, '04_Modelos_Entrenados' ya existe, pero esta línea es una buena práctica de seguridad)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(poisson_results.summary().as_text())
        
    print("\n" + "="*80)
    print("      ✅ ENTRENAMIENTO DEL MODELO POISSON V6.0 COMPLETADO")
    print(f"      Resultados guardados en: {output_path.name}")
    print("="*80)
    print("\n--- Resumen de Coeficientes (Modelo V6.0) ---")
    print(poisson_results.summary().as_text())
    print("\n--- ¡Analiza los coeficientes y p-valores! ---")


if __name__ == "__main__":
    entrenar_modelo_poisson(BASE_MODELADO_PATH, OUTPUT_SUMMARY_PATH)