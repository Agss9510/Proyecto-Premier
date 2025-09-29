import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import poisson
from sklearn.metrics import mean_squared_error
import pickle

# --- CONFIGURACIÓN DE RUTAS (CORREGIDAS) ---

# BASE_DIR es ahora '01_scripts/V1_corners/'
BASE_DIR = Path(__file__).resolve().parent

# Usamos .parent.parent para subir dos niveles y alcanzar la raíz del proyecto (Proyecto Premier/)
PROYECTO_ROOT = BASE_DIR.parent.parent

# Rutas de archivos
BASE_V1_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_V1_corners.csv'
MODELO_NOMBRE = 'modelo_v1_corners.pkl'
MODELO_OUTPUT_PATH = PROYECTO_ROOT / '04_Modelos_Entrenados' / MODELO_NOMBRE


def entrenar_modelo_v1(base_path, output_path):
    """
    Entrena un modelo de Regresión de Poisson usando solo métricas de córners.
    """
    try:
        df = pd.read_csv(base_path)
    except FileNotFoundError:
        print(f"Error: No se encontró la base V1 en {base_path}.")
        return

    # 1. Preparación de Datos
    # Las variables X son las 4 métricas de promedio móvil de córners
    # La variable Y es el total de córners del partido
    FORMULA = (
        "CORNERS_TOTAL_PARTIDO ~ "
        "Local_CORNERS_A_FAVOR_AVG + Local_CORNERS_EN_CONTRA_AVG + "
        "Visitante_CORNERS_A_FAVOR_AVG + Visitante_CORNERS_EN_CONTRA_AVG"
    )

    # 2. División Entrenamiento y Prueba
    # Usaremos una división simple de 80/20
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    
    # 3. Entrenamiento del Modelo de Poisson
    print("Iniciando entrenamiento del modelo de Regresión de Poisson...")
    poisson_results = poisson(FORMULA, data=df_train).fit(
        method='newton', 
        maxiter=100
    )
    print("Entrenamiento completado.")

    # 4. Evaluación del Modelo (RMSE)
    # Predecir sobre el conjunto de prueba
    y_true = df_test['CORNERS_TOTAL_PARTIDO']
    y_pred = poisson_results.predict(df_test)
    
    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 5. Guardar el Modelo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump(poisson_results, file)
        
    # 6. Ejemplo de Predicción (Primer partido del set de prueba)
    ejemplo_partido = df_test.iloc[[0]]
    ejemplo_prediccion = poisson_results.predict(ejemplo_partido).iloc[0]

    # --- RESULTADOS FINALES ---
    print("\n" + "="*70)
    print("       ¡MODELO V1 (CÓRNERS) ENTRENADO Y GUARDADO!")
    print("="*70)
    print(f"Modelo guardado en: {output_path}")
    print(f"Métrica de Evaluación (RMSE en Test): {rmse:.4f}")
    print("El RMSE representa el error promedio en la predicción de córners totales (cuanto más bajo, mejor).")
    print("="*70)
    print("--- Ejemplo de Predicción ---")
    print(f"Predicción de Córners Totales (Media esperada λ): {ejemplo_prediccion:.2f}")


# --- SECCIÓN DE EJECUCIÓN ---
if __name__ == "__main__":
    entrenar_modelo_v1(BASE_V1_PATH, MODELO_OUTPUT_PATH)