import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import pickle

# --- CONFIGURACIÓN DE RUTAS ---
PROYECTO_ROOT_ABSOLUTO = Path(r"C:\Users\PC\Desktop\Proyecto Premier") # ¡Asegúrate de que esta ruta sea correcta!
INPUT_PATH = PROYECTO_ROOT_ABSOLUTO / '03_Datos_Limpios' / 'premier_league_BASE_V5_C5_ST10.csv'
OUTPUT_MODELO_PATH = PROYECTO_ROOT_ABSOLUTO / '04_Modelos_Entrenados' / 'modelo_v5_C5_ST10.pkl'

def entrenar_modelo_v5(input_path, output_model_path):
    
    # Cargar datos
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Datos calculados V5.1 no encontrados en {input_path}. Ejecuta primero el script de cálculo.")
        return

    # Definición de la fórmula del Modelo de Poisson
    # Incluye el factor Local, las 4 métricas de Córners (AF/EC Local/Visitante) y las 4 métricas de Tiros a Puerta (AF/EC Local/Visitante)
    formula_v5 = (
        "CORNERS_TOTAL_PARTIDO ~ FACTOR_LOCAL + "
        "Local_CORNERS_AF_AVG + Local_CORNERS_EC_AVG + "
        "Visitante_CORNERS_AF_AVG + Visitante_CORNERS_EC_AVG + "
        "Local_ST_AF_AVG + Local_ST_EC_AVG + "
        "Visitante_ST_AF_AVG + Visitante_ST_EC_AVG"
    )
    
    # 1. Ajustar el modelo
    try:
        # Usamos el modelo de Regresión de Poisson
        modelo = sm.formula.glm(formula_v5, data=df, family=sm.families.Poisson()).fit()
        
        # Calcular el RMSE
        y_true = df['CORNERS_TOTAL_PARTIDO']
        y_pred = modelo.predict(df)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
    except Exception as e:
        print(f"Error durante el entrenamiento del modelo: {e}")
        return

    # 2. Guardar el modelo entrenado
    with open(output_model_path, 'wb') as file:
        pickle.dump(modelo, file)

    print("\n" + "="*70)
    print("      🎉 MODELO V5.1 ENTRENADO Y GUARDADO")
    print(f"      Modelo: Regresión de Poisson (C5 + Tiros a Puerta ST10)")
    print(f"      RMSE (Error Cuadrático Medio): {rmse:.4f}")
    print(f"      Modelo guardado en: {output_model_path.name}")
    print(f"      El RMSE anterior era 3.4275. ¡Comparemos!")
    print("="*70)


if __name__ == "__main__":
    entrenar_modelo_v5(INPUT_PATH, OUTPUT_MODELO_PATH)