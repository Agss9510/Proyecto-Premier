import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from scipy.stats import poisson

# --- CONFIGURACIÓN DE RUTAS (CORREGIDAS) ---
BASE_DIR = Path(__file__).resolve().parent

# Usamos .parent.parent para subir dos niveles y alcanzar la raíz del proyecto (Proyecto Premier/)
PROYECTO_ROOT = BASE_DIR.parent.parent

# Rutas de archivos (usando la nueva variable PROYECTO_ROOT)
MODELO_PATH = PROYECTO_ROOT / '04_Modelos_Entrenados' / 'modelo_v1_corners.pkl'
BASE_V1_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_V1_corners.csv'

# La línea de córners que nos interesa
LINEA_CORNERS = 10.5


def obtener_ultimas_metricas(df, equipo_local, equipo_visitante):
    """
    Busca las últimas métricas de córners (promedios móviles) para los dos equipos 
    en el momento de su último partido registrado.
    """
    
    # Nos aseguramos de que el DataFrame esté ordenado por fecha para tomar el dato más reciente
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df_sorted = df.sort_values(by='Fecha', ascending=False)
    
    # Buscar el dato más reciente para el equipo local
    local_data = df_sorted[(df_sorted['Local'] == equipo_local) | (df_sorted['Visitante'] == equipo_local)].head(1)
    # Buscar el dato más reciente para el equipo visitante
    visitante_data = df_sorted[(df_sorted['Local'] == equipo_visitante) | (df_sorted['Visitante'] == equipo_visitante)].head(1)
        
    if local_data.empty or visitante_data.empty:
        raise ValueError(f"No se pudieron encontrar datos recientes para {equipo_local} o {equipo_visitante}.")

    # --- Creación del DataFrame de Predicción (X_new) ---
    # NOTA: Necesitamos los promedios del equipo local (AF/EC) y del visitante (AF/EC)
    
    # 1. Obtener los promedios del equipo LOCAL (sin importar si en esa fila fue Local o Visitante)
    # Si la última aparición de Local fue como Local, toma las métricas 'Local_CORNERS_...'
    if local_data['Local'].iloc[0] == equipo_local:
        local_af = local_data['Local_CORNERS_A_FAVOR_AVG'].iloc[0]
        local_ec = local_data['Local_CORNERS_EN_CONTRA_AVG'].iloc[0]
    # Si la última aparición de Local fue como Visitante, toma las métricas 'Visitante_CORNERS_...'
    else:
        local_af = local_data['Visitante_CORNERS_A_FAVOR_AVG'].iloc[0]
        local_ec = local_data['Visitante_CORNERS_EN_CONTRA_AVG'].iloc[0]
    
    # 2. Obtener los promedios del equipo VISITANTE
    # Si la última aparición de Visitante fue como Local, toma las métricas 'Local_CORNERS_...'
    if visitante_data['Local'].iloc[0] == equipo_visitante:
        visitante_af = visitante_data['Local_CORNERS_A_FAVOR_AVG'].iloc[0]
        visitante_ec = visitante_data['Local_CORNERS_EN_CONTRA_AVG'].iloc[0]
    # Si la última aparición de Visitante fue como Visitante, toma las métricas 'Visitante_CORNERS_...'
    else:
        visitante_af = visitante_data['Visitante_CORNERS_A_FAVOR_AVG'].iloc[0]
        visitante_ec = visitante_data['Visitante_CORNERS_EN_CONTRA_AVG'].iloc[0]


    # Construir el DataFrame de 1 fila para la predicción, usando las métricas correctas
    X_new = pd.DataFrame({
        'Local_CORNERS_A_FAVOR_AVG': [local_af],
        'Local_CORNERS_EN_CONTRA_AVG': [local_ec],
        'Visitante_CORNERS_A_FAVOR_AVG': [visitante_af],
        'Visitante_CORNERS_EN_CONTRA_AVG': [visitante_ec]
    })
    
    return X_new


def predecir_over_under_poisson(equipo_local, equipo_visitante, linea_corners):
    """
    Carga el modelo V1, obtiene las métricas y calcula la probabilidad de Over/Under 10.5.
    """
    try:
        # 1. Cargar el modelo entrenado
        with open(MODELO_PATH, 'rb') as file:
            poisson_results = pickle.load(file)
            
        # 2. Cargar la base de datos para obtener las últimas métricas
        df_v1 = pd.read_csv(BASE_V1_PATH)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error al cargar archivos: {e}. Asegúrate de que los archivos .pkl y .csv existen en las rutas correctas.")
        return

    # 3. Obtener las características de los equipos para el nuevo partido
    try:
        X_new = obtener_ultimas_metricas(df_v1, equipo_local, equipo_visitante)
    except ValueError as e:
        print(f"Error de datos: {e}")
        return

    # 4. Predecir la media esperada (lambda)
    lambda_pred = poisson_results.predict(X_new).iloc[0]
    
    # 5. Calcular la Probabilidad de "Under"
    # El Under 10.5 significa 0, 1, 2, ..., hasta 10 córners.
    limite_superior_under = int(np.floor(linea_corners)) 
    
    # Usamos la Función de Masa de Probabilidad Acumulada (CDF) de Poisson.
    prob_under = poisson.cdf(limite_superior_under, lambda_pred)
    prob_over = 1 - prob_under
    
    # --- RESULTADOS ---
    print("\n" + "="*80)
    print(f"       PREDICCIÓN OVER/UNDER {linea_corners} CÓRNERS: {equipo_local} vs {equipo_visitante}")
    print("="*80)
    print(f"Métricas usadas (Promedios AF/EC del Local): {X_new.iloc[0]['Local_CORNERS_A_FAVOR_AVG']:.2f} / {X_new.iloc[0]['Local_CORNERS_EN_CONTRA_AVG']:.2f}")
    print(f"Métricas usadas (Promedios AF/EC del Visitante): {X_new.iloc[0]['Visitante_CORNERS_A_FAVOR_AVG']:.2f} / {X_new.iloc[0]['Visitante_CORNERS_EN_CONTRA_AVG']:.2f}")
    print("-" * 80)
    print(f"➡️ Media Esperada de Córners (λ) para el partido: {lambda_pred:.2f}")
    print("-" * 80)
    print(f"PROBABILIDAD UNDER {linea_corners} (<= {limite_superior_under} Córners): {prob_under:.2%}")
    print(f"PROBABILIDAD OVER {linea_corners} (>= {limite_superior_under + 1} Córners): {prob_over:.2%}")
    print("="*80)
    

# --- SECCIÓN DE EJECUCIÓN ---
if __name__ == "__main__":
    # ¡Recuerda! Los nombres de los equipos deben coincidir EXACTAMENTE con los de tu CSV.
    LOCAL = "Bournemouth" 
    VISITANTE = "Fulham"

    predecir_over_under_poisson(LOCAL, VISITANTE, LINEA_CORNERS)