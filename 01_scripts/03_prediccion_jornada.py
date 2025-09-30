import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import re

# --- CONFIGURACIÓN DE RUTAS ---
PROYECTO_ROOT_ABSOLUTO = Path(r"C:\Users\PC\Desktop\Proyecto Premier") # ¡RUTA ABSOLUTA!

# RUTA DEL MODELO ACTUALIZADA AL V5.1
MODELO_PATH = PROYECTO_ROOT_ABSOLUTO / '04_Modelos_Entrenados' / 'modelo_v5_C5_ST10.pkl' # <--- CAMBIO AQUÍ

# Rutas de archivos
BASE_CONSOLIDADA_PATH = PROYECTO_ROOT_ABSOLUTO / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'

# Las ventanas móviles usadas en el modelo V5.1 ganador
N_CORNERS = 5 
N_ST = 10 # Ahora usamos N_ST (Tiros a Puerta)

# Mapeo de nombres (se mantiene igual)
MAPEO_NOMBRES = {
    "AFC Bournemouth": "Bournemouth", 
    "Arsenal": "Arsenal", 
    "Aston Villa": "Aston Villa", 
    "Brentford": "Brentford", 
    "Brighton & Hove Albion": "Brighton", 
    "Burnley": "Burnley", 
    "Chelsea": "Chelsea", 
    "Crystal Palace": "Crystal Palace", 
    "Everton": "Everton", 
    "Fulham": "Fulham", 
    "Leeds United": "Leeds", 
    "Liverpool": "Liverpool", 
    "Manchester City": "Man City", 
    "Manchester United": "Man United", 
    "Newcastle United": "Newcastle", 
    "Nottingham Forest": "Nott'm Forest",
    "Sunderland": "Sunderland", 
    "Tottenham Hotspur": "Spurs", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves",
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Spurs": "Spurs",
    "Tottenham": "Spurs"
}

# --- 1. DEFINICIÓN DE PARTIDOS DE LA JORNADA ---
PARTIDOS_JORNADA = [
    ("AFC Bournemouth", "Fulham"),
    ("Leeds United", "Tottenham Hotspur"),
    ("Arsenal", "West Ham United"),
    ("Manchester United", "Sunderland"),
    ("Chelsea", "Liverpool"),
    ("Aston Villa", "Burnley"),
    ("Everton", "Crystal Palace"),
    ("Newcastle United", "Nottingham Forest"),
    ("Wolverhampton Wanderers", "Brighton & Hove Albion"),
    ("Brentford", "Manchester City"),
]

# --- 2. FUNCIÓN DE CÁLCULO DE MÉTRICAS (CORREGIDA PARA H/A) ---

def calcular_metricas_historicas(df, equipo, metrica_abr, N):
    """Calcula los promedios móviles (AF y EC) para un equipo en las últimas N jornadas."""
    
    df_equipo_calc = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)].copy()
    
    metrica_nombre = metrica_abr.replace('H', '') # Ej: 'HC' -> 'C'
    col_home = metrica_abr
    col_away = metrica_abr.replace('H', 'A') # Ej: 'AC' o 'AST'
    
    # A FAVOR (AF)
    af = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_home], df_equipo_calc[col_away])
    # EN CONTRA (EC)
    ec = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_away], df_equipo_calc[col_home])
    
    af_series = pd.Series(af, index=df_equipo_calc.index)
    ec_series = pd.Series(ec, index=df_equipo_calc.index)
    
    # Calcular el promedio móvil MÁS RECIENTE
    avg_af = af_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]
    avg_ec = ec_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]

    return {
        f'{metrica_nombre}_A_FAVOR_AVG': avg_af,
        f'{metrica_nombre}_EN_CONTRA_AVG': avg_ec
    }

# --- 3. FUNCIÓN PRINCIPAL DE PREDICCIÓN ---

def predecir_jornada(base_path, modelo_path, partidos):
    
    # Cargar Base Consolidada (Limpieza y preparación)
    try:
        df = pd.read_csv(base_path)
    except FileNotFoundError:
        print(f"Error: Base consolidada no encontrada en {base_path}")
        return

    # Estandarizar nombres para cálculo (HC/AC=Córners, HST/AST=Tiros a Puerta)
    df.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                  'HC', 'AC', 'HST', 'AST'] # <--- COLUMNAS CAMBIADAS A HST/AST
    
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    df = df.sort_values(by='Fecha').reset_index(drop=True)
    
    # Calcular las medias históricas de la liga para imputación de ascendidos
    liga_media = {
        'C_A_FAVOR_AVG': df['HC'].mean(), 
        'C_EN_CONTRA_AVG': df['AC'].mean(), 
        'ST_A_FAVOR_AVG': df['HST'].mean(), # <--- USA HST
        'ST_EN_CONTRA_AVG': df['AST'].mean(), # <--- USA AST
    }

    # Cargar Modelo Ganador
    try:
        with open(modelo_path, 'rb') as file:
            modelo = pickle.load(file)
        try:
             rmse = 3.4030 # RMSE fijo del modelo V5.1
        except AttributeError:
             rmse = 3.4030 
             
    except FileNotFoundError:
        print(f"Error: Modelo V5.1 ('modelo_v5_C5_ST10.pkl') no encontrado en {modelo_path}. ¡Asegúrate de que el archivo exista!")
        return
    
    # Diccionario para almacenar el rendimiento más reciente de cada equipo
    rendimiento_reciente = {}
    equipos_unicos = pd.concat([df['Local'], df['Visitante']]).unique()

    # Cálculo del rendimiento más reciente para Córners (N=5) y Tiros a Puerta (N=10)
    for equipo in equipos_unicos:
        
        # --- Córners (N=5) ---
        res_c = calcular_metricas_historicas(df, equipo, 'HC', N_CORNERS)
        
        # --- Tiros a Puerta (N=10) ---
        res_st = calcular_metricas_historicas(df, equipo, 'HST', N_ST)
        
        # Almacenar resultados
        rendimiento_reciente[equipo] = {
            'C_A_FAVOR_AVG': res_c['C_A_FAVOR_AVG'],
            'C_EN_CONTRA_AVG': res_c['C_EN_CONTRA_AVG'],
            'ST_A_FAVOR_AVG': res_st['ST_A_FAVOR_AVG'], # <--- ST
            'ST_EN_CONTRA_AVG': res_st['ST_EN_CONTRA_AVG'], # <--- ST
        }

    
    # --- 4. PREPARAR EL DATAFRAME DE PREDICCIÓN CON IMPUTACIÓN ---
    df_prediccion = pd.DataFrame()
    partidos_finales = []
    
    for local_largo, visitante_largo in partidos:
        local = MAPEO_NOMBRES.get(local_largo, local_largo)
        visitante = MAPEO_NOMBRES.get(visitante_largo, visitante_largo)

        datos_local = rendimiento_reciente.get(local, {})
        datos_visitante = rendimiento_reciente.get(visitante, {})

        # IMPUTACIÓN: Si hay valores NaN (ej. equipo ascendido), se reemplazan con la media de la liga
        for key in liga_media.keys():
            # Imputamos si la clave no existe o si el valor es NaN
            if pd.isna(datos_local.get(key)) or not datos_local:
                 datos_local[key] = liga_media[key]
            if pd.isna(datos_visitante.get(key)) or not datos_visitante:
                 datos_visitante[key] = liga_media[key]

        
        nueva_fila = {
            'Local': local,
            'Visitante': visitante,
            'Local_CORNERS_AF_AVG': datos_local['C_A_FAVOR_AVG'],
            'Local_CORNERS_EC_AVG': datos_local['C_EN_CONTRA_AVG'],
            'Visitante_CORNERS_AF_AVG': datos_visitante['C_A_FAVOR_AVG'],
            'Visitante_CORNERS_EC_AVG': datos_visitante['C_EN_CONTRA_AVG'],
            'Local_ST_AF_AVG': datos_local['ST_A_FAVOR_AVG'], # <--- ST
            'Local_ST_EC_AVG': datos_local['ST_EN_CONTRA_AVG'], # <--- ST
            'Visitante_ST_AF_AVG': datos_visitante['ST_A_FAVOR_AVG'], # <--- ST
            'Visitante_ST_EC_AVG': datos_visitante['ST_EN_CONTRA_AVG'], # <--- ST
        }
        
        df_prediccion = pd.concat([df_prediccion, pd.DataFrame([nueva_fila])], ignore_index=True)
        partidos_finales.append((local_largo, visitante_largo)) 

    if df_prediccion.empty:
        print("No hay partidos válidos para predecir.")
        return

    # 5. Generar Predicciones
    
    # Las columnas deben coincidir con las usadas en la fórmula del modelo V5.1
    df_prediccion['CORNERS_TOTAL_PARTIDO'] = 0 
    df_prediccion['FACTOR_LOCAL'] = 1 
    
    df_prediccion['CORNERS_PREDICHOS'] = modelo.predict(df_prediccion)
    
    # 6. Reporte Final
    
    reporte = pd.DataFrame({
        'Local': [p[0] for p in partidos_finales],
        'Visitante': [p[1] for p in partidos_finales],
        'Prediccion_Córners_Totales': df_prediccion['CORNERS_PREDICHOS'].round(2)
    })
    
    reporte['Prediccion_Córners_Totales'] = reporte['Prediccion_Córners_Totales'].astype(str) + ' (AVG)'
    
    print("\n" + "="*70)
    print("      ⚽ PREDICCIÓN DE CÓRNERS - JORNADA DE OCTUBRE 2025")
    print(f"      Modelo Usado: V5.1 (RMSE {rmse:.4f}) - ¡NUEVO CAMPEÓN!")
    print("      *Equipos ascendidos (ej. Leeds) imputados con la media de la liga.")
    print("="*70)
    print(reporte.to_string(index=False))
    print("="*70)


# --- SECCIÓN DE EJECUCIÓN ---
if __name__ == "__main__":
    predecir_jornada(BASE_CONSOLIDADA_PATH, MODELO_PATH, PARTIDOS_JORNADA)