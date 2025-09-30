import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import re

# --- CONFIGURACIÓN DE RUTAS ---
PROYECTO_ROOT_ABSOLUTO = Path(r"C:\Users\PC\Desktop\Proyecto Premier") # <-- RUTA ABSOLUTA

BASE_CONSOLIDADA_PATH = PROYECTO_ROOT_ABSOLUTO / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'
MODELO_PATH = PROYECTO_ROOT_ABSOLUTO / '04_Modelos_Entrenados' / 'modelo_v2_C5_T10.pkl' 

# Las ventanas móviles usadas en el modelo V2.1 ganador
N_CORNERS = 5 
N_SHOTS = 10 

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
    "Spurs": "Spurs", # Aseguramos que el nombre corto Spurs se maneje
    "Tottenham": "Spurs" # Agregamos un posible nombre corto alternativo
}

# --- 1. DEFINICIÓN DE PARTIDOS DE LA JORNADA ---
PARTIDOS_JORNADA = [
    ("AFC Bournemouth", "Fulham"),
    ("Leeds United", "Tottenham Hotspur"), # Este es el partido problemático
    ("Arsenal", "West Ham United"),
    ("Manchester United", "Sunderland"),
    ("Chelsea", "Liverpool"),
    ("Aston Villa", "Burnley"),
    ("Everton", "Crystal Palace"),
    ("Newcastle United", "Nottingham Forest"),
    ("Wolverhampton Wanderers", "Brighton & Hove Albion"),
    ("Brentford", "Manchester City"),
]

# --- 2. FUNCIÓN DE CÁLCULO DE MÉTRICAS ---

def calcular_metricas_historicas(df, equipo, metricas, N):
    df_equipo_calc = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)].copy()
    
    resultados_dict = {}

    for metrica_abr, metrica_nombre in metricas.items():
        af = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[f'{metrica_nombre}_LOCAL'], df_equipo_calc[f'{metrica_nombre}_VISITANTE'])
        ec = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[f'{metrica_nombre}_VISITANTE'], df_equipo_calc[f'{metrica_nombre}_LOCAL'])
        
        af_series = pd.Series(af, index=df_equipo_calc.index)
        ec_series = pd.Series(ec, index=df_equipo_calc.index)
        
        # El cálculo del promedio móvil devolverá NaN si no hay N partidos previos
        avg_af = af_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]
        avg_ec = ec_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]

        resultados_dict[f'{metrica_nombre}_A_FAVOR_AVG'] = avg_af
        resultados_dict[f'{metrica_nombre}_EN_CONTRA_AVG'] = avg_ec
        
    return resultados_dict


# --- 3. FUNCIÓN PRINCIPAL DE PREDICCIÓN ---

def predecir_jornada(base_path, modelo_path, partidos):
    
    # Cargar Base Consolidada
    try:
        df = pd.read_csv(base_path)
    except FileNotFoundError:
        print(f"Error: Base consolidada no encontrada en {base_path}")
        print(f"Ruta de búsqueda: {base_path}")
        return

    # Preparación de datos (Igual que antes)
    df.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                  'CORNERS_LOCAL', 'CORNERS_VISITANTE', 
                  'TIROS_LOCAL', 'TIROS_VISITANTE']
    
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    df = df.sort_values(by='Fecha').reset_index(drop=True)
    
    # Calcular las medias históricas de la liga para imputación de ascendidos
    liga_media = {
        'CORNERS_A_FAVOR_AVG': df['CORNERS_LOCAL'].mean(), # Proxy de media de corners AF
        'CORNERS_EN_CONTRA_AVG': df['CORNERS_VISITANTE'].mean(), # Proxy de media de corners EC
        'TIROS_A_FAVOR_AVG': df['TIROS_LOCAL'].mean(),
        'TIROS_EN_CONTRA_AVG': df['TIROS_VISITANTE'].mean(),
    }
    
    # Cargar Modelo Ganador
    try:
        with open(modelo_path, 'rb') as file:
            modelo = pickle.load(file)
        try:
             rmse = np.sqrt(modelo.mse_resid)
        except AttributeError:
             rmse = 3.4275 
             
    except FileNotFoundError:
        print(f"Error: Modelo V2.1 ('modelo_v2_C5_T10.pkl') no encontrado en {modelo_path}. ¡Asegúrate de que el archivo exista!")
        return
    
    rendimiento_reciente = {}
    equipos_unicos = pd.concat([df['Local'], df['Visitante']]).unique()

    # Cálculo del rendimiento más reciente
    for equipo in equipos_unicos:
        
        metricas_corners = {'CORNERS': 'CORNERS'}
        res_c = calcular_metricas_historicas(df, equipo, metricas_corners, N_CORNERS)
        
        metricas_tiros = {'TIROS': 'TIROS'}
        res_t = calcular_metricas_historicas(df, equipo, metricas_tiros, N_SHOTS)
        
        # Almacenar resultados
        rendimiento_reciente[equipo] = {
            'CORNERS_A_FAVOR_AVG': res_c['CORNERS_A_FAVOR_AVG'],
            'CORNERS_EN_CONTRA_AVG': res_c['CORNERS_EN_CONTRA_AVG'],
            'TIROS_A_FAVOR_AVG': res_t['TIROS_A_FAVOR_AVG'],
            'TIROS_EN_CONTRA_AVG': res_t['TIROS_EN_CONTRA_AVG'],
        }

    
    # --- 4. PREPARAR EL DATAFRAME DE PREDICCIÓN CON IMPUTACIÓN ---
    df_prediccion = pd.DataFrame()
    partidos_finales = []
    
    for local_largo, visitante_largo in partidos:
        local = MAPEO_NOMBRES.get(local_largo, local_largo)
        visitante = MAPEO_NOMBRES.get(visitante_largo, visitante_largo)
        
        # Intentamos obtener los datos
        datos_local = rendimiento_reciente.get(local, {})
        datos_visitante = rendimiento_reciente.get(visitante, {})
        
        # IMPUTACIÓN: Si hay valores NaN (ej. equipo ascendido), se reemplazan con la media de la liga
        for key in liga_media.keys():
            if pd.isna(datos_local.get(key)):
                 datos_local[key] = liga_media[key]
            if pd.isna(datos_visitante.get(key)):
                 datos_visitante[key] = liga_media[key]

        # Si el equipo no tiene NINGÚN dato histórico y no fue encontrado en el diccionario (caso raro), saltamos
        if not datos_local or not datos_visitante:
            print(f"Advertencia: El equipo '{local}' o '{visitante}' no pudo ser mapeado. Saltando.")
            continue
        
        # Si el equipo ascendido se llenó con medias, ¡se predice!
        
        nueva_fila = {
            'Local': local,
            'Visitante': visitante,
            'Local_CORNERS_A_FAVOR_AVG': datos_local['CORNERS_A_FAVOR_AVG'],
            'Local_CORNERS_EN_CONTRA_AVG': datos_local['CORNERS_EN_CONTRA_AVG'],
            'Visitante_CORNERS_A_FAVOR_AVG': datos_visitante['CORNERS_A_FAVOR_AVG'],
            'Visitante_CORNERS_EN_CONTRA_AVG': datos_visitante['CORNERS_EN_CONTRA_AVG'],
            'Local_TIROS_A_FAVOR_AVG': datos_local['TIROS_A_FAVOR_AVG'],
            'Local_TIROS_EN_CONTRA_AVG': datos_local['TIROS_EN_CONTRA_AVG'],
            'Visitante_TIROS_A_FAVOR_AVG': datos_visitante['TIROS_A_FAVOR_AVG'],
            'Visitante_TIROS_EN_CONTRA_AVG': datos_visitante['TIROS_EN_CONTRA_AVG'],
        }
        
        df_prediccion = pd.concat([df_prediccion, pd.DataFrame([nueva_fila])], ignore_index=True)
        partidos_finales.append((local_largo, visitante_largo)) 

    if df_prediccion.empty:
        print("No hay partidos válidos para predecir.")
        return

    # 5. Generar Predicciones
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
    print(f"      Modelo Usado: V2.1 (RMSE {rmse:.4f})")
    print("      *Equipos ascendidos (ej. Leeds) imputados con la media de la liga.")
    print("="*70)
    print(reporte.to_string(index=False))
    print("="*70)


# --- SECCIÓN DE EJECUCIÓN ---
if __name__ == "__main__":
    predecir_jornada(BASE_CONSOLIDADA_PATH, MODELO_PATH, PARTIDOS_JORNADA)