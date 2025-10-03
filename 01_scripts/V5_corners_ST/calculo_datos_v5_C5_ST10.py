import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# ¡Asegúrate de que esta ruta sea correcta!
PROYECTO_ROOT_ABSOLUTO = Path(r"C:\Users\PC\Desktop\Proyecto Premier") 
BASE_CONSOLIDADA_PATH = PROYECTO_ROOT_ABSOLUTO / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'
OUTPUT_PATH = PROYECTO_ROOT_ABSOLUTO / '03_Datos_Limpios' / 'premier_league_BASE_V5_C5_ST10.csv'

# Parámetros del Modelo V5.1
N_CORNERS = 5   # Ventana para Córners (HC/AC)
N_ST = 10       # Ventana para Tiros a Puerta (HST/AST)

def calcular_metricas(df, equipo, metrica_abr, N):
    """
    Calcula los promedios móviles (A Favor y En Contra) para un equipo.
    metrica_abr debe ser la versión Home, ej: 'HC', 'HST'. 
    Asumimos que la versión Away es la misma con 'H' reemplazada por 'A' (ej: 'AC', 'AST').
    """
    
    df_equipo_calc = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)].copy()
    df_equipo_calc = df_equipo_calc.sort_values(by='Fecha').reset_index(drop=True)
    
    metrica_nombre = metrica_abr.replace('H', '') # Ej: 'HC' -> 'C'
    
    # 1. Columnas de Córners/Tiros A Favor (AF) y En Contra (EC)
    col_home = metrica_abr     # Ej: 'HC' o 'HST'
    col_away = metrica_abr.replace('H', 'A') # Ej: 'AC' o 'AST'
    
    # A FAVOR (AF): Si es local, usa col_home. Si es visitante, usa col_away.
    af = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_home], df_equipo_calc[col_away])
    
    # EN CONTRA (EC): Si es local, usa col_away. Si es visitante, usa col_home.
    ec = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_away], df_equipo_calc[col_home])
    
    af_series = pd.Series(af)
    ec_series = pd.Series(ec)
    
    # Promedio móvil de las N jornadas anteriores (shift(1) para no incluir el partido actual)
    df_equipo_calc[f'{metrica_nombre}_AF_AVG'] = af_series.shift(1).rolling(window=N, min_periods=1).mean()
    df_equipo_calc[f'{metrica_nombre}_EC_AVG'] = ec_series.shift(1).rolling(window=N, min_periods=1).mean()
    
    cols = [f'{metrica_nombre}_AF_AVG', f'{metrica_nombre}_EC_AVG']

    return df_equipo_calc[['Fecha', 'Local', 'Visitante'] + cols].copy()


def preparar_datos_v5(base_path, output_path):
    try:
        df = pd.read_csv(base_path)
    except FileNotFoundError:
        print(f"Error: Base consolidada no encontrada en {base_path}")
        return

    # --- CORRECCIÓN 1: Eliminación del renombramiento incorrecto ---
    # La consolidación ya debe proveer los nombres correctos (16 columnas).
    # Se comentó el bloque que intentaba renombrar 16 columnas con solo 8 nombres.
    # df.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
    #              'HC', 'AC',   
    #              'HST', 'AST'] 

    # 1. Preparación general
    # --- CORRECCIÓN 2: Tratamiento de fechas con errores='coerce' ---
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    
    # Eliminar filas con fechas no válidas que errors='coerce' convirtió a NaT
    df = df.dropna(subset=['Fecha']).sort_values(by='Fecha').reset_index(drop=True)
    
    all_teams = pd.concat([df['Local'], df['Visitante']]).unique()
    df_final = df.copy()

    # 2. Iterar y calcular métricas (se asume que las columnas 'HC', 'AC', 'HST', 'AST' existen y son correctas)
    for equipo in all_teams:
        
        # Córners (C5) - metrica_abr='HC'
        df_c = calcular_metricas(df, equipo, 'HC', N_CORNERS)
        
        # Tiros a Puerta (ST10) - metrica_abr='HST'
        df_st = calcular_metricas(df, equipo, 'HST', N_ST)
        
        # Unir Córners al DataFrame principal
        df_c = df_c.drop(columns=['Local', 'Visitante'])
        df_c = df_c.rename(columns={'C_AF_AVG': f'{equipo}_CORNERS_AF_AVG',
                                     'C_EC_AVG': f'{equipo}_CORNERS_EC_AVG'})
        df_final = df_final.merge(df_c, on=['Fecha'], how='left')

        # Unir Tiros a Puerta al DataFrame principal
        df_st = df_st.drop(columns=['Local', 'Visitante'])
        df_st = df_st.rename(columns={'ST_AF_AVG': f'{equipo}_ST_AF_AVG',
                                       'ST_EC_AVG': f'{equipo}_ST_EC_AVG'})
        df_final = df_final.merge(df_st, on=['Fecha'], how='left')

    # 3. Mapear métricas al partido (Local y Visitante)
    df_final['Local_CORNERS_AF_AVG'] = df_final.apply(lambda row: row[f"{row['Local']}_CORNERS_AF_AVG"], axis=1)
    df_final['Local_CORNERS_EC_AVG'] = df_final.apply(lambda row: row[f"{row['Local']}_CORNERS_EC_AVG"], axis=1)
    df_final['Visitante_CORNERS_AF_AVG'] = df_final.apply(lambda row: row[f"{row['Visitante']}_CORNERS_AF_AVG"], axis=1)
    df_final['Visitante_CORNERS_EC_AVG'] = df_final.apply(lambda row: row[f"{row['Visitante']}_CORNERS_EC_AVG"], axis=1)

    df_final['Local_ST_AF_AVG'] = df_final.apply(lambda row: row[f"{row['Local']}_ST_AF_AVG"], axis=1)
    df_final['Local_ST_EC_AVG'] = df_final.apply(lambda row: row[f"{row['Local']}_ST_EC_AVG"], axis=1)
    df_final['Visitante_ST_AF_AVG'] = df_final.apply(lambda row: row[f"{row['Visitante']}_ST_AF_AVG"], axis=1)
    df_final['Visitante_ST_EC_AVG'] = df_final.apply(lambda row: row[f"{row['Visitante']}_ST_EC_AVG"], axis=1)

    # 4. Limpieza final y selección de columnas para el modelo
    # Solo modelamos partidos donde tenemos las 4 métricas de ambos equipos
    df_modelado = df_final.dropna(subset=['Local_CORNERS_AF_AVG', 'Local_ST_AF_AVG']) 
    
    # Uso de .loc para evitar la advertencia SettingWithCopyWarning
    df_modelado.loc[:, 'CORNERS_TOTAL_PARTIDO'] = df_modelado['HC'] + df_modelado['AC']
    df_modelado.loc[:, 'FACTOR_LOCAL'] = 1 # Variable constante para el sesgo de jugar en casa

    # Seleccionar solo las columnas necesarias para el modelo
    cols_modelo = [
        'Local', 'Visitante', 'Fecha', 'CORNERS_TOTAL_PARTIDO', 'FACTOR_LOCAL',
        'Local_CORNERS_AF_AVG', 'Local_CORNERS_EC_AVG', 
        'Visitante_CORNERS_AF_AVG', 'Visitante_CORNERS_EC_AVG',
        'Local_ST_AF_AVG', 'Local_ST_EC_AVG', 
        'Visitante_ST_AF_AVG', 'Visitante_ST_EC_AVG'
    ]
    
    df_modelado = df_modelado[cols_modelo]
    df_modelado.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("      ✅ CÁLCULO DE MÉTRICAS V5.1 COMPLETADO")
    print(f"      Córners (N={N_CORNERS}), Tiros a Puerta (N={N_ST})")
    print(f"      Datos listos para modelar guardados en: {output_path.name}")
    print(f"      Partidos listos para modelar: {len(df_modelado)}")
    print("="*70)


if __name__ == "__main__":
    preparar_datos_v5(BASE_CONSOLIDADA_PATH, OUTPUT_PATH)