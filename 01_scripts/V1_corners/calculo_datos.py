import pandas as pd
import numpy as np
from pathlib import Path

# --- COLUMNAS NECESARIAS PARA V1 (CÓRNERS) ---
COLUMNAS_REQUERIDAS_V1 = ['DATE', 'HOMETEAM', 'AWAYTEAM', 'FTR', 'HC', 'AC'] 
NOMBRES_AMIGABLES_V1 = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 'CORNERS_LOCAL', 'CORNERS_VISITANTE']


def calcular_metricas_v1(base_path, n_partidos=5):
    """
    Calcula los promedios móviles (AF/EC) de Córners para crear la Base V1.
    """
    
    # 1. Carga de la Base Consolidada
    try:
        df = pd.read_csv(base_path)
    except FileNotFoundError:
        print(f"Error: No se encontró la base consolidada en {base_path}.")
        return None
    except Exception as e:
        print(f"Error CRÍTICO al leer la Base Consolidada: {e}")
        return None
    
    # 2. Selección y Limpieza de Córners
    try:
        df_clean = df[COLUMNAS_REQUERIDAS_V1].copy()
    except KeyError as e:
        print(f"Error de Columna: Falta una columna esencial en la base consolidada: {e}")
        print("Asegúrate de que 00_consolidacion_datos.py se ejecutó y las columnas están en MAYÚSCULAS.")
        return None

    for col in ['HC', 'AC']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
    df_clean.columns = NOMBRES_AMIGABLES_V1
    
    # 3. Preparación
    df_clean['Fecha'] = pd.to_datetime(df_clean['Fecha'])
    df_clean = df_clean.sort_values(by='Fecha').reset_index(drop=True)
    df_clean['CORNERS_TOTAL_PARTIDO'] = df_clean['CORNERS_LOCAL'] + df_clean['CORNERS_VISITANTE']

    # --- INICIALIZAR COLUMNAS FINALES CON NaN ---
    df_clean['Local_CORNERS_A_FAVOR_AVG'] = np.nan
    df_clean['Local_CORNERS_EN_CONTRA_AVG'] = np.nan
    df_clean['Visitante_CORNERS_A_FAVOR_AVG'] = np.nan
    df_clean['Visitante_CORNERS_EN_CONTRA_AVG'] = np.nan
    
    # 4. Cálculo del Rendimiento Reciente (Promedio Móvil)
    equipos = pd.concat([df_clean['Local'], df_clean['Visitante']]).unique()
    
    for equipo in equipos:
        df_equipo_indices = df_clean[(df_clean['Local'] == equipo) | (df_clean['Visitante'] == equipo)].index
        df_equipo_calc = df_clean.loc[df_equipo_indices].copy()
        
        df_equipo_calc['C_AF'] = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc['CORNERS_LOCAL'], df_equipo_calc['CORNERS_VISITANTE'])
        df_equipo_calc['C_EC'] = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc['CORNERS_VISITANTE'], df_equipo_calc['CORNERS_LOCAL'])
        
        # Calcular el promedio móvil
        avg_af = df_equipo_calc['C_AF'].shift(1).rolling(window=n_partidos, min_periods=1).mean()
        avg_ec = df_equipo_calc['C_EC'].shift(1).rolling(window=n_partidos, min_periods=1).mean()
        
        # --- ASIGNACIÓN DIRECTA A LA COLUMNA FINAL ---
        # Asignar a filas donde el equipo es LOCAL
        idx_local = df_equipo_indices[df_equipo_calc['Local'] == equipo]
        df_clean.loc[idx_local, 'Local_CORNERS_A_FAVOR_AVG'] = avg_af.loc[idx_local]
        df_clean.loc[idx_local, 'Local_CORNERS_EN_CONTRA_AVG'] = avg_ec.loc[idx_local]

        # Asignar a filas donde el equipo es VISITANTE
        idx_visitante = df_equipo_indices[df_equipo_calc['Visitante'] == equipo]
        df_clean.loc[idx_visitante, 'Visitante_CORNERS_A_FAVOR_AVG'] = avg_af.loc[idx_visitante]
        df_clean.loc[idx_visitante, 'Visitante_CORNERS_EN_CONTRA_AVG'] = avg_ec.loc[idx_visitante]
        
    # 5. Selección y Limpieza Final
    columnas_a_mantener = [
        'Fecha', 'Local', 'Visitante', 'Resultado_Final', 'CORNERS_TOTAL_PARTIDO', 
        'Local_CORNERS_A_FAVOR_AVG', 'Local_CORNERS_EN_CONTRA_AVG',
        'Visitante_CORNERS_A_FAVOR_AVG', 'Visitante_CORNERS_EN_CONTRA_AVG'
    ]
    
    df_final = df_clean.filter(columnas_a_mantener).copy()
    
    # La limpieza final
    df_final_clean = df_final.dropna(subset=['Local_CORNERS_A_FAVOR_AVG', 'Visitante_CORNERS_A_FAVOR_AVG'])
    
    return df_final_clean


# --- SECCIÓN DE EJECUCIÓN (CON RUTA CORREGIDA) ---

# BASE_DIR es ahora '01_scripts/V1_corners/'
BASE_DIR = Path(__file__).resolve().parent

# Usamos .parent.parent para subir dos niveles y alcanzar la raíz del proyecto
PROYECTO_ROOT = BASE_DIR.parent.parent

# Rutas de entrada y salida
base_consolidada_path = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'
output_path = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_V1_corners.csv' 

# Ejecución del cálculo
datos_limpios = calcular_metricas_v1(base_consolidada_path)

if datos_limpios is not None and not datos_limpios.empty:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datos_limpios.to_csv(output_path, index=False)
    
    print("="*70)
    print("🎉 ¡BASE V1 (CÓRNERS) RE-CREADA CON ÉXITO EN LA NUEVA ESTRUCTURA!")
    print(f"El archivo se guardó en: {output_path}") 
    print(f"Total de partidos para el modelo V1: {len(datos_limpios)}")
    print("="*70)
else:
    print("Fallo en el procesamiento de datos V1. La base consolidada no tiene datos válidos después de la limpieza.")