import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_ROOT = BASE_DIR.parent 

# Rutas de entrada y salida
BASE_CONSOLIDADA_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'
# Renombramos la salida para reflejar que las métricas totales fueron excluidas si eran el problema.
OUTPUT_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_V6_C5_ST10_FINAL.csv' 

# Parámetros del modelo V6.1
N_CORNERS = 5 
N_ST = 10 

def calcular_promedios_moviles(df, equipo, metrica, N):
    """
    Calcula los promedios móviles a favor y en contra (AF/EC) para un equipo.
    Devuelve un DataFrame con el índice original del partido para un merge seguro.
    """
    df_equipo_calc = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)].copy()

    if metrica == 'HC':
        col_home = 'HC'
        col_away = 'AC'
    else:
        col_home = metrica + '_H'
        col_away = metrica + '_A'
    
    af = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_home], df_equipo_calc[col_away])
    ec = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_away], df_equipo_calc[col_home])
    
    af_series = pd.Series(af, index=df_equipo_calc.index)
    ec_series = pd.Series(ec, index=df_equipo_calc.index)
    
    avg_af = af_series.shift(1).rolling(window=N, min_periods=1).mean()
    avg_ec = ec_series.shift(1).rolling(window=N, min_periods=1).mean()

    return pd.DataFrame({
        'Index': df_equipo_calc.index,
        f'{metrica}_AF_AVG': avg_af,
        f'{metrica}_EC_AVG': avg_ec
    }).set_index('Index')


def generar_base_modelado(base_path, output_path):
    # Cargar Base Consolidada
    df = pd.read_csv(base_path)

    # Renombrar columnas
    df.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                  'HC', 'AC', 'ST_H', 'ST_A', 'FT_H', 'FT_A', 'OFF_H', 'OFF_A',
                  'Total_Tiros', 'Total_Tiros_Libres', 'Total_Offsides', 'Total_Corners'] 
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values(by='Fecha').reset_index(drop=True)
    
    equipos = pd.concat([df['Local'], df['Visitante']]).unique()
    
    # 1. Calcular las métricas por equipo
    metricas_c = [calcular_promedios_moviles(df, equipo, 'HC', N_CORNERS) for equipo in equipos]
    metricas_st = [calcular_promedios_moviles(df, equipo, 'ST', N_ST) for equipo in equipos]

    # -----------------------------------------------------------
    # CONSOLIDACIÓN Y MERGE POR ÍNDICE
    # -----------------------------------------------------------
    
    df_c_unique = pd.concat(metricas_c).groupby(level=0).mean()
    df_st_unique = pd.concat(metricas_st).groupby(level=0).mean()

    df_modelado = df.copy()

    # 2. Unir métricas del EQUIPO LOCAL (Merge por índice)
    
    df_modelado = df_modelado.merge(df_c_unique, left_index=True, right_index=True, how='left')
    df_modelado.rename(columns={'HC_AF_AVG': 'Local_CORNERS_AF_AVG', 'HC_EC_AVG': 'Local_CORNERS_EC_AVG'}, inplace=True)

    df_modelado = df_modelado.merge(df_st_unique, left_index=True, right_index=True, how='left')
    df_modelado.rename(columns={'ST_AF_AVG': 'Local_ST_AF_AVG', 'ST_EC_AVG': 'Local_ST_EC_AVG'}, inplace=True)
    
    
    # 3. Calcular métricas del VISITANTE (Usando la relación AF/EC)
    
    df_modelado['Visitante_CORNERS_AF_AVG'] = df_modelado['Local_CORNERS_EC_AVG']
    df_modelado['Visitante_CORNERS_EC_AVG'] = df_modelado['Local_CORNERS_AF_AVG']
    
    df_modelado['Visitante_ST_AF_AVG'] = df_modelado['Local_ST_EC_AVG']
    df_modelado['Visitante_ST_EC_AVG'] = df_modelado['Local_ST_AF_AVG']

    # 4. Preparación final del DataFrame de modelado
    
    df_modelado['CORNERS_TOTAL_PARTIDO'] = df_modelado['HC'] + df_modelado['AC']
    df_modelado['FACTOR_LOCAL'] = 1 
    
    # 🚨 LISTA FINAL: Excluimos las métricas totales para garantizar filas útiles.
    columnas_modelo_final_v6 = [
        # Promedios móviles (8 columnas)
        'Local_CORNERS_AF_AVG', 'Local_CORNERS_EC_AVG', 'Visitante_CORNERS_AF_AVG', 'Visitante_CORNERS_EC_AVG',
        'Local_ST_AF_AVG', 'Local_ST_EC_AVG', 'Visitante_ST_AF_AVG', 'Visitante_ST_EC_AVG',
        
        'FACTOR_LOCAL', # Sesgo de Local
        
        'CORNERS_TOTAL_PARTIDO' # Variable dependiente (Y)
    ]
    
    # Aquí es donde se eliminan los primeros N partidos sin datos previos (lo normal).
    df_final = df_modelado.dropna(subset=columnas_modelo_final_v6).copy()
    
    # Guardar el archivo listo para modelar
    df_final[columnas_modelo_final_v6].to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("      ✅ CÁLCULO DE MÉTRICAS V6.0 COMPLETADO (¡Listo para Modelar!)")
    print(f"      Córners (N={N_CORNERS}), Tiros a Puerta (N={N_ST})")
    print(f"      Datos listos para modelar guardados en: {output_path.name}")
    print(f"      Partidos listos para modelar: {len(df_final)}")
    print("="*80)

if __name__ == "__main__":
    generar_base_modelado(BASE_CONSOLIDADA_PATH, OUTPUT_PATH)