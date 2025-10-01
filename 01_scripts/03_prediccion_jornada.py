import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_ROOT = BASE_DIR.parent 

# Archivos de entrada/salida
BASE_CONSOLIDADA_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'
OUTPUT_PROBABILIDADES_PATH = PROYECTO_ROOT / '04_Modelos_Entrenados' / 'predicciones_jornada_V6_REAL.csv'
CUOTAS_PATH = PROYECTO_ROOT / '04_Modelos_Entrenados' / 'cuotas_jornada.csv' # Archivo que debes crear

# --- COEFICIENTES DEL MODELO POISSON V6.0 ---
COEFS_V6 = {
    'Local_CORNERS_AF_AVG': 0.0098,
    'Local_CORNERS_EC_AVG': 0.0172,
    'Visitante_CORNERS_AF_AVG': 0.0172,
    'Visitante_CORNERS_EC_AVG': 0.0098,
    'Local_ST_AF_AVG': 0.0052,
    'Local_ST_EC_AVG': 0.0011,
    'Visitante_ST_AF_AVG': 0.0011,
    'Visitante_ST_EC_AVG': 0.0052,
    'FACTOR_LOCAL': 1.8835
}

# --- PARÁMETROS DE CÁLCULO ---
N_CORNERS = 5 
N_ST = 10
UMBRALES_ENTEROS = [7, 8, 9, 10, 11, 12] # Umbrales X.5 a calcular (de 7.5 a 12.5)

# 🚨 DEFINICIÓN MANUAL DE LA PRÓXIMA JORNADA 🚨
JORNADA_FUTURA = pd.DataFrame({
    'Fecha': ['2025-10-03', '2025-10-04', '2025-10-04', '2025-10-04', '2025-10-04', 
             '2025-10-05', '2025-10-05', '2025-10-05', '2025-10-05', '2025-10-05'],
    'Local': ['Bournemouth', 'Leeds', 'Arsenal', 'Man United', 'Chelsea', 
              'Aston Villa', 'Everton', 'Newcastle', 'Wolverhampton Wanderers', 'Brentford'],
    'Visitante': ['Fulham', 'Tottenham', 'West Ham', 'Sunderland', 'Liverpool', 
                  'Burnley', 'Crystal Palace', 'Nottingham Forest', 'Brighton & Hove Albion', 'Man City']
})

# --- FUNCIONES BASE DEL MODELO ---

def calcular_promedios_instantaneos(df_historial, equipo, metrica, N):
    """Calcula el promedio móvil FINAL (previo al partido) para un equipo."""
    # ... (El código de esta función es el mismo que enviaste) ...
    df_equipo_calc = df_historial[(df_historial['Local'] == equipo) | (df_historial['Visitante'] == equipo)].copy()
    if df_equipo_calc.empty:
        return {f'{metrica}_AF_AVG': np.nan, f'{metrica}_EC_AVG': np.nan}

    if metrica == 'HC':
        col_home, col_away = 'HC', 'AC'
    else:
        col_home, col_away = metrica + '_H', metrica + '_A'
    
    af = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_home], df_equipo_calc[col_away])
    ec = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_away], df_equipo_calc[col_home])
    
    af_series, ec_series = pd.Series(af), pd.Series(ec)
    avg_af = af_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]
    avg_ec = ec_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]

    return {f'{metrica}_AF_AVG': avg_af, f'{metrica}_EC_AVG': avg_ec}

def calcular_lambda(row):
    """Calcula la Tasa de Córners Esperada (lambda) para un partido."""
    if row.isnull().any():
        return np.nan

    lambda_exp = (
        COEFS_V6['Local_CORNERS_AF_AVG'] * row['Local_CORNERS_AF_AVG'] +
        COEFS_V6['Local_CORNERS_EC_AVG'] * row['Local_CORNERS_EC_AVG'] +
        COEFS_V6['Visitante_CORNERS_AF_AVG'] * row['Visitante_CORNERS_AF_AVG'] +
        COEFS_V6['Visitante_CORNERS_EC_AVG'] * row['Visitante_CORNERS_EC_AVG'] +
        COEFS_V6['Local_ST_AF_AVG'] * row['Local_ST_AF_AVG'] +
        COEFS_V6['Local_ST_EC_AVG'] * row['Local_ST_EC_AVG'] +
        COEFS_V6['Visitante_ST_AF_AVG'] * row['Visitante_ST_AF_AVG'] +
        COEFS_V6['Visitante_ST_EC_AVG'] * row['Visitante_ST_EC_AVG'] +
        COEFS_V6['FACTOR_LOCAL'] * row['FACTOR_LOCAL']
    )
    return np.exp(lambda_exp)

# --- FUNCIÓN DE ANÁLISIS DE KELLY (NUEVA) ---

def analizar_valor_kelly(df_probabilidades, df_cuotas):
    """
    Combina las probabilidades del modelo (Pm) con las cuotas (C) para calcular
    el valor de Kelly (f) y encuentra el umbral óptimo para cada partido.
    """
    UMBRALES_FLOAT = [float(f"{X}.5") for X in UMBRALES_ENTEROS]
    resultados_kelly = []

    for index, row in df_probabilidades.iterrows():
        # Formato de nombre de partido para coincidir con el CSV de cuotas
        partido_match = f"{row['Local']} vs {row['Visitante']}"
        
        # Verificar si tenemos cuotas para este partido
        if partido_match not in df_cuotas['Partido'].values:
            continue
            
        # Fila de cuotas para el partido actual
        cuotas_row = df_cuotas.loc[df_cuotas['Partido'] == partido_match].iloc[0]
        
        for umbral in UMBRALES_FLOAT:
            X = int(umbral - 0.5) 
            
            # --- CÁLCULO DE MÁS DE X.5 ---
            p_m_mas = row[f'Prob_MAS_{X}_5']
            try:
                c_mas = cuotas_row[f'Mas_{umbral}']
            except KeyError:
                continue # Saltar si la cuota no existe en el CSV
                
            # Fórmula de Kelly (f): f = (P*C - 1) / (C - 1)
            f_mas = (p_m_mas * c_mas - 1) / (c_mas - 1) if c_mas > 1 else -1 
            
            if f_mas > 0:
                resultados_kelly.append({
                    'Partido': partido_match,
                    'Umbral': f'Más {umbral}',
                    'Cuota': c_mas,
                    'Prob_Modelo': p_m_mas,
                    'Fraccion_Kelly': f_mas,
                    'Kelly_Media': f_mas / 2 
                })

            # --- CÁLCULO DE MENOS DE X.5 ---
            p_m_menos = row[f'Prob_MENOS_{X}_5']
            try:
                c_menos = cuotas_row[f'Menos_{umbral}']
            except KeyError:
                continue
                
            f_menos = (p_m_menos * c_menos - 1) / (c_menos - 1) if c_menos > 1 else -1 
            
            if f_menos > 0:
                resultados_kelly.append({
                    'Partido': partido_match,
                    'Umbral': f'Menos {umbral}',
                    'Cuota': c_menos,
                    'Prob_Modelo': p_m_menos,
                    'Fraccion_Kelly': f_menos,
                    'Kelly_Media': f_menos / 2 
                })
        
    df_resultados_final = pd.DataFrame(resultados_kelly)
    
    # Encontrar el umbral óptimo (máximo Kelly Media) para cada partido
    if df_resultados_final.empty:
        return pd.DataFrame()
        
    idx_max = df_resultados_final.groupby('Partido')['Kelly_Media'].idxmax()
    df_optimos = df_resultados_final.loc[idx_max].sort_values(by='Kelly_Media', ascending=False)
    
    return df_optimos.reset_index(drop=True)

# --- FUNCIÓN PRINCIPAL DE PREDICCIÓN ---

def predecir_jornada_real(consolidada_path, output_path, jornada_df):
    
    try:
        # 1. Cargar y limpiar el historial de partidos
        df_historial = pd.read_csv(consolidada_path)
        df_historial.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                                'HC', 'AC', 'ST_H', 'ST_A', 'FT_H', 'FT_A', 'OFF_H', 'OFF_A',
                                'Total_Tiros', 'Total_Tiros_Libres', 'Total_Offsides', 'Total_Corners'] 
        df_historial['Fecha'] = pd.to_datetime(df_historial['Fecha'])
        df_historial = df_historial.sort_values(by='Fecha').reset_index(drop=True)
    except Exception as e:
        print(f"🚨 ERROR al cargar la base consolidada: {e}")
        return
    
    df_prediccion = jornada_df.copy()
    metricas_jornada = []

    # 2. Calcular Métricas
    for index, row in df_prediccion.iterrows():
        # ... (Cálculo de métricas) ...
        local, visitante = row['Local'], row['Visitante']
        c_local = calcular_promedios_instantaneos(df_historial, local, 'HC', N_CORNERS)
        st_local = calcular_promedios_instantaneos(df_historial, local, 'ST', N_ST)
        c_visitante = calcular_promedios_instantaneos(df_historial, visitante, 'HC', N_CORNERS)
        st_visitante = calcular_promedios_instantaneos(df_historial, visitante, 'ST', N_ST)
        
        metricas = {
            'Local': local, 'Visitante': visitante,
            'Local_CORNERS_AF_AVG': c_local['HC_AF_AVG'], 'Local_CORNERS_EC_AVG': c_local['HC_EC_AVG'],
            'Visitante_CORNERS_AF_AVG': c_visitante['HC_AF_AVG'], 'Visitante_CORNERS_EC_AVG': c_visitante['HC_EC_AVG'],
            'Local_ST_AF_AVG': st_local['ST_AF_AVG'], 'Local_ST_EC_AVG': st_local['ST_EC_AVG'],
            'Visitante_ST_AF_AVG': st_visitante['ST_AF_AVG'], 'Visitante_ST_EC_AVG': st_visitante['ST_EC_AVG'],
            'FACTOR_LOCAL': 1.0 
        }
        metricas_jornada.append(metricas)

    # 3. Unir las métricas y aplicar el modelo
    df_prediccion_con_metricas = pd.DataFrame(metricas_jornada)
    df_prediccion_con_metricas['Lambda_Esperada'] = df_prediccion_con_metricas.apply(calcular_lambda, axis=1)
    
    # 4. Calcular TODAS las Probabilidades de Umbral (P.M.)
    for X in UMBRALES_ENTEROS:
        df_prediccion_con_metricas[f'Prob_MAS_{X}_5'] = poisson.sf(X, df_prediccion_con_metricas['Lambda_Esperada'])
        df_prediccion_con_metricas[f'Prob_MENOS_{X}_5'] = poisson.cdf(X, df_prediccion_con_metricas['Lambda_Esperada'])
    
    # 5. Generar Previsiones Finales
    columnas_finales = ['Local', 'Visitante', 'Lambda_Esperada'] + \
                       [col for col in df_prediccion_con_metricas.columns if 'Prob_' in col]
                       
    df_final = df_prediccion_con_metricas[columnas_finales].copy()
    df_final.rename(columns={'Lambda_Esperada': 'Lambda'}, inplace=True)
    
    # Guardar Resultados
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    return df_final


# --- EJECUCIÓN DEL SCRIPT ---

if __name__ == "__main__":
    
    # 1. Ejecutar Predicción ML y obtener la tabla de probabilidades
    df_probabilidades = predecir_jornada_real(BASE_CONSOLIDADA_PATH, OUTPUT_PROBABILIDADES_PATH, JORNADA_FUTURA)
    
    # 2. Intentar cargar las cuotas y realizar el análisis de Kelly
    print("\n" + "="*80)
    print("      ⚙️ INICIANDO ANÁLISIS DE VALOR ÓPTIMO (KELLY) ⚙️")
    print("="*80)
    
    try:
        # Cargar el archivo de cuotas que el usuario debe crear/actualizar
        df_cuotas = pd.read_csv(CUOTAS_PATH)
        df_cuotas['Partido'] = df_cuotas['Local'] + ' vs ' + df_cuotas['Visitante']
        
    except FileNotFoundError:
        print(f"\n🚨 ERROR: No se encontró el archivo de cuotas: {CUOTAS_PATH.name}")
        print("Crea y llena el archivo 'cuotas_jornada.csv' para realizar el análisis de valor.")
        exit()

    # 3. Aplicar el Criterio de Kelly a todos los umbrales
    df_valor_optimo = analizar_valor_kelly(df_probabilidades, df_cuotas)

    # 4. Mostrar Resultados Finales
    print("\n" + "="*80)
    print("      🏆 REPORTE FINAL DE APUESTAS DE VALOR 🏆")
    print("="*80)
    
    if not df_valor_optimo.empty:
        # Calcular el total del Kelly_Media para normalizar el peso de cada apuesta
        total_kelly = df_valor_optimo['Kelly_Media'].sum()
        df_valor_optimo['Peso_Relativo'] = (df_valor_optimo['Kelly_Media'] / total_kelly) * 100
        
        df_reporte = df_valor_optimo[['Partido', 'Umbral', 'Cuota', 'Prob_Modelo', 'Peso_Relativo']].copy()
        df_reporte['Prob_Modelo'] = (df_reporte['Prob_Modelo'] * 100).round(2).astype(str) + '%'
        df_reporte['Peso_Relativo'] = df_reporte['Peso_Relativo'].round(2).astype(str) + '%'
        
        print(df_reporte.to_string(index=False))
        print(f"\n✅ Total de Capital Recomendado a Invertir: {total_kelly * 100:.2f}% de tu Bankroll.")
        print("💡 Los 'Pesos Relativos' muestran cómo distribuir esa cantidad total.")
    else:
        print("⚠️ No se encontró ninguna apuesta con valor positivo (Kelly > 0) en los umbrales analizados.")