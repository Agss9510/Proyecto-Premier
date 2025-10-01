import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
# Retrocede un directorio desde '01_scripts' para llegar a la raíz del proyecto
PROYECTO_ROOT = BASE_DIR.parent 

# Ruta a la base consolidada
BASE_CONSOLIDADA_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'

# --- COEFICIENTES DEL MODELO POISSON V6.0 ---
COEFS_V6 = {
    'Local_CORNERS_AF_AVG': 0.0098, 'Local_CORNERS_EC_AVG': 0.0172,
    'Visitante_CORNERS_AF_AVG': 0.0172, 'Visitante_CORNERS_EC_AVG': 0.0098,
    'Local_ST_AF_AVG': 0.0052, 'Local_ST_EC_AVG': 0.0011,
    'Visitante_ST_AF_AVG': 0.0011, 'Visitante_ST_EC_AVG': 0.0052,
    'FACTOR_LOCAL': 1.8835
}

# --- PARÁMETROS DE CÁLCULO ---
N_CORNERS = 5 
N_ST = 10
UMBRALES_ENTEROS = [7, 8, 9, 10, 11, 12] # Umbrales X.5 a revisar (de 7.5 a 12.5)

# --- FUNCIONES BASE DEL MODELO V6.0 ---

def calcular_promedios_instantaneos(df_historial, equipo, metrica, N):
    """Calcula el promedio móvil FINAL (previo al partido) para un equipo, basado en N partidos."""
    
    df_equipo_calc = df_historial[(df_historial['Local'] == equipo) | (df_historial['Visitante'] == equipo)].copy()

    if df_equipo_calc.empty:
        # Manejo de equipos sin historial
        return {f'{metrica}_AF_AVG': np.nan, f'{metrica}_EC_AVG': np.nan}

    if metrica == 'HC':
        col_home, col_away = 'HC', 'AC'
    else:
        col_home, col_away = metrica + '_H', metrica + '_A'
    
    # Calcular AF (Atacando/A favor) y EC (En contra) para cada partido
    af = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_home], df_equipo_calc[col_away])
    ec = np.where(df_equipo_calc['Local'] == equipo, df_equipo_calc[col_away], df_equipo_calc[col_home])
    
    af_series, ec_series = pd.Series(af), pd.Series(ec)
    
    # Aplicar promedio móvil .shift(1) para usar el promedio ANTES del partido.
    avg_af = af_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]
    avg_ec = ec_series.shift(1).rolling(window=N, min_periods=1).mean().iloc[-1]

    return {f'{metrica}_AF_AVG': avg_af, f'{metrica}_EC_AVG': avg_ec}


def calcular_lambda(row):
    """Calcula la Tasa de Córners Esperada (lambda) para un partido usando el Modelo Poisson."""
    
    if row.isnull().any():
        return np.nan

    # Cálculo basado en los coeficientes V6 que has compartido
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

# --- FUNCIÓN PRINCIPAL DE PREDICCIÓN (Adaptada para un solo partido) ---

def predecir_partido_unico(local, visitante):
    """Calcula el Lambda y todas las PM para un solo partido."""
    try:
        df_historial = pd.read_csv(BASE_CONSOLIDADA_PATH)
        # Aseguramos que las columnas sean las esperadas por el modelo V6.0
        df_historial.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                                'HC', 'AC', 'ST_H', 'ST_A', 'FT_H', 'FT_A', 'OFF_H', 'OFF_A',
                                'Total_Tiros', 'Total_Tiros_Libres', 'Total_Offsides', 'Total_Corners'] 
        df_historial['Fecha'] = pd.to_datetime(df_historial['Fecha']).sort_values()
    except Exception as e:
        print(f"🚨 ERROR al cargar la base consolidada: {e}")
        return None

    # 1. Calcular Métricas para ambos equipos
    c_local = calcular_promedios_instantaneos(df_historial, local, 'HC', N_CORNERS)
    st_local = calcular_promedios_instantaneos(df_historial, local, 'ST', N_ST)
    c_visitante = calcular_promedios_instantaneos(df_historial, visitante, 'HC', N_CORNERS)
    st_visitante = calcular_promedios_instantaneos(df_historial, visitante, 'ST', N_ST)
    
    metricas = {
        'Local': [local], 'Visitante': [visitante],
        'Local_CORNERS_AF_AVG': c_local['HC_AF_AVG'], 'Local_CORNERS_EC_AVG': c_local['HC_EC_AVG'],
        'Visitante_CORNERS_AF_AVG': c_visitante['HC_AF_AVG'], 'Visitante_CORNERS_EC_AVG': c_visitante['HC_EC_AVG'],
        'Local_ST_AF_AVG': st_local['ST_AF_AVG'], 'Local_ST_EC_AVG': st_local['ST_EC_AVG'],
        'Visitante_ST_AF_AVG': st_visitante['ST_AF_AVG'], 'Visitante_ST_EC_AVG': st_visitante['ST_EC_AVG'],
        'FACTOR_LOCAL': 1.0 
    }
    df_prediccion = pd.DataFrame(metricas)

    # 2. Calcular Lambda
    df_prediccion['Lambda'] = df_prediccion.apply(calcular_lambda, axis=1)
    lambda_val = df_prediccion['Lambda'].iloc[0]

    if np.isnan(lambda_val):
        print(f"⚠️ No se pudo calcular Lambda. Uno de los equipos ('{local}' o '{visitante}') no se encontró en el historial de datos.")
        return None

    # 3. Calcular TODAS las Probabilidades de Umbral (P.M.)
    probabilidades = {'Lambda': lambda_val}
    for X in UMBRALES_ENTEROS:
        # P(Córners > X) = P(Córners >= X + 1) -> poisson.sf(X, lambda)
        probabilidades[f'Prob_MAS_{X}_5'] = poisson.sf(X, lambda_val)
        # P(Córners <= X) -> poisson.cdf(X, lambda)
        probabilidades[f'Prob_MENOS_{X}_5'] = poisson.cdf(X, lambda_val)
        
    return probabilidades

# --- FUNCIÓN DE ANÁLISIS DE VALOR KELLY ---

def analizar_kelly(probabilidades, cuotas):
    """Aplica Kelly y devuelve el umbral óptimo."""
    UMBRALES_FLOAT = [float(f"{X}.5") for X in UMBRALES_ENTEROS]
    resultados_kelly = []
    
    for umbral in UMBRALES_FLOAT:
        X = int(umbral - 0.5) 
        
        # Más de X.5
        p_m_mas = probabilidades[f'Prob_MAS_{X}_5']
        c_mas = cuotas.get(f'Mas_{umbral}', 0)
        # Fórmula de Kelly (f): f = (P*C - 1) / (C - 1)
        f_mas = (p_m_mas * c_mas - 1) / (c_mas - 1) if c_mas > 1 else -1 
        
        if f_mas > 0:
            resultados_kelly.append({
                'Umbral': f'Más {umbral}', 'Cuota': c_mas,
                'Prob_Modelo': p_m_mas, 'Kelly_Media': f_mas / 2 
            })

        # Menos de X.5
        p_m_menos = probabilidades[f'Prob_MENOS_{X}_5']
        c_menos = cuotas.get(f'Menos_{umbral}', 0)
        f_menos = (p_m_menos * c_menos - 1) / (c_menos - 1) if c_menos > 1 else -1 
        
        if f_menos > 0:
            resultados_kelly.append({
                'Umbral': f'Menos {umbral}', 'Cuota': c_menos,
                'Prob_Modelo': p_m_menos, 'Kelly_Media': f_menos / 2 
            })
            
    if not resultados_kelly:
        return None, None
        
    df_resultados = pd.DataFrame(resultados_kelly)
    # Encontrar el máximo valor Kelly (la mejor apuesta)
    optimo = df_resultados.loc[df_resultados['Kelly_Media'].idxmax()]
    
    # Calcular el Peso Relativo (para mostrar el % de riesgo que lleva cada apuesta con valor)
    df_resultados['Peso_Relativo'] = (df_resultados['Kelly_Media'] / df_resultados['Kelly_Media'].sum()) * 100
    
    return optimo, df_resultados.sort_values(by='Kelly_Media', ascending=False)

# --- ENTRADA DE DATOS DEL USUARIO ---

def obtener_cuotas_usuario():
    """Pide las cuotas al usuario y las devuelve en un diccionario."""
    print("\n--- 📝 INGRESO DE CUOTAS DE LA CASA (Solo valores numéricos) ---")
    cuotas = {}
    for X in UMBRALES_ENTEROS:
        umbral = f"{X}.5"
        try:
            # Entrada para 'Más de X.5'
            mas_c = input(f"Cuota para MÁS de {umbral} (o Enter): ")
            cuotas[f'Mas_{umbral}'] = float(mas_c.replace(',', '.')) if mas_c.strip() else 0
            
            # Entrada para 'Menos de X.5'
            menos_c = input(f"Cuota para MENOS de {umbral} (o Enter): ")
            cuotas[f'Menos_{umbral}'] = float(menos_c.replace(',', '.')) if menos_c.strip() else 0
        except ValueError:
            print("⚠️ Valor inválido. Ingresa solo números. Saliendo.")
            return None
    return cuotas

# --- EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("      🚀 ANÁLISIS DE VALOR - PARTIDO ÚNICO (CÓRNERS V6.0) 🚀")
    print("="*80)
    
    # 1. Entrada de equipos
    local = input("Introduce el nombre del equipo LOCAL (ej: Chelsea): ").strip()
    visitante = input("Introduce el nombre del equipo VISITANTE (ej: Liverpool): ").strip()
    
    if not local or not visitante:
        print("🚫 Debes ingresar ambos nombres de equipo.")
        exit()

    # 2. Calcular probabilidades (Lambda y PM)
    probabilidades = predecir_partido_unico(local, visitante)
    
    if probabilidades is None:
        print("\n🚫 Proceso abortado debido a errores en la predicción.")
        exit()
        
    print(f"\n✅ Predicción de Modelo V6.0 completada: Lambda = {probabilidades['Lambda']:.4f}")
    
    # 3. Entrada de cuotas
    cuotas = obtener_cuotas_usuario()
    if cuotas is None:
        exit()
        
    # 4. Aplicar Kelly y encontrar el óptimo
    resultado_optimo, df_todos_los_valores = analizar_kelly(probabilidades, cuotas)
    
    # 5. Mostrar Resultados
    print("\n" + "="*80)
    print(f"      📈 REPORTE DE VALOR PARA {local} vs {visitante} 🏆")
    print("="*80)
    
    if resultado_optimo is not None:
        print("--- 🥇 UMFRAL ÓPTIMO DE APUESTA (MÁXIMO KELLY) 🥇 ---")
        print(f"  Apuesta: {resultado_optimo['Umbral']}")
        print(f"  Cuota: {resultado_optimo['Cuota']:.2f}")
        print(f"  Prob. Modelo (PM): {resultado_optimo['Prob_Modelo'] * 100:.2f}%")
        print(f"  Peso Kelly (f/2): {resultado_optimo['Kelly_Media'] * 100:.2f}% del Bankroll")

        print("\n--- 🥈 TODOS LOS MERCADOS CON VALOR POSITIVO ---")
        df_display = df_todos_los_valores[['Umbral', 'Cuota', 'Prob_Modelo', 'Kelly_Media', 'Peso_Relativo']].copy()
        df_display['Prob_Modelo'] = (df_display['Prob_Modelo'] * 100).round(2).astype(str) + '%'
        df_display['Kelly_Media'] = (df_display['Kelly_Media'] * 100).round(2).astype(str) + '%'
        df_display['Peso_Relativo'] = (df_display['Peso_Relativo']).round(2).astype(str) + '%'
        print(df_display.to_string(index=False))

    else:
        print("⚠️ No se encontró ninguna apuesta con valor positivo (Kelly > 0) en los umbrales analizados.")