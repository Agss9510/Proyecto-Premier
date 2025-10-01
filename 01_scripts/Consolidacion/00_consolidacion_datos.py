import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_ROOT = BASE_DIR.parent.parent

DATOS_RAW_PATH = PROYECTO_ROOT / '02_Datos_Brutos' 
OUTPUT_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'

# NUEVAS COLUMNAS ESENCIALES (Añadimos Tiros Libres y Offsides para calcular Totales)
COLUMNAS_ESENCIALES = [
    'DATE', 'HOMETEAM', 'AWAYTEAM', 'FTR', 
    'HC', 'AC',    # Córners (Total_Corners se calculará aquí, pero no se usará como predictora)
    'HS', 'AS',    # Tiros (Shots)
    'FT', 'AT',    # Tiros Libres (Free Kicks)
    'HO', 'AO'     # Offsides
]

# Diccionario de mapeo de nombres para estandarización (Asegúrate de que este esté completo en tu versión real)
NAME_MAPPING = {
    "Man City": "Man City", "Man Utd": "Man United", "Spurs": "Tottenham Hotspur", 
    "Nott'm Forest": "Nottingham Forest", "WBA": "West Brom", "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion", # Agrega todos los nombres inconsistentes que conozcas
}


def consolidar_datos(raw_path, output_path):
    all_data = []
    
    print(f"Buscando archivos en: {raw_path}")
    
    csv_files = raw_path.glob('*.[Cc][Ss][Vv]')
    file_list = list(csv_files)
    
    if not file_list:
        print("Fallo en la consolidación: No se encontraron archivos CSV.")
        return

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, encoding='latin1', dtype=str)
            df.columns = [col.upper().replace(' ', '_') for col in df.columns]
            df_selected = df.filter(COLUMNAS_ESENCIALES, axis=1)
            
            # 1. Manejo de columnas faltantes
            for col in COLUMNAS_ESENCIALES:
                if col not in df_selected.columns:
                    # Rellenar con NA si falta alguna columna esencial
                    df_selected[col] = pd.NA 

            all_data.append(df_selected)
            print(f"✅ Procesado y Limpiado: {file_path.name} ({len(df_selected)} filas)")
            
        except Exception as e:
            print(f"❌ Error al procesar {file_path.name}: {e}")
            continue

    df_consolidated = pd.concat(all_data, ignore_index=True)
    
    # 2. Convertir todas las columnas de estadísticas a numérico
    stats_cols = ['HC', 'AC', 'HS', 'AS', 'FT', 'AT', 'HO', 'AO']
    for col in stats_cols:
        df_consolidated[col] = pd.to_numeric(df_consolidated[col], errors='coerce')
        
    # 3. Calcular las Columnas Totales del Partido
    df_consolidated['Total_Tiros'] = df_consolidated['HS'] + df_consolidated['AS'] # HS/AS
    df_consolidated['Total_Tiros_Libres'] = df_consolidated['FT'] + df_consolidated['AT'] # FT/AT
    df_consolidated['Total_Offsides'] = df_consolidated['HO'] + df_consolidated['AO'] # HO/AO
    df_consolidated['Total_Corners'] = df_consolidated['HC'] + df_consolidated['AC'] # HC/AC
    
    
    # 4. Limpieza final y estandarización
    df_consolidated = df_consolidated.dropna(subset=['DATE', 'HOMETEAM', 'AWAYTEAM', 'HC', 'AC', 'HS', 'AS'])
    
    df_consolidated['DATE'] = pd.to_datetime(df_consolidated['DATE'], dayfirst=True)
    df_consolidated['HOMETEAM'] = df_consolidated['HOMETEAM'].replace(NAME_MAPPING)
    df_consolidated['AWAYTEAM'] = df_consolidated['AWAYTEAM'].replace(NAME_MAPPING)
    
    df_consolidado = df_consolidated.sort_values(by='DATE').reset_index(drop=True)
    
    # Renombrar a español (solo las columnas que usará el modelo)
    df_consolidado.columns = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                              'HC', 'AC', 'HST', 'AST', # HST/AST se renombran aquí para consistencia con V5.1
                              'FT_H', 'FT_A', 'OFF_H', 'OFF_A',
                              'Total_Tiros', 'Total_Tiros_Libres', 'Total_Offsides', 'Total_Corners']
    
    # Eliminar columnas duplicadas y reordenar para la salida final (solo las originales + las nuevas Totales)
    cols_final = ['Fecha', 'Local', 'Visitante', 'Resultado_Final', 
                  'HC', 'AC', 'HST', 'AST', 'FT_H', 'FT_A', 'OFF_H', 'OFF_A',
                  'Total_Tiros', 'Total_Tiros_Libres', 'Total_Offsides', 'Total_Corners']
    
    df_consolidado = df_consolidado[cols_final]
    
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_consolidado.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("      🎉 ¡CONSOLIDACIÓN V6.1 COMPLETADA! (Incluye Totales)")
    print(f"Archivo guardado en: {output_path.name}")
    print(f"Columnas Totales Creadas: Total_Tiros, Total_Tiros_Libres, Total_Offsides")
    print(f"Total de partidos listos: {len(df_consolidado)}")
    print("="*80)


# --- SECCIÓN DE EJECUCIÓN ---
if __name__ == "__main__":
    consolidar_datos(DATOS_RAW_PATH, OUTPUT_PATH)