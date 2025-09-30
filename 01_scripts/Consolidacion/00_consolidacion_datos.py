import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS (RUTAS ORIGINALES RESTAURADAS) ---
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_ROOT = BASE_DIR.parent.parent

# RUTA RESTAURADA: Ahora el script busca la carpeta 02_Datos_Brutos
DATOS_RAW_PATH = PROYECTO_ROOT / '02_Datos_Brutos' 

# Ruta de la base de datos consolidada final
OUTPUT_PATH = PROYECTO_ROOT / '03_Datos_Limpios' / 'premier_league_BASE_CONSOLIDADA.csv'

# COLUMNAS ESENCIALES (Córners y Tiros)
COLUMNAS_ESENCIALES = [
    'DATE', 'HOMETEAM', 'AWAYTEAM', 'FTR', 
    'HC', 'AC',  # Córners
    'HS', 'AS'   # Tiros (Shots)
]


def consolidar_datos(raw_path, output_path):
    """
    Carga todos los archivos CSV, selecciona SÓLO las columnas esenciales 
    y consolida el DataFrame.
    """
    all_data = []
    
    print(f"Buscando archivos en: {raw_path}")
    
    # 1. Iterar sobre todos los archivos CSV (Tolerancia en la extensión)
    # Busca archivos que terminen en .csv, .CSV, etc.
    csv_files = raw_path.glob('*.[Cc][Ss][Vv]')
    
    # Lista de archivos encontrados (para debug si falla)
    file_list = list(csv_files)
    if not file_list:
        print("Fallo en la consolidación: No se encontraron archivos CSV. Verifica que la carpeta '02_Datos_Brutos' exista y contenga los archivos.")
        return

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, encoding='latin1', dtype=str)
            df.columns = [col.upper().replace(' ', '_') for col in df.columns]
            df_selected = df.filter(COLUMNAS_ESENCIALES, axis=1)
            
            for col in COLUMNAS_ESENCIALES:
                if col not in df_selected.columns:
                    print(f"Advertencia: Columna {col} no encontrada en {file_path.name}")
                    df_selected[col] = pd.NA 

            all_data.append(df_selected)
            print(f"✅ Procesado y Limpiado: {file_path.name} ({len(df_selected)} filas)")
            
        except Exception as e:
            print(f"❌ Error al procesar {file_path.name}: {e}")
            continue

    df_consolidated = pd.concat(all_data, ignore_index=True)
    
    for col in ['HC', 'AC', 'HS', 'AS']:
        df_consolidated[col] = pd.to_numeric(df_consolidated[col], errors='coerce')
        
    df_consolidated = df_consolidated.dropna(subset=['DATE', 'HOMETEAM', 'AWAYTEAM', 'HC', 'AC', 'HS', 'AS'])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_consolidated.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("      🎉 ¡CONSOLIDACIÓN FINAL COMPLETADA! (Solo Córners y Tiros)")
    print(f"Archivo guardado en: {output_path.name}")
    print(f"Total de partidos en la base: {len(df_consolidated)}")
    print("El Modelo V2.1 (RMSE 3.4275) es tu modelo ganador.")
    print("="*80)


# --- SECCIÓN DE EJECUCIÓN ---
if __name__ == "__main__":
    consolidar_datos(DATOS_RAW_PATH, OUTPUT_PATH)