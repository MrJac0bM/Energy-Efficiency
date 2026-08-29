"""Transformacion del dataset Energy Efficiency.

Modulo para renombrado de columnas, conversion de tipos categoricos,
validacion y generacion del dataset limpio.
"""

import pandas as pd


RAW_DATA_PATH = '../data/raw/energy_efficiency.xlsx'
PROCESSED_DATA_PATH = '../data/processed/energy_efficiency_clean.csv'

COLUMN_MAPPING = {
    'X1': 'relative_compactness',
    'X2': 'surface_area',
    'X3': 'wall_area',
    'X4': 'roof_area',
    'X5': 'overall_height',
    'X6': 'orientation',
    'X7': 'glazing_area',
    'X8': 'glazing_distribution',
    'Y1': 'heating_load',
    'Y2': 'cooling_load',
}

CATEGORICAL_COLS = ['orientation', 'glazing_distribution']


def load_raw_data(path=RAW_DATA_PATH):
    """Carga el dataset crudo.

    Args:
        path: Ruta al archivo Excel crudo.

    Returns:
        DataFrame crudo.
    """
    return pd.read_excel(path)


def rename_columns(df, mapping=COLUMN_MAPPING):
    """Renombra columnas segun el mapeo definido.

    Args:
        df: DataFrame de entrada.
        mapping: Diccionario de mapeo de nombres.

    Returns:
        DataFrame con columnas renombradas.
    """
    return df.rename(columns=mapping)


def convert_to_category(df, cols=CATEGORICAL_COLS):
    """Convierte columnas indicadas a tipo category.

    Args:
        df: DataFrame de entrada.
        cols: Lista de columnas categoricas.

    Returns:
        DataFrame con tipos convertidos.
    """
    for col in cols:
        df[col] = df[col].astype('category')
    return df


def validate(df):
    """Valida dimensiones, nulos y duplicados.

    Args:
        df: DataFrame a validar.

    Returns:
        Diccionario con resultados de validacion.
    """
    return {
        'shape': df.shape,
        'nulls': df.isna().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes,
    }


def save_processed(df, path=PROCESSED_DATA_PATH):
    """Guarda el dataset procesado en CSV.

    Args:
        df: DataFrame procesado.
        path: Ruta de salida.
    """
    df.to_csv(path, index=False)


def run_transformation(
    input_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH,
    verbose=True,
):
    """Ejecuta el flujo completo de transformacion con metricas.

    Args:
        input_path: Ruta del archivo crudo.
        output_path: Ruta del archivo procesado.
        verbose: Si True imprime metricas como en 02_transformation.ipynb.

    Returns:
        DataFrame transformado.
    """
    import pathlib as _pl
    df = load_raw_data(input_path)

    if verbose:
        print("\n" + "="*70)
        print(" TRANSFORMACION - Energy Efficiency")
        print("="*70)
        print(f"\n[INFO] Crudo: {df.shape[0]} filas x {df.shape[1]} columnas")
        print("\n--- tail() crudo ---")
        print(df.tail().to_string())

    df = rename_columns(df)
    df = convert_to_category(df)
    info = validate(df)

    if verbose:
        print("\n--- dtypes tras conversion ---")
        print(info['dtypes'].to_string())
        print("\n--- describe() ---")
        print(df.describe().to_string())
        print("\n--- info() ---")
        df.info()
        print("\n--- Validaciones finales ---")
        print(f"shape: {info['shape']}")
        print("nulos por columna:")
        print(info['nulls'].to_string())
        print(f"duplicados: {info['duplicates']}")

    # Asegurar que la carpeta de salida existe
    _pl.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_processed(df, output_path)

    if verbose:
        print(f"\n[OK] Guardado procesado en: {output_path}")
        print(f"[OK] Filas: {df.shape[0]} | Columnas: {df.shape[1]}\n")

    return df


if __name__ == "__main__":
    run_transformation()
