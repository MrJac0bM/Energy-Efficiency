"""Exploracion del dataset Energy Efficiency.

Modulo para la inspeccion inicial, validacion de integridad y
visualizacion de relaciones entre variables y carga
energetica.
"""

import matplotlib
matplotlib.use("Agg")  #
import matplotlib.pyplot as plt
import pandas as pd


RAW_DATA_PATH = '../data/raw/energy_efficiency.xlsx'

NUMERIC_COLS = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'Y1', 'Y2']


def load_data(path=RAW_DATA_PATH):
    """Carga el dataset desde un archivo Excel.

    Args:
        path: Ruta al archivo Excel.

    Returns:
        DataFrame con los datos cargados.
    """
    return pd.read_excel(path)


def inspect_data(df):
    """Realiza inspeccion basica del DataFrame.

    Args:
        df: DataFrame a inspeccionar.

    Returns:
        Tupla con (head, tail, info, describe, min_max).
    """
    head = df.head()
    tail = df.tail()
    describe = df.describe()
    min_max = df[NUMERIC_COLS].agg(['min', 'max'])
    return head, tail, describe, min_max


def check_integrity(df):
    """Verifica nulos y duplicados.

    Args:
        df: DataFrame a validar.

    Returns:
        Tupla con (nulos por columna, conteo de duplicados).
    """
    nulls = df.isnull().sum()
    duplicates = df.duplicated().sum()
    return nulls, duplicates


def summarize_categoricals(df):
    """Resume variables categoricas X6 y X8.

    Args:
        df: DataFrame de entrada.

    Returns:
        Diccionario con valores unicos y conteos.
    """
    return {
        'X6_unique': df['X6'].unique(),
        'X6_counts': df['X6'].value_counts(),
        'X8_unique': df['X8'].unique(),
        'X8_counts': df['X8'].value_counts(),
    }


def plot_distributions(df, cols=NUMERIC_COLS):
    """Grafica histogramas de variables numericas.

    Args:
        df: DataFrame de entrada.
        cols: Lista de columnas numericas a graficar.
    """
    df[cols].hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, cols=NUMERIC_COLS):
    """Grafica boxplots de variables numericas.

    Args:
        df: DataFrame de entrada.
        cols: Lista de columnas numericas a graficar.
    """
    plt.figure(figsize=(14, 7))
    df[cols].boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplots de valores numericos')
    plt.ylabel('Value')
    plt.show()


def plot_correlation(df, cols=NUMERIC_COLS):
    """Grafica matriz de correlacion.

    Args:
        df: DataFrame de entrada.
        cols: Lista de columnas numericas.
    """
    correlation = df[cols].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
    )
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_scatter(df, x_col, y_col, xlabel, ylabel, title):
    """Grafica diagrama de dispersion generico.

    Args:
        df: DataFrame de entrada.
        x_col: Columna para eje X.
        y_col: Columna para eje Y.
        xlabel: Etiqueta del eje X.
        ylabel: Etiqueta del eje Y.
        title: Titulo del grafico.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def run_exploration(path=RAW_DATA_PATH, verbose=True, save_figs=True):
    """Ejecuta el flujo completo de exploracion con metricas y graficas.

    Args:
        path: Ruta al archivo de datos.
        verbose: Si True imprime metricas en consola (como en 01_exploration.ipynb).
        save_figs: Si True guarda las figuras en disco ademas de mostrarlas.

    Returns:
        DataFrame explorado.
    """
    import pathlib as _pl
    df = load_data(path)

    # --- 1. Inspeccion basica ---
    head, tail, describe, min_max = inspect_data(df)
    nulls, duplicates = check_integrity(df)
    cat_summary = summarize_categoricals(df)

    if verbose:
        print("\n" + "="*70)
        print(" EXPLORACION - Energy Efficiency")
        print("="*70)
        print(f"\n[INFO] Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
        print("\n--- head() ---")
        print(head.to_string())
        print("\n--- tail() ---")
        print(tail.to_string())
        print("\n--- describe() ---")
        print(describe.to_string())
        print("\n--- min / max (numericas) ---")
        print(min_max.to_string())
        print("\n--- info() ---")
        df.info()
        print("\n--- nulos por columna ---")
        print(nulls.to_string())
        print(f"\n--- duplicados: {duplicates} ---")
        print("\n--- X6 (orientation) unique ---")
        print(cat_summary['X6_unique'])
        print(cat_summary['X6_counts'].to_string())
        print("\n--- X8 (glazing_distribution) unique ---")
        print(cat_summary['X8_unique'])
        print(cat_summary['X8_counts'].to_string())
        corr = df[NUMERIC_COLS].corr()
        print("\n--- matriz de correlacion ---")
        print(corr.to_string())

    # --- 2. Visualizaciones (con guardado para que no se pierdan en headless) ---
    _orig_show = None
    fig_dir = None
    if save_figs:
        # Directorio robusto: <repo>/outputs/figures y <repo>/data/processed/figures como fallback
        # Usar ubicacion del archivo, no del cwd
        repo_root = _pl.Path(__file__).resolve().parent.parent
        fig_dir = repo_root / "outputs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _orig_show = plt.show
        _fig_counter = {"n": 0}
        def _show_and_save(*args, **kwargs):
            _fig_counter["n"] += 1
            try:
                fname = fig_dir / f"exploration_fig_{_fig_counter['n']:02d}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                if verbose:
                    print(f"[FIG] Guardada: {fname}")
            except Exception as e:
                print(f"[WARN] No se pudo guardar figura: {e}")
            # En Agg no intentamos mostrar ventana; solo cerramos para liberar memoria
            try:
                plt.close("all")
            except Exception:
                pass
            return None
        plt.show = _show_and_save

    try:
        plot_distributions(df)
        plot_boxplots(df)
        plot_correlation(df)
        plot_scatter(
            df, 'X1', 'Y1',
            'Relative Compactness', 'Heating Load',
            'Relative Compactness vs Heating Load',
        )
        plot_scatter(
            df, 'X1', 'Y2',
            'Relative Compactness', 'Cooling Load',
            'Relative Compactness vs Cooling Load',
        )
        plot_scatter(
            df, 'X2', 'Y1',
            'Surface Area', 'Heating Load',
            'Surface Area vs Heating Load',
        )
        plot_scatter(
            df, 'X2', 'Y2',
            'Surface Area', 'Cooling Load',
            'Surface Area vs Cooling Load',
        )
        plot_scatter(
            df, 'X4', 'Y1',
            'Roof Area', 'Heating Load',
            'Roof Area vs Heating Load',
        )
        plot_scatter(
            df, 'X4', 'Y2',
            'Roof Area', 'Cooling Load',
            'Roof Area vs Cooling Load',
        )
    finally:
        if save_figs and _orig_show is not None:
            plt.show = _orig_show  # restaurar
            plt.close("all")

    if verbose:
        print("\n[OK] Exploracion completada - graficas mostradas y guardadas.\n")

    return df


if __name__ == "__main__":
    run_exploration()
