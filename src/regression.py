

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROCESSED_DATA_PATH = '../data/processed/energy_efficiency_clean.csv'

FEATURE_COLUMNS = [
    'relative_compactness',
    'surface_area',
    'wall_area',
    'roof_area',
    'overall_height',
    'orientation',
    'glazing_area',
    'glazing_distribution',
]

TARGET = 'heating_load'

CATEGORICAL_COLUMNS = ['orientation', 'glazing_distribution']

NUMERIC_COLUMNS = [
    'relative_compactness',
    'surface_area',
    'wall_area',
    'roof_area',
    'overall_height',
    'glazing_area',
]


def load_processed_data(path=PROCESSED_DATA_PATH):
    return pd.read_csv(path)


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Divide los datos en entrenamiento y prueba de forma aleatoria.

    Args:
        X: Matriz de caracteristicas.
        y: Vector objetivo.
        test_size: Proporcion del conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Returns:
        Tupla con (X_train, X_test, y_train, y_test).
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test


def one_hot_encode_manual(X_train, X_test, categorical_columns):

    X_train = X_train.copy()
    X_test = X_test.copy()
    for column in categorical_columns:
        categories = sorted(X_train[column].unique())[1:]
        for category in categories:
            new_column = f'{column}_{category}'
            X_train[new_column] = (X_train[column] == category).astype(int)
            X_test[new_column] = (X_test[column] == category).astype(int)
        X_train.drop(columns=[column], inplace=True)
        X_test.drop(columns=[column], inplace=True)
    return X_train, X_test


def scale_manual(X_train, X_test, columns_to_scale):
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaling_params = {}
    for column in columns_to_scale:
        mean = X_train[column].mean()
        std = X_train[column].std()
        scaling_params[column] = {'mean': mean, 'std': std}
        X_train[column] = (X_train[column] - mean) / std
        X_test[column] = (X_test[column] - mean) / std
    return X_train, X_test, scaling_params


def add_intercept(X):

    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    m = X_arr.shape[0]
    n = X_arr.shape[1]
    X_b = np.ones((m, n + 1))
    for i in range(m):
        for j in range(n):
            X_b[i, j + 1] = X_arr[i, j]
    return X_b


def predict_linear(X, theta):

    X_b = add_intercept(X)
    m = X_b.shape[0]
    n = X_b.shape[1]
    predictions = np.zeros(m)
    for i in range(m):
        suma = 0.0
        for j in range(n):
            suma += X_b[i, j] * theta[j]
        predictions[i] = suma
    return predictions


def compute_cost(X_b, y, theta):

    m = len(y)
    n = X_b.shape[1]
    total_error = 0.0
    for i in range(m):
        pred = 0.0
        for j in range(n):
            pred += X_b[i, j] * theta[j]
        error = pred - y[i]
        total_error += error ** 2
    return total_error / (2 * m)


def gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    n_iterations=100000,
):
    X_b = add_intercept(X_train)
    y_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    m = len(y_arr)
    n = X_b.shape[1]
    theta = np.zeros(n)
    cost_history = []

    for _ in range(n_iterations):
        # 1. Calcular predicciones
        predictions = np.zeros(m)
        for i in range(m):
            suma = 0.0
            for j in range(n):
                suma += X_b[i, j] * theta[j]
            predictions[i] = suma

        # 2. Calcular errores
        errors = np.zeros(m)
        for i in range(m):
            errors[i] = predictions[i] - y_arr[i]

        # 3. Calcular gradiente para cada theta_j
        gradient = np.zeros(n)
        for j in range(n):
            suma_grad = 0.0
            for i in range(m):
                suma_grad += X_b[i, j] * errors[i]
            gradient[j] = suma_grad / m

        # 4. Actualizar theta
        for j in range(n):
            theta[j] = theta[j] - learning_rate * gradient[j]

        # 5. Calcular costo
        cost = compute_cost(X_b, y_arr, theta)
        cost_history.append(cost)

    return theta, cost_history


def evaluate_regression(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }


def predict_building(
    datos_usuario,
    X_train_encoded,
    X_train_scaled,
    scaling_params,
    theta,
    categorical_columns=CATEGORICAL_COLUMNS,
    numeric_columns=NUMERIC_COLUMNS,
):
    df = pd.DataFrame([datos_usuario])
    for col in categorical_columns:
        dummy_cols = [
            c for c in X_train_encoded.columns if c.startswith(col + '_')
        ]
        for dummy in dummy_cols:
            category = dummy.split('_', 1)[1]
            try:
                category_val = int(category)
                is_match = df[col].iloc[0] == category_val
            except ValueError:
                is_match = df[col].iloc[0] == category
            df[dummy] = int(is_match)
        df.drop(columns=[col], inplace=True)
    for col in numeric_columns:
        mean = scaling_params[col]['mean']
        std = scaling_params[col]['std']
        df[col] = (df[col] - mean) / std
    df = df.reindex(columns=X_train_scaled.columns, fill_value=0)

    X_b = add_intercept(df)
    m = X_b.shape[0]
    n = X_b.shape[1]
    predicciones = np.zeros(m)
    for i in range(m):
        suma = 0.0
        for j in range(n):
            suma += X_b[i, j] * theta[j]
        predicciones[i] = suma
    return predicciones[0]


def plot_cost_history(cost_history):
    """Grafica la convergencia del costo.

    Args:
        cost_history: Historial de costo por iteracion.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iteracion')
    plt.ylabel('Costo (MSE/2)')
    plt.title('Convergencia del Gradient Descent')
    plt.grid(True)
    plt.show()


def plot_predictions(
    y_train, y_pred_train, y_test, y_pred_test, mae_train, mae_test
):

    y_train_arr = (
        y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    )
    y_test_arr = (
        y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    )
    min_val = min(y_train_arr.min(), y_test_arr.min())
    max_val = max(y_train_arr.max(), y_test_arr.max())
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(
        y_train_arr,
        y_pred_train,
        alpha=0.6,
        color='blue',
        edgecolors='k',
        linewidth=0.5,
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label='Prediccion Perfecta (y=x)',
    )
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.title(f'Train: Real vs Predicho (MAE = {mae_train:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.scatter(
        y_test_arr,
        y_pred_test,
        alpha=0.6,
        color='green',
        edgecolors='k',
        linewidth=0.5,
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label='Prediccion Perfecta (y=x)',
    )
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.title(f'Test: Real vs Predicho (MAE = {mae_test:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_regression(path=PROCESSED_DATA_PATH, verbose=True, save_figs=True):

    import pathlib as _pl
    df = load_processed_data(path)
    if verbose:
        print("\n" + "="*70)
        print(" REGRESION - Energy Efficiency")
        print("="*70)
        print(f"\n[INFO] Dataset procesado: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"[INFO] Features: {FEATURE_COLUMNS}")
        print(f"[INFO] Target: {TARGET}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    if verbose:
        print(f"\n[INFO] X.shape={X.shape} y.shape={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    if verbose:
        print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    X_train_encoded, X_test_encoded = one_hot_encode_manual(
        X_train, X_test, CATEGORICAL_COLUMNS,
    )
    X_train_scaled, X_test_scaled, scaling_params = scale_manual(
        X_train_encoded, X_test_encoded, NUMERIC_COLUMNS,
    )
    if verbose:
        print("\n--- verificacion escalado (media~0, std~1 en train) ---")
        print(X_train_scaled[NUMERIC_COLUMNS].mean().to_string())
        print(X_train_scaled[NUMERIC_COLUMNS].std().to_string())

    theta_gd, cost_history = gradient_descent(
        X_train_scaled, y_train, learning_rate=0.001, n_iterations=25000,
    )

    if verbose:
        print("\nTheta con Gradient Descent:")
        print(theta_gd)
        print(f"\n[INFO] Costo inicial: {cost_history[0]:.4f} -> Costo final: {cost_history[-1]:.4f}")

    # Guardado de figuras (para que no se pierdan en headless)
    _orig_show = None
    fig_dir = None
    if save_figs:
        repo_root = _pl.Path(__file__).resolve().parent.parent
        fig_dir = repo_root / "outputs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _orig_show = plt.show
        _counter = {"n": 0}
        def _show_and_save(*args, **kwargs):
            _counter["n"] += 1
            try:
                fname = fig_dir / f"regression_fig_{_counter['n']:02d}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                if verbose:
                    print(f"[FIG] Guardada: {fname}")
            except Exception as e:
                print(f"[WARN] No se pudo guardar figura: {e}")
            try:
                plt.close("all")
            except Exception:
                pass
            return None
        plt.show = _show_and_save

    try:
        plot_cost_history(cost_history)
        y_pred_train = predict_linear(X_train_scaled, theta_gd)
        y_pred_test = predict_linear(X_test_scaled, theta_gd)
        results_train = evaluate_regression(y_train, y_pred_train)
        results_test = evaluate_regression(y_test, y_pred_test)

        if verbose:
            print("\n=== Gradient Descent ===")
            print(f"Train: {results_train}")
            print(f"Test:  {results_test}")

            # Tablas Real vs Predicho (primeras y ultimas 10)
            y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
            y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
            print("\n" + "="*50)
            print("PRIMERAS 10 MUESTRAS - TRAIN (Real vs Predicho)")
            print("="*50)
            print(pd.DataFrame({
                'Real (y)': y_train_arr[:10],
                'Predicción (ŷ)': y_pred_train[:10],
                'Error': y_train_arr[:10] - y_pred_train[:10]
            }).to_string(index=False))
            print("\n" + "="*50)
            print("PRIMERAS 10 MUESTRAS - TEST (Real vs Predicho)")
            print("="*50)
            print(pd.DataFrame({
                'Real (y)': y_test_arr[:10],
                'Predicción (ŷ)': y_pred_test[:10],
                'Error': y_test_arr[:10] - y_pred_test[:10]
            }).to_string(index=False))
            print("\n" + "="*50)
            print("ÚLTIMAS 10 MUESTRAS - TRAIN (Real vs Predicho)")
            print("="*50)
            print(pd.DataFrame({
                'Real (y)': y_train_arr[-10:],
                'Predicción (ŷ)': y_pred_train[-10:],
                'Error': y_train_arr[-10:] - y_pred_train[-10:]
            }).to_string(index=False))
            print("\n" + "="*50)
            print("ÚLTIMAS 10 MUESTRAS - TEST (Real vs Predicho)")
            print("="*50)
            print(pd.DataFrame({
                'Real (y)': y_test_arr[-10:],
                'Predicción (ŷ)': y_pred_test[-10:],
                'Error': y_test_arr[-10:] - y_pred_test[-10:]
            }).to_string(index=False))

            # Demo prediccion edificio nuevo
            edificio_nuevo = {
                'relative_compactness': 0.82,
                'surface_area': 612.5,
                'wall_area': 318.5,
                'roof_area': 147.0,
                'overall_height': 7,
                'glazing_area': 0.10,
                'orientation': 2,
                'glazing_distribution': 2
            }
            pred = predict_building(edificio_nuevo, X_train_encoded, X_train_scaled, scaling_params, theta_gd)
            print(f"\n[DEMO] Predicción edificio nuevo {edificio_nuevo} -> heating_load ≈ {pred:.2f}")

        plot_predictions(
            y_train, y_pred_train, y_test, y_pred_test,
            results_train['MAE'], results_test['MAE'],
        )
    finally:
        if save_figs and _orig_show is not None:
            plt.show = _orig_show
            plt.close("all")

    if verbose:
        print("\n[OK] Regresion completada.\n")

    return {
        'theta_gd': theta_gd,
        'scaling_params': scaling_params,
        'X_train_encoded': X_train_encoded,
        'X_train_scaled': X_train_scaled,
        'cost_history': cost_history,
        'metrics_train': results_train,
        'metrics_test': results_test,
    }


if __name__ == "__main__":
    run_regression()