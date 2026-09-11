# src/regression_sklearn.py

"""
Regresión con scikit-learn usando Random Forest.

"""

import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    REPO_ROOT / "data" / "processed" / "energy_efficiency_clean.csv"
)

FIG_DIR = REPO_ROOT / "outputs" / "figures_sklearn"


TARGET = "heating_load"

FEATURE_COLUMNS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_distribution",
]

CATEGORICAL_COLUMNS = [
    "orientation",
    "glazing_distribution",
]

NUMERIC_COLUMNS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "glazing_area",
]


def load_processed_data(path=PROCESSED_DATA_PATH):
    """Carga el dataset procesado por transformation.py."""
    return pd.read_csv(path)


def split_train_validation_test(X, y, random_state=42):
    """Divide los datos en entrenamiento, validación y prueba (80/10/10).

    La semilla fija permite reproducir la misma partición en cada ejecución.
    """
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(X))
    n_train = int(len(X) * 0.80)
    n_validation = (len(X) - n_train) // 2
    train_idx = indices[:n_train]
    validation_idx = indices[n_train:n_train + n_validation]
    test_idx = indices[n_train + n_validation:]
    return (
        X.iloc[train_idx], X.iloc[validation_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[validation_idx], y.iloc[test_idx],
    )


def evaluate_regression(y_true, y_pred):
    """Calcula métricas de regresión."""
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
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def create_one_hot_encoder():
    """
    Crea OneHotEncoder compatible con versiones nuevas y antiguas
    de scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def plot_real_vs_predicho(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    results_train,
    results_test,
    fig_dir,
    verbose=True,
):
    """Gráfica 1: Real vs Predicho para Train y Test."""

    y_train_arr = (
        y_train.values if hasattr(y_train, "values") else np.array(y_train)
    )
    y_test_arr = (
        y_test.values if hasattr(y_test, "values") else np.array(y_test)
    )

    min_val = min(
        y_train_arr.min(),
        y_test_arr.min(),
        y_pred_train.min(),
        y_pred_test.min(),
    )

    max_val = max(
        y_train_arr.max(),
        y_test_arr.max(),
        y_pred_train.max(),
        y_pred_test.max(),
    )

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(
        y_train_arr,
        y_pred_train,
        alpha=0.6,
        color="blue",
        edgecolors="k",
        linewidth=0.5,
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Predicción perfecta",
    )
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(
        f"Train: Real vs Predicho\n"
        f"MAE={results_train['MAE']:.2f} | R2={results_train['R2']:.3f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(
        y_test_arr,
        y_pred_test,
        alpha=0.6,
        color="green",
        edgecolors="k",
        linewidth=0.5,
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Predicción perfecta",
    )
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(
        f"Test: Real vs Predicho\n"
        f"MAE={results_test['MAE']:.2f} | R2={results_test['R2']:.3f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = fig_dir / "01_rf_real_vs_predicho.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[FIG] Guardada: {fig_path}")


def plot_histograma_residuos(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    fig_dir,
    verbose=True,
):
    """Gráfica 2: Histograma de residuos."""

    residuos_train = y_train - y_pred_train
    residuos_test = y_test - y_pred_test

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.hist(
        residuos_train,
        bins=30,
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    plt.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Distribución de residuos - Train")
    plt.xlabel("Error = Real - Predicho")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(
        residuos_test,
        bins=30,
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    plt.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Distribución de residuos - Test")
    plt.xlabel("Error = Real - Predicho")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = fig_dir / "02_rf_histograma_residuos.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[FIG] Guardada: {fig_path}")


def plot_comparacion_metricas(
    results_train,
    results_validation,
    results_test,
    fig_dir,
    verbose=True,
):
    """Gráfica 5: Comparación de métricas Train vs Test."""

    metricas = ["MAE", "RMSE", "R2"]

    train_values = [
        results_train["MAE"],
        results_train["RMSE"],
        results_train["R2"],
    ]

    validation_values = [
        results_validation["MAE"],
        results_validation["RMSE"],
        results_validation["R2"],
    ]

    test_values = [
        results_test["MAE"],
        results_test["RMSE"],
        results_test["R2"],
    ]

    x = np.arange(len(metricas))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    barras_train = ax.bar(
        x - width,
        train_values,
        width,
        label="Train",
        color="blue",
        alpha=0.7,
    )

    barras_validation = ax.bar(
        x,
        validation_values,
        width,
        label="Validation",
        color="orange",
        alpha=0.7,
    )

    barras_test = ax.bar(
        x + width,
        test_values,
        width,
        label="Test",
        color="green",
        alpha=0.7,
    )

    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas - Random Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax.bar_label(barras_train, padding=3, fmt="%.3f")
    ax.bar_label(barras_validation, padding=3, fmt="%.3f")
    ax.bar_label(barras_test, padding=3, fmt="%.3f")

    plt.tight_layout()

    fig_path = fig_dir / "05_rf_comparacion_metricas.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[FIG] Guardada: {fig_path}")


def run_regression_sklearn(
    path=PROCESSED_DATA_PATH,
    verbose=True,
    save_figs=True,
):
    """Entrena Random Forest con sklearn y genera solo 3 gráficas."""

    df = load_processed_data(path)

    if verbose:
        print("\n" + "=" * 70)
        print(" REGRESION SKLEARN - Energy Efficiency")
        print("=" * 70)
        print(
            f"\n[INFO] Dataset procesado: "
            f"{df.shape[0]} filas x {df.shape[1]} columnas"
        )
        print(f"[INFO] Features: {FEATURE_COLUMNS}")
        print(f"[INFO] Target: {TARGET}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    if verbose:
        print(f"\n[INFO] X.shape={X.shape} y.shape={y.shape}")

    (
        X_train, X_validation, X_test,
        y_train, y_validation, y_test,
    ) = split_train_validation_test(X, y)

    if verbose:
        print(
            f"[INFO] Train: {len(X_train)} | "
            f"Validation: {len(X_validation)} | Test: {len(X_test)}"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                create_one_hot_encoder(),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    )


    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if verbose:
        print("\n[INFO] Modelo: RandomForestRegressor")
        print(
            "[INFO] Hiperparámetros: "
            "n_estimators=300, max_depth=15, random_state=42"
        )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_validation = model.predict(X_validation)
    y_pred_test = model.predict(X_test)

    results_train = evaluate_regression(y_train, y_pred_train)
    results_validation = evaluate_regression(y_validation, y_pred_validation)
    results_test = evaluate_regression(y_test, y_pred_test)

    if verbose:
        print("\n=== Random Forest (sklearn) ===")
        print(f"Train: {results_train}")
        print(f"Validation: {results_validation}")
        print(f"Test:  {results_test}")

        y_train_arr = (
            y_train.values if hasattr(y_train, "values") else np.array(y_train)
        )
        y_test_arr = (
            y_test.values if hasattr(y_test, "values") else np.array(y_test)
        )

        print("\n" + "=" * 50)
        print("PRIMERAS 10 MUESTRAS - TRAIN (Real vs Predicho)")
        print("=" * 50)
        print(
            pd.DataFrame(
                {
                    "Real (y)": y_train_arr[:10],
                    "Predicción (ŷ)": y_pred_train[:10],
                    "Error": y_train_arr[:10] - y_pred_train[:10],
                }
            ).to_string(index=False)
        )

        print("\n" + "=" * 50)
        print("PRIMERAS 10 MUESTRAS - TEST (Real vs Predicho)")
        print("=" * 50)
        print(
            pd.DataFrame(
                {
                    "Real (y)": y_test_arr[:10],
                    "Predicción (ŷ)": y_pred_test[:10],
                    "Error": y_test_arr[:10] - y_pred_test[:10],
                }
            ).to_string(index=False)
        )

        print("\n" + "=" * 50)
        print("ÚLTIMAS 10 MUESTRAS - TRAIN (Real vs Predicho)")
        print("=" * 50)
        print(
            pd.DataFrame(
                {
                    "Real (y)": y_train_arr[-10:],
                    "Predicción (ŷ)": y_pred_train[-10:],
                    "Error": y_train_arr[-10:] - y_pred_train[-10:],
                }
            ).to_string(index=False)
        )

        print("\n" + "=" * 50)
        print("ÚLTIMAS 10 MUESTRAS - TEST (Real vs Predicho)")
        print("=" * 50)
        print(
            pd.DataFrame(
                {
                    "Real (y)": y_test_arr[-10:],
                    "Predicción (ŷ)": y_pred_test[-10:],
                    "Error": y_test_arr[-10:] - y_pred_test[-10:],
                }
            ).to_string(index=False)
        )

        edificio_nuevo = {
            "relative_compactness": 0.82,
            "surface_area": 612.5,
            "wall_area": 318.5,
            "roof_area": 147.0,
            "overall_height": 7,
            "glazing_area": 0.10,
            "orientation": 2,
            "glazing_distribution": 2,
        }

        df_nuevo = pd.DataFrame([edificio_nuevo])
        df_nuevo = df_nuevo[FEATURE_COLUMNS]

        pred = model.predict(df_nuevo)[0]

        print(
            f"\n[DEMO] Predicción edificio nuevo {edificio_nuevo} "
            f"-> heating_load ≈ {pred:.2f}"
        )

    if save_figs:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        plot_real_vs_predicho(
            y_train,
            y_pred_train,
            y_test,
            y_pred_test,
            results_train,
            results_test,
            FIG_DIR,
            verbose=verbose,
        )

        plot_histograma_residuos(
            y_train,
            y_pred_train,
            y_test,
            y_pred_test,
            FIG_DIR,
            verbose=verbose,
        )

        plot_comparacion_metricas(
            results_train,
            results_validation,
            results_test,
            FIG_DIR,
            verbose=verbose,
        )

    if verbose:
        print("\n[OK] Regresion completada.\n")

    return {
        "model": model,
        "metrics_train": results_train,
        "metrics_validation": results_validation,
        "metrics_test": results_test,
        "y_pred_train": y_pred_train,
        "y_pred_validation": y_pred_validation,
        "y_pred_test": y_pred_test,
    }


if __name__ == "__main__":
    run_regression_sklearn()
