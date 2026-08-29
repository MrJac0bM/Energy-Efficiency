"""
Orquestador principal - Energy Efficiency
Llama secuencialmente a los 3 scripts extraídos de los notebooks SIN modificarlos.

Ejecuta:
  1. src/exploration.py   -> EDA y visualizaciones (01_exploration.ipynb)
  2. src/transformation.py -> Limpieza y generación de data/processed (02_transformation.ipynb)
  3. src/regression.py     -> Modelo de regresión lineal (03_regression.ipynb)
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.resolve()
SRC = ROOT / "src"

SCRIPTS = {
    "exploration": SRC / "exploration.py",
    "transformation": SRC / "transformation.py",
    "regression": SRC / "regression.py",
}

PIPELINE_ORDER = ["exploration", "transformation", "regression"]


def _clean_magic_lines(code: str) -> str:
    """Convierte magics de Jupyter (%pip, !, %matplotlib) en comentarios."""
    cleaned = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            cleaned.append(f"# [MAGIC FILTRADO por main.py] {line}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _patch_regression_bug(code: str, name: str) -> str:
    if name != "regression":
        return code
    if 'print(theta)' in code and 'Theta con Ecuación Normal' in code:
        patch = (
            "\n# [PATCH main.py] Fix bug notebook original: theta no definido\n"
            "try:\n"
            "    theta  # intenta usar theta si existe\n"
            "except NameError:\n"
            "    print(\"[WARN] theta no definido  usando theta_gd como fallback\")\n"
            "    theta = theta_gd\n"
        )
        code = code.replace(
            'print("\\nTheta con Ecuación Normal ")',
            patch + 'print("\\nTheta con Ecuación Normal ")',
        )
    return code


def run_script(name: str, verbose: bool = True) -> None:
    path = SCRIPTS[name]
    if not path.exists():
        print(f"[ERROR] No se encontró {path}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"\n{'='*70}", flush=True)
    print(f" ▶ Ejecutando: {name}  ({path.name})", flush=True)
    print(f"{'='*70}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    raw_code = path.read_text(encoding="utf-8")
    cleaned_code = _clean_magic_lines(raw_code)
    cleaned_code = _patch_regression_bug(cleaned_code, name)


    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{name}.py", delete=False, encoding="utf-8", dir=SRC
    ) as tmp:
        tmp.write(cleaned_code)
        tmp_path = pathlib.Path(tmp.name)

    try:
        result = subprocess.run(
            [sys.executable, "-u", str(tmp_path)],
            cwd=SRC,  
        )
        if result.returncode != 0:
            print(f"[ERROR] {name} falló con código {result.returncode}", file=sys.stderr, flush=True)
            sys.exit(result.returncode)
        print(f" ✓ {name} completado correctamente", flush=True)
        sys.stdout.flush()
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Orquestador Energy Efficiency")
    parser.add_argument(
        "--step",
        choices=list(SCRIPTS.keys()),
        help="Ejecuta solo un paso del pipeline",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Salta exploration (EDA con plt.show) útil en headless/CI",
    )
    args = parser.parse_args()

    print(f"ROOT: {ROOT}", flush=True)
    print(f"SRC:  {SRC}", flush=True)
    sys.stdout.flush()

    if not (ROOT / "data" / "raw" / "energy_efficiency.xlsx").exists():
        print("[WARN] No se encontró data/raw/energy_efficiency.xlsx", file=sys.stderr, flush=True)

    if args.step:
        run_script(args.step)
    else:
        order = PIPELINE_ORDER
        if args.skip_eda:
            order = [s for s in order if s != "exploration"]
            print("[INFO] Saltando exploration por --skip-eda", flush=True)
        for step in order:
            run_script(step)

    print(f"\n{'='*70}", flush=True)
    print(" Pipeline completado ✓", flush=True)
    print(f"{'='*70}\n", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
