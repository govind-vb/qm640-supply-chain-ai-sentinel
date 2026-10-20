#!/usr/bin/env bash
# Bootstrap a local Python environment for QM640 Supply Chain AI Sentinel notebooks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Project root: $ROOT"
echo "==> Creating venv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing requirements"
python -m pip install -r requirements.txt

echo "==> Registering Jupyter kernel: qm640-sentinel"
python -m ipykernel install --user --name qm640-sentinel --display-name "Python (qm640-sentinel)"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env from .env.example (add your API keys)"
else
  echo "==> .env already exists (left unchanged)"
fi

mkdir -p data/raw data/processed \
  notebooks/outputs/latest/figures \
  notebooks/outputs/latest/tables \
  notebooks/outputs/iter_01

echo "==> Setup complete"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run: jupyter lab notebooks/"
echo "Select kernel: Python (qm640-sentinel)"
