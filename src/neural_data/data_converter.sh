#!/usr/bin/env bash
set -euo pipefail

# Quiet mode by default; set VERBOSE=1 to see full logs
QUIET=${QUIET:-1}
if [[ "${VERBOSE:-}" == "1" ]]; then
  QUIET=0
fi

# Helper: run a command quietly (stdout only) when QUIET=1; keep stderr for errors
runq() {
  if [[ "$QUIET" == "1" ]]; then
    "$@" >/dev/null
  else
    "$@"
  fi
}

# Try to use conda if available, otherwise fall back to a local venv
USE_CONDA=true
if [[ "${FORCE_VENV:-}" == "1" ]]; then
  USE_CONDA=false
fi

# Detect conda in PATH; if missing, try common install locations
if ! command -v conda >/dev/null 2>&1; then
  for csh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$csh" ]; then
      # shellcheck disable=SC1090
      . "$csh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  USE_CONDA=false
fi

if $USE_CONDA; then
  echo "Creating conda environment"

  # ensure conda functions are available in this shell
  eval "$(conda shell.bash hook)"

  ENV_NAME=allen_legacy
  # Create env only if it does not exist
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [[ "$QUIET" == "1" ]]; then QFLAG="-q"; else QFLAG=""; fi
    runq conda create -y $QFLAG -n "$ENV_NAME" python=3.7
    runq conda install -y $QFLAG -n "$ENV_NAME" -c conda-forge pandas=0.25.3 xarray=0.15.1 zarr=2.* numcodecs tqdm
  fi
  conda activate "$ENV_NAME"
else
  echo "Creating venv"
  PYBIN=${PYBIN:-python3}
  VENV_DIR=${VENV_DIR:-.venv_allen_legacy}
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYBIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  # Install broadly compatible versions when using venv
  python -m pip install -q pandas xarray zarr numcodecs tqdm
fi

# use arguments passed or default
inputs=("$@")
if [ ${#inputs[@]} -eq 0 ]; then
  inputs=(AllenData/neuropixels.pkl AllenData/calcium.pkl)
fi

for p in "${inputs[@]}"; do
  if [ ! -f "$p" ]; then
    echo "File not found: $p" >&2
    continue
  fi
  if [[ "$QUIET" == "1" ]]; then SILENT=1 python src/converter.py "$p"; else python src/converter.py "$p"; fi
done

if $USE_CONDA; then
  conda deactivate || true
  # remove only if explicitly requested
  if [[ "${REMOVE_ENV:-}" == "1" ]]; then
    if [[ "$QUIET" == "1" ]]; then QFLAG="-q"; else QFLAG=""; fi
    runq conda remove -y $QFLAG -n "$ENV_NAME" --all || true
  fi
else
  deactivate || true
  if [[ "${REMOVE_ENV:-}" == "1" ]]; then
    rm -rf "$VENV_DIR"
  fi
fi