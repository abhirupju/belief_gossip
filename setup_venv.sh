#!/usr/bin/env bash
# Create code/.venv on this machine (Mac or BlueBear) and install everything in requirements.txt.
# Usage (from anywhere):  bash code/setup_venv.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Activate with: source $(pwd)/.venv/bin/activate"
echo "Mock tests:   (cd $(pwd) && .venv/bin/python -m pytest)"
echo "Real model:   (cd $(pwd) && .venv/bin/python -m pytest -m slow)"
