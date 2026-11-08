#!/usr/bin/env bash
# Download HiddenBench and a model into project space. Run on a LOGIN node:
#   bash bluebear/download.sh --model Qwen/Qwen3.5-0.8B
# Any main.py option can be added; the BlueBEAR Test 1 config is used unless --config is given.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BG_CODE"
"$BG_VENV/bin/python" src/main.py --config config/hiddenbench_test1_bluebear.yaml --download_only "$@"
