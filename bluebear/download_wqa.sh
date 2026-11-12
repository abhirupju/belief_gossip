#!/usr/bin/env bash
# On a LOGIN node: build data/wearableqa/tasks.jsonl from WearableQA_raw.json and download a model.
#   bash bluebear/download_wqa.sh --model Qwen/Qwen3.5-2B
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BG_CODE"
"$BG_VENV/bin/python" src/wqa_exp.py --config config/wearableqa_bluebear.yaml --download_only "$@"
