#!/usr/bin/env bash
# Submit a WearableQA phenotype run from a login node (repo root on BlueBEAR):
#   bash bluebear/submit_wqa.sh --model Qwen/Qwen3.5-2B --variant strict_key
# Everything after the script name is passed to wqa_exp.py. First, on the login node:
#   copy WearableQA_raw.json to $BG_STORE/data/wearableqa/, then
#   bash bluebear/download_wqa.sh --model Qwen/Qwen3.5-2B   (builds tasks.jsonl and downloads the model)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$BG_STORE/logs"
cd "$BG_STORE/logs"   # BlueBEAR writes slurm-<jobid>.stats to the submit folder
sbatch --account="$BG_ACCOUNT" --export=ALL "$BG_CODE/bluebear/run_wqa.sbatch" "$@"
