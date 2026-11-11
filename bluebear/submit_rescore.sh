#!/usr/bin/env bash
# Submit the C4b re-scoring cache job from the repo root on BlueBEAR:
#   bash bluebear/submit_rescore.sh --model Qwen/Qwen3.5-4B --facts_per_agent 1 --option_rotation cyclic
# Everything after the script name is passed to belief_update.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BG_STORE/logs"
sbatch --export=ALL "$BG_CODE/bluebear/rescore.sbatch" "$@"
