#!/usr/bin/env bash
# Submit a job to the A100 queue from a login node. Logs go to $BG_STORE/logs/<job name>_<jobid>.out.
#   bash bluebear/submit.sh --model Qwen/Qwen3.5-0.8B --facts_per_agent 1 --option_rotation cyclic
#   BG_SBATCH=check_env.sbatch bash bluebear/submit.sh        (environment check: versions, CUDA, unit tests)
# The Slurm account comes from BG_ACCOUNT in env.sh. Download models first (bluebear/download.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$BG_STORE/logs"
cd "$BG_STORE/logs"   # BlueBEAR writes slurm-<jobid>.stats to the submit folder, so submit from the logs folder
sbatch --account="$BG_ACCOUNT" --export=ALL "$BG_CODE/bluebear/${BG_SBATCH:-run_experiment.sbatch}" "$@"
