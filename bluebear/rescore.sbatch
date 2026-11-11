#!/bin/bash
#SBATCH --job-name=bg_rescore
#SBATCH --account=ghoshaz-multi-agent-llms
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/rds/projects/g/ghoshaz-multi-agent-llms/belief_gossip/logs/%x_%j.out
# Build the C4b belief-update re-scoring cache (prompt-only; variant D). Submit with:
#   bash bluebear/submit_rescore.sh --model Qwen/Qwen3.5-4B --facts_per_agent 1 --option_rotation cyclic
# Arguments after the script name are passed to belief_update.py.
set -euo pipefail
: "${BG_CODE:?submit with: bash bluebear/submit_rescore.sh [belief_update.py options]}"
source "$BG_CODE/bluebear/env.sh"
cd "$BG_CODE"

echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "code commit: $(git -C "$BG_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$BG_VENV/bin/python" src/belief_update.py --config config/hiddenbench_test1_bluebear.yaml --device cuda "$@"

echo "finished at $(date -Is)"
