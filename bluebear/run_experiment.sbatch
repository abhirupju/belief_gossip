#!/bin/bash
#SBATCH --job-name=bg_exp
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/rds/projects/g/ghoshaz-multi-agent-llms/belief_gossip/logs/%x_%j.out
# Submit with bluebear/submit.sh (it sets BG_CODE and the account, and runs sbatch from the logs folder).
# Arguments after the script name are passed to main.py, e.g. --model X --facts_per_agent 1.
set -euo pipefail
: "${BG_CODE:?submit with: bash bluebear/submit.sh [main.py options]}"
source "$BG_CODE/bluebear/env.sh"
cd "$BG_CODE"

echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "code commit: $(git -C "$BG_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$BG_VENV/bin/python" src/main.py --config config/hiddenbench_test1_bluebear.yaml --device cuda "$@"

echo "finished at $(date -Is)"
