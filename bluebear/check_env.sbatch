#!/bin/bash
#SBATCH --job-name=bg_check
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/rds/projects/g/ghoshaz-multi-agent-llms/belief_gossip/logs/%x_%j.out
# Environment check on one A100: versions, CUDA, a bf16 matmul, and the unit tests.
#   BG_SBATCH=check_env.sbatch bash bluebear/submit.sh   (account is passed by submit.sh)
set -euo pipefail
: "${BG_CODE:?submit with: BG_SBATCH=check_env.sbatch bash bluebear/submit.sh}"
source "$BG_CODE/bluebear/env.sh"
cd "$BG_CODE"

echo "job $SLURM_JOB_ID on $(hostname) (BB_CPU=${BB_CPU:-unknown}) at $(date -Is), venv $BG_VENV"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$BG_VENV/bin/python" - <<'PY'
import sys
import torch, torchvision, transformers, accelerate, tokenizers, huggingface_hub
print("python", sys.version.split()[0])
for m in (torch, torchvision, transformers, accelerate, tokenizers, huggingface_hub):
    print(f"{m.__name__:16s} {m.__version__:12s} {m.__file__}")
print("torch CUDA build", torch.version.cuda, "| cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), "check_env: torch.cuda.is_available() is False on a GPU node"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok on", torch.cuda.get_device_name(0), float((x @ x).float().abs().mean()))
PY

"$BG_VENV/bin/python" -m pytest -q
echo "finished at $(date -Is)"
