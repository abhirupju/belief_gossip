#!/usr/bin/env bash
# BlueBEAR environment: storage, caches, modules and venv. Sourced by every BlueBEAR script.
# Rule: everything large (data, models, results, logs, venv, caches) lives in BG_STORE (project space),
# never in /rds/homes. The code repo links to it through symlinks (see setup_bluebear.sh).
#
# Software (tested 8 Nov 2026 on login node bear-pg-login06 and an A100 40 GB node):
#   torch 2.7.1 (CUDA 12.6)  from the BEAR module PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 (bear-apps/2024a, EL8-icelake;
#                            loads on the login nodes too, but CUDA works only in GPU jobs)
#   transformers 5.22.2 etc. from PyPI in the venv $BG_STORE/venv-bearmod (built with --system-site-packages)
# BEAR's own Transformers module stops at 4.42.0 (2023a, PyTorch 2.1.2), too old for Qwen3.5.

export BG_CODE="${BG_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (the folder with src/)
source "$BG_CODE/bluebear/cache_env.sh"                                          # BG_STORE and all cache locations
export BG_ACCOUNT="${BG_ACCOUNT:-ghoshaz-multi-agent-llms}"   # Slurm account = BEAR project (plain 'ghoshaz' is rejected)
export BG_VENV="$BG_STORE/venv-bearmod"
export PYTHONDONTWRITEBYTECODE=1

module purge
module load bluebear
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0   # also loads Python 3.12.3, SciPy-bundle, Python-bundle-PyPI

# Modules put their packages on PYTHONPATH, which Python searches before the venv. Put the venv first so its
# newer packages win (e.g. Python-bundle-PyPI has regex 2024.5.15; transformers 5.22 needs >= 2025.10.22).
export PYTHONPATH="$BG_VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
