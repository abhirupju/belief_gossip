#!/usr/bin/env bash
# Cache locations for everything that downloads or compiles: Hugging Face, pip, torch, Triton, generic XDG caches.
# Without these, tools write to ~/.cache under /rds/homes (small home quota). With them, caches go to the
# project's RDS space. Sourced by env.sh, and by ~/.bashrc (setup_bluebear.sh adds the line) so that interactive
# shells, `hf download` and manual `pip install` use the same places.

export BG_STORE=/rds/projects/g/ghoshaz-multi-agent-llms/belief_gossip
export HF_HOME="$BG_STORE/hf_cache"                         # Hugging Face hub + xet caches
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"       # the token stays private in home (tiny file), not in project space
export PIP_CACHE_DIR="$BG_STORE/pip_cache"
export XDG_CACHE_HOME="$BG_STORE/cache"                     # generic fallback for tools that use ~/.cache
export TORCH_HOME="$BG_STORE/cache/torch"
export TRITON_CACHE_DIR="$BG_STORE/cache/triton"            # default ~/.triton
export TORCHINDUCTOR_CACHE_DIR="$BG_STORE/cache/torchinductor"
