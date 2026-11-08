#!/usr/bin/env bash
# One-time BlueBEAR setup, run on a LOGIN node from the repo root (the folder with src/ and bluebear/):
#   bash bluebear/setup_bluebear.sh
# Creates the storage folders in project space, symlinks them into the repo root, builds the venv there,
# and points the caches of interactive shells at project space (one line in ~/.bashrc). Safe to rerun.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
tf_version="${BG_TF_VERSION:-5.22.2}"   # needs Python 3.11+ and torch 2.6+
tv_version="${BG_TV_VERSION:-0.22.1}"   # torchvision release paired with torch 2.7.1 (PyPI wheel built for CUDA 12.6)

mkdir -p "$BG_STORE"/{data,model,results,logs,hf_cache,pip_cache,cache} "$(dirname "$HF_TOKEN_PATH")"

# <repo root>/<name> -> $BG_STORE/<target>. Refuses to replace a real folder (move its contents first).
for pair in data:data model:model results:results logs:logs .venv:venv-bearmod; do
  name="${pair%%:*}"; target="$BG_STORE/${pair##*:}"; link="$BG_CODE/$name"
  if [ -L "$link" ] && [ "$(readlink "$link")" != "$target" ]; then
    if [ "$name" = ".venv" ]; then
      ln -sfn "$target" "$link"   # earlier layouts pointed .venv at $BG_STORE/venv
    else
      echo "ERROR: $link points to $(readlink "$link"), expected $target"; exit 1
    fi
  elif [ -e "$link" ] && [ ! -L "$link" ]; then
    echo "ERROR: $link is a real folder; move its contents to $target, remove it, then rerun"; exit 1
  elif [ ! -e "$link" ]; then
    ln -s "$target" "$link"
  fi
  echo "$link -> $(readlink "$link")"
done

# Venv on top of the PyTorch module. torch must come from the module, so nothing may install torch from PyPI:
# transformers does not depend on torch; accelerate and torchvision do, so they go in with --no-deps.
[ -x "$BG_VENV/bin/python" ] || python3 -m venv --system-site-packages "$BG_VENV"
pip="$BG_VENV/bin/pip"
"$pip" install --upgrade pip
grep -v -E '^(torch|torchvision|accelerate|transformers)([<>=! ]|$)' "$BG_CODE/requirements.txt" > "$BG_STORE/cache/requirements_bluebear.txt"
"$pip" install -r "$BG_STORE/cache/requirements_bluebear.txt" "transformers==$tf_version"
# pip counts the module's regex 2024.5.15 as installed; transformers checks for >= 2025.10.22 at import time.
"$pip" install --ignore-installed --no-deps "regex>=2025.10.22"
"$pip" install --no-deps accelerate "torchvision==$tv_version"

site="$BG_VENV/lib/python3.12/site-packages"
[ ! -d "$site/torch" ] || { echo "ERROR: pip put a PyPI torch in $site; it would shadow the module's torch 2.7.1"; exit 1; }
# transformers checks its dependency versions on import, so a too-old module package fails here, not in a job.
"$BG_VENV/bin/python" - "$tf_version" <<'PY'
import sys
import accelerate, regex, torch, torchvision, transformers
for m in (torch, transformers, torchvision, accelerate, regex):
    print(f"{m.__name__:13s} {m.__version__:14s} {m.__file__}")
assert torch.__file__.startswith("/rds/bear-apps/"), f"setup: torch loaded from {torch.__file__}, expected the BEAR module"
assert transformers.__version__ == sys.argv[1], f"setup: transformers {transformers.__version__}, expected {sys.argv[1]}"
PY

# Interactive shells (hf download, pip install, python) get the same cache locations as the scripts.
marker="# >>> belief_gossip caches >>>"
if ! grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
  printf '\n%s\n[ -f "%s" ] && source "%s"\n# <<< belief_gossip caches <<<\n' \
    "$marker" "$BG_CODE/bluebear/cache_env.sh" "$BG_CODE/bluebear/cache_env.sh" >> "$HOME/.bashrc"
  echo "Added cache_env.sh to ~/.bashrc (takes effect in new shells)"
fi
# A token saved before HF_TOKEN_PATH was set sits in project space; move it to the private path.
if [ -f "$HF_HOME/token" ] && [ ! -f "$HF_TOKEN_PATH" ]; then
  mv "$HF_HOME/token" "$HF_TOKEN_PATH" && chmod 600 "$HF_TOKEN_PATH" && echo "Moved HF token to $HF_TOKEN_PATH"
fi
# Report caches that already landed in the home folder; move or delete them by hand.
for d in "$HOME/.cache/huggingface/hub" "$HOME/.cache/huggingface/xet" "$HOME/.cache/pip" "$HOME/.cache/torch" "$HOME/.triton"; do
  if [ -d "$d" ]; then echo "WARNING: $(du -sh "$d" | cut -f1) of cache in $d (home quota); move it to $BG_STORE or delete it"; fi
done

echo "Setup done. Next (login node): bash bluebear/download.sh --model <hf_name>; then test a GPU job with"
echo "  BG_SBATCH=check_env.sbatch bash bluebear/submit.sh"
