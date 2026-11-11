# belief_gossip code — rules for working in this repo

Multi-agent LLM experiments on HiddenBench (conformal set debate, MobiCom target). Agents get the shared facts plus
k private facts each; we save generated answers and prior/posterior option log-probabilities, then analyse
evidence-lift, fusion rules and conformal sets. How to run everything: `docs/README.md`.

## Where things run and live

| | Mac (development) | BlueBEAR (large runs) |
|---|---|---|
| Hardware | MacBook M4, 8 GB, MPS | A100 40 GB (`--qos=bbgpu --gres=gpu:a100:1`), up to 2 days/job |
| Code (git repo root = folder with `src/`) | `~/Code/belief_gossip/code` | `/rds/homes/g/ghoshaz/belief_gossip` |
| Data, models, results, logs, venv, caches | `code/data`, `code/model`, `code/results` (local folders) | **`/rds/projects/g/ghoshaz-multi-agent-llms/belief_gossip`**, symlinked into the repo root as `data`, `model`, `results`, `logs`, `.venv` (→ `venv-bearmod`) |
| What to run there | 0.8B model, short runs, all unit tests | 2B–8B models, full Test 1 sweeps |

**Storage rule (BlueBEAR):** everything memory- or disk-intensive — datasets, model weights, results, traces,
sbatch logs, the venv, Hugging Face / pip / torch caches — goes in
`/rds/projects/g/ghoshaz-multi-agent-llms/belief_gossip`. Never write large files under `/rds/homes`.
`bluebear/cache_env.sh` sets `HF_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR` there (sourced by `bluebear/env.sh` and, after setup, by `~/.bashrc`); any new script that
downloads or writes large files must source `env.sh`. The HF token stays in home (`HF_TOKEN_PATH`). New large-output folders get a symlink in
`bluebear/setup_bluebear.sh` and an entry in `.gitignore`.

BlueBEAR software: torch 2.7.1 from the BEAR module `PyTorch/2.7.1-foss-2024a-CUDA-12.6.0`, transformers 5.22.2 from
PyPI in `$BG_STORE/venv-bearmod`; `env.sh` puts the venv first on `PYTHONPATH`. Never `pip install` torch into that venv.

BlueBEAR workflow: `bluebear/setup_bluebear.sh` (once, login node) → `bluebear/download.sh --model ...` (login
node) → `bluebear/submit.sh [main.py options]` (logs in `logs/bg_exp_<jobid>.out` = `$BG_STORE/logs`).
Slurm account: `ghoshaz-multi-agent-llms` (`BG_ACCOUNT` in `env.sh`; `submit.sh` passes it).

## Coding rules

- All logic inside functions or methods; outside them only imports, function/class definitions and the
  `if __name__ == "__main__"` guard. No module-level constants or state.
- No `try`/`except`. Check inputs with `assert` statements whose message names the function, the bad value and
  what was expected. Never run with `python -O`.
- Import `torch`, `transformers`, `huggingface_hub` inside the functions that use them, so unit tests run without them.
- Type hints and a one-line docstring on every function; `pathlib.Path` for paths.
- Every feature gets unit tests with fake data and the fake model backend (`tests/conftest.py`); no real model in
  unit tests. Real-model tests are marked `@pytest.mark.slow`.
- Prompts live only in `prompts/` YAML templates; configs in `config/`. Results files keep the same fields in every
  mode (`schema_version` in records and summaries; bump it when a field changes meaning).
- Run `python -m pytest` from the repo root before handing back changes.

## Pointers

- Run guide and all options: `docs/README.md`
- Analysis: `python src/analyze.py --predictions <predictions file>`
- Multi-round protocol (C4a): `python src/simulate_protocol.py --predictions <k1 predictions file>`
- Design plan (Claude doc): "HiddenBench Agent Codebase — Implementation Plan"
