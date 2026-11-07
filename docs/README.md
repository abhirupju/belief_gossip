# Running the belief_gossip code

A step-by-step guide to running experiments and analysing results. All commands are run **from the repo root** (`belief_gossip/`) unless stated otherwise. Relative paths in configs and options are resolved against `code/`, so commands work from any folder.

---

## 1. One-time setup

```bash
bash code/setup_venv.sh            # creates code/.venv and installs requirements.txt
source code/.venv/bin/activate
```

On BlueBear, run the same script once in your project space (it creates `code/.venv` there).

## 2. Check that everything works

| Check | Command | Needs |
|---|---|---|
| Mock tests (fake data and model, seconds) | `cd code && python -m pytest` | No model, no download |
| Real-model smoke test (2 tasks, first token is a letter, memory stays flat) | `cd code && python -m pytest -m slow` | Downloaded model and data |

## 3. Download the dataset and model

```bash
python code/src/main.py --download_only
```

- HiddenBench goes to `code/data/hiddenbench/` (converted once to JSONL plus an offset index).
- The model goes to `code/model/<org>__<name>/`.
- On BlueBear, run this on the **login node** (compute nodes usually have no internet) and pass `--data_dir` and `--model_dir` with your `/rds` paths.

## 4. Run an experiment

```bash
python code/src/main.py [options]
```

It prints the per-agent and majority accuracy and the path of the predictions file.

### All options

| Option | Default | What it does |
|---|---|---|
| `--config` | `config/hiddenbench_qwen35_08b.yaml` | Config file (relative to `code/`) |
| `--model` | from config | Hugging Face name; replaces the default model (class detected automatically) |
| `--num_samples N` | all 65 | Run only the first N tasks; folder becomes `..._first{N}_run` |
| `--facts_per_agent k` | from config (`all`) | Hidden facts per agent: an integer, or `all` for one agent with every fact |
| `--option_rotation` | from config (`none`) | `cyclic`: answer once per cyclic shift of the options (removes letter bias) |
| `--prompt_template` | `answer_only` | Template name in the prompt folder (without `.yaml`) |
| `--prompt_dir` | from config | Folder of templates |
| `--reasoning` | off | Ask for reasoning, then the answer; uses `reasoning_answer` unless a template is named |
| `--dtype` | from config (`auto`) | `auto` (faster of bf16/fp16, measured once per device), `bfloat16`, `float16`, `float32` |
| `--device` | from config (`auto`) | `auto`, `mps`, `cuda`, `cpu` |
| `--max_new_tokens` | from config | Generation length for the chosen mode |
| `--data_dir`, `--model_dir`, `--results_dir` | `data`, `model`, `results` | Roots (relative to `code/`, or absolute `/rds` paths) |
| `--no_trace` | off | Do not write the OpenTelemetry trace file |
| `--no_trace_content` | off | Keep the trace, but leave prompts and replies out of it |
| `--otlp_endpoint` | none | Also send spans live to a collector, e.g. `http://localhost:4318` |
| `--download_only` | off | Fetch dataset and models, then exit |

### Recipes

| Experiment | Command (add `--model ...` for another model) |
|---|---|
| Quick check (5 tasks) | `python code/src/main.py --num_samples 5` |
| Full Profile: one agent sees every fact | `python code/src/main.py --facts_per_agent all` |
| Full Profile without letter bias | `python code/src/main.py --facts_per_agent all --option_rotation cyclic` |
| **Test 1**: one fact per agent | `python code/src/main.py --facts_per_agent 1 --option_rotation cyclic` |
| Two facts per agent (last agent may have fewer) | `python code/src/main.py --facts_per_agent 2 --option_rotation cyclic` |
| Reasoning before the answer | `python code/src/main.py --reasoning` (no option scores are saved in this mode) |
| Another prompt wording | Copy a template in `code/prompts/hiddenbench_qwen35_08b/`, change `name:` to match the file name, then `--prompt_template <new_name>` |
| Larger model on BlueBear | `python code/src/main.py --config config/hiddenbench_test1_bluebear.yaml --model <hf_name> --data_dir /rds/... --model_dir /rds/...` |
| Long run, small trace | add `--no_trace_content` |

A run takes about 1 s per agent answer for the 0.8B model on the M4 (65 tasks with `--facts_per_agent all` ≈ 75 s; with cyclic rotation ≈ 4 min). `--facts_per_agent 1` multiplies this by the number of facts per task (about 3–4).

## 5. Where results go

`code/results/<dataset>_<model>_<scope>/`, where `<scope>` is `full_run` (all tasks) or `first{N}_run`. File names carry the settings: `<template>_k<k>_rot<rotation>_N<N>`.

| File | Contents |
|---|---|
| `predictions_<stem>.json` | One record per agent per rotation: task, agent id, fact indices and private facts used, rotation and option order, raw reply, reasoning (empty if not asked), predicted and true option, `correct`, `parse_ok`, prior and posterior option log-probabilities, token counts, latency, trace and span ids |
| `summary_<stem>.json` | Accuracy (per agent, per rotation, per agent index), majority-vote accuracy, models and dtype, template text, config, run time |
| `prompt_<stem>.yaml` | Exact copy of the template used |
| `traces_<stem>.otlp.jsonl` | OpenTelemetry trace of the run |
| `analysis_<stem>.json` | Written by the analysis step below |

A rerun with the same settings overwrites these files.

## 6. Analyse results

```bash
python code/src/analyze.py --predictions code/results/<run_folder>/predictions_<stem>.json
```

It prints tables and saves `analysis_<stem>.json` next to the predictions file. Several files can be given at once.

| Option | Default | What it does |
|---|---|---|
| `--alpha` | `0.1 0.2` | Target miss rates for the conformal sets |
| `--k` | `1 2` | Quorum sizes (1 = intersection; k only matters with several agents per task) |
| `--score` | `lift lac` | Nonconformity scores: `lift` (−evidence-lift) or `lac` (1 − posterior probability) |
| `--conformal_rule` | `joint markov` | Threshold rule (see below) |
| `--splits` | 200 | Random 50/50 calibration/test splits over tasks |
| `--bootstrap` | 5000 | Resamples for the 95% intervals |
| `--seed` | 0 | Seed for splits and bootstrap |
| `--no_save` | off | Print only |

### What the report contains

**Baselines.** Chance (uniform guessing), and "always the most common true letter". The second reads the answer key, so treat it as an oracle; in HiddenBench the answer is C in 39 of 65 tasks.

**Decision rules**, evaluated per task (scores averaged over rotations, then combined over agents) and per answer (each record alone):

| Rule | Picks the option with… |
|---|---|
| `generated_majority` | the most generated votes (a tie or no parseable answer counts as wrong) |
| `prior` | the highest prior: shared facts only |
| `anti_prior` | the lowest prior: tests "just move away from the shared facts" |
| `posterior_sum` | the highest summed posterior (counts the shared prior once per agent) |
| `lift_sum` | the highest summed evidence-lift (ignores the prior) |
| `prior_plus_lift` | the highest prior + Σ lift: Bayes-optimal if facts are independent given the answer |

With one agent per task, `prior_plus_lift` and `posterior_sum` both equal the agent's posterior.

**Paired comparisons.** For two rules, the number of tasks where only A is right and where only B is right, with an exact sign-test p-value.

**Conformal sets.** Each agent excludes options whose score exceeds a threshold; the fused set keeps an option unless at least k agents exclude it.
- `joint`: threshold from the k-th largest true-answer score across a task's agents. Valid coverage for the fused set, with no Bonferroni penalty.
- `markov`: per-agent threshold at level k·α/N. Approximate, because agents of one task are pooled.
- Reported per setting: coverage, mean set size, singleton rate and singleton accuracy, as the mean and the 10th/90th percentiles over splits.

With 65 tasks, coverage varies by roughly ±5–10% between splits. Read the analysis as a feasibility signal, not a paper result.

## 7. View traces (observe only)

Traces are OTLP JSON lines, one trace per run: run → task → round → `invoke_agent agent-i` → `chat <model>`, with timestamped events (`prompt_rendered`, `answer_parsed`, `agent_scored`, `task_scored`). One way to view them, **not yet tested on this project**: an OpenTelemetry Collector (contrib build) with the `otlpjsonfile` receiver pointed at the trace file, exporting to Jaeger.

```yaml
# collector.yaml (sketch)
receivers:
  otlpjsonfile:
    include: ["/path/to/code/results/*/traces_*.otlp.jsonl"]
    start_at: beginning
exporters:
  otlp:
    endpoint: localhost:4317
    tls: {insecure: true}
service:
  pipelines:
    traces: {receivers: [otlpjsonfile], exporters: [otlp]}
```

In Jaeger, filter on `gen_ai.agent.id` to follow one agent, or on `bg.task.index` to follow one task.

## 8. Config file reference (`code/config/*.yaml`)

| Section | Keys |
|---|---|
| `experiment` | `name` (run name in summary and trace), `dataset` (`hiddenbench`), `protocol` (`single_round`) |
| `models.<key>` | `hf_name`, `model_class` (`auto` / `multimodal` / `causal`), `dtype`, `device` |
| `agents` | `facts_per_agent`, `default_model` (a model key), `overrides` (`{agent_index: model_key}`) |
| `prompts` | `dir`, `template`, `option_rotation` |
| `generation` | `max_new_tokens`, `max_new_tokens_reasoning`, `do_sample`, `seed` |
| `telemetry` | `enabled`, `capture_content`, `otlp_endpoint` |

Command-line options override the config; the final config is saved in the summary.

## 9. When something fails

- Errors are assertions whose message names the function, the bad value and what was expected; read the last line of the traceback.
- Never run with `python -O`: it strips the assertions.
- "dataset not found": run `--download_only` first, or check `--data_dir`.
- "refusing to load a second copy": the same model was requested with two different devices or dtypes in one process.
- "--reasoning is on but template ... has mode answer_only": pass a reasoning template, or drop `--reasoning`.
- The analysis says "reasoning-mode runs have none": reasoning runs save no option scores; analyse an answer-only run.
