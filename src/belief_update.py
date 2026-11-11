"""C4b (variant D): build the re-scoring cache for the belief-update protocol. Online, prompt-only; runs on BlueBEAR.

For every task, agent and reachable ruled-out set R (every R that leaves at least two options, including the empty
ruling for round 0), the model does a prior pass (shared facts + group conclusion, no private fact) and a posterior
pass (+ the private fact) under a `{group_conclusion}` naming R, averaged over cyclic rotations. The protocol driver
in `analysis/protocol.py` then reads this cache offline (variant D) with the same calibration and evaluation as C4a.

Cache file `rescored_<stem>.json`: {"by_task": {task_index: {agent_index: {ruling_key: {prior: {...}, posterior:
{...}}}}}} where ruling_key is the comma-joined sorted ruled-out letters in the dataset's own option order ("" = none).
All scores are log-probabilities in the dataset's option order, averaged over rotations. The model never sees the truth.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from agent import get_model_key_for_agent, load_config, partition_facts, validate_config
from agent_exp import get_dataset_adapter, get_run_dir, load_all_models
from analysis.scores import normalize
from utility.model_utils import ModelHandle, build_chat_messages, score_options
from utility.prompt_utils import get_option_order, get_rotations, load_template, render_prompt, unrotate_scores


def ruling_key(ruled_out: tuple[str, ...]) -> str:
    """Canonical cache key for a ruled-out set: its letters, sorted, comma-joined ('' for the empty ruling)."""
    return ",".join(sorted(ruled_out))


def reachable_rulings(letters: list[str]) -> list[tuple[str, ...]]:
    """Every ruled-out set that leaves at least two options (including the empty ruling), as sorted letter tuples."""
    assert len(letters) >= 2, f"reachable_rulings: need at least two options, got {letters}"
    rulings = []
    for size in range(0, len(letters) - 1):  # leave >= 2 options, so rule out at most M - 2
        rulings.extend(tuple(sorted(combo)) for combo in combinations(sorted(letters), size))
    return rulings


def displayed_of(letters: list[str], rotation: int) -> dict[str, str]:
    """Map each dataset letter to the label it is shown under at this cyclic rotation."""
    order = get_option_order(len(letters), rotation)
    return {letters[original]: letters[position] for position, original in enumerate(order)}


def score_pass(handle: ModelHandle, template: dict[str, Any], fields: dict[str, Any],
               token_ids: dict[str, list[int]], num_options: int, rotation: int) -> dict[str, float]:
    """Prompt-only option log-probabilities for one rendered prompt, re-keyed to the dataset's option order."""
    system_text, user_text = render_prompt(template, fields)
    messages = build_chat_messages(system_text, user_text, handle.model_class)
    raw = score_options(handle, messages, token_ids)
    return unrotate_scores(raw, num_options, rotation)


def average_passes(per_rotation: list[dict[str, float]]) -> dict[str, float]:
    """Mean of per-rotation score dicts (all in the dataset's option order)."""
    assert per_rotation, "average_passes: nothing to average"
    letters = sorted(per_rotation[0])
    return {letter: sum(d[letter] for d in per_rotation) / len(per_rotation) for letter in letters}


def rescore_agent_ruling(handle: ModelHandle, template: dict[str, Any], adapter: dict[str, Callable],
                         sample: dict[str, Any], block: list[int], ruled_out: tuple[str, ...],
                         rotations: list[int], which: str) -> dict[str, float]:
    """One pass kind under one ruling ('prior' drops the private facts), rotation-averaged and renormalised.

    Renormalising after the average matches variant A's convention (scores.agent_view), so raw-lift scores
    stay on one scale; margin and every restricted-to-S score are invariant to it either way.
    """
    assert which in ("prior", "posterior"), f"rescore_agent_ruling: which is {which!r}; use 'prior' or 'posterior'"
    passes = []
    for rotation in rotations:
        fields = adapter["prompt_fields"](sample, block, rotation)
        letters = fields["letters"]
        disp = displayed_of(letters, rotation)
        shown_ruled = [disp[d] for d in ruled_out]
        surviving = [disp[d] for d in letters if d not in ruled_out]
        token_ids = handle.backend.option_token_ids(letters)
        pass_fields = {**fields, "ruled_out": shown_ruled, "surviving": surviving}
        if which == "prior":
            pass_fields = {**pass_fields, "private_facts": []}
        passes.append(score_pass(handle, template, pass_fields, token_ids, len(letters), rotation))
    return normalize(average_passes(passes))


def template_for_ruling(ruled_out: tuple[str, ...], round0_template: dict[str, Any],
                        ruling_template: dict[str, Any]) -> dict[str, Any]:
    """The empty ruling (round 0) uses the plain single-shot template; any actual ruling uses the belief-update one.

    This keeps round 0 byte-identical to the one-shot run, so its scores match the saved predictions and the
    group-conclusion line is added only when it carries real information.
    """
    return round0_template if not ruled_out else ruling_template


def build_rescore_cache(config: dict[str, Any], adapter: dict[str, Callable], handles: dict[str, ModelHandle],
                        round0_template: dict[str, Any], ruling_template: dict[str, Any],
                        data_dir: Path, num_samples: int) -> dict[str, Any]:
    """Loop tasks x agents x reachable rulings; the prior (no private fact) is computed once per task and ruling."""
    rotation_mode = config["prompts"]["option_rotation"]
    facts_per_agent = config["agents"]["facts_per_agent"]
    assert not config["agents"].get("overrides"), (
        "build_rescore_cache: the prior pass is shared across a task's agents, which assumes one model; "
        "agents.overrides must be empty")
    by_task: dict[str, Any] = {}
    for task_index in range(num_samples):
        sample = adapter["load_sample"](task_index, data_dir)
        letters = adapter["prompt_fields"](sample, [], 0)["letters"]
        rotations = get_rotations(adapter["num_options"](sample), rotation_mode)
        blocks = partition_facts(adapter["num_private_facts"](sample), facts_per_agent)
        rulings = reachable_rulings(letters)
        handle = handles[get_model_key_for_agent(config, 0)]
        shared_prior = {ruling_key(r): rescore_agent_ruling(handle, template_for_ruling(r, round0_template,
                        ruling_template), adapter, sample, [], r, rotations, "prior") for r in rulings}
        task_cache: dict[str, Any] = {}
        for agent_index, block in enumerate(blocks):
            agent_cache = {}
            for r in rulings:
                tmpl = template_for_ruling(r, round0_template, ruling_template)
                posterior = rescore_agent_ruling(handle, tmpl, adapter, sample, block, r, rotations, "posterior")
                agent_cache[ruling_key(r)] = {"prior": shared_prior[ruling_key(r)], "posterior": posterior}
            task_cache[str(agent_index)] = agent_cache
        by_task[str(task_index)] = task_cache
    return by_task


def cache_stem(config: dict[str, Any], num_samples: int) -> str:
    """File stem matching the predictions it extends: <template>_k<k>_rot<rotation>_N<n>."""
    k = config["agents"]["facts_per_agent"]
    return f"{config['prompts']['template']}_k{k}_rot{config['prompts']['option_rotation']}_N{num_samples}"


def save_rescore_cache(by_task: dict[str, Any], config: dict[str, Any], round0_template: dict[str, Any],
                       ruling_template: dict[str, Any], run_dir: Path, stem: str) -> Path:
    """Write rescored_<stem>.json into the run folder; returns the path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"rescored_{stem}.json"
    payload = {"variant": "D", "round0_template": round0_template["name"], "ruling_template": ruling_template["name"],
               "round0_template_text": round0_template["raw_text"], "ruling_template_text": ruling_template["raw_text"],
               "option_rotation": config["prompts"]["option_rotation"], "by_task": by_task}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments for building the C4b re-scoring cache."""
    from main import parse_facts_per_agent
    parser = argparse.ArgumentParser(description="Build the C4b belief-update re-scoring cache (variant D).")
    parser.add_argument("--config", default="config/hiddenbench_qwen35_08b.yaml", help="config file (relative to code/)")
    parser.add_argument("--model", default=None, help="Hugging Face model name; replaces the default model")
    parser.add_argument("--num_samples", type=int, default=None, help="build the cache for only the first N tasks")
    parser.add_argument("--facts_per_agent", type=parse_facts_per_agent, default=None, help="k, or 'all'")
    parser.add_argument("--option_rotation", choices=["none", "cyclic"], default=None)
    parser.add_argument("--prompt_dir", default=None, help="folder of prompt templates (relative to code/)")
    parser.add_argument("--prompt_template", default="belief_update",
                        help="template used once options have been ruled out (rounds >= 1)")
    parser.add_argument("--round0_template", default="answer_only",
                        help="template for the empty ruling (round 0); 'answer_only' keeps round 0 identical to the "
                             "one-shot run, or pass the ruling template to score round 0 with an empty conclusion")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    parser.add_argument("--download_only", action="store_true", help="fetch dataset and models, then exit")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, code_root: Path) -> dict[str, Any]:
    """Load the config, apply the overrides this CLI supports, and force the belief-update template."""
    from main import resolve_path
    config = load_config(resolve_path(args.config, code_root))
    default_model = config["models"][config["agents"]["default_model"]]
    if args.model is not None:
        default_model["hf_name"], default_model["model_class"] = args.model, "auto"
    if args.dtype is not None:
        default_model["dtype"] = args.dtype
    if args.device is not None:
        for model in config["models"].values():
            model["device"] = args.device
    if args.facts_per_agent is not None:
        config["agents"]["facts_per_agent"] = args.facts_per_agent
    if args.option_rotation is not None:
        config["prompts"]["option_rotation"] = args.option_rotation
    if args.prompt_dir is not None:
        config["prompts"]["dir"] = args.prompt_dir
    config["prompts"]["template"] = args.prompt_template
    validate_config(config)
    return config


def main(argv: list[str] | None = None, loader_fn: Callable | None = None) -> Path | None:
    """Build and save the re-scoring cache for the first N tasks; returns the cache path (None for --download_only)."""
    from main import get_code_root, resolve_path
    args = parse_args(argv)
    code_root = get_code_root()
    config = build_config(args, code_root)
    paths = {name: resolve_path(getattr(args, name), code_root) for name in ("data_dir", "model_dir", "results_dir")}
    prompt_dir = resolve_path(args.prompt_dir or config["prompts"]["dir"], code_root)
    adapter = get_dataset_adapter(config["experiment"]["dataset"])
    if not adapter["is_prepared"](paths["data_dir"]):
        adapter["prepare"](paths["data_dir"])
    assert adapter["is_prepared"](paths["data_dir"]), (
        f"belief_update: dataset not found under {paths['data_dir']}; run main.py --download_only first")
    total = adapter["num_samples"](paths["data_dir"])
    n = total if args.num_samples is None else args.num_samples
    assert type(n) is int and 1 <= n <= total, f"belief_update: num_samples {n!r} must be between 1 and {total}"
    ruling_template = load_template(prompt_dir, config["prompts"]["template"])
    round0_template = load_template(prompt_dir, args.round0_template)
    handles = load_all_models(config, paths["model_dir"], loader_fn, True)
    if args.download_only:
        return None
    by_task = build_rescore_cache(config, adapter, handles, round0_template, ruling_template, paths["data_dir"], n)
    default_model = config["models"][config["agents"]["default_model"]]["hf_name"]
    run_dir = get_run_dir(paths["results_dir"], config["experiment"]["dataset"], default_model, n, total)
    path = save_rescore_cache(by_task, config, round0_template, ruling_template, run_dir, cache_stem(config, n))
    print(f"Saved {path}  ({n} tasks)")
    return path


if __name__ == "__main__":
    main()
