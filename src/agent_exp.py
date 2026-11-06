"""Experiment loop: load models once, then answer each task with ceil(F / k) agents and save results."""

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent import create_agents, partition_facts
from utility import hiddenbench_utils
from utility.model_utils import ModelHandle, choose_dtype, load_model, memory_report, resolve_device
from utility.prompt_utils import check_template_mode, get_option_order, get_rotations, load_template
from utility.telemetry_utils import Telemetry, add_event, setup_telemetry, shutdown_telemetry, start_span


def get_schema_version() -> int:
    """Version of the record and summary format; bump when fields change meaning."""
    return 1


def get_dataset_adapter(name: str) -> dict[str, Callable]:
    """The functions the experiment needs from a dataset; HiddenBench is the first adapter."""
    adapters = {
        "hiddenbench": {
            "prepare": hiddenbench_utils.download_hiddenbench,
            "is_prepared": hiddenbench_utils.is_prepared,
            "num_samples": hiddenbench_utils.get_num_samples,
            "load_sample": hiddenbench_utils.load_sample,
            "num_private_facts": hiddenbench_utils.get_num_private_facts,
            "num_options": hiddenbench_utils.get_num_options,
            "prompt_fields": hiddenbench_utils.build_prompt_fields,
            "true_letter": hiddenbench_utils.get_true_letter,
            "letter_to_answer": hiddenbench_utils.letter_to_answer,
        },
    }
    assert name in adapters, f"get_dataset_adapter: unknown dataset {name!r}; available: {sorted(adapters)}"
    return adapters[name]


def get_protocol(name: str) -> Callable:
    """The per-task function for a protocol; only single_round exists now."""
    protocols = {"single_round": run_single_round_task}
    assert name in protocols, f"get_protocol: unknown protocol {name!r}; available: {sorted(protocols)}"
    return protocols[name]


def get_needed_model_keys(config: dict[str, Any]) -> list[str]:
    """Model keys any agent can use: the default plus every override."""
    keys = [config["agents"]["default_model"]] + list((config["agents"].get("overrides") or {}).values())
    return list(dict.fromkeys(keys))


def load_all_models(config: dict[str, Any], model_root: Path, loader_fn: Callable | None = None,
                    download: bool = True) -> dict[str, ModelHandle]:
    """Load every model any agent can use, once, keyed by model key."""
    handles = {}
    for key in get_needed_model_keys(config):
        model_cfg = config["models"][key]
        device = resolve_device(model_cfg["device"])
        dtype = model_cfg["dtype"]
        if dtype == "auto":
            dtype = choose_dtype(model_cfg["hf_name"], model_root, device, model_cfg["model_class"], loader_fn, download)
        handles[key] = load_model(model_cfg["hf_name"], model_root, device, dtype, model_cfg["model_class"],
                                  loader_fn, download)
    return handles


def make_record(run_id: str, task_index: int, sample: dict[str, Any], adapter: dict[str, Callable],
                output: Any) -> dict[str, Any]:
    """One prediction record (same fields in every mode; letters in the dataset's own option order)."""
    true_letter = adapter["true_letter"](sample)
    letter = output.predicted_letter
    return {
        "schema_version": get_schema_version(),
        "run_id": run_id,
        "round": output.round,
        "agent_id": output.agent_id,
        "role": output.role,
        "trace_id": output.trace_id,
        "span_id": output.span_id,
        "task_index": task_index,
        "task_id": sample["id"],
        "task_name": sample["name"],
        "agent_index": output.agent_index,
        "model_name": output.model_name,
        "fact_indices": output.fact_indices,
        "private_facts": output.private_facts,
        "rotation": output.rotation,
        "option_order": get_option_order(adapter["num_options"](sample), output.rotation),
        "raw_response": output.raw_response,
        "reasoning": output.reasoning,
        "predicted_letter": letter,
        "predicted_answer": adapter["letter_to_answer"](sample, letter) if letter is not None else None,
        "true_letter": true_letter,
        "true_answer": adapter["letter_to_answer"](sample, true_letter),
        "correct": letter == true_letter,
        "parse_ok": output.parse_ok,
        "option_logprobs_prior": output.option_logprobs_prior,
        "option_logprobs_posterior": output.option_logprobs_posterior,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "latency_s": output.latency_s,
    }


def majority_vote(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority answer of one task (one rotation): a tie or no parseable answer counts as wrong."""
    votes = Counter(r["predicted_letter"] for r in records if r["predicted_letter"] is not None)
    if not votes:
        return {"letter": None, "correct": False, "tie": False, "no_answer": True}
    ranked = votes.most_common()
    tie = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
    letter = None if tie else ranked[0][0]
    return {"letter": letter, "correct": letter is not None and letter == records[0]["true_letter"],
            "tie": tie, "no_answer": False}


def run_single_round_task(task_index: int, sample: dict[str, Any], adapter: dict[str, Callable],
                          handles: dict[str, ModelHandle], config: dict[str, Any], template: dict[str, Any],
                          telemetry: Telemetry, run_id: str) -> list[dict[str, Any]]:
    """Every agent answers once (per rotation) with its own block of k private facts."""
    blocks = partition_facts(adapter["num_private_facts"](sample), config["agents"]["facts_per_agent"])
    agents = create_agents(blocks, handles, config, template, telemetry)
    rotations = get_rotations(adapter["num_options"](sample), config["prompts"]["option_rotation"])
    records = []
    with start_span(telemetry, f"task {task_index}", {"bg.task.index": task_index, "bg.task.id": sample["id"],
                                                      "bg.num_agents": len(agents)}) as task_span:
        for rotation in rotations:
            with start_span(telemetry, "round 0", {"bg.round": 0, "bg.rotation": rotation}):
                group = []
                for agent, block in zip(agents, blocks):
                    agent.set_task(task_index, adapter["prompt_fields"](sample, block, rotation), block, rotation)
                    record = make_record(run_id, task_index, sample, adapter, agent.answer(round_idx=0))
                    add_event(telemetry, task_span, "agent_scored",
                              {"gen_ai.agent.id": record["agent_id"], "bg.rotation": rotation,
                               "bg.correct": record["correct"], "bg.parse_ok": record["parse_ok"]})
                    group.append(record)
                    agent.reset()
                vote = majority_vote(group)
                add_event(telemetry, task_span, "task_scored",
                          {"bg.rotation": rotation, "bg.majority_letter": vote["letter"],
                           "bg.correct": vote["correct"], "bg.tie": vote["tie"], "bg.no_answer": vote["no_answer"]})
                records.extend(group)
    return records


def accuracy_by(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Accuracy grouped by a record field."""
    groups: dict[Any, list[bool]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record["correct"])
    return {str(k): sum(v) / len(v) for k, v in sorted(groups.items())}


def compute_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-agent accuracy: correct agent answers / all agent answers (unparseable answers count as wrong)."""
    assert records, "compute_accuracy: no records"
    return {
        "per_agent_accuracy": sum(r["correct"] for r in records) / len(records),
        "num_answers": len(records),
        "num_correct": sum(r["correct"] for r in records),
        "parse_failures": sum(not r["parse_ok"] for r in records),
        "per_rotation": accuracy_by(records, "rotation"),
        "per_agent_index": accuracy_by(records, "agent_index"),
    }


def compute_majority_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-task majority-vote accuracy over (task, rotation) groups; ties and no-answer count as wrong."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["task_index"], record["rotation"]), []).append(record)
    votes = {key: majority_vote(group) for key, group in sorted(groups.items())}
    per_rotation: dict[int, list[bool]] = {}
    for (_, rotation), vote in votes.items():
        per_rotation.setdefault(rotation, []).append(vote["correct"])
    return {
        "majority_accuracy": sum(v["correct"] for v in votes.values()) / len(votes),
        "num_groups": len(votes),
        "ties": sum(v["tie"] for v in votes.values()),
        "no_answer": sum(v["no_answer"] for v in votes.values()),
        "per_rotation": {str(r): sum(v) / len(v) for r, v in sorted(per_rotation.items())},
    }


def get_run_dir(results_dir: Path, dataset: str, model_name: str, num_samples: int, total_samples: int) -> Path:
    """results/<dataset>_<model>_full_run, or ..._first{N}_run when only the first N tasks run."""
    scope = "full_run" if num_samples == total_samples else f"first{num_samples}_run"
    return Path(results_dir) / f"{dataset}_{model_name.replace('/', '__')}_{scope}"


def get_file_stem(template_name: str, facts_per_agent: int | str, option_rotation: str, num_samples: int) -> str:
    """'answer_only_kall_rotnone_N65'."""
    return f"{template_name}_k{facts_per_agent}_rot{option_rotation}_N{num_samples}"


def save_results(run_dir: Path, stem: str, records: list[dict[str, Any]], summary: dict[str, Any],
                 template: dict[str, Any]) -> dict[str, str]:
    """Write predictions, summary and an exact copy of the template; return the file paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "predictions": run_dir / f"predictions_{stem}.json",
        "summary": run_dir / f"summary_{stem}.json",
        "prompt": run_dir / f"prompt_{stem}.yaml",
    }
    files["predictions"].write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    files["prompt"].write_text(template["raw_text"], encoding="utf-8")
    summary["files"] = {key: str(path) for key, path in files.items()}
    files["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary["files"]


def run_experiment(config: dict[str, Any], paths: dict[str, Path], num_samples: int | None = None,
                   reasoning: bool = False, loader_fn: Callable | None = None, span_exporter: Any = None,
                   download: bool = True) -> dict[str, Any]:
    """Run the configured protocol over the first N tasks and save results; returns the summary."""
    started = time.perf_counter()
    experiment = config["experiment"]
    adapter = get_dataset_adapter(experiment["dataset"])
    if download:
        adapter["prepare"](paths["data_dir"])
    assert adapter["is_prepared"](paths["data_dir"]), (
        f"run_experiment: dataset not found under {paths['data_dir']}; run main.py --download_only first"
    )
    total = adapter["num_samples"](paths["data_dir"])
    n = total if num_samples is None else num_samples
    assert type(n) is int and 1 <= n <= total, (
        f"run_experiment: num_samples is {n!r}; must be between 1 and the dataset size {total}"
    )
    template = load_template(paths["prompt_dir"], config["prompts"]["template"])
    check_template_mode(template, reasoning)
    protocol = get_protocol(experiment["protocol"])
    handles = load_all_models(config, paths["model_dir"], loader_fn, download)
    default_model = config["models"][config["agents"]["default_model"]]["hf_name"]
    run_dir = get_run_dir(paths["results_dir"], experiment["dataset"], default_model, n, total)
    stem = get_file_stem(template["name"], config["agents"]["facts_per_agent"],
                         config["prompts"]["option_rotation"], n)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_info = {"bg.run.id": run_id, "bg.run.name": experiment["name"], "bg.dataset": experiment["dataset"],
                "bg.protocol": experiment["protocol"], "bg.template": template["name"],
                "bg.k": str(config["agents"]["facts_per_agent"]), "bg.n": n,
                "bg.option_rotation": config["prompts"]["option_rotation"], "bg.model": default_model}
    trace_path = run_dir / f"traces_{stem}.otlp.jsonl"
    if config["telemetry"]["enabled"] and span_exporter is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
    telemetry = setup_telemetry(trace_path, run_info, config["telemetry"]["enabled"],
                                config["telemetry"]["capture_content"], config["telemetry"]["otlp_endpoint"],
                                span_exporter)
    records = []
    with start_span(telemetry, f"run {experiment['name']}", run_info):
        for task_index in range(n):
            sample = adapter["load_sample"](task_index, paths["data_dir"])
            records.extend(protocol(task_index, sample, adapter, handles, config, template, telemetry, run_id))
    shutdown_telemetry(telemetry)
    summary = {
        "schema_version": get_schema_version(),
        "run_id": run_id,
        "run_name": experiment["name"],
        "dataset": experiment["dataset"],
        "protocol": experiment["protocol"],
        "num_samples": n,
        "total_samples": total,
        "facts_per_agent": config["agents"]["facts_per_agent"],
        "option_rotation": config["prompts"]["option_rotation"],
        "num_agent_answers": len(records),
        "template_name": template["name"],
        "template_text": template["raw_text"],
        "reasoning": reasoning,
        "models": {key: {"hf_name": h.name, "device": h.device, "dtype": h.dtype, "model_class": h.model_class}
                   for key, h in handles.items()},
        "accuracy": compute_accuracy(records),
        "majority": compute_majority_accuracy(records),
        "memory": memory_report(next(iter(handles.values())).device),
        "run_time_s": time.perf_counter() - started,
        "trace_file": str(trace_path) if telemetry.enabled and span_exporter is None else None,
        "config": config,
    }
    save_results(run_dir, stem, records, summary, template)
    return summary
