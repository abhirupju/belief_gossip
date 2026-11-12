"""WearableQA phenotype runner: score every device agent (and the centralised agent) on every task, save records.

Each agent answers once per task in answer-only mode. The posterior is the first-token log-probability over the
option letters with the agent's data; the prior is the same prompt with the data replaced by `empty_data`, scored
once per agent (the prompt is identical across tasks). Records use the same fields as `agent_exp.make_record`, so
the HiddenBench analysis code (`analysis.scores`, `analysis.conformal`) reads them unchanged.

Prompt variants: strict_key (main), no_key (overlay folder without the clinical keys), daily_rows (one line per day
instead of summaries), original (the original WearableQA wording, one agent, ten options; scores are kept for the
four gold-capable options under the fixed letters A-D), axis_questions (each device agent answers only its own
two-option question, in both letter orders; the two answers are spread evenly over the four classes; the centralised
agent answers both questions, combined, and the four-option question).

Example (BlueBEAR, through bluebear/submit_wqa.sh):
  python src/wqa_exp.py --config config/wearableqa_bluebear.yaml --model Qwen/Qwen3.5-2B --variant strict_key
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from agent_exp import get_schema_version, load_all_models
from utility import wqa_split as ws
from utility.model_utils import (ModelHandle, assistant_prefill_message, build_chat_messages, download_model,
                                 logsumexp, memory_report, parse_option_letter, query_model, score_options)
from utility.telemetry_utils import (Telemetry, add_event, get_span_ids, set_attributes, setup_telemetry,
                                     shutdown_telemetry, start_span)
from utility.wqa_original import original_core_letter_map, original_letters, render_original
from utility.wqa_prompts import (asked_axis, axis_letter_sides, axis_names, class_side, load_wqa_templates,
                                 option_letters, render_agent, render_axis_agent, render_axis_centralised,
                                 render_centralised, side_classes, template_version, validate_axis_shared)
from wqa_prepare import ensure_tasks


# ----------------------------------------------------------------------------------------------- config


def get_variants() -> dict[str, dict[str, Any]]:
    """Prompt variants: which template folder, summaries or daily rows, device agents or the original prompt."""
    return {
        "strict_key": {"overlay": None, "daily": False, "original": False, "axis": False},
        "no_key": {"overlay": "nokey_dir", "daily": False, "original": False, "axis": False},
        "daily_rows": {"overlay": None, "daily": True, "original": False, "axis": False},
        "original": {"overlay": None, "daily": False, "original": True, "axis": False},
        "axis_questions": {"overlay": "axis_dir", "daily": False, "original": False, "axis": True},
    }


def validate_wqa_config(config: dict[str, Any]) -> None:
    """Assert the sections and keys the runner reads."""
    for section in ("experiment", "models", "agents", "data", "prompts", "generation", "telemetry"):
        assert isinstance(config.get(section), dict), f"validate_wqa_config: missing or invalid section '{section}'"
    assert config["experiment"].get("name"), "validate_wqa_config: experiment.name is missing"
    assert config["experiment"].get("dataset") == "wearableqa_phenotype", (
        f"validate_wqa_config: experiment.dataset is {config['experiment'].get('dataset')!r}; "
        f"expected wearableqa_phenotype"
    )
    assert config["models"], "validate_wqa_config: models section is empty"
    for key, model in config["models"].items():
        assert isinstance(model, dict) and isinstance(model.get("hf_name"), str), (
            f"validate_wqa_config: models.{key} needs an hf_name"
        )
        assert model.get("model_class") in ("auto", "multimodal", "causal"), (
            f"validate_wqa_config: models.{key}.model_class is {model.get('model_class')!r}"
        )
        assert model.get("dtype") in ("auto", "bfloat16", "float16", "float32"), (
            f"validate_wqa_config: models.{key}.dtype is {model.get('dtype')!r}"
        )
        assert model.get("device") in ("auto", "mps", "cuda", "cpu"), (
            f"validate_wqa_config: models.{key}.device is {model.get('device')!r}"
        )
    agents = config["agents"]
    assert agents.get("default_model") in config["models"], (
        f"validate_wqa_config: agents.default_model {agents.get('default_model')!r} is not one of "
        f"{list(config['models'])}"
    )
    assert not agents.get("overrides"), "validate_wqa_config: per-agent model overrides are not supported here"
    devices = agents.get("devices")
    assert isinstance(devices, list) and devices and len(set(devices)) == len(devices), (
        f"validate_wqa_config: agents.devices must be a list of distinct agent names, got {devices!r}"
    )
    assert type(agents.get("centralised")) is bool, "validate_wqa_config: agents.centralised must be true or false"
    listed = agents.get("centralised_devices")
    assert listed is None or (isinstance(listed, list) and listed and set(listed) <= set(devices)), (
        f"validate_wqa_config: agents.centralised_devices must be null or a non-empty subset of the devices, "
        f"got {listed!r}"
    )
    data = config["data"]
    for key in ("raw_file", "tasks_file"):
        assert isinstance(data.get(key), str) and data[key], f"validate_wqa_config: data.{key} is missing"
    assert type(data.get("window_days")) is int and data["window_days"] >= 1, (
        "validate_wqa_config: data.window_days must be a positive integer"
    )
    prompts = config["prompts"]
    assert isinstance(prompts.get("dir"), str) and prompts["dir"], "validate_wqa_config: prompts.dir is missing"
    assert prompts.get("variant") in get_variants(), (
        f"validate_wqa_config: prompts.variant is {prompts.get('variant')!r}; use one of {sorted(get_variants())}"
    )
    overlay_key = get_variants()[prompts["variant"]]["overlay"]
    if overlay_key is not None:
        assert isinstance(prompts.get(overlay_key), str) and prompts[overlay_key], (
            f"validate_wqa_config: variant {prompts['variant']} needs prompts.{overlay_key}"
        )
    assert prompts.get("axis_orders", 2) in (1, 2), (
        f"validate_wqa_config: prompts.axis_orders is {prompts.get('axis_orders')!r}; use 1 or 2"
    )
    generation = config["generation"]
    assert type(generation.get("max_new_tokens")) is int and generation["max_new_tokens"] >= 1, (
        "validate_wqa_config: generation.max_new_tokens must be a positive integer"
    )
    assert type(generation.get("do_sample")) is bool, "validate_wqa_config: generation.do_sample must be true or false"
    assert type(generation.get("seed")) is int, "validate_wqa_config: generation.seed must be an integer"
    for key in ("enabled", "capture_content"):
        assert type(config["telemetry"].get(key)) is bool, f"validate_wqa_config: telemetry.{key} must be true or false"


def load_wqa_config(path: Path) -> dict[str, Any]:
    """Read the YAML config and validate it."""
    path = Path(path)
    assert path.is_file(), f"load_wqa_config: config file {path} does not exist"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict), f"load_wqa_config: {path} does not contain a YAML mapping"
    validate_wqa_config(config)
    return config


# ----------------------------------------------------------------------------------------------- records


def chat_messages(system_text: str, user_text: str, model_class: str,
                  prefix: str | None = None) -> list[dict[str, Any]]:
    """Chat messages; an empty system text (original prompt) sends the user message alone; a prefix is a prefilled
    start of the reply (e.g. 'Answer:'), so the next token is the answer letter."""
    messages = build_chat_messages(system_text, user_text, model_class)
    messages = messages if system_text.strip() else messages[1:]
    return messages + [assistant_prefill_message(prefix, model_class)] if prefix else messages


def class_of_letter(letter: str | None) -> str | None:
    """Phenotype code of a fixed letter (A = CH, ...), None for no letter."""
    return {v: k for k, v in ws.fixed_letter_map().items()}.get(letter) if letter else None


def make_wqa_record(run_id: str, task: dict[str, Any], agent_id: str, agent_index: int, role: str,
                    model_name: str, result: Any, letter: str | None, prior: dict[str, float],
                    posterior: dict[str, float], latency: float, span: Any, extra: dict[str, Any]) -> dict[str, Any]:
    """One prediction record: the fields of agent_exp.make_record plus the WearableQA ones."""
    trace_id, span_id = get_span_ids(span)
    return {
        "schema_version": get_schema_version(),
        "run_id": run_id,
        "round": 0,
        "agent_id": agent_id,
        "role": role,
        "trace_id": trace_id,
        "span_id": span_id,
        "task_index": task["task_index"],
        "task_id": task["task_id"],
        "task_name": task["task_id"],
        "agent_index": agent_index,
        "model_name": model_name,
        "fact_indices": [],
        "private_facts": [],
        "rotation": 0,
        "option_order": list(range(len(prior))),
        "raw_response": result.text,
        "reasoning": "",
        "predicted_letter": letter,
        "predicted_answer": class_of_letter(letter),
        "true_letter": task["gold_letter"],
        "true_answer": task["gold_class"],
        "correct": letter == task["gold_letter"],
        "parse_ok": letter is not None,
        "option_logprobs_prior": prior,
        "option_logprobs_posterior": posterior,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": latency,
        "dataset": "wearableqa_phenotype",
        "user_id": task["user_id"],
        "gold_class": task["gold_class"],
        **extra,
    }


def generate_with_scores(handle: ModelHandle, messages: list[dict[str, Any]], token_ids: dict[str, list[int]],
                         generation: dict[str, Any], telemetry: Telemetry, agent_id: str) -> tuple[Any, float]:
    """Greedy generation with first-token option log-probabilities; returns (result, latency)."""
    with start_span(telemetry, f"chat {handle.name}", {"gen_ai.operation.name": "chat",
                                                       "gen_ai.request.model": handle.name,
                                                       "gen_ai.agent.id": agent_id}) as chat_span:
        started = time.perf_counter()
        result = query_model(handle, messages, generation["max_new_tokens"], generation["do_sample"], token_ids,
                             generation["seed"])
        latency = time.perf_counter() - started
        set_attributes(chat_span, {"gen_ai.usage.input_tokens": result.input_tokens,
                                   "gen_ai.usage.output_tokens": result.output_tokens, "bg.latency_s": latency})
        add_event(telemetry, chat_span, "reply", {"text": result.text}, content_keys=("text",))
    return result, latency


def score_prior(handle: ModelHandle, messages: list[dict[str, Any]], token_ids: dict[str, list[int]],
                telemetry: Telemetry, agent_id: str) -> dict[str, float]:
    """Prompt-only option log-probabilities of a no-data prompt."""
    with start_span(telemetry, f"score_options {handle.name}", {"gen_ai.request.model": handle.name,
                                                                "gen_ai.agent.id": agent_id, "bg.pass": "prior"}):
        return score_options(handle, messages, token_ids)


def device_agent_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The agents of a device-prompt run: every device, then the centralised agent if asked for."""
    devices = config["agents"]["devices"]
    specs = [{"agent_id": name, "agent_index": i, "role": "device", "devices": [name]}
             for i, name in enumerate(devices)]
    if config["agents"]["centralised"]:
        listed = config["agents"].get("centralised_devices") or devices
        specs.append({"agent_id": "centralised", "agent_index": len(devices), "role": "centralised",
                      "devices": list(listed)})
    return specs


def spec_axis(templates: dict[str, Any], spec: dict[str, Any]) -> str:
    """The judgement an agent's data covers: its template axis, or 'both' for the centralised agent."""
    return "both" if spec["role"] == "centralised" else templates["agents"][spec["devices"][0]]["axis"]


def render_spec(templates: dict[str, Any], spec: dict[str, Any], split: dict[str, Any], with_data: bool,
                daily: bool, window_days: int) -> tuple[str, str]:
    """(system, user) of one device agent or the centralised agent."""
    if spec["role"] == "centralised":
        return render_centralised(templates, split, spec["devices"], with_data, daily, window_days)
    return render_agent(templates, spec["devices"][0], split, with_data, daily, window_days)


def run_device_agents(tasks: list[dict[str, Any]], templates: dict[str, Any], handle: ModelHandle,
                      config: dict[str, Any], telemetry: Telemetry, run_id: str,
                      specs: list[dict[str, Any]] | None = None) -> tuple[list[dict], dict]:
    """The four-option question for every device agent and the centralised agent (or only the given specs) on every
    task; returns (records, priors by agent)."""
    variant = config["prompts"]["variant"]
    daily = get_variants()[variant]["daily"]
    window_days = config["data"]["window_days"]
    letters = option_letters(templates)
    option_texts = [o["text"] for o in templates["shared"]["options"]]
    token_ids = handle.backend.option_token_ids(letters)
    prefix = templates["shared"].get("assistant_prefix")
    specs = device_agent_specs(config) if specs is None else specs
    missing = sorted({d for s in specs for d in s["devices"]} - set(templates["agents"]))
    assert not missing, f"run_device_agents: agents {missing} have no template in {sorted(templates['agents'])}"
    version = template_version(templates)
    priors = {}
    for spec in specs:
        system, user = render_spec(templates, spec, tasks[0]["split"], False, daily, window_days)
        priors[spec["agent_id"]] = score_prior(handle, chat_messages(system, user, handle.model_class, prefix),
                                               token_ids, telemetry, spec["agent_id"])
    records = []
    for task in tasks:
        with start_span(telemetry, f"task {task['task_index']}", {"bg.task.index": task["task_index"],
                                                                  "bg.task.id": task["task_id"]}):
            for spec in specs:
                with start_span(telemetry, f"invoke_agent {spec['agent_id']}",
                                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.id": spec["agent_id"],
                                 "bg.agent.role": spec["role"], "bg.task.index": task["task_index"]}) as span:
                    system, user = render_spec(templates, spec, task["split"], True, daily, window_days)
                    add_event(telemetry, span, "prompt_rendered", {"system": system, "user": user},
                              content_keys=("system", "user"))
                    result, latency = generate_with_scores(handle, chat_messages(system, user, handle.model_class,
                                                                                 prefix),
                                                           token_ids, config["generation"], telemetry,
                                                           spec["agent_id"])
                    letter = parse_option_letter(result.text, letters, option_texts)
                    add_event(telemetry, span, "answer_parsed",
                              {"letter": letter, "option_logprobs_posterior": result.option_logprobs})
                    records.append(make_wqa_record(
                        run_id, task, spec["agent_id"], spec["agent_index"], spec["role"], handle.name, result,
                        letter, priors[spec["agent_id"]], result.option_logprobs, latency, span,
                        {"condition": spec["role"], "agent_devices": spec["devices"], "agent_axis": spec_axis(
                            templates, spec), "prompt_variant": variant, "template_version": version}))
    return records, priors


def side_logprobs(scores_by_order: list[dict[str, float]], axis: str) -> dict[str, float]:
    """Letter scores of each order -> side log-probabilities, averaged over the orders and renormalised."""
    per_side: dict[str, list[float]] = {}
    for order, scores in enumerate(scores_by_order):
        for letter, side in axis_letter_sides(axis, order):
            per_side.setdefault(side, []).append(scores[letter])
    mean = {side: sum(values) / len(values) for side, values in per_side.items()}
    total = logsumexp(list(mean.values()))
    return {side: value - total for side, value in mean.items()}


def classes_from_sides(axis: str, sides: dict[str, float]) -> dict[str, float]:
    """Four-class log-probabilities (fixed letters) from one axis: each side's mass split evenly over its classes."""
    fixed = ws.fixed_letter_map()
    out = {}
    for side, codes in side_classes()[axis].items():
        for code in codes:
            out[fixed[code]] = sides[side] - math.log(len(codes))
    return dict(sorted(out.items()))


def classes_from_both_axes(fitness: dict[str, float], metabolic: dict[str, float]) -> dict[str, float]:
    """Four-class log-probabilities from both axes: log p(fitness side) + log p(metabolic side)."""
    fixed = ws.fixed_letter_map()
    return dict(sorted((fixed[code], fitness[class_side(code, "fitness")] + metabolic[class_side(code, "metabolic")])
                       for code in fixed))


def parsed_side(text: str, axis: str, order: int, templates: dict[str, Any]) -> str | None:
    """The side a two-option reply chose, or None."""
    texts = {o["side"]: o["text"] for o in templates["shared"]["axis_questions"][axis]["options"]}
    pairs = axis_letter_sides(axis, order)
    letter = parse_option_letter(text, [l for l, _ in pairs], [texts[s] for _, s in pairs])
    return dict(pairs).get(letter) if letter else None


def ask_axis(handle: ModelHandle, render: Callable[[int, bool], tuple[str, str]], axis: str, orders: list[int],
             prior_scores: list[dict[str, float]] | None, templates: dict[str, Any], config: dict[str, Any],
             telemetry: Telemetry, agent_id: str) -> dict[str, Any]:
    """One two-option question in every letter order: generate in order 0, prompt-only scores in the others."""
    prefix = templates["shared"].get("assistant_prefix")
    token_ids = handle.backend.option_token_ids(["A", "B"])
    if prior_scores is None:
        prior_scores = []
        for order in orders:
            system, user = render(order, False)
            prior_scores.append(score_prior(handle, chat_messages(system, user, handle.model_class, prefix),
                                            token_ids, telemetry, agent_id))
    scores, result, latency = [], None, 0.0
    for order in orders:
        system, user = render(order, True)
        messages = chat_messages(system, user, handle.model_class, prefix)
        if order == 0:
            result, latency = generate_with_scores(handle, messages, token_ids, config["generation"], telemetry,
                                                   agent_id)
            scores.append(result.option_logprobs)
        else:
            scores.append(score_options(handle, messages, token_ids))
    return {"axis": axis, "result": result, "latency": latency, "scores_by_order": scores,
            "prior_scores_by_order": prior_scores, "side": parsed_side(result.text, axis, 0, templates),
            "posterior_sides": side_logprobs(scores, axis), "prior_sides": side_logprobs(prior_scores, axis)}


def axis_record_fields(answer: dict[str, Any], gold_class: str) -> dict[str, Any]:
    """The per-axis fields of a record."""
    axis = answer["axis"]
    return {"axis_question": axis, "predicted_side": answer["side"], "true_side": class_side(gold_class, axis),
            "side_correct": answer["side"] == class_side(gold_class, axis),
            "option_logprobs_axis_prior": answer["prior_sides"],
            "option_logprobs_axis_posterior": answer["posterior_sides"],
            "option_logprobs_by_order": answer["scores_by_order"], "raw_response_by_axis": answer["result"].text}


def run_axis_agents(tasks: list[dict[str, Any]], templates: dict[str, Any], handle: ModelHandle,
                    config: dict[str, Any], telemetry: Telemetry, run_id: str) -> tuple[list[dict], dict]:
    """Axis-question variant: every device agent answers its own two-option question; the centralised agent answers
    both (combined, agent 'centralised_axis') and the four-option question (agent 'centralised')."""
    validate_axis_shared(templates["shared"])
    orders = list(range(config["prompts"].get("axis_orders", 2)))
    window_days = config["data"]["window_days"]
    version = template_version(templates)
    variant = config["prompts"]["variant"]
    specs = device_agent_specs(config)
    devices = [s for s in specs if s["role"] == "device"]
    missing = sorted({s["devices"][0] for s in devices} - set(templates["agents"]))
    assert not missing, f"run_axis_agents: agents {missing} have no template in {sorted(templates['agents'])}"
    central = [s for s in specs if s["role"] == "centralised"]
    priors: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for task in tasks:
        split = task["split"]
        with start_span(telemetry, f"task {task['task_index']}", {"bg.task.index": task["task_index"],
                                                                  "bg.task.id": task["task_id"]}):
            for spec in devices:
                name = spec["devices"][0]
                axis = asked_axis(templates, name)
                with start_span(telemetry, f"invoke_agent {name}", {"gen_ai.agent.id": name,
                                                                     "bg.axis": axis}) as span:
                    answer = ask_axis(
                        handle, lambda order, with_data: render_axis_agent(templates, name, split, order, with_data,
                                                                           window_days),
                        axis, orders, priors.get(name), templates, config, telemetry, name)
                    priors[name] = answer["prior_scores_by_order"]
                    fields = axis_record_fields(answer, task["gold_class"])
                    records.append(make_wqa_record(
                        run_id, task, name, spec["agent_index"], "device", handle.name, answer["result"], None,
                        classes_from_sides(axis, answer["prior_sides"]),
                        classes_from_sides(axis, answer["posterior_sides"]), answer["latency"], span,
                        {"condition": "device", "agent_devices": [name], "agent_axis": axis,
                         "prompt_variant": variant, "template_version": version, **fields,
                         "predicted_answer": answer["side"], "parse_ok": answer["side"] is not None,
                         "correct": fields["side_correct"]}))
            for spec in central:
                agent_id = "centralised_axis"
                with start_span(telemetry, f"invoke_agent {agent_id}", {"gen_ai.agent.id": agent_id}) as span:
                    answers = {}
                    for axis in axis_names():
                        key = f"{agent_id}:{axis}"
                        answers[axis] = ask_axis(
                            handle, lambda order, with_data, axis=axis: render_axis_centralised(
                                templates, split, spec["devices"], axis, order, with_data, window_days),
                            axis, orders, priors.get(key), templates, config, telemetry, agent_id)
                        priors[key] = answers[axis]["prior_scores_by_order"]
                    fit, met = answers["fitness"], answers["metabolic"]
                    letter = None
                    if fit["side"] and met["side"]:
                        code = next(c for c in ws.core_classes() if class_side(c, "fitness") == fit["side"]
                                    and class_side(c, "metabolic") == met["side"])
                        letter = ws.fixed_letter_map()[code]
                    records.append(make_wqa_record(
                        run_id, task, agent_id, spec["agent_index"] + 1, "centralised", handle.name, fit["result"],
                        letter, classes_from_both_axes(fit["prior_sides"], met["prior_sides"]),
                        classes_from_both_axes(fit["posterior_sides"], met["posterior_sides"]),
                        fit["latency"] + met["latency"], span,
                        {"condition": "centralised", "agent_devices": spec["devices"], "agent_axis": "both",
                         "prompt_variant": variant, "template_version": version,
                         "axes": {axis: axis_record_fields(a, task["gold_class"]) for axis, a in answers.items()},
                         "raw_response": f"fitness: {fit['result'].text!r}; metabolic: {met['result'].text!r}"}))
    if central:
        four_records, four_priors = run_device_agents(tasks, templates, handle, config, telemetry, run_id, central)
        records += four_records
        priors.update(four_priors)
    return records, priors


def restrict_to_core(scores: dict[str, float], letter_map: dict[str, str]) -> dict[str, float]:
    """Original-letter scores -> fixed-letter scores (A-D) for the four gold-capable options."""
    return {fixed: scores[original] for original, fixed in sorted(letter_map.items(), key=lambda kv: kv[1])}


def run_original_agent(tasks: list[dict[str, Any]], handle: ModelHandle, config: dict[str, Any],
                       telemetry: Telemetry, run_id: str) -> tuple[list[dict], dict]:
    """The original WearableQA prompt (one agent, ten options) on every task; the prior is scored per task."""
    window_days = config["data"]["window_days"]
    records, priors = [], {}
    for task in tasks:
        question = task["question"]
        letters = original_letters(question)
        texts = [question["options"][letter] for letter in letters]
        letter_map = original_core_letter_map(question)
        token_ids = handle.backend.option_token_ids(letters)
        with start_span(telemetry, f"task {task['task_index']}", {"bg.task.index": task["task_index"],
                                                                  "bg.task.id": task["task_id"]}):
            with start_span(telemetry, "invoke_agent original", {"gen_ai.operation.name": "invoke_agent",
                                                                 "gen_ai.agent.id": "original"}) as span:
                system, user = render_original(question, task["rows"], task["cohort_reference"], False, window_days)
                prior_full = score_prior(handle, chat_messages(system, user, handle.model_class), token_ids,
                                         telemetry, "original")
                system, user = render_original(question, task["rows"], task["cohort_reference"], True, window_days)
                add_event(telemetry, span, "prompt_rendered", {"user": user}, content_keys=("user",))
                result, latency = generate_with_scores(handle, chat_messages(system, user, handle.model_class),
                                                       token_ids, config["generation"], telemetry, "original")
                original_letter = parse_option_letter(result.text, letters, texts)
                letter = letter_map.get(original_letter) if original_letter else None
                priors[task["task_id"]] = prior_full
                records.append(make_wqa_record(
                    run_id, task, "original", 0, "original", handle.name, result, letter,
                    restrict_to_core(prior_full, letter_map), restrict_to_core(result.option_logprobs, letter_map),
                    latency, span,
                    {"condition": "original", "agent_devices": [], "agent_axis": "both", "prompt_variant": "original",
                     "template_version": "original_wording_28day_summary:1", "original_letter_map": letter_map,
                     "predicted_letter_original": original_letter, "parse_ok": original_letter is not None,
                     "option_logprobs_prior_full": prior_full,
                     "option_logprobs_posterior_full": result.option_logprobs}))
    return records, priors


# ----------------------------------------------------------------------------------------------- run


def accuracy_by_agent(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Accuracy and parse failures of the generated answer, per agent id (for an agent asked a two-option axis
    question: accuracy of the side it chose on its own question)."""
    out: dict[str, dict[str, float]] = {}
    for agent_id in dict.fromkeys(r["agent_id"] for r in records):
        own = [r for r in records if r["agent_id"] == agent_id]
        out[agent_id] = {"accuracy": sum(r["correct"] for r in own) / len(own), "answers": len(own),
                         "parse_failures": sum(not r["parse_ok"] for r in own)}
    return out


def get_wqa_run_dir(results_dir: Path, model_name: str, variant: str, num_samples: int, total: int) -> Path:
    """results/wqa_<model>_<variant>_full_run, or ..._first{N}_run."""
    scope = "full_run" if num_samples == total else f"first{num_samples}_run"
    return Path(results_dir) / f"wqa_{model_name.replace('/', '__')}_{variant}_{scope}"


def save_wqa_results(run_dir: Path, stem: str, records: list[dict[str, Any]], summary: dict[str, Any],
                     priors: dict[str, Any], templates: dict[str, Any] | None, config: dict[str, Any]) -> dict:
    """Predictions, summary, priors, config and a copy of every prompt file used; returns the file paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {"predictions": run_dir / f"predictions_{stem}.json", "summary": run_dir / f"summary_{stem}.json",
             "priors": run_dir / f"priors_{stem}.json", "config": run_dir / f"config_{stem}.yaml"}
    files["predictions"].write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    files["priors"].write_text(json.dumps(priors, indent=2, ensure_ascii=False), encoding="utf-8")
    files["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if templates is not None:
        prompt_dir = run_dir / f"prompts_{stem}"
        prompt_dir.mkdir(exist_ok=True)
        for name, text in templates["raw_text"].items():
            (prompt_dir / f"{name}.yaml").write_text(text, encoding="utf-8")
        files["prompts"] = prompt_dir
    summary["files"] = {key: str(path) for key, path in files.items()}
    files["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary["files"]


def run_wqa_experiment(config: dict[str, Any], paths: dict[str, Path], num_samples: int | None = None,
                       loader_fn: Callable | None = None, span_exporter: Any = None,
                       download: bool = True) -> dict[str, Any]:
    """Score the configured agents on the first N tasks and save the run folder; returns the summary."""
    started = time.perf_counter()
    variant = config["prompts"]["variant"]
    settings = get_variants()[variant]
    tasks = ensure_tasks(paths["tasks"], paths["raw"], config["data"]["window_days"])
    total = len(tasks)
    n = total if num_samples is None else num_samples
    assert type(n) is int and 1 <= n <= total, (
        f"run_wqa_experiment: num_samples is {n!r}; must be between 1 and the number of tasks {total}"
    )
    tasks = tasks[:n]
    templates = None
    if not settings["original"]:
        overlay = paths["prompt_root"] / config["prompts"][settings["overlay"]] if settings["overlay"] else None
        templates = load_wqa_templates(paths["prompt_root"] / config["prompts"]["dir"], overlay)
    handles = load_all_models(config, paths["model_dir"], loader_fn, download)
    handle = handles[config["agents"]["default_model"]]
    run_dir = get_wqa_run_dir(paths["results_dir"], handle.name, variant, n, total)
    stem = f"wqa_{variant}_N{n}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_info = {"bg.run.id": run_id, "bg.run.name": config["experiment"]["name"],
                "bg.dataset": "wearableqa_phenotype", "bg.variant": variant, "bg.n": n, "bg.model": handle.name}
    trace_path = run_dir / f"traces_{stem}.otlp.jsonl"
    if config["telemetry"]["enabled"] and span_exporter is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
    telemetry = setup_telemetry(trace_path, run_info, config["telemetry"]["enabled"],
                                config["telemetry"]["capture_content"], config["telemetry"].get("otlp_endpoint"),
                                span_exporter)
    with start_span(telemetry, f"run {config['experiment']['name']}", run_info):
        if settings["original"]:
            records, priors = run_original_agent(tasks, handle, config, telemetry, run_id)
        elif settings["axis"]:
            records, priors = run_axis_agents(tasks, templates, handle, config, telemetry, run_id)
        else:
            records, priors = run_device_agents(tasks, templates, handle, config, telemetry, run_id)
    shutdown_telemetry(telemetry)
    summary = {
        "schema_version": get_schema_version(),
        "run_id": run_id,
        "run_name": config["experiment"]["name"],
        "dataset": "wearableqa_phenotype",
        "prompt_variant": variant,
        "num_samples": n,
        "total_samples": total,
        "num_records": len(records),
        "agents": sorted(dict.fromkeys(r["agent_id"] for r in records)),
        "template_version": template_version(templates) if templates is not None else "original",
        "model": {"hf_name": handle.name, "device": handle.device, "dtype": handle.dtype,
                  "model_class": handle.model_class},
        "accuracy": accuracy_by_agent(records),
        "memory": memory_report(handle.device),
        "run_time_s": time.perf_counter() - started,
        "trace_file": str(trace_path) if telemetry.enabled and span_exporter is None else None,
        "config": config,
    }
    save_wqa_results(run_dir, stem, records, summary, priors, templates, config)
    return summary


# ----------------------------------------------------------------------------------------------- CLI


def get_code_root() -> Path:
    """The code/ folder (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path, root: Path) -> Path:
    """Absolute paths stay; relative paths are taken relative to root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments (defaults come from the config unless given)."""
    parser = argparse.ArgumentParser(description="Score WearableQA phenotype tasks with device agents.")
    parser.add_argument("--config", default="config/wearableqa_bluebear.yaml", help="config file (relative to code/)")
    parser.add_argument("--model", default=None, help="Hugging Face name; a model listed in the config is selected, "
                                                      "any other name replaces the default model")
    parser.add_argument("--variant", choices=sorted(get_variants()), default=None, help="prompt variant")
    parser.add_argument("--num_samples", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    parser.add_argument("--data_dir", default="data", help="data root (relative to code/, or absolute)")
    parser.add_argument("--model_dir", default="model", help="model root (relative to code/, or absolute)")
    parser.add_argument("--results_dir", default="results", help="parent of the run folders")
    parser.add_argument("--no_trace", action="store_true", help="do not write the OpenTelemetry trace file")
    parser.add_argument("--no_trace_content", action="store_true", help="leave prompts and replies out of the trace")
    parser.add_argument("--download_only", action="store_true", help="download the model, build tasks.jsonl, exit")
    return parser.parse_args(argv)


def apply_wqa_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply command-line overrides and re-validate."""
    if args.model is not None:
        listed = [key for key, m in config["models"].items() if m["hf_name"] == args.model]
        if listed:
            config["agents"]["default_model"] = listed[0]
        else:
            model = config["models"][config["agents"]["default_model"]]
            model["hf_name"], model["model_class"] = args.model, "auto"
    default_model = config["models"][config["agents"]["default_model"]]
    if args.dtype is not None:
        default_model["dtype"] = args.dtype
    if args.device is not None:
        for model in config["models"].values():
            model["device"] = args.device
    if args.variant is not None:
        config["prompts"]["variant"] = args.variant
    if args.no_trace:
        config["telemetry"]["enabled"] = False
    if args.no_trace_content:
        config["telemetry"]["capture_content"] = False
    validate_wqa_config(config)
    return config


def get_wqa_paths(config: dict[str, Any], args: argparse.Namespace, code_root: Path) -> dict[str, Path]:
    """Every path the run needs, resolved against code/ (data files against the data root)."""
    data_dir = resolve_path(args.data_dir, code_root)
    return {"data_dir": data_dir, "raw": resolve_path(config["data"]["raw_file"], data_dir),
            "tasks": resolve_path(config["data"]["tasks_file"], data_dir),
            "model_dir": resolve_path(args.model_dir, code_root),
            "results_dir": resolve_path(args.results_dir, code_root), "prompt_root": code_root}


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    """Load config, apply overrides, run (or only download and prepare), print per-agent accuracy."""
    args = parse_args(argv)
    code_root = get_code_root()
    config = apply_wqa_overrides(load_wqa_config(resolve_path(args.config, code_root)), args)
    paths = get_wqa_paths(config, args, code_root)
    if args.download_only:
        ensure_tasks(paths["tasks"], paths["raw"], config["data"]["window_days"])
        download_model(config["models"][config["agents"]["default_model"]]["hf_name"], paths["model_dir"])
        print(f"Tasks ready at {paths['tasks']}; model in {paths['model_dir']}")
        return None
    summary = run_wqa_experiment(config, paths, args.num_samples)
    print(json.dumps({"accuracy": summary["accuracy"], "results": summary["files"]["predictions"]}, indent=2))
    return summary


if __name__ == "__main__":
    main()
