"""The axis-question variant: prompts, score mapping, end-to-end run with the fake backend, F1 analysis, prefill."""

import copy
import json
import math
from pathlib import Path

import re

import pytest

import analyze_wqa as aw
import wqa_exp
import wqa_prepare
from utility import model_utils as mu
from utility import wqa_prompts as wp
from utility import wqa_split as ws


def axis_reply(text: str) -> str:
    """Fit when mean resting HR is below 60, adverse when glucose is 120; picks the letter of that answer's text."""
    if "fitness/activity favourable or low" in text:
        rhr = re.search(r"Resting heart rate: mean ([0-9.]+)", text)
        fit = rhr is not None and float(rhr.group(1)) < 60
        wanted = "Favourable fitness/activity." if fit else "Low fitness/activity."
    elif "metabolic profile favourable or adverse" in text:
        wanted = "Adverse metabolic profile." if "Fasting glucose: 120" in text else "Favourable metabolic profile."
    else:
        return "C"
    return "A" if f"A. {wanted}" in text else "B"


def test_axis_prompts_both_orders(wqa_prompt_dir, wqa_axis_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir, wqa_axis_dir)
    split = ws.split_question(ws.select_questions(wqa_raw)[0], wqa_raw)
    _, first = wp.render_axis_agent(templates, "watch", split, 0)
    _, second = wp.render_axis_agent(templates, "watch", split, 1)
    assert "A. Favourable fitness/activity.\nB. Low fitness/activity." in first
    assert "A. Low fitness/activity.\nB. Favourable fitness/activity." in second
    assert "says nothing about the metabolic judgement" not in first and "(A or B)" in first
    system, _ = wp.render_axis_agent(templates, "watch", split, 0)
    assert system.endswith("Reply with one letter only.")
    _, ring = wp.render_axis_agent(templates, "ring", split, 0)
    assert "fitness/activity favourable or low" in ring
    _, clinic = wp.render_axis_agent(templates, "clinic_lipid", split, 0, with_data=False)
    assert "metabolic profile favourable or adverse" in clinic and "Triglycerides" not in clinic
    _, central = wp.render_axis_centralised(templates, split, ["watch", "clinic_lipid"], "metabolic", 0)
    assert "Resting heart rate" in central and "Triglycerides" in central and "{" not in central
    assert templates["shared"]["assistant_prefix"] == "Answer:"


def test_main_templates_have_no_axis_questions(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    split = ws.split_question(ws.select_questions(wqa_raw)[0], wqa_raw)
    with pytest.raises(AssertionError, match="axis overlay"):
        wp.render_axis_agent(templates, "watch", split, 0)
    with pytest.raises(AssertionError, match="no axis question"):
        wp.asked_axis(templates, "ring")


def test_side_and_class_scores():
    order0 = {"A": math.log(0.8), "B": math.log(0.2)}   # A = favourable
    order1 = {"A": math.log(0.4), "B": math.log(0.6)}   # A = low
    sides = wqa_exp.side_logprobs([order0, order1], "fitness")
    expected_fav = (math.log(0.8) + math.log(0.6)) / 2
    expected_low = (math.log(0.2) + math.log(0.4)) / 2
    total = math.log(math.exp(expected_fav) + math.exp(expected_low))
    assert sides["favourable"] == pytest.approx(expected_fav - total)
    classes = wqa_exp.classes_from_sides("fitness", sides)
    assert sum(math.exp(v) for v in classes.values()) == pytest.approx(1.0)
    assert classes["A"] == classes["D"] and classes["B"] == classes["C"]  # CH = D-FA, CA = D-LF
    metabolic = {"favourable": math.log(0.9), "adverse": math.log(0.1)}
    both = wqa_exp.classes_from_both_axes(sides, metabolic)
    assert sum(math.exp(v) for v in both.values()) == pytest.approx(1.0)
    assert both["A"] == pytest.approx(sides["favourable"] + math.log(0.9))   # CH: fit, favourable labs
    assert both["D"] == pytest.approx(sides["favourable"] + math.log(0.1))   # D-FA: fit, adverse labs


def test_chat_messages_prefill():
    messages = wqa_exp.chat_messages("sys", "user", "causal", "Answer:")
    assert messages[-1] == {"role": "assistant", "content": "Answer:"}
    multimodal = wqa_exp.chat_messages("sys", "user", "multimodal", "Answer:")
    assert multimodal[-1]["content"] == [{"type": "text", "text": "Answer:"}]
    assert wqa_exp.chat_messages("sys", "user", "causal")[-1]["role"] == "user"


def test_hf_backend_continues_a_prefill():
    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        tokenizer = None

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return FakeInputs()

    processor = FakeProcessor()
    backend = mu.HFBackend(None, processor, "cpu")
    backend.build_inputs([{"role": "user", "content": "q"}])
    assert processor.kwargs["add_generation_prompt"] and not processor.kwargs["continue_final_message"]
    backend.build_inputs([{"role": "user", "content": "q"}, {"role": "assistant", "content": "Answer:"}])
    assert not processor.kwargs["add_generation_prompt"] and processor.kwargs["continue_final_message"]


def test_axis_run_end_to_end(wqa_config, wqa_paths, make_loader):
    loader, tracker = make_loader(axis_reply)
    config = copy.deepcopy(wqa_config)
    config["prompts"]["variant"] = "axis_questions"
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, loader_fn=loader, download=False)
    records = json.loads(Path(summary["files"]["predictions"]).read_text(encoding="utf-8"))
    devices = ["watch", "phone", "ring", "clinic_glycaemic", "clinic_lipid"]
    agents = devices + ["centralised_axis", "centralised"]
    assert sorted({r["agent_id"] for r in records}) == sorted(agents) and len(records) == 8 * len(agents)
    calls = tracker["backends"][0].calls
    priors = 5 * 2 + 2 * 2 + 1                       # devices x orders, centralised two axes x orders, four-option
    assert sum(c["kind"] == "score" for c in calls) == priors + 8 * (5 + 2)
    assert sum(c["kind"] == "generate" for c in calls) == 8 * (5 + 2 + 1)
    tasks = {t["task_id"]: t for t in wqa_prepare.load_tasks(wqa_paths["tasks"])}
    for record in records:
        assert set(record["option_logprobs_posterior"]) == {"A", "B", "C", "D"}
        assert sum(math.exp(v) for v in record["option_logprobs_posterior"].values()) == pytest.approx(1.0)
    watch = [r for r in records if r["agent_id"] == "watch"]
    assert all(r["axis_question"] == "fitness" and r["predicted_letter"] is None for r in watch)
    assert all(r["side_correct"] == r["correct"] for r in watch)
    assert all(r["side_correct"] for r in watch)  # the fake reads resting HR perfectly
    glycaemic = [r for r in records if r["agent_id"] == "clinic_glycaemic"]
    assert all(r["correct"] for r in glycaemic)
    central = [r for r in records if r["agent_id"] == "centralised_axis"]
    assert all(r["predicted_letter"] == tasks[r["task_id"]]["gold_letter"] for r in central)
    ring = [r for r in records if r["agent_id"] == "ring"]
    assert {r["agent_axis"] for r in ring} == {"fitness"}
    priors_file = json.loads(Path(summary["files"]["priors"]).read_text(encoding="utf-8"))
    assert len(priors_file["watch"]) == 2 and "centralised_axis:metabolic" in priors_file
    report = aw.analyze_wqa(Path(summary["files"]["predictions"]), [0.2], [1], ["lift"], ["joint"], n_splits=10,
                            tasks=list(tasks.values()))
    watch_f1 = report["f1"]["watch"]
    assert watch_f1["two_option"] and watch_f1["own_axis"]["generated"]["macro"] == 1.0
    assert "ideal" in watch_f1["own_axis"]
    assert report["f1"]["centralised_axis"]["per_class"]["generated"]["macro"] == 1.0
    assert "fusion_no_ring" in report["conformal"]
    assert "own judgement" in aw.format_wqa_report(report, "lift", "joint")


def test_axis_single_order(wqa_config, wqa_paths, make_loader):
    loader, tracker = make_loader(axis_reply)
    config = copy.deepcopy(wqa_config)
    config["prompts"].update(variant="axis_questions", axis_orders=1)
    config["agents"].update(devices=["watch"], centralised=False)
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, num_samples=2, loader_fn=loader, download=False)
    records = json.loads(Path(summary["files"]["predictions"]).read_text(encoding="utf-8"))
    assert len(records) == 2 and all(len(r["option_logprobs_by_order"]) == 1 for r in records)


def test_f1_per_label():
    f1 = aw.f1_per_label(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"])
    assert f1["a"] == pytest.approx(2 / 3)                 # P 1, R 1/2
    assert f1["b"] == pytest.approx(1 / 2)                 # P 1/2, R 1/2 (unparsed counts as a miss)
    assert f1["macro"] == pytest.approx(7 / 12)
