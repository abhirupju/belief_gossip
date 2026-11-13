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


def test_mean_log_distribution():
    mean = aw.mean_log_distribution([{"a": math.log(0.9), "b": math.log(0.1)}, {"a": math.log(0.5), "b": math.log(0.5)}])
    assert math.exp(mean["a"]) == pytest.approx(0.7) and math.exp(mean["b"]) == pytest.approx(0.3)


def test_empirical_prior_views():
    def view(user, p):
        posterior = {"A": math.log(p), "B": math.log(1 - p)}
        return {"user_id": user, "agents": [{"prior": {"A": math.log(0.99), "B": math.log(0.01)},
                                             "posterior": posterior, "lift": {}}]}
    views = [view("u1", 0.8), view("u2", 0.4), view("u3", 0.1)]
    out = aw.empirical_prior_views(views, {"u1", "u2"})
    prior = out[0]["agents"][0]["prior"]
    assert math.exp(prior["A"]) == pytest.approx(0.6)                      # mean of 0.8 and 0.4; u3 unused
    assert all(v["agents"][0]["prior"] == prior for v in out)
    assert out[2]["agents"][0]["lift"]["A"] == pytest.approx(math.log(0.1) - math.log(0.6))
    assert views[0]["agents"][0]["prior"]["A"] == pytest.approx(math.log(0.99))   # input untouched


def test_axis_order_and_empirical_prior_analysis(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader(axis_reply)
    config = copy.deepcopy(wqa_config)
    config["prompts"]["variant"] = "axis_questions"
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, loader_fn=loader, download=False)
    path = Path(summary["files"]["predictions"])
    records = json.loads(path.read_text(encoding="utf-8"))
    priors = json.loads(aw.priors_path_for(path).read_text(encoding="utf-8"))
    for order in (0, 1):
        rebuilt = aw.records_at_axis_order(records, priors, order)
        watch = next(r for r in rebuilt if r["agent_id"] == "watch")
        scores = watch["option_logprobs_by_order"][order]
        favourable = scores["A"] if order == 0 else scores["B"]
        total = math.log(math.exp(scores["A"]) + math.exp(scores["B"]))
        assert watch["option_logprobs_axis_posterior"]["favourable"] == pytest.approx(favourable - total)
        assert sum(math.exp(v) for v in watch["option_logprobs_posterior"].values()) == pytest.approx(1.0)
        central = next(r for r in rebuilt if r["agent_id"] == "centralised_axis")
        assert central["option_logprobs_posterior"] == wqa_exp.classes_from_both_axes(
            central["axes"]["fitness"]["option_logprobs_axis_posterior"],
            central["axes"]["metabolic"]["option_logprobs_axis_posterior"])
        assert next(r for r in rebuilt if r["agent_id"] == "centralised") == next(
            r for r in records if r["agent_id"] == "centralised")
    tasks = wqa_prepare.load_tasks(wqa_paths["tasks"])
    report = aw.analyze_wqa(path, [0.2], [1], ["lift"], ["joint"], n_splits=5, tasks=tasks, axis_order="1")
    assert report["settings"] == dict(report["settings"], prior="empirical", axis_order="1")
    assert all(r["prior"] == "empirical" for results in report["conformal"].values() for r in results)
    assert "argmax_lift_empirical" in report["f1"]["watch"]["own_axis"]
    assert "argmax_lift_empirical" in report["f1"]["centralised"]["per_class"]
    assert aw.save_wqa_report(report).name.endswith("_prior-empirical_order-1.json")
    prompt = aw.analyze_wqa(path, [0.2], [1], ["lift"], ["joint"], n_splits=5, prior="prompt")
    assert aw.report_suffix(prompt) == ""
    with pytest.raises(AssertionError, match="not scored"):
        single = [dict(r, option_logprobs_by_order=r["option_logprobs_by_order"][:1]) for r in records
                  if "axis_question" in r]
        aw.records_at_axis_order(single, priors, 1)


def test_classwise_thresholds_and_sets():
    score_fn = aw.get_score_functions()["lac"]

    def view(true, p_a):
        posterior = {"A": math.log(p_a), "B": math.log(1 - p_a)}
        return {"true_letter": true, "letters": ["A", "B"],
                "agents": [{"posterior": posterior, "prior": posterior, "lift": {"A": 0.0, "B": 0.0}}]}
    calibration = [view("A", p) for p in (0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.75, 0.65, 0.55, 0.99,
                                          0.91, 0.81, 0.71, 0.61, 0.51, 0.92, 0.82, 0.72, 0.62)] + [view("B", 0.3)]
    thresholds = aw.classwise_thresholds(calibration, score_fn, 0.1, 1)
    assert thresholds["A"] == pytest.approx(1 - 0.55)          # rank ceil(20 x 0.9) = 18 of 19 scores
    assert math.isinf(thresholds["B"])                          # B: 1 question < 1/0.1 - 1
    assert aw.classwise_set(view("A", 0.4), score_fn, thresholds, 1) == ["B"]            # A excluded, B always kept
    assert aw.classwise_set(view("A", 0.6), score_fn, thresholds, 1) == ["A", "B"]


def test_classes_from_side_sets():
    fixed = ws.fixed_letter_map()
    assert aw.classes_from_side_sets(["favourable"], ["adverse"]) == [fixed["D-FA"]]
    assert aw.classes_from_side_sets(["low"], ["favourable", "adverse"]) == sorted([fixed["CA"], fixed["D-LF"]])
    assert aw.classes_from_side_sets([], ["adverse"]) == []


def test_structured_and_classwise_report(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader(axis_reply)
    config = copy.deepcopy(wqa_config)
    config["prompts"]["variant"] = "axis_questions"
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, loader_fn=loader, download=False)
    path = Path(summary["files"]["predictions"])
    records = json.loads(path.read_text(encoding="utf-8"))
    groups = aw.judgement_conditions(records)
    assert groups["fusion_no_ring"] == {"fitness": ["watch", "phone"], "metabolic": ["clinic_glycaemic", "clinic_lipid"]}
    assert groups["fusion_with_ring"]["fitness"] == ["watch", "phone", "ring"]
    assert groups["watch_and_clinics"]["fitness"] == ["watch"]
    assert groups["centralised_axis"]["metabolic"] == ["centralised_axis"]
    views = aw.side_views(records, ["watch", "phone"], "fitness")
    assert len(views) == 8 and all(v["letters"] == ["favourable", "low"] for v in views)
    assert all(v["true_letter"] == aw.side_of(v["gold_class"], "fitness") for v in views)
    with pytest.raises(AssertionError, match="did not answer"):
        aw.side_views(records, ["watch"], "metabolic")
    report = aw.analyze_wqa(path, [0.2], [1], ["lac"], ["joint"], n_splits=10)
    assert set(report["structured"]) == set(groups) and set(report["classwise"]) == set(report["conditions"])
    result = report["structured"]["fusion_no_ring"]["results"][0]
    assert result["alpha_per_judgement"] == pytest.approx(0.1)
    assert set(result["per_side"]) == {"fitness", "metabolic"} and "D-FA" in result["per_class"]
    assert all(0.0 <= result["metrics"][m]["mean"] for m in ("coverage", "mean_set_size"))
    classwise = report["classwise"]["fusion_no_ring"][0]
    assert classwise["infinite_threshold_share"]["D-FA"] == 1.0          # 1 D-FA question: never calibratable
    assert classwise["per_class"]["D-FA"]["coverage"] == 1.0
    text = aw.format_wqa_report(report, "lac", "joint")
    assert "structured conformal" in text and "classwise conformal" in text and "D-FA" in text


def test_no_structured_sets_without_axis_questions(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader(lambda text: "A")
    summary = wqa_exp.run_wqa_experiment(copy.deepcopy(wqa_config), wqa_paths, num_samples=4, loader_fn=loader,
                                         download=False)
    records = json.loads(Path(summary["files"]["predictions"]).read_text(encoding="utf-8"))
    assert aw.judgement_conditions(records) == {}
