"""WearableQA analysis: user-grouped splits, calibration exclusion, agreement with run_conformal, full report."""

import json
import math
import random
from pathlib import Path

import pytest

import analyze_wqa as aw
import wqa_exp
import wqa_prepare
from analysis.conformal import run_conformal


def synthetic_views(n_tasks: int, n_agents: int, seed: int, users: list[str], classes: list[str]):
    """Random task views in the shape analysis.scores.build_task_views makes, with user and class."""
    rng = random.Random(seed)
    letters = ["A", "B", "C", "D"]
    letter_of = {"CH": "A", "CA": "B", "D-LF": "C", "D-FA": "D"}
    views = []
    for i in range(n_tasks):
        agents = []
        for a in range(n_agents):
            raw = {l: rng.gauss(0, 1) for l in letters}
            total = math.log(sum(math.exp(v) for v in raw.values()))
            posterior = {l: v - total for l, v in raw.items()}
            prior = {l: -math.log(4) for l in letters}
            agents.append({"agent_index": a, "prior": prior, "posterior": posterior,
                           "lift": {l: posterior[l] - prior[l] for l in letters}, "generated": []})
        views.append({"task_index": i, "true_letter": letter_of[classes[i]], "letters": letters, "agents": agents,
                      "user_id": users[i], "gold_class": classes[i]})
    return views


def test_group_splits_keep_users_together():
    users = ["u1", "u1", "u2", "u3", "u3", "u3", "u4", "u5", "u6"]
    classes = ["CH", "CA", "D-LF", "D-FA", "CH", "CA", "CH", "CA", "D-FA"]
    views = synthetic_views(len(users), 2, 0, users, classes)
    splits = aw.make_group_splits(users, 50, 0)
    assert splits == aw.make_group_splits(users, 50, 0)  # deterministic
    for groups in splits:
        assert len(groups) == 3
        calibration, test = aw.split_views(views, groups, ("D-FA",))
        assert not {v["user_id"] for v in calibration} & {v["user_id"] for v in test}
        assert all(v["gold_class"] != "D-FA" for v in calibration)
        assert len(calibration) + len(test) + sum(v["gold_class"] == "D-FA" and v["user_id"] in groups
                                                  for v in views) == len(views)


def test_group_splits_need_four_groups():
    with pytest.raises(AssertionError, match="at least 4 groups"):
        aw.make_group_splits(["a", "b", "c"], 5, 0)


def test_one_user_per_task_and_no_exclusion_reproduces_run_conformal():
    n = 40
    users = [f"t{i:02d}" for i in range(n)]
    classes = [["CH", "CA", "D-LF", "D-FA"][i % 4] for i in range(n)]
    views = synthetic_views(n, 3, 1, users, classes)
    for score, rule, alpha, k in [("lift", "joint", 0.1, 1), ("lac", "markov", 0.2, 2), ("margin", "joint", 0.2, 2)]:
        old = run_conformal(views, score, rule, alpha, k, n_splits=30, seed=7)
        new = aw.run_grouped_conformal(views, aw.make_group_splits(users, 30, 7), score, rule, alpha, k, ())
        for metric, values in old["metrics"].items():
            for stat, value in values.items():
                assert (math.isnan(value) and math.isnan(new["metrics"][metric][stat])) or \
                    value == pytest.approx(new["metrics"][metric][stat]), (score, metric, stat)


def test_per_class_and_axis_tallies():
    tallies = {"per_class": {}, "per_axis": {}}
    test = [{"gold_class": "CH", "true_letter": "A"}, {"gold_class": "D-FA", "true_letter": "D"}]
    aw.class_and_axis_tallies(test, [["A"], ["B", "C"]], tallies)
    summary = aw.summarize_tallies(tallies)
    assert summary["per_class"]["CH"] == {"n": 1, "coverage": 1.0, "mean_set_size": 1.0}
    assert summary["per_class"]["D-FA"]["coverage"] == 0.0
    # D-FA is fit + adverse labs; {CA (low, adverse), D-LF (low, favourable)} misses fitness, covers metabolic
    assert summary["per_axis"]["fitness"]["coverage"] == 0.5 and summary["per_axis"]["metabolic"]["coverage"] == 1.0


def test_k_larger_than_agents_is_refused():
    users = ["a", "b", "c", "d"]
    views = synthetic_views(4, 1, 0, users, ["CH", "CA", "CH", "CA"])
    with pytest.raises(AssertionError, match="larger than"):
        aw.run_grouped_conformal(views, aw.make_group_splits(users, 2, 0), "lift", "joint", 0.2, 2)


def test_guideline_rule_on_fake_tasks(wqa_raw):
    tasks = wqa_prepare.build_tasks(wqa_raw)
    assert [aw.guideline_rule_class(t) for t in tasks] == [t["gold_class"] for t in tasks]
    assert aw.guideline_rule_report(tasks)["accuracy"] == 1.0


def test_full_report_on_a_fake_run(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader(lambda text: "A" if "Resting heart rate: mean 5" in text else "B")
    summary = wqa_exp.run_wqa_experiment(wqa_config, wqa_paths, loader_fn=loader, download=False)
    path = Path(summary["files"]["predictions"])
    tasks = wqa_prepare.load_tasks(wqa_paths["tasks"])
    report = aw.analyze_wqa(path, [0.2], [1, 2], ["lift", "lac"], ["joint"], n_splits=20, seed=0, tasks=tasks,
                            with_rf=True)
    assert report["conditions"]["fusion_no_ring"] == ["watch", "phone", "clinic_glycaemic", "clinic_lipid"]
    assert report["conditions"]["fusion_with_ring"][2] == "ring"
    assert "single:centralised" in report["conformal"]
    assert len(report["conformal"]["single:watch"]) == 2  # k=2 skipped for one agent
    assert len(report["conformal"]["fusion_no_ring"]) == 4
    watch = report["agents"]["watch"]
    assert watch["agent_axis"] == "fitness" and "fitness_axis_mass" in watch and "metabolic_axis_mass" not in watch
    assert set(report["agents"]["centralised"]) >= {"fitness_axis_mass", "metabolic_axis_mass"}
    assert report["baselines"]["guideline_rule"]["accuracy"] == 1.0
    assert 0.0 <= report["baselines"]["random_forest"]["accuracy_argmax"] <= 1.0
    for result in report["conformal"]["fusion_no_ring"]:
        assert "D-FA" not in result["exclude_from_calibration"] or result["n_calibration_mean"] > 0
        assert set(result["per_axis"]) == {"fitness", "metabolic"}
        assert 0.0 <= result["coverage_calibrated_classes"] <= 1.0
    text = aw.format_wqa_report(report, "lift", "joint")
    assert "fusion_no_ring" in text and "guideline rule accuracy 1.000" in text
    saved = aw.save_wqa_report(report)
    assert json.loads(saved.read_text(encoding="utf-8"))["num_tasks"] == 8
    figures = aw.plot_wqa_report(report, "lift", "joint")
    assert all(f.exists() for f in figures) and len(figures) == 3


def test_cli_end_to_end(wqa_config, wqa_paths, make_loader, capsys):
    loader, _ = make_loader()
    summary = wqa_exp.run_wqa_experiment(wqa_config, wqa_paths, loader_fn=loader, download=False)
    reports = aw.main(["--predictions", summary["files"]["predictions"], "--tasks", str(wqa_paths["tasks"]),
                       "--alpha", "0.2", "--splits", "10", "--no_plots"])
    assert len(reports) == 1 and "Saved" in capsys.readouterr().out


def test_original_run_is_analysed_as_one_agent(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader(lambda text: "A")
    config = dict(wqa_config, prompts=dict(wqa_config["prompts"], variant="original"))
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, loader_fn=loader, download=False)
    report = aw.analyze_wqa(Path(summary["files"]["predictions"]), [0.2], [1], ["lac"], ["joint"], n_splits=10)
    assert list(report["conditions"]) == ["single:original"]
    assert report["agents"]["original"]["agent_axis"] == "both"
