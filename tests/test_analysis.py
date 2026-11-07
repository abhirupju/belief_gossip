"""Analysis: decision rules, rotation averaging, statistics, conformal sets and the report (synthetic records)."""

import json
import math
import random

import pytest

import analyze as analyze_cli
from analysis import conformal as cf
from analysis import scores as sc
from analysis import stats as st
from analysis.report import analyze, save_report


def logs(probs: dict[str, float]) -> dict[str, float]:
    return {k: math.log(v) for k, v in probs.items()}


def record(task, agent, true, prior, posterior, predicted=None, rotation=0):
    return {"task_index": task, "agent_index": agent, "rotation": rotation, "true_letter": true,
            "predicted_letter": predicted, "correct": predicted == true,
            "option_logprobs_prior": logs(prior), "option_logprobs_posterior": logs(posterior)}


def worked_example_records():
    """Prior favours A (wrong); agent 0's fact rules out A, agent 1's fact rules out B; truth is C."""
    prior = {"A": 0.6, "B": 0.2, "C": 0.2}
    return [record(0, 0, "C", prior, {"A": 0.10, "B": 0.45, "C": 0.45}, "B"),
            record(0, 1, "C", prior, {"A": 0.50, "B": 0.05, "C": 0.45}, "A")]


def test_rules_on_worked_example():
    view = sc.build_task_views(worked_example_records())[0]
    assert sc.rule_prior(view) == "A"
    assert sc.rule_anti_prior(view) in ("B", "C")
    assert sc.rule_prior_plus_lift(view) == "C"
    assert sc.rule_lift_sum(view) == "C"
    assert sc.rule_generated_majority(view) is None
    fused = sc.normalize(sc.add_scores(sc.shared_prior(view), *[a["lift"] for a in view["agents"]]))
    assert math.exp(fused["C"]) == pytest.approx(0.84, abs=0.01)


def test_posterior_sum_counts_the_prior_once_per_agent():
    """Weak evidence for C from 3 agents: prior+lift recovers C, summed posteriors stay on the prior's A."""
    prior = {"A": 0.7, "B": 0.15, "C": 0.15}
    posterior = {"A": 0.55, "B": 0.15, "C": 0.30}
    view = sc.build_task_views([record(0, i, "C", prior, posterior) for i in range(3)])[0]
    assert sc.rule_prior_plus_lift(view) == "C"
    assert sc.rule_posterior_sum(view) == "A"


def test_rotation_scores_are_averaged_per_agent():
    prior = {"A": 0.5, "B": 0.5}
    recs = [record(0, 0, "A", prior, {"A": 0.9, "B": 0.1}, "A", 0), record(0, 0, "A", prior, {"A": 0.3, "B": 0.7}, "B", 1)]
    views = sc.build_task_views(recs)
    assert len(views) == 1 and len(views[0]["agents"]) == 1
    agent = views[0]["agents"][0]
    assert agent["generated"] == ["A", "B"]
    expected = sc.normalize({"A": (math.log(0.9) + math.log(0.3)) / 2, "B": (math.log(0.1) + math.log(0.7)) / 2})
    assert agent["posterior"] == pytest.approx(expected)
    assert len(sc.build_answer_views(recs)) == 2


def test_baselines():
    views = sc.build_task_views(worked_example_records() + [record(1, 0, "C", {"A": .5, "B": .5, "C": 1e-9 + 0.0},
                                                                   {"A": .2, "B": .2, "C": .6})])
    assert sc.chance_accuracy(views) == pytest.approx(1 / 3)
    key = sc.answer_key_majority_accuracy(views)
    assert key["letter"] == "C" and key["accuracy"] == 1.0


def test_load_predictions_rejects_reasoning_runs(tmp_path):
    path = tmp_path / "predictions_x.json"
    rec = worked_example_records()[0]
    rec["option_logprobs_prior"] = None
    path.write_text(json.dumps([rec]))
    with pytest.raises(AssertionError, match="reasoning-mode"):
        sc.load_predictions(path)


def test_sign_test_and_bootstrap():
    assert st.sign_test(16, 6) == pytest.approx(0.0525, abs=0.001)
    assert st.sign_test(0, 0) == 1.0
    low, high = st.bootstrap_ci([1.0] * 30 + [0.0] * 30, n_boot=2000, seed=1)
    assert low < 0.5 < high
    assert st.paired_comparison([True, True, False], [False, True, True]) == {"only_a": 1, "only_b": 1,
                                                                             "p_value": 1.0}


def test_conformal_quantile_edges():
    assert cf.conformal_quantile(list(range(1, 10)), 0.1) == 9
    assert cf.conformal_quantile([1, 2, 3, 4, 5], 0.1) == math.inf
    with pytest.raises(AssertionError, match="alpha"):
        cf.conformal_quantile([1, 2], 0.0)


def test_quorum_sets_grow_with_k():
    view = sc.build_task_views(worked_example_records())[0]
    thresholds = [0.5, 0.5]
    k1 = cf.quorum_set(view, cf.score_lift, thresholds, 1)
    k2 = cf.quorum_set(view, cf.score_lift, thresholds, 2)
    assert set(k1) <= set(k2)
    assert k1 == ["C"] and k2 == ["A", "B", "C"]


def synthetic_views(n_tasks=80, n_agents=3, seed=0):
    """Each agent's evidence pushes the truth up and one random wrong option down, with noise."""
    rng = random.Random(seed)
    recs = []
    for task in range(n_tasks):
        letters = ["A", "B", "C"]
        true = rng.choice(letters)
        prior = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        for agent in range(n_agents):
            weights = {y: math.exp((1.0 if y == true else 0.0) + rng.gauss(0, 0.8)) for y in letters}
            total = sum(weights.values())
            recs.append(record(task, agent, true, prior, {y: w / total for y, w in weights.items()}))
    return sc.build_task_views(recs)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("k", [1, 2])
def test_joint_rule_reaches_target_coverage(alpha, k):
    result = cf.run_conformal(synthetic_views(), "lift", "joint", alpha, k, n_splits=100, seed=0)
    assert result["metrics"]["coverage"]["mean"] >= 1 - alpha - 0.03


def test_larger_alpha_gives_smaller_sets():
    views = synthetic_views()
    tight = cf.run_conformal(views, "lift", "joint", 0.05, 1, n_splits=50)["metrics"]["mean_set_size"]["mean"]
    loose = cf.run_conformal(views, "lift", "joint", 0.3, 1, n_splits=50)["metrics"]["mean_set_size"]["mean"]
    assert loose < tight


def test_markov_rule_runs_and_covers():
    result = cf.run_conformal(synthetic_views(), "lac", "markov", 0.2, 1, n_splits=50)
    assert result["metrics"]["coverage"]["mean"] >= 0.75


def test_report_end_to_end(tmp_path):
    views_records = []
    rng = random.Random(3)
    for task in range(12):
        true = rng.choice(["A", "B", "C"])
        prior = {"A": 0.6, "B": 0.2, "C": 0.2}
        posterior = {y: (0.6 if y == true else 0.2) for y in "ABC"}
        for agent in range(2):
            for rotation in range(3):
                views_records.append(record(task, agent, true, prior, posterior, true if agent == 0 else "A", rotation))
    path = tmp_path / "predictions_answer_only_k1_rotcyclic_N12.json"
    path.write_text(json.dumps(views_records))
    report = analyze(path, [0.2], [1], ["lift"], ["joint"], n_splits=20, n_boot=200)
    assert report["num_tasks"] == 12 and report["rotations_per_agent"] == 3
    assert report["per_task"]["prior_plus_lift"]["accuracy"] == 1.0
    assert set(report["comparisons"]) >= {"lift_sum_vs_anti_prior", "prior_plus_lift_vs_anti_prior"}
    saved = save_report(report)
    assert saved.name == "analysis_answer_only_k1_rotcyclic_N12.json" and json.loads(saved.read_text())
    printed = analyze_cli.main(["--predictions", str(path), "--alpha", "0.2", "--k", "1", "--splits", "10",
                                "--bootstrap", "200", "--no_save"])
    assert printed[0]["num_records"] == 72
