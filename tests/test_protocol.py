"""C4a protocol simulator: unit tests (layer 1), properties (layer 2), synthetic coverage (layer 3), regression and CLI."""

import json
import math
import random

import pytest

import simulate_protocol as sim_cli
from analysis import conformal as cf
from analysis import protocol as pr
from analysis import scores as sc


def logs(probs: dict[str, float]) -> dict[str, float]:
    return {k: math.log(v) for k, v in probs.items()}


def agent_from(prior: dict[str, float], posterior: dict[str, float]) -> dict[str, any]:
    p, q = sc.normalize(logs(prior)), sc.normalize(logs(posterior))
    return {"prior": p, "posterior": q, "lift": sc.lift_of(p, q)}


def agent_with_lift(lift: dict[str, float]) -> dict[str, any]:
    """Agent with a uniform prior whose posterior has the given lift (up to normalisation)."""
    prior = {y: 1.0 / len(lift) for y in lift}
    weights = {y: math.exp(v) for y, v in lift.items()}
    total = sum(weights.values())
    return agent_from(prior, {y: w / total for y, w in weights.items()})


def decoy_agent():
    """Shared facts favour the decoy B; the agent's fact rules B out. Truth is A."""
    return agent_from({"A": 0.05, "B": 0.90, "C": 0.05}, {"A": 0.80, "B": 0.05, "C": 0.15})


def view_of(agents, true="A", letters=("A", "B", "C"), task=0):
    return {"task_index": task, "true_letter": true, "letters": list(letters), "agents": agents}


def random_views(n_tasks, n_agents=(1, 5), n_options=(3, 4), signal=1.0, noise=0.8, seed=0):
    """Exchangeable synthetic tasks: each agent's posterior favours the truth by `signal`, plus noise."""
    rng = random.Random(seed)
    views = []
    for task in range(n_tasks):
        letters = "ABCD"[:rng.randint(*n_options)]
        true = rng.choice(letters)
        prior_w = {y: math.exp(rng.gauss(0, 1)) for y in letters}
        prior = {y: w / sum(prior_w.values()) for y, w in prior_w.items()}
        agents = []
        for _ in range(rng.randint(*n_agents)):
            post_w = {y: prior[y] * math.exp((signal if y == true else 0.0) + rng.gauss(0, noise)) for y in letters}
            agents.append(agent_from(prior, {y: w / sum(post_w.values()) for y, w in post_w.items()}))
        views.append(view_of(agents, true, letters, task))
    return views


# ---------- Layer 1: unit tests on hand-built tasks ----------

def test_restricted_logprobs_sum_to_one():
    agent = decoy_agent()
    restricted = pr.restrict_logprobs(agent["posterior"], ["A", "C"])
    assert sorted(restricted) == ["A", "C"]
    assert abs(sum(math.exp(v) for v in restricted.values()) - 1.0) < 1e-12


def test_lift_over_s_is_lift_minus_one_constant_and_unchanged_on_the_full_set():
    agent = decoy_agent()
    full = pr.restricted_lift(agent, ["A", "B", "C"])
    assert full == {y: agent["lift"][y] for y in "ABC"}
    restricted = pr.restricted_lift(agent, ["A", "C"])
    shifts = {y: agent["lift"][y] - restricted[y] for y in restricted}
    assert abs(shifts["A"] - shifts["C"]) < 1e-12
    assert abs(shifts["A"] - math.log(0.95 / 0.10)) < 1e-9


def test_worked_decoy_example_values():
    restricted = pr.restricted_lift(decoy_agent(), ["A", "C"])
    assert abs(restricted["A"] - math.log(0.80 / 0.95 / 0.5)) < 1e-9
    assert abs(restricted["C"] - math.log(0.15 / 0.95 / 0.5)) < 1e-9
    assert restricted["C"] < 0 < decoy_agent()["lift"]["C"]


def test_margin_is_shift_invariant_and_zero_for_the_top_option():
    agent = decoy_agent()
    for S in (["A", "B", "C"], ["A", "C"], ["B", "C"]):
        margin = pr.restricted_scores(agent, S, "margin")
        lift = pr.restricted_lift(agent, S)
        assert min(margin.values()) == 0.0
        best = max(lift.values())
        assert all(abs(margin[y] - (best - lift[y])) < 1e-12 for y in S)
    assert abs(pr.restricted_scores(agent, ["A", "C"], "margin")["C"]
               - pr.restricted_scores(agent, ["A", "B", "C"], "margin")["C"]) < 1e-12


def test_decoy_protocol_lift_needs_a_second_round_margin_does_not():
    view = view_of([decoy_agent()])
    lift_run = pr.run_protocol(view, "lift", [0.3, 0.3])
    assert lift_run["sets"] == [["A", "C"], ["A"]] and lift_run["stop"] == "decided"
    margin_run = pr.run_protocol(view, "margin", [1.0, 1.0])
    assert margin_run["sets"] == [["A"]] and margin_run["stop"] == "decided" and margin_run["rounds_used"] == 1


def test_stop_rules():
    a_top = agent_with_lift({"A": 2.0, "B": 0.0, "C": 0.0})
    b_top = agent_with_lift({"A": 0.0, "B": 2.0, "C": 0.0})
    assert pr.run_protocol(view_of([a_top, b_top]), "margin", [0.5])["stop"] == "empty"
    assert pr.run_protocol(view_of([a_top]), "margin", [0.5])["stop"] == "decided"
    unchanged = pr.run_protocol(view_of([a_top]), "margin", [5.0, 5.0, 5.0])
    assert unchanged["stop"] == "no_change" and unchanged["rounds_used"] == 1
    ladder = view_of([agent_with_lift({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0})], letters="ABCD")
    capped = pr.run_protocol(ladder, "margin", [2.5, 1.5])
    assert capped["sets"] == [["A", "B", "C"], ["A", "B"]] and capped["stop"] == "max_rounds"
    tighter = pr.run_protocol(ladder, "margin", [5.0, 1.5])
    assert tighter["stop"] != "no_change" and tighter["sets"][-1] == ["A", "B"]


def test_bits_are_counted_per_round():
    ladder = view_of([agent_with_lift({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0})], letters="ABCD")
    run = pr.run_protocol(ladder, "margin", [2.5, 1.5, 0.5])
    assert run["sets"] == [["A", "B", "C"], ["A", "B"], ["A"]]
    assert run["bits_up"] == 4 + 3 + 2 and run["bits_down"] == 3 * 4


def test_protocol_never_reads_the_true_answer():
    view = view_of([decoy_agent(), agent_with_lift({"A": 1.0, "B": 0.0, "C": 0.5})])
    del view["true_letter"]
    assert pr.run_protocol(view, "margin", [0.8, 0.4])["final_set"]


def test_calibration_and_test_tasks_and_folds_are_disjoint():
    for fraction in (0.5, 0.7):
        cal, test = pr.split_tasks(65, fraction, random.Random(1))
        assert not set(cal) & set(test) and len(cal) + len(test) == 65
    folds = pr.make_folds(list(range(22)), 3)
    assert [len(f) for f in folds] == [8, 7, 7]
    assert sorted(x for f in folds for x in f) == list(range(22))


def test_split_alpha_sums_to_alpha():
    assert pr.split_alpha(0.1, 0) == [0.1]
    assert all(abs(sum(pr.split_alpha(0.2, r, share)) - 0.2) < 1e-12 for r in (1, 2, 3) for share in (None, 0.7))
    assert pr.split_alpha(0.2, 1, 0.7) == pytest.approx([0.14, 0.06])


def test_lift_beyond_round_zero_is_refused():
    with pytest.raises(AssertionError, match="no coverage argument"):
        pr.run_multiround(random_views(20), "lift", "global", 0.1, 1, n_splits=2)


# ---------- Layer 2: properties on random tasks ----------

def test_sets_only_shrink():
    for view in random_views(200, seed=1):
        thresholds = sorted((random.Random(view["task_index"]).uniform(0, 2) for _ in range(4)), reverse=True)
        run = pr.run_protocol(view, "margin", thresholds)
        entering = view["letters"]
        for S in run["sets"]:
            assert set(S) <= set(entering)
            entering = S


def test_margin_with_one_threshold_never_changes_after_round_zero():
    rng = random.Random(2)
    for view in random_views(300, seed=2):
        tau = rng.uniform(0, 2.5)
        one = pr.run_protocol(view, "margin", [tau])
        many = pr.run_protocol(view, "margin", [tau] * 4, stop_when_unchanged=False)
        assert many["final_set"] == one["final_set"]


def test_spend_with_no_extra_round_equals_global():
    views = random_views(60, seed=3)
    assert pr.calibrate(views, "margin", "spend", 0.1, 0) == pr.calibrate(views, "margin", "global", 0.1, 0)


def test_relabelling_options_permutes_the_sets():
    mapping = {"A": "C", "B": "A", "C": "D", "D": "B"}
    for view in random_views(100, seed=4):
        renamed_agents = [{key: {mapping[y]: v for y, v in agent[key].items()} for key in ("prior", "posterior", "lift")}
                          for agent in view["agents"]]
        renamed = view_of(renamed_agents, mapping[view["true_letter"]], sorted(mapping[y] for y in view["letters"]))
        original = pr.run_protocol(view, "margin", [1.2, 0.6])
        relabelled = pr.run_protocol(renamed, "margin", [1.2, 0.6])
        assert sorted(mapping[y] for y in original["final_set"]) == sorted(relabelled["final_set"])


@pytest.mark.parametrize("score", ["margin", "lift"])
def test_round_zero_matches_run_conformal_exactly(score):
    views = random_views(65, n_agents=(3, 4), seed=5)
    one_round = cf.run_conformal(views, score, "joint", 0.1, 1, n_splits=40, seed=7)["metrics"]
    protocol = pr.run_multiround(views, score, "global", 0.1, 0, n_splits=40, seed=7)["metrics"]
    for name, value in one_round.items():
        if not math.isnan(value["mean"]):
            assert protocol[name] == value


# ---------- Layer 3: coverage on synthetic data with a known truth ----------

@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_global_coverage_is_on_target(alpha):
    views = random_views(1000, seed=6)
    for rounds in (0, 2):
        m = pr.run_multiround(views, "margin", "global", alpha, rounds, n_splits=20, seed=0)["metrics"]
        n_cal = 500
        assert 1 - alpha - 0.01 <= m["coverage"]["mean"] <= 1 - alpha + 1 / (n_cal + 1) + 0.02


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("share", [None, 0.7])
def test_spend_coverage_and_per_round_budget(alpha, share):
    views = random_views(1000, seed=7)
    result = pr.run_multiround(views, "margin", "spend", alpha, 1, cal_fraction=0.5, round0_share=share,
                               n_splits=20, seed=0)
    m, alphas = result["metrics"], result["alphas"]
    assert m["coverage"]["mean"] >= 1 - alpha - 0.01
    for r, alpha_r in enumerate(alphas):
        assert m[f"missed_round_{r}"]["mean"] <= alpha_r + 0.02


def test_null_agents_keep_coverage_and_rarely_decide():
    views = random_views(600, signal=0.0, seed=8)
    m = pr.run_multiround(views, "margin", "spend", 0.1, 1, n_splits=20)["metrics"]
    assert m["coverage"]["mean"] >= 0.88
    assert m["singleton_rate"]["mean"] < 0.25


def test_oracle_agents_decide_almost_everything_correctly():
    views = random_views(400, signal=6.0, noise=0.3, seed=9)
    m = pr.run_multiround(views, "margin", "global", 0.1, 1, n_splits=20)["metrics"]
    assert m["singleton_rate"]["mean"] > 0.85 and m["singleton_accuracy"]["mean"] == 1.0


# ---------- CLI ----------

def fake_predictions(tmp_path, n_tasks=80):
    rng = random.Random(10)
    records = []
    for task in range(n_tasks):
        true = rng.choice("ABC")
        prior = {"A": 0.6, "B": 0.2, "C": 0.2}
        for agent in range(3):
            post_w = {y: prior[y] * math.exp((1.5 if y == true else 0.0) + rng.gauss(0, 0.7)) for y in "ABC"}
            post = {y: w / sum(post_w.values()) for y, w in post_w.items()}
            for rotation in range(2):
                records.append({"task_index": task, "agent_index": agent, "rotation": rotation, "true_letter": true,
                                "predicted_letter": "A", "correct": true == "A",
                                "option_logprobs_prior": logs(prior), "option_logprobs_posterior": logs(post)})
    path = tmp_path / "predictions_answer_only_k1_rotcyclic_N80.json"
    path.write_text(json.dumps(records))
    return path


def test_shuffle_agents_keeps_options_and_moves_evidence():
    views = random_views(60, seed=11)
    shuffled = pr.shuffle_agents(views, seed=0)
    assert len(shuffled) == len(views)
    moved = 0
    for original, new in zip(views, shuffled):
        assert new["true_letter"] == original["true_letter"] and new["letters"] == original["letters"]
        if new["agents"] is not original["agents"]:
            moved += 1
        assert all(sorted(a["lift"]) == original["letters"] for a in new["agents"])
    assert moved >= 0.8 * len(views)  # most same-option tasks get someone else's agents


def test_shuffled_control_keeps_coverage_but_decides_less():
    views = random_views(500, signal=1.4, seed=12)
    real = pr.run_multiround(views, "margin", "global", 0.1, 1, n_splits=20, seed=0)["metrics"]
    shuffled = pr.run_multiround(views, "margin", "global", 0.1, 1, n_splits=20, seed=0, shuffle=True)["metrics"]
    assert shuffled["coverage"]["mean"] >= 0.88
    assert shuffled["singleton_rate"]["mean"] < real["singleton_rate"]["mean"]


def test_cli_runs_saves_and_traces(tmp_path, capsys):
    path = fake_predictions(tmp_path)
    reports = sim_cli.main(["--predictions", str(path), "--rounds", "0", "1", "--alpha", "0.2",
                            "--cal_fraction", "0.5", "--splits", "5", "--trace_tasks", "2"])
    printed = capsys.readouterr().out
    configs = [(r["score"], r["schedule"], r["rounds"], r["round0_share"]) for r in reports[0]["results"]]
    assert ("lift", "global", 0, None) in configs and ("lift", "global", 1, None) not in configs
    assert ("margin", "spend", 1, 0.7) in configs and ("margin", "spend", 0, None) not in configs
    assert "Traces:" in printed and "round 0" in printed
    saved = tmp_path / "protocol_A_answer_only_k1_rotcyclic_N80.json"
    assert saved.is_file() and json.loads(saved.read_text())["num_tasks"] == 80


def test_cli_trace_prefers_a_config_whose_later_round_can_act():
    args = sim_cli.parse_args(["--predictions", "x", "--rounds", "0", "1"])
    results = [{"score": s, "schedule": sc, "rounds": r, "shuffle": sh}
               for s in ("lift", "margin") for sc in ("global", "spend") for r in (0, 1) for sh in (False, True)]
    chosen = sim_cli.choose_trace_config(results)
    assert (chosen["score"], chosen["schedule"], chosen["rounds"], chosen["shuffle"]) == ("margin", "spend", 1, False)


def test_cli_infeasible_spend_is_dropped_unless_kept():
    base = ["--predictions", "x", "--score", "margin", "--schedule", "spend", "--rounds", "3",
            "--alpha", "0.1", "--cal_fraction", "0.5", "--round0_share"]
    with pytest.raises(AssertionError, match="no runnable configuration"):
        sim_cli.build_configs(sim_cli.parse_args(base), num_tasks=65)
    kept = sim_cli.build_configs(sim_cli.parse_args(base + ["--keep_infeasible"]), num_tasks=65)
    assert kept and all(c["schedule"] == "spend" for c in kept)


def test_cli_shuffle_control_adds_shuffled_rows(tmp_path):
    path = fake_predictions(tmp_path)
    reports = sim_cli.main(["--predictions", str(path), "--score", "margin", "--schedule", "global",
                            "--rounds", "0", "--alpha", "0.2", "--cal_fraction", "0.5", "--splits", "5",
                            "--shuffle_control", "--no_save"])
    flags = sorted(r["shuffle"] for r in reports[0]["results"])
    assert flags == [False, True]
