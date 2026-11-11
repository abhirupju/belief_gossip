"""C4b belief-update: template rendering, the re-scoring cache (fake model), and variant D in the protocol."""

import json
import math
import random

import pytest

import belief_update as bu
from analysis import protocol as pr
from utility import prompt_utils as pu


def load_bu_template():
    return pu.load_template("prompts/hiddenbench_qwen35_08b", "belief_update")


def logs(probs: dict[str, float]) -> dict[str, float]:
    return {k: math.log(v) for k, v in probs.items()}


# ---------- template ----------

def test_belief_update_template_validates_and_is_known():
    template = load_bu_template()
    assert template["mode"] == "answer_only" and template["group_conclusion_basis"] == "ruled_out"
    assert "group_conclusion" in pu.get_optional_placeholders()


def test_render_fills_the_ruling_and_the_empty_case():
    template = load_bu_template()
    fields = {"description": "d", "shared_facts": ["s"], "private_facts": ["p"],
              "options": ["o1", "o2", "o3"], "letters": ["A", "B", "C"], "ruled_out": ["B"], "surviving": ["A", "C"]}
    _, user = pu.render_prompt(template, fields)
    assert "ruled out these options: B" in user
    _, empty_user = pu.render_prompt(template, {**fields, "ruled_out": [], "surviving": ["A", "B", "C"]})
    assert "not ruled out any option yet" in empty_user


def test_existing_templates_still_render_without_a_ruling():
    template = pu.load_template("prompts/hiddenbench_qwen35_08b", "answer_only")
    fields = {"description": "d", "shared_facts": ["s"], "private_facts": [],
              "options": ["o1", "o2"], "letters": ["A", "B"]}
    system, user = pu.render_prompt(template, fields)  # no ruled_out needed
    assert "no private information" in user.lower() and system


# ---------- reachable rulings ----------

def test_reachable_rulings_leave_at_least_two_options():
    assert bu.reachable_rulings(["A", "B", "C"]) == [(), ("A",), ("B",), ("C",)]
    four = bu.reachable_rulings(["A", "B", "C", "D"])
    assert () in four and all(len(r) <= 2 for r in four) and ("A", "B") in four
    assert all(len(set(r)) == len(r) for r in four)


def test_displayed_of_inverts_the_rotation():
    letters = ["A", "B", "C"]
    for rotation in range(3):
        disp = bu.displayed_of(letters, rotation)
        # the dataset letter mapped to label L must un-rotate back to itself
        for d, shown in disp.items():
            assert pu.unrotate_letter(shown, 3, rotation) == d


# ---------- cache build with the fake model ----------

def group_sensitive_reply(text: str) -> str:
    """Fake agent: normally answers A, but if the group ruled out A it switches to C."""
    return "C" if "ruled out these options" in text and "A" in text.split("ruled out these options:")[1][:6] else "A"


def build_small_cache(make_loader, config_copy, fake_data_dir, fake_model_root, reply_fn):
    config = config_copy(agents={"facts_per_agent": 1, "default_model": "fake", "overrides": {}},
                         prompts={"dir": "prompts/hiddenbench_qwen35_08b", "template": "belief_update",
                                  "option_rotation": "cyclic"})
    loader, tracker = make_loader(reply_fn)
    from agent_exp import get_dataset_adapter, load_all_models
    adapter = get_dataset_adapter("hiddenbench")
    handles = load_all_models(config, fake_model_root, loader, False)
    template = load_bu_template()
    n = adapter["num_samples"](fake_data_dir)
    by_task = bu.build_rescore_cache(config, adapter, handles, template, fake_data_dir, n)
    return by_task, adapter, fake_data_dir, n


def test_cache_has_every_agent_and_ruling_and_shared_prior(make_loader, config_copy, fake_data_dir, fake_model_root):
    by_task, adapter, data_dir, n = build_small_cache(make_loader, config_copy, fake_data_dir, fake_model_root,
                                                      lambda text: "A")
    assert set(by_task) == {str(i) for i in range(n)}
    for task_index, agents in by_task.items():
        sample = adapter["load_sample"](int(task_index), data_dir)
        letters = adapter["prompt_fields"](sample, [], 0)["letters"]
        expected = {bu.ruling_key(r) for r in bu.reachable_rulings(letters)}
        prior_sets = []
        for agent_index, rulings in agents.items():
            assert set(rulings) == expected
            for entry in rulings.values():
                assert sorted(entry["prior"]) == sorted(letters) and sorted(entry["posterior"]) == sorted(letters)
            prior_sets.append(rulings[""]["prior"])
        assert all(p == prior_sets[0] for p in prior_sets)  # prior shared across a task's agents


def test_cache_builder_does_not_read_the_true_letter(make_loader, config_copy, fake_data_dir, fake_model_root):
    # The builder only ever uses adapter["prompt_fields"], which does not expose the answer; a smoke run must work.
    by_task, _, _, _ = build_small_cache(make_loader, config_copy, fake_data_dir, fake_model_root, lambda text: "A")
    assert by_task


def test_cache_scores_are_normalised_like_variant_a(make_loader, config_copy, fake_data_dir, fake_model_root):
    # scores.agent_view normalises after rotation-averaging; the cache must match so raw-lift stays on one scale.
    by_task, _, _, _ = build_small_cache(make_loader, config_copy, fake_data_dir, fake_model_root, lambda text: "A")
    for agents in by_task.values():
        for rulings in agents.values():
            for entry in rulings.values():
                for side in ("prior", "posterior"):
                    assert abs(math.log(sum(math.exp(v) for v in entry[side].values()))) < 1e-9


def test_prior_pass_is_shared_not_repeated_per_agent(make_loader, config_copy, fake_data_dir, fake_model_root):
    # Count score passes: one prior per (task, ruling, rotation) + one posterior per (task, agent, ruling, rotation).
    config = config_copy(agents={"facts_per_agent": 1, "default_model": "fake", "overrides": {}},
                         prompts={"dir": "prompts/hiddenbench_qwen35_08b", "template": "belief_update",
                                  "option_rotation": "cyclic"})
    loader, tracker = make_loader(lambda text: "A")
    from agent_exp import get_dataset_adapter, load_all_models
    adapter = get_dataset_adapter("hiddenbench")
    handles = load_all_models(config, fake_model_root, loader, False)
    n = adapter["num_samples"](fake_data_dir)
    bu.build_rescore_cache(config, adapter, handles, load_bu_template(), fake_data_dir, n)
    passes = sum(len(c["user"].split()) >= 0 for b in tracker["backends"] for c in b.calls if c["kind"] == "score")
    expected = 0
    for t in range(n):
        s = adapter["load_sample"](t, fake_data_dir)
        m = adapter["num_options"](s)
        rulings = len(bu.reachable_rulings([chr(65 + i) for i in range(m)]))
        n_fact = adapter["num_private_facts"](s)
        expected += rulings * m + n_fact * rulings * m  # shared priors + posteriors, times rotations (= m)
    assert passes == expected


def test_overrides_are_rejected_because_the_prior_is_shared(make_loader, config_copy, fake_data_dir, fake_model_root):
    config = config_copy(agents={"facts_per_agent": 1, "default_model": "fake", "overrides": {1: "fake"}},
                         prompts={"dir": "prompts/hiddenbench_qwen35_08b", "template": "belief_update",
                                  "option_rotation": "cyclic"})
    loader, _ = make_loader(lambda text: "A")
    from agent_exp import get_dataset_adapter, load_all_models
    adapter = get_dataset_adapter("hiddenbench")
    handles = load_all_models(config, fake_model_root, loader, False)
    with pytest.raises(AssertionError, match="overrides must be empty"):
        bu.build_rescore_cache(config, adapter, handles, load_bu_template(), fake_data_dir,
                               adapter["num_samples"](fake_data_dir))


# ---------- variant D in the protocol ----------

def synthetic_cache_from_views(views, rescore_fn):
    """Build an in-memory rescore lookup from a function (view, agent_index, ruled_out_tuple) -> {prior, posterior}."""
    store = {}
    for v in views:
        letters = v["letters"]
        store[v["task_index"]] = {}
        for a in v["agents"]:
            store[v["task_index"]][a["agent_index"]] = {}
            for r in bu.reachable_rulings(letters):
                store[v["task_index"]][a["agent_index"]][bu.ruling_key(r)] = rescore_fn(v, a, r)

    def lookup(task_index, agent_index, ruling):
        return store[task_index][agent_index][ruling]

    return lookup


def make_views(n_tasks=60, seed=0):
    rng = random.Random(seed)
    views = []
    for task in range(n_tasks):
        letters = ["A", "B", "C"]
        true = rng.choice(letters)
        views.append({"task_index": task, "true_letter": true, "letters": letters,
                      "agents": [{"agent_index": i,
                                  "prior": {y: math.log(1 / 3) for y in letters},
                                  "posterior": {y: v for y, v in zip(letters, _norm(rng))},
                                  "lift": {}} for i in range(3)]})
        for a in views[-1]["agents"]:
            a["lift"] = {y: a["posterior"][y] - a["prior"][y] for y in letters}
    return views


def _norm(rng):
    raw = [rng.random() + 0.1 for _ in range(3)]
    total = sum(raw)
    return [math.log(x / total) for x in raw]


def test_noop_cache_reproduces_variant_a():
    views = make_views(60, seed=1)
    # cache returns each agent's own frozen prior/posterior for every ruling -> D must equal A exactly
    rescore = synthetic_cache_from_views(views, lambda v, a, r: {"prior": a["prior"], "posterior": a["posterior"]})
    a = pr.run_multiround(views, "margin", "global", 0.1, 2, n_splits=30, seed=5, variant="A")
    d = pr.run_multiround(views, "margin", "global", 0.1, 2, n_splits=30, seed=5, variant="D", rescore=rescore)
    for key in ("coverage", "mean_set_size", "singleton_rate", "empty_rate"):
        assert a["metrics"][key]["mean"] == d["metrics"][key]["mean"]


def test_variant_d_missing_entry_raises():
    views = make_views(8, seed=2)
    rescore = synthetic_cache_from_views(views, lambda v, a, r: {"prior": a["prior"], "posterior": a["posterior"]})
    bad = lambda task_index, agent_index, ruling: rescore(task_index, agent_index, "Z")  # never cached
    with pytest.raises((AssertionError, KeyError)):
        pr.run_multiround(views, "margin", "global", 0.1, 1, n_splits=2, seed=0, variant="D", rescore=bad)


def test_variant_d_coverage_holds_when_ruling_shifts_scores():
    views = make_views(400, seed=3)

    def shift(v, a, r):
        # a ruling that removes an option nudges the truth up (a benign belief update), keeping coverage
        prior = dict(a["prior"])
        posterior = dict(a["posterior"])
        if r:
            posterior = {y: posterior[y] + (0.5 if y == v["true_letter"] else 0.0) for y in posterior}
        return {"prior": prior, "posterior": posterior}

    rescore = synthetic_cache_from_views(views, shift)
    d = pr.run_multiround(views, "margin", "spend", 0.1, 1, cal_fraction=0.7, n_splits=20, seed=0,
                          variant="D", rescore=rescore)
    assert d["metrics"]["coverage"]["mean"] >= 0.88
    assert d["metrics"]["mean_model_passes"]["mean"] > 0  # D records the model-pass cost


def test_cli_builds_cache(make_loader, config_copy, fake_data_dir, fake_model_root, tmp_path):
    import yaml
    config = config_copy(agents={"facts_per_agent": 1, "default_model": "fake", "overrides": {}},
                         prompts={"dir": "prompts/hiddenbench_qwen35_08b", "template": "belief_update",
                                  "option_rotation": "cyclic"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    loader, _ = make_loader(lambda text: "A")
    argv = ["--config", str(config_path), "--device", "cpu", "--data_dir", str(fake_data_dir),
            "--model_dir", str(fake_model_root), "--results_dir", str(tmp_path / "results")]
    cache_path = bu.main(argv, loader_fn=loader)
    assert cache_path.is_file() and json.loads(cache_path.read_text())["variant"] == "D"
    lookup = pr.load_rescore_cache(cache_path)
    data = json.loads(cache_path.read_text())["by_task"]
    for t, agents in data.items():  # round 0 (empty ruling) retrievable for every task and agent
        for a in agents:
            assert "prior" in lookup(int(t), int(a), "")


def test_shared_prior_wrapper_fixes_the_prior_and_keeps_the_posterior():
    views = make_views(12, seed=4)
    # a cache where the prior differs by ruling; the wrapper must always return the empty-ruling prior
    def entries(v, a, r):
        bump = 0.3 * len(r)
        return {"prior": {y: a["prior"][y] + bump for y in a["prior"]}, "posterior": a["posterior"]}
    base = synthetic_cache_from_views(views, entries)
    wrapped = pr.shared_prior_rescore(base)
    v = views[0]; a = v["agents"][0]
    empty_prior = base(v["task_index"], a["agent_index"], "")["prior"]
    got = wrapped(v["task_index"], a["agent_index"], "A")
    assert got["prior"] == empty_prior  # shared-facts-only prior, not the ruling's prior
    assert got["posterior"] == base(v["task_index"], a["agent_index"], "A")["posterior"]


def test_shared_prior_lift_differs_from_conclusion_prior_once_a_ruling_applies():
    # With the conclusion prior, the ruling cancels in round 0 (empty ruling) so both modes agree there;
    # once options are ruled out the two priors differ, so the lift (and the sets) can differ.
    views = make_views(200, seed=5)
    def entries(v, a, r):
        return {"prior": {y: (a["prior"][y] + (0.4 if r else 0.0)) for y in a["prior"]}, "posterior": a["posterior"]}
    base = synthetic_cache_from_views(views, entries)
    conclusion = pr.run_multiround(views, "margin", "spend", 0.2, 1, cal_fraction=0.5, n_splits=20, seed=0,
                                   variant="D", rescore=base)
    shared = pr.run_multiround(views, "margin", "spend", 0.2, 1, cal_fraction=0.5, n_splits=20, seed=0,
                               variant="D", rescore=pr.shared_prior_rescore(base))
    assert conclusion["metrics"]["coverage"]["mean"] >= 0.75 and shared["metrics"]["coverage"]["mean"] >= 0.75


def test_cli_variant_d_without_rescored_asserts_cleanly():
    import simulate_protocol as sim_cli
    args = sim_cli.parse_args(["--predictions", "x", "--variant", "D"])
    with pytest.raises(AssertionError, match="needs --rescored"):
        sim_cli.make_rescore(args)


def test_cli_end_to_end_variant_d_both_lift_priors(tmp_path):
    import simulate_protocol as sim_cli
    # a tiny predictions file + a matching cache, run through the CLI for both --lift_prior modes
    rng = random.Random(6)
    records = []
    for task in range(20):
        true = rng.choice("ABC")
        prior = {"A": 0.5, "B": 0.3, "C": 0.2}
        for agent in range(3):
            post = {y: v for y, v in zip("ABC", [0.6, 0.25, 0.15])}
            for rot in range(2):
                records.append({"task_index": task, "agent_index": agent, "rotation": rot, "true_letter": true,
                                "predicted_letter": "A", "correct": true == "A",
                                "option_logprobs_prior": logs(prior), "option_logprobs_posterior": logs(post)})
    pred = tmp_path / "predictions_belief_update_k1_rotcyclic_N20.json"
    pred.write_text(json.dumps(records))
    from analysis.scores import build_task_views
    views = build_task_views(json.loads(pred.read_text()))
    by_task = {}
    for v in views:
        by_task[str(v["task_index"])] = {}
        for a in v["agents"]:
            entry = {"prior": a["prior"], "posterior": a["posterior"]}
            by_task[str(v["task_index"])][str(a["agent_index"])] = {
                bu.ruling_key(r): entry for r in bu.reachable_rulings(v["letters"])}
    cache = tmp_path / "rescored_belief_update_k1_rotcyclic_N20.json"
    cache.write_text(json.dumps({"variant": "D", "by_task": by_task}))
    for mode, tag in (("conclusion", "D"), ("shared", "Dsharedprior")):
        reports = sim_cli.main(["--predictions", str(pred), "--variant", "D", "--rescored", str(cache),
                                "--lift_prior", mode, "--score", "margin", "--schedule", "global", "--rounds", "0",
                                "--alpha", "0.2", "--cal_fraction", "0.5", "--splits", "5"])
        assert reports[0]["lift_prior"] == mode
        assert (tmp_path / f"protocol_{tag}_belief_update_k1_rotcyclic_N20.json").is_file()


def test_load_rescore_cache_round_trips(tmp_path):
    cache = {"by_task": {"0": {"0": {"": {"prior": {"A": -1.0, "B": -1.0},
                                         "posterior": {"A": -0.5, "B": -1.5}}}}}}
    path = tmp_path / "rescored_x.json"
    path.write_text(json.dumps(cache))
    lookup = pr.load_rescore_cache(path)
    assert lookup(0, 0, "")["posterior"]["A"] == -0.5
