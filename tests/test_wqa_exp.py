"""End-to-end WearableQA runs with the fake backend: records, priors, files, variants, config checks."""

import copy
import json
from pathlib import Path

import pytest
import yaml

import wqa_exp
import wqa_prepare
from agent_exp import run_experiment
from utility import model_utils as mu
from utility import wqa_split as ws


@pytest.fixture
def hiddenbench_record_fields(config_copy, make_paths, make_loader):
    """The field names of a HiddenBench record written by agent_exp.make_record (fake run)."""
    loader, _ = make_loader()
    summary = run_experiment(config_copy(), make_paths(), num_samples=1, loader_fn=loader, download=False)
    mu.get_model_registry().clear()  # the WearableQA run in the same test loads its own fake model
    return set(read_predictions(summary)[0])


def scripted_reply(text: str) -> str:
    """Watch says A when its resting HR is low; clinics say B on high glucose; everything else C."""
    if "Resting heart rate: mean 5" in text:
        return "A"
    if "Fasting glucose: 120" in text:
        return "B"
    return "C"


def read_predictions(summary):
    with open(summary["files"]["predictions"], encoding="utf-8") as handle:
        return json.load(handle)


def test_device_run_records_and_files(wqa_config, wqa_paths, make_loader, hiddenbench_record_fields):
    loader, tracker = make_loader(scripted_reply)
    summary = wqa_exp.run_wqa_experiment(wqa_config, wqa_paths, loader_fn=loader, download=False)
    records = read_predictions(summary)
    agents = ["watch", "phone", "ring", "clinic_glycaemic", "clinic_lipid", "centralised"]
    assert len(records) == 8 * len(agents)
    assert summary["agents"] == sorted(agents)
    backend = tracker["backends"][0]
    assert sum(c["kind"] == "score" for c in backend.calls) == len(agents)  # one prior per agent
    assert sum(c["kind"] == "generate" for c in backend.calls) == len(records)
    added = {"dataset", "user_id", "gold_class", "condition", "prompt_variant", "template_version", "agent_devices",
             "agent_axis"}
    for record in records:
        assert set(record) - added == hiddenbench_record_fields
        assert set(record["option_logprobs_prior"]) == {"A", "B", "C", "D"}
        assert record["fact_indices"] == [] and record["rotation"] == 0
    watch = [r for r in records if r["agent_id"] == "watch"]
    assert len({json.dumps(r["option_logprobs_prior"]) for r in watch}) == 1  # same prior on every task
    gold = {r["task_id"]: r["true_letter"] for r in records}
    tasks = wqa_prepare.load_tasks(wqa_paths["tasks"])
    assert gold == {t["task_id"]: t["gold_letter"] for t in tasks}
    first_watch = watch[0]
    assert first_watch["predicted_letter"] == "A" and first_watch["predicted_answer"] == "CH"
    assert first_watch["agent_axis"] == "fitness"
    assert {r["agent_axis"] for r in records if r["agent_id"] == "centralised"} == {"both"}
    files = summary["files"]
    for key in ("predictions", "summary", "priors", "config", "prompts"):
        assert Path(files[key]).exists()
    assert sorted(p.stem for p in Path(files["prompts"]).glob("*.yaml")) == sorted(
        ["shared", "watch", "phone", "ring", "clinic_glycaemic", "clinic_lipid"])
    assert "wqa_fake__causal-model_strict_key_full_run" in files["predictions"]
    priors = json.loads(Path(files["priors"]).read_text(encoding="utf-8"))
    assert set(priors) == set(agents)
    assert Path(summary["trace_file"]).read_text(encoding="utf-8").strip()


def test_records_feed_the_existing_analysis(wqa_config, wqa_paths, make_loader):
    from analysis.scores import build_task_views, load_predictions

    loader, _ = make_loader(scripted_reply)
    summary = wqa_exp.run_wqa_experiment(wqa_config, wqa_paths, loader_fn=loader, download=False)
    records = load_predictions(Path(summary["files"]["predictions"]))
    views = build_task_views([r for r in records if r["role"] == "device"])
    assert len(views) == 8 and all(len(v["agents"]) == 5 for v in views)


def test_num_samples_and_ring_off(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader()
    config = copy.deepcopy(wqa_config)
    config["agents"]["devices"] = ["watch", "clinic_glycaemic"]
    config["agents"]["centralised"] = False
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, num_samples=3, loader_fn=loader, download=False)
    records = read_predictions(summary)
    assert len(records) == 6 and {r["task_index"] for r in records} == {0, 1, 2}
    assert "first3_run" in summary["files"]["predictions"]


def test_original_variant_restricts_to_core_options(wqa_config, wqa_paths, make_loader, wqa_raw):
    questions = ws.select_questions(wqa_raw)
    gold_original = {q["id"]: q["gt_letter"] for q in questions}
    loader, tracker = make_loader(lambda text: "A")
    config = copy.deepcopy(wqa_config)
    config["prompts"]["variant"] = "original"
    summary = wqa_exp.run_wqa_experiment(config, wqa_paths, loader_fn=loader, download=False)
    records = read_predictions(summary)
    assert len(records) == 8 and {r["agent_id"] for r in records} == {"original"}
    backend = tracker["backends"][0]
    assert sum(c["kind"] == "score" for c in backend.calls) == 8  # options differ per question: prior per task
    for record, question in zip(records, questions):
        assert set(record["option_logprobs_posterior"]) == {"A", "B", "C", "D"}
        assert len(record["option_logprobs_posterior_full"]) == 10
        letter_map = record["original_letter_map"]
        assert letter_map[gold_original[record["task_id"]]] == record["true_letter"]
        assert record["predicted_letter_original"] == "A" and record["parse_ok"]
        assert record["predicted_letter"] == letter_map.get("A")  # None when A is not a gold-capable option
        for original, fixed in letter_map.items():
            assert record["option_logprobs_posterior"][fixed] == record["option_logprobs_posterior_full"][original]
    assert "prompts" not in summary["files"]


def test_nokey_and_daily_variants_run(wqa_config, wqa_paths, make_loader):
    loader, tracker = make_loader()
    for variant in ("no_key", "daily_rows"):
        config = copy.deepcopy(wqa_config)
        config["prompts"]["variant"] = variant
        summary = wqa_exp.run_wqa_experiment(config, wqa_paths, num_samples=2, loader_fn=loader, download=False)
        records = read_predictions(summary)
        assert {r["prompt_variant"] for r in records} == {variant}
    texts = [c["user"] for c in tracker["backends"][0].calls]
    assert any("2024-01-31: rhr=" in t for t in texts)
    assert any("CLINICAL KEY" not in t and "Resting heart rate: mean" in t for t in texts)


def test_config_validation(wqa_config):
    wqa_exp.validate_wqa_config(wqa_config)
    for mutate, message in [
        (lambda c: c["experiment"].update(dataset="hiddenbench"), "wearableqa_phenotype"),
        (lambda c: c["agents"].update(devices=["watch", "watch"]), "distinct"),
        (lambda c: c["prompts"].update(variant="softened"), "variant"),
        (lambda c: c["prompts"].update(variant="no_key", nokey_dir=None), "nokey_dir"),
        (lambda c: c["agents"].update(overrides={0: "fake2"}), "overrides"),
    ]:
        config = copy.deepcopy(wqa_config)
        mutate(config)
        with pytest.raises(AssertionError, match=message):
            wqa_exp.validate_wqa_config(config)


def test_unknown_device_has_no_template(wqa_config, wqa_paths, make_loader):
    loader, _ = make_loader()
    config = copy.deepcopy(wqa_config)
    config["agents"]["devices"] = ["watch", "fridge"]
    with pytest.raises(AssertionError, match="no template"):
        wqa_exp.run_wqa_experiment(config, wqa_paths, loader_fn=loader, download=False)


def test_cli_overrides_select_listed_model(wqa_config, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(wqa_config), encoding="utf-8")
    config = wqa_exp.load_wqa_config(path)
    args = wqa_exp.parse_args(["--model", "fake/second-model", "--variant", "daily_rows", "--device", "cpu"])
    config = wqa_exp.apply_wqa_overrides(config, args)
    assert config["agents"]["default_model"] == "fake2" and config["prompts"]["variant"] == "daily_rows"
    args = wqa_exp.parse_args(["--model", "other/model"])
    config = wqa_exp.apply_wqa_overrides(config, args)
    assert config["models"]["fake2"]["hf_name"] == "other/model"


def test_repository_config_is_valid():
    path = Path(__file__).resolve().parent.parent / "config" / "wearableqa_bluebear.yaml"
    config = wqa_exp.load_wqa_config(path)
    assert config["agents"]["devices"] == list(ws.default_holder_table())


def test_stale_tasks_file_is_refused(tmp_path, wqa_raw):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(wqa_raw), encoding="utf-8")
    tasks_path = tmp_path / "tasks.jsonl"
    assert len(wqa_prepare.ensure_tasks(tasks_path, raw, 28)) == 8
    with pytest.raises(AssertionError, match="window_days"):
        wqa_prepare.ensure_tasks(tasks_path, raw, 14)


def test_prepare_roundtrip(tmp_path, wqa_raw):
    tasks = wqa_prepare.build_tasks(wqa_raw)
    path = wqa_prepare.write_tasks(tmp_path / "t.jsonl", tasks)
    loaded = wqa_prepare.load_tasks(path)
    assert [t["gold_letter"] for t in loaded] == ["A", "A", "B", "C", "D", "B", "A", "C"]
    assert loaded[0]["user_id"] == "u1" and loaded[0]["original_gt_letter"] == wqa_raw["mcqs"][0]["gt_letter"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(loaded[0], schema_version=99)) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="schema versions"):
        wqa_prepare.load_tasks(bad)
