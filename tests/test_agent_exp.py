"""End-to-end runs with the fake backend: accuracy, files, naming, rotations, reasoning, overrides."""

import json
import math

import pytest

from agent_exp import run_experiment
from utility.prompt_utils import load_template


def read_outputs(summary):
    files = summary["files"]
    with open(files["predictions"], encoding="utf-8") as handle:
        records = json.load(handle)
    return records, files


def test_full_profile_run(config_copy, make_paths, make_loader, real_prompt_dir):
    loader, tracker = make_loader(lambda text: "A")
    summary = run_experiment(config_copy(), make_paths(), loader_fn=loader, download=False)
    records, files = read_outputs(summary)
    assert len(records) == 3 and all(len(r["fact_indices"]) == len(r["private_facts"]) for r in records)
    assert [len(r["private_facts"]) for r in records] == [2, 3, 4]
    assert math.isclose(summary["accuracy"]["per_agent_accuracy"], 1 / 3)
    assert math.isclose(summary["majority"]["majority_accuracy"], 1 / 3)
    assert len(tracker["loads"]) == 1
    assert "hiddenbench_fake__causal-model_full_run" in files["predictions"]
    assert files["predictions"].endswith("predictions_answer_only_kall_rotnone_N3.json")
    template = load_template(real_prompt_dir, "answer_only")
    with open(files["prompt"], encoding="utf-8") as handle:
        assert handle.read() == template["raw_text"]
    assert summary["template_text"] == template["raw_text"]
    record = records[0]
    assert "prompt" not in record and "system" not in record
    assert record["round"] == 0 and record["role"] == "agent" and record["agent_id"] == "agent-0"
    assert record["reasoning"] == "" and record["schema_version"] == 1
    assert record["predicted_letter"] == "A" and record["true_letter"] == "A" and record["correct"]
    assert set(record["option_logprobs_prior"]) == {"A", "B", "C"}
    assert max(record["option_logprobs_posterior"], key=record["option_logprobs_posterior"].get) == "A"
    trace_file = summary["trace_file"]
    with open(trace_file, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert lines and all("resourceSpans" in line for line in lines)


def test_one_fact_per_agent(config_copy, make_paths, make_loader):
    loader, _ = make_loader(lambda text: "A")
    summary = run_experiment(config_copy(agents={"facts_per_agent": 1}), make_paths(), loader_fn=loader,
                             download=False)
    records, files = read_outputs(summary)
    assert len(records) == 9 and all(len(r["private_facts"]) == 1 for r in records)
    assert math.isclose(summary["accuracy"]["per_agent_accuracy"], 2 / 9)
    assert summary["accuracy"]["per_agent_index"]["0"] == pytest.approx(1 / 3)
    assert files["predictions"].endswith("predictions_answer_only_k1_rotnone_N3.json")


def test_num_samples_limits_tasks_and_names_folder(config_copy, make_paths, make_loader):
    loader, _ = make_loader()
    summary = run_experiment(config_copy(), make_paths(), num_samples=2, loader_fn=loader, download=False)
    records, files = read_outputs(summary)
    assert {r["task_index"] for r in records} == {0, 1}
    assert "hiddenbench_fake__causal-model_first2_run" in files["predictions"]
    assert files["predictions"].endswith("_N2.json")


@pytest.mark.parametrize("bad", [0, 4])
def test_num_samples_out_of_range(config_copy, make_paths, make_loader, bad):
    loader, _ = make_loader()
    with pytest.raises(AssertionError, match="between 1 and the dataset size 3"):
        run_experiment(config_copy(), make_paths(), num_samples=bad, loader_fn=loader, download=False)


def test_cyclic_rotation(config_copy, make_paths, make_loader):
    loader, _ = make_loader(lambda text: "A")
    summary = run_experiment(config_copy(prompts={"option_rotation": "cyclic"}), make_paths(), loader_fn=loader,
                             download=False)
    records, files = read_outputs(summary)
    assert len(records) == 3 + 3 + 4
    for task_index in range(3):
        task_records = [r for r in records if r["task_index"] == task_index]
        assert sum(r["correct"] for r in task_records) == 1
        assert sorted(r["rotation"] for r in task_records) == list(range(len(task_records)))
    rotated = [r for r in records if r["task_index"] == 0 and r["rotation"] == 1][0]
    assert rotated["option_order"] == [1, 2, 0] and rotated["predicted_letter"] == "B"
    assert max(rotated["option_logprobs_posterior"], key=rotated["option_logprobs_posterior"].get) == "B"
    assert math.isclose(summary["accuracy"]["per_agent_accuracy"], 3 / 10)
    assert set(summary["accuracy"]["per_rotation"]) == {"0", "1", "2", "3"}
    assert files["predictions"].endswith("predictions_answer_only_kall_rotcyclic_N3.json")


def test_tie_counts_as_wrong(config_copy, make_paths, make_loader):
    loader, _ = make_loader(lambda text: "A" if "fact zero of task one" in text else "B")
    summary = run_experiment(config_copy(agents={"facts_per_agent": 1}), make_paths(), num_samples=1,
                             loader_fn=loader, download=False)
    assert summary["majority"]["ties"] == 1 and summary["majority"]["majority_accuracy"] == 0
    assert summary["accuracy"]["per_agent_accuracy"] == 0.5


def test_unparseable_replies_count_as_wrong(config_copy, make_paths, make_loader):
    loader, _ = make_loader(lambda text: "zzz")
    summary = run_experiment(config_copy(), make_paths(), loader_fn=loader, download=False)
    records, _ = read_outputs(summary)
    assert summary["accuracy"]["parse_failures"] == 3 and summary["accuracy"]["per_agent_accuracy"] == 0
    assert summary["majority"]["no_answer"] == 3
    assert all(r["predicted_letter"] is None and not r["parse_ok"] for r in records)


def test_reasoning_mode_saves_reasoning(config_copy, make_paths, make_loader):
    loader, tracker = make_loader(lambda text: "Reasoning: because it is safe.\nAnswer: A")
    config = config_copy(prompts={"template": "reasoning_answer"})
    summary = run_experiment(config, make_paths(), reasoning=True, loader_fn=loader, download=False)
    records, files = read_outputs(summary)
    assert all(r["reasoning"] == "because it is safe." for r in records)
    assert all(r["option_logprobs_prior"] is None and r["option_logprobs_posterior"] is None for r in records)
    assert records[0]["correct"] and summary["reasoning"] is True
    assert tracker["backends"][0].calls[0]["max_new_tokens"] == 64
    assert files["predictions"].endswith("predictions_reasoning_answer_kall_rotnone_N3.json")


def test_reasoning_flag_must_match_template(config_copy, make_paths, make_loader):
    loader, _ = make_loader()
    with pytest.raises(AssertionError, match="--reasoning is on"):
        run_experiment(config_copy(), make_paths(), reasoning=True, loader_fn=loader, download=False)


def test_overrides_load_each_model_once(config_copy, make_paths, make_loader):
    loader, tracker = make_loader()
    config = config_copy(models={"other": {"hf_name": "fake/second-model", "model_class": "auto",
                                           "dtype": "float32", "device": "cpu"}},
                         agents={"facts_per_agent": 1, "overrides": {1: "other"}})
    summary = run_experiment(config, make_paths(), loader_fn=loader, download=False)
    records, _ = read_outputs(summary)
    assert len(tracker["loads"]) == 2
    assert {r["model_name"] for r in records if r["agent_index"] == 1} == {"fake/second-model"}
    assert {r["model_name"] for r in records if r["agent_index"] != 1} == {"fake/causal-model"}


def test_tracing_off_writes_no_trace(config_copy, make_paths, make_loader):
    loader, _ = make_loader()
    summary = run_experiment(config_copy(telemetry={"enabled": False}), make_paths(), loader_fn=loader,
                             download=False)
    records, _ = read_outputs(summary)
    assert summary["trace_file"] is None
    assert all(r["trace_id"] == "0" * 32 for r in records)
