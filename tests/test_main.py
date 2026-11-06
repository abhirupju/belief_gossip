"""Command-line parsing and config overrides."""

import copy

import pytest

from main import apply_overrides, parse_args, resolve_path


def test_overrides_apply_and_revalidate(base_config):
    args = parse_args(["--model", "org/other-model", "--num_samples", "5", "--facts_per_agent", "2",
                       "--option_rotation", "cyclic", "--reasoning", "--device", "cpu", "--dtype", "float32",
                       "--max_new_tokens", "128", "--no_trace_content"])
    config = apply_overrides(copy.deepcopy(base_config), args)
    model = config["models"]["fake"]
    assert model["hf_name"] == "org/other-model" and model["model_class"] == "auto"
    assert config["agents"]["facts_per_agent"] == 2
    assert config["prompts"]["option_rotation"] == "cyclic"
    assert config["prompts"]["template"] == "reasoning_answer"
    assert config["generation"]["max_new_tokens_reasoning"] == 128
    assert config["telemetry"]["capture_content"] is False
    assert args.num_samples == 5


def test_facts_per_agent_all_and_invalid():
    assert parse_args(["--facts_per_agent", "all"]).facts_per_agent == "all"
    with pytest.raises(AssertionError, match="facts_per_agent"):
        parse_args(["--facts_per_agent", "0"])


def test_relative_paths_resolve_against_code_root(tmp_path):
    assert resolve_path("data", tmp_path) == tmp_path / "data"
    assert resolve_path("/rds/projects/x", tmp_path).as_posix() == "/rds/projects/x"
