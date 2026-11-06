"""Fact partitioning, config validation, model choice per agent and agent prompts."""

import pytest

from agent import create_agents, get_model_key_for_agent, partition_facts, validate_config
from utility import model_utils as mu
from utility.prompt_utils import load_template
from utility.telemetry_utils import setup_telemetry


@pytest.mark.parametrize("num_facts, k, expected", [
    (4, "all", [[0, 1, 2, 3]]),
    (4, 1, [[0], [1], [2], [3]]),
    (4, 2, [[0, 1], [2, 3]]),
    (5, 2, [[0, 1], [2, 3], [4]]),
    (4, 3, [[0, 1, 2], [3]]),
    (4, 9, [[0, 1, 2, 3]]),
])
def test_partition_facts(num_facts, k, expected):
    assert partition_facts(num_facts, k) == expected


@pytest.mark.parametrize("bad_k", [0, -1, 1.5, "two", True])
def test_partition_rejects_bad_k(bad_k):
    with pytest.raises(AssertionError, match="facts_per_agent"):
        partition_facts(4, bad_k)


@pytest.mark.parametrize("section, values, message", [
    ("agents", {"default_model": "nope"}, "default_model"),
    ("agents", {"overrides": {0: "nope"}}, "unknown model"),
    ("agents", {"overrides": {"0": "fake"}}, "non-negative integer"),
    ("agents", {"facts_per_agent": 0}, "facts_per_agent"),
    ("experiment", {"protocol": ""}, "protocol"),
    ("prompts", {"option_rotation": "random"}, "option_rotation"),
    ("generation", {"do_sample": "no"}, "do_sample"),
])
def test_config_validation_errors(config_copy, section, values, message):
    with pytest.raises(AssertionError, match=message):
        validate_config(config_copy(**{section: values}))


def test_override_vs_default(config_copy):
    config = config_copy(models={"other": {"hf_name": "fake/second-model", "model_class": "auto",
                                           "dtype": "float32", "device": "cpu"}},
                         agents={"overrides": {1: "other"}})
    validate_config(config)
    assert get_model_key_for_agent(config, 0) == "fake"
    assert get_model_key_for_agent(config, 1) == "other"


def test_create_agents_shares_one_handle(base_config, fake_model_root, make_loader, real_prompt_dir):
    loader, tracker = make_loader()
    handle = mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    telemetry = setup_telemetry(None, {}, enabled=False)
    template = load_template(real_prompt_dir, "answer_only")
    agents = create_agents(partition_facts(3, 1), {"fake": handle}, base_config, template, telemetry)
    assert [a.agent_id for a in agents] == ["agent-0", "agent-1", "agent-2"]
    assert all(a.model_handle is handle for a in agents) and len(tracker["loads"]) == 1


def test_agent_prompt_has_only_its_own_facts(base_config, fake_model_root, make_loader, real_prompt_dir, fake_tasks):
    from utility.hiddenbench_utils import build_prompt_fields

    loader, tracker = make_loader(lambda text: "B")
    handle = mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    telemetry = setup_telemetry(None, {}, enabled=False)
    template = load_template(real_prompt_dir, "answer_only")
    task = fake_tasks[1]
    agents = create_agents(partition_facts(3, 1), {"fake": handle}, base_config, template, telemetry)
    agent = agents[1]
    agent.set_task(1, build_prompt_fields(task, [1], 0), [1], 0)
    output = agent.answer()
    posterior_prompt = [c for c in tracker["backends"][0].calls if c["kind"] == "generate"][-1]["user"]
    assert task["description"] in posterior_prompt and "Route info." in posterior_prompt
    assert "fact one of task two" in posterior_prompt
    assert "fact zero of task two" not in posterior_prompt and "fact two of task two" not in posterior_prompt
    prior_prompt = [c for c in tracker["backends"][0].calls if c["kind"] == "score"][-1]["user"]
    assert "fact one of task two" not in prior_prompt and "no private information" in prior_prompt
    assert output.predicted_letter == "B" and output.parse_ok
    assert output.private_facts == ["fact one of task two"] and output.fact_indices == [1]
    assert set(output.option_logprobs_prior) == set(output.option_logprobs_posterior) == {"A", "B", "C"}
