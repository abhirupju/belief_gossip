"""Config handling, fact partitioning and the Agent class.

An agent holds an index, a reference to a shared model handle and its own prompt; it never loads a model.
It is dataset-agnostic: the experiment builds its prompt fields through the dataset adapter.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utility.model_utils import (ModelHandle, build_chat_messages, parse_option_letter, parse_reasoning_response,
                                 query_model, score_options)
from utility.prompt_utils import render_prompt, unrotate_letter, unrotate_scores
from utility.telemetry_utils import (Telemetry, add_event, agent_span_attributes, get_span_ids, set_attributes,
                                     start_span)


def load_config(path: Path) -> dict[str, Any]:
    """Read the YAML config and validate it."""
    path = Path(path)
    assert path.is_file(), f"load_config: config file {path} does not exist"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict), f"load_config: {path} does not contain a YAML mapping"
    validate_config(config)
    return config


def validate_facts_per_agent(value: Any) -> None:
    """k must be 'all' or a positive integer."""
    assert value == "all" or (type(value) is int and value >= 1), (
        f"validate_facts_per_agent: facts_per_agent is {value!r}; expected a positive integer or 'all'"
    )


def validate_config(config: dict[str, Any]) -> None:
    """Assert required sections and keys, and that every agent model points to a defined model."""
    for section in ("experiment", "models", "agents", "prompts", "generation", "telemetry"):
        assert isinstance(config.get(section), dict), f"validate_config: missing or invalid section '{section}'"
    for key in ("name", "dataset", "protocol"):
        assert config["experiment"].get(key), f"validate_config: experiment.{key} is missing"
    assert config["models"], "validate_config: models section is empty"
    for key, model in config["models"].items():
        assert isinstance(model, dict) and isinstance(model.get("hf_name"), str), (
            f"validate_config: models.{key} needs an hf_name"
        )
        assert model.get("model_class") in ("auto", "multimodal", "causal"), (
            f"validate_config: models.{key}.model_class is {model.get('model_class')!r}; use auto, multimodal or causal"
        )
        assert model.get("dtype") in ("auto", "bfloat16", "float16", "float32"), (
            f"validate_config: models.{key}.dtype is {model.get('dtype')!r}; use auto, bfloat16, float16 or float32"
        )
        assert model.get("device") in ("auto", "mps", "cuda", "cpu"), (
            f"validate_config: models.{key}.device is {model.get('device')!r}; use auto, mps, cuda or cpu"
        )
    agents = config["agents"]
    validate_facts_per_agent(agents.get("facts_per_agent"))
    assert agents.get("default_model") in config["models"], (
        f"validate_config: agents.default_model {agents.get('default_model')!r} is not one of {list(config['models'])}"
    )
    overrides = agents.get("overrides") or {}
    assert isinstance(overrides, dict), "validate_config: agents.overrides must be a mapping {agent_index: model_key}"
    for index, key in overrides.items():
        assert type(index) is int and index >= 0, (
            f"validate_config: override key {index!r} must be a non-negative integer agent index"
        )
        assert key in config["models"], f"validate_config: override for agent {index} uses unknown model {key!r}"
    prompts = config["prompts"]
    assert prompts.get("dir") and prompts.get("template"), "validate_config: prompts.dir and prompts.template are required"
    assert prompts.get("option_rotation") in ("none", "cyclic"), (
        f"validate_config: prompts.option_rotation is {prompts.get('option_rotation')!r}; use none or cyclic"
    )
    generation = config["generation"]
    for key in ("max_new_tokens", "max_new_tokens_reasoning", "seed"):
        assert type(generation.get(key)) is int and generation[key] >= 0, (
            f"validate_config: generation.{key} must be a non-negative integer"
        )
    assert generation["max_new_tokens"] >= 1 and generation["max_new_tokens_reasoning"] >= 1, (
        "validate_config: generation lengths must be at least 1"
    )
    assert type(generation.get("do_sample")) is bool, "validate_config: generation.do_sample must be true or false"
    telemetry = config["telemetry"]
    for key in ("enabled", "capture_content"):
        assert type(telemetry.get(key)) is bool, f"validate_config: telemetry.{key} must be true or false"
    assert telemetry.get("otlp_endpoint") is None or isinstance(telemetry["otlp_endpoint"], str), (
        "validate_config: telemetry.otlp_endpoint must be null or a URL string"
    )


def partition_facts(num_facts: int, facts_per_agent: int | str) -> list[list[int]]:
    """Fact indices per agent: blocks of k in dataset order; only the last block can be smaller."""
    assert type(num_facts) is int and num_facts >= 1, f"partition_facts: num_facts must be >= 1, got {num_facts!r}"
    validate_facts_per_agent(facts_per_agent)
    k = num_facts if facts_per_agent == "all" else min(facts_per_agent, num_facts)
    return [list(range(start, min(start + k, num_facts))) for start in range(0, num_facts, k)]


def get_model_key_for_agent(config: dict[str, Any], agent_idx: int) -> str:
    """Override if present, else the default model key."""
    overrides = config["agents"].get("overrides") or {}
    return overrides.get(agent_idx, config["agents"]["default_model"])


@dataclass
class AgentOutput:
    """One agent's answer in one round (letters and scores in the dataset's own option order)."""

    agent_id: str
    agent_index: int
    role: str
    round: int
    rotation: int
    model_name: str
    fact_indices: list[int]
    private_facts: list[str]
    raw_response: str
    reasoning: str
    predicted_letter: str | None
    parse_ok: bool
    option_logprobs_prior: dict[str, float] | None
    option_logprobs_posterior: dict[str, float] | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    trace_id: str
    span_id: str


@dataclass
class Agent:
    """One agent. Keeps a message history so later protocols (debate, gossip) can append to it."""

    agent_idx: int
    model_handle: ModelHandle
    generation_cfg: dict[str, Any]
    template: dict[str, Any]
    telemetry: Telemetry
    role: str = "agent"
    task_index: int | None = None
    fields: dict[str, Any] | None = None
    fact_indices: list[int] = field(default_factory=list)
    rotation: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agent_id(self) -> str:
        """Stable id used in records and traces, e.g. 'agent-0'."""
        return f"{self.role}-{self.agent_idx}"

    def set_task(self, task_index: int, fields: dict[str, Any], fact_indices: list[int], rotation: int) -> None:
        """Store this agent's prompt fields (its own private facts only) for one task and rotation."""
        assert fields["private_facts"] is not None and len(fields["private_facts"]) == len(fact_indices), (
            f"set_task: {self.agent_id} got {len(fields['private_facts'])} private facts for indices {fact_indices}"
        )
        self.task_index, self.fields, self.fact_indices, self.rotation = task_index, fields, list(fact_indices), rotation

    def reset(self) -> None:
        """Forget the current task and history."""
        self.task_index, self.fields, self.fact_indices, self.rotation = None, None, [], 0
        self.history = []

    def render_messages(self, with_private_facts: bool) -> tuple[list[dict[str, Any]], str, str]:
        """Chat messages for the current task, with or without this agent's private facts."""
        fields = self.fields if with_private_facts else {**self.fields, "private_facts": []}
        system_text, user_text = render_prompt(self.template, fields)
        return build_chat_messages(system_text, user_text, self.model_handle.model_class), system_text, user_text

    def answer(self, round_idx: int = 0) -> AgentOutput:
        """Query the model once (plus a prompt-only prior pass in answer-only mode) and parse the reply."""
        assert self.fields is not None, f"answer: {self.agent_id} has no task; call set_task first"
        handle, generation, telemetry = self.model_handle, self.generation_cfg, self.telemetry
        reasoning_mode = self.template["mode"] == "reasoning"
        letters, options = self.fields["letters"], self.fields["options"]
        attributes = agent_span_attributes(self.agent_id, self.role, round_idx, self.task_index,
                                           self.fact_indices, self.rotation)
        with start_span(telemetry, f"invoke_agent {self.agent_id}", attributes) as span:
            messages, system_text, user_text = self.render_messages(with_private_facts=True)
            add_event(telemetry, span, "prompt_rendered",
                      {"template": self.template["name"], "system": system_text, "user": user_text},
                      content_keys=("system", "user"))
            token_ids = None if reasoning_mode else handle.backend.option_token_ids(letters)
            max_tokens = generation["max_new_tokens_reasoning"] if reasoning_mode else generation["max_new_tokens"]
            with start_span(telemetry, f"chat {handle.name}",
                            {"gen_ai.operation.name": "chat", "gen_ai.request.model": handle.name,
                             "gen_ai.agent.id": self.agent_id}) as chat_span:
                started = time.perf_counter()
                result = query_model(handle, messages, max_tokens, generation["do_sample"], token_ids,
                                     generation["seed"])
                latency = time.perf_counter() - started
                set_attributes(chat_span, {"gen_ai.usage.input_tokens": result.input_tokens,
                                           "gen_ai.usage.output_tokens": result.output_tokens,
                                           "bg.latency_s": latency})
                add_event(telemetry, chat_span, "reply", {"text": result.text}, content_keys=("text",))
            prior = None
            if not reasoning_mode:
                prior_messages, _, _ = self.render_messages(with_private_facts=False)
                with start_span(telemetry, f"score_options {handle.name}",
                                {"gen_ai.request.model": handle.name, "gen_ai.agent.id": self.agent_id,
                                 "bg.pass": "prior"}):
                    prior = score_options(handle, prior_messages, token_ids)
            if reasoning_mode:
                reasoning, displayed = parse_reasoning_response(result.text, letters, options,
                                                                self.template["answer_marker"])
            else:
                reasoning, displayed = "", parse_option_letter(result.text, letters, options)
            num_options = len(letters)
            letter = unrotate_letter(displayed, num_options, self.rotation)
            posterior = unrotate_scores(result.option_logprobs, num_options, self.rotation)
            prior = unrotate_scores(prior, num_options, self.rotation)
            add_event(telemetry, span, "answer_parsed",
                      {"letter": letter, "displayed_letter": displayed, "parse_ok": letter is not None,
                       "option_logprobs_posterior": posterior, "option_logprobs_prior": prior,
                       "reasoning": reasoning}, content_keys=("reasoning",))
            trace_id, span_id = get_span_ids(span)
        self.history.append({"round": round_idx, "role": "assistant", "content": result.text})
        return AgentOutput(
            agent_id=self.agent_id, agent_index=self.agent_idx, role=self.role, round=round_idx,
            rotation=self.rotation, model_name=handle.name, fact_indices=list(self.fact_indices),
            private_facts=list(self.fields["private_facts"]), raw_response=result.text, reasoning=reasoning,
            predicted_letter=letter, parse_ok=letter is not None, option_logprobs_prior=prior,
            option_logprobs_posterior=posterior, input_tokens=result.input_tokens,
            output_tokens=result.output_tokens, latency_s=latency, trace_id=trace_id, span_id=span_id,
        )


def create_agents(fact_blocks: list[list[int]], model_handles: dict[str, ModelHandle], config: dict[str, Any],
                  template: dict[str, Any], telemetry: Telemetry) -> list[Agent]:
    """One agent per fact block, each given the already-loaded handle of its model."""
    agents = []
    for agent_idx, _ in enumerate(fact_blocks):
        key = get_model_key_for_agent(config, agent_idx)
        assert key in model_handles, f"create_agents: model {key!r} for agent {agent_idx} was not loaded"
        agents.append(Agent(agent_idx, model_handles[key], config["generation"], template, telemetry))
    return agents
