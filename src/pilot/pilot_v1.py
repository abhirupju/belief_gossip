"""Pilot v1: load Qwen3.5-0.8B and HiddenBench samples; write every agent's prompt to an HTML file.

Run from the repo root: PYTHONPATH=src python src/pilot/pilot_v1.py
"""

import json
import math
from pathlib import Path
from typing import Any

from agent import partition_facts
from utility import hiddenbench_utils as hb
from utility.model_utils import (ModelHandle, build_chat_messages, detect_model_class, load_model, model_dir_for,
                                 get_option_token_ids, parse_option_letter, query_model,
                                 resolve_device)
from utility.prompt_utils import load_template, render_prompt


def get_code_root() -> Path:
    """The code/ folder (two levels above src/pilot/)."""
    return Path(__file__).resolve().parents[2]


def load_qwen(code_root: Path, model_name: str = "Qwen/Qwen3.5-0.8B", device: str = "auto",
              dtype: str = "bfloat16") -> ModelHandle:
    """Load the model from code/model (no download); model class is read from config.json."""
    return load_model(model_name, code_root / "model", resolve_device(device), dtype, "auto", download=False)


def load_samples(code_root: Path, num_samples: int) -> list[dict[str, Any]]:
    """The first num_samples HiddenBench tasks from code/data."""
    data_dir = code_root / "data"
    total = hb.get_num_samples(data_dir)
    assert 1 <= num_samples <= total, f"load_samples: num_samples {num_samples} must be in 1..{total}"
    return [hb.load_sample(i, data_dir) for i in range(num_samples)]


def get_model_class(code_root: Path, model_name: str = "Qwen/Qwen3.5-0.8B") -> str:
    """'multimodal' or 'causal', read from the model's config.json (no weights loaded)."""
    return detect_model_class(model_dir_for(model_name, code_root / "model"))


def load_prompt_template(code_root: Path, name: str = "answer_only") -> dict[str, Any]:
    """One YAML template from prompts/hiddenbench_qwen35_08b (answer_only, reasoning_answer, ...)."""
    return load_template(code_root / "prompts" / "hiddenbench_qwen35_08b", name)


def build_messages(sample: dict[str, Any], template: dict[str, Any], model_class: str,
                   fact_indices: list[int] | None = None) -> list[dict[str, Any]]:
    """Chat messages for one agent: shared facts plus the hidden facts at fact_indices (default: all)."""
    if fact_indices is None:
        fact_indices = list(range(hb.get_num_private_facts(sample)))
    fields = hb.build_prompt_fields(sample, fact_indices)
    system_text, user_text = render_prompt(template, fields)
    return build_chat_messages(system_text, user_text, model_class)


def create_agents(sample: dict[str, Any], template: dict[str, Any], model_class: str,
                  facts_per_agent: int = 1) -> list[dict[str, Any]]:
    """A 'no-private' baseline (shared facts only), one agent per block of hidden facts, and an 'all-facts' agent."""
    num_facts = hb.get_num_private_facts(sample)
    blocks = [[]] + partition_facts(num_facts, facts_per_agent) + [list(range(num_facts))]
    names = ["no-private"] + [f"agent-{i}" for i in range(len(blocks) - 2)] + ["all-facts"]
    return [{"agent_id": name, "fact_indices": block, "messages": build_messages(sample, template, model_class, block)}
            for name, block in zip(names, blocks)]


def ask_agent(handle: ModelHandle, messages: list[dict[str, Any]], letters: list[str],
              max_new_tokens: int = 10) -> dict[str, Any]:
    """Greedy reply, its parsed letter (None if unparseable) and first-token option log-probs (normalized over options)."""
    option_token_ids = get_option_token_ids(handle.backend.tokenizer, letters)
    result = query_model(handle, messages, max_new_tokens, do_sample=False, option_token_ids=option_token_ids)
    return {"text": result.text, "letter": parse_option_letter(result.text, letters),
            "option_logprobs": result.option_logprobs}


def probability_change(prior: dict[str, float], posterior: dict[str, float], true_letter: str) -> dict[str, Any]:
    """Per-option change in probability (posterior - prior) and the true answer's log-prob lift."""
    assert prior.keys() == posterior.keys(), f"probability_change: options differ: {list(prior)} vs {list(posterior)}"
    assert true_letter in prior, f"probability_change: true letter {true_letter!r} not in options {list(prior)}"
    return {"delta_p": {k: math.exp(posterior[k]) - math.exp(prior[k]) for k in prior},
            "true_lift": posterior[true_letter] - prior[true_letter]}


def format_probs(logprobs: dict[str, float]) -> str:
    """'A=0.332 B=0.293 C=0.376'."""
    return " ".join(f"{k}={math.exp(v):.3f}" for k, v in logprobs.items())


def message_text(message: dict[str, Any]) -> str:
    """Plain text of one chat message (list-of-parts content for multimodal models, a string otherwise)."""
    content = message["content"]
    return content if isinstance(content, str) else "".join(part["text"] for part in content if part["type"] == "text")


def agent_result(reply: dict[str, Any], prior: dict[str, float], true_letter: str) -> dict[str, Any]:
    """Reply, option probabilities, change from the no-private prior and true-answer lift, for the viewer."""
    change = probability_change(prior, reply["option_logprobs"], true_letter)
    return {"reply": reply["text"], "letter": reply["letter"],
            "probs": {k: math.exp(v) for k, v in reply["option_logprobs"].items()},
            "delta_p": change["delta_p"], "true_lift": change["true_lift"]}


def sample_record(sample: dict[str, Any], agents: list[dict[str, Any]],
                  results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What the viewer shows for one sample: facts, true answer, options, each agent's prompt and its result."""
    return {
        "id": sample["id"], "name": sample["name"], "true_letter": hb.get_true_letter(sample),
        "correct_answer": sample["correct_answer"], "shared": hb.get_shared_facts(sample),
        "hidden": hb.get_private_facts(sample), "options": dict(zip(hb.get_option_letters(sample), sample["possible_answers"])),
        "agents": [{"agent_id": a["agent_id"], "fact_indices": a["fact_indices"],
                    "messages": [{"role": m["role"], "text": message_text(m)} for m in a["messages"]],
                    **results[a["agent_id"]]}
                   for a in agents],
    }


def write_viewer(template_path: Path, out_path: Path, records: list[dict[str, Any]]) -> Path:
    """Fill the viewer template's /*DATA*/ placeholder with the records as JSON and write the page."""
    page = template_path.read_text(encoding="utf-8")
    assert "/*DATA*/" in page, f"write_viewer: {template_path} has no /*DATA*/ placeholder"
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page.replace("/*DATA*/", data), encoding="utf-8")
    return out_path


def run_sample(handle: ModelHandle, template: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Query every agent of one sample, print its probabilities and lift, and return the viewer record."""
    agents = create_agents(sample, template, handle.model_class, facts_per_agent=1)
    letters, true_letter = hb.get_option_letters(sample), hb.get_true_letter(sample)
    replies = {agent["agent_id"]: ask_agent(handle, agent["messages"], letters) for agent in agents}
    prior = replies["no-private"]["option_logprobs"]
    results = {agent_id: agent_result(reply, prior, true_letter) for agent_id, reply in replies.items()}
    print(f"Sample {sample['id']} ({sample['name']}): {len(agents)} agents, true letter {true_letter}")
    for agent in agents:
        result = results[agent["agent_id"]]
        delta = " ".join(f"{k}={v:+.3f}" for k, v in result["delta_p"].items())
        print(f"  {agent['agent_id']:<10} facts {str(agent['fact_indices']):<12} "
              f"p: {format_probs(replies[agent['agent_id']]['option_logprobs'])}  dp: {delta}  "
              f"lift(true)={result['true_lift']:+.3f}  reply {result['letter']}")
    return sample_record(sample, agents, results)


def main() -> None:
    """Query the no-private, one-fact and all-facts agents on every sample and write the HTML viewer."""
    code_root = get_code_root()
    handle = load_qwen(code_root)
    print(f"Loaded {handle.name} on {handle.device} ({handle.dtype}, {handle.model_class})")
    template = load_prompt_template(code_root)
    num_samples = hb.get_num_samples(code_root / "data")
    records = [run_sample(handle, template, sample) for sample in load_samples(code_root, num_samples)]
    out_path = write_viewer(Path(__file__).with_name("pilot_v1_viewer.html"),
                            code_root / "results" / "pilot" / "pilot_v1_prompts.html", records)
    print(f"Wrote {len(records)} samples to {out_path}")


if __name__ == "__main__":
    main()
