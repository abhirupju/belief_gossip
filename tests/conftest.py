"""Shared fixtures: three fake tasks, a fake model backend, fake templates and an in-memory span exporter."""

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from utility import hiddenbench_utils as hb
from utility import model_utils as mu


def make_fake_tasks() -> list[dict[str, Any]]:
    """2 facts / 3 options; 3 facts / 3 options; 4 facts / 4 options."""
    return [
        {"id": 1, "name": "fake_two_facts", "description": "Pick the safest city.",
         "shared_information": ["Shared fact one.", "Shared fact two."],
         "hidden_information": ["fact zero of task one", "fact one of task one"],
         "possible_answers": ["Alpha City", "Beta Town", "Gamma Hill"], "correct_answer": "Alpha City",
         "rationale": None},
        {"id": 2, "name": "fake_three_facts", "description": "Pick the best route.",
         "shared_information": ["Route info."],
         "hidden_information": ["fact zero of task two", "fact one of task two", "fact two of task two"],
         "possible_answers": ["North", "South", "East"], "correct_answer": "South", "rationale": None},
        {"id": 3, "name": "fake_four_facts", "description": "Pick the right number.",
         "shared_information": ["Numbers are tricky."],
         "hidden_information": ["fact zero of task three", "fact one of task three", "fact two of task three",
                                "fact three of task three"],
         "possible_answers": ["One", "Two", "Three", "Four"], "correct_answer": "Three", "rationale": None},
    ]


def user_text_of(messages: list[dict[str, Any]]) -> str:
    """The user message text, whether content is a string or a list of parts."""
    content = messages[-1]["content"]
    return content if isinstance(content, str) else " ".join(part["text"] for part in content)


def fake_vocab_logprobs(favoured: str | None, letters: list[str]) -> list[float]:
    """A 128-entry 'vocabulary' (token id = ord(char)); favoured letter gets most of the mass."""
    vector = [-20.0] * 128
    for letter in letters:
        vector[ord(letter)] = -0.1 if letter == favoured else (-1.0 if favoured is None else -3.0)
    return vector


class FakeBackend:
    """Scripted stand-in for HFBackend; records calls and how many copies are alive."""

    def __init__(self, reply_fn: Callable[[str], str], tracker: dict[str, Any], dtype: str, delays: dict[str, float]):
        """reply_fn(user_text) -> reply; tracker counts live copies; delays[dtype] slows generate()."""
        self.reply_fn, self.tracker, self.dtype, self.delays = reply_fn, tracker, dtype, delays
        self.calls: list[dict[str, Any]] = []
        tracker["live"] += 1
        tracker["max_live"] = max(tracker["max_live"], tracker["live"])

    def option_token_ids(self, letters: list[str]) -> dict[str, list[int]]:
        """Token id of a letter is its character code."""
        return {letter: [ord(letter)] for letter in letters}

    def generate(self, messages, max_new_tokens, do_sample, option_token_ids=None, seed=0) -> mu.GenerationResult:
        """Scripted reply; first-token log-probabilities favour the letter the reply starts with."""
        text = user_text_of(messages)
        self.calls.append({"kind": "generate", "user": text, "max_new_tokens": max_new_tokens})
        time.sleep(self.delays.get(self.dtype, 0.0))
        reply = self.reply_fn(text)
        logprobs = None
        if option_token_ids is not None:
            letters = list(option_token_ids)
            first = reply.strip()[:1].upper()
            logprobs = mu.extract_option_logprobs(fake_vocab_logprobs(first if first in letters else None, letters),
                                                  option_token_ids)
        return mu.GenerationResult(reply, logprobs, len(text.split()), len(reply.split()))

    def score_options(self, messages, option_token_ids) -> dict[str, float]:
        """Uniform when the agent has no private facts, else favours the scripted reply's letter."""
        text = user_text_of(messages)
        self.calls.append({"kind": "score", "user": text})
        letters = list(option_token_ids)
        first = None if "no private information" in text else self.reply_fn(text).strip()[:1].upper()
        return mu.extract_option_logprobs(fake_vocab_logprobs(first if first in letters else None, letters),
                                          option_token_ids)

    def release(self) -> None:
        """Mark this copy as freed."""
        self.tracker["live"] -= 1


@pytest.fixture(autouse=True)
def clean_registry():
    """Every test starts and ends with no loaded models."""
    mu.get_model_registry().clear()
    yield
    mu.get_model_registry().clear()


@pytest.fixture
def fake_tasks() -> list[dict[str, Any]]:
    """Three fake HiddenBench tasks."""
    return make_fake_tasks()


@pytest.fixture
def fake_data_dir(tmp_path: Path, fake_tasks) -> Path:
    """Fake raw JSON array, converted with the real conversion and offset-index functions."""
    data_dir = tmp_path / "data"
    paths = hb.dataset_paths(data_dir)
    paths["raw_dir"].mkdir(parents=True)
    raw = paths["raw_dir"] / "hiddenbench.json"
    raw.write_text(json.dumps(fake_tasks, indent=2), encoding="utf-8")
    hb.convert_json_to_jsonl(raw, paths["jsonl"])
    hb.build_offset_index(paths["jsonl"], paths["offsets"])
    return data_dir


@pytest.fixture
def real_prompt_dir() -> Path:
    """The repository's own templates."""
    return Path(__file__).resolve().parent.parent / "prompts" / "hiddenbench_qwen35_08b"


@pytest.fixture
def fake_prompt_dir(tmp_path: Path) -> Path:
    """Two small fake templates in a temporary folder."""
    folder = tmp_path / "prompts"
    folder.mkdir()
    (folder / "short.yaml").write_text(
        'name: short\nmode: answer_only\nempty_private_facts: "(none)"\nsystem: "Be brief."\n'
        'user: "{description}\\nShared:\\n{shared_facts}\\nPrivate:\\n{private_facts}\\n'
        'Options:\\n{options}\\nPick {letters}."\n', encoding="utf-8")
    (folder / "think.yaml").write_text(
        'name: think\nmode: reasoning\nanswer_marker: "Answer:"\nempty_private_facts: "(none)"\n'
        'system: "Reason."\nuser: "{private_facts}\\n{options}"\n', encoding="utf-8")
    return folder


@pytest.fixture
def fake_model_root(tmp_path: Path) -> Path:
    """Model folders with a config.json so model_class 'auto' can be resolved."""
    root = tmp_path / "model"
    for name, arch in (("fake/causal-model", "FakeForCausalLM"), ("fake/second-model", "FakeForCausalLM"),
                       ("fake/vision-model", "FakeForConditionalGeneration")):
        folder = mu.model_dir_for(name, root)
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"architectures": [arch]}), encoding="utf-8")
    return root


@pytest.fixture
def make_loader() -> Callable:
    """Factory: make_loader(reply_fn, delays) -> (loader_fn, tracker)."""

    def factory(reply_fn: Callable[[str], str] = lambda text: "A", delays: dict[str, float] | None = None):
        tracker = {"live": 0, "max_live": 0, "loads": [], "backends": []}

        def loader(model_path, device, dtype, model_class):
            tracker["loads"].append({"path": str(model_path), "device": device, "dtype": dtype,
                                     "model_class": model_class})
            backend = FakeBackend(reply_fn, tracker, dtype, delays or {})
            tracker["backends"].append(backend)
            return backend

        return loader, tracker

    return factory


@pytest.fixture
def base_config() -> dict[str, Any]:
    """A valid config using the fake causal model on CPU."""
    return {
        "experiment": {"name": "test_run", "dataset": "hiddenbench", "protocol": "single_round"},
        "models": {"fake": {"hf_name": "fake/causal-model", "model_class": "auto", "dtype": "float32",
                            "device": "cpu"}},
        "agents": {"facts_per_agent": "all", "default_model": "fake", "overrides": {}},
        "prompts": {"dir": "prompts/hiddenbench_qwen35_08b", "template": "answer_only", "option_rotation": "none"},
        "generation": {"max_new_tokens": 4, "max_new_tokens_reasoning": 64, "do_sample": False, "seed": 0},
        "telemetry": {"enabled": True, "capture_content": True, "otlp_endpoint": None},
    }


@pytest.fixture
def make_paths(tmp_path: Path, fake_data_dir: Path, fake_model_root: Path, real_prompt_dir: Path) -> Callable:
    """Factory for the paths dict run_experiment expects."""

    def factory(prompt_dir: Path | None = None) -> dict[str, Path]:
        return {"data_dir": fake_data_dir, "model_dir": fake_model_root, "results_dir": tmp_path / "results",
                "prompt_dir": prompt_dir or real_prompt_dir}

    return factory


@pytest.fixture
def span_exporter():
    """In-memory span exporter from the OpenTelemetry SDK."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    return InMemorySpanExporter()


@pytest.fixture
def config_copy(base_config) -> Callable:
    """Factory returning a deep copy of base_config with nested overrides applied."""

    def factory(**sections: dict[str, Any]) -> dict[str, Any]:
        config = copy.deepcopy(base_config)
        for section, values in sections.items():
            config[section].update(values)
        return config

    return factory


# ----------------------------------------------------------------------------------------------- WearableQA fakes


def make_wqa_fake_raw() -> dict[str, Any]:
    """A small WearableQA source: 6 users, 8 multimodal-phenotype questions (all four gold classes) + 1 other."""
    from utility import wqa_split as ws

    codes = [code for _, code in ws.phenotype_option_rules()]
    texts = {code: prefix + " - fake detail" for prefix, code in ws.phenotype_option_rules()}

    def options_for(gold: str, shift: int) -> tuple[dict[str, str], str]:
        letters = "ABCDEFGHIJ"
        options = {letters[i]: texts[codes[(i + shift) % 10]] for i in range(10)}
        return options, next(l for l, t in options.items() if ws.option_class(t) == gold)

    def history(rhr: float, steps: float | None, start_day: int = 1) -> list[dict[str, Any]]:
        rows = []
        for day in range(start_day, 32):
            row = {"date": f"2024-01-{day:02d}", "resting_hr": rhr + day % 3, "sleep_duration": 7.0 + (day % 4) / 4,
                   "hrv": 45.0, "exercise_count": float(day % 2), "exercise_hours": 0.5 if day % 2 else 0.0,
                   "avg_stress_level": -1.0, "_source": "x"}
            if steps is not None and day % 7:
                row["steps"] = steps + 10 * day
            rows.append(row)
        return rows

    good = {"hgba1c": 5.2, "glu": 85.0, "ins": 5.0, "tg": 90.0, "hdl": 60.0, "ldl": 100.0, "tsh": 2.0}
    bad = {"hgba1c": 5.9, "glu": 120.0, "ins": 20.0, "tg": 220.0, "hdl": 35.0, "ldl": 160.0}
    no_ins = {k: v for k, v in good.items() if k != "ins"}
    spec = [  # (question id, user, gold, blood, sex)
        ("q1", "u1", "CH", good, "m"), ("q2", "u1", "CH", no_ins, "m"), ("q3", "u2", "CA", bad, "f"),
        ("q4", "u3", "D-LF", good, "f"), ("q5", "u4", "D-FA", bad, "m"), ("q6", "u5", "CA", bad, "m"),
        ("q7", "u6", "CH", good, "f"), ("q8", "u5", "D-LF", no_ins, "m"),
    ]
    fit = {"u1": 50.0, "u2": 75.0, "u3": 72.0, "u4": 48.0, "u5": 78.0, "u6": 52.0}
    histories = {u: history(r, None if u == "u3" else 9000.0 if r < 60 else 4000.0) for u, r in fit.items()}
    mcqs = []
    for i, (qid, user, gold, blood, sex) in enumerate(spec):
        options, gold_letter = options_for(gold, i)
        mcqs.append({"id": qid, "user_id": user, "question_type": "multimodal_phenotype",
                     "stem": "Based on this user's most recent 28-day wearable record and the available blood "
                             "markers, which phenotype label best characterizes this individual?",
                     "options": options, "gt_letter": gold_letter, "reasoning_group": "health", "signal": "cross",
                     "grounding": "literature",
                     "context": {"demographics": {"age": 40.0, "sex": sex, "bmi": 26.5, "ethnicity": "white"},
                                 "blood_panel": dict(blood)},
                     "end_date": "2024-01-31", "window_size": 28, "full_dropped_metrics": [], "no_cohort": i == 7})
    other = dict(mcqs[0], id="q9", question_type="signal_summary")
    cohort = {"rhr": {"p10": 49.0, "p25": 54.0, "p50": 60.0, "p75": 66.0, "p90": 71.0, "_n_users": 10},
              "steps": {"p10": 4900.0, "p25": 6300.0, "p50": 8100.0, "p75": 10200.0, "p90": 12800.0},
              "sleep_duration": {"p10": 6.1, "p25": 6.7, "p50": 7.2, "p75": 7.7, "p90": 8.2}}
    return {"n_mcqs": 9, "mcqs": mcqs + [other], "user_histories": histories, "cohort_reference": cohort,
            "description": "fake WearableQA"}


@pytest.fixture
def wqa_raw() -> dict[str, Any]:
    """Fake WearableQA source (see make_wqa_fake_raw)."""
    return make_wqa_fake_raw()


@pytest.fixture
def wqa_prompt_dir() -> Path:
    """The repository's WearableQA templates."""
    return Path(__file__).resolve().parent.parent / "prompts" / "wearableqa_phenotype"


@pytest.fixture
def wqa_nokey_dir() -> Path:
    """The repository's no-key overlay."""
    return Path(__file__).resolve().parent.parent / "prompts" / "wearableqa_phenotype_nokey"


@pytest.fixture
def wqa_config():
    """A valid WearableQA config using the fake causal model on CPU."""
    return {
        "experiment": {"name": "wqa_test", "dataset": "wearableqa_phenotype", "protocol": "single_round"},
        "models": {"fake": {"hf_name": "fake/causal-model", "model_class": "auto", "dtype": "float32",
                            "device": "cpu"},
                   "fake2": {"hf_name": "fake/second-model", "model_class": "auto", "dtype": "float32",
                             "device": "cpu"}},
        "agents": {"default_model": "fake", "devices": ["watch", "phone", "ring", "clinic_glycaemic", "clinic_lipid"],
                   "centralised": True, "centralised_devices": None},
        "data": {"raw_file": "wearableqa/WearableQA_raw.json", "tasks_file": "wearableqa/tasks.jsonl",
                 "window_days": 28},
        "prompts": {"dir": "prompts/wearableqa_phenotype", "nokey_dir": "prompts/wearableqa_phenotype_nokey",
                    "variant": "strict_key"},
        "generation": {"max_new_tokens": 4, "do_sample": False, "seed": 0},
        "telemetry": {"enabled": True, "capture_content": True, "otlp_endpoint": None},
    }


@pytest.fixture
def wqa_paths(tmp_path, wqa_raw, fake_model_root):
    """Raw file in a temporary data root; prompts from the repository."""
    data_dir = tmp_path / "data"
    raw = data_dir / "wearableqa" / "WearableQA_raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps(wqa_raw), encoding="utf-8")
    return {"data_dir": data_dir, "raw": raw, "tasks": data_dir / "wearableqa" / "tasks.jsonl",
            "model_dir": fake_model_root, "results_dir": tmp_path / "results",
            "prompt_root": Path(__file__).resolve().parent.parent}
