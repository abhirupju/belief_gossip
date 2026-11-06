"""Registry, model-class detection, answer/reasoning parsing, option log-probabilities and dtype choice."""

import json
import math

import pytest

from utility import model_utils as mu


def test_same_model_loads_once_and_is_shared(fake_model_root, make_loader):
    loader, tracker = make_loader()
    first = mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    second = mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    assert first is second and first.backend is second.backend
    assert len(tracker["loads"]) == 1


def test_different_model_loads_a_second_copy(fake_model_root, make_loader):
    loader, tracker = make_loader()
    mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    mu.load_model("fake/second-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    assert len(tracker["loads"]) == 2


def test_same_model_on_other_device_is_refused(fake_model_root, make_loader):
    loader, _ = make_loader()
    mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    with pytest.raises(AssertionError, match="refusing to load a second copy"):
        mu.load_model("fake/causal-model", fake_model_root, "mps", "float16", "auto", loader, download=False)


def test_clear_registry_releases_models(fake_model_root, make_loader):
    loader, tracker = make_loader()
    mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    mu.clear_model_registry()
    assert mu.get_model_registry() == {} and tracker["live"] == 0


def test_model_class_auto_detection(fake_model_root):
    assert mu.detect_model_class(mu.model_dir_for("fake/causal-model", fake_model_root)) == "causal"
    assert mu.detect_model_class(mu.model_dir_for("fake/vision-model", fake_model_root)) == "multimodal"


def test_chat_message_shapes():
    multimodal = mu.build_chat_messages("sys", "usr", "multimodal")
    assert multimodal[1]["content"] == [{"type": "text", "text": "usr"}]
    causal = mu.build_chat_messages("sys", "usr", "causal")
    assert causal[1]["content"] == "usr"


@pytest.mark.parametrize("reply, expected", [
    ("B", "B"), (" b ", "B"), ("(C)", "C"), ("A.", "A"), ("Answer: C", "C"), ("The answer is B", "B"),
    ("<think>maybe A</think>C", "C"), ("West City", "A"), ("I pick C because it is safe", "C"),
    ("D", None), ("", None), ("A or B", None), ("Answer: West", None),
])
def test_parse_option_letter(reply, expected):
    options = ["West City", "East Town", "North Hill"]
    assert mu.parse_option_letter(reply, ["A", "B", "C"], options) == expected


@pytest.mark.parametrize("reply, reasoning, letter", [
    ("Reasoning: x\nAnswer: B", "x", "B"),
    ("Reasoning: the answer: is unclear\nAnswer: C", "the answer: is unclear", "C"),
    ("Reasoning: fine\nanswer: c", "fine", "C"),
    ("No marker here, maybe A", "No marker here, maybe A", None),
])
def test_parse_reasoning_response(reply, reasoning, letter):
    assert mu.parse_reasoning_response(reply, ["A", "B", "C"], None, "Answer:") == (reasoning, letter)


def test_extract_option_logprobs_combines_variants_and_normalizes():
    vocab = [-30.0] * 10
    vocab[1], vocab[2], vocab[3] = math.log(0.2), math.log(0.2), math.log(0.1)
    scores = mu.extract_option_logprobs(vocab, {"A": [1, 2], "B": [3]})
    assert math.isclose(math.exp(scores["A"]), 0.8, rel_tol=1e-9)
    assert math.isclose(sum(math.exp(v) for v in scores.values()), 1.0, rel_tol=1e-9)
    with pytest.raises(AssertionError, match="outside the vocabulary"):
        mu.extract_option_logprobs(vocab, {"A": [99]})


def test_get_option_token_ids_with_fake_tokenizer():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            table = {"A": [1], " A": [2], "B": [3], " B": [4, 5], "C": [1]}
            return table[text] if text in table else [9, 9]

    assert mu.get_option_token_ids(Tokenizer(), ["A", "B"]) == {"A": [1, 2], "B": [3]}
    with pytest.raises(AssertionError, match="share token ids"):
        mu.get_option_token_ids(Tokenizer(), ["A", "C"])
    with pytest.raises(AssertionError, match="not a single token"):
        mu.get_option_token_ids(Tokenizer(), ["Z"])


def test_query_and_score_pass_through(fake_model_root, make_loader):
    loader, tracker = make_loader(lambda text: "B")
    handle = mu.load_model("fake/causal-model", fake_model_root, "cpu", "float32", "auto", loader, download=False)
    ids = handle.backend.option_token_ids(["A", "B", "C"])
    messages = mu.build_chat_messages("s", "Information only you know:\n- x", "causal")
    result = mu.query_model(handle, messages, 4, False, ids)
    assert result.text == "B" and max(result.option_logprobs, key=result.option_logprobs.get) == "B"
    prior = mu.score_options(handle, mu.build_chat_messages("s", "(You have no private information)", "causal"), ids)
    assert len(set(round(v, 9) for v in prior.values())) == 1
    with pytest.raises(AssertionError, match="max_new_tokens"):
        mu.query_model(handle, messages, 0, False)


def test_choose_dtype_cpu_is_float32_without_loading(fake_model_root, make_loader):
    loader, tracker = make_loader()
    assert mu.choose_dtype("fake/causal-model", fake_model_root, "cpu", "auto", loader, download=False) == "float32"
    assert tracker["loads"] == []


def test_choose_dtype_reuses_saved_choice(fake_model_root, make_loader):
    loader, tracker = make_loader()
    folder = mu.model_dir_for("fake/causal-model", fake_model_root)
    (folder / "dtype_choice_mps.json").write_text(json.dumps({"dtype": "bfloat16"}), encoding="utf-8")
    assert mu.choose_dtype("fake/causal-model", fake_model_root, "mps", "auto", loader, download=False) == "bfloat16"
    assert tracker["loads"] == []


def test_choose_dtype_benchmarks_one_copy_at_a_time(fake_model_root, make_loader):
    loader, tracker = make_loader(delays={"bfloat16": 0.03, "float16": 0.0})
    best = mu.choose_dtype("fake/causal-model", fake_model_root, "mps", "auto", loader, download=False)
    assert best == "float16"
    assert [load["dtype"] for load in tracker["loads"]] == ["bfloat16", "float16"]
    assert tracker["max_live"] == 1 and tracker["live"] == 0
    assert mu.get_model_registry() == {}
    saved = json.loads((mu.model_dir_for("fake/causal-model", fake_model_root) / "dtype_choice_mps.json").read_text())
    assert saved["dtype"] == "float16"
