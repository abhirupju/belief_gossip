"""Smoke test with the real Qwen3.5-0.8B (skipped unless run with: pytest -m slow)."""

import pytest

from agent_exp import run_experiment
from main import get_code_root, resolve_path
from agent import load_config
from utility import model_utils as mu


@pytest.mark.slow
def test_real_model_two_tasks_and_flat_memory(tmp_path):
    code_root = get_code_root()
    config = load_config(code_root / "config" / "hiddenbench_qwen35_08b.yaml")
    paths = {"data_dir": resolve_path("data", code_root), "model_dir": resolve_path("model", code_root),
             "results_dir": tmp_path / "results", "prompt_dir": resolve_path(config["prompts"]["dir"], code_root)}
    summary = run_experiment(config, paths, num_samples=2)
    assert summary["accuracy"]["num_answers"] == 2
    handle = next(iter(mu.get_model_registry().values()))
    letters = ["A", "B", "C"]
    ids = handle.backend.option_token_ids(letters)
    messages = mu.build_chat_messages("Answer with a single letter only.",
                                      "Which is a fruit?\nA. Apple\nB. Brick\nC. Chair\nReply with only the letter.",
                                      handle.model_class)
    first = mu.query_model(handle, messages, 1, False, ids)
    assert first.text.strip()[:1] in letters, f"first generated token is {first.text!r}, not an option letter"
    before = mu.memory_report(handle.device)["allocated_mb"]
    for _ in range(20):
        mu.query_model(handle, messages, 4, False, ids)
    after = mu.memory_report(handle.device)["allocated_mb"]
    assert after <= before * 1.05 + 50, f"memory grew from {before:.0f} MB to {after:.0f} MB over 20 queries"
