"""Template loading, validation, rendering (zero/one/several facts) and option rotation."""

import pytest

from utility import prompt_utils as pu


def fields(private_facts):
    return {"description": "Pick one.", "shared_facts": ["s1", "s2"], "private_facts": private_facts,
            "options": ["Alpha", "Beta", "Gamma"], "letters": ["A", "B", "C"]}


def test_list_and_load_fake_templates(fake_prompt_dir):
    assert pu.list_templates(fake_prompt_dir) == ["short", "think"]
    template = pu.load_template(fake_prompt_dir, "short")
    assert template["mode"] == "answer_only" and "raw_text" in template


def test_unknown_template_lists_available(fake_prompt_dir):
    with pytest.raises(AssertionError, match=r"available: \['short', 'think'\]"):
        pu.load_template(fake_prompt_dir, "missing")


def test_real_templates_are_valid(real_prompt_dir):
    assert pu.load_template(real_prompt_dir, "answer_only")["mode"] == "answer_only"
    assert pu.load_template(real_prompt_dir, "reasoning_answer")["mode"] == "reasoning"


@pytest.mark.parametrize("change, message", [
    ({"user": "{private_facts} {options} {unknown}"}, "unknown placeholders"),
    ({"user": "{options}"}, r"must use \{private_facts\}"),
    ({"empty_private_facts": ""}, "empty_private_facts"),
    ({"mode": "chat"}, "has mode"),
])
def test_invalid_templates_fail(change, message):
    template = {"name": "t", "mode": "answer_only", "system": "s", "user": "{private_facts} {options}",
                "empty_private_facts": "(none)", **change}
    with pytest.raises(AssertionError, match=message):
        pu.validate_template(template)


def test_template_without_empty_private_facts_key_fails():
    with pytest.raises(AssertionError, match="missing keys"):
        pu.validate_template({"name": "t", "mode": "answer_only", "system": "s", "user": "{private_facts} {options}"})


def test_reasoning_template_needs_marker():
    with pytest.raises(AssertionError, match="answer_marker"):
        pu.validate_template({"name": "t", "mode": "reasoning", "system": "s", "user": "{private_facts} {options}",
                              "empty_private_facts": "(none)"})


def test_mode_must_match_reasoning_flag(fake_prompt_dir):
    short, think = pu.load_template(fake_prompt_dir, "short"), pu.load_template(fake_prompt_dir, "think")
    pu.check_template_mode(short, reasoning=False)
    pu.check_template_mode(think, reasoning=True)
    with pytest.raises(AssertionError, match="--reasoning is on"):
        pu.check_template_mode(short, reasoning=True)
    with pytest.raises(AssertionError, match="--reasoning is off"):
        pu.check_template_mode(think, reasoning=False)


def test_render_with_zero_one_and_several_facts(real_prompt_dir):
    template = pu.load_template(real_prompt_dir, "answer_only")
    _, none_text = pu.render_prompt(template, fields([]))
    assert "Information only you know:\n(You have no private information for this task.)" in none_text
    _, one_text = pu.render_prompt(template, fields(["fact x"]))
    assert "Information only you know:\n- fact x" in one_text
    _, many_text = pu.render_prompt(template, fields(["fact x", "fact y"]))
    assert "- fact x\n- fact y" in many_text
    assert "A. Alpha\nB. Beta\nC. Gamma" in many_text and "(A, B or C)" in many_text
    assert "{" not in many_text


def test_rotation_helpers():
    assert pu.get_rotations(3, "none") == [0]
    assert pu.get_rotations(3, "cyclic") == [0, 1, 2]
    assert pu.get_option_order(3, 1) == [1, 2, 0]
    assert pu.unrotate_letter("A", 3, 1) == "B"
    assert pu.unrotate_letter("C", 3, 1) == "A"
    assert pu.unrotate_letter(None, 3, 1) is None
    assert pu.unrotate_scores({"A": -0.1, "B": -2.0, "C": -3.0}, 3, 1) == {"A": -3.0, "B": -0.1, "C": -2.0}
    with pytest.raises(AssertionError, match="rotation"):
        pu.get_option_order(3, 3)
