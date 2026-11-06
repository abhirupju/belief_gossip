"""Prompt templates (YAML files), rendering, and option-order rotation helpers."""

import string
from pathlib import Path
from typing import Any

import yaml


def get_known_placeholders() -> set[str]:
    """Placeholders a template may use."""
    return {"description", "shared_facts", "private_facts", "options", "letters"}


def get_template_modes() -> set[str]:
    """Allowed values of a template's `mode`."""
    return {"answer_only", "reasoning"}


def list_templates(prompt_dir: Path) -> list[str]:
    """Template names (file stems) in the folder."""
    prompt_dir = Path(prompt_dir)
    assert prompt_dir.is_dir(), f"list_templates: prompt folder {prompt_dir} does not exist"
    return sorted(p.stem for p in prompt_dir.glob("*.yaml"))


def load_template(prompt_dir: Path, name: str) -> dict[str, Any]:
    """Read and validate one template; keeps the raw file text under 'raw_text' for saving with results."""
    path = Path(prompt_dir) / f"{name}.yaml"
    assert path.is_file(), (
        f"load_template: template '{name}' not found in {prompt_dir}; available: {list_templates(prompt_dir)}"
    )
    raw_text = path.read_text(encoding="utf-8")
    template = yaml.safe_load(raw_text)
    assert isinstance(template, dict), f"load_template: {path} does not contain a YAML mapping"
    validate_template(template)
    assert template["name"] == name, (
        f"load_template: {path} declares name '{template['name']}' but the file is '{name}.yaml'; make them match"
    )
    template["raw_text"] = raw_text
    template["path"] = str(path)
    return template


def get_placeholders(text: str) -> set[str]:
    """Placeholder names used in a format string."""
    return {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}


def validate_template(template: dict[str, Any]) -> None:
    """Assert required keys, a known mode, only known placeholders, and the required placeholders."""
    required = ["name", "mode", "system", "user", "empty_private_facts"]
    missing = [key for key in required if key not in template]
    assert not missing, f"validate_template: template {template.get('name', '?')} is missing keys {missing}"
    assert template["mode"] in get_template_modes(), (
        f"validate_template: template {template['name']} has mode {template['mode']!r}; "
        f"expected one of {sorted(get_template_modes())}"
    )
    if template["mode"] == "reasoning":
        assert isinstance(template.get("answer_marker"), str) and template["answer_marker"].strip(), (
            f"validate_template: reasoning template {template['name']} needs a non-empty answer_marker"
        )
    assert isinstance(template["empty_private_facts"], str) and template["empty_private_facts"].strip(), (
        f"validate_template: template {template['name']} needs non-empty empty_private_facts text"
    )
    used = get_placeholders(template["system"]) | get_placeholders(template["user"])
    unknown = used - get_known_placeholders()
    assert not unknown, (
        f"validate_template: template {template['name']} uses unknown placeholders {sorted(unknown)}; "
        f"known: {sorted(get_known_placeholders())}"
    )
    for needed in ("private_facts", "options"):
        assert needed in get_placeholders(template["user"]), (
            f"validate_template: template {template['name']} must use {{{needed}}} in its user text"
        )


def check_template_mode(template: dict[str, Any], reasoning: bool) -> None:
    """Assert that --reasoning and the template's mode agree."""
    expected = "reasoning" if reasoning else "answer_only"
    assert template["mode"] == expected, (
        f"check_template_mode: --reasoning is {'on' if reasoning else 'off'} but template "
        f"'{template['name']}' has mode '{template['mode']}'; use a {expected} template"
    )


def format_bullets(items: list[str]) -> str:
    """'- a\\n- b'."""
    return "\n".join(f"- {item}" for item in items)


def format_letters(letters: list[str]) -> str:
    """'A, B or C'."""
    assert letters, "format_letters: no letters"
    return letters[0] if len(letters) == 1 else ", ".join(letters[:-1]) + f" or {letters[-1]}"


def render_prompt(template: dict[str, Any], fields: dict[str, Any]) -> tuple[str, str]:
    """Fill the template; returns (system_text, user_text). No private facts -> empty_private_facts text."""
    missing = get_known_placeholders() - set(fields)
    assert not missing, f"render_prompt: fields are missing {sorted(missing)}"
    assert len(fields["options"]) == len(fields["letters"]), (
        f"render_prompt: {len(fields['options'])} options but {len(fields['letters'])} letters"
    )
    values = {
        "description": fields["description"],
        "shared_facts": format_bullets(fields["shared_facts"]),
        "private_facts": format_bullets(fields["private_facts"]) if fields["private_facts"]
        else template["empty_private_facts"],
        "options": "\n".join(f"{l}. {o}" for l, o in zip(fields["letters"], fields["options"])),
        "letters": format_letters(fields["letters"]),
    }
    return template["system"].format(**values).strip(), template["user"].format(**values).strip()


def get_rotations(num_options: int, option_rotation: str) -> list[int]:
    """Rotation indices to run: [0] for 'none', 0..M-1 for 'cyclic'."""
    assert option_rotation in ("none", "cyclic"), (
        f"get_rotations: option_rotation is {option_rotation!r}; expected 'none' or 'cyclic'"
    )
    return [0] if option_rotation == "none" else list(range(num_options))


def get_option_order(num_options: int, rotation: int) -> list[int]:
    """Original option index shown at each displayed position for this cyclic rotation."""
    assert type(num_options) is int and num_options >= 1, f"get_option_order: bad num_options {num_options!r}"
    assert type(rotation) is int and 0 <= rotation < num_options, (
        f"get_option_order: rotation {rotation!r} must be an int in 0..{num_options - 1}"
    )
    return [(position + rotation) % num_options for position in range(num_options)]


def unrotate_letter(letter: str | None, num_options: int, rotation: int) -> str | None:
    """Map a displayed letter back to the letter in the dataset's own option order (None stays None)."""
    if letter is None:
        return None
    position = ord(letter) - ord("A")
    assert 0 <= position < num_options, f"unrotate_letter: {letter!r} is outside the {num_options} options"
    return chr(ord("A") + get_option_order(num_options, rotation)[position])


def unrotate_scores(scores: dict[str, float] | None, num_options: int, rotation: int) -> dict[str, float] | None:
    """Re-key per-letter scores from displayed letters to the dataset's own option order."""
    if scores is None:
        return None
    assert len(scores) == num_options, f"unrotate_scores: {len(scores)} scores for {num_options} options"
    return dict(sorted((unrotate_letter(letter, num_options, rotation), value) for letter, value in scores.items()))
