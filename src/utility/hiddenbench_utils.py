"""HiddenBench access: one-time download and conversion, then lazy one-task-at-a-time loading.

The dataset is stored as JSONL plus a byte-offset index, so load_sample(i) is one seek and one
readline and only task i is ever held in memory.
"""

import json
from pathlib import Path
from typing import Any

from utility.prompt_utils import get_option_order


def dataset_paths(data_dir: Path) -> dict[str, Path]:
    """Return the raw folder, JSONL file and offset-index file for HiddenBench under data_dir."""
    assert data_dir is not None, "dataset_paths: data_dir is None; pass the data folder (e.g. code/data)"
    root = Path(data_dir) / "hiddenbench"
    return {
        "root": root,
        "raw_dir": root / "raw",
        "jsonl": root / "hiddenbench.jsonl",
        "offsets": root / "hiddenbench.offsets.json",
    }


def is_prepared(data_dir: Path) -> bool:
    """True when the JSONL file and its offset index both exist."""
    paths = dataset_paths(data_dir)
    return paths["jsonl"].is_file() and paths["offsets"].is_file()


def download_hiddenbench(data_dir: Path, repo_id: str = "YuxuanLi1225/HiddenBench") -> Path:
    """Download HiddenBench once, convert it to JSONL and build the offset index; return the JSONL path."""
    paths = dataset_paths(data_dir)
    if is_prepared(data_dir):
        return paths["jsonl"]
    from huggingface_hub import snapshot_download

    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(paths["raw_dir"]))
    raw_path = find_raw_json(paths["raw_dir"])
    convert_json_to_jsonl(raw_path, paths["jsonl"])
    build_offset_index(paths["jsonl"], paths["offsets"])
    return paths["jsonl"]


def find_raw_json(raw_dir: Path) -> Path:
    """Return the single .json/.jsonl data file inside the downloaded snapshot."""
    candidates = sorted(
        p for p in Path(raw_dir).rglob("*")
        if p.suffix in (".json", ".jsonl") and ".cache" not in p.parts and p.is_file()
    )
    assert len(candidates) == 1, (
        f"find_raw_json: expected exactly one .json/.jsonl file in {raw_dir}, found {len(candidates)}: "
        f"{[str(c) for c in candidates]}"
    )
    return candidates[0]


def first_non_space_char(path: Path) -> str:
    """Read characters until the first non-whitespace one (to tell a JSON array from JSON lines)."""
    with open(path, "r", encoding="utf-8") as handle:
        char = handle.read(1)
        while char and char.isspace():
            char = handle.read(1)
    assert char, f"first_non_space_char: {path} is empty"
    return char


def convert_json_to_jsonl(raw_path: Path, jsonl_path: Path) -> int:
    """Stream the raw file item by item (never fully loaded) and write one task per line; return the count."""
    import ijson

    first = first_non_space_char(raw_path)
    assert first in ("[", "{"), (
        f"convert_json_to_jsonl: {raw_path} starts with {first!r}; expected a JSON array '[' or JSON lines '{{'"
    )
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(raw_path, "rb") as source, open(jsonl_path, "w", encoding="utf-8") as target:
        items = ijson.items(source, "item", use_float=True) if first == "[" else ijson.items(
            source, "", multiple_values=True, use_float=True
        )
        for item in items:
            assert isinstance(item, dict), (
                f"convert_json_to_jsonl: item {count} in {raw_path} is a {type(item).__name__}, expected an object"
            )
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    assert count > 0, f"convert_json_to_jsonl: no tasks found in {raw_path}"
    return count


def build_offset_index(jsonl_path: Path, offsets_path: Path) -> int:
    """Record the byte offset of every line of the JSONL file; return the number of lines."""
    offsets = []
    position = 0
    with open(jsonl_path, "rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(position)
            position += len(line)
    assert offsets, f"build_offset_index: {jsonl_path} has no non-empty lines"
    Path(offsets_path).write_text(json.dumps({"num_samples": len(offsets), "offsets": offsets}), encoding="utf-8")
    return len(offsets)


def read_offsets(data_dir: Path) -> list[int]:
    """Read the offset index (one integer per task: metadata, not data)."""
    paths = dataset_paths(data_dir)
    assert paths["offsets"].is_file(), (
        f"read_offsets: {paths['offsets']} not found; run download_hiddenbench(data_dir) first "
        f"(or main.py --download_only)"
    )
    index = json.loads(paths["offsets"].read_text(encoding="utf-8"))
    assert index["num_samples"] == len(index["offsets"]), (
        f"read_offsets: corrupt index {paths['offsets']}: num_samples={index['num_samples']} "
        f"but {len(index['offsets'])} offsets"
    )
    return index["offsets"]


def get_num_samples(data_dir: Path) -> int:
    """Number of tasks, read from the offset index."""
    return len(read_offsets(data_dir))


def load_sample(index: int, data_dir: Path) -> dict[str, Any]:
    """Load task `index` (0-based) only: one seek and one readline."""
    assert type(index) is int, (
        f"load_sample: index must be an int, got {type(index).__name__} ({index!r})"
    )
    offsets = read_offsets(data_dir)
    assert 0 <= index < len(offsets), (
        f"load_sample: index {index} is out of range; HiddenBench has {len(offsets)} samples "
        f"(valid indices 0..{len(offsets) - 1})"
    )
    with open(dataset_paths(data_dir)["jsonl"], "rb") as handle:
        handle.seek(offsets[index])
        line = handle.readline()
    sample = json.loads(line.decode("utf-8"))
    validate_sample(sample)
    return sample


def validate_sample(sample: dict[str, Any]) -> None:
    """Assert required fields and types; correct_answer must be one of possible_answers."""
    assert isinstance(sample, dict), f"validate_sample: sample is a {type(sample).__name__}, expected dict"
    required = ["id", "name", "description", "shared_information", "hidden_information",
                "possible_answers", "correct_answer"]
    missing = [key for key in required if key not in sample]
    assert not missing, f"validate_sample: task {sample.get('id', '?')} is missing fields {missing}"
    assert isinstance(sample["description"], str), f"validate_sample: task {sample['id']} description is not a string"
    for key in ("shared_information", "hidden_information", "possible_answers"):
        assert isinstance(sample[key], list) and all(isinstance(x, str) for x in sample[key]), (
            f"validate_sample: task {sample['id']} field {key} must be a list of strings"
        )
    assert len(sample["hidden_information"]) >= 1, f"validate_sample: task {sample['id']} has no hidden facts"
    assert 2 <= len(sample["possible_answers"]) <= 26, (
        f"validate_sample: task {sample['id']} has {len(sample['possible_answers'])} options; expected 2..26"
    )
    assert sample["correct_answer"] in sample["possible_answers"], (
        f"validate_sample: task {sample['id']} correct_answer {sample['correct_answer']!r} is not one of "
        f"possible_answers {sample['possible_answers']}"
    )


def get_private_facts(sample: dict[str, Any]) -> list[str]:
    """The task's hidden facts, in dataset order (split among agents by partition_facts)."""
    return list(sample["hidden_information"])


def get_num_private_facts(sample: dict[str, Any]) -> int:
    """Number of hidden facts (F) in the task."""
    return len(sample["hidden_information"])


def get_shared_facts(sample: dict[str, Any]) -> list[str]:
    """The facts every agent sees."""
    return list(sample["shared_information"])


def get_num_options(sample: dict[str, Any]) -> int:
    """Number of answer options."""
    return len(sample["possible_answers"])


def get_option_letters(sample: dict[str, Any]) -> list[str]:
    """['A', 'B', 'C'] (or more) for this task."""
    return [chr(ord("A") + i) for i in range(get_num_options(sample))]


def get_displayed_options(sample: dict[str, Any], rotation: int = 0) -> list[str]:
    """Option texts in the order shown to the agent for this rotation."""
    order = get_option_order(get_num_options(sample), rotation)
    return [sample["possible_answers"][i] for i in order]


def format_options(sample: dict[str, Any], rotation: int = 0) -> str:
    """'A. West City\\nB. East Town\\nC. North Hill' (in the rotated order)."""
    letters = get_option_letters(sample)
    return "\n".join(f"{letter}. {text}" for letter, text in zip(letters, get_displayed_options(sample, rotation)))


def get_true_letter(sample: dict[str, Any]) -> str:
    """Letter of correct_answer in the dataset's own option order."""
    return get_option_letters(sample)[sample["possible_answers"].index(sample["correct_answer"])]


def letter_to_answer(sample: dict[str, Any], letter: str) -> str:
    """Option text for a letter in the dataset's own option order."""
    letters = get_option_letters(sample)
    assert letter in letters, (
        f"letter_to_answer: '{letter}' is not an option of task {sample['id']}; valid letters are {letters}"
    )
    return sample["possible_answers"][letters.index(letter)]


def build_prompt_fields(sample: dict[str, Any], fact_indices: list[int], rotation: int = 0) -> dict[str, Any]:
    """Values a template fills for one agent: description, shared facts, its private facts, options, letters."""
    facts = get_private_facts(sample)
    assert all(type(i) is int and 0 <= i < len(facts) for i in fact_indices), (
        f"build_prompt_fields: fact indices {fact_indices} are invalid for task {sample['id']} "
        f"with {len(facts)} hidden facts"
    )
    return {
        "description": sample["description"],
        "shared_facts": get_shared_facts(sample),
        "private_facts": [facts[i] for i in fact_indices],
        "options": get_displayed_options(sample, rotation),
        "letters": get_option_letters(sample),
    }
