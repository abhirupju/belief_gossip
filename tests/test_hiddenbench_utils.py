"""Dataset conversion, lazy loading, validation, options and prompt fields (fake data only)."""

import json

import pytest

from utility import hiddenbench_utils as hb


def test_round_trip_loads_each_task(fake_data_dir, fake_tasks):
    assert hb.is_prepared(fake_data_dir)
    assert hb.get_num_samples(fake_data_dir) == 3
    for index, task in enumerate(fake_tasks):
        assert hb.load_sample(index, fake_data_dir) == task


@pytest.mark.parametrize("bad_index, message", [
    (-1, r"out of range; HiddenBench has 3 samples \(valid indices 0\.\.2\)"),
    (3, r"out of range; HiddenBench has 3 samples \(valid indices 0\.\.2\)"),
    (1.0, "must be an int"),
    (True, "must be an int"),
    ("2", "must be an int"),
])
def test_invalid_index_raises_with_clear_message(fake_data_dir, bad_index, message):
    with pytest.raises(AssertionError, match=message):
        hb.load_sample(bad_index, fake_data_dir)


def test_lazy_loading_reads_only_the_requested_line(fake_data_dir, fake_tasks):
    """Corrupt every other line (same byte length, so offsets stay valid): task 1 still loads."""
    jsonl = hb.dataset_paths(fake_data_dir)["jsonl"]
    lines = jsonl.read_bytes().split(b"\n")
    corrupted = [line if i == 1 or not line else b"#" * len(line) for i, line in enumerate(lines)]
    jsonl.write_bytes(b"\n".join(corrupted))
    assert hb.load_sample(1, fake_data_dir) == fake_tasks[1]


def test_missing_dataset_says_how_to_fix(tmp_path):
    with pytest.raises(AssertionError, match="download_hiddenbench"):
        hb.load_sample(0, tmp_path / "nothing")


def test_json_lines_raw_file_also_converts(tmp_path, fake_tasks):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(t) for t in fake_tasks), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert hb.convert_json_to_jsonl(raw, out) == 3
    assert hb.build_offset_index(out, tmp_path / "offsets.json") == 3


def test_validate_rejects_answer_not_in_options(fake_tasks):
    task = dict(fake_tasks[0], correct_answer="Delta")
    with pytest.raises(AssertionError, match="is not one of possible_answers"):
        hb.validate_sample(task)


def test_validate_rejects_missing_fields(fake_tasks):
    task = {k: v for k, v in fake_tasks[0].items() if k != "hidden_information"}
    with pytest.raises(AssertionError, match="missing fields"):
        hb.validate_sample(task)


def test_letters_formatting_and_mapping(fake_tasks):
    task = fake_tasks[2]
    assert hb.get_option_letters(task) == ["A", "B", "C", "D"]
    assert hb.format_options(task) == "A. One\nB. Two\nC. Three\nD. Four"
    assert hb.format_options(task, rotation=1) == "A. Two\nB. Three\nC. Four\nD. One"
    assert hb.get_true_letter(task) == "C"
    assert hb.letter_to_answer(task, "D") == "Four"
    with pytest.raises(AssertionError, match="valid letters"):
        hb.letter_to_answer(task, "E")


def test_prompt_fields_give_only_the_agents_facts(fake_tasks):
    fields = hb.build_prompt_fields(fake_tasks[1], [1, 2], rotation=0)
    assert fields["private_facts"] == ["fact one of task two", "fact two of task two"]
    assert fields["shared_facts"] == ["Route info."]
    assert fields["letters"] == ["A", "B", "C"]
    with pytest.raises(AssertionError, match="fact indices"):
        hb.build_prompt_fields(fake_tasks[0], [5])
