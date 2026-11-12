"""Unit tests for utility/wqa_split.py on a small fake WearableQA source (no real data, no model)."""

import json
import math

import pytest

from utility import wqa_split as ws


def fake_options(gold_code: str) -> tuple[dict[str, str], str]:
    """Ten phenotype options (one per class) and the letter of the given gold class."""
    texts = [prefix + " — fake detail" for prefix, _ in ws.phenotype_option_rules()]
    codes = [code for _, code in ws.phenotype_option_rules()]
    letters = "ABCDEFGHIJ"
    options = {letters[i]: texts[(i + 3) % 10] for i in range(10)}  # shuffled order
    gold_letter = next(letter for letter, text in options.items() if ws.option_class(text) == gold_code)
    assert codes  # rules non-empty
    return options, gold_letter


def fake_raw() -> dict:
    """Two users, three questions; uses metric aliases, masks, zero exercise values and missing days."""
    hist_u1 = []
    for day in range(1, 31):
        row = {"date": f"2024-01-{day:02d}", "resting_hr": 60.0 + day % 3, "steps": 8000.0 + day,
               "sleep_duration": 7.0, "hrv": 40.0, "exercise_mets": 0.0, "avg_stress_level": -1.0, "_source": "x"}
        if day % 5 == 0:
            row.pop("steps")
        hist_u1.append(row)
    hist_u2 = [{"date": f"2024-01-{day:02d}", "rhr": 70.0, "vo2Max": 30.0} for day in range(10, 31)]
    opts_ch, gold_ch = fake_options("CH")
    opts_ca, gold_ca = fake_options("CA")
    blood = {"hgba1c": 5.2, "glu": 85.0, "tg": 90.0, "hdl": 60.0, "ins": 5.0}
    bad_blood = {"hgba1c": 5.9, "glu": 120.0, "tg": 220.0, "hdl": 35.0, "ins": 20.0}
    demo = {"age": 40.0, "sex": "m", "bmi": 25.0, "ethnicity": "white"}
    q1 = {"id": "q1", "user_id": 1, "question_type": "multimodal_phenotype", "stem": "Which phenotype?",
          "options": opts_ch, "gt_letter": gold_ch, "reasoning_group": "health", "signal": "cross",
          "grounding": "literature", "context": {"demographics": demo, "blood_panel": blood},
          "end_date": "2024-01-30", "window_size": 28, "full_dropped_metrics": [], "no_cohort": False}
    q2 = dict(q1, id="q2", full_dropped_metrics=["hrv"], hidden_metric="sleep_duration")
    q3 = dict(q1, id="q3", user_id=2, options=opts_ca, gt_letter=gold_ca,
              context={"demographics": dict(demo, sex="f"), "blood_panel": bad_blood}, no_cohort=True)
    q4 = dict(q1, id="q4", question_type="signal_summary")
    return {"n_mcqs": 4, "mcqs": [q1, q2, q3, q4], "user_histories": {"1": hist_u1, "2": hist_u2},
            "cohort_reference": {"rhr": {"p50": 60.0}}}


def test_load_raw_roundtrip(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(fake_raw()))
    raw = ws.load_raw(path)
    assert len(raw["mcqs"]) == 4


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(AssertionError):
        ws.load_raw(tmp_path / "nope.json")


def test_select_questions_filters_type():
    qs = ws.select_questions(fake_raw())
    assert [q["id"] for q in qs] == ["q1", "q2", "q3"]


def test_option_classes_and_gold():
    raw = fake_raw()
    q1, _, q3 = ws.select_questions(raw)
    assert ws.gold_class(q1) == "CH"
    assert ws.gold_class(q3) == "CA"
    assert sorted(ws.option_classes(q1).values()) == sorted(ws.phenotype_class_names())


def test_option_class_rejects_unknown_text():
    with pytest.raises(AssertionError):
        ws.option_class("something else entirely")


def test_default_holder_table_is_valid():
    ws.validate_holder_table(ws.default_holder_table())


def test_holder_table_with_missing_metric_fails():
    holders = ws.default_holder_table()
    holders["watch"]["metrics"].remove("rhr")
    with pytest.raises(AssertionError):
        ws.validate_holder_table(holders)


def test_holder_table_with_duplicate_metric_fails():
    holders = ws.default_holder_table()
    holders["phone"]["metrics"].append("rhr")
    with pytest.raises(AssertionError):
        ws.validate_holder_table(holders)


def test_question_rows_normalises_and_masks():
    raw = fake_raw()
    q1, q2, q3 = ws.select_questions(raw)
    rows = ws.question_rows(q1, raw)
    assert rows[0]["date"] == "2024-01-02"  # window is end - 28 days .. end
    assert len(rows) == 29
    assert all("rhr" in r and "resting_hr" not in r for r in rows)
    assert all("exercise_mets" not in r and "avg_stress_level" not in r for r in rows)  # zero / non-positive
    assert all("_source" not in r for r in rows)
    rows2 = ws.question_rows(q2, raw)
    assert all("hrv" not in r and "sleep_duration" not in r for r in rows2)
    rows3 = ws.question_rows(q3, raw)
    assert all(r["vo2_max"] == 30.0 for r in rows3)


def test_split_partitions_every_value():
    raw = fake_raw()
    for q in ws.select_questions(raw):
        split = ws.split_question(q, raw)
        ws.check_partition(q, split, raw)
        assert set(split["agents"]) == {"watch", "phone", "ring", "clinic_glycaemic", "clinic_lipid"}
        glyc = split["agents"]["clinic_glycaemic"]["context"]["blood_panel"]
        lipid = split["agents"]["clinic_lipid"]["context"]["blood_panel"]
        assert set(glyc) <= {"glu", "hgba1c", "ins"} and set(lipid) <= {"tg", "hdl"}
        for name in ("phone", "clinic_glycaemic", "clinic_lipid"):
            assert "demographics" in split["agents"][name]["context"]
        assert split["agents"]["clinic_glycaemic"]["rows"] == []


def test_split_history_days_extends_window():
    raw = fake_raw()
    q1 = ws.select_questions(raw)[0]
    short = ws.split_question(q1, raw)
    long = ws.split_question(q1, raw, history_days=10)
    ws.check_partition(q1, long, raw)
    assert len(long["agents"]["watch"]["rows"]) > len(short["agents"]["watch"]["rows"])


def test_split_respects_no_cohort():
    raw = fake_raw()
    _, _, q3 = ws.select_questions(raw)
    assert ws.split_question(q3, raw)["shared"]["cohort_reference"] == {}


def test_agent_summary_and_render():
    raw = fake_raw()
    q1 = ws.select_questions(raw)[0]
    split = ws.split_question(q1, raw)
    summary = ws.agent_summary(split["agents"]["phone"])
    assert summary["steps"]["n_days"] == 23  # days 2..30 minus multiples of 5
    assert math.isnan(summary["active_burn"]["mean"])
    text = ws.render_agent_text(split["agents"]["phone"], max_rows=2)
    assert "PHONE" in text and "demographics" in text and "earlier days not shown" in text


def test_lab_flags():
    raw = fake_raw()
    q1, _, q3 = ws.select_questions(raw)
    assert sum(ws.lab_flags(q1["context"]["blood_panel"], "m").values()) == 0
    assert sum(ws.lab_flags(q3["context"]["blood_panel"], "f").values()) == 5


def test_feature_rows():
    raw = fake_raw()
    rows = ws.feature_rows(ws.select_questions(raw), raw)
    assert [r["gold_class"] for r in rows] == ["CH", "CH", "CA"]
    assert rows[0]["n_steps"] == 23 and rows[2]["n_lab_flags"] == 5


def test_holder_table_with_doubled_marker_fails():
    holders = ws.default_holder_table()
    holders["clinic_lipid"]["blood_markers"].append("glu")
    with pytest.raises(AssertionError, match="more than one holder"):
        ws.validate_holder_table(holders)


def test_holder_table_with_whole_panel_and_markers_fails():
    holders = ws.default_holder_table()
    holders["ring"]["context"].append("blood_panel")
    with pytest.raises(AssertionError, match="not both"):
        ws.validate_holder_table(holders)


def test_whole_blood_panel_holder_still_works():
    raw = fake_raw()
    holders = ws.default_holder_table()
    del holders["clinic_lipid"]
    holders["clinic_glycaemic"] = {"metrics": [], "context": ["blood_panel"], "blood_markers": []}
    for q in ws.select_questions(raw):
        split = ws.split_question(q, raw, holders)
        ws.check_partition(q, split, raw)
        assert split["agents"]["clinic_glycaemic"]["context"]["blood_panel"] == q["context"]["blood_panel"]
        assert "demographics" not in split["agents"]["clinic_glycaemic"]["context"]


def test_check_partition_catches_a_leaked_marker():
    raw = fake_raw()
    q = ws.select_questions(raw)[0]
    split = ws.split_question(q, raw)
    split["agents"]["clinic_lipid"]["context"]["blood_panel"]["glu"] = q["context"]["blood_panel"]["glu"]
    with pytest.raises(AssertionError, match="does not own"):
        ws.check_partition(q, split, raw)
