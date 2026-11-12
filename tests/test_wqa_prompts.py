"""Unit tests for utility/wqa_prompts.py and utility/wqa_original.py on the fake WearableQA source."""

import pytest

from utility import wqa_original as wo
from utility import wqa_prompts as wp
from utility import wqa_split as ws


def splits_of(raw):
    """Split of every phenotype question, by id."""
    return {q["id"]: ws.split_question(q, raw) for q in ws.select_questions(raw)}


def test_templates_load_and_match_the_holder_table(wqa_prompt_dir):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    assert set(templates["agents"]) == set(ws.default_holder_table())
    assert wp.option_letters(templates) == ["A", "B", "C", "D"]
    assert [o["code"] for o in templates["shared"]["options"]] == ws.core_classes()
    assert "watch:1" in wp.template_version(templates)


def test_nokey_overlay_replaces_keys(wqa_prompt_dir, wqa_nokey_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir, wqa_nokey_dir)
    assert all(t["key"] == "" for t in templates["agents"].values())
    split = splits_of(wqa_raw)["q1"]
    for agent in templates["agents"]:
        system, user = wp.render_agent(templates, agent, split)
        assert "CLINICAL KEY" not in user and "clinical key" not in (system + user).lower()
        assert "\n\n\n" not in user


def test_overlay_with_unknown_file_fails(tmp_path, wqa_prompt_dir):
    (tmp_path / "extra.yaml").write_text("name: extra\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="no base template"):
        wp.load_wqa_templates(wqa_prompt_dir, tmp_path)


def test_every_placeholder_is_filled(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    for split in splits_of(wqa_raw).values():
        for agent in templates["agents"]:
            for with_data in (True, False):
                system, user = wp.render_agent(templates, agent, split, with_data)
                assert "{" not in user and "}" not in user and "{" not in system
        system, user = wp.render_centralised(templates, split, list(templates["agents"]))
        assert "{" not in user and user.count("=== YOUR DATA") == len(templates["agents"])
        assert user.count("=== CLINICAL KEY") == len(templates["agents"])


def test_prior_uses_empty_data_and_keeps_the_key(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    split = splits_of(wqa_raw)["q1"]
    _, prior = wp.render_agent(templates, "watch", split, with_data=False)
    _, posterior = wp.render_agent(templates, "watch", split)
    assert templates["shared"]["empty_data"].strip() in prior
    assert "Resting heart rate:" not in prior and "Resting heart rate:" in posterior
    assert "CLINICAL KEY (watch)" in prior
    other = splits_of(wqa_raw)["q3"]
    assert wp.render_agent(templates, "watch", other, with_data=False) == wp.render_agent(templates, "watch", split,
                                                                                          with_data=False)


def test_not_measured_for_missing_metric_and_insulin(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    splits = splits_of(wqa_raw)
    _, phone = wp.render_agent(templates, "phone", splits["q4"])  # user u3 has no steps
    assert "Steps per day: not measured" in phone
    _, clinic = wp.render_agent(templates, "clinic_glycaemic", splits["q2"])  # no insulin
    assert "Fasting insulin: not measured" in clinic and "HOMA-IR (glucose x insulin / 405): not measured" in clinic
    _, clinic = wp.render_agent(templates, "clinic_glycaemic", splits["q1"])
    assert "HOMA-IR (glucose x insulin / 405): 1.05" in clinic  # 85 * 5 / 405
    assert "LDL" not in clinic and "tsh" not in clinic.lower()


def test_summary_uses_only_the_last_28_days(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    split = splits_of(wqa_raw)["q1"]
    fields = wp.agent_fields(templates, "watch", split)
    assert fields["rhr"].startswith("mean ") and "28/28 days" in fields["rhr"]
    assert "Population: p10=49.0, p25=54.0, p50=60.0, p75=66.0, p90=71.0" in fields["rhr"]
    assert fields["exercise"].startswith("14 sessions in 28 days")
    assert fields["vo2_max"] == "not measured"


def test_no_cohort_hides_population(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    split = splits_of(wqa_raw)["q8"]
    assert split["shared"]["cohort_reference"] == {}
    assert "Population" not in wp.agent_fields(templates, "watch", split)["rhr"]


def test_demographics_in_phone_and_both_clinics(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    split = splits_of(wqa_raw)["q3"]
    for agent in ("phone", "clinic_glycaemic", "clinic_lipid"):
        assert wp.agent_fields(templates, agent, split)["sex"] == "female"
    assert "sex" not in wp.agent_fields(templates, "watch", split)


@pytest.mark.parametrize("value, expected", [(40.0, "below p10"), (49.0, "between p10 and p25"),
                                             (60.0, "between p25 and p50"), (61.0, "between p50 and p75"),
                                             (71.0, "between p75 and p90"), (80.0, "above p90")])
def test_cohort_position_edges(value, expected):
    percentiles = {"p10": 49.0, "p25": 54.0, "p50": 60.0, "p75": 66.0, "p90": 71.0, "_n_users": 3}
    assert wp.cohort_position(value, percentiles) == expected


def test_metric_summary_and_formatting():
    assert wp.metric_summary([], "bpm", 1, 28, "not measured") == "not measured"
    text = wp.metric_summary([60.0, 62.0, 70.0], "bpm", 1, 28, "nm")
    assert text == "mean 64.0 bpm, median 62.0, range 60.0-70.0, 3/28 days"
    assert wp.metric_summary([12000.0], "steps/day", 0, 28, "nm") == "mean 12,000 steps/day, median 12,000, " \
                                                                     "range 12,000-12,000, 1/28 days"
    assert wp.with_unit("45.0", "%") == "45.0%"
    assert wp.homa_ir(100.0, 10.0) == 2.47 and wp.homa_ir(None, 3.0) is None


def test_daily_rows_ablation_lists_days(wqa_prompt_dir, wqa_raw):
    templates = wp.load_wqa_templates(wqa_prompt_dir)
    split = splits_of(wqa_raw)["q1"]
    _, user = wp.render_agent(templates, "watch", split, daily=True)
    assert "2024-01-31: rhr=" in user and "2024-01-03:" not in user  # window = Jan 4 .. Jan 31
    assert "mean " not in user.split("=== CLINICAL KEY")[0]
    _, clinic = wp.render_agent(templates, "clinic_lipid", split, daily=True)
    assert "Triglycerides: 90 mg/dL" in clinic


def test_bad_shared_template_fails(tmp_path, wqa_prompt_dir):
    text = (wqa_prompt_dir / "shared.yaml").read_text(encoding="utf-8").replace("code: D-FA", "code: XX")
    (tmp_path / "shared.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(AssertionError, match="options are"):
        wp.load_wqa_templates(wqa_prompt_dir, tmp_path)


def test_original_prompt_keeps_ten_options_and_full_panel(wqa_raw):
    q = ws.select_questions(wqa_raw)[0]
    rows = ws.question_rows(q, wqa_raw)
    system, user = wo.render_original(q, rows, wqa_raw["cohort_reference"])
    assert system == ""
    assert user.count("\n", user.index("=== OPTIONS ===")) == 10
    assert "ldl: 100.0" in user and "tsh: 2.00" in user
    assert "RHR: mean" in user and "/28 days" in user and "2024-01-31" not in user
    assert "_n_users" not in user and "rhr: p10=49.0" in user
    letter_map = wo.original_core_letter_map(q)
    assert sorted(letter_map.values()) == ["A", "B", "C", "D"] and len(letter_map) == 4
    assert letter_map[q["gt_letter"]] == ws.fixed_gold_letter(q)
    _, prior = wo.render_original(q, rows, wqa_raw["cohort_reference"], with_data=False)
    assert "BLOOD" not in prior and "=== OPTIONS ===" in prior


def test_original_without_cohort_drops_the_phrase(wqa_raw):
    q = ws.select_questions(wqa_raw)[7]
    _, user = wo.render_original(q, ws.question_rows(q, wqa_raw), wqa_raw["cohort_reference"])
    assert "cohort reference" not in user.lower()
