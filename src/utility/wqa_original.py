"""The original WearableQA prompt wording (E5 baseline), with the sensor data as 28-day summary statistics.

Re-implemented from reading the official renderer (`render_raw.py`, function `render_question`); that downloaded
script is never run. Kept from the original: the instruction, the profile line, the full blood panel (no units, as
in the original), the cohort reference block, the question wording and all ten options in their original order.
Changed, as agreed for E5: only the last 28 days, and one summary line per metric instead of one line per day.
"""

from typing import Any

from utility import wqa_split as ws
from utility.wqa_prompts import metric_decimals, metric_summary, metric_values, window_rows


def original_instruction(with_cohort: bool) -> str:
    """The original instruction text (the cohort phrase is dropped when the cohort reference is withheld)."""
    text = ("You are given a user's demographics, wearable health sensor history, blood biomarker panel, a cohort "
            "reference distribution, and a multiple-choice question. Analyze the data carefully and select the "
            "single best answer.")
    return text if with_cohort else text.replace(", a cohort reference distribution,", ",")


def original_field_labels() -> dict[str, str]:
    """Display labels used by the original renderer."""
    return {"steps": "Steps", "rhr": "RHR", "active_burn": "Active Cal", "sleep_duration": "Sleep(h)",
            "sleep_efficiency": "Sleep Eff", "deep_percentage": "Deep%", "rem_percentage": "REM%", "hrv": "HRV",
            "avg_stress_level": "Stress", "exercise_mets": "Ex METs", "exercise_hours": "Ex Hours",
            "exercise_avg_hr": "Ex Avg HR", "exercise_count": "Ex Count", "vo2_max": "VO2max",
            "bed_time_std": "Bedtime Std(min)", "calories_bmr": "BMR Cal"}


def original_format_value(value: Any) -> str | None:
    """The original renderer's number format (3 / 2 / 1 / 0 decimals by magnitude); None for missing or NaN."""
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:
            return None
        magnitude = abs(value)
        if magnitude < 1:
            return f"{value:.3f}"
        if magnitude < 10:
            return f"{value:.2f}"
        if magnitude < 1000:
            return f"{value:.1f}"
        return f"{value:.0f}"
    return str(value)


def original_hidden_cohort_keys() -> set[str]:
    """Cohort-reference keys the original renderer leaves out."""
    return {"_n_users", "_median_days_per_user"}


def original_question_text(stem: str, window_size: int) -> str:
    """The question line as the original renderer words it."""
    low = stem.lower()
    if any(p in low for p in ("based on", "-day window", "day window", "recent days", "days of data")):
        return stem
    if "Based on this user" in stem:
        return stem.replace("Based on this user's", "Based on this user's recent")
    return f"Based on the most recent {window_size} days of data, {stem[0].lower()}{stem[1:]}"


def original_letters(question: dict[str, Any]) -> list[str]:
    """The question's option letters in order (A..J)."""
    return [letter for letter in "ABCDEFGHIJ" if letter in question["options"]]


def original_core_letter_map(question: dict[str, Any]) -> dict[str, str]:
    """Original option letter -> fixed letter (A-D), for the four gold-capable classes only."""
    fixed = ws.fixed_letter_map()
    codes = ws.option_classes(question)
    return {letter: fixed[code] for letter, code in codes.items() if code in fixed}


def sensor_summary_block(rows: list[dict[str, Any]], end_date: str, window_days: int) -> str:
    """One summary line per metric present in the last `window_days` days."""
    rows = window_rows(rows, end_date, window_days)
    labels, decimals = original_field_labels(), metric_decimals()
    lines = []
    for metric in sorted(labels):
        values = metric_values(rows, metric)
        if values:
            lines.append(f"  {labels[metric]}: {metric_summary(values, '', decimals[metric], window_days, '')}")
    if not lines:
        return "=== SENSOR DATA ===\n  (no wearable time-series available)"
    return f"=== SENSOR DATA (summary of the last {window_days} days) ===\n" + "\n".join(lines)


def render_original(question: dict[str, Any], rows: list[dict[str, Any]], cohort: dict[str, Any],
                    with_data: bool = True, window_days: int = 28) -> tuple[str, str]:
    """(system, user): system is empty, as in the original one-message prompt; with_data=False gives the prior."""
    context = question.get("context") or {}
    demo = context.get("demographics") or {}
    blood = context.get("blood_panel") or {}
    coh = {} if question.get("no_cohort") else (cohort or {})
    parts = [original_instruction(bool(coh)) + "\n"]
    if with_data:
        profile = []
        if demo.get("age"):
            profile.append(f"Age: {demo['age']:.0f}")
        if demo.get("sex"):
            profile.append(f"Sex: {demo['sex']}")
        if demo.get("bmi"):
            profile.append(f"BMI: {demo['bmi']:.1f}")
        if demo.get("ethnicity"):
            profile.append(f"Ethnicity: {demo['ethnicity']}")
        if profile:
            parts += ["=== USER PROFILE ===", "  " + ", ".join(profile) + "\n"]
        parts.append(sensor_summary_block(rows, question["end_date"], window_days))
        if blood:
            parts.append("\n=== BLOOD BIOMARKER PANEL ===")
            parts.append(",".join(f" {k}: {original_format_value(v)}" for k, v in blood.items()))
        if coh:
            parts.append("\n=== COHORT REFERENCE (population percentiles) ===")
            for metric, pcs in coh.items():
                if isinstance(pcs, dict):
                    shown = ", ".join(f"{k}={v}" for k, v in pcs.items() if k not in original_hidden_cohort_keys())
                    parts.append(f"  {metric}: {shown}")
    else:
        parts.append("(No data available for this question.)")
    parts.append(f"\n=== QUESTION ===\n{original_question_text(question['stem'], window_days)}\n")
    parts.append("=== OPTIONS ===")
    parts += [f"{letter}. {question['options'][letter]}" for letter in original_letters(question)]
    return "", "\n".join(parts)
