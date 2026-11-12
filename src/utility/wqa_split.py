"""WearableQA multimodal-phenotype questions: load, label, and split each question's data into per-device private views.

Source: WearableQA (Lee et al., arXiv 2609.05405), file `WearableQA_raw.json` from
https://github.com/facebookresearch/WearableQA. Licence: CC BY-NC 4.0 with extra terms (no re-identification, no
clinical use, research and benchmarking only).

A question's data has three parts:
- the user's daily wearable history (`raw["user_histories"][user_id]`, one dict per day, 16 metrics, sparse);
- the question's context: `demographics` and `blood_panel`;
- the shared cohort reference percentiles (`raw["cohort_reference"]`), withheld when `no_cohort` is true.

`split_question` gives each holder (watch, phone, ring, glycaemic clinic, lipid clinic in the default table) its
private view: every metric goes to exactly one holder; a context block goes to one holder, except the blocks in
`shared_context_blocks()` (demographics), which several holders may copy; the blood panel can be split by marker
(`blood_markers`), and markers in no holder are left out. The question's own masks are applied first
(`full_dropped_metrics`, `hidden_metric`, `window_dropped_metrics`). It returns one view per holder plus the shared
part. The window follows the official renderer: the days from `end_date - (window_days + history_days)` to
`end_date`, inclusive (the prompts in `wqa_prompts` then show the last `window_days` of them).
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------------------------- loading


def default_raw_path(code_dir: Path) -> Path:
    """Where the notebook expects the structured source: code/data/wearableqa/WearableQA_raw.json."""
    return Path(code_dir) / "data" / "wearableqa" / "WearableQA_raw.json"


def load_raw(path: Path) -> dict[str, Any]:
    """Load WearableQA_raw.json and check its top-level structure."""
    path = Path(path)
    assert path.exists(), (
        f"load_raw: {path} not found; download WearableQA_raw.json from "
        f"https://media.githubusercontent.com/media/facebookresearch/WearableQA/main/WearableQA_raw.json"
    )
    with open(path) as f:
        raw = json.load(f)
    for key in ("mcqs", "user_histories", "cohort_reference", "n_mcqs"):
        assert key in raw, f"load_raw: key '{key}' missing from {path}; expected the WearableQA structured source"
    assert len(raw["mcqs"]) == raw["n_mcqs"], (
        f"load_raw: {len(raw['mcqs'])} questions but n_mcqs says {raw['n_mcqs']}"
    )
    return raw


def select_questions(raw: dict[str, Any], question_type: str = "multimodal_phenotype") -> list[dict[str, Any]]:
    """All questions of one type, in file order."""
    questions = [q for q in raw["mcqs"] if q["question_type"] == question_type]
    assert questions, f"select_questions: no questions of type '{question_type}'"
    for q in questions:
        assert str(q["user_id"]) in raw["user_histories"], (
            f"select_questions: question {q['id']} has user {q['user_id']} with no history"
        )
        assert q["gt_letter"] in q["options"], f"select_questions: question {q['id']} gold letter not in options"
    return questions


# ----------------------------------------------------------------------------------------------- metrics


def metric_aliases() -> dict[str, str]:
    """Alternative metric names in the raw histories mapped to the canonical name."""
    return {"resting_hr": "rhr", "active_kcal": "active_burn", "vo2Max": "vo2_max"}


def all_metrics() -> list[str]:
    """The 16 canonical daily wearable metrics."""
    return [
        "steps", "active_burn", "calories_bmr",
        "rhr", "hrv", "vo2_max",
        "exercise_count", "exercise_hours", "exercise_mets", "exercise_avg_hr", "avg_stress_level",
        "sleep_duration", "sleep_efficiency", "deep_percentage", "rem_percentage", "bed_time_std",
    ]


def canonical_metric(name: str) -> str:
    """Canonical metric name for a raw key."""
    return metric_aliases().get(name, name)


def normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """One history day with canonical metric names; drops bookkeeping keys and missing values."""
    assert "date" in row, f"normalise_row: row without a date: {row}"
    out: dict[str, Any] = {"date": row["date"]}
    for key, value in row.items():
        if key in ("date", "_source", "_period") or value is None:
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        name = canonical_metric(key)
        assert name in all_metrics(), f"normalise_row: unknown metric '{key}' on {row['date']}"
        out[name] = value
    return out


# ----------------------------------------------------------------------------------------------- classes


def phenotype_option_rules() -> list[tuple[str, str]]:
    """(option-text prefix, class code) for the 10 multimodal-phenotype options."""
    return [
        ("concordant healthy phenotype", "CH"),
        ("concordant adverse phenotype", "CA"),
        ("discordant phenotype — low wearable", "D-LF"),
        ("discordant phenotype — favourable wearable", "D-FA"),
        ("intermediate phenotype", "INT"),
        ("established type 2 diabetes", "T2D"),
        ("acute infection phenotype", "INF"),
        ("athletic elite-endurance", "ATH"),
        ("the phenotype is attributable entirely to body-mass index", "BMI"),
        ("the wearable and blood streams measure unrelated systems", "UNR"),
    ]


def phenotype_class_names() -> dict[str, str]:
    """Class code to a short readable name."""
    return {
        "CH": "concordant healthy (fit + good labs)",
        "CA": "concordant adverse (low fitness + bad labs)",
        "D-LF": "discordant (low fitness + good labs)",
        "D-FA": "discordant (fit + bad labs)",
        "INT": "intermediate (both near cohort middle)",
        "T2D": "established type 2 diabetes",
        "INF": "acute infection",
        "ATH": "elite endurance athlete",
        "BMI": "attributable to BMI alone",
        "UNR": "streams unrelated",
    }


def core_classes() -> list[str]:
    """The four 2 x 2 classes (fitness axis x lab axis); every gold answer is one of these."""
    return ["CH", "CA", "D-LF", "D-FA"]


def option_class(text: str) -> str:
    """Class code of one option text."""
    lowered = text.strip().lower()
    matches = [code for prefix, code in phenotype_option_rules() if lowered.startswith(prefix.lower())]
    assert len(matches) == 1, f"option_class: option text matches {matches}, expected exactly one class: {text!r}"
    return matches[0]


def option_classes(question: dict[str, Any]) -> dict[str, str]:
    """Option letter to class code for one question; the 10 options must cover the 10 classes once each."""
    mapping = {letter: option_class(text) for letter, text in question["options"].items()}
    assert sorted(mapping.values()) == sorted(phenotype_class_names()), (
        f"option_classes: question {question['id']} options map to {sorted(mapping.values())}"
    )
    return mapping


def gold_class(question: dict[str, Any]) -> str:
    """Class code of the gold answer."""
    return option_classes(question)[question["gt_letter"]]


def fixed_letter_map() -> dict[str, str]:
    """The fixed option letter of each of the four gold-capable classes (A = CH, B = CA, C = D-LF, D = D-FA)."""
    return {"CH": "A", "CA": "B", "D-LF": "C", "D-FA": "D"}


def fixed_gold_letter(question: dict[str, Any]) -> str:
    """Gold letter in the fixed four-option order, not the original shuffled gt_letter."""
    code = gold_class(question)
    assert code in fixed_letter_map(), (
        f"fixed_gold_letter: question {question.get('id')} has gold class {code}; expected one of {core_classes()}"
    )
    return fixed_letter_map()[code]


# ----------------------------------------------------------------------------------------------- holders


def blood_markers() -> list[str]:
    """Marker keys that can appear in a question's blood panel."""
    return ["hgba1c", "glu", "ins", "hdl", "ldl", "tg", "chol", "hscrp", "alt", "ast", "tsh", "hb", "hct", "fer", "fe",
            "d", "cor"]


def shared_context_blocks() -> set[str]:
    """Context blocks that more than one holder may hold (a copy each, on purpose)."""
    return {"demographics"}


def default_holder_table() -> dict[str, dict[str, list[str]]]:
    """Which device or site holds which metrics, context blocks and blood markers. Edit a copy to try others.

    Two clinics split the blood tests: a glycaemic clinic (glucose, HbA1c, insulin) and a lipid clinic
    (triglycerides, HDL). Markers in no holder (LDL, liver, thyroid, ...) are left out of every view.
    Demographics go to the phone and to both clinics, as a lab report carries age and sex.
    """
    return {
        "watch": {
            "metrics": ["rhr", "hrv", "vo2_max", "exercise_count", "exercise_hours", "exercise_mets",
                        "exercise_avg_hr", "avg_stress_level"],
            "context": [], "blood_markers": [],
        },
        "phone": {"metrics": ["steps", "active_burn", "calories_bmr"], "context": ["demographics"], "blood_markers": []},
        "ring": {
            "metrics": ["sleep_duration", "sleep_efficiency", "deep_percentage", "rem_percentage", "bed_time_std"],
            "context": [], "blood_markers": [],
        },
        "clinic_glycaemic": {"metrics": [], "context": ["demographics"], "blood_markers": ["glu", "hgba1c", "ins"]},
        "clinic_lipid": {"metrics": [], "context": ["demographics"], "blood_markers": ["tg", "hdl"]},
    }


def validate_holder_table(holders: dict[str, dict[str, list[str]]]) -> None:
    """Every metric in exactly one holder; context blocks once unless shared; blood markers at most once each."""
    for name, spec in holders.items():
        extra = set(spec) - {"metrics", "context", "blood_markers"}
        assert not extra and "metrics" in spec and "context" in spec, (
            f"validate_holder_table: holder {name} has keys {sorted(spec)}; expected metrics, context, blood_markers"
        )
    assigned = [m for spec in holders.values() for m in spec["metrics"]]
    for m in assigned:
        assert m in all_metrics(), f"validate_holder_table: unknown metric '{m}'"
    missing = sorted(set(all_metrics()) - set(assigned))
    doubled = sorted({m for m in assigned if assigned.count(m) > 1})
    assert not missing, f"validate_holder_table: metrics with no holder: {missing}"
    assert not doubled, f"validate_holder_table: metrics with more than one holder: {doubled}"
    contexts = [c for spec in holders.values() for c in spec["context"]]
    unknown = sorted(set(contexts) - {"demographics", "blood_panel"})
    assert not unknown, f"validate_holder_table: unknown context blocks {unknown}; use demographics or blood_panel"
    assert "demographics" in contexts, "validate_holder_table: no holder has the demographics block"
    for block in set(contexts) - shared_context_blocks():
        assert contexts.count(block) == 1, (
            f"validate_holder_table: {block} held by {contexts.count(block)} holders; it can only have one"
        )
    markers = [m for spec in holders.values() for m in spec.get("blood_markers", [])]
    for m in markers:
        assert m in blood_markers(), f"validate_holder_table: unknown blood marker '{m}'; known: {blood_markers()}"
    doubled_markers = sorted({m for m in markers if markers.count(m) > 1})
    assert not doubled_markers, f"validate_holder_table: blood markers with more than one holder: {doubled_markers}"
    assert not ("blood_panel" in contexts and markers), (
        "validate_holder_table: give the blood panel either whole (context blood_panel) or by blood_markers, not both"
    )
    assert "blood_panel" in contexts or markers, "validate_holder_table: no holder has any blood marker"


def metric_holder(holders: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    """Metric or blood marker ('blood:<marker>') to the holder that owns it."""
    owner = {}
    for name, spec in holders.items():
        for m in spec["metrics"]:
            owner[m] = name
        for m in spec.get("blood_markers", []):
            owner[f"blood:{m}"] = name
    return owner


# ----------------------------------------------------------------------------------------------- windows and masks


def question_masks(question: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(metrics hidden on every day, metrics hidden inside the window) for one question."""
    full = {canonical_metric(m) for m in (question.get("full_dropped_metrics") or [])}
    if question.get("hidden_metric"):
        full.add(canonical_metric(question["hidden_metric"]))
    window = {canonical_metric(m) for m in (question.get("window_dropped_metrics") or [])}
    return full, window


def question_rows(question: dict[str, Any], raw: dict[str, Any], window_days: int = 28,
                  history_days: int = 0) -> list[dict[str, Any]]:
    """The question's daily rows after masking, with only the metrics that have a value (days with none dropped)."""
    assert window_days > 0 and history_days >= 0, (
        f"question_rows: window_days={window_days}, history_days={history_days}; expected >0 and >=0"
    )
    end = date.fromisoformat(question["end_date"])
    start = end - timedelta(days=window_days + history_days)
    window_start = end - timedelta(days=window_days)
    full_drop, window_drop = question_masks(question)
    rows = []
    for row in raw["user_histories"][str(question["user_id"])]:
        day = date.fromisoformat(row["date"])
        if not start <= day <= end:
            continue
        clean = normalise_row(row)
        for m in list(clean):
            if m == "date":
                continue
            if m in full_drop or (m in window_drop and day >= window_start):
                del clean[m]
            elif m in ("exercise_avg_hr", "exercise_mets", "exercise_hours") and clean[m] == 0:
                del clean[m]  # the renderer treats a zero here as missing
            elif m == "avg_stress_level" and float(clean[m]) <= 0:
                del clean[m]
        if len(clean) > 1:
            rows.append(clean)
    return rows


# ----------------------------------------------------------------------------------------------- the split


def split_question(question: dict[str, Any], raw: dict[str, Any],
                   holders: dict[str, dict[str, list[str]]] | None = None,
                   window_days: int = 28, history_days: int = 0) -> dict[str, Any]:
    """One question as a shared part plus one private view per holder."""
    holders = holders if holders is not None else default_holder_table()
    validate_holder_table(holders)
    rows = question_rows(question, raw, window_days, history_days)
    agents = {}
    for name, spec in holders.items():
        agent_rows = []
        for row in rows:
            values = {m: row[m] for m in spec["metrics"] if m in row}
            if values:
                agent_rows.append({"date": row["date"], **values})
        context = {}
        for block in spec["context"]:
            value = (question.get("context") or {}).get(block) or {}
            if value:
                context[block] = dict(value)
        blood = (question.get("context") or {}).get("blood_panel") or {}
        markers = list(spec.get("blood_markers", []))
        held = {m: blood[m] for m in markers if m in blood}
        if held:
            context["blood_panel"] = held
        agents[name] = {"holder": name, "metrics": list(spec["metrics"]), "rows": agent_rows, "context": context,
                        "context_blocks": list(spec["context"]), "blood_markers": markers}
    return {
        "id": question["id"],
        "user_id": question["user_id"],
        "end_date": question["end_date"],
        "window_days": window_days,
        "history_days": history_days,
        "gold_letter": question["gt_letter"],
        "gold_class": gold_class(question),
        "option_classes": option_classes(question),
        "shared": {
            "query": question["stem"],
            "options": dict(question["options"]),
            "cohort_reference": {} if question.get("no_cohort") else raw["cohort_reference"],
        },
        "agents": agents,
    }


def check_partition(question: dict[str, Any], split: dict[str, Any], raw: dict[str, Any]) -> None:
    """Every (date, metric) value in exactly one agent; context blocks with their declared holders; each owned
    blood marker in exactly its owner."""
    expected = {(r["date"], m): v for r in question_rows(question, raw, split["window_days"], split["history_days"])
                for m, v in r.items() if m != "date"}
    seen: dict[tuple[str, str], Any] = {}
    for name, agent in split["agents"].items():
        for row in agent["rows"]:
            for m, v in row.items():
                if m == "date":
                    continue
                key = (row["date"], m)
                assert key not in seen, f"check_partition: {key} held by more than one agent in {split['id']}"
                assert m in agent["metrics"], f"check_partition: agent {name} holds {m}, which it does not own"
                seen[key] = v
    assert seen == expected, (
        f"check_partition: {split['id']} agents hold {len(seen)} values, the question has {len(expected)}"
    )
    question_context = question.get("context") or {}
    for block in ("demographics", "blood_panel"):
        declared = [n for n, a in split["agents"].items() if block in a["context_blocks"]]
        holding = [n for n, a in split["agents"].items() if block in a["context"] and block in a["context_blocks"]]
        expected_holders = declared if question_context.get(block) else []
        assert holding == expected_holders, (
            f"check_partition: {block} held by {holding} in {split['id']}; expected {expected_holders}"
        )
    blood = question_context.get("blood_panel") or {}
    marker_owner: dict[str, str] = {}
    for name, agent in split["agents"].items():
        if "blood_panel" in agent["context_blocks"]:
            continue
        for m, v in (agent["context"].get("blood_panel") or {}).items():
            assert m in agent["blood_markers"], f"check_partition: agent {name} holds marker {m}, which it does not own"
            assert m not in marker_owner, f"check_partition: marker {m} held by {marker_owner.get(m)} and {name}"
            assert blood.get(m) == v, f"check_partition: marker {m} of {name} differs from the question in {split['id']}"
            marker_owner[m] = name
    owned = {m for a in split["agents"].values() for m in a["blood_markers"]}
    assert set(marker_owner) == owned & set(blood), (
        f"check_partition: {split['id']} markers held {sorted(marker_owner)}, expected {sorted(owned & set(blood))}"
    )


# ----------------------------------------------------------------------------------------------- summaries


def agent_summary(agent: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per metric: number of days with a value, mean, min and max over the agent's rows."""
    out = {}
    for m in agent["metrics"]:
        values = [float(r[m]) for r in agent["rows"] if m in r]
        if values:
            out[m] = {"n_days": len(values), "mean": sum(values) / len(values), "min": min(values),
                      "max": max(values)}
        else:
            out[m] = {"n_days": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return out


def render_agent_text(agent: dict[str, Any], max_rows: int | None = None) -> str:
    """The agent's private information as plain text: context blocks, then one line per day."""
    lines = [f"=== {agent['holder'].upper()} (private) ==="]
    for block, values in agent["context"].items():
        items = ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())
        lines.append(f"{block}: {items}")
    rows = agent["rows"] if max_rows is None else agent["rows"][-max_rows:]
    if not agent["rows"] and agent["metrics"]:
        lines.append("(no readings in the window)")
    for row in rows:
        items = ", ".join(f"{m}={row[m]:.4g}" if isinstance(row[m], float) else f"{m}={row[m]}"
                          for m in agent["metrics"] if m in row)
        lines.append(f"{row['date']}: {items}")
    if max_rows is not None and len(agent["rows"]) > max_rows:
        lines.insert(1 + len(agent["context"]), f"... ({len(agent['rows']) - max_rows} earlier days not shown)")
    return "\n".join(lines)


def lab_flag_thresholds() -> dict[str, float]:
    """Common clinical cut-offs used to count adverse lab flags (an approximation; the benchmark's rule is unpublished)."""
    return {"hgba1c": 5.7, "glu": 100.0, "tg": 150.0, "hdl_male": 40.0, "hdl_female": 50.0, "homa_ir": 2.5}


def lab_flags(blood: dict[str, float], sex: str | None) -> dict[str, bool]:
    """Adverse lab flags for one blood panel; a flag is False when its markers are missing."""
    t = lab_flag_thresholds()
    hdl_cut = t["hdl_male"] if (sex or "").lower().startswith("m") else t["hdl_female"]
    flags = {
        "hba1c_high": blood.get("hgba1c") is not None and blood["hgba1c"] >= t["hgba1c"],
        "glucose_high": blood.get("glu") is not None and blood["glu"] >= t["glu"],
        "tg_high": blood.get("tg") is not None and blood["tg"] >= t["tg"],
        "hdl_low": blood.get("hdl") is not None and blood["hdl"] < hdl_cut,
        "homa_ir_high": (blood.get("glu") is not None and blood.get("ins") is not None
                         and blood["glu"] * blood["ins"] / 405.0 >= t["homa_ir"]),
    }
    return flags


def feature_rows(questions: list[dict[str, Any]], raw: dict[str, Any], window_days: int = 28) -> list[dict[str, Any]]:
    """One flat record per question: ids, gold class, demographics, window mean and day count per metric, blood
    markers and lab flags. Meant to be turned into a pandas DataFrame in the notebook."""
    records = []
    for q in questions:
        rows = question_rows(q, raw, window_days, 0)
        demo = (q.get("context") or {}).get("demographics") or {}
        blood = (q.get("context") or {}).get("blood_panel") or {}
        rec: dict[str, Any] = {
            "id": q["id"], "user_id": q["user_id"], "end_date": q["end_date"], "gold_letter": q["gt_letter"],
            "gold_class": gold_class(q), "age": demo.get("age"), "sex": demo.get("sex"), "bmi": demo.get("bmi"),
            "ethnicity": demo.get("ethnicity"), "window_rows": len(rows),
        }
        for m in all_metrics():
            values = [float(r[m]) for r in rows if m in r]
            rec[f"n_{m}"] = len(values)
            rec[f"mean_{m}"] = sum(values) / len(values) if values else float("nan")
        for marker, value in blood.items():
            rec[f"blood_{marker}"] = value
        flags = lab_flags(blood, demo.get("sex"))
        for flag, value in flags.items():
            rec[f"flag_{flag}"] = value
        rec["n_lab_flags"] = sum(flags.values())
        records.append(rec)
    return records
