"""Prompt rendering for the WearableQA phenotype task: one prompt per device agent, plus a centralised prompt.

Templates live in `prompts/wearableqa_phenotype/` (YAML): `shared.yaml` holds the system text, the question, the four
fixed options and the answer line; every other file is one agent (`device`, `axis`, `units`, `data`, `key`,
`steps`). An overlay folder (e.g. `prompts/wearableqa_phenotype_nokey/`) can replace some of those files for a
prompt ablation. The data block shows 28-day summary statistics, never raw daily values, unless the daily-rows
ablation is asked for.

A rendered prompt is (system, user): system = shared `system` with `{device}`; user = shared `user` with
`{options}`, `{data}`, `{key}`, `{steps}` and `{letters}` filled in.
"""

import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from utility import wqa_split as ws
from utility.prompt_utils import format_letters, get_placeholders


# ----------------------------------------------------------------------------------------------- templates


def shared_keys() -> list[str]:
    """Keys every shared.yaml must have."""
    return ["name", "version", "mode", "system", "system_centralised", "user", "options", "not_measured",
            "empty_data", "steps_centralised"]


def agent_keys() -> list[str]:
    """Keys every agent template must have."""
    return ["name", "version", "device", "axis", "units", "data", "key", "steps"]


def shared_user_placeholders() -> set[str]:
    """Placeholders the shared user text may use."""
    return {"options", "data", "key", "steps", "letters"}


def demographic_placeholders() -> set[str]:
    """Data-block placeholders filled from the demographics block."""
    return {"age", "sex", "bmi"}


def blood_placeholder_markers() -> dict[str, str]:
    """Data-block placeholder -> blood-panel marker key (HOMA-IR is computed, not a marker)."""
    return {"glu": "glu", "hba1c": "hgba1c", "ins": "ins", "tg": "tg", "hdl": "hdl", "ldl": "ldl", "chol": "chol"}


def known_data_placeholders() -> set[str]:
    """Every placeholder an agent's data block may use."""
    return (set(ws.all_metrics()) | {"exercise", "homa"} | demographic_placeholders()
            | set(blood_placeholder_markers()))


def placeholders(text: str) -> set[str]:
    """Placeholder names in a format string (prompt_utils.get_placeholders)."""
    return get_placeholders(text)


def read_yaml(path: Path) -> tuple[dict[str, Any], str]:
    """(mapping, raw text) of one YAML file."""
    raw_text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    assert isinstance(data, dict), f"read_yaml: {path} does not contain a YAML mapping"
    return data, raw_text


def validate_shared(shared: dict[str, Any]) -> None:
    """Required keys, known placeholders, and the four fixed options A-D = CH, CA, D-LF, D-FA."""
    missing = [k for k in shared_keys() if k not in shared]
    assert not missing, f"validate_shared: shared template lacks keys {missing}"
    assert shared["mode"] == "answer_only", f"validate_shared: mode is {shared['mode']!r}; only answer_only is used"
    unknown = placeholders(shared["user"]) - shared_user_placeholders()
    assert not unknown, f"validate_shared: user text uses unknown placeholders {sorted(unknown)}"
    for needed in ("options", "data"):
        assert needed in placeholders(shared["user"]), f"validate_shared: user text must use {{{needed}}}"
    assert placeholders(shared["system"]) <= {"device"}, (
        f"validate_shared: system text may only use {{device}}, found {sorted(placeholders(shared['system']))}"
    )
    options = shared["options"]
    expected = [(letter, code) for code, letter in ws.fixed_letter_map().items()]
    found = [(o.get("letter"), o.get("code")) for o in options] if isinstance(options, list) else None
    assert found == expected, f"validate_shared: options are {found}; expected {expected} in this order"
    assert all(isinstance(o.get("text"), str) and o["text"].strip() for o in options), (
        "validate_shared: every option needs a non-empty text"
    )


def validate_agent(agent: dict[str, Any]) -> None:
    """Required keys, a known axis, and only known placeholders in the data block."""
    missing = [k for k in agent_keys() if k not in agent]
    assert not missing, f"validate_agent: agent template {agent.get('name', '?')} lacks keys {missing}"
    assert agent["axis"] in ("fitness", "metabolic", "none"), (
        f"validate_agent: {agent['name']} has axis {agent['axis']!r}; use fitness, metabolic or none"
    )
    unknown = placeholders(agent["data"]) - known_data_placeholders()
    assert not unknown, f"validate_agent: {agent['name']} data block uses unknown placeholders {sorted(unknown)}"
    for key in ("key", "steps"):
        assert not placeholders(agent[key]), f"validate_agent: {agent['name']} {key} text must have no placeholders"
    assert isinstance(agent["units"], dict), f"validate_agent: {agent['name']} units must be a mapping"
    assert agent.get("ask", "fitness") in axis_names(), (
        f"validate_agent: {agent['name']} has ask {agent.get('ask')!r}; use one of {axis_names()}"
    )


def template_files(folder: Path) -> dict[str, Path]:
    """YAML files of a template folder by stem."""
    folder = Path(folder)
    assert folder.is_dir(), f"template_files: prompt folder {folder} does not exist"
    return {p.stem: p for p in sorted(folder.glob("*.yaml"))}


def load_wqa_templates(prompt_dir: Path, overlay_dir: Path | None = None) -> dict[str, Any]:
    """Shared and agent templates; files in overlay_dir replace the files of the same name in prompt_dir."""
    files = template_files(prompt_dir)
    if overlay_dir is not None:
        overlay = template_files(overlay_dir)
        unknown = sorted(set(overlay) - set(files))
        assert not unknown, f"load_wqa_templates: overlay {overlay_dir} has files {unknown} with no base template"
        files.update(overlay)
    assert "shared" in files, f"load_wqa_templates: no shared.yaml in {prompt_dir}"
    shared, shared_raw = read_yaml(files["shared"])
    validate_shared(shared)
    agents, raw_text = {}, {"shared": shared_raw}
    for stem, path in files.items():
        if stem == "shared":
            continue
        agent, text = read_yaml(path)
        validate_agent(agent)
        assert agent["name"] == stem, f"load_wqa_templates: {path} declares name {agent['name']!r}; make it {stem!r}"
        agents[stem], raw_text[stem] = agent, text
    assert agents, f"load_wqa_templates: no agent templates in {prompt_dir}"
    return {"shared": shared, "agents": agents, "raw_text": raw_text,
            "files": {stem: str(path) for stem, path in files.items()}}


def template_version(templates: dict[str, Any]) -> str:
    """'shared:1,watch:1,...' from the version field of every file."""
    parts = [f"shared:{templates['shared']['version']}"]
    parts += [f"{name}:{t['version']}" for name, t in sorted(templates["agents"].items())]
    return ",".join(parts)


# ----------------------------------------------------------------------------------------------- formatting


def metric_decimals() -> dict[str, int]:
    """Decimals shown for each metric's summary statistics."""
    return {"rhr": 1, "hrv": 1, "vo2_max": 1, "steps": 0, "active_burn": 0, "calories_bmr": 0,
            "sleep_duration": 1, "sleep_efficiency": 2, "deep_percentage": 1, "rem_percentage": 1,
            "bed_time_std": 0, "exercise_hours": 2, "exercise_mets": 1, "exercise_avg_hr": 0, "exercise_count": 0,
            "avg_stress_level": 0}


def fmt(value: float, decimals: int) -> str:
    """Number with a fixed number of decimals; integers get thousands separators."""
    return f"{value:,.0f}" if decimals == 0 else f"{value:.{decimals}f}"


def with_unit(text: str, unit: str) -> str:
    """'61.2 bpm', or '61.2' for an empty unit, or '45.0%' for a percent sign."""
    if not unit:
        return text
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}"


def window_rows(rows: list[dict[str, Any]], end_date: str, window_days: int) -> list[dict[str, Any]]:
    """Rows of the last `window_days` days up to and including end_date."""
    assert window_days >= 1, f"window_rows: window_days must be >= 1, got {window_days}"
    end = date.fromisoformat(end_date)
    first = end - timedelta(days=window_days - 1)
    return [r for r in rows if first <= date.fromisoformat(r["date"]) <= end]


def metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    """The metric's values over the rows that have it."""
    return [float(r[metric]) for r in rows if metric in r]


def metric_summary(values: list[float], unit: str, decimals: int, window_days: int, not_measured: str) -> str:
    """'mean 61.2 bpm, median 61.0, range 55.0-70.0, 27/28 days', or not_measured when there are no values."""
    if not values:
        return not_measured
    mean = fmt(statistics.fmean(values), decimals)
    median = fmt(statistics.median(values), decimals)
    low, high = fmt(min(values), decimals), fmt(max(values), decimals)
    return f"mean {with_unit(mean, unit)}, median {median}, range {low}-{high}, {len(values)}/{window_days} days"


def cohort_percentile_keys() -> list[str]:
    """Percentile keys of the cohort reference, low to high."""
    return ["p10", "p25", "p50", "p75", "p90"]


def cohort_position(value: float, percentiles: dict[str, float]) -> str:
    """Where a value sits among the cohort percentiles: 'below p10', 'between p50 and p75', 'above p90'."""
    keys = [k for k in cohort_percentile_keys() if k in percentiles]
    assert len(keys) >= 2, f"cohort_position: need at least two percentiles, got {sorted(percentiles)}"
    if value < percentiles[keys[0]]:
        return f"below {keys[0]}"
    if value > percentiles[keys[-1]]:
        return f"above {keys[-1]}"
    bands = [(low, high) for low, high in zip(keys, keys[1:]) if percentiles[low] <= value <= percentiles[high]]
    assert bands, f"cohort_position: percentiles {percentiles} are not increasing; cannot place {value}"
    return f"between {bands[0][0]} and {bands[0][1]}"


def metric_text(values: list[float], unit: str, decimals: int, window_days: int, cohort: dict[str, float] | None,
                not_measured: str) -> str:
    """Summary, plus population percentiles and the mean's position when the cohort reference has the metric."""
    summary = metric_summary(values, unit, decimals, window_days, not_measured)
    if not values or not cohort:
        return summary
    shown = ", ".join(f"{k}={cohort[k]}" for k in cohort_percentile_keys() if k in cohort)
    position = cohort_position(statistics.fmean(values), cohort)
    return f"{summary}. Population: {shown}. Position: {position}."


def exercise_text(rows: list[dict[str, Any]], window_days: int, not_measured: str) -> str:
    """'<sessions> sessions in 28 days, mean <h> h on exercise days', or not_measured."""
    counts = metric_values(rows, "exercise_count")
    hours = metric_values(rows, "exercise_hours")
    if not counts and not hours:
        return not_measured
    sessions = f"{fmt(sum(counts), 0)} sessions in {window_days} days" if counts else "sessions not measured"
    duration = f", mean {fmt(statistics.fmean(hours), 2)} h on exercise days" if hours else ""
    return sessions + duration


def blood_text(value: float | None, unit: str, decimals: int, not_measured: str) -> str:
    """'85.0 mg/dL', or not_measured when the value is missing."""
    return not_measured if value is None else with_unit(fmt(float(value), decimals), unit)


def homa_ir(glu: float | None, ins: float | None) -> float | None:
    """HOMA-IR = fasting glucose (mg/dL) x fasting insulin (uIU/mL) / 405, rounded to 2 decimals; None if missing."""
    if glu is None or ins is None:
        return None
    return round(float(glu) * float(ins) / 405.0, 2)


def blood_decimals() -> dict[str, int]:
    """Decimals shown for each blood placeholder."""
    return {"glu": 0, "hba1c": 1, "ins": 1, "homa": 2, "tg": 0, "hdl": 0, "ldl": 0, "chol": 0}


def sex_text(sex: str | None, not_measured: str) -> str:
    """'male' / 'female' from 'm' / 'f'."""
    return {"m": "male", "f": "female"}.get(str(sex).lower(), not_measured) if sex else not_measured


# ----------------------------------------------------------------------------------------------- fields


def agent_fields(templates: dict[str, Any], agent_name: str, split: dict[str, Any],
                 window_days: int = 28) -> dict[str, str]:
    """Every placeholder of the agent's data block, filled from that agent's private view in the split."""
    assert agent_name in templates["agents"], (
        f"agent_fields: no template for agent {agent_name!r}; templates: {sorted(templates['agents'])}"
    )
    assert agent_name in split["agents"], (
        f"agent_fields: split {split['id']} has no agent {agent_name!r}; agents: {sorted(split['agents'])}"
    )
    template, agent = templates["agents"][agent_name], split["agents"][agent_name]
    not_measured = templates["shared"]["not_measured"]
    units, decimals = template["units"], metric_decimals()
    rows = window_rows(agent["rows"], split["end_date"], window_days)
    cohort = split["shared"]["cohort_reference"] or {}
    demographics = agent["context"].get("demographics") or {}
    blood = agent["context"].get("blood_panel") or {}
    fields = {}
    for name in placeholders(template["data"]):
        if name in ws.all_metrics():
            assert name in agent["metrics"], f"agent_fields: {agent_name} template shows {name}, which it does not hold"
            fields[name] = metric_text(metric_values(rows, name), units.get(name, ""), decimals[name], window_days,
                                       cohort.get(name), not_measured)
        elif name == "exercise":
            fields[name] = exercise_text(rows, window_days, not_measured)
        elif name == "age":
            fields[name] = fmt(demographics["age"], 0) if demographics.get("age") is not None else not_measured
        elif name == "sex":
            fields[name] = sex_text(demographics.get("sex"), not_measured)
        elif name == "bmi":
            fields[name] = fmt(demographics["bmi"], 1) if demographics.get("bmi") is not None else not_measured
        elif name == "homa":
            fields[name] = blood_text(homa_ir(blood.get("glu"), blood.get("ins")), units.get(name, ""),
                                      blood_decimals()[name], not_measured)
        else:
            marker = blood_placeholder_markers()[name]
            fields[name] = blood_text(blood.get(marker), units.get(name, ""), blood_decimals()[name], not_measured)
    return fields


# ----------------------------------------------------------------------------------------------- rendering


def format_options(shared: dict[str, Any]) -> str:
    """'A. ...' lines in the fixed order."""
    return "\n".join(f"{o['letter']}. {o['text']}" for o in shared["options"])


def option_letters(templates: dict[str, Any]) -> list[str]:
    """['A', 'B', 'C', 'D']."""
    return [o["letter"] for o in templates["shared"]["options"]]


def fill_user(templates: dict[str, Any], data: str, key: str, steps: str) -> str:
    """The shared user text with every placeholder filled."""
    shared = templates["shared"]
    values = {"options": format_options(shared), "data": data.strip(), "key": key.strip(), "steps": steps.strip(),
              "letters": format_letters(option_letters(templates))}
    text = shared["user"].format(**values)
    while "\n\n\n" in text:  # an empty key (no-key ablation) leaves blank lines behind
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def data_block(templates: dict[str, Any], agent_name: str, split: dict[str, Any], window_days: int = 28) -> str:
    """The agent's data block with 28-day summaries."""
    template = templates["agents"][agent_name]
    return template["data"].format(**agent_fields(templates, agent_name, split, window_days)).strip()


def daily_data_block(templates: dict[str, Any], agent_name: str, split: dict[str, Any], window_days: int = 28) -> str:
    """Ablation: the header and the non-metric lines of the data block, then one line per day of raw values."""
    template = templates["agents"][agent_name]
    agent = split["agents"][agent_name]
    fields = agent_fields(templates, agent_name, split, window_days)
    metric_names = set(ws.all_metrics()) | {"exercise"}
    lines = [line for line in template["data"].strip().splitlines() if not placeholders(line) & metric_names]
    kept = "\n".join(lines).format(**fields)
    rows = window_rows(agent["rows"], split["end_date"], window_days)
    if not agent["metrics"]:
        return kept
    decimals, units = metric_decimals(), template["units"]
    day_lines = []
    for row in rows:
        parts = [f"{m}={with_unit(fmt(float(row[m]), decimals[m]), units.get(m, ''))}"
                 for m in agent["metrics"] if m in row]
        if parts:
            day_lines.append(f"{row['date']}: " + ", ".join(parts))
    days = "\n".join(day_lines) if day_lines else templates["shared"]["not_measured"]
    return f"{kept}\n{days}"


def render_agent(templates: dict[str, Any], agent_name: str, split: dict[str, Any], with_data: bool = True,
                 daily: bool = False, window_days: int = 28) -> tuple[str, str]:
    """(system, user) for one device agent; with_data=False puts empty_data in place of the data (the prior)."""
    shared, template = templates["shared"], templates["agents"][agent_name]
    if not with_data:
        data = shared["empty_data"]
    elif daily:
        data = daily_data_block(templates, agent_name, split, window_days)
    else:
        data = data_block(templates, agent_name, split, window_days)
    system = shared["system"].format(device=template["device"]).strip()
    return system, fill_user(templates, data, template["key"], template["steps"])


def render_centralised(templates: dict[str, Any], split: dict[str, Any], agent_names: list[str],
                       with_data: bool = True, daily: bool = False, window_days: int = 28) -> tuple[str, str]:
    """(system, user) for one agent that sees every listed agent's data and key."""
    assert agent_names, "render_centralised: no agents listed"
    shared = templates["shared"]
    if with_data:
        blocks = [daily_data_block(templates, a, split, window_days) if daily
                  else data_block(templates, a, split, window_days) for a in agent_names]
        data = "\n\n".join(blocks)
    else:
        data = shared["empty_data"]
    keys = [templates["agents"][a]["key"].strip() for a in agent_names if templates["agents"][a]["key"].strip()]
    return shared["system_centralised"].strip(), fill_user(templates, data, "\n\n".join(keys),
                                                           shared["steps_centralised"])


# ----------------------------------------------------------------------------------------------- axis questions


def axis_names() -> list[str]:
    """The two judgements: fitness (wearables) and metabolic (blood)."""
    return ["fitness", "metabolic"]


def axis_sides() -> dict[str, list[str]]:
    """The two answers of each axis question, favourable first."""
    return {"fitness": ["favourable", "low"], "metabolic": ["favourable", "adverse"]}


def side_classes() -> dict[str, dict[str, list[str]]]:
    """The two classes each side of an axis is consistent with."""
    return {"fitness": {"favourable": ["CH", "D-FA"], "low": ["CA", "D-LF"]},
            "metabolic": {"favourable": ["CH", "D-LF"], "adverse": ["CA", "D-FA"]}}


def class_side(code: str, axis: str) -> str:
    """The side of an axis a class sits on."""
    return next(side for side, codes in side_classes()[axis].items() if code in codes)


def validate_axis_shared(shared: dict[str, Any]) -> None:
    """The keys the axis-question variant needs in shared.yaml."""
    for key in ("user_axis", "axis_questions", "steps_centralised_axis"):
        assert key in shared, f"validate_axis_shared: shared template lacks {key!r}; use the axis overlay folder"
    unknown = placeholders(shared["user_axis"]) - (shared_user_placeholders() | {"question"})
    assert not unknown, f"validate_axis_shared: user_axis uses unknown placeholders {sorted(unknown)}"
    for axis in axis_names():
        question = shared["axis_questions"].get(axis) or {}
        sides = [o.get("side") for o in question.get("options") or []]
        assert sides == axis_sides()[axis], (
            f"validate_axis_shared: {axis} options have sides {sides}; expected {axis_sides()[axis]}"
        )
        assert isinstance(question.get("question"), str) and question["question"].strip(), (
            f"validate_axis_shared: {axis} question text is missing"
        )
        assert isinstance(shared["steps_centralised_axis"].get(axis), str), (
            f"validate_axis_shared: steps_centralised_axis has no {axis} entry"
        )


def asked_axis(templates: dict[str, Any], agent_name: str) -> str:
    """The question an agent answers in the axis variant: its own axis, or `ask` (e.g. the ring)."""
    template = templates["agents"][agent_name]
    axis = template.get("ask", template["axis"])
    assert axis in axis_names(), f"asked_axis: agent {agent_name} has no axis question (axis {axis!r}); set `ask`"
    return axis


def axis_letter_sides(axis: str, order: int) -> list[tuple[str, str]]:
    """[(letter, side)] of an axis question: order 0 puts the favourable answer first (A), order 1 last."""
    assert order in (0, 1), f"axis_letter_sides: order must be 0 or 1, got {order}"
    sides = axis_sides()[axis] if order == 0 else list(reversed(axis_sides()[axis]))
    return list(zip(["A", "B"], sides))


def fill_axis_user(templates: dict[str, Any], axis: str, order: int, data: str, key: str, steps: str) -> str:
    """The axis-question user text for one axis and letter order."""
    shared = templates["shared"]
    validate_axis_shared(shared)
    question = shared["axis_questions"][axis]
    texts = {o["side"]: o["text"] for o in question["options"]}
    pairs = axis_letter_sides(axis, order)
    values = {"question": question["question"].strip(), "options": "\n".join(f"{l}. {texts[s]}" for l, s in pairs),
              "data": data.strip(), "key": key.strip(), "steps": steps.strip(),
              "letters": format_letters([l for l, _ in pairs])}
    text = shared["user_axis"].format(**values)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def render_axis_agent(templates: dict[str, Any], agent_name: str, split: dict[str, Any], order: int,
                      with_data: bool = True, window_days: int = 28) -> tuple[str, str]:
    """(system, user) of one device agent's own two-option question; with_data=False gives the prior."""
    shared, template = templates["shared"], templates["agents"][agent_name]
    data = data_block(templates, agent_name, split, window_days) if with_data else shared["empty_data"]
    system = shared["system"].format(device=template["device"]).strip()
    return system, fill_axis_user(templates, asked_axis(templates, agent_name), order, data, template["key"],
                                  template["steps"])


def render_axis_centralised(templates: dict[str, Any], split: dict[str, Any], agent_names: list[str], axis: str,
                            order: int, with_data: bool = True, window_days: int = 28) -> tuple[str, str]:
    """(system, user) of the centralised agent's two-option question on one axis, with every listed agent's data."""
    shared = templates["shared"]
    data = ("\n\n".join(data_block(templates, a, split, window_days) for a in agent_names) if with_data
            else shared["empty_data"])
    keys = [templates["agents"][a]["key"].strip() for a in agent_names if templates["agents"][a]["key"].strip()]
    return shared["system_centralised"].strip(), fill_axis_user(templates, axis, order, data, "\n\n".join(keys),
                                                                shared["steps_centralised_axis"][axis])
