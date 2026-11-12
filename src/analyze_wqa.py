"""Analyse a WearableQA run: per-agent metrics, conformal sets fused over agent groups, and baselines.

The conformal machinery is the existing one (`analysis.conformal`: scores, calibration rules, quorum sets,
evaluate_sets, summarize_splits), used unchanged. What is specific to this dataset lives here:
- calibration/test splits are drawn over users (a user's questions never sit in both halves);
- tasks of the classes in `exclude_from_calibration` (D-FA by default) are kept out of calibration, not out of test;
  their coverage is not controlled, so `coverage_calibrated_classes` (test tasks of the other classes only) is the
  figure the 1 - alpha target applies to;
- coverage and set size are also reported per gold class and per judgement axis;
- agent groups ("conditions") are chosen here from one run's records: single agents, fusion without and with the
  ring, the centralised agent, the original prompt.

Example (from code/):
  python src/analyze_wqa.py --predictions results/<run>/predictions_wqa_strict_key_N75.json \
      --tasks data/wearableqa/tasks.jsonl --rf
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from analysis.conformal import (calibrate, evaluate_sets, get_score_functions, quorum_set, summarize_splits,
                                thresholds_for)
from analysis.scores import build_task_views, load_predictions
from utility import wqa_split as ws
from utility.wqa_prompts import window_rows
from wqa_prepare import load_tasks


# ----------------------------------------------------------------------------------------------- classes and axes


def fit_classes() -> set[str]:
    """Classes with favourable wearable fitness."""
    return {"CH", "D-FA"}


def favourable_lab_classes() -> set[str]:
    """Classes with a favourable metabolic (blood) profile."""
    return {"CH", "D-LF"}


def letter_class() -> dict[str, str]:
    """Fixed letter -> class code."""
    return {letter: code for code, letter in ws.fixed_letter_map().items()}


def axis_value(code: str, axis: str) -> bool:
    """The class's side of one axis: fit (fitness) or favourable labs (metabolic)."""
    assert axis in ("fitness", "metabolic"), f"axis_value: axis {axis!r} must be fitness or metabolic"
    return code in (fit_classes() if axis == "fitness" else favourable_lab_classes())


# ----------------------------------------------------------------------------------------------- records and views


def load_wqa_predictions(path: Path) -> list[dict[str, Any]]:
    """Records of one WearableQA run (checked by the shared loader, then for the WearableQA fields)."""
    records = load_predictions(path)
    needed = ["dataset", "agent_id", "user_id", "gold_class", "agent_axis", "role"]
    missing = [key for key in needed if key not in records[0]]
    assert not missing, f"load_wqa_predictions: records in {path} lack {missing}; is this a wqa_exp.py run?"
    assert {r["dataset"] for r in records} == {"wearableqa_phenotype"}, (
        f"load_wqa_predictions: {path} is not a WearableQA run"
    )
    return records


def get_conditions(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Agent groups to evaluate: each agent alone, fusion without / with the ring, centralised, original."""
    agent_ids = list(dict.fromkeys(r["agent_id"] for r in records))
    roles = {r["agent_id"]: r["role"] for r in records}
    devices = [a for a in agent_ids if roles[a] == "device"]
    conditions = {f"single:{a}": [a] for a in agent_ids}
    without_ring = [a for a in devices if a != "ring"]
    if len(without_ring) >= 2:
        conditions["fusion_no_ring"] = without_ring
    if "ring" in devices and len(devices) >= 2:
        conditions["fusion_with_ring"] = devices
    return conditions


def condition_views(records: list[dict[str, Any]], agent_ids: list[str]) -> list[dict[str, Any]]:
    """Per-task views (analysis.scores.build_task_views) of the listed agents, with user and gold class added."""
    chosen = [r for r in records if r["agent_id"] in agent_ids]
    assert chosen, f"condition_views: no records for agents {agent_ids}"
    info = {r["task_index"]: r for r in chosen}
    views = build_task_views(chosen)
    for view in views:
        record = info[view["task_index"]]
        view["user_id"], view["gold_class"], view["task_id"] = record["user_id"], record["gold_class"], record["task_id"]
        assert len(view["agents"]) == len(agent_ids), (
            f"condition_views: task {view['task_index']} has {len(view['agents'])} of the {len(agent_ids)} agents"
        )
    return views


# ----------------------------------------------------------------------------------------------- grouped conformal


def make_group_splits(groups: list[str], n_splits: int, seed: int) -> list[set[str]]:
    """Calibration groups of each split: half of the distinct groups (users), drawn with a fixed seed."""
    distinct = sorted(set(groups))
    assert len(distinct) >= 4, f"make_group_splits: need at least 4 groups, got {len(distinct)}"
    rng = random.Random(seed)
    splits = []
    for _ in range(n_splits):
        order = list(distinct)
        rng.shuffle(order)
        splits.append(set(order[:len(order) // 2]))
    return splits


def split_views(views: list[dict[str, Any]], calibration_groups: set[str],
                exclude_from_calibration: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """(calibration, test): calibration = views of the calibration users minus the excluded classes."""
    calibration = [v for v in views if v["user_id"] in calibration_groups
                   and v["gold_class"] not in exclude_from_calibration]
    test = [v for v in views if v["user_id"] not in calibration_groups]
    return calibration, test


def class_and_axis_tallies(test: list[dict[str, Any]], sets: list[list[str]], tallies: dict[str, Any]) -> None:
    """Add one split's per-class coverage and size, and per-axis coverage, to running tallies."""
    codes = letter_class()
    for view, kept in zip(test, sets):
        cls = view["gold_class"]
        entry = tallies["per_class"].setdefault(cls, {"n": 0, "covered": 0, "size_sum": 0})
        entry["n"] += 1
        entry["covered"] += view["true_letter"] in kept
        entry["size_sum"] += len(kept)
        for axis in ("fitness", "metabolic"):
            axis_entry = tallies["per_axis"].setdefault(axis, {"n": 0, "covered": 0})
            axis_entry["n"] += 1
            axis_entry["covered"] += any(axis_value(codes[l], axis) == axis_value(cls, axis) for l in kept)


def summarize_tallies(tallies: dict[str, Any]) -> dict[str, Any]:
    """Pooled coverage and mean set size per class, coverage per axis (with the counts behind them)."""
    per_class = {cls: {"n": e["n"], "coverage": e["covered"] / e["n"], "mean_set_size": e["size_sum"] / e["n"]}
                 for cls, e in sorted(tallies["per_class"].items())}
    per_axis = {axis: {"n": e["n"], "coverage": e["covered"] / e["n"]} for axis, e in tallies["per_axis"].items()}
    return {"per_class": per_class, "per_axis": per_axis}


def run_grouped_conformal(views: list[dict[str, Any]], splits: list[set[str]], score: str, rule: str, alpha: float,
                          k: int, exclude_from_calibration: tuple[str, ...] = ("D-FA",)) -> dict[str, Any]:
    """analysis.conformal's calibrate / thresholds_for / quorum_set / evaluate_sets over user-grouped splits."""
    assert type(k) is int and k >= 1, f"run_grouped_conformal: k must be a positive integer, got {k!r}"
    n_agents = min(len(v["agents"]) for v in views)
    assert k <= n_agents, f"run_grouped_conformal: k={k} is larger than the {n_agents} agents per task"
    score_fn = get_score_functions()[score]
    per_split, sizes = [], []
    tallies: dict[str, Any] = {"per_class": {}, "per_axis": {}}
    for calibration_groups in splits:
        calibration, test = split_views(views, calibration_groups, exclude_from_calibration)
        assert calibration and test, "run_grouped_conformal: a split left calibration or test empty"
        calibrated = calibrate(calibration, score_fn, rule, alpha, k)
        sets = [quorum_set(v, score_fn, thresholds_for(v, calibrated, rule, alpha, k), k) for v in test]
        per_split.append(evaluate_sets(test, sets))
        class_and_axis_tallies(test, sets, tallies)
        sizes.append(len(calibration))
    summary = summarize_tallies(tallies)
    kept = [e for cls, e in tallies["per_class"].items() if cls not in exclude_from_calibration]
    covered = sum(e["covered"] for e in kept)
    total = sum(e["n"] for e in kept)
    return {"score": score, "rule": rule, "alpha": alpha, "k": k, "n_splits": len(splits),
            "n_calibration_mean": statistics.fmean(sizes), "exclude_from_calibration": list(exclude_from_calibration),
            "metrics": summarize_splits(per_split),
            "coverage_calibrated_classes": covered / total if total else math.nan, **summary}


# ----------------------------------------------------------------------------------------------- per-agent metrics


def posterior_probabilities(view_agent: dict[str, Any]) -> dict[str, float]:
    """Normalised posterior probabilities of one agent view."""
    return {letter: math.exp(value) for letter, value in view_agent["posterior"].items()}


def agent_metrics(records: list[dict[str, Any]], agent_id: str) -> dict[str, Any]:
    """Generated-letter accuracy, argmax accuracy, axis accuracy, mass on the gold axis side, lift by class."""
    own = [r for r in records if r["agent_id"] == agent_id]
    axis = own[0]["agent_axis"]
    views = condition_views(records, [agent_id])
    codes = letter_class()
    argmax_letters = [max(v["agents"][0]["posterior"], key=v["agents"][0]["posterior"].get) for v in views]
    out: dict[str, Any] = {
        "agent_axis": axis,
        "n": len(own),
        "accuracy_generated": statistics.fmean(float(r["correct"]) for r in own),
        "parse_failures": sum(not r["parse_ok"] for r in own),
        "accuracy_argmax": statistics.fmean(float(l == v["true_letter"]) for l, v in zip(argmax_letters, views)),
    }
    for side in ("fitness", "metabolic"):
        if axis not in (side, "both"):
            continue
        out[f"{side}_axis_accuracy_argmax"] = statistics.fmean(
            float(axis_value(codes[l], side) == axis_value(v["gold_class"], side)) for l, v in zip(argmax_letters, views))
        masses = []
        for v in views:
            probs = posterior_probabilities(v["agents"][0])
            gold_side = axis_value(v["gold_class"], side)
            masses.append(sum(p for l, p in probs.items() if axis_value(codes[l], side) == gold_side))
        out[f"{side}_axis_mass"] = statistics.fmean(masses)
    lift_by_class: dict[str, dict[str, float]] = {}
    for cls in ws.core_classes():
        lifts = [v["agents"][0]["lift"] for v in views if v["gold_class"] == cls]
        if lifts:
            lift_by_class[cls] = {codes[l]: statistics.fmean(d[l] for d in lifts) for l in sorted(lifts[0])}
    out["mean_lift_by_gold_class"] = lift_by_class
    out["mean_posterior_by_gold_class"] = {
        cls: {codes[l]: statistics.fmean(posterior_probabilities(v["agents"][0])[l] for v in views
                                         if v["gold_class"] == cls) for l in sorted(views[0]["letters"])}
        for cls in ws.core_classes() if any(v["gold_class"] == cls for v in views)}
    return out


# ----------------------------------------------------------------------------------------------- F1


def f1_per_label(gold: list[str], predicted: list[str | None], labels: list[str]) -> dict[str, float]:
    """F1 of each label (0 when a label is never predicted and never true), plus 'macro' over the labels."""
    out = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        out[label] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    out["macro"] = statistics.fmean(out[label] for label in labels)
    return out


def argmax_key(scores: dict[str, float]) -> str:
    """Key with the highest score (ties broken by key order)."""
    return max(sorted(scores), key=lambda key: scores[key])


def axis_sides_of(axis: str) -> list[str]:
    """The two sides of an axis, favourable first."""
    return {"fitness": ["favourable", "low"], "metabolic": ["favourable", "adverse"]}[axis]


def side_of(code: str | None, axis: str) -> str | None:
    """The side of an axis a class sits on (None for no class)."""
    if code is None:
        return None
    return "favourable" if axis_value(code, axis) else axis_sides_of(axis)[1]


def ideal_class(agent_id: str, task: dict[str, Any]) -> str | None:
    """The best a perfect reader of the agent's own data could do with the strict clinical key, taking the more
    common class on the judgement it cannot see (fit -> CH, not fit -> CA; favourable labs -> CH, adverse -> CA).
    Watch: resting HR below the cohort median; phone: >= 7,500 steps a day (somewhat active or more); clinics: their
    own guideline flags; centralised: both. None for agents with no judgement of their own (ring, original)."""
    rows = window_rows(task["rows"], task["end_date"], task["window_days"])
    context = task["question"].get("context") or {}
    blood, sex = context.get("blood_panel") or {}, (context.get("demographics") or {}).get("sex")
    homa = blood["glu"] * blood["ins"] / 405.0 if "glu" in blood and "ins" in blood else None
    glycaemic = any([blood.get("glu", 0) >= 100, blood.get("hgba1c", 0) >= 5.7, homa is not None and homa >= 2.5])
    lipid = any([blood.get("tg", 0) >= 150, "hdl" in blood and blood["hdl"] < (40 if sex == "m" else 50)])
    rhr = [float(r["rhr"]) for r in rows if "rhr" in r]
    steps = [float(r["steps"]) for r in rows if "steps" in r]
    fit_rhr = bool(rhr) and statistics.fmean(rhr) < task["cohort_reference"]["rhr"]["p50"]
    rules = {"watch": lambda: "CH" if fit_rhr else "CA",
             "phone": lambda: "CH" if steps and statistics.fmean(steps) >= 7500 else "CA",
             "clinic_glycaemic": lambda: "CA" if glycaemic else "CH",
             "clinic_lipid": lambda: "CA" if lipid else "CH",
             "centralised": lambda: guideline_rule_class(task),
             "centralised_axis": lambda: guideline_rule_class(task)}
    return rules[agent_id]() if agent_id in rules else None


def agent_f1(records: list[dict[str, Any]], agent_id: str,
             tasks: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """F1 of one agent: per class (four-option agents) and on its own judgement (both kinds), for the generated
    answer, the most probable option, the option with the largest lift, and the ideal reader (when tasks are given)."""
    own = [r for r in records if r["agent_id"] == agent_id]
    gold = [r["gold_class"] for r in own]
    codes = letter_class()
    axis = own[0]["agent_axis"]
    axes = [axis] if axis in ("fitness", "metabolic") else (["fitness", "metabolic"] if axis == "both" else [])
    out: dict[str, Any] = {"agent_axis": axis, "two_option": "axis_question" in own[0]}
    if out["two_option"]:
        sides = {
            "generated": [r["predicted_side"] for r in own],
            "argmax_posterior": [argmax_key(r["option_logprobs_axis_posterior"]) for r in own],
            "argmax_lift": [argmax_key({k: r["option_logprobs_axis_posterior"][k] - r["option_logprobs_axis_prior"][k]
                                        for k in r["option_logprobs_axis_posterior"]}) for r in own],
        }
        if tasks is not None and ideal_class(agent_id, tasks[own[0]["task_id"]]) is not None:
            sides["ideal"] = [side_of(ideal_class(agent_id, tasks[r["task_id"]]), axis) for r in own]
        truth = [side_of(g, axis) for g in gold]
        out["own_axis"] = {name: f1_per_label(truth, pred, axis_sides_of(axis)) for name, pred in sides.items()}
        return out
    predictions = {
        "generated": [codes.get(r["predicted_letter"]) if r["predicted_letter"] else None for r in own],
        "argmax_posterior": [codes[argmax_key(r["option_logprobs_posterior"])] for r in own],
        "argmax_lift": [codes[argmax_key({l: r["option_logprobs_posterior"][l] - r["option_logprobs_prior"][l]
                                          for l in r["option_logprobs_posterior"]})] for r in own],
    }
    if tasks is not None and ideal_class(agent_id, tasks[own[0]["task_id"]]) is not None:
        predictions["ideal"] = [ideal_class(agent_id, tasks[r["task_id"]]) for r in own]
    out["per_class"] = {name: f1_per_label(gold, pred, ws.core_classes()) for name, pred in predictions.items()}
    out["own_axis"] = {f"{name}:{a}": f1_per_label([side_of(g, a) for g in gold], [side_of(p, a) for p in pred],
                                                   axis_sides_of(a))
                       for name, pred in predictions.items() for a in axes}
    return out


# ----------------------------------------------------------------------------------------------- baselines


def guideline_rule_class(task: dict[str, Any]) -> str:
    """Untrained rule from the strict clinical key: fitness from mean resting HR vs the cohort median; metabolic
    adverse if any guideline flag is set (glucose >= 100, HbA1c >= 5.7, HOMA-IR >= 2.5, TG >= 150, low HDL)."""
    split = task["split"]
    rows = window_rows(task["rows"], task["end_date"], task["window_days"])
    rhr = [float(r["rhr"]) for r in rows if "rhr" in r]
    median = task["cohort_reference"]["rhr"]["p50"]
    fit = bool(rhr) and statistics.fmean(rhr) < median
    blood = (task["question"].get("context") or {}).get("blood_panel") or {}
    sex = ((task["question"].get("context") or {}).get("demographics") or {}).get("sex")
    homa = blood["glu"] * blood["ins"] / 405.0 if "glu" in blood and "ins" in blood else None
    flags = [blood.get("glu", 0) >= 100, blood.get("hgba1c", 0) >= 5.7, homa is not None and homa >= 2.5,
             blood.get("tg", 0) >= 150, "hdl" in blood and blood["hdl"] < (40 if sex == "m" else 50)]
    adverse = any(flags)
    assert split["id"] == task["task_id"], f"guideline_rule_class: task {task['task_id']} has a mismatched split"
    return {(True, False): "CH", (False, True): "CA", (False, False): "D-LF", (True, True): "D-FA"}[(fit, adverse)]


def guideline_rule_report(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy of the untrained guideline rule, overall and per gold class."""
    right = [(t["gold_class"], guideline_rule_class(t) == t["gold_class"]) for t in tasks]
    per_class = {cls: statistics.fmean(float(ok) for c, ok in right if c == cls)
                 for cls in ws.core_classes() if any(c == cls for c, _ in right)}
    return {"accuracy": statistics.fmean(float(ok) for _, ok in right), "per_class": per_class, "n": len(right)}


def task_feature_vector(task: dict[str, Any]) -> dict[str, float]:
    """28-day means of every metric, the six prompt blood values (with HOMA-IR), age, sex and BMI."""
    rows = window_rows(task["rows"], task["end_date"], task["window_days"])
    features: dict[str, float] = {}
    for metric in ws.all_metrics():
        values = [float(r[metric]) for r in rows if metric in r]
        features[f"mean_{metric}"] = statistics.fmean(values) if values else math.nan
    context = task["question"].get("context") or {}
    blood, demo = context.get("blood_panel") or {}, context.get("demographics") or {}
    for marker in ("glu", "hgba1c", "ins", "tg", "hdl"):
        features[f"blood_{marker}"] = float(blood[marker]) if marker in blood else math.nan
    features["homa_ir"] = (blood["glu"] * blood["ins"] / 405.0) if "glu" in blood and "ins" in blood else math.nan
    features["age"] = float(demo["age"]) if demo.get("age") is not None else math.nan
    features["bmi"] = float(demo["bmi"]) if demo.get("bmi") is not None else math.nan
    features["sex_m"] = 1.0 if demo.get("sex") == "m" else 0.0
    return features


def rf_views(tasks: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """One-agent views from out-of-fold random-forest probabilities (up to 5 folds grouped by user); prior = uniform."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    names = list(task_feature_vector(tasks[0]))
    X = np.array([[task_feature_vector(t)[n] for n in names] for t in tasks], dtype=float)
    y = np.array([t["gold_letter"] for t in tasks])
    groups = np.array([t["user_id"] for t in tasks])
    letters = sorted(ws.fixed_letter_map().values())
    proba = np.zeros((len(tasks), len(letters)))
    n_folds = min(5, len(set(groups.tolist())), max(int((y == letter).sum()) for letter in set(y.tolist())))
    assert n_folds >= 2, f"rf_views: only {n_folds} fold(s) possible; need more users or tasks"
    for train, test in StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y, groups):
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                     random_state=seed))
        model.fit(X[train], y[train])
        fold = model.predict_proba(X[test])
        for j, cls in enumerate(model.classes_):
            proba[test, letters.index(cls)] = fold[:, j]
    views = []
    for i, task in enumerate(tasks):
        posterior = {l: math.log(max(proba[i, j], 1e-6)) for j, l in enumerate(letters)}
        total = math.log(sum(math.exp(v) for v in posterior.values()))
        posterior = {l: v - total for l, v in posterior.items()}
        prior = {l: -math.log(len(letters)) for l in letters}
        views.append({"task_index": task["task_index"], "true_letter": task["gold_letter"], "letters": letters,
                      "user_id": task["user_id"], "gold_class": task["gold_class"], "task_id": task["task_id"],
                      "agents": [{"agent_index": 0, "prior": prior, "posterior": posterior,
                                  "lift": {l: posterior[l] - prior[l] for l in letters}, "generated": []}]})
    return views


# ----------------------------------------------------------------------------------------------- report


def analyze_wqa(path: Path, alphas: list[float], ks: list[int], scores: list[str], rules: list[str],
                n_splits: int = 200, seed: int = 0, exclude_from_calibration: tuple[str, ...] = ("D-FA",),
                tasks: list[dict[str, Any]] | None = None, with_rf: bool = False) -> dict[str, Any]:
    """Every metric for one predictions file (plus baselines when the tasks are given)."""
    records = load_wqa_predictions(path)
    users = [r["user_id"] for r in records]
    splits = make_group_splits(users, n_splits, seed)
    conditions = get_conditions(records)
    report: dict[str, Any] = {
        "predictions_file": str(path), "num_records": len(records),
        "num_tasks": len({r["task_index"] for r in records}), "num_users": len(set(users)),
        "prompt_variant": records[0].get("prompt_variant"), "model": records[0]["model_name"],
        "conditions": conditions,
        "agents": {a: agent_metrics(records, a) for a in dict.fromkeys(r["agent_id"] for r in records)},
        "f1": {a: agent_f1(records, a, {t["task_id"]: t for t in tasks} if tasks is not None else None)
               for a in dict.fromkeys(r["agent_id"] for r in records)},
        "conformal": {},
    }
    for name, agent_ids in conditions.items():
        views = condition_views(records, agent_ids)
        report["conformal"][name] = [
            run_grouped_conformal(views, splits, score, rule, alpha, k, exclude_from_calibration)
            for score in scores for rule in rules for alpha in alphas for k in ks if k <= len(agent_ids)]
    if tasks is not None:
        indices = {r["task_index"] for r in records}
        chosen = [t for t in tasks if t["task_index"] in indices]
        report["baselines"] = {"guideline_rule": guideline_rule_report(chosen)}
        if with_rf:
            views = rf_views(chosen, seed)
            report["baselines"]["random_forest"] = {
                "accuracy_argmax": statistics.fmean(
                    float(max(v["agents"][0]["posterior"], key=v["agents"][0]["posterior"].get) == v["true_letter"])
                    for v in views),
                "conformal": [run_grouped_conformal(views, splits, "lac", "joint", alpha, 1, exclude_from_calibration)
                              for alpha in alphas]}
    report["settings"] = {"alphas": alphas, "ks": ks, "scores": scores, "rules": rules, "n_splits": n_splits,
                          "seed": seed, "exclude_from_calibration": list(exclude_from_calibration)}
    return report


def headline_rows(report: dict[str, Any], score: str, rule: str, k: int) -> list[dict[str, Any]]:
    """One row per condition and alpha: coverage, mean set size, singleton rate."""
    rows = []
    for name, results in report["conformal"].items():
        for result in results:
            if (result["score"], result["rule"], result["k"]) == (score, rule, k):
                m = result["metrics"]
                rows.append({"condition": name, "alpha": result["alpha"], "coverage": m["coverage"]["mean"],
                             "mean_set_size": m["mean_set_size"]["mean"], "singleton_rate": m["singleton_rate"]["mean"]})
    return rows


def format_wqa_report(report: dict[str, Any], score: str, rule: str) -> str:
    """Plain-text tables: per-agent metrics, then the k=1 conformal grid for one score and rule."""
    lines = [f"{report['predictions_file']}", f"model {report['model']}, variant {report['prompt_variant']}, "
             f"{report['num_tasks']} tasks, {report['num_users']} users", "",
             f"{'agent':18s} {'axis':9s} {'acc_gen':>8s} {'acc_max':>8s} {'axis_acc':>8s} {'axis_mass':>9s}"]
    for agent_id, m in report["agents"].items():
        side = m["agent_axis"] if m["agent_axis"] in ("fitness", "metabolic") else None
        axis_acc = m.get(f"{side}_axis_accuracy_argmax", math.nan) if side else math.nan
        axis_mass = m.get(f"{side}_axis_mass", math.nan) if side else math.nan
        lines.append(f"{agent_id:18s} {m['agent_axis']:9s} {m['accuracy_generated']:8.3f} {m['accuracy_argmax']:8.3f} "
                     f"{axis_acc:8.3f} {axis_mass:9.3f}")
    lines += ["", "F1: agent, prediction, per class / own judgement (favourable, other side), macro"]
    for agent_id, f in report["f1"].items():
        for name, values in (f.get("per_class") or {}).items():
            cells = " ".join(f"{values[c]:.2f}" for c in ws.core_classes())
            lines.append(f"  {agent_id:18s} {name:17s} classes CH CA D-LF D-FA: {cells} | macro {values['macro']:.2f}")
        for name, values in f["own_axis"].items():
            cells = " ".join(f"{v:.2f}" for k, v in values.items() if k != "macro")
            lines.append(f"  {agent_id:18s} {name:17s} own judgement: {cells} | macro {values['macro']:.2f}")
    lines += ["", f"conformal ({score}, {rule}, k=1): condition, alpha, coverage, mean size, singleton rate"]
    for row in headline_rows(report, score, rule, 1):
        lines.append(f"  {row['condition']:28s} {row['alpha']:5.2f} {row['coverage']:6.3f} "
                     f"{row['mean_set_size']:6.2f} {row['singleton_rate']:6.3f}")
    if "baselines" in report:
        lines += ["", f"guideline rule accuracy {report['baselines']['guideline_rule']['accuracy']:.3f}"]
        if "random_forest" in report["baselines"]:
            lines.append(f"random forest (out-of-fold) argmax accuracy "
                         f"{report['baselines']['random_forest']['accuracy_argmax']:.3f}")
    return "\n".join(lines)


def save_wqa_report(report: dict[str, Any]) -> Path:
    """analysis/analysis_<predictions stem>.json next to the predictions file."""
    path = Path(report["predictions_file"])
    out_dir = path.parent / "analysis"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"analysis_{path.stem.removeprefix('predictions_')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def plot_wqa_report(report: dict[str, Any], score: str, rule: str) -> list[Path]:
    """Set size against alpha per condition, per-agent posterior heatmaps, coverage against set size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(report["predictions_file"])
    out_dir = path.parent / "analysis"
    out_dir.mkdir(exist_ok=True)
    stem = path.stem.removeprefix("predictions_")
    files = []
    rows = headline_rows(report, score, rule, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in dict.fromkeys(r["condition"] for r in rows):
        own = sorted((r["alpha"], r["mean_set_size"]) for r in rows if r["condition"] == name)
        ax.plot([a for a, _ in own], [s for _, s in own], marker="o", label=name)
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean set size")
    ax.set_title(f"Set size against alpha ({score}, {rule}, k=1)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    files.append(out_dir / f"set_size_vs_alpha_{stem}.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    agents = list(report["agents"])
    fig, axes = plt.subplots(1, len(agents), figsize=(3 * len(agents), 3), squeeze=False)
    for ax, agent_id in zip(axes[0], agents):
        table = report["agents"][agent_id]["mean_posterior_by_gold_class"]
        classes = list(table)
        options = ws.core_classes()
        ax.imshow([[table[c][o] for o in options] for c in classes], vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(options)), options, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(agent_id, fontsize=8)
    fig.suptitle("Mean posterior probability: gold class (rows) x option (columns)", fontsize=9)
    fig.tight_layout()
    files.append(out_dir / f"posterior_heatmaps_{stem}.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, results in report["conformal"].items():
        ax.scatter([r["metrics"]["mean_set_size"]["mean"] for r in results],
                   [r["metrics"]["coverage"]["mean"] for r in results], s=12, label=name)
    for alpha in report["settings"]["alphas"]:
        ax.axhline(1 - alpha, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("mean set size")
    ax.set_ylabel("coverage")
    ax.set_title("Coverage against set size, whole grid")
    ax.legend(fontsize=6)
    fig.tight_layout()
    files.append(out_dir / f"coverage_vs_size_{stem}.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyse WearableQA prediction files written by wqa_exp.py.")
    parser.add_argument("--predictions", nargs="+", required=True, help="one or more predictions_wqa_*.json files")
    parser.add_argument("--tasks", default=None, help="tasks.jsonl, for the guideline-rule and RF baselines")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.1, 0.2], help="target miss rates")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2], help="quorum sizes (1 = intersection)")
    parser.add_argument("--score", nargs="+", choices=["lift", "margin", "lac"], default=["lift", "margin", "lac"])
    parser.add_argument("--conformal_rule", nargs="+", choices=["joint", "markov"], default=["joint", "markov"])
    parser.add_argument("--splits", type=int, default=200, help="user-grouped calibration/test splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude_from_calibration", nargs="*", default=["D-FA"], help="gold classes kept out of "
                                                                                         "calibration")
    parser.add_argument("--rf", action="store_true", help="add the random-forest baseline (needs --tasks, sklearn)")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_save", action="store_true", help="print only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    """Analyse each file, print the tables, and save the report and plots under <run>/analysis/."""
    args = parse_args(argv)
    assert not args.rf or args.tasks, "main: --rf needs --tasks"
    tasks = load_tasks(Path(args.tasks).expanduser()) if args.tasks else None
    reports = []
    for path in args.predictions:
        report = analyze_wqa(Path(path).expanduser().resolve(), args.alpha, args.k, args.score, args.conformal_rule,
                             args.splits, args.seed, tuple(args.exclude_from_calibration), tasks, args.rf)
        print(format_wqa_report(report, args.score[0], args.conformal_rule[0]))
        if not args.no_save:
            print(f"\nSaved {save_wqa_report(report)}")
            if not args.no_plots:
                for figure in plot_wqa_report(report, args.score[0], args.conformal_rule[0]):
                    print(f"Saved {figure}")
        reports.append(report)
    return reports


if __name__ == "__main__":
    main()
