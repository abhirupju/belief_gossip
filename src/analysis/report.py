"""Build, print and save the analysis report for one predictions file."""

import json
import math
from pathlib import Path
from typing import Any

from analysis.conformal import run_conformal
from analysis.scores import (answer_key_majority_accuracy, build_answer_views, build_task_views, chance_accuracy,
                             correctness, get_rules, load_predictions)
from analysis.stats import bootstrap_ci, mean, paired_comparison


def rule_table(views: list[dict[str, Any]], rule_names: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    """Accuracy and bootstrap interval of each rule on a list of views."""
    rules = get_rules()
    table = {}
    for name in rule_names:
        right = correctness(views, rules[name])
        low, high = bootstrap_ci([float(r) for r in right], n_boot, seed)
        table[name] = {"accuracy": mean([float(r) for r in right]), "ci95": [low, high], "correct": right}
    return table


def get_comparisons() -> list[tuple[str, str]]:
    """Paired comparisons reported (rule A vs rule B)."""
    return [("lift_sum", "anti_prior"), ("prior_plus_lift", "anti_prior"), ("prior_plus_lift", "lift_sum"),
            ("prior_plus_lift", "posterior_sum"), ("prior_plus_lift", "generated_majority")]


def analyze(path: Path, alphas: list[float], ks: list[int], scores: list[str], conformal_rules: list[str],
            n_splits: int = 200, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Every metric for one predictions file."""
    records = load_predictions(path)
    task_views = build_task_views(records)
    answer_views = build_answer_views(records)
    rule_names = list(get_rules())
    per_task = rule_table(task_views, rule_names, n_boot, seed)
    per_answer = rule_table(answer_views, [n for n in rule_names if n != "generated_majority"], n_boot, seed)
    per_answer["generated"] = {"accuracy": mean([float(r["correct"]) for r in records]),
                               "ci95": list(bootstrap_ci([float(r["correct"]) for r in records], n_boot, seed))}
    comparisons = {f"{a}_vs_{b}": paired_comparison(per_task[a]["correct"], per_task[b]["correct"])
                   for a, b in get_comparisons()}
    conformal = [run_conformal(task_views, score, rule, alpha, k, n_splits, seed)
                 for score in scores for rule in conformal_rules for alpha in alphas for k in ks]
    agents_per_task = [len(v["agents"]) for v in task_views]
    return {
        "predictions_file": str(path),
        "num_records": len(records),
        "num_tasks": len(task_views),
        "agents_per_task": {"min": min(agents_per_task), "max": max(agents_per_task),
                            "mean": mean([float(n) for n in agents_per_task])},
        "rotations_per_agent": mean([float(len(a["generated"])) for v in task_views for a in v["agents"]]),
        "baselines": {"chance": chance_accuracy(task_views),
                      "answer_key_majority_letter": answer_key_majority_accuracy(task_views)},
        "per_task": {k: {kk: vv for kk, vv in v.items() if kk != "correct"} for k, v in per_task.items()},
        "per_answer": {k: {kk: vv for kk, vv in v.items() if kk != "correct"} for k, v in per_answer.items()},
        "comparisons": comparisons,
        "conformal": conformal,
        "settings": {"alphas": alphas, "ks": ks, "scores": scores, "conformal_rules": conformal_rules,
                     "n_splits": n_splits, "n_boot": n_boot, "seed": seed},
    }


def fmt(value: float) -> str:
    """Three decimals, or '-' for NaN."""
    return "-" if isinstance(value, float) and math.isnan(value) else f"{value:.3f}"


def format_report(report: dict[str, Any]) -> str:
    """Plain-text tables for the terminal."""
    lines = [f"Analysis of {report['predictions_file']}",
             f"{report['num_tasks']} tasks, {report['num_records']} records, agents per task "
             f"{report['agents_per_task']['min']}-{report['agents_per_task']['max']}, "
             f"rotations per agent {report['rotations_per_agent']:.1f}", ""]
    key = report["baselines"]["answer_key_majority_letter"]
    lines += [f"Chance: {fmt(report['baselines']['chance'])}   Always '{key['letter']}' (answer-key oracle): "
              f"{fmt(key['accuracy'])}   true letters: {key['true_letter_counts']}", ""]
    lines.append("Per task (scores averaged over rotations, fused over agents)")
    lines.append(f"  {'rule':<22}{'accuracy':>10}{'95% CI':>20}")
    for name, row in report["per_task"].items():
        lines.append(f"  {name:<22}{fmt(row['accuracy']):>10}{'[' + fmt(row['ci95'][0]) + ', ' + fmt(row['ci95'][1]) + ']':>20}")
    lines += ["", "Per answer (each record on its own)"]
    for name, row in report["per_answer"].items():
        lines.append(f"  {name:<22}{fmt(row['accuracy']):>10}{'[' + fmt(row['ci95'][0]) + ', ' + fmt(row['ci95'][1]) + ']':>20}")
    lines += ["", "Paired comparisons per task (only A right / only B right, sign-test p)"]
    for name, row in report["comparisons"].items():
        lines.append(f"  {name:<44}{row['only_a']:>4} /{row['only_b']:>4}   p={row['p_value']:.3f}")
    lines += ["", "Conformal sets (mean over random 50/50 task splits; [10th, 90th percentile])",
              f"  {'score':<6}{'rule':<8}{'alpha':>6}{'k':>3}{'coverage':>22}{'set size':>22}"
              f"{'singletons':>12}{'singleton acc':>15}"]
    for row in report["conformal"]:
        m = row["metrics"]
        lines.append(f"  {row['score']:<6}{row['rule']:<8}{row['alpha']:>6}{row['k']:>3}"
                     f"{fmt(m['coverage']['mean']) + ' [' + fmt(m['coverage']['p10']) + ', ' + fmt(m['coverage']['p90']) + ']':>22}"
                     f"{fmt(m['mean_set_size']['mean']) + ' [' + fmt(m['mean_set_size']['p10']) + ', ' + fmt(m['mean_set_size']['p90']) + ']':>22}"
                     f"{fmt(m['singleton_rate']['mean']):>12}{fmt(m['singleton_accuracy']['mean']):>15}")
    return "\n".join(lines)


def save_report(report: dict[str, Any]) -> Path:
    """analysis_<stem>.json next to the predictions file."""
    predictions = Path(report["predictions_file"])
    target = predictions.with_name(predictions.name.replace("predictions_", "analysis_", 1))
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
