"""Split-conformal prediction sets per agent, fused by a k-veto quorum, evaluated over random task splits.

Each agent excludes option y when its nonconformity score s_i(y) exceeds a threshold. The quorum keeps y unless
at least k agents exclude it (k = 1 is plain intersection).

Two ways to set the threshold:
- "joint": per task, T = k-th largest true-answer score across its agents; the threshold is the conformal
  quantile of T over calibration tasks. Valid coverage for the fused set (tasks are the exchangeable units),
  with no Bonferroni penalty.
- "markov": per-agent threshold at level k*alpha/N (N = agents in the task) from pooled calibration agent
  scores. Coverage P(miss) <= sum(alpha_i)/k by Markov's inequality; approximate because agents of one task
  are pooled together.
"""

import math
import random
from typing import Any, Callable

from analysis.stats import mean, percentile


def score_lift(agent: dict[str, Any], letter: str) -> float:
    """Nonconformity = -lift: high when the agent's evidence pushes against the option."""
    return -agent["lift"][letter]


def score_lac(agent: dict[str, Any], letter: str) -> float:
    """Nonconformity = 1 - posterior probability (LAC)."""
    return 1.0 - math.exp(agent["posterior"][letter])


def get_score_functions() -> dict[str, Callable[[dict[str, Any], str], float]]:
    """Available nonconformity scores by name."""
    return {"lift": score_lift, "lac": score_lac}


def conformal_quantile(values: list[float], alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest value; +inf when n is too small for that level."""
    assert 0.0 < alpha < 1.0, f"conformal_quantile: alpha {alpha} must be in (0, 1)"
    assert values, "conformal_quantile: no calibration values"
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    return math.inf if rank > len(values) else sorted(values)[rank - 1]


def kth_largest_true_score(view: dict[str, Any], score_fn: Callable, k: int) -> float:
    """k-th largest true-answer score among the task's agents; -inf when it has fewer than k agents."""
    scores = sorted((score_fn(agent, view["true_letter"]) for agent in view["agents"]), reverse=True)
    return scores[k - 1] if len(scores) >= k else -math.inf


def quorum_set(view: dict[str, Any], score_fn: Callable, thresholds: list[float], k: int) -> list[str]:
    """Options excluded by fewer than k agents (thresholds[i] belongs to agent i)."""
    kept = []
    for letter in view["letters"]:
        exclusions = sum(score_fn(agent, letter) > threshold for agent, threshold in zip(view["agents"], thresholds))
        if exclusions < k:
            kept.append(letter)
    return kept


def calibrate(calibration: list[dict[str, Any]], score_fn: Callable, rule: str, alpha: float, k: int) -> Any:
    """Return what the chosen rule needs at test time (one threshold, or pooled agent scores)."""
    if rule == "joint":
        return conformal_quantile([kth_largest_true_score(v, score_fn, k) for v in calibration], alpha)
    assert rule == "markov", f"calibrate: unknown rule {rule!r}; use 'joint' or 'markov'"
    return [score_fn(agent, view["true_letter"]) for view in calibration for agent in view["agents"]]


def thresholds_for(view: dict[str, Any], calibrated: Any, rule: str, alpha: float, k: int) -> list[float]:
    """Per-agent thresholds for one test task."""
    if rule == "joint":
        return [calibrated] * len(view["agents"])
    agent_alpha = min(0.999, k * alpha / len(view["agents"]))
    return [conformal_quantile(calibrated, agent_alpha)] * len(view["agents"])


def evaluate_sets(test: list[dict[str, Any]], sets: list[list[str]]) -> dict[str, float]:
    """Coverage, set size, singleton rate and accuracy, empty rate on one test split."""
    sizes = [len(s) for s in sets]
    covered = [view["true_letter"] in s for view, s in zip(test, sets)]
    singletons = [view["true_letter"] in s for view, s in zip(test, sets) if len(s) == 1]
    return {
        "coverage": mean([float(c) for c in covered]),
        "mean_set_size": mean([float(n) for n in sizes]),
        "mean_relative_size": mean([len(s) / len(v["letters"]) for v, s in zip(test, sets)]),
        "singleton_rate": mean([float(n == 1) for n in sizes]),
        "singleton_accuracy": mean([float(c) for c in singletons]) if singletons else math.nan,
        "empty_rate": mean([float(n == 0) for n in sizes]),
    }


def summarize_splits(per_split: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and 10th/90th percentiles of each metric across splits (NaNs ignored)."""
    summary = {}
    for metric in per_split[0]:
        values = sorted(v[metric] for v in per_split if not math.isnan(v[metric]))
        summary[metric] = ({"mean": mean(values), "p10": percentile(values, 0.1), "p90": percentile(values, 0.9)}
                           if values else {"mean": math.nan, "p10": math.nan, "p90": math.nan})
    return summary


def run_conformal(views: list[dict[str, Any]], score: str, rule: str, alpha: float, k: int,
                  n_splits: int = 200, seed: int = 0) -> dict[str, Any]:
    """Calibrate on a random half of the tasks, test on the other half; repeat over n_splits splits."""
    assert len(views) >= 4, f"run_conformal: need at least 4 tasks, got {len(views)}"
    assert type(k) is int and k >= 1, f"run_conformal: k must be a positive integer, got {k!r}"
    score_fn = get_score_functions()[score]
    rng = random.Random(seed)
    per_split = []
    for _ in range(n_splits):
        order = list(range(len(views)))
        rng.shuffle(order)
        half = len(views) // 2
        calibration, test = [views[i] for i in order[:half]], [views[i] for i in order[half:]]
        calibrated = calibrate(calibration, score_fn, rule, alpha, k)
        sets = [quorum_set(v, score_fn, thresholds_for(v, calibrated, rule, alpha, k), k) for v in test]
        per_split.append(evaluate_sets(test, sets))
    return {"score": score, "rule": rule, "alpha": alpha, "k": k, "n_splits": n_splits,
            "n_calibration": len(views) // 2, "metrics": summarize_splits(per_split)}
