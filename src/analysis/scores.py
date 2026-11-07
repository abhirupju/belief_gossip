"""Turn prediction records into per-task views and apply decision and fusion rules.

Notation (per option y): prior(y) = log p(y | shared context), posterior(y) = log p(y | context + agent's facts),
lift(y) = posterior(y) - prior(y). All scores are already in the dataset's own option order.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from utility.model_utils import logsumexp


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Read a predictions file and check it has the option scores the analysis needs."""
    path = Path(path)
    assert path.is_file(), f"load_predictions: {path} does not exist"
    records = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and records, f"load_predictions: {path} holds no records"
    needed = ["task_index", "agent_index", "rotation", "true_letter", "predicted_letter",
              "option_logprobs_prior", "option_logprobs_posterior"]
    missing = [key for key in needed if key not in records[0]]
    assert not missing, f"load_predictions: records in {path} lack {missing}; rerun with the current code"
    assert all(r["option_logprobs_prior"] is not None and r["option_logprobs_posterior"] is not None
               for r in records), (
        f"load_predictions: {path} has records without option scores (reasoning-mode runs have none); "
        f"analyse an answer-only run"
    )
    return records


def normalize(scores: dict[str, float]) -> dict[str, float]:
    """Log-softmax over the options (scores become log-probabilities summing to 1)."""
    total = logsumexp(list(scores.values()))
    return {letter: value - total for letter, value in scores.items()}


def average_scores(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Mean of several per-option score dicts (e.g. over rotations)."""
    assert score_dicts, "average_scores: nothing to average"
    letters = sorted(score_dicts[0])
    assert all(sorted(d) == letters for d in score_dicts), "average_scores: score dicts have different options"
    return {letter: sum(d[letter] for d in score_dicts) / len(score_dicts) for letter in letters}


def lift_of(prior: dict[str, float], posterior: dict[str, float]) -> dict[str, float]:
    """lift(y) = posterior(y) - prior(y)."""
    return {letter: posterior[letter] - prior[letter] for letter in prior}


def add_scores(*score_dicts: dict[str, float]) -> dict[str, float]:
    """Element-wise sum of per-option score dicts."""
    letters = sorted(score_dicts[0])
    return {letter: sum(d[letter] for d in score_dicts) for letter in letters}


def argmax(scores: dict[str, float]) -> str:
    """Option with the highest score (ties broken by letter order)."""
    return max(sorted(scores), key=lambda letter: scores[letter])


def argmin(scores: dict[str, float]) -> str:
    """Option with the lowest score (ties broken by letter order)."""
    return min(sorted(scores), key=lambda letter: scores[letter])


def agent_view(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One agent on one task: scores averaged over its rotations, plus its generated letters."""
    prior = normalize(average_scores([r["option_logprobs_prior"] for r in records]))
    posterior = normalize(average_scores([r["option_logprobs_posterior"] for r in records]))
    return {"agent_index": records[0]["agent_index"], "prior": prior, "posterior": posterior,
            "lift": lift_of(prior, posterior), "generated": [r["predicted_letter"] for r in records]}


def build_task_views(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per task: true letter and one rotation-averaged view per agent."""
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["task_index"], {}).setdefault(record["agent_index"], []).append(record)
    views = []
    for task_index, agents in sorted(grouped.items()):
        any_record = next(iter(agents.values()))[0]
        views.append({
            "task_index": task_index,
            "true_letter": any_record["true_letter"],
            "letters": sorted(any_record["option_logprobs_prior"]),
            "agents": [agent_view(agent_records) for _, agent_records in sorted(agents.items())],
        })
    return views


def build_answer_views(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every record as its own one-agent 'task' (no rotation averaging)."""
    return [{"task_index": r["task_index"], "true_letter": r["true_letter"],
             "letters": sorted(r["option_logprobs_prior"]), "agents": [agent_view([r])]} for r in records]


def shared_prior(view: dict[str, Any]) -> dict[str, float]:
    """The task's prior: the mean of the agents' priors (identical when agents share model and context)."""
    return normalize(average_scores([agent["prior"] for agent in view["agents"]]))


def rule_generated_majority(view: dict[str, Any]) -> str | None:
    """Majority of all generated letters on the task; a tie or no parseable answer gives None (wrong)."""
    votes = Counter(letter for agent in view["agents"] for letter in agent["generated"] if letter is not None)
    if not votes:
        return None
    ranked = votes.most_common()
    return None if len(ranked) > 1 and ranked[0][1] == ranked[1][1] else ranked[0][0]


def rule_prior(view: dict[str, Any]) -> str:
    """Most likely option given the shared context only."""
    return argmax(shared_prior(view))


def rule_anti_prior(view: dict[str, Any]) -> str:
    """Least likely option under the shared prior (baseline for 'move away from the shared facts')."""
    return argmin(shared_prior(view))


def rule_lift_sum(view: dict[str, Any]) -> str:
    """argmax of the summed lifts (drops the prior)."""
    return argmax(add_scores(*[agent["lift"] for agent in view["agents"]]))


def rule_prior_plus_lift(view: dict[str, Any]) -> str:
    """Product of experts: argmax of prior + sum of lifts (Bayes-optimal if facts are independent given y)."""
    return argmax(add_scores(shared_prior(view), *[agent["lift"] for agent in view["agents"]]))


def rule_posterior_sum(view: dict[str, Any]) -> str:
    """argmax of summed posteriors (counts the shared prior once per agent)."""
    return argmax(add_scores(*[agent["posterior"] for agent in view["agents"]]))


def get_rules() -> dict[str, Callable[[dict[str, Any]], str | None]]:
    """Every decision rule the report evaluates, by name."""
    return {
        "generated_majority": rule_generated_majority,
        "prior": rule_prior,
        "anti_prior": rule_anti_prior,
        "posterior_sum": rule_posterior_sum,
        "lift_sum": rule_lift_sum,
        "prior_plus_lift": rule_prior_plus_lift,
    }


def correctness(views: list[dict[str, Any]], rule: Callable[[dict[str, Any]], str | None]) -> list[bool]:
    """Per-view right/wrong for a rule."""
    return [rule(view) == view["true_letter"] for view in views]


def chance_accuracy(views: list[dict[str, Any]]) -> float:
    """Expected accuracy of uniform guessing."""
    return sum(1 / len(view["letters"]) for view in views) / len(views)


def answer_key_majority_accuracy(views: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of the most common true letter (an oracle baseline: it reads the answer key)."""
    counts = Counter(view["true_letter"] for view in views)
    letter, count = counts.most_common(1)[0]
    return {"letter": letter, "accuracy": count / len(views), "true_letter_counts": dict(sorted(counts.items()))}
