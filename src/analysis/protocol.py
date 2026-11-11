"""C4a: offline multi-round set protocol, variant A (restrict and recalibrate), on saved round-0 scores.

Round r: the hub has broadcast the surviving set S_r (S_0 = every option). Each agent renormalises its own saved
prior and posterior over S_r, scores every option in S_r, and vetoes the options whose score exceeds tau_r. The hub
keeps the options no agent vetoed (k = 1) and broadcasts the new set. The protocol stops when one option is left
(decided), none is left (empty: a miss), nothing changed and no later round is stricter (no_change), or after
round R (max_rounds). Agents never see each other's scores or facts; the hub never sees the true answer.

Scores of option y in S (nonconformity, high = evidence against y):
- lift:   -lift_S(y), with lift_S(y) = lift(y) - log[q(S) / p(S)] (prior p and posterior q renormalised over S)
- margin: max over y' in S of lift(y') - lift(y) (the renormalisation constant cancels)

Threshold schedules (one per calibration split):
- global: one tau for every round = the joint conformal quantile of the round-0 truth score. Under margin the later
  rounds can never veto anything new (margins only shrink as S shrinks), so the result equals the one-round set.
- spend:  alpha_0 + ... + alpha_R = alpha; the calibration tasks are split into R + 1 disjoint folds; tau_r is the
  conformal quantile at alpha_r of the round-r truth score on fold r, with tau_0 .. tau_{r-1} already fixed.
  P(truth vetoed in round r) <= alpha_r, so P(miss) <= alpha by the union bound.
"""

import math
import random
from typing import Any

from analysis.conformal import conformal_quantile, evaluate_sets, summarize_splits
from utility.model_utils import logsumexp


def get_protocol_scores() -> list[str]:
    """Scores the protocol supports."""
    return ["margin", "lift"]


def get_schedules() -> list[str]:
    """Threshold schedules the protocol supports."""
    return ["global", "spend"]


def check_subset(agent: dict[str, Any], S: list[str]) -> None:
    """Assert S is a non-empty list of distinct options the agent has scores for."""
    assert S, "check_subset: the surviving set S is empty"
    assert len(set(S)) == len(S), f"check_subset: S {S} repeats an option"
    unknown = [y for y in S if y not in agent["lift"]]
    assert not unknown, f"check_subset: options {unknown} in S are not among the agent's options {sorted(agent['lift'])}"


def restrict_logprobs(logprobs: dict[str, float], S: list[str]) -> dict[str, float]:
    """Renormalise log-probabilities over the options in S (condition on 'the answer is in S')."""
    assert S and all(y in logprobs for y in S), f"restrict_logprobs: S {S} is empty or not within {sorted(logprobs)}"
    total = logsumexp([logprobs[y] for y in S])
    return {y: logprobs[y] - total for y in S}


def restricted_lift(agent: dict[str, Any], S: list[str]) -> dict[str, float]:
    """Lift over S: lift(y) minus log[q(S)/p(S)]; exactly the saved lift when S holds every option."""
    check_subset(agent, S)
    if len(S) == len(agent["lift"]):
        return {y: agent["lift"][y] for y in S}
    offset = logsumexp([agent["posterior"][y] for y in S]) - logsumexp([agent["prior"][y] for y in S])
    return {y: agent["lift"][y] - offset for y in S}


def restricted_scores(agent: dict[str, Any], S: list[str], score: str) -> dict[str, float]:
    """Nonconformity score of every option in S for one agent (lift or margin over S)."""
    assert score in get_protocol_scores(), f"restricted_scores: unknown score {score!r}; use {get_protocol_scores()}"
    lift = restricted_lift(agent, S)
    if score == "lift":
        return {y: -value for y, value in lift.items()}
    best = max(lift.values())
    return {y: best - value for y, value in lift.items()}


def agent_vetoes(agent: dict[str, Any], S: list[str], score: str, tau: float) -> list[str]:
    """The options in S this agent vetoes at threshold tau (its veto bits); uses only its own scores and S."""
    scores = restricted_scores(agent, S, score)
    return [y for y in S if scores[y] > tau]


def run_protocol(view: dict[str, Any], score: str, thresholds: list[float],
                 stop_when_unchanged: bool = True) -> dict[str, Any]:
    """Run rounds 0..R (R = len(thresholds) - 1) on one task; never reads the true answer."""
    assert thresholds, "run_protocol: need at least one threshold (round 0)"
    assert view["agents"], f"run_protocol: task {view.get('task_index')} has no agents"
    letters = list(view["letters"])
    n_agents = len(view["agents"])
    S = list(letters)
    sets, vetoes = [], []
    bits_up = bits_down = 0
    stop = "max_rounds"
    for r, tau in enumerate(thresholds):
        round_vetoes = [agent_vetoes(agent, S, score, tau) for agent in view["agents"]]
        vetoed = {y for agent_veto in round_vetoes for y in agent_veto}
        new_S = [y for y in S if y not in vetoed]
        bits_up += n_agents * len(S)
        bits_down += len(letters)
        sets.append(new_S)
        vetoes.append(round_vetoes)
        if not new_S:
            stop = "empty"
            break
        if len(new_S) == 1:
            stop = "decided"
            break
        later = thresholds[r + 1:]
        if stop_when_unchanged and later and new_S == S and min(later) >= tau:
            stop = "no_change"
            break
        S = new_S
    return {"task_index": view.get("task_index"), "sets": sets, "vetoes": vetoes, "stop": stop,
            "rounds_used": len(sets), "bits_up": bits_up, "bits_down": bits_down, "final_set": sets[-1]}


def truth_score(view: dict[str, Any], S: list[str], score: str) -> float:
    """Largest score any agent gives the true answer within S (the truth is vetoed when this exceeds tau)."""
    truth = view["true_letter"]
    assert truth in S, f"truth_score: true answer {truth} is not in S {S}"
    return max(restricted_scores(agent, S, score)[truth] for agent in view["agents"])


def truth_score_at_round(view: dict[str, Any], score: str, earlier_thresholds: list[float]) -> float:
    """Truth score in round r = len(earlier_thresholds), after rounds 0..r-1; -inf when the truth is not at risk."""
    if not earlier_thresholds:
        return truth_score(view, list(view["letters"]), score)
    trajectory = run_protocol(view, score, earlier_thresholds, stop_when_unchanged=False)
    S = trajectory["final_set"]
    if trajectory["stop"] != "max_rounds" or view["true_letter"] not in S:
        return -math.inf
    return truth_score(view, S, score)


def floor_threshold(tau: float, score: str) -> float:
    """Margin thresholds never go below 0 (an agent never vetoes its own top option); raising tau is conservative."""
    return max(tau, 0.0) if score == "margin" else tau


def calibrate_global(calibration: list[dict[str, Any]], score: str, alpha: float, rounds: int) -> list[float]:
    """One threshold for every round: the joint conformal quantile of the round-0 truth score."""
    assert rounds >= 0, f"calibrate_global: rounds must be >= 0, got {rounds}"
    tau = conformal_quantile([truth_score(v, list(v["letters"]), score) for v in calibration], alpha)
    return [floor_threshold(tau, score)] * (rounds + 1)


def split_alpha(alpha: float, rounds: int, round0_share: float | None = None) -> list[float]:
    """alpha_0 .. alpha_R summing to alpha: equal shares, or round0_share of alpha in round 0 and the rest equal."""
    assert 0.0 < alpha < 1.0, f"split_alpha: alpha {alpha} must be in (0, 1)"
    assert rounds >= 0, f"split_alpha: rounds must be >= 0, got {rounds}"
    if rounds == 0:
        return [alpha]
    if round0_share is None:
        return [alpha / (rounds + 1)] * (rounds + 1)
    assert 0.0 < round0_share < 1.0, f"split_alpha: round0_share {round0_share} must be in (0, 1)"
    return [alpha * round0_share] + [alpha * (1.0 - round0_share) / rounds] * rounds


def make_folds(calibration: list[dict[str, Any]], n_folds: int) -> list[list[dict[str, Any]]]:
    """Split the (already shuffled) calibration tasks into n_folds disjoint, nearly equal folds."""
    assert 1 <= n_folds <= len(calibration), (
        f"make_folds: cannot split {len(calibration)} calibration tasks into {n_folds} folds")
    size, extra = divmod(len(calibration), n_folds)
    folds, start = [], 0
    for f in range(n_folds):
        end = start + size + (1 if f < extra else 0)
        folds.append(calibration[start:end])
        start = end
    return folds


def calibrate_spend(calibration: list[dict[str, Any]], score: str, alphas: list[float]) -> list[float]:
    """tau_r from fold r at level alpha_r, with tau_0 .. tau_{r-1} fixed (alpha spending + union bound)."""
    folds = make_folds(calibration, len(alphas))
    thresholds: list[float] = []
    for fold, alpha_r in zip(folds, alphas):
        values = [truth_score_at_round(v, score, thresholds) for v in fold]
        thresholds.append(floor_threshold(conformal_quantile(values, alpha_r), score))
    return thresholds


def calibrate(calibration: list[dict[str, Any]], score: str, schedule: str, alpha: float, rounds: int,
              round0_share: float | None = None) -> list[float]:
    """Thresholds [tau_0 .. tau_R] for the chosen schedule."""
    assert schedule in get_schedules(), f"calibrate: unknown schedule {schedule!r}; use {get_schedules()}"
    if schedule == "global":
        return calibrate_global(calibration, score, alpha, rounds)
    return calibrate_spend(calibration, score, split_alpha(alpha, rounds, round0_share))


def first_miss_round(view: dict[str, Any], trajectory: dict[str, Any]) -> int | None:
    """Round in which the true answer was vetoed, or None when it survived."""
    for r, S in enumerate(trajectory["sets"]):
        if view["true_letter"] not in S:
            return r
    return None


def evaluate_protocol(test: list[dict[str, Any]], score: str, thresholds: list[float]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Run the protocol on test tasks; return the metrics and the trajectories (with coverage added)."""
    trajectories = [run_protocol(view, score, thresholds) for view in test]
    metrics = evaluate_sets(test, [t["final_set"] for t in trajectories])
    rounds = len(thresholds)
    metrics["mean_rounds"] = sum(t["rounds_used"] for t in trajectories) / len(trajectories)
    metrics["mean_bits_up"] = sum(t["bits_up"] for t in trajectories) / len(trajectories)
    metrics["mean_bits_down"] = sum(t["bits_down"] for t in trajectories) / len(trajectories)
    for r in range(rounds):
        metrics[f"decided_round_{r}"] = sum(t["stop"] == "decided" and t["rounds_used"] == r + 1
                                            for t in trajectories) / len(trajectories)
        metrics[f"missed_round_{r}"] = sum(first_miss_round(v, t) == r for v, t in zip(test, trajectories)) / len(test)
    for view, trajectory in zip(test, trajectories):
        trajectory["true_letter"] = view["true_letter"]
        trajectory["covered"] = view["true_letter"] in trajectory["final_set"]
    return metrics, trajectories


def split_tasks(n_tasks: int, cal_fraction: float, rng: random.Random) -> tuple[list[int], list[int]]:
    """Random calibration/test task indices; cal_fraction 0.5 reproduces run_conformal's split exactly."""
    assert 0.0 < cal_fraction < 1.0, f"split_tasks: cal_fraction {cal_fraction} must be in (0, 1)"
    order = list(range(n_tasks))
    rng.shuffle(order)
    n_cal = n_tasks // 2 if cal_fraction == 0.5 else int(n_tasks * cal_fraction)
    assert 1 <= n_cal < n_tasks, f"split_tasks: {n_tasks} tasks leave no calibration or no test tasks at {cal_fraction}"
    calibration, test = order[:n_cal], order[n_cal:]
    assert not set(calibration) & set(test), "split_tasks: calibration and test tasks overlap"
    return calibration, test


def shuffle_agents(views: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Reassign each task's agents to a different task with the same options, keeping its true answer and options.

    A control: the evidence an agent brings no longer belongs to the task it lands on, so coverage must still hold
    (calibration adapts) but the decisive rate must fall toward the null level if the real gains came from the facts.
    """
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, view in enumerate(views):
        groups.setdefault(tuple(view["letters"]), []).append(i)
    source_of = list(range(len(views)))
    for letters, idx in groups.items():
        donors = idx[:]
        rng.shuffle(donors)
        if len(donors) > 1:  # derange so no task keeps its own agents
            while any(d == i for d, i in zip(donors, idx)):
                rng.shuffle(donors)
        for target, donor in zip(idx, donors):
            source_of[target] = donor
    return [{**views[target], "agents": views[source_of[target]]["agents"]} for target in range(len(views))]


def run_multiround(views: list[dict[str, Any]], score: str, schedule: str, alpha: float, rounds: int,
                   cal_fraction: float = 0.5, round0_share: float | None = None,
                   n_splits: int = 200, seed: int = 0, shuffle: bool = False) -> dict[str, Any]:
    """Calibrate and test the protocol over random task splits; metrics as mean and 10th/90th percentiles."""
    assert len(views) >= 4, f"run_multiround: need at least 4 tasks, got {len(views)}"
    assert score in get_protocol_scores(), f"run_multiround: unknown score {score!r}"
    assert score == "margin" or rounds == 0, (
        f"run_multiround: score {score!r} has no coverage argument beyond round 0; use rounds 0 or score 'margin'")
    if shuffle:
        views = shuffle_agents(views, seed)
    rng = random.Random(seed)
    per_split, kept_trajectories, thresholds_seen = [], [], []
    for split in range(n_splits):
        cal_idx, test_idx = split_tasks(len(views), cal_fraction, rng)
        calibration, test = [views[i] for i in cal_idx], [views[i] for i in test_idx]
        thresholds = calibrate(calibration, score, schedule, alpha, rounds, round0_share)
        metrics, trajectories = evaluate_protocol(test, score, thresholds)
        metrics["infinite_threshold_rounds"] = float(sum(math.isinf(t) for t in thresholds))
        per_split.append(metrics)
        thresholds_seen.append(thresholds)
        if split == 0:
            kept_trajectories = trajectories
    n_cal = len(split_tasks(len(views), cal_fraction, random.Random(0))[0])
    return {"variant": "A", "score": score, "schedule": schedule, "alpha": alpha, "rounds": rounds,
            "cal_fraction": cal_fraction, "round0_share": round0_share, "shuffle": shuffle,
            "n_splits": n_splits, "seed": seed,
            "n_calibration": n_cal, "fold_size": n_cal // (rounds + 1) if schedule == "spend" else n_cal,
            "alphas": split_alpha(alpha, rounds, round0_share) if schedule == "spend" else [alpha],
            "metrics": summarize_splits(per_split), "thresholds_split0": thresholds_seen[0],
            "trajectories_split0": kept_trajectories}
