"""C4a: simulate the multi-round set protocol (variant A) offline on saved predictions files.

Example (from the repo root):
  python code/src/simulate_protocol.py --predictions code/results/<run_folder>/predictions_<stem>.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analysis.protocol import (agents_for_round, get_protocol_scores, get_schedules, load_rescore_cache,
                               make_folds, restricted_lift, run_multiround, shared_prior_rescore, split_alpha,
                               truth_score)
from analysis.scores import build_task_views, load_predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate the multi-round set protocol (C4a, variant A).")
    parser.add_argument("--predictions", nargs="+", required=True, help="one or more predictions_*.json files")
    parser.add_argument("--variant", choices=["A", "D"], default="A",
                        help="A = restrict and recalibrate on saved scores; D = belief update (needs --rescored)")
    parser.add_argument("--rescored", default=None, help="variant D: the rescored_*.json cache from belief_update.py")
    parser.add_argument("--lift_prior", choices=["conclusion", "shared"], default="conclusion",
                        help="variant D: prior pass context - 'conclusion' (group conclusion in both passes) or "
                             "'shared' (the empty-ruling prior: shared facts only, fixed across rounds)")
    parser.add_argument("--score", nargs="+", choices=get_protocol_scores(), default=["margin", "lift"],
                        help="lift is run at --rounds 0 only (no coverage argument for later rounds)")
    parser.add_argument("--rounds", nargs="+", type=int, default=[0, 1, 2, 3], help="extra rounds R after round 0")
    parser.add_argument("--schedule", nargs="+", choices=get_schedules(), default=["global", "spend"])
    parser.add_argument("--round0_share", nargs="*", type=float, default=[0.7],
                        help="spend: also run with this share of alpha in round 0 (equal shares always run)")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.1, 0.2], help="target miss rates")
    parser.add_argument("--cal_fraction", nargs="+", type=float, default=[0.5, 0.7], help="share of tasks calibrating")
    parser.add_argument("--splits", type=int, default=200, help="random calibration/test splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_control", action="store_true",
                        help="also run every config on agents shuffled across same-option tasks (facts vs mechanics)")
    parser.add_argument("--keep_infeasible", action="store_true",
                        help="keep spend configs whose smallest fold is too small for its alpha share (all inf tau)")
    parser.add_argument("--trace_tasks", type=int, default=0, help="print the round-by-round trajectory of n tasks")
    parser.add_argument("--no_save", action="store_true", help="print only; do not write protocol_A_*.json")
    return parser.parse_args(argv)


def smallest_fold(num_tasks: int, cal_fraction: float, rounds: int) -> int:
    """Calibration tasks per fold a spend run would get (its binding fold is the smallest)."""
    n_cal = num_tasks // 2 if cal_fraction == 0.5 else int(num_tasks * cal_fraction)
    return min(make_folds(list(range(n_cal)), rounds + 1), key=len).__len__() if n_cal >= rounds + 1 else 0


def spend_feasible(num_tasks: int, cal_fraction: float, alpha: float, rounds: int, share: float | None) -> bool:
    """True when every fold is large enough for its alpha share to give a finite threshold (n >= 1/alpha_r - 1)."""
    fold = smallest_fold(num_tasks, cal_fraction, rounds)
    return fold > 0 and fold >= 1.0 / min(split_alpha(alpha, rounds, share)) - 1.0


def build_configs(args: argparse.Namespace, num_tasks: int) -> list[dict[str, Any]]:
    """Runnable combinations; skips lift beyond round 0, spend at R = 0, and (unless --keep_infeasible) tiny-fold spend."""
    configs = []
    for score in args.score:
        for schedule in args.schedule:
            for rounds in args.rounds:
                assert rounds >= 0, f"build_configs: --rounds must be >= 0, got {rounds}"
                if (score == "lift" and rounds > 0) or (schedule == "spend" and rounds == 0):
                    continue
                shares = [None] + list(args.round0_share) if schedule == "spend" else [None]
                for share in shares:
                    for alpha in args.alpha:
                        for cal_fraction in args.cal_fraction:
                            if (schedule == "spend" and not args.keep_infeasible
                                    and not spend_feasible(num_tasks, cal_fraction, alpha, rounds, share)):
                                continue
                            configs.append({"score": score, "schedule": schedule, "rounds": rounds,
                                            "round0_share": share, "alpha": alpha, "cal_fraction": cal_fraction})
    assert configs, ("build_configs: no runnable configuration; on a small dataset spend needs a larger "
                     "--cal_fraction or alpha, or pass --keep_infeasible")
    return configs


def make_rescore(args: argparse.Namespace) -> Any:
    """The variant-D cache lookup with the lift-prior ablation applied, or None for variant A."""
    if args.variant == "A":
        return None
    assert args.rescored is not None, "simulate: --variant D needs --rescored <cache file from belief_update.py>"
    rescore = load_rescore_cache(Path(args.rescored).expanduser().resolve())
    return shared_prior_rescore(rescore) if args.lift_prior == "shared" else rescore


def simulate_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run every configuration on one predictions file (plus the shuffled control when asked)."""
    views = build_task_views(load_predictions(path))
    rescore = make_rescore(args)
    configs = build_configs(args, len(views))
    shuffles = [False, True] if args.shuffle_control else [False]
    results = [run_multiround(views, c["score"], c["schedule"], c["alpha"], c["rounds"], c["cal_fraction"],
                              c["round0_share"], args.splits, args.seed, shuffle, args.variant, rescore)
               for shuffle in shuffles for c in configs]
    return {"predictions_file": str(path), "variant": args.variant, "lift_prior": args.lift_prior,
            "num_tasks": len(views),
            "agents_per_task": [min(len(v["agents"]) for v in views), max(len(v["agents"]) for v in views)],
            "results": results}


def fmt_metric(metrics: dict[str, Any], name: str, digits: int = 2, spread: bool = False) -> str:
    """Mean of one metric (optionally with its 10th/90th percentiles) as text."""
    m = metrics[name]
    if math.isnan(m["mean"]):
        return "-"
    text = f"{m['mean']:.{digits}f}"
    return f"{text} [{m['p10']:.{digits}f}, {m['p90']:.{digits}f}]" if spread else text


def format_results(report: dict[str, Any]) -> str:
    """Printable table: one row per configuration."""
    mode = "" if report["variant"] == "A" else f" (lift prior: {report['lift_prior']})"
    lines = [f"Protocol {report['variant']}{mode} on {report['predictions_file']}",
             f"{report['num_tasks']} tasks, agents per task {report['agents_per_task'][0]}-{report['agents_per_task'][1]}",
             "",
             f"{'score':>6} {'schedule':>10} {'R':>2} {'alpha':>5} {'cal':>4} {'fold':>4} {'src':>4}  {'coverage':>8} "
             f"{'size':>5} {'decided':>18} {'dec acc':>7} {'empty':>5} {'rounds':>6} {'bits up':>7} {'inf tau':>7}"]
    for r in report["results"]:
        m = r["metrics"]
        name = r["schedule"] if r["round0_share"] is None else f"spend{r['round0_share']:g}"
        src = "shuf" if r.get("shuffle") else "real"
        lines.append(f"{r['score']:>6} {name:>10} {r['rounds']:>2} {r['alpha']:>5} {r['cal_fraction']:>4} "
                     f"{r['fold_size']:>4} {src:>4}  {fmt_metric(m, 'coverage', 3):>8} {fmt_metric(m, 'mean_set_size'):>5} "
                     f"{fmt_metric(m, 'singleton_rate', 2, True):>18} {fmt_metric(m, 'singleton_accuracy'):>7} "
                     f"{fmt_metric(m, 'empty_rate'):>5} {fmt_metric(m, 'mean_rounds'):>6} "
                     f"{fmt_metric(m, 'mean_bits_up', 1):>7} {fmt_metric(m, 'infinite_threshold_rounds'):>7}")
    lines += ["", "decided = share of test tasks ending with one option (mean [10th, 90th percentile] over splits); "
              "fold = calibration tasks per threshold; inf tau = rounds per split whose threshold was infinite "
              "(fold too small for its alpha share)"]
    return "\n".join(lines)


def format_trace(view: dict[str, Any], trajectory: dict[str, Any], score: str,
                 variant: str = "A", rescore: Any = None) -> str:
    """Round-by-round account of one task for checking by hand; variant D shows the cache's re-scored lifts."""
    lines = [f"task {trajectory['task_index']} (true {view['true_letter']}): stop={trajectory['stop']}, "
             f"final {trajectory['final_set']}"]
    full = list(view["letters"])
    for i, agent in enumerate(agents_for_round(view, full, variant, rescore)):
        lift = ", ".join(f"{y} {agent['lift'][y]:+.2f}" for y in full)
        lines.append(f"  agent {i} round-0 lift: {lift}")
    entering = full
    for r, (S, vetoes) in enumerate(zip(trajectory["sets"], trajectory["vetoes"])):
        lines.append(f"  round {r}: S={entering} truth score {truth_score(view, entering, score, variant, rescore):.2f}"
                     if view["true_letter"] in entering else f"  round {r}: S={entering}")
        if variant == "D" and len(entering) < len(full):
            for i, agent in enumerate(agents_for_round(view, entering, variant, rescore)):
                lifts = restricted_lift(agent, entering)
                lines.append(f"           agent {i} lift over S: "
                             + ", ".join(f"{y} {lifts[y]:+.2f}" for y in entering))
        lines.append(f"           vetoes {vetoes} -> {S}")
        entering = S
    return "\n".join(lines)


def choose_trace_config(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a config whose later rounds can actually act: real (not shuffled) margin, spend over global, R >= 1."""
    real = [r for r in results if not r.get("shuffle")]
    rank = lambda r: (r["score"] == "margin", r["schedule"] == "spend", r["rounds"] >= 1, r["rounds"])
    return max(real or results, key=rank)


def order_trace_tasks(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Front-load the instructive tasks: one decided after round 0, then the rest in order."""
    later = [t for t in trajectories if t["stop"] == "decided" and t["rounds_used"] > 1]
    rest = [t for t in trajectories if t not in later[:1]]
    return (later[:1] + rest) if later else trajectories


def print_traces(path: Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    """Trace n tasks of split 0 for the most informative configuration (a later round can change the set)."""
    views = {v["task_index"]: v for v in build_task_views(load_predictions(path))}
    chosen = choose_trace_config(report["results"])
    rescore = make_rescore(args)
    print(f"\nTraces: {chosen['score']} {chosen['schedule']} R={chosen['rounds']} alpha={chosen['alpha']} "
          f"cal={chosen['cal_fraction']}, thresholds {[round(t, 3) for t in chosen['thresholds_split0']]}")
    for trajectory in order_trace_tasks(chosen["trajectories_split0"])[:args.trace_tasks]:
        print(format_trace(views[trajectory["task_index"]], trajectory, chosen["score"], args.variant, rescore))


def save_results(report: dict[str, Any]) -> Path:
    """protocol_<variant>_<stem>.json next to the predictions file (D with the shared prior: protocol_Dsharedprior_)."""
    predictions = Path(report["predictions_file"])
    tag = report["variant"] + ("sharedprior" if report["variant"] == "D" and report["lift_prior"] == "shared" else "")
    target = predictions.with_name(predictions.name.replace("predictions_", f"protocol_{tag}_", 1))
    target.write_text(json.dumps(report, indent=2, default=lambda x: None), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    """Simulate each file, print the table (and traces), and save protocol_A_<stem>.json next to it."""
    args = parse_args(argv)
    reports = []
    for name in args.predictions:
        path = Path(name).expanduser().resolve()
        report = simulate_file(path, args)
        print(format_results(report))
        if args.trace_tasks > 0:
            print_traces(path, report, args)
        if not args.no_save:
            print(f"\nSaved {save_results(report)}\n")
        reports.append(report)
    return reports


if __name__ == "__main__":
    main()
