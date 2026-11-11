"""Analyse one or more predictions files: decision rules, baselines, conformal sets, statistics.

Example (from the repo root):
  python code/src/analyze.py --predictions code/results/<run_folder>/predictions_<stem>.json
"""

import argparse
from pathlib import Path

from analysis.report import analyze, format_report, save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyse prediction files written by main.py.")
    parser.add_argument("--predictions", nargs="+", required=True, help="one or more predictions_*.json files")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.1, 0.2], help="target miss rates")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2], help="quorum sizes (1 = intersection)")
    parser.add_argument("--score", nargs="+", choices=["lift", "margin", "lac"], default=["lift", "margin", "lac"])
    parser.add_argument("--conformal_rule", nargs="+", choices=["joint", "markov"], default=["joint", "markov"])
    parser.add_argument("--splits", type=int, default=200, help="random calibration/test splits")
    parser.add_argument("--bootstrap", type=int, default=5000, help="bootstrap resamples for intervals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_save", action="store_true", help="print only; do not write analysis_*.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict]:
    """Analyse each file, print the tables, and save analysis_<stem>.json next to it."""
    args = parse_args(argv)
    reports = []
    for path in args.predictions:
        report = analyze(Path(path).expanduser().resolve(), args.alpha, args.k, args.score, args.conformal_rule,
                         args.splits, args.bootstrap, args.seed)
        print(format_report(report))
        if not args.no_save:
            print(f"\nSaved {save_report(report)}\n")
        reports.append(report)
    return reports


if __name__ == "__main__":
    main()
