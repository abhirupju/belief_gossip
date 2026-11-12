"""Prepare the WearableQA phenotype tasks: raw JSON -> tasks.jsonl (one self-contained task per question).

Each task holds the per-device split (rows and context per agent), the question itself (for the original-prompt
baseline), the question's 28-day rows over all metrics, the cohort reference and the gold labels. A preview file with
the rendered prompts of the first few tasks is written next to it for reading by eye (E0).

Example (from code/):
  python src/wqa_prepare.py --raw data/wearableqa/WearableQA_raw.json --preview 10
"""

import argparse
import json
from pathlib import Path
from typing import Any

from utility import wqa_split as ws
from utility.wqa_original import render_original
from utility.wqa_prompts import load_wqa_templates, render_agent, render_centralised


def get_task_schema_version() -> int:
    """Version of the task record format; bump when a field changes meaning."""
    return 1


def default_tasks_path(data_dir: Path) -> Path:
    """data/wearableqa/tasks.jsonl."""
    return Path(data_dir) / "wearableqa" / "tasks.jsonl"


def build_task(task_index: int, question: dict[str, Any], raw: dict[str, Any],
               holders: dict[str, dict[str, list[str]]] | None = None, window_days: int = 28) -> dict[str, Any]:
    """One task: the checked per-device split plus everything the baselines need."""
    split = ws.split_question(question, raw, holders, window_days=window_days, history_days=0)
    ws.check_partition(question, split, raw)
    return {
        "schema_version": get_task_schema_version(),
        "task_index": task_index,
        "task_id": question["id"],
        "user_id": str(question["user_id"]),
        "end_date": question["end_date"],
        "gold_class": ws.gold_class(question),
        "gold_letter": ws.fixed_gold_letter(question),
        "original_gt_letter": question["gt_letter"],
        "split": split,
        "question": question,
        "rows": ws.question_rows(question, raw, window_days, 0),
        "cohort_reference": raw["cohort_reference"],
        "window_days": window_days,
    }


def build_tasks(raw: dict[str, Any], holders: dict[str, dict[str, list[str]]] | None = None,
                window_days: int = 28) -> list[dict[str, Any]]:
    """Every multimodal-phenotype question as a task, in file order."""
    questions = ws.select_questions(raw, "multimodal_phenotype")
    assert questions, "build_tasks: no multimodal_phenotype questions in the raw file"
    return [build_task(i, q, raw, holders, window_days) for i, q in enumerate(questions)]


def write_tasks(path: Path, tasks: list[dict[str, Any]]) -> Path:
    """One JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    return path


def load_tasks(path: Path) -> list[dict[str, Any]]:
    """Read tasks.jsonl and check its schema version and task order."""
    path = Path(path)
    assert path.is_file(), f"load_tasks: {path} does not exist; run src/wqa_prepare.py first"
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert tasks, f"load_tasks: {path} holds no tasks"
    versions = {t.get("schema_version") for t in tasks}
    assert versions == {get_task_schema_version()}, (
        f"load_tasks: {path} has schema versions {versions}; expected {get_task_schema_version()}; prepare it again"
    )
    assert [t["task_index"] for t in tasks] == list(range(len(tasks))), f"load_tasks: {path} task_index out of order"
    return tasks


def ensure_tasks(tasks_path: Path, raw_path: Path, window_days: int = 28) -> list[dict[str, Any]]:
    """Load tasks.jsonl, building it from the raw file first if it does not exist."""
    if not Path(tasks_path).is_file():
        write_tasks(tasks_path, build_tasks(ws.load_raw(raw_path), window_days=window_days))
    tasks = load_tasks(tasks_path)
    windows = {t["window_days"] for t in tasks}
    assert windows == {window_days}, (
        f"ensure_tasks: {tasks_path} was built with window_days {sorted(windows)}, the config asks for {window_days}; "
        f"delete it (it is rebuilt from {raw_path}) or run src/wqa_prepare.py --window_days {window_days}"
    )
    return tasks


def preview_text(tasks: list[dict[str, Any]], templates: dict[str, Any], n: int) -> str:
    """Rendered prompts (every agent, prior, centralised, original) of the first n tasks, with lengths."""
    agents = list(templates["agents"])
    out = []
    for task in tasks[:n]:
        split = task["split"]
        out.append(f"{'#' * 100}\nTASK {task['task_index']} {task['task_id']}  gold {task['gold_class']} "
                   f"({task['gold_letter']})\n")
        for agent in agents:
            system, user = render_agent(templates, agent, split, window_days=task["window_days"])
            out.append(f"----- {agent} ({len(system) + len(user)} chars)\n[system]\n{system}\n[user]\n{user}\n")
        system, user = render_centralised(templates, split, agents, window_days=task["window_days"])
        out.append(f"----- centralised ({len(system) + len(user)} chars)\n[system]\n{system}\n[user]\n{user}\n")
        _, user = render_original(task["question"], task["rows"], task["cohort_reference"],
                                  window_days=task["window_days"])
        out.append(f"----- original ({len(user)} chars)\n{user}\n")
    if tasks:
        split = tasks[0]["split"]
        for agent in agents:
            system, user = render_agent(templates, agent, split, with_data=False)
            out.append(f"----- prior of {agent}\n[system]\n{system}\n[user]\n{user}\n")
    return "\n".join(out)


def resolve_code_path(value: str | Path, code_root: Path) -> Path:
    """Absolute paths stay; relative paths are taken relative to code/."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else code_root / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Build data/wearableqa/tasks.jsonl from WearableQA_raw.json.")
    parser.add_argument("--raw", default="data/wearableqa/WearableQA_raw.json", help="raw file (relative to code/)")
    parser.add_argument("--out", default="data/wearableqa/tasks.jsonl", help="tasks file (relative to code/)")
    parser.add_argument("--prompt_dir", default="prompts/wearableqa_phenotype", help="templates for the preview")
    parser.add_argument("--window_days", type=int, default=28)
    parser.add_argument("--preview", type=int, default=10, help="write rendered prompts of the first N tasks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    """Build and save the tasks; write the prompt preview; print class counts."""
    args = parse_args(argv)
    code_root = Path(__file__).resolve().parent.parent
    tasks = build_tasks(ws.load_raw(resolve_code_path(args.raw, code_root)), window_days=args.window_days)
    out = write_tasks(resolve_code_path(args.out, code_root), tasks)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["gold_class"]] = counts.get(task["gold_class"], 0) + 1
    print(f"wrote {len(tasks)} tasks from {len({t['user_id'] for t in tasks})} users to {out}; classes {counts}")
    if args.preview > 0:
        preview = out.parent / "prompt_preview.txt"
        templates = load_wqa_templates(resolve_code_path(args.prompt_dir, code_root))
        preview.write_text(preview_text(tasks, templates, args.preview),
                           encoding="utf-8")
        print(f"wrote the rendered prompts of the first {args.preview} tasks to {preview}")
    return tasks


if __name__ == "__main__":
    main()
