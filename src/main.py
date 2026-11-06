"""Entry point: parse arguments, load the config, apply overrides, run the experiment, print accuracy.

Example (from the repo root): python code/src/main.py --num_samples 5 --device mps
"""

import argparse
import json
from pathlib import Path
from typing import Any

from agent import load_config, validate_config, validate_facts_per_agent
from agent_exp import get_dataset_adapter, get_needed_model_keys, run_experiment
from utility.model_utils import download_model


def get_code_root() -> Path:
    """The code/ folder (parent of src/); every relative path is resolved against it."""
    return Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path, code_root: Path) -> Path:
    """Absolute paths stay; relative paths are taken relative to code/."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (code_root / path)


def parse_facts_per_agent(value: str) -> int | str:
    """'all' or a positive integer, from the command line."""
    parsed = value if value == "all" else int(value) if value.isdigit() else value
    validate_facts_per_agent(parsed)
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments (defaults come from the config unless given)."""
    parser = argparse.ArgumentParser(description="Run agents on a dataset and save predictions, summary and traces.")
    parser.add_argument("--config", default="config/hiddenbench_qwen35_08b.yaml", help="config file (relative to code/)")
    parser.add_argument("--dataset", default=None, help="dataset name (only hiddenbench for now)")
    parser.add_argument("--model", default=None, help="Hugging Face model name; replaces the default model")
    parser.add_argument("--num_samples", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--facts_per_agent", type=parse_facts_per_agent, default=None, help="k, or 'all'")
    parser.add_argument("--option_rotation", choices=["none", "cyclic"], default=None)
    parser.add_argument("--prompt_dir", default=None, help="folder of prompt templates (relative to code/)")
    parser.add_argument("--prompt_template", default=None, help="template name without .yaml")
    parser.add_argument("--reasoning", action="store_true", help="ask for reasoning before the answer")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--data_dir", default="data", help="dataset root (relative to code/, or an absolute /rds path)")
    parser.add_argument("--model_dir", default="model", help="model root (relative to code/, or an absolute /rds path)")
    parser.add_argument("--results_dir", default="results", help="parent of the run folders")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None, help="generation length for the chosen mode")
    parser.add_argument("--no_trace", action="store_true", help="do not write the OpenTelemetry trace file")
    parser.add_argument("--no_trace_content", action="store_true", help="leave prompts and replies out of the trace")
    parser.add_argument("--otlp_endpoint", default=None, help="also send spans live, e.g. http://localhost:4318")
    parser.add_argument("--download_only", action="store_true", help="fetch dataset and models, then exit")
    return parser.parse_args(argv)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply command-line overrides to the config and re-validate it."""
    if args.dataset is not None:
        config["experiment"]["dataset"] = args.dataset
    default_key = config["agents"]["default_model"]
    default_model = config["models"][default_key]
    if args.model is not None:
        default_model["hf_name"] = args.model
        default_model["model_class"] = "auto"
    if args.dtype is not None:
        default_model["dtype"] = args.dtype
    if args.device is not None:
        for model in config["models"].values():
            model["device"] = args.device
    if args.facts_per_agent is not None:
        config["agents"]["facts_per_agent"] = args.facts_per_agent
    if args.option_rotation is not None:
        config["prompts"]["option_rotation"] = args.option_rotation
    if args.prompt_dir is not None:
        config["prompts"]["dir"] = args.prompt_dir
    if args.prompt_template is not None:
        config["prompts"]["template"] = args.prompt_template
    elif args.reasoning:
        config["prompts"]["template"] = "reasoning_answer"
    if args.max_new_tokens is not None:
        key = "max_new_tokens_reasoning" if args.reasoning else "max_new_tokens"
        config["generation"][key] = args.max_new_tokens
    if args.no_trace:
        config["telemetry"]["enabled"] = False
    if args.no_trace_content:
        config["telemetry"]["capture_content"] = False
    if args.otlp_endpoint is not None:
        config["telemetry"]["otlp_endpoint"] = args.otlp_endpoint
    validate_config(config)
    return config


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    """Load config, apply overrides, download if needed, run, print accuracy."""
    args = parse_args(argv)
    code_root = get_code_root()
    config = apply_overrides(load_config(resolve_path(args.config, code_root)), args)
    paths = {
        "data_dir": resolve_path(args.data_dir, code_root),
        "model_dir": resolve_path(args.model_dir, code_root),
        "results_dir": resolve_path(args.results_dir, code_root),
        "prompt_dir": resolve_path(config["prompts"]["dir"], code_root),
    }
    if args.download_only:
        get_dataset_adapter(config["experiment"]["dataset"])["prepare"](paths["data_dir"])
        for key in get_needed_model_keys(config):
            download_model(config["models"][key]["hf_name"], paths["model_dir"])
        print(f"Downloaded dataset to {paths['data_dir']} and models to {paths['model_dir']}")
        return None
    summary = run_experiment(config, paths, args.num_samples, args.reasoning)
    accuracy, majority = summary["accuracy"], summary["majority"]
    print(json.dumps({"per_agent_accuracy": round(accuracy["per_agent_accuracy"], 4),
                      "majority_accuracy": round(majority["majority_accuracy"], 4),
                      "answers": accuracy["num_answers"], "parse_failures": accuracy["parse_failures"],
                      "results": summary["files"]["predictions"]}, indent=2))
    return summary


if __name__ == "__main__":
    main()
