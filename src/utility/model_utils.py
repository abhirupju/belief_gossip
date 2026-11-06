"""Model download, one-instance-per-model registry, querying, option scoring and answer parsing.

torch / transformers / huggingface_hub are imported inside the functions that need them, so the
unit tests (which use a fake backend) run without them.
"""

import functools
import gc
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class GenerationResult:
    """What one model call returns."""

    text: str
    option_logprobs: dict[str, float] | None
    input_tokens: int
    output_tokens: int


@dataclass
class ModelHandle:
    """A loaded model shared by every agent that uses it."""

    name: str
    backend: Any
    device: str
    dtype: str
    model_class: str


class HFBackend:
    """Hugging Face backend: generate() and score_options() on one loaded model."""

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        """Hold the model, its processor/tokenizer and the device."""
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.device = device

    def build_inputs(self, messages: list[dict[str, Any]]) -> Any:
        """Apply the chat template (thinking disabled) and move tensors to the device."""
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        )
        return inputs.to(self.device)

    def option_token_ids(self, letters: list[str]) -> dict[str, list[int]]:
        """Token ids for each option letter (with and without a leading space)."""
        return get_option_token_ids(self.tokenizer, letters)

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, do_sample: bool,
                 option_token_ids: dict[str, list[int]] | None = None, seed: int = 0) -> GenerationResult:
        """Generate a reply; with option_token_ids, also return first-token option log-probabilities."""
        import torch

        inputs = self.build_inputs(messages)
        input_len = int(inputs["input_ids"].shape[-1])
        if do_sample:
            torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                return_dict_in_generate=True, output_logits=option_token_ids is not None,
            )
        new_tokens = out.sequences[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        option_logprobs = None
        if option_token_ids is not None:
            first_step = torch.log_softmax(out.logits[0][0].float(), dim=-1)
            option_logprobs = extract_option_logprobs(first_step, option_token_ids)
        return GenerationResult(text, option_logprobs, input_len, int(new_tokens.shape[0]))

    def score_options(self, messages: list[dict[str, Any]], option_token_ids: dict[str, list[int]]) -> dict[str, float]:
        """One forward pass over the prompt (no generation): option log-probabilities at the answer position."""
        import torch

        inputs = self.build_inputs(messages)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        return extract_option_logprobs(torch.log_softmax(logits.float(), dim=-1), option_token_ids)

    def release(self) -> None:
        """Drop the model and free device memory."""
        self.model = None
        release_device_memory()


def logsumexp(values: list[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    assert values, "logsumexp: empty list"
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(v - top) for v in values))


def get_option_token_ids(tokenizer: Any, letters: list[str]) -> dict[str, list[int]]:
    """Single-token ids for each letter, as 'A' and ' A'; asserts every letter has one and none overlap."""
    token_ids = {}
    for letter in letters:
        ids = []
        for variant in (letter, " " + letter):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(int(encoded[0]))
        assert ids, f"get_option_token_ids: letter {letter!r} is not a single token for this tokenizer"
        token_ids[letter] = ids
    flat = [i for ids in token_ids.values() for i in ids]
    assert len(flat) == len(set(flat)), f"get_option_token_ids: letters share token ids: {token_ids}"
    return token_ids


def extract_option_logprobs(vocab_logprobs: Any, option_token_ids: dict[str, list[int]]) -> dict[str, float]:
    """Combine each letter's token variants and normalize over the options (log-probabilities sum to 1)."""
    assert option_token_ids, "extract_option_logprobs: no option tokens given"
    size = len(vocab_logprobs)
    per_letter = {}
    for letter, ids in option_token_ids.items():
        assert ids and all(0 <= i < size for i in ids), (
            f"extract_option_logprobs: token ids {ids} for {letter!r} are outside the vocabulary (size {size})"
        )
        per_letter[letter] = logsumexp([float(vocab_logprobs[i]) for i in ids])
    total = logsumexp(list(per_letter.values()))
    assert math.isfinite(total), f"extract_option_logprobs: option letters have zero probability: {per_letter}"
    return {letter: value - total for letter, value in per_letter.items()}


def model_dir_for(model_name: str, model_root: Path) -> Path:
    """model/Qwen__Qwen3.5-0.8B (slash replaced)."""
    assert model_name and "/" in model_name, (
        f"model_dir_for: {model_name!r} is not a Hugging Face name like 'Qwen/Qwen3.5-0.8B'"
    )
    return Path(model_root) / model_name.replace("/", "__")


def download_model(model_name: str, model_root: Path) -> Path:
    """snapshot_download into the model folder if config.json is missing; return the folder."""
    target = model_dir_for(model_name, model_root)
    if not (target / "config.json").is_file():
        from huggingface_hub import snapshot_download

        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_name, local_dir=str(target))
    assert (target / "config.json").is_file(), f"download_model: {target} has no config.json after download"
    return target


def detect_model_class(model_path: Path) -> str:
    """'multimodal' or 'causal', from the architectures field of config.json."""
    config_path = Path(model_path) / "config.json"
    assert config_path.is_file(), f"detect_model_class: {config_path} not found; download the model first"
    architectures = json.loads(config_path.read_text(encoding="utf-8")).get("architectures") or []
    assert architectures, f"detect_model_class: {config_path} lists no architectures; set model_class explicitly"
    if any(a.endswith("ForConditionalGeneration") or "ImageTextToText" in a for a in architectures):
        return "multimodal"
    assert any(a.endswith("ForCausalLM") for a in architectures), (
        f"detect_model_class: cannot tell the class from {architectures}; set model_class to multimodal or causal"
    )
    return "causal"


def resolve_device(requested: str) -> str:
    """'auto' -> cuda, else mps, else cpu; asserts that an explicitly requested device exists."""
    assert requested in ("auto", "cuda", "mps", "cpu"), (
        f"resolve_device: device {requested!r} must be auto, cuda, mps or cpu"
    )
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    available = torch.cuda.is_available() if requested == "cuda" else torch.backends.mps.is_available()
    assert available, f"resolve_device: device {requested!r} was requested but is not available on this machine"
    return requested


def get_torch_dtype(dtype: str) -> Any:
    """torch dtype object for a dtype name."""
    import torch

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    assert dtype in mapping, f"get_torch_dtype: dtype {dtype!r} must be one of {sorted(mapping)}"
    return mapping[dtype]


def hf_loader(model_path: Path, device: str, dtype: str, model_class: str) -> HFBackend:
    """Real loader: AutoProcessor + multimodal model class, or AutoTokenizer + AutoModelForCausalLM."""
    import transformers

    assert model_class in ("multimodal", "causal"), f"hf_loader: model_class {model_class!r} must be resolved first"
    if model_class == "multimodal":
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None) or getattr(
            transformers, "AutoModelForImageTextToText", None)
        assert model_cls is not None, (
            f"hf_loader: transformers {transformers.__version__} has no multimodal auto class; upgrade transformers"
        )
        processor = transformers.AutoProcessor.from_pretrained(str(model_path))
    else:
        model_cls = transformers.AutoModelForCausalLM
        processor = transformers.AutoTokenizer.from_pretrained(str(model_path))
    model = model_cls.from_pretrained(str(model_path), dtype=get_torch_dtype(dtype))
    model.to(device).eval()
    return HFBackend(model, processor, device)


@functools.lru_cache(maxsize=1)
def get_model_registry() -> dict[str, ModelHandle]:
    """The process-wide dict of loaded models (created once, on first call)."""
    return {}


def release_device_memory() -> None:
    """Garbage-collect and empty the device cache if torch is already loaded."""
    gc.collect()
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def clear_model_registry() -> None:
    """Release and forget every loaded model."""
    registry = get_model_registry()
    for handle in registry.values():
        handle.backend.release()
    registry.clear()
    release_device_memory()


def load_model(model_name: str, model_root: Path, device: str, dtype: str, model_class: str,
               loader_fn: Callable | None = None, download: bool = True) -> ModelHandle:
    """Return the shared handle for model_name, loading it only on the first call."""
    registry = get_model_registry()
    if model_name in registry:
        handle = registry[model_name]
        assert (handle.device, handle.dtype) == (device, dtype), (
            f"load_model: {model_name} is already loaded on {handle.device}/{handle.dtype}; refusing to load a "
            f"second copy on {device}/{dtype}"
        )
        return handle
    model_path = download_model(model_name, model_root) if download else model_dir_for(model_name, model_root)
    resolved_class = detect_model_class(model_path) if model_class == "auto" else model_class
    backend = (loader_fn or hf_loader)(model_path, device, dtype, resolved_class)
    handle = ModelHandle(model_name, backend, device, dtype, resolved_class)
    registry[model_name] = handle
    return handle


def default_benchmark_messages(model_class: str) -> list[dict[str, Any]]:
    """A fixed short prompt used to time dtypes."""
    return build_chat_messages(
        "Answer with a single letter only.",
        "Which option is a fruit?\nA. Apple\nB. Brick\nReply with only the letter (A or B).",
        model_class,
    )


def benchmark_dtypes(model_path: Path, device: str, model_class: str, loader_fn: Callable | None = None,
                     candidates: tuple[str, ...] = ("bfloat16", "float16"), repeats: int = 3) -> dict[str, float]:
    """Seconds per query for each dtype; loads one copy at a time, outside the registry."""
    loader = loader_fn or hf_loader
    messages = default_benchmark_messages(model_class)
    timings = {}
    for dtype in candidates:
        backend = loader(model_path, device, dtype, model_class)
        token_ids = backend.option_token_ids(["A", "B"])
        scores = backend.score_options(messages, token_ids)
        assert all(math.isfinite(v) for v in scores.values()), (
            f"benchmark_dtypes: {dtype} gives non-finite scores {scores} on {device}; it cannot be used"
        )
        backend.generate(messages, 2, False)
        start = time.perf_counter()
        for _ in range(repeats):
            backend.generate(messages, 2, False)
        timings[dtype] = (time.perf_counter() - start) / repeats
        backend.release()
        del backend
        release_device_memory()
    return timings


def choose_dtype(model_name: str, model_root: Path, device: str, model_class: str,
                 loader_fn: Callable | None = None, download: bool = True) -> str:
    """For dtype 'auto': float32 on CPU; otherwise the faster of bf16/fp16, measured once and saved."""
    if device == "cpu":
        return "float32"
    model_path = download_model(model_name, model_root) if download else model_dir_for(model_name, model_root)
    choice_path = model_path / f"dtype_choice_{device}.json"
    if choice_path.is_file():
        return json.loads(choice_path.read_text(encoding="utf-8"))["dtype"]
    resolved_class = detect_model_class(model_path) if model_class == "auto" else model_class
    timings = benchmark_dtypes(model_path, device, resolved_class, loader_fn)
    best = min(timings, key=timings.get)
    choice_path.write_text(json.dumps({"dtype": best, "seconds_per_query": timings}, indent=2), encoding="utf-8")
    return best


def build_chat_messages(system_text: str, user_text: str, model_class: str) -> list[dict[str, Any]]:
    """Messages in the shape the chat template expects (list-of-parts content for multimodal models)."""
    assert model_class in ("multimodal", "causal"), f"build_chat_messages: unresolved model_class {model_class!r}"
    if model_class == "multimodal":
        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


def query_model(handle: ModelHandle, messages: list[dict[str, Any]], max_new_tokens: int, do_sample: bool,
                option_token_ids: dict[str, list[int]] | None = None, seed: int = 0) -> GenerationResult:
    """Generate with the handle's backend."""
    assert max_new_tokens >= 1, f"query_model: max_new_tokens must be >= 1, got {max_new_tokens}"
    return handle.backend.generate(messages, max_new_tokens, do_sample, option_token_ids, seed)


def score_options(handle: ModelHandle, messages: list[dict[str, Any]],
                  option_token_ids: dict[str, list[int]]) -> dict[str, float]:
    """Prompt-only option log-probabilities (no generation)."""
    return handle.backend.score_options(messages, option_token_ids)


def memory_report(device: str) -> dict[str, float]:
    """Allocated device memory in MB (0 on CPU or when torch is not loaded)."""
    if device == "cpu" or "torch" not in sys.modules:
        return {"device": device, "allocated_mb": 0.0}
    torch = sys.modules["torch"]
    allocated = torch.cuda.memory_allocated() if device == "cuda" else torch.mps.current_allocated_memory()
    return {"device": device, "allocated_mb": allocated / 2**20}


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and surrounding whitespace."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_option_letter(response: str, valid_letters: list[str], options: list[str] | None = None) -> str | None:
    """Answer letter or None.

    Rules, in order: (1) strip whitespace and think blocks; (2) the whole reply is one valid letter,
    optionally bracketed or followed by '.', ')' or ':' (case-insensitive); (3) 'Answer: C' / 'the answer
    is C' (case-insensitive, last one wins); (4) exactly one distinct standalone upper-case valid letter;
    (5) exactly one option's text appears; (6) otherwise None.
    """
    assert isinstance(response, str), f"parse_option_letter: response is {type(response).__name__}, expected str"
    assert valid_letters, "parse_option_letter: no valid letters"
    valid = [v.upper() for v in valid_letters]
    text = strip_think(response)
    if not text:
        return None
    whole = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?\s*[.:)]?", text)
    if whole:
        return whole.group(1).upper() if whole.group(1).upper() in valid else None
    phrases = re.findall(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([A-Za-z])\)?(?![A-Za-z])", text, flags=re.IGNORECASE)
    phrases = [p.upper() for p in phrases if p.upper() in valid]
    if phrases:
        return phrases[-1]
    standalone = {m for m in re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text) if m in valid}
    if len(standalone) == 1:
        return standalone.pop()
    if options:
        found = [i for i, option in enumerate(options) if option.lower() in text.lower()]
        if len(found) == 1 and found[0] < len(valid):
            return valid[found[0]]
    return None


def parse_reasoning_response(response: str, valid_letters: list[str], options: list[str] | None,
                             answer_marker: str) -> tuple[str, str | None]:
    """(reasoning, letter): split at the last answer marker; no marker -> (whole reply, None)."""
    assert answer_marker, "parse_reasoning_response: empty answer_marker"
    text = strip_think(response)
    cut = text.lower().rfind(answer_marker.lower())
    if cut < 0:
        return text, None
    reasoning = re.sub(r"^\s*reasoning\s*:\s*", "", text[:cut], flags=re.IGNORECASE).strip()
    return reasoning, parse_option_letter(text[cut + len(answer_marker):], valid_letters, options)
