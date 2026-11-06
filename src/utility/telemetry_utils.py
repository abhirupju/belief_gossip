"""OpenTelemetry tracing: one trace per run, written to an OTLP JSON lines file for later viewing.

No global tracer provider is set: setup_telemetry returns a Telemetry object that is passed explicitly
to the experiment and the agents, so tests can inject an in-memory exporter.
"""

import contextlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Telemetry:
    """Tracer plus the settings every caller needs."""

    provider: Any
    tracer: Any
    enabled: bool
    capture_content: bool


def make_file_exporter(path: Path) -> Any:
    """The only place that touches the beta OTLP JSON file exporter."""
    from opentelemetry.exporter.otlp.json.file.trace_exporter import FileSpanExporter

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return FileSpanExporter(path=str(path))


def make_http_exporter(endpoint: str) -> Any:
    """Optional live export to a collector (needs opentelemetry-exporter-otlp-proto-http)."""
    assert importlib.util.find_spec("opentelemetry.exporter.otlp.proto.http") is not None, (
        "make_http_exporter: --otlp_endpoint needs `pip install opentelemetry-exporter-otlp-proto-http`"
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")


def clean_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep OpenTelemetry-compatible values: drop None, JSON-encode dicts and mixed lists."""
    cleaned = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, bool, int, float)) for v in value) and (
                len({type(v) for v in value}) <= 1):
            cleaned[key] = list(value)
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False)
    return cleaned


def setup_telemetry(trace_path: Path | None, run_info: dict[str, Any], enabled: bool = True,
                    capture_content: bool = True, otlp_endpoint: str | None = None,
                    exporter: Any = None) -> Telemetry:
    """Tracer provider with the run's resource attributes and a file (or injected) exporter."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    if not enabled:
        return Telemetry(None, trace.NoOpTracer(), False, False)
    assert exporter is not None or trace_path is not None, "setup_telemetry: give a trace_path or an exporter"
    resource = Resource.create(clean_attributes({"service.name": "belief_gossip", **run_info}))
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter if exporter is not None else make_file_exporter(trace_path)))
    if otlp_endpoint:
        provider.add_span_processor(SimpleSpanProcessor(make_http_exporter(otlp_endpoint)))
    return Telemetry(provider, provider.get_tracer("belief_gossip"), True, capture_content)


@contextlib.contextmanager
def start_span(telemetry: Telemetry, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Context manager for a child span of the current one."""
    with telemetry.tracer.start_as_current_span(name, attributes=clean_attributes(attributes or {})) as span:
        yield span


def add_event(telemetry: Telemetry, span: Any, name: str, attributes: dict[str, Any],
              content_keys: tuple[str, ...] = ()) -> None:
    """Timestamped event; content_keys (prompt/reply text) are dropped when content capture is off."""
    kept = {k: v for k, v in attributes.items() if telemetry.capture_content or k not in content_keys}
    span.add_event(name, attributes=clean_attributes(kept))


def set_attributes(span: Any, attributes: dict[str, Any]) -> None:
    """Add attributes to an open span."""
    span.set_attributes(clean_attributes(attributes))


def get_span_ids(span: Any) -> tuple[str, str]:
    """(trace_id, span_id) as hex strings; zeros when tracing is off."""
    context = span.get_span_context()
    return format(context.trace_id, "032x"), format(context.span_id, "016x")


def agent_span_attributes(agent_id: str, role: str, round_idx: int, task_index: int,
                          fact_indices: list[int], rotation: int) -> dict[str, Any]:
    """Standard attributes on every agent span, so all callers label agents the same way."""
    return {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.id": agent_id,
        "gen_ai.agent.name": agent_id,
        "bg.agent.role": role,
        "bg.round": round_idx,
        "bg.task.index": task_index,
        "bg.fact_indices": list(fact_indices),
        "bg.rotation": rotation,
    }


def shutdown_telemetry(telemetry: Telemetry) -> None:
    """Flush and close the trace file."""
    if telemetry.provider is not None:
        telemetry.provider.force_flush()
        telemetry.provider.shutdown()
