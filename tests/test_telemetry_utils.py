"""Span tree, agent attributes, timestamps, events, content capture and the OTLP JSON file."""

import json

from agent_exp import run_experiment
from utility import telemetry_utils as tu


def run_with_exporter(config, paths, loader, exporter):
    summary = run_experiment(config, paths, loader_fn=loader, span_exporter=exporter, download=False)
    with open(summary["files"]["predictions"], encoding="utf-8") as handle:
        return summary, json.load(handle), exporter.get_finished_spans()


def test_span_tree_and_agent_attributes(config_copy, make_paths, make_loader, span_exporter):
    loader, _ = make_loader(lambda text: "A")
    _, records, spans = run_with_exporter(config_copy(agents={"facts_per_agent": 1}), make_paths(), loader,
                                          span_exporter)
    by_id = {s.context.span_id: s for s in spans}

    def parent_name(span):
        return by_id[span.parent.span_id].name if span.parent else None

    names = [s.name for s in spans]
    assert sum(n.startswith("run ") for n in names) == 1
    assert sum(n.startswith("task ") for n in names) == 3
    agent_spans = [s for s in spans if s.name.startswith("invoke_agent")]
    assert len(agent_spans) == 9
    for span in agent_spans:
        assert parent_name(span) == "round 0"
        assert span.attributes["gen_ai.agent.id"].startswith("agent-")
        assert "bg.task.index" in span.attributes and span.attributes["bg.round"] == 0
        assert span.start_time <= span.end_time
        assert {e.name for e in span.events} >= {"prompt_rendered", "answer_parsed"}
    for span in spans:
        if span.name.startswith("chat "):
            assert parent_name(span).startswith("invoke_agent")
            assert span.attributes["gen_ai.request.model"] == "fake/causal-model"
        if span.name == "round 0":
            assert parent_name(span).startswith("task ")
        if span.name.startswith("task "):
            assert parent_name(span).startswith("run ")
    span_ids = {format(s.context.span_id, "016x") for s in agent_spans}
    assert {r["span_id"] for r in records} == span_ids
    assert {r["trace_id"] for r in records} == {format(agent_spans[0].context.trace_id, "032x")}


def test_content_capture_off_drops_text(config_copy, make_paths, make_loader, span_exporter):
    loader, _ = make_loader(lambda text: "A")
    _, _, spans = run_with_exporter(config_copy(telemetry={"capture_content": False}), make_paths(), loader,
                                    span_exporter)
    for span in spans:
        for event in span.events:
            assert "user" not in event.attributes and "text" not in event.attributes
    rendered = [e for s in spans for e in s.events if e.name == "prompt_rendered"]
    assert rendered and all(e.attributes["template"] == "answer_only" for e in rendered)


def test_content_capture_on_keeps_prompt(config_copy, make_paths, make_loader, span_exporter):
    loader, _ = make_loader(lambda text: "A")
    _, _, spans = run_with_exporter(config_copy(), make_paths(), loader, span_exporter)
    rendered = [e for s in spans for e in s.events if e.name == "prompt_rendered"]
    assert all("Information only you know" in e.attributes["user"] for e in rendered)


def test_file_exporter_writes_otlp_json_lines(tmp_path):
    path = tmp_path / "trace.otlp.jsonl"
    telemetry = tu.setup_telemetry(path, {"bg.run.id": "r1"})
    with tu.start_span(telemetry, "run test", {"bg.k": "all", "skip": None, "facts": [1, 2], "obj": {"a": 1}}) as span:
        tu.add_event(telemetry, span, "hello", {"x": 1})
    tu.shutdown_telemetry(telemetry)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1 and "resourceSpans" in lines[0]
    text = json.dumps(lines[0])
    assert "run test" in text and "bg.run.id" in text and "hello" in text


def test_disabled_telemetry_is_a_no_op():
    telemetry = tu.setup_telemetry(None, {}, enabled=False)
    with tu.start_span(telemetry, "x", {}) as span:
        tu.add_event(telemetry, span, "e", {"a": 1})
    assert tu.get_span_ids(span) == ("0" * 32, "0" * 16)
    tu.shutdown_telemetry(telemetry)


def test_clean_attributes():
    cleaned = tu.clean_attributes({"a": None, "b": 1, "c": [1, 2], "d": {"k": 1}, "e": [1, "x"]})
    assert cleaned == {"b": 1, "c": [1, 2], "d": '{"k": 1}', "e": '[1, "x"]'}
