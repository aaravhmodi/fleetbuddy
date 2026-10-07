"""Dashboard numbers computed from the stored traces."""

from __future__ import annotations

import math
from typing import Any

from . import state
from .config import INPUT_PRICE_PER_TOKEN, OUTPUT_PRICE_PER_TOKEN


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": trace["id"],
        "dataset_id": trace["dataset_id"],
        "created_at": trace["created_at"],
        "question": trace["question"],
        "outcome": trace["outcome"],
        "duration_ms": trace["duration_ms"],
        "input_tokens": trace["input_tokens"],
        "output_tokens": trace["output_tokens"],
        "total_tokens": trace["input_tokens"] + trace["output_tokens"],
        "cost": trace["cost"],
        "model_calls": sum(step["type"] == "model" for step in trace["steps"]),
        "tool_calls": sum(step["type"] == "tool" for step in trace["steps"]),
        "fallback_used": bool(trace.get("fallback_used")),
        "grounding_status": (trace.get("grounding") or {}).get("status"),
        "previous_trace_id": trace.get("previous_trace_id"),
        "reused_evidence": bool(trace.get("reused_evidence")),
        "untraced_duration_ms": trace.get("untraced_duration_ms", 0.0),
        "timing_coverage_pct": trace.get("timing_coverage_pct", 0.0),
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile_value) - 1))
    return round(ordered[index], 2)


def analytics() -> dict[str, Any]:
    # Dashboard totals across all traces: tokens, cost, latency, failure rates, per-tool counts.
    with state.STATE_LOCK:
        traces = list(state.TRACES)
    model_steps = [step for trace in traces for step in trace["steps"] if step["type"] == "model"]
    tool_steps = [step for trace in traces for step in trace["steps"] if step["type"] == "tool"]
    by_tool: dict[str, Any] = {}
    for step in tool_steps:
        item = by_tool.setdefault(step["name"], {"calls": 0, "duration_ms": 0.0, "errors": 0})
        item["calls"] += 1
        item["duration_ms"] += step["duration_ms"]
        item["errors"] += int(bool(step.get("error")))
    failed = sum(trace["outcome"] == "failed" for trace in traces)
    stopped = sum(trace["outcome"] == "stopped_at_cap" for trace in traces)
    ungrounded = sum(trace["outcome"] == "ungrounded" for trace in traces)
    fallback_count = sum(bool(trace.get("fallback_used")) for trace in traces)
    latencies = [trace["duration_ms"] for trace in traces]
    return {
        "turns": len(traces),
        "model_calls": len(model_steps),
        "tool_calls": len(tool_steps),
        "tool_breakdown": by_tool,
        "input_tokens": sum(trace["input_tokens"] for trace in traces),
        "output_tokens": sum(trace["output_tokens"] for trace in traces),
        "total_tokens": sum(trace["input_tokens"] + trace["output_tokens"] for trace in traces),
        "total_cost": round(sum(trace["cost"] for trace in traces), 8),
        "average_turn_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_turn_latency_ms": percentile(latencies, 0.95),
        "failed_share": round(failed / len(traces), 4) if traces else 0.0,
        "stopped_at_cap_share": round(stopped / len(traces), 4) if traces else 0.0,
        "ungrounded_share": round(ungrounded / len(traces), 4) if traces else 0.0,
        "fallback_share": round(fallback_count / len(traces), 4) if traces else 0.0,
        "price_per_million_tokens": {"input": INPUT_PRICE_PER_TOKEN * 1_000_000, "output": OUTPUT_PRICE_PER_TOKEN * 1_000_000},
    }
