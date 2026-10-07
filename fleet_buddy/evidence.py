"""Turns tool results into the chart, the evidence box, and cleaned-up maths in the reply."""

from __future__ import annotations

import json
import re
from typing import Any


def metric_label(metric: str) -> str:
    return {
        "sum_nitrogen": "Nitrogen applied (L)",
        "sum_distance": "Distance driven (m)",
        "efficiency_l_per_km": "Efficiency (L/km)",
        "avg_battery": "Average battery (%)",
        "max_battery": "Maximum battery (%)",
        "min_battery": "Minimum battery (%)",
        "row_count": "Intervals / rows",
        "robot_count": "Distinct robots observed",
    }.get(metric, metric.replace("_", " ").title())


def build_visualization(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Turn the structured tool result into a small, honest chart for the UI."""
    aggregate_groups: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
    query_candidates: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    weather_candidates: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for step_index, step in enumerate(trace.get("steps", [])):
        if step.get("type") != "tool" or step.get("error"):
            continue
        result = step.get("result") or {}
        rows = result.get("rows") or []
        if not rows:
            continue
        if step.get("name") == "aggregate_runs":
            metrics = result.get("metrics") or []
            metric = next((name for name in metrics if any(isinstance(row.get(name), (int, float)) for row in rows)), None)
            if metric is None:
                continue
            group_by = tuple(result.get("group_by") or [])
            filters = result.get("filters") or {}
            comparable_filters = {key: value for key, value in filters.items() if key != "robot_id"}
            group_key = (metric, group_by, json.dumps(comparable_filters, sort_keys=True))
            group = aggregate_groups.setdefault(group_key, {"step": step, "step_index": step_index, "metric": metric, "group_by": group_by, "rows": {}})
            for row in rows:
                row_key = tuple(str(row.get(name)) for name in group_by) or (metric,)
                group["rows"][row_key] = row
        if step.get("name") == "query_runs":
            query_candidates.append((step_index, step, rows))
        if step.get("name") == "get_weather":
            weather_candidates.append((step_index, step, rows))
    if aggregate_groups:
        def chart_priority(item: dict[str, Any]) -> tuple[int, float, int, int]:
            numeric_total = sum(
                abs(float(row.get(item["metric"]) or 0))
                for row in item["rows"].values()
                if isinstance(row.get(item["metric"]), (int, float))
            )
            fallback_priority = 2 if item["step"].get("fallback") else 0
            return (fallback_priority + (1 if numeric_total > 0 else 0), numeric_total, len(item["rows"]), item["step_index"])

        group = max(aggregate_groups.values(), key=chart_priority)
        step = group["step"]
        metric = group["metric"]
        group_by = group["group_by"]
        rows = list(group["rows"].values())
        group_name = group_by[0] if group_by else "metric"
        labels = [str(row.get(group_name, metric_label(metric))) for row in rows]
        values = [float(row.get(metric, 0) or 0) for row in rows]
        return {
            "type": "bar",
            "title": metric_label(metric) + (f" by {group_name.replace('_', ' ')}" if group_by else ""),
            "labels": labels,
            "datasets": [{"label": metric_label(metric), "data": values}],
            "source_step_id": step.get("id"),
        }
    # Raw event queries become a count chart, usually faults by date.
    if query_candidates:
        _, step, rows = query_candidates[-1]
        if all("ts" in row for row in rows):
            counts: dict[str, int] = {}
            for row in rows:
                date = str(row["ts"])[:10]
                counts[date] = counts.get(date, 0) + 1
            return {
                "type": "bar",
                "title": "Matching intervals by date",
                "labels": list(counts),
                "datasets": [{"label": "Intervals", "data": list(counts.values())}],
                "source_step_id": step.get("id"),
            }
    if weather_candidates:
        _, step, rows = weather_candidates[-1]
        soil_metrics = [
            ("soil_moisture_0_to_7cm_m3_m3", "Soil moisture 0-7 cm (m³/m³)"),
            ("soil_moisture_7_to_28cm_m3_m3", "Soil moisture 7-28 cm (m³/m³)"),
            ("soil_moisture_28_to_100cm_m3_m3", "Soil moisture 28-100 cm (m³/m³)"),
            ("soil_moisture_100_to_255cm_m3_m3", "Soil moisture 100-255 cm (m³/m³)"),
        ]
        soil_metric = next(((name, label) for name, label in soil_metrics if any(row.get(name) is not None for row in rows)), None)
        if soil_metric:
            metric, title = soil_metric
            return {
                "type": "bar",
                "title": "External Open-Meteo: " + title,
                "labels": [str(row.get("date")) for row in rows],
                "datasets": [{"label": metric, "data": [float(row.get(metric) or 0) for row in rows]}],
                "source_step_id": step.get("id"),
            }
        metric = "temperature_mean_c" if any(row.get("temperature_mean_c") is not None for row in rows) else "precipitation_mm"
        labels = [str(row.get("date")) for row in rows]
        values = [float(row.get(metric) or 0) for row in rows]
        return {
            "type": "bar",
            "title": "External weather: " + ("Mean temperature (°C)" if metric == "temperature_mean_c" else "Precipitation (mm)"),
            "labels": labels,
            "datasets": [{"label": metric, "data": values}],
            "source_step_id": step.get("id"),
        }
    return None


def build_evidence(trace: dict[str, Any]) -> dict[str, Any] | None:
    # The "evidence" box under each answer: which tools ran, date ranges, filters, and row counts.
    tool_steps = [step for step in trace.get("steps", []) if step.get("type") == "tool" and not step.get("error")]
    if not tool_steps:
        grounding = trace.get("grounding")
        if not grounding:
            return None
        return {
            "tools": [],
            "step_ids": [],
            "effective_time_ranges": [],
            "filters": [],
            "sources": [],
            "result_rows": 0,
            "truncated": False,
            "fallback_used": bool(trace.get("fallback_used")),
            "grounding": grounding,
        }
    ranges = []
    filters = []
    sources = []
    result_rows = 0
    truncated = False
    for step in tool_steps:
        result = step.get("result") or {}
        if result.get("effective_time_range") and result["effective_time_range"] not in ranges:
            ranges.append(result["effective_time_range"])
        if result.get("filters"):
            filters.append(result["filters"])
        if result.get("source") and result["source"] not in sources:
            sources.append(result["source"])
        result_rows += len(result.get("rows") or [])
        truncated = truncated or bool(result.get("truncated"))
    return {
        "tools": list(dict.fromkeys(step["name"] for step in tool_steps)),
        "step_ids": [step["id"] for step in tool_steps],
        "effective_time_ranges": ranges,
        "filters": filters,
        "sources": sources,
        "result_rows": result_rows,
        "truncated": truncated,
        "fallback_used": bool(trace.get("fallback_used")),
        "grounding": trace.get("grounding"),
    }


def normalize_latex_response(content: str) -> str:
    """Normalize common model pseudo-LaTeX before returning the API reply."""
    block_pattern = re.compile(r"(^|\n)\s*\[\s*([\s\S]*?)\s*\](?=\s*(?:\n|$))")

    def replace_block(match: re.Match[str]) -> str:
        expression = match.group(2).strip()
        if not re.search(r"\\[A-Za-z]+|[=^_{}]", expression):
            return match.group(0)
        return f"{match.group(1)}\n$$\n{expression}\n$$"

    normalized = block_pattern.sub(replace_block, str(content))
    inline_pattern = re.compile(
        r"\(\s*((?:\\(?:frac|text|approx|sqrt|sum|int|mathrm|mathbf|cdot|times|div)\b[\s\S]*?))\s*\)"
    )
    return inline_pattern.sub(lambda match: f"\\({match.group(1).strip()}\\)", normalized)
