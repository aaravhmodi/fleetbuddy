"""The tools the model can call, their descriptions, and the router that runs them."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import MAX_TOOL_ROWS, VALID_STATES
from .data import filtered_frame, format_rows
from .state import Dataset
from .utils import json_safe
from .weather import run_get_weather


def run_query_runs(dataset: Dataset, args: dict[str, Any]) -> dict[str, Any]:
    # Tool 1: return matching raw rows (max 50), e.g. "when did MR-03 fault?"
    frame = filtered_frame(dataset, args)
    limit = int(args.get("limit", MAX_TOOL_ROWS))
    if limit < 1 or limit > MAX_TOOL_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_TOOL_ROWS}.")
    result = format_rows(frame.sort_values("ts"), limit)
    result["filters"] = {key: value for key, value in args.items() if value is not None}
    return result


def run_aggregate_runs(dataset: Dataset, args: dict[str, Any]) -> dict[str, Any]:
    # Tool 2: totals, averages, and counts, optionally grouped by robot, field, state, or date.
    # To add a metric, add it to allowed_metrics below AND to the "metrics" enum in TOOLS.
    frame = filtered_frame(dataset, args)
    group_by = args.get("group_by", [])
    if isinstance(group_by, str):
        group_by = [group_by]
    valid_groups = {"robot_id", "field", "state", "date"}
    if any(group not in valid_groups for group in group_by):
        raise ValueError("group_by values must be robot_id, field, state, or date.")
    if len(group_by) != len(set(group_by)):
        raise ValueError("group_by values must be unique.")
    if len(group_by) > 2:
        raise ValueError("group_by supports at most two dimensions at a time.")
    metrics = args.get("metrics", ["sum_nitrogen", "sum_distance"])
    allowed_metrics = {
        "sum_nitrogen": ("nitrogen_applied_l", "sum"),
        "sum_distance": ("distance_m", "sum"),
        "avg_battery": ("battery_pct", "mean"),
        "max_battery": ("battery_pct", "max"),
        "min_battery": ("battery_pct", "min"),
        "row_count": ("ts", "count"),
        "robot_count": ("robot_id", "nunique"),
        "efficiency_l_per_km": None,
    }
    if not metrics or any(metric not in allowed_metrics for metric in metrics):
        raise ValueError("metrics must use sum_nitrogen, sum_distance, efficiency_l_per_km, avg_battery, max_battery, min_battery, row_count, or robot_count.")
    if len(metrics) != len(set(metrics)):
        raise ValueError("metrics values must be unique.")

    def calculate_metric(source: pd.DataFrame, metric: str) -> Any:
        if metric == "efficiency_l_per_km":
            nitrogen = source["nitrogen_applied_l"].sum(min_count=1)
            distance = source["distance_m"].sum(min_count=1)
            return nitrogen / distance * 1000 if pd.notna(distance) and distance > 0 else None
        column, operation = allowed_metrics[metric]
        return getattr(source[column], operation)()

    work = frame.copy()
    if "date" in group_by:
        work["date"] = work["ts"].dt.strftime("%Y-%m-%d")
    if group_by:
        grouped = work.groupby(group_by, dropna=False)
        rows: list[dict[str, Any]] = []
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {name: json_safe(value) for name, value in zip(group_by, keys)}
            for metric in metrics:
                row[metric] = json_safe(calculate_metric(group, metric))
            rows.append(row)
        result_frame = pd.DataFrame(rows)
    else:
        result_frame = pd.DataFrame(
            [{metric: json_safe(calculate_metric(work, metric)) for metric in metrics}]
        )
    result = format_rows(result_frame, MAX_TOOL_ROWS, frame)
    result["filters"] = {key: value for key, value in args.items() if key not in {"group_by", "metrics"} and value is not None}
    result["group_by"] = group_by
    result["metrics"] = metrics
    return result


# Tool descriptions sent to the model. The model reads these to decide which tool to call and
# with what arguments. Every tool here also needs a matching branch in call_tool().
TOOLS = [
    {
        "type": "function",
        "name": "query_runs",
        "description": "Return up to 50 raw robot-run rows after filtering. Use this for exact timestamps and individual events such as faults.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "date": {"type": "string", "description": "UTC calendar date YYYY-MM-DD"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "state": {"type": "string", "enum": sorted(VALID_STATES)},
                "field": {"type": "string", "description": "Farm field/location, such as North 40, Creekside, or Home Quarter. This is not a metric or CSV column name."},
                "exclude_states": {"type": "array", "items": {"type": "string", "enum": sorted(VALID_STATES)}, "description": "States to exclude, for questions such as everything except idle."},
                "exclude_robot_ids": {"type": "array", "items": {"type": "string"}},
                "exclude_fields": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_ROWS},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "aggregate_runs",
        "description": "Aggregate robot-run metrics for comparisons, totals, rates, and counts. Use exclude_states, exclude_robot_ids, or exclude_fields for requests containing except, excluding, or without. efficiency_l_per_km is calculated server-side as sum nitrogen_applied_l / sum distance_m * 1000 and is null when distance is missing or zero. robot_count is the distinct robots observed in each group, not a simultaneous headcount. For fleet-wide rankings or trends, omit robot_id and use group_by to get all robots or dates in one call; do not call once per robot unless the user names specific robots. For named-robot comparisons, group by robot_id and select the requested rows. Each source row is a five-minute interval, so row_count for charging can be converted to minutes by multiplying by 5. If both date and start/end dates are supplied, date is treated as the authoritative UTC calendar day.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "date": {"type": "string", "description": "UTC calendar date YYYY-MM-DD"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "state": {"type": "string", "enum": sorted(VALID_STATES)},
                "field": {"type": "string", "description": "Farm field/location, such as North 40, Creekside, or Home Quarter. This is not a metric or CSV column name."},
                "exclude_states": {"type": "array", "items": {"type": "string", "enum": sorted(VALID_STATES)}, "description": "States to exclude, for questions such as everything except idle."},
                "exclude_robot_ids": {"type": "array", "items": {"type": "string"}},
                "exclude_fields": {"type": "array", "items": {"type": "string"}},
                "group_by": {"type": "array", "items": {"type": "string", "enum": ["robot_id", "field", "state", "date"]}},
                "metrics": {"type": "array", "items": {"type": "string", "enum": ["sum_nitrogen", "sum_distance", "efficiency_l_per_km", "avg_battery", "max_battery", "min_battery", "row_count", "robot_count"]}},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Retrieve historical weather or soil-moisture reanalysis from Open-Meteo for a supplied city/location or latitude and longitude. The CSV has no coordinates, so never infer a farm location from field names. If the user asks for soil moisture, set data_type to soil_moisture. Use this only when the user supplies a location or coordinates; otherwise ask for the location. Results are external data and must be labeled as such.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City, region, or postal code supplied by the user."},
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "data_type": {"type": "string", "enum": ["weather", "soil_moisture"]},
                "date": {"type": "string", "description": "UTC calendar date YYYY-MM-DD"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
]


def call_tool(dataset: Dataset, name: str, args: dict[str, Any]) -> dict[str, Any]:
    # Sends the model's tool request to the matching Python function.
    if name == "query_runs":
        return run_query_runs(dataset, args)
    if name == "aggregate_runs":
        return run_aggregate_runs(dataset, args)
    if name == "get_weather":
        return run_get_weather(dataset, args)
    raise ValueError(f"Unknown tool: {name}")
