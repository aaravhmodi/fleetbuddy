from __future__ import annotations

import io
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field


EXPECTED_COLUMNS = [
    "ts",
    "robot_id",
    "field",
    "state",
    "battery_pct",
    "nitrogen_applied_l",
    "distance_m",
]
VALID_STATES = {"applying", "driving", "charging", "idle", "fault"}
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
INPUT_PRICE_PER_TOKEN = 0.15 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000
MAX_MODEL_CALLS = 8
MAX_TOOL_ROWS = 50


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


class EvalCase(BaseModel):
    messages: list[Message] = Field(min_length=1)
    expected: str


class Dataset:
    def __init__(self, dataset_id: str, frame: pd.DataFrame, profile: dict[str, Any]):
        self.id = dataset_id
        self.frame = frame
        self.profile = profile


DATASETS: dict[str, Dataset] = {}
TRACES: list[dict[str, Any]] = []


app = FastAPI(title="Fleet Buddy", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def profile_frame(frame: pd.DataFrame) -> dict[str, Any]:
    type_names: dict[str, str] = {
        "ts": "datetime",
        "robot_id": "string",
        "field": "string",
        "state": "string",
        "battery_pct": "number",
        "nitrogen_applied_l": "number",
        "distance_m": "number",
    }
    return {
        "columns": [{"name": name, "type": type_names[name]} for name in EXPECTED_COLUMNS],
        "row_count": int(len(frame)),
        "time_range": {
            "start": frame["ts"].min().isoformat().replace("+00:00", "Z"),
            "end": frame["ts"].max().isoformat().replace("+00:00", "Z"),
        },
        "robot_ids": sorted(frame["robot_id"].dropna().unique().tolist()),
        "fields": sorted(frame["field"].dropna().unique().tolist()),
        "states": sorted(frame["state"].dropna().unique().tolist()),
        "missing_values": {column: int(frame[column].isna().sum()) for column in EXPECTED_COLUMNS},
        "notes": [
            "ts is normalized to UTC.",
            "Numeric blanks are retained as null and ignored by aggregations.",
            "Each source row represents a five-minute robot interval.",
        ],
    }


def parse_dataset(raw: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not raw:
        raise ValueError("The CSV file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The CSV file has a header but no data rows.")
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing columns: {', '.join(missing)}")
        if extra:
            pieces.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError("Unexpected robot run shape; " + "; ".join(pieces) + ".")
    frame = frame[EXPECTED_COLUMNS].copy()
    timestamps = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    if timestamps.isna().any():
        count = int(timestamps.isna().sum())
        raise ValueError(f"ts contains {count} invalid or blank timestamp(s); expected ISO 8601 values.")
    frame["ts"] = timestamps
    if frame["robot_id"].isna().any() or frame["robot_id"].astype(str).str.strip().eq("").any():
        raise ValueError("robot_id cannot be blank.")
    if frame["field"].isna().any() or frame["state"].isna().any():
        raise ValueError("field and state cannot be blank.")
    frame["robot_id"] = frame["robot_id"].astype(str).str.strip()
    frame["field"] = frame["field"].astype(str).str.strip()
    frame["state"] = frame["state"].astype(str).str.strip().str.lower()
    bad_states = sorted(set(frame["state"]) - VALID_STATES)
    if bad_states:
        raise ValueError(f"state contains unsupported value(s): {', '.join(bad_states)}.")
    for column in ["battery_pct", "nitrogen_applied_l", "distance_m"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["battery_pct"].dropna().lt(0).any() or frame["battery_pct"].dropna().gt(100).any():
        raise ValueError("battery_pct must be between 0 and 100.")
    return frame, profile_frame(frame)


def parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date or timestamp: {value}")
    return parsed


def filtered_frame(dataset: Dataset, args: dict[str, Any]) -> pd.DataFrame:
    frame = dataset.frame
    robot_id = args.get("robot_id")
    state = args.get("state")
    field = args.get("field")
    if robot_id and str(robot_id).lower() not in {"all", "any", "fleet"}:
        frame = frame[frame["robot_id"] == robot_id]
    if state and str(state).lower() not in {"all", "any"}:
        if state not in VALID_STATES:
            raise ValueError(f"state must be one of: {', '.join(sorted(VALID_STATES))}")
        frame = frame[frame["state"] == state]
    if field and str(field).lower() not in {"all", "any"}:
        valid_fields = set(dataset.profile.get("fields", []))
        if field not in valid_fields:
            raise ValueError(f"field must be one of: {', '.join(sorted(valid_fields))}")
        frame = frame[frame["field"] == field]
    if args.get("date"):
        start = parse_date(str(args["date"]))
        assert start is not None
        frame = frame[frame["ts"].dt.date == start.date()]
    else:
        start = parse_date(args.get("start_date"))
        end = parse_date(args.get("end_date"))
        if start is not None:
            frame = frame[frame["ts"] >= start]
        if end is not None:
            # Date-only end dates are inclusive; timestamps remain exact.
            end_value = end + pd.Timedelta(days=1) if len(str(args["end_date"])) <= 10 else end
            frame = frame[frame["ts"] < end_value]
    return frame


def format_rows(frame: pd.DataFrame, limit: int = MAX_TOOL_ROWS, time_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    limited = frame.head(limit).copy()
    rows = []
    for row in limited.to_dict(orient="records"):
        rows.append(json_safe(row))
    result = {"rows": rows, "row_count": int(len(frame)), "truncated": len(frame) > limit}
    range_frame = time_frame if time_frame is not None else frame
    if len(range_frame) and "ts" in range_frame.columns:
        result["effective_time_range"] = {
            "start": range_frame["ts"].min().isoformat().replace("+00:00", "Z"),
            "end": range_frame["ts"].max().isoformat().replace("+00:00", "Z"),
        }
    else:
        result["effective_time_range"] = None
    return result


def run_query_runs(dataset: Dataset, args: dict[str, Any]) -> dict[str, Any]:
    frame = filtered_frame(dataset, args)
    limit = int(args.get("limit", MAX_TOOL_ROWS))
    if limit < 1 or limit > MAX_TOOL_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_TOOL_ROWS}.")
    result = format_rows(frame.sort_values("ts"), limit)
    result["filters"] = {key: value for key, value in args.items() if value is not None}
    return result


def run_aggregate_runs(dataset: Dataset, args: dict[str, Any]) -> dict[str, Any]:
    frame = filtered_frame(dataset, args)
    group_by = args.get("group_by", [])
    if isinstance(group_by, str):
        group_by = [group_by]
    valid_groups = {"robot_id", "field", "state", "date"}
    if any(group not in valid_groups for group in group_by):
        raise ValueError("group_by values must be robot_id, field, state, or date.")
    metrics = args.get("metrics", ["sum_nitrogen", "sum_distance"])
    allowed_metrics = {
        "sum_nitrogen": ("nitrogen_applied_l", "sum"),
        "sum_distance": ("distance_m", "sum"),
        "avg_battery": ("battery_pct", "mean"),
        "max_battery": ("battery_pct", "max"),
        "min_battery": ("battery_pct", "min"),
        "row_count": ("ts", "count"),
    }
    if not metrics or any(metric not in allowed_metrics for metric in metrics):
        raise ValueError("metrics must use sum_nitrogen, sum_distance, avg_battery, max_battery, min_battery, or row_count.")
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
                column, operation = allowed_metrics[metric]
                row[metric] = json_safe(getattr(group[column], operation)())
            rows.append(row)
        result_frame = pd.DataFrame(rows)
    else:
        result_frame = pd.DataFrame(
            [{
                metric: json_safe(getattr(work[column], operation)())
                for metric in metrics
                for column, operation in [allowed_metrics[metric]]
            }]
        )
    result = format_rows(result_frame, MAX_TOOL_ROWS, frame)
    result["filters"] = {key: value for key, value in args.items() if key not in {"group_by", "metrics"} and value is not None}
    result["group_by"] = group_by
    result["metrics"] = metrics
    return result


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
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_ROWS},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "aggregate_runs",
        "description": "Aggregate robot-run metrics for comparisons, totals, rates, and counts. Each source row is a five-minute interval, so row_count for charging can be converted to minutes by multiplying by 5.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "date": {"type": "string", "description": "UTC calendar date YYYY-MM-DD"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "state": {"type": "string", "enum": sorted(VALID_STATES)},
                "field": {"type": "string", "description": "Farm field/location, such as North 40, Creekside, or Home Quarter. This is not a metric or CSV column name."},
                "group_by": {"type": "array", "items": {"type": "string", "enum": ["robot_id", "field", "state", "date"]}},
                "metrics": {"type": "array", "items": {"type": "string", "enum": ["sum_nitrogen", "sum_distance", "avg_battery", "max_battery", "min_battery", "row_count"]}},
            },
            "additionalProperties": False,
        },
    },
]


SYSTEM_PROMPT_TEMPLATE = """You are Fleet Buddy, an accurate analyst for a robot-run CSV.
Answer only from tool results. The dataset profile below describes available columns and scope; it deliberately contains no rows.
If the question asks for information outside the profile/data (for example weather), say clearly that the dataset cannot answer it. Do not invent facts.
Use aggregate_runs for totals, comparisons, rates, and counts; use query_runs for exact events and timestamps. When calculating litres per kilometre, sum litres divided by sum metres times 1000. State assumptions and units, and round sensibly.
The CSV column named field means farm field/location. Never put a metric name such as nitrogen_applied_l in field; metric names belong only in metrics.
Format answers with Markdown: use **bold** for key results, headings with `##`, and bullet lists when useful. For equations, use LaTeX delimiters such as `\\( ... \\)` for inline math or `\\[ ... \\]` for display math. Do not wrap ordinary text in math delimiters.
Always state the effective timeframe for time-based answers. If the user did not provide a date or date range, explicitly say "across the full dataset" and include the profile's start and end dates. Never call it a "selected timeframe" unless the user actually selected or supplied one.

DATASET PROFILE:
{profile}
"""


def client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def usage_values(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def function_calls(response: Any) -> list[dict[str, Any]]:
    calls = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call" or (isinstance(item, dict) and item.get("type") == "function_call"):
            dumped = model_dump(item)
            calls.append(dumped)
    return calls


def output_items(response: Any) -> list[dict[str, Any]]:
    return [model_dump(item) for item in (getattr(response, "output", []) or [])]


def call_tool(dataset: Dataset, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "query_runs":
        return run_query_runs(dataset, args)
    if name == "aggregate_runs":
        return run_aggregate_runs(dataset, args)
    raise ValueError(f"Unknown tool: {name}")


def metric_label(metric: str) -> str:
    return {
        "sum_nitrogen": "Nitrogen applied (L)",
        "sum_distance": "Distance driven (m)",
        "avg_battery": "Average battery (%)",
        "max_battery": "Maximum battery (%)",
        "min_battery": "Minimum battery (%)",
        "row_count": "Intervals / rows",
    }.get(metric, metric.replace("_", " ").title())


def build_visualization(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Turn the structured tool result into a small, honest chart for the UI."""
    aggregate_groups: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
    query_candidates: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
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
    if aggregate_groups:
        group = max(aggregate_groups.values(), key=lambda item: (len(item["rows"]), item["step_index"]))
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
    return None


def run_chat(dataset: Dataset, messages: list[Message]) -> dict[str, Any]:
    trace = {
        "id": f"tr_{uuid.uuid4().hex[:12]}",
        "dataset_id": dataset.id,
        "created_at": now_iso(),
        "question": next((message.content for message in reversed(messages) if message.role == "user"), ""),
        "steps": [],
        "model": MODEL,
        "outcome": "answered",
        "reply": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "duration_ms": 0.0,
        "visualization": None,
    }
    started = time.perf_counter()
    prompt = SYSTEM_PROMPT_TEMPLATE.format(profile=json.dumps(dataset.profile, indent=2))
    input_items: list[dict[str, Any]] = [message.model_dump() for message in messages]
    try:
        api_client = client()
        reply = ""
        for call_number in range(1, MAX_MODEL_CALLS + 1):
            step_start = time.perf_counter()
            step_started_at = now_iso()
            response = api_client.responses.create(model=MODEL, instructions=prompt, input=input_items, tools=TOOLS)
            input_tokens, output_tokens = usage_values(response)
            trace["input_tokens"] += input_tokens
            trace["output_tokens"] += output_tokens
            trace["cost"] += input_tokens * INPUT_PRICE_PER_TOKEN + output_tokens * OUTPUT_PRICE_PER_TOKEN
            calls = function_calls(response)
            text_output = getattr(response, "output_text", "") or ""
            model_step = {
                "id": f"step_{len(trace['steps']) + 1}",
                "type": "model",
                "name": "responses.create",
                "call_number": call_number,
                "started_at": step_started_at,
                "start_offset_ms": round((step_start - started) * 1000, 2),
                "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": round(input_tokens * INPUT_PRICE_PER_TOKEN + output_tokens * OUTPUT_PRICE_PER_TOKEN, 8),
                "function_calls": [{"name": c.get("name"), "call_id": c.get("call_id")} for c in calls],
                "output_text": text_output,
            }
            trace["steps"].append(model_step)
            if not calls:
                reply = text_output.strip()
                break
            if call_number >= MAX_MODEL_CALLS:
                trace["outcome"] = "stopped_at_cap"
                reply = "I stopped after reaching the model-call limit before I could finish the answer."
                break
            input_items.extend(output_items(response))
            for call in calls:
                tool_started = time.perf_counter()
                tool_started_at = now_iso()
                name = str(call.get("name", ""))
                raw_args = call.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    result = call_tool(dataset, name, args)
                    tool_error = None
                except Exception as exc:  # Tool errors are intentionally returned to the model.
                    args = args if "args" in locals() and isinstance(args, dict) else {"raw": raw_args}
                    result = {"error": str(exc)}
                    tool_error = str(exc)
                trace["steps"].append({
                    "id": f"step_{len(trace['steps']) + 1}",
                    "type": "tool",
                    "name": name,
                    "started_at": tool_started_at,
                    "start_offset_ms": round((tool_started - started) * 1000, 2),
                    "duration_ms": round((time.perf_counter() - tool_started) * 1000, 2),
                    "arguments": json_safe(args),
                    "result": json_safe(result),
                    "error": tool_error,
                })
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": json.dumps(json_safe(result)),
                })
        else:
            trace["outcome"] = "stopped_at_cap"
            reply = "I stopped after reaching the model-call limit before I could finish the answer."
        trace["reply"] = reply or "I could not produce an answer from the available data."
    except Exception as exc:
        trace["outcome"] = "failed"
        trace["error"] = str(exc)
        trace["reply"] = "I couldn't complete that turn. Please check the server configuration and try again."
    trace["visualization"] = build_visualization(trace)
    trace["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace["cost"] = round(trace["cost"], 8)
    TRACES.append(trace)
    return {"reply": trace["reply"], "trace_id": trace["id"], "visualization": trace["visualization"], "trace": trace}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%+-]+", " ", value.lower())).strip()


def score_reply(reply: str, expected: str) -> tuple[bool, str]:
    actual = normalize_text(reply)
    target = normalize_text(expected)
    if not actual:
        return False, "empty reply"
    if target in actual:
        return True, "expected answer found in reply"
    target_tokens = [token for token in target.split() if token not in {"the", "a", "an", "is", "was", "of", "on", "and", "to", "in", "for", "with"}]
    numeric_or_ids = [token for token in target_tokens if any(character.isdigit() for character in token) or token.startswith("mr-")]
    missing_facts = [token for token in numeric_or_ids if token not in actual]
    if missing_facts:
        return False, f"missing key fact(s): {', '.join(missing_facts)}"
    matched = sum(token in actual for token in target_tokens)
    threshold = max(1, math.ceil(len(target_tokens) * 0.65))
    if matched >= threshold:
        return True, f"matched {matched}/{len(target_tokens)} expected facts"
    return False, f"matched {matched}/{len(target_tokens)} expected facts"


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
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile_value) - 1))
    return round(ordered[index], 2)


def analytics() -> dict[str, Any]:
    traces = list(TRACES)
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
        "price_per_million_tokens": {"input": INPUT_PRICE_PER_TOKEN * 1_000_000, "output": OUTPUT_PRICE_PER_TOKEN * 1_000_000},
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    try:
        frame, profile = parse_dataset(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    DATASETS[dataset_id] = Dataset(dataset_id, frame, profile)
    return {"dataset_id": dataset_id, "profile": profile}


@app.post("/datasets/{dataset_id}/chat")
def chat(dataset_id: str, request: ChatRequest, response: Response) -> dict[str, Any]:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    result = run_chat(dataset, request.messages)
    response.headers["X-Trace-ID"] = result["trace_id"]
    return {"reply": result["reply"], "trace_id": result["trace_id"], "visualization": result["visualization"]}


@app.get("/traces")
def list_traces() -> list[dict[str, Any]]:
    return [summarize_trace(trace) for trace in reversed(TRACES)]


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    trace = next((trace for trace in TRACES if trace["id"] == trace_id), None)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


@app.get("/analytics")
def get_analytics() -> dict[str, Any]:
    return analytics()


@app.get("/evals")
def get_evals() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "evals.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/datasets/{dataset_id}/evals")
def run_evals(dataset_id: str, cases: list[EvalCase]) -> dict[str, Any]:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    results = []
    for index, case in enumerate(cases):
        run = run_chat(dataset, case.messages)
        passed, reason = score_reply(run["reply"], case.expected)
        results.append({
            "index": index,
            "passed": passed,
            "reason": reason,
            "expected": case.expected,
            "reply": run["reply"],
            "visualization": run["visualization"],
            "trace_id": run["trace_id"],
        })
    passed_count = sum(result["passed"] for result in results)
    return {"pass_rate": round(passed_count / len(results), 4) if results else 0.0, "passed": passed_count, "total": len(results), "results": results}
