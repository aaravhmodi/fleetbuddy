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
import httpx
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
MAX_QUERY_WINDOW_DAYS = 90
MAX_FILTER_STRING_LENGTH = 64
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


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
WEATHER_CACHE: dict[str, dict[str, Any]] = {}


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
    for column in ["nitrogen_applied_l", "distance_m"]:
        values = frame[column].dropna()
        if not values.map(math.isfinite).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        if values.lt(0).any():
            raise ValueError(f"{column} cannot contain negative values.")
    return frame, profile_frame(frame)


def parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date or timestamp: {value}")
    return parsed


def validate_tool_window(args: dict[str, Any]) -> None:
    for key in ("robot_id", "field"):
        value = args.get(key)
        if value is not None and len(str(value)) > MAX_FILTER_STRING_LENGTH:
            raise ValueError(f"{key} exceeds maximum length of {MAX_FILTER_STRING_LENGTH}.")
    if args.get("date") and (args.get("start_date") or args.get("end_date")):
        raise ValueError("Use date or start_date/end_date, not both.")
    if args.get("date"):
        parse_date(str(args["date"]))
        return
    start = parse_date(args.get("start_date"))
    end = parse_date(args.get("end_date"))
    if start is None or end is None:
        return
    effective_end = end + pd.Timedelta(days=1) if len(str(args["end_date"])) <= 10 else end
    if effective_end <= start:
        raise ValueError("end_date must be after start_date.")
    if effective_end - start > pd.Timedelta(days=MAX_QUERY_WINDOW_DAYS):
        raise ValueError(f"Time window cannot exceed {MAX_QUERY_WINDOW_DAYS} days.")


def filtered_frame(dataset: Dataset, args: dict[str, Any]) -> pd.DataFrame:
    validate_tool_window(args)
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


def run_get_weather(dataset: Dataset, args: dict[str, Any]) -> dict[str, Any]:
    location = str(args.get("location") or "").strip()
    latitude = args.get("latitude")
    longitude = args.get("longitude")
    if len(location) > MAX_FILTER_STRING_LENGTH:
        raise ValueError(f"location exceeds maximum length of {MAX_FILTER_STRING_LENGTH}.")
    if location and (latitude is not None or longitude is not None):
        raise ValueError("Use location or latitude/longitude, not both.")
    if not location and (latitude is None or longitude is None):
        raise ValueError("A weather location is required. Provide location or both latitude and longitude.")
    if latitude is not None or longitude is not None:
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError) as exc:
            raise ValueError("latitude and longitude must be numeric.") from exc
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude must be between -90 and 90 and longitude between -180 and 180.")

    validate_tool_window(args)
    date_value = args.get("date")
    if date_value:
        start_date = end_date = parse_date(str(date_value)).strftime("%Y-%m-%d")
    else:
        start = parse_date(args.get("start_date"))
        end = parse_date(args.get("end_date"))
        start_date = (start or parse_date(dataset.profile["time_range"]["start"])).strftime("%Y-%m-%d")
        end_date = (end or start or parse_date(dataset.profile["time_range"]["end"])).strftime("%Y-%m-%d")
    if end_date < start_date:
        raise ValueError("end_date must be after start_date.")

    if location:
        cache_key = json.dumps({"location": location.lower()}, sort_keys=True)
        geocoded = WEATHER_CACHE.get(cache_key)
        if geocoded is None:
            response = httpx.get(OPEN_METEO_GEOCODING_URL, params={"name": location, "count": 1, "language": "en", "format": "json"}, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                raise ValueError(f"Open-Meteo could not find a location matching '{location}'.")
            match = results[0]
            geocoded = {
                "name": match.get("name") or location,
                "admin1": match.get("admin1"),
                "country": match.get("country"),
                "latitude": float(match["latitude"]),
                "longitude": float(match["longitude"]),
            }
            WEATHER_CACHE[cache_key] = geocoded
        latitude = geocoded["latitude"]
        longitude = geocoded["longitude"]
        resolved_location = ", ".join(part for part in [geocoded.get("name"), geocoded.get("admin1"), geocoded.get("country")] if part)
    else:
        resolved_location = f"{latitude:.4f}, {longitude:.4f}"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weather_code,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max",
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    response = httpx.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason") or "Open-Meteo returned a weather error.")
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    rows = []
    for index, date in enumerate(times):
        rows.append({
            "date": date,
            "weather_code": (daily.get("weather_code") or [None] * len(times))[index],
            "temperature_mean_c": (daily.get("temperature_2m_mean") or [None] * len(times))[index],
            "temperature_max_c": (daily.get("temperature_2m_max") or [None] * len(times))[index],
            "temperature_min_c": (daily.get("temperature_2m_min") or [None] * len(times))[index],
            "precipitation_mm": (daily.get("precipitation_sum") or [None] * len(times))[index],
            "rain_mm": (daily.get("rain_sum") or [None] * len(times))[index],
            "wind_max_kmh": (daily.get("wind_speed_10m_max") or [None] * len(times))[index],
        })
    return {
        "source": "Open-Meteo historical weather API",
        "location": resolved_location,
        "latitude": latitude,
        "longitude": longitude,
        "effective_time_range": {"start": f"{start_date}T00:00:00", "end": f"{end_date}T23:59:59"},
        "rows": rows,
        "units": {"temperature": "°C", "precipitation": "mm", "wind": "km/h"},
        "note": "Historical weather is external reanalysis data, not a measurement stored in the uploaded CSV.",
    }


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
        "description": "Aggregate robot-run metrics for comparisons, totals, rates, and counts. efficiency_l_per_km is calculated server-side as sum nitrogen_applied_l / sum distance_m * 1000 and is null when distance is missing or zero. robot_count is the distinct robots observed in each group, not a simultaneous headcount. For fleet-wide rankings or trends, omit robot_id and use group_by to get all robots or dates in one call; do not call once per robot unless the user names specific robots. Each source row is a five-minute interval, so row_count for charging can be converted to minutes by multiplying by 5.",
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
                "metrics": {"type": "array", "items": {"type": "string", "enum": ["sum_nitrogen", "sum_distance", "efficiency_l_per_km", "avg_battery", "max_battery", "min_battery", "row_count", "robot_count"]}},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Retrieve daily historical weather from Open-Meteo for a supplied city/location or latitude and longitude. The CSV has no coordinates, so never infer a farm location from field names. Use this only when the user supplies a location or coordinates; otherwise ask for the location. Weather is external data and must be labeled as such.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City, region, or postal code, such as Toronto, Canada."},
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "date": {"type": "string", "description": "UTC calendar date YYYY-MM-DD"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
]


SYSTEM_PROMPT_TEMPLATE = """You are Fleet Buddy, an accurate analyst for a robot-run CSV.
Answer only from tool results. The dataset profile below describes available columns and scope; it deliberately contains no rows.
If the question asks for information outside the CSV, do not invent facts. For weather, use get_weather only when the user supplies a city/location or coordinates; if no location is supplied, clearly ask for it. Label Open-Meteo results as external historical weather and never imply the CSV contained weather.
Use aggregate_runs for totals, comparisons, rates, and counts; use query_runs for exact events and timestamps. For litres per kilometre, request the server-side efficiency_l_per_km metric and include sum_nitrogen and sum_distance when the numerator and denominator help explain the result. Never divide by zero or treat a null denominator as zero. `row_count` means five-minute source intervals, not distinct robots or events; use `robot_count` for distinct robots observed in a group and do not present it as simultaneous headcount. State assumptions and units, and round sensibly.
The CSV column named field means farm field/location. Never put a metric name such as nitrogen_applied_l in field; metric names belong only in metrics.
Format answers with Markdown: use **bold** for key results, headings with `##`, and bullet lists when useful. For equations, use `\\( ... \\)` for inline math and `$$ ... $$` for display math. For example, write `MR-01: \\(\\frac{{173.05}}{{16901}} \\approx 0.01024\\) L/m`, never `( \\frac{{...}} )` without delimiters. Never use bare `[` and `]` lines to delimit an equation. Do not wrap ordinary text in math delimiters.
Always state the effective timeframe for time-based answers. If the user did not provide a date or date range, explicitly say "across the full dataset" and include the profile's start and end dates. Never call it a "selected timeframe" unless the user actually selected or supplied one.
Use the full conversation history to resolve follow-ups, pronouns, and relative dates such as "the day before". Carry forward an earlier data limitation: if the prior question asked for unavailable weather or another missing field, a follow-up about another date is also unanswerable rather than an invitation to guess.
For fleet-wide rankings, totals by robot, and trends, make one aggregate_runs call with robot_id omitted and group_by set to robot_id or date. Use separate calls only when comparing explicitly named robots or when the first result is insufficient.

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
    if name == "get_weather":
        return run_get_weather(dataset, args)
    raise ValueError(f"Unknown tool: {name}")


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
    if weather_candidates:
        _, step, rows = weather_candidates[-1]
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
    tool_steps = [step for step in trace.get("steps", []) if step.get("type") == "tool" and not step.get("error")]
    if not tool_steps:
        return None
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
        "evidence": None,
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
                reply = normalize_latex_response(text_output.strip())
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
    trace["evidence"] = build_evidence(trace)
    trace["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace["cost"] = round(trace["cost"], 8)
    TRACES.append(trace)
    return {"reply": trace["reply"], "trace_id": trace["id"], "visualization": trace["visualization"], "evidence": trace["evidence"], "trace": trace}


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
    return {"reply": result["reply"], "trace_id": result["trace_id"], "visualization": result["visualization"], "evidence": result["evidence"]}


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
            "evidence": run["evidence"],
            "trace_id": run["trace_id"],
        })
    passed_count = sum(result["passed"] for result in results)
    return {"pass_rate": round(passed_count / len(results), 4) if results else 0.0, "passed": passed_count, "total": len(results), "results": results}
