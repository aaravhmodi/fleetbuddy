"""Loading and validating the uploaded CSV, plus the shared filters the tools use."""

from __future__ import annotations

import io
import math
from typing import Any

import pandas as pd

from .config import EXPECTED_COLUMNS, MAX_FILTER_STRING_LENGTH, MAX_QUERY_WINDOW_DAYS, MAX_TOOL_ROWS, VALID_STATES
from .state import Dataset
from .utils import json_safe


def profile_frame(frame: pd.DataFrame) -> dict[str, Any]:
    # A summary of the dataset (columns, date range, robots, fields). This is what the model
    # sees in its prompt; it never sees the raw rows.
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
    # Reads the uploaded CSV and rejects wrong columns, bad timestamps, unknown states,
    # or impossible numbers (e.g. battery over 100%). Blank numbers are allowed.
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
    # Models sometimes expand a calendar date into an equivalent full-day range.
    # Treat the explicit calendar date as authoritative so that harmless duplicate
    # context does not trigger a tool-call retry loop.
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


def string_list_argument(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key) or []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings.")
    if len(value) > 20:
        raise ValueError(f"{key} supports at most 20 values.")
    cleaned = [item.strip() for item in value if item.strip()]
    if any(len(item) > MAX_FILTER_STRING_LENGTH for item in cleaned):
        raise ValueError(f"{key} values cannot exceed {MAX_FILTER_STRING_LENGTH} characters.")
    return cleaned


def filtered_frame(dataset: Dataset, args: dict[str, Any]) -> pd.DataFrame:
    # Shared filtering for both data tools: robot, state, field, exclusions, then date or date range.
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
    exclude_states = [value.lower() for value in string_list_argument(args, "exclude_states")]
    invalid_states = sorted(set(exclude_states) - VALID_STATES)
    if invalid_states:
        raise ValueError(f"exclude_states contains unsupported value(s): {', '.join(invalid_states)}.")
    exclude_robot_ids = string_list_argument(args, "exclude_robot_ids")
    invalid_robots = sorted(set(exclude_robot_ids) - set(dataset.profile.get("robot_ids", [])))
    if invalid_robots:
        raise ValueError(f"exclude_robot_ids contains unsupported value(s): {', '.join(invalid_robots)}.")
    exclude_fields = string_list_argument(args, "exclude_fields")
    invalid_fields = sorted(set(exclude_fields) - set(dataset.profile.get("fields", [])))
    if invalid_fields:
        raise ValueError(f"exclude_fields contains unsupported value(s): {', '.join(invalid_fields)}.")
    if exclude_states:
        frame = frame[~frame["state"].isin(exclude_states)]
    if exclude_robot_ids:
        frame = frame[~frame["robot_id"].isin(exclude_robot_ids)]
    if exclude_fields:
        frame = frame[~frame["field"].isin(exclude_fields)]
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
