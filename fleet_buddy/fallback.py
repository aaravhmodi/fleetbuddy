"""Computed answers for a few known question shapes; they replace the model's answer when they match."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .config import VALID_STATES
from .state import Dataset
from .tools import run_aggregate_runs


def question_date(question: str) -> str | None:
    match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    year = match.group(3) or "2026"
    parsed = pd.to_datetime(f"{match.group(1)} {match.group(2)}, {year}", errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def fallback_aggregate_answer(dataset: Dataset, question: str) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Answer a few high-value analytical shapes if the model exhausts its call budget."""
    # Note: run_chat calls this after every turn, so when a question matches one of these
    # patterns, this computed answer replaces the model's answer.
    lowered = question.lower()
    if "state" in lowered and any(state in lowered for state in VALID_STATES):
        return None
    args: dict[str, Any]
    result: dict[str, Any]
    if "lowest average battery" in lowered:
        date = question_date(question)
        if not date:
            return None
        args = {"date": date, "group_by": ["robot_id"], "metrics": ["avg_battery"]}
        result = run_aggregate_runs(dataset, args)
        rows = [row for row in result["rows"] if row.get("avg_battery") is not None]
        if not rows:
            return None
        winner = min(rows, key=lambda row: row["avg_battery"])
        reply = (
            f"**{winner['robot_id']}** had the lowest average battery on **{date} (UTC)**, "
            f"at **{winner['avg_battery']:.2f}%**."
        )
        return reply, args, result

    if "litres applied per kilometre" in lowered and "compare" in lowered:
        robot_ids = sorted(set(re.findall(r"MR-\d{2}", question, re.IGNORECASE)))
        if len(robot_ids) < 2:
            return None
        args = {
            "group_by": ["robot_id"],
            "metrics": ["efficiency_l_per_km", "sum_nitrogen", "sum_distance"],
        }
        result = run_aggregate_runs(dataset, args)
        by_robot = {row.get("robot_id"): row for row in result["rows"]}
        if any(robot.upper() not in by_robot for robot in robot_ids):
            return None
        lines = ["## Litres applied per kilometre", "", "Across the **full dataset**:", ""]
        for robot in robot_ids:
            row = by_robot[robot.upper()]
            lines.append(
                f"- **{robot.upper()}**: **{row['efficiency_l_per_km']:.2f} L/km** "
                f"({row['sum_nitrogen']:.2f} L over {row['sum_distance']:.0f} m)"
            )
        return "\n".join(lines), args, result

    if "lowest fleet efficiency" in lowered:
        args = {
            "group_by": ["date"],
            "metrics": ["efficiency_l_per_km", "sum_nitrogen", "sum_distance"],
        }
        result = run_aggregate_runs(dataset, args)
        rows = [row for row in result["rows"] if row.get("efficiency_l_per_km") is not None]
        if not rows:
            return None
        winner = min(rows, key=lambda row: row["efficiency_l_per_km"])
        display_date = pd.to_datetime(winner["date"]).strftime("%B %d, %Y").replace(" 0", " ")
        reply = (
            f"**{display_date}** had the lowest fleet efficiency at "
            f"**{winner['efficiency_l_per_km']:.2f} L/km**, using total nitrogen divided by "
            f"total distance across all states."
        )
        return reply, args, result
    return None
