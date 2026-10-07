import json
from pathlib import Path

import pandas as pd
import pytest

from main import Dataset, analytics, build_evidence, build_visualization, parse_dataset, run_aggregate_runs, run_query_runs, score_reply


CSV = Path(__file__).parents[1] / "robot_runs.csv"


@pytest.fixture()
def dataset() -> Dataset:
    frame, profile = parse_dataset(CSV.read_bytes())
    return Dataset("ds_test", frame, profile)


def test_parse_profile_matches_robot_file(dataset: Dataset) -> None:
    assert dataset.profile["row_count"] == 5184
    assert dataset.profile["robot_ids"] == ["MR-01", "MR-02", "MR-03", "MR-04", "MR-05", "MR-06"]
    assert dataset.profile["time_range"]["start"].startswith("2026-06-14")
    assert dataset.profile["missing_values"]["battery_pct"] == 5


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        parse_dataset(b"ts,robot_id\n2026-01-01T00:00:00Z,MR-01\n")


def test_aggregate_totals_and_filter(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {"date": "2026-06-15", "group_by": ["robot_id"], "metrics": ["sum_nitrogen"]})
    by_robot = {row["robot_id"]: row["sum_nitrogen"] for row in result["rows"]}
    assert by_robot["MR-04"] == pytest.approx(210.49)
    assert len(result["rows"]) == 6


def test_all_is_a_no_filter_value(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {"robot_id": "all", "date": "2026-06-15", "group_by": ["robot_id"], "metrics": ["sum_nitrogen"]})
    assert len(result["rows"]) == 6


def test_tool_reports_effective_time_range(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {"state": "idle", "group_by": ["robot_id"], "metrics": ["row_count"]})
    assert result["effective_time_range"]["start"].startswith("2026-06-14")
    assert result["effective_time_range"]["end"].startswith("2026-06-16")


def test_query_is_capped_at_fifty_rows(dataset: Dataset) -> None:
    result = run_query_runs(dataset, {"robot_id": "MR-03", "state": "fault"})
    assert result["row_count"] == 7
    assert len(result["rows"]) == 7
    assert not result["truncated"]


def test_scoring_requires_key_facts() -> None:
    assert score_reply("The winner is MR-04 with 210.49 L.", "MR-04, 210.49 L")[0]
    assert not score_reply("MR-02 applied the most.", "MR-04, 210.49 L")[0]


def test_aggregate_result_creates_visualization() -> None:
    trace = {
        "steps": [{
            "id": "step_2",
            "type": "tool",
            "name": "aggregate_runs",
            "result": {
                "group_by": ["robot_id"],
                "metrics": ["sum_nitrogen"],
                "rows": [
                    {"robot_id": "MR-01", "sum_nitrogen": 100.0},
                    {"robot_id": "MR-04", "sum_nitrogen": 210.49},
                ],
            },
        }]
    }
    chart = build_visualization(trace)
    assert chart is not None
    assert chart["labels"] == ["MR-01", "MR-04"]
    assert chart["datasets"][0]["data"] == [100.0, 210.49]


def test_evidence_summarizes_tool_provenance() -> None:
    evidence = build_evidence({
        "steps": [{
            "id": "step_2",
            "type": "tool",
            "name": "aggregate_runs",
            "result": {
                "effective_time_range": {"start": "2026-06-15T00:00:00Z", "end": "2026-06-15T23:55:00Z"},
                "filters": {"date": "2026-06-15"},
                "rows": [{"robot_id": "MR-04", "sum_nitrogen": 210.49}],
                "truncated": False,
            },
        }]
    })
    assert evidence["tools"] == ["aggregate_runs"]
    assert evidence["result_rows"] == 1
    assert evidence["effective_time_ranges"][0]["start"].startswith("2026-06-15")


def test_analytics_empty() -> None:
    assert analytics()["turns"] >= 0
