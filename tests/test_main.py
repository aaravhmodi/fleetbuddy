import json
from pathlib import Path

import pandas as pd
import pytest

import main
from main import Dataset, analytics, build_evidence, build_visualization, normalize_latex_response, parse_dataset, run_aggregate_runs, run_chat, run_query_runs, score_reply


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


def test_server_calculates_efficiency_from_sums(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {
        "robot_id": "MR-04",
        "date": "2026-06-14",
        "state": "applying",
        "metrics": ["efficiency_l_per_km", "sum_nitrogen", "sum_distance"],
    })
    row = result["rows"][0]
    assert row["efficiency_l_per_km"] == pytest.approx(192.79 / 14941 * 1000)
    assert row["sum_nitrogen"] == pytest.approx(192.79)
    assert row["sum_distance"] == pytest.approx(14941)


def test_robot_count_is_distinct_within_each_group(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {"group_by": ["state"], "metrics": ["robot_count"]})
    assert {row["state"]: row["robot_count"] for row in result["rows"]} == {
        "applying": 6,
        "charging": 6,
        "driving": 6,
        "fault": 2,
        "idle": 6,
    }


def test_tool_rejects_invalid_or_oversized_time_windows(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="after"):
        run_query_runs(dataset, {"start_date": "2026-06-16", "end_date": "2026-06-15"})
    with pytest.raises(ValueError, match="90 days"):
        run_query_runs(dataset, {"start_date": "2026-01-01", "end_date": "2026-04-02"})


def test_parse_rejects_negative_measurements() -> None:
    raw = b"ts,robot_id,field,state,battery_pct,nitrogen_applied_l,distance_m\n2026-01-01T00:00:00Z,MR-01,F,driving,50,-1,2\n"
    with pytest.raises(ValueError, match="nitrogen_applied_l cannot contain negative"):
        parse_dataset(raw)


def test_server_normalizes_model_latex_delimiters() -> None:
    reply = normalize_latex_response(r"""[
\text{Efficiency} = \frac{\text{Total Nitrogen Applied (L)}}{\text{Total Distance Traveled (m)}}
]
MR-01: ( \frac{173.05}{16901} \approx 0.01024 ) L/m""")
    assert "$$" in reply
    assert r"\(" in reply and r"\)" in reply
    assert "\n[" not in reply and "\n]" not in reply


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


def test_query_truncates_large_result_to_fifty_rows(dataset: Dataset) -> None:
    result = run_query_runs(dataset, {})
    assert result["row_count"] == 5184
    assert len(result["rows"]) == 50
    assert result["truncated"]


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


class FakeCall:
    type = "function_call"

    def __init__(self, name: str, arguments: str, call_id: str = "call_test") -> None:
        self.name = name
        self.arguments = arguments
        self.call_id = call_id

    def model_dump(self, **_: object) -> dict[str, str]:
        return {"type": "function_call", "name": self.name, "arguments": self.arguments, "call_id": self.call_id}


class FakeResponse:
    def __init__(self, output: list[FakeCall], output_text: str = "") -> None:
        self.output = output
        self.output_text = output_text
        self.usage = type("Usage", (), {"input_tokens": 3, "output_tokens": 2})()


class FakeResponsesClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = self
        self._responses = iter(responses)

    def create(self, **_: object) -> FakeResponse:
        return next(self._responses)


def test_tool_error_is_returned_and_turn_recovers(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([
        FakeResponse([FakeCall("query_runs", '{"limit": 51}')]),
        FakeResponse([], "I could not use that limit, so I continued without returning the oversized result."),
    ])
    monkeypatch.setattr(main, "client", lambda: fake)
    monkeypatch.setattr(main, "TRACES", [])
    result = run_chat(dataset, [main.Message(role="user", content="Show me the runs")])
    assert result["reply"].startswith("I could not use")
    assert result["trace"]["outcome"] == "answered"
    assert result["trace"]["steps"][1]["error"]
    assert result["trace"]["steps"][1]["result"]["error"]


def test_turn_stops_at_eight_model_calls(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([FakeCall("aggregate_runs", "{}", f"call_{i}")]) for i in range(8)])
    monkeypatch.setattr(main, "client", lambda: fake)
    monkeypatch.setattr(main, "TRACES", [])
    result = run_chat(dataset, [main.Message(role="user", content="Keep calling tools")])
    assert result["trace"]["outcome"] == "stopped_at_cap"
    assert sum(step["type"] == "model" for step in result["trace"]["steps"]) == 8


def test_separate_turns_have_separate_trace_ids_and_ordered_offsets(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([], "first"), FakeResponse([], "second")])
    monkeypatch.setattr(main, "client", lambda: fake)
    monkeypatch.setattr(main, "TRACES", [])
    first = run_chat(dataset, [main.Message(role="user", content="first")])
    second = run_chat(dataset, [main.Message(role="user", content="second")])
    assert first["trace_id"] != second["trace_id"]
    assert first["trace"]["steps"][0]["start_offset_ms"] >= 0
    assert second["trace"]["steps"][0]["start_offset_ms"] >= 0


def test_analytics_empty() -> None:
    assert analytics()["turns"] >= 0
