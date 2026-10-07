import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from fleet_buddy import chat, state, weather
from fleet_buddy.analytics import analytics
from fleet_buddy.chat import run_chat
from fleet_buddy.data import parse_dataset
from fleet_buddy.evals import score_reply
from fleet_buddy.evidence import build_evidence, build_visualization, normalize_latex_response
from fleet_buddy.fallback import fallback_aggregate_answer
from fleet_buddy.schemas import Message
from fleet_buddy.state import Dataset
from fleet_buddy.tools import run_aggregate_runs, run_query_runs
from fleet_buddy.weather import run_get_weather


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


def test_exclusion_filters_are_applied_server_side(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {
        "exclude_states": ["idle"],
        "exclude_robot_ids": ["MR-06"],
        "exclude_fields": ["North 40"],
        "metrics": ["row_count"],
    })
    expected = dataset.frame[
        (dataset.frame["state"] != "idle")
        & (dataset.frame["robot_id"] != "MR-06")
        & (dataset.frame["field"] != "North 40")
    ]
    assert result["rows"][0]["row_count"] == len(expected)
    assert result["filters"] == {
        "exclude_states": ["idle"],
        "exclude_robot_ids": ["MR-06"],
        "exclude_fields": ["North 40"],
    }


def test_exclusion_filters_reject_unknown_values(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        run_query_runs(dataset, {"exclude_states": ["sleeping"]})


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


def test_calendar_date_wins_over_redundant_full_day_range(dataset: Dataset) -> None:
    result = run_aggregate_runs(dataset, {
        "date": "2026-06-15",
        "start_date": "2026-06-15T00:00:00Z",
        "end_date": "2026-06-15T23:55:00Z",
        "group_by": ["robot_id"],
        "metrics": ["sum_nitrogen"],
    })
    assert len(result["rows"]) == 6


def test_date_conflicting_with_wider_range_is_an_error(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        run_aggregate_runs(dataset, {
            "robot_id": "MR-01",
            "date": "2026-06-15",
            "start_date": "2026-06-14T00:00:00Z",
            "end_date": "2026-06-16T23:55:00Z",
            "group_by": ["state"],
            "metrics": ["row_count"],
        })


def test_scorer_accepts_rounding_but_not_wrong_values() -> None:
    expected = "MR-03 approximately 45.27% average battery"
    assert score_reply("MR-03 averaged 45.3% battery", expected)[0]
    assert not score_reply("MR-03 averaged 45% battery", expected)[0]
    assert not score_reply("MR-03 averaged 46.3% battery", expected)[0]


def test_unrequested_state_filter_is_rejected() -> None:
    from fleet_buddy.grounding import unrequested_state_error

    distance = [Message(role="user", content="Which robot had the greatest total distance traveled?")]
    assert unrequested_state_error({"state": "applying"}, distance)
    assert unrequested_state_error({}, distance) is None
    nitrogen = [Message(role="user", content="Which robot applied the most nitrogen?")]
    assert unrequested_state_error({"state": "applying"}, nitrogen) is None
    faults = [Message(role="user", content="How many times did MR-03 fault?")]
    assert unrequested_state_error({"state": "fault"}, faults) is None


def test_location_only_weather_reply_states_csv_limitation() -> None:
    from fleet_buddy.grounding import ensure_environment_follow_up, reply_states_limitation

    messages = [Message(role="user", content="What was the soil moisture on June 15?")]
    reply = ensure_environment_follow_up(messages, "Please provide a specific location for June 15.", [])
    assert reply.startswith("The uploaded CSV doesn't contain soil moisture data")
    assert reply_states_limitation(reply)
    assert "Open-Meteo" not in reply  # the model already asked for a location


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


def test_fallback_planner_uses_dataset_aggregates(dataset: Dataset) -> None:
    result = fallback_aggregate_answer(dataset, "Which robot had the lowest average battery percentage on June 15?")
    assert result is not None
    reply, args, _ = result
    assert "MR-02" in reply and "57.13%" in reply
    assert args == {"date": "2026-06-15", "group_by": ["robot_id"], "metrics": ["avg_battery"]}

    result = fallback_aggregate_answer(dataset, "Which date had the lowest fleet efficiency in litres per kilometre?")
    assert result is not None
    assert "June 16, 2026" in result[0] and "7.85 L/km" in result[0]


def test_weather_requires_a_location(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="location is required"):
        run_get_weather(dataset, {"date": "2026-06-15"})


def test_weather_uses_geocoding_and_archive(monkeypatch: pytest.MonkeyPatch, dataset: Dataset) -> None:
    class FakeHTTPResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self.payload

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_get(url: str, *, params: dict[str, object], timeout: float) -> FakeHTTPResponse:
        calls.append((url, params))
        if "geocoding" in url:
            return FakeHTTPResponse({"results": [{"name": "Example City", "admin1": "Example Region", "country": "Example Country", "latitude": 1.23, "longitude": 4.56}]})
        return FakeHTTPResponse({
            "daily": {
                "time": ["2026-06-15"],
                "weather_code": [3],
                "temperature_2m_mean": [20.1],
                "temperature_2m_max": [25.0],
                "temperature_2m_min": [15.2],
                "precipitation_sum": [1.4],
                "rain_sum": [1.4],
                "wind_speed_10m_max": [24.0],
            }
        })

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    state.WEATHER_CACHE.clear()
    result = run_get_weather(dataset, {"location": "Example City", "date": "2026-06-15"})
    assert result["source"] == "Open-Meteo historical weather API"
    assert result["data_type"] == "weather"
    assert result["rows"][0]["temperature_mean_c"] == 20.1
    assert len(calls) == 2


def test_soil_moisture_uses_open_meteo_hourly_data(monkeypatch: pytest.MonkeyPatch, dataset: Dataset) -> None:
    class FakeHTTPResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self.payload

    archive_params: dict[str, object] = {}

    def fake_get(url: str, *, params: dict[str, object], timeout: float) -> FakeHTTPResponse:
        if "geocoding" in url:
            return FakeHTTPResponse({"results": [{"name": "Guelph", "admin1": "Ontario", "country": "Canada", "latitude": 43.54, "longitude": -80.25}]})
        archive_params.update(params)
        return FakeHTTPResponse({
            "hourly": {
                "time": ["2026-06-15T00:00", "2026-06-15T01:00"],
                "soil_moisture_0_to_7cm": [0.2, 0.3],
                "soil_moisture_7_to_28cm": [0.3, 0.4],
                "soil_moisture_28_to_100cm": [0.4, 0.5],
                "soil_moisture_100_to_255cm": [0.5, 0.6],
            }
        })

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    state.WEATHER_CACHE.clear()
    result = run_get_weather(dataset, {
        "location": "Guelph",
        "latitude": 0,
        "longitude": 0,
        "date": "2026-06-15",
        "data_type": "soil_moisture",
    })
    assert "soil_moisture_0_to_7cm" in str(archive_params["hourly"])
    assert result["data_type"] == "soil_moisture"
    assert result["units"] == {"soil_moisture": "m³/m³"}
    assert result["rows"] == [{
        "date": "2026-06-15",
        "soil_moisture_0_to_7cm_m3_m3": 0.25,
        "soil_moisture_7_to_28cm_m3_m3": 0.35,
        "soil_moisture_28_to_100cm_m3_m3": 0.45,
        "soil_moisture_100_to_255cm_m3_m3": 0.55,
    }]
    assert "reanalysis" in result["note"]


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
    assert score_reply("The fleet total was 1,012.0 L.", "1012.00 L")[0]
    assert score_reply("Creekside covered 11,036 m.", "Creekside, 11036 m")[0]
    assert not score_reply("MR-01 averaged 57.24%.", "MR-01, 57.13%")[0]
    assert not score_reply("MR-03 had 3 faults and MR-05 had 4 faults.", "MR-03 4 faults; MR-05 3 faults")[0]


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


def test_visualization_prefers_nonzero_result_when_model_explores_states() -> None:
    trace = {
        "steps": [
            {"id": "idle", "type": "tool", "name": "aggregate_runs", "result": {
                "filters": {"state": "idle"}, "group_by": ["robot_id"], "metrics": ["sum_nitrogen"],
                "rows": [{"robot_id": "MR-01", "sum_nitrogen": 0.0}],
            }},
            {"id": "applying", "type": "tool", "name": "aggregate_runs", "result": {
                "filters": {"state": "applying"}, "group_by": ["robot_id"], "metrics": ["sum_nitrogen"],
                "rows": [{"robot_id": "MR-01", "sum_nitrogen": 12.0}],
            }},
        ]
    }
    assert build_visualization(trace)["source_step_id"] == "applying"


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
        self.id = f"resp_{id(self)}"
        self.status = "completed"


class FakeResponsesClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = self
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return next(self._responses)


def test_progress_events_are_emitted_in_execution_order(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([
        FakeResponse([FakeCall("aggregate_runs", '{"state":"applying","metrics":["row_count"]}')]),
        FakeResponse([], "The applying interval count came from the tool result."),
    ])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    events: list[dict[str, object]] = []
    run_chat(dataset, [Message(role="user", content="How many applying intervals are there?")], on_event=events.append)
    event_names = [event["event"] for event in events]
    assert event_names[0] == "turn_start"
    assert event_names[-1] == "turn_complete"
    assert [event["step"]["type"] for event in events if event["event"] == "step"] == ["model", "tool", "model"]
    assert any(event.get("label") == "Running tool · aggregate_runs" for event in events)
    tool_start = next(index for index, event in enumerate(events) if event["event"] == "tool_start")
    tool_step = next(index for index, event in enumerate(events) if event["event"] == "step" and event["step"]["type"] == "tool")
    assert tool_start < tool_step
    assert events[tool_start]["name"] == "aggregate_runs"
    assert "applying" in events[tool_start]["arguments"]


def test_previous_trace_evidence_can_be_reused_without_rerunning_tool(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    first_fake = FakeResponsesClient([
        FakeResponse([FakeCall("aggregate_runs", '{"state":"applying","metrics":["row_count"]}')]),
        FakeResponse([], "There are applying intervals in the dataset."),
    ])
    monkeypatch.setattr(chat, "client", lambda: first_fake)
    monkeypatch.setattr(state, "TRACES", [])
    first = run_chat(dataset, [Message(role="user", content="How many applying intervals are there?")])

    second_fake = FakeResponsesClient([FakeResponse([], "The exact prior tool output is shown above.")])
    monkeypatch.setattr(chat, "client", lambda: second_fake)
    second = run_chat(
        dataset,
        [
            Message(role="user", content="How many applying intervals are there?"),
            Message(role="assistant", content=first["reply"]),
            Message(role="user", content="Show me the exact previous tool output."),
        ],
        previous_trace=first["trace"],
    )
    assert second["trace"]["previous_trace_id"] == first["trace_id"]
    assert second["trace"]["grounding"]["status"] == "reused_evidence"
    assert second["trace"]["reused_evidence"] == ["step_2"]
    assert sum(step["type"] == "tool" for step in second["trace"]["steps"]) == 0
    assert "PRIOR TURN TOOL EVIDENCE" in str(second_fake.calls[0]["instructions"])
    assert second["evidence"]["previous_trace_id"] == first["trace_id"]


def test_tool_error_is_returned_and_turn_recovers(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([
        FakeResponse([FakeCall("query_runs", '{"limit": 51}')]),
        FakeResponse([], "I could not use that limit, so I continued without returning the oversized result."),
    ])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    result = run_chat(dataset, [Message(role="user", content="Show me the runs")])
    assert result["reply"].startswith("I could not use")
    assert result["trace"]["outcome"] == "answered"
    assert result["trace"]["steps"][1]["error"]
    assert result["trace"]["steps"][1]["result"]["error"]


@pytest.mark.parametrize(
    ("call", "error_fragment"),
    [
        (FakeCall("query_runs", "{not-json"), "Expecting property name"),
        (FakeCall("missing_tool", "{}"), "Unknown tool"),
    ],
)
def test_malformed_or_unknown_tool_is_traced_and_recoverable(
    dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    call: FakeCall,
    error_fragment: str,
) -> None:
    fake = FakeResponsesClient([
        FakeResponse([call]),
        FakeResponse([], "I could not use that tool call, so I did not invent an answer."),
    ])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    result = run_chat(dataset, [Message(role="user", content="Show me the runs")])
    tool_step = result["trace"]["steps"][1]
    assert error_fragment in tool_step["error"]
    assert tool_step["result"]["error"] == tool_step["error"]
    if call.arguments == "{not-json":
        assert tool_step["arguments"] == {"raw": "{not-json"}


def test_data_answer_without_tool_evidence_is_blocked(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([], "MR-04 is definitely the best robot.")])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    result = run_chat(dataset, [Message(role="user", content="Which robot is best?")])
    assert result["trace"]["outcome"] == "ungrounded"
    assert result["trace"]["grounding"]["status"] == "missing_tool_evidence"
    assert "won't guess" in result["reply"]


def test_weather_follow_up_records_limitation_grounding(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([
        FakeResponse([], "For June 14, the CSV contains no weather data. Please provide a location."),
    ])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    result = run_chat(dataset, [
        Message(role="user", content="What was the weather on June 15?"),
        Message(role="assistant", content="The CSV contains no weather data."),
        Message(role="user", content="And the day before?"),
    ])
    assert result["trace"]["grounding"] == {
        "required": True,
        "status": "limitation",
        "successful_tool_steps": [],
        "previous_trace_id": None,
        "reused_step_ids": [],
    }


def test_soil_moisture_limitation_adds_location_follow_up(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([
        FakeResponse([], "The CSV doesn't contain soil-moisture measurements for June 15."),
    ])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    result = run_chat(dataset, [Message(role="user", content="What was the soil moisture on June 15?")])
    assert "Open-Meteo" in result["reply"]
    assert "What city, region, or coordinates" in result["reply"]
    assert result["trace"]["grounding"]["status"] == "limitation"
    passed, _ = score_reply(result["reply"], "June 15, soil moisture isn't available in the CSV, Open-Meteo, location")
    assert passed


def test_turn_stops_at_eight_model_calls(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([FakeCall("aggregate_runs", "{}", f"call_{i}")]) for i in range(8)])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    result = run_chat(dataset, [Message(role="user", content="Keep calling tools")])
    assert result["trace"]["outcome"] == "stopped_at_cap"
    assert sum(step["type"] == "model" for step in result["trace"]["steps"]) == 8


def test_separate_turns_have_separate_trace_ids_and_ordered_offsets(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([], "first"), FakeResponse([], "second")])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    first = run_chat(dataset, [Message(role="user", content="first")])
    second = run_chat(dataset, [Message(role="user", content="second")])
    assert first["trace_id"] != second["trace_id"]
    assert first["trace"]["steps"][0]["start_offset_ms"] >= 0
    assert second["trace"]["steps"][0]["start_offset_ms"] >= 0


def test_concurrent_turns_keep_trace_state_separate(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "client", lambda: FakeResponsesClient([FakeResponse([], "Hello.")]))
    monkeypatch.setattr(state, "TRACES", [])
    questions = [f"hello {index}" for index in range(12)]

    def invoke(question: str) -> dict[str, object]:
        return run_chat(dataset, [Message(role="user", content=question)])

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(invoke, questions))
    assert len({result["trace_id"] for result in results}) == len(questions)
    assert {trace["question"] for trace in state.TRACES} == set(questions)


def test_trace_timing_tokens_and_cost_reconcile(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([], "Hello.")])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    trace = run_chat(dataset, [Message(role="user", content="hello")])["trace"]
    assert trace["input_tokens"] == sum(step.get("input_tokens", 0) for step in trace["steps"])
    assert trace["output_tokens"] == sum(step.get("output_tokens", 0) for step in trace["steps"])
    assert trace["cost"] == pytest.approx(sum(step.get("cost", 0) for step in trace["steps"]))
    assert trace["duration_ms"] == pytest.approx(trace["step_duration_ms"] + trace["untraced_duration_ms"], abs=0.02)
    assert 0 <= trace["timing_coverage_pct"] <= 100
    assert trace["messages"] == [{"role": "user", "content": "hello"}]
    assert trace["steps"][0]["response_id"].startswith("resp_")


def test_follow_up_sends_full_conversation_history(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResponsesClient([FakeResponse([], "The dataset cannot answer that.")])
    monkeypatch.setattr(chat, "client", lambda: fake)
    monkeypatch.setattr(state, "TRACES", [])
    messages = [
        Message(role="user", content="What was the weather on June 15?"),
        Message(role="assistant", content="The dataset cannot answer because it contains no weather data."),
        Message(role="user", content="And the day before?"),
    ]
    run_chat(dataset, messages)
    assert fake.calls[0]["input"] == [message.model_dump() for message in messages]


def test_analytics_empty() -> None:
    assert analytics()["turns"] >= 0
