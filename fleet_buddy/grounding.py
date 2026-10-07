"""Checks that data answers are backed by tool results, plus follow-up helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import Message


# Word lists for the grounding check: if the user asks about fleet data, the answer must come
# from a tool result or clearly say the data can't answer it.
DATA_QUESTION_TERMS = {
    "robot", "fleet", "nitrogen", "battery", "distance", "fault", "charging",
    "driving", "applying", "idle", "efficiency", "field", "interval", "run",
    "weather", "temperature", "rain", "precipitation", "soil", "moisture",
}


LIMITATION_PHRASES = (
    "cannot answer", "can't answer", "could not answer", "not available", "isn't available",
    "not included", "contains no", "need a location", "provide a location", "what city",
    "what location", "coordinates should i use", "cannot determine", "can't determine",
    "could not use", "doesn't contain", "does not contain", "doesn't include",
    "does not include", "isn't included", "is not included", "not stored",
)


ENVIRONMENT_QUERY_TERMS = ("weather", "soil moisture", "temperature", "rain", "precipitation")


LOCATION_REQUEST_RE = re.compile(
    r"\b(what|which|provide|share|specify|tell me|need)\b[^.?!]{0,40}\b(city|location|region|coordinates)\b"
)


def grounding_required(messages: list[Message]) -> bool:
    user_text = " ".join(message.content.lower() for message in messages if message.role == "user")
    return any(re.search(rf"\b{re.escape(term)}s?\b", user_text) for term in DATA_QUESTION_TERMS)


def reply_states_limitation(reply: str) -> bool:
    lowered = reply.lower().replace("’", "'")
    return any(phrase in lowered for phrase in LIMITATION_PHRASES)


def ensure_environment_follow_up(messages: list[Message], reply: str, steps: list[dict[str, Any]]) -> str:
    # If the user asked about weather and no tool ran, make sure the reply says the CSV can't
    # answer it and asks for a location for the external lookup.
    user_text = " ".join(message.content.lower() for message in messages if message.role == "user")
    environmental = any(term in user_text for term in ENVIRONMENT_QUERY_TERMS)
    any_tool_succeeded = any(step.get("type") == "tool" and not step.get("error") for step in steps)
    if not environmental or any_tool_succeeded:
        return reply
    subject = "soil moisture" if "soil moisture" in user_text else "weather"
    if not reply_states_limitation(reply):
        reply = f"The uploaded CSV doesn't contain {subject} data, so it can't answer this directly. " + reply.lstrip()
    if not LOCATION_REQUEST_RE.search(reply.lower()):
        reply = (
            reply.rstrip()
            + f"\n\nI can look up external historical {subject} from Open-Meteo. "
            + "What city, region, or coordinates should I use?"
        )
    return reply


def previous_trace_context(previous_trace: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    # Gives the model the previous turn's tool results so follow-ups like "show your work" can reuse them.
    if not previous_trace:
        return "", []
    tool_steps = [
        {
            "step_id": step.get("id"),
            "tool": step.get("name"),
            "arguments": step.get("arguments"),
            "result": step.get("result"),
        }
        for step in previous_trace.get("steps", [])
        if step.get("type") == "tool" and not step.get("error")
    ]
    if not tool_steps:
        return "", []
    context = (
        "\n\nPRIOR TURN TOOL EVIDENCE:\n"
        + json.dumps({
            "trace_id": previous_trace.get("id"),
            "question": previous_trace.get("question"),
            "reply": previous_trace.get("reply"),
            "tool_steps": tool_steps,
        }, indent=2)
        + "\nUse this evidence only when the user asks to inspect, explain, or reuse the prior result. "
        + "For a new scope, date, entity, filter, or metric, call a tool again."
    )
    return context, tool_steps


def question_requests_evidence_reuse(question: str) -> bool:
    lowered = question.lower()
    phrases = (
        "previous tool", "prior tool", "exact tool", "tool output", "tool result",
        "show your work", "show the evidence", "support that", "how did you calculate",
        "how was that calculated", "where did that come from", "same result",
    )
    return any(phrase in lowered for phrase in phrases)
