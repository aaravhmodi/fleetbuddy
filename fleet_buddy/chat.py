"""The chat loop: model call -> tools -> model call ... with every step recorded in a trace."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Callable

from openai import OpenAI

from . import state
from .config import INPUT_PRICE_PER_TOKEN, MAX_MODEL_CALLS, MODEL, OUTPUT_PRICE_PER_TOKEN, PROMPT_VERSION
from .evidence import build_evidence, build_visualization, normalize_latex_response
from .fallback import fallback_aggregate_answer
from .grounding import (
    ensure_environment_follow_up,
    grounding_required,
    previous_trace_context,
    question_requests_evidence_reuse,
    reply_states_limitation,
    unrequested_state_error,
)
from .prompt import SYSTEM_PROMPT_TEMPLATE
from .schemas import Message
from .state import Dataset
from .tools import TOOLS, call_tool
from .utils import json_safe, model_dump, now_iso


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


ProgressCallback = Callable[[dict[str, Any]], None]


def run_chat(
    dataset: Dataset,
    messages: list[Message],
    previous_trace: dict[str, Any] | None = None,
    on_event: ProgressCallback | None = None,
) -> dict[str, Any]:
    # The main chat loop: ask the model -> run any tools it asks for -> send results back -> repeat
    # until it gives a final answer (at most MAX_MODEL_CALLS times). Every step goes into the trace.
    def emit(event: str, **payload: Any) -> None:
        if on_event is not None:
            on_event(json_safe({"event": event, **payload}))

    prior_context, prior_tool_steps = previous_trace_context(previous_trace)
    trace = {
        "id": f"tr_{uuid.uuid4().hex[:12]}",
        "dataset_id": dataset.id,
        "created_at": now_iso(),
        "question": next((message.content for message in reversed(messages) if message.role == "user"), ""),
        "messages": [message.model_dump() for message in messages],
        "previous_trace_id": previous_trace.get("id") if previous_trace else None,
        "prior_evidence_step_ids": [step["step_id"] for step in prior_tool_steps],
        "reused_evidence": [],
        "steps": [],
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "model_config": {
            "max_model_calls": MAX_MODEL_CALLS,
            "tool_names": [tool["name"] for tool in TOOLS],
            "input_price_per_token": INPUT_PRICE_PER_TOKEN,
            "output_price_per_token": OUTPUT_PRICE_PER_TOKEN,
        },
        "outcome": "answered",
        "reply": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "duration_ms": 0.0,
        "visualization": None,
        "evidence": None,
        "fallback_used": False,
        "grounding": None,
    }
    started = time.perf_counter()
    prompt = SYSTEM_PROMPT_TEMPLATE.format(profile=json.dumps(dataset.profile, indent=2)) + prior_context
    input_items: list[dict[str, Any]] = [message.model_dump() for message in messages]
    emit("turn_start", trace_id=trace["id"], question=trace["question"], previous_trace_id=trace["previous_trace_id"])
    try:
        api_client = client()
        reply = ""
        for call_number in range(1, MAX_MODEL_CALLS + 1):
            emit("status", trace_id=trace["id"], label=f"Calling model · pass {call_number}")
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
                "response_id": getattr(response, "id", None),
                "response_status": getattr(response, "status", None),
                "function_calls": [{"name": c.get("name"), "call_id": c.get("call_id")} for c in calls],
                "output_text": text_output,
            }
            trace["steps"].append(model_step)
            emit("step", trace_id=trace["id"], step=model_step)
            # No tool requests means the model has given its final answer.
            if not calls:
                reply = normalize_latex_response(text_output.strip())
                break
            if call_number >= MAX_MODEL_CALLS:
                trace["outcome"] = "stopped_at_cap"
                reply = "I stopped after reaching the model-call limit before I could finish the answer."
                break
            # Run each requested tool and add its result to the conversation for the next model call.
            input_items.extend(output_items(response))
            for call in calls:
                tool_started = time.perf_counter()
                tool_started_at = now_iso()
                name = str(call.get("name", ""))
                raw_args = call.get("arguments", "{}")
                args: dict[str, Any] = {"raw": raw_args}
                emit("status", trace_id=trace["id"], label=f"Running tool · {name}")
                # Tell the UI which tool is starting (and with what arguments) before it finishes.
                emit("tool_start", trace_id=trace["id"], name=name, call_id=call.get("call_id"), arguments=raw_args)
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(parsed_args, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    args = parsed_args
                    state_error = unrequested_state_error(args, messages)
                    if state_error:
                        raise ValueError(state_error)
                    result = call_tool(dataset, name, args)
                    tool_error = None
                except Exception as exc:  # Tool errors are intentionally returned to the model.
                    result = {"error": str(exc)}
                    tool_error = str(exc)
                tool_step = {
                    "id": f"step_{len(trace['steps']) + 1}",
                    "type": "tool",
                    "name": name,
                    "started_at": tool_started_at,
                    "start_offset_ms": round((tool_started - started) * 1000, 2),
                    "duration_ms": round((time.perf_counter() - tool_started) * 1000, 2),
                    "arguments": json_safe(args),
                    "result": json_safe(result),
                    "error": tool_error,
                }
                trace["steps"].append(tool_step)
                emit("step", trace_id=trace["id"], step=tool_step)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": json.dumps(json_safe(result)),
                })
        else:
            trace["outcome"] = "stopped_at_cap"
            reply = "I stopped after reaching the model-call limit before I could finish the answer."
        # Only rescue a turn that ran out of model calls; the outcome stays stopped_at_cap.
        fallback = fallback_aggregate_answer(dataset, trace["question"]) if trace["outcome"] == "stopped_at_cap" else None
        if fallback:
            fallback_reply, fallback_args, fallback_result = fallback
            fallback_step = {
                "id": f"step_{len(trace['steps']) + 1}",
                "type": "tool",
                "name": "aggregate_runs",
                "started_at": now_iso(),
                "start_offset_ms": round((time.perf_counter() - started) * 1000, 2),
                "duration_ms": 0.0,
                "arguments": json_safe(fallback_args),
                "result": json_safe(fallback_result),
                "error": None,
                "fallback": True,
            }
            trace["steps"].append(fallback_step)
            emit("step", trace_id=trace["id"], step=fallback_step)
            trace["fallback_used"] = True
            reply = fallback_reply
        trace["reply"] = reply or "I could not produce an answer from the available data."
        successful_tools = [
            step for step in trace["steps"]
            if step.get("type") == "tool" and not step.get("error")
        ]
        trace["reply"] = ensure_environment_follow_up(messages, trace["reply"], trace["steps"])
        # Grounding check: a data answer must be backed by a tool result, reused evidence,
        # or an honest "can't answer". Otherwise replace it instead of letting the model guess.
        required = grounding_required(messages)
        limitation = reply_states_limitation(trace["reply"])
        reuse_requested = bool(prior_tool_steps) and question_requests_evidence_reuse(trace["question"])
        if successful_tools:
            grounding_status = "grounded"
        elif reuse_requested:
            grounding_status = "reused_evidence"
            trace["reused_evidence"] = list(trace["prior_evidence_step_ids"])
        elif required and limitation:
            grounding_status = "limitation"
        elif required:
            grounding_status = "missing_tool_evidence"
            if trace["outcome"] == "answered":  # keep stopped_at_cap visible in traces and analytics
                trace["outcome"] = "ungrounded"
            trace["reply"] = "I couldn't verify that answer from tool results, so I won't guess."
        else:
            grounding_status = "not_required"
        trace["grounding"] = {
            "required": required,
            "status": grounding_status,
            "successful_tool_steps": [step["id"] for step in successful_tools],
            "previous_trace_id": trace["previous_trace_id"] if reuse_requested else None,
            "reused_step_ids": trace["reused_evidence"],
        }
    except Exception as exc:
        trace["outcome"] = "failed"
        trace["error"] = str(exc)
        trace["reply"] = "I couldn't complete that turn. Please check the server configuration and try again."
        trace["grounding"] = {
            "required": grounding_required(messages),
            "status": "failed",
            "successful_tool_steps": [],
        }
    if (trace.get("grounding") or {}).get("status") == "reused_evidence" and previous_trace:
        trace["visualization"] = previous_trace.get("visualization")
        prior_evidence = dict(previous_trace.get("evidence") or {})
        prior_evidence.update({
            "grounding": trace["grounding"],
            "previous_trace_id": trace["previous_trace_id"],
            "reused_evidence": True,
        })
        trace["evidence"] = prior_evidence
    else:
        trace["visualization"] = build_visualization(trace)
        trace["evidence"] = build_evidence(trace)
    # Final timing numbers, then save the trace so /traces and /analytics can see it.
    trace["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    step_duration_ms = round(sum(float(step.get("duration_ms", 0)) for step in trace["steps"]), 2)
    trace["step_duration_ms"] = step_duration_ms
    trace["untraced_duration_ms"] = round(max(0.0, trace["duration_ms"] - step_duration_ms), 2)
    trace["timing_coverage_pct"] = round(
        min(100.0, step_duration_ms / trace["duration_ms"] * 100), 2
    ) if trace["duration_ms"] else 100.0
    trace["cost"] = round(trace["cost"], 8)
    with state.STATE_LOCK:
        state.TRACES.append(trace)
    result = {"reply": trace["reply"], "trace_id": trace["id"], "visualization": trace["visualization"], "evidence": trace["evidence"], "trace": trace}
    emit("turn_complete", trace_id=trace["id"], outcome=trace["outcome"], duration_ms=trace["duration_ms"])
    return result
