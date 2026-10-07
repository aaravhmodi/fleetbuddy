"""HTTP endpoints. Each one is thin: look things up, call the right module, return JSON."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Iterator

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import state
from .analytics import analytics, summarize_trace
from .chat import ProgressCallback, run_chat
from .data import parse_dataset
from .evals import evaluation_result
from .schemas import ChatRequest, EvalCase
from .state import Dataset
from .utils import json_safe

# Project root (one level above this package): holds static/ and evals.json.
ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Fleet Buddy", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def resolve_previous_trace(dataset_id: str, trace_id: str | None) -> dict[str, Any] | None:
    if not trace_id:
        return None
    with state.STATE_LOCK:
        trace = next((item for item in state.TRACES if item["id"] == trace_id), None)
    if trace is None:
        raise HTTPException(status_code=404, detail="Previous trace not found.")
    if trace["dataset_id"] != dataset_id:
        raise HTTPException(status_code=400, detail="Previous trace belongs to a different dataset.")
    return trace


def stream_worker(worker: Callable[[ProgressCallback], Any]) -> Iterator[str]:
    # Runs the work in a background thread and streams its progress to the browser, one JSON line per event.
    events: Queue[Any] = Queue()
    finished = object()

    def emit(event: dict[str, Any]) -> None:
        events.put(event)

    def target() -> None:
        try:
            result = worker(emit)
            events.put({"event": "done", "result": json_safe(result)})
        except Exception as exc:
            events.put({"event": "error", "message": str(exc)})
        finally:
            events.put(finished)

    threading.Thread(target=target, daemon=True).start()
    while True:
        event = events.get()
        if event is finished:
            break
        yield json.dumps(json_safe(event), separators=(",", ":")) + "\n"


# --- HTTP endpoints ---
@app.get("/")
def index() -> FileResponse:
    # no-store so the browser always loads the latest UI instead of a cached copy.
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict[str, Any]:
    with state.STATE_LOCK:
        return {"status": "ok", "datasets": len(state.DATASETS), "traces": len(state.TRACES)}


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    try:
        frame, profile = parse_dataset(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    with state.STATE_LOCK:
        state.DATASETS[dataset_id] = Dataset(dataset_id, frame, profile)
    return {"dataset_id": dataset_id, "profile": profile}


@app.post("/datasets/{dataset_id}/chat")
def chat(dataset_id: str, request: ChatRequest, response: Response) -> dict[str, Any]:
    with state.STATE_LOCK:
        dataset = state.DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    previous_trace = resolve_previous_trace(dataset_id, request.previous_trace_id)
    result = run_chat(dataset, request.messages, previous_trace=previous_trace)
    response.headers["X-Trace-ID"] = result["trace_id"]
    return {"reply": result["reply"], "trace_id": result["trace_id"], "visualization": result["visualization"], "evidence": result["evidence"]}


@app.post("/datasets/{dataset_id}/chat/stream")
def stream_chat(dataset_id: str, request: ChatRequest) -> StreamingResponse:
    with state.STATE_LOCK:
        dataset = state.DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    previous_trace = resolve_previous_trace(dataset_id, request.previous_trace_id)
    return StreamingResponse(
        stream_worker(lambda emit: run_chat(dataset, request.messages, previous_trace=previous_trace, on_event=emit)),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/traces")
def list_traces() -> list[dict[str, Any]]:
    with state.STATE_LOCK:
        traces = list(state.TRACES)
    return [summarize_trace(trace) for trace in reversed(traces)]


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    with state.STATE_LOCK:
        trace = next((trace for trace in state.TRACES if trace["id"] == trace_id), None)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


@app.get("/analytics")
def get_analytics() -> dict[str, Any]:
    return analytics()


@app.get("/evals")
def get_evals() -> list[dict[str, Any]]:
    path = ROOT / "evals.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/datasets/{dataset_id}/evals")
def run_evals(dataset_id: str, cases: list[EvalCase]) -> dict[str, Any]:
    with state.STATE_LOCK:
        dataset = state.DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    results = []
    for index, case in enumerate(cases):
        run = run_chat(dataset, case.messages)
        results.append(evaluation_result(index, case, run))
    passed_count = sum(result["passed"] for result in results)
    return {"pass_rate": round(passed_count / len(results), 4) if results else 0.0, "passed": passed_count, "total": len(results), "results": results}


@app.post("/datasets/{dataset_id}/evals/stream")
def stream_evals(dataset_id: str, cases: list[EvalCase]) -> StreamingResponse:
    with state.STATE_LOCK:
        dataset = state.DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    def worker(emit: ProgressCallback) -> dict[str, Any]:
        results = []
        for index, case in enumerate(cases):
            question = next((message.content for message in reversed(case.messages) if message.role == "user"), "")
            emit({"event": "case_start", "case_index": index, "total": len(cases), "question": question})

            def case_event(event: dict[str, Any]) -> None:
                emit({**event, "case_index": index, "total": len(cases)})

            run = run_chat(dataset, case.messages, on_event=case_event)
            result = evaluation_result(index, case, run, include_trace=True)
            results.append(result)
            emit({"event": "case_result", "case_index": index, "total": len(cases), "result": result})
        passed_count = sum(result["passed"] for result in results)
        return {
            "pass_rate": round(passed_count / len(results), 4) if results else 0.0,
            "passed": passed_count,
            "total": len(results),
            "results": results,
        }

    return StreamingResponse(
        stream_worker(worker),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
