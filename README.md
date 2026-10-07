# Fleet Buddy

Fleet Buddy is a small FastAPI application for uploading robot-run CSV data, asking questions through OpenAI Responses API function calling, and inspecting the cost and timing of every turn.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Create a local environment file (Windows PowerShell)
Copy-Item .env.example .env
# Then replace the placeholder in .env with your OpenAI API key.
uvicorn main:app --reload --env-file .env
```

Open http://127.0.0.1:8000, upload `robot_runs.csv`, and use the chat and dashboard. The default model is `gpt-4o-mini`; set `OPENAI_MODEL` in `.env` to override it. The real `.env` is ignored by Git and must not be committed.

Run the API-key-free tests with:

```bash
pytest -q
```

## Project layout

`main.py` is only the entry point (`uvicorn main:app`). The code lives in `fleet_buddy/`:

| File | What it does |
|---|---|
| `api.py` | HTTP endpoints and the NDJSON streaming helper |
| `chat.py` | The model → tool → model loop; builds and stores each trace |
| `tools.py` | `query_runs`, `aggregate_runs`, the tool schemas sent to the model, and `call_tool` |
| `weather.py` | `get_weather` (Open-Meteo geocoding + archive) |
| `data.py` | CSV validation, dataset profile, shared filters |
| `prompt.py` | System prompt |
| `grounding.py` | Grounding check and follow-up helpers |
| `fallback.py` | Computed answers for a few known question shapes |
| `evidence.py` | Chart, evidence summary, and LaTeX clean-up |
| `evals.py` | Local eval scorer |
| `analytics.py` | Dashboard aggregates over traces |
| `config.py`, `schemas.py`, `state.py`, `utils.py` | Settings, request models, in-memory stores, small helpers |

## Design decisions

The server keeps datasets and traces in process memory, as requested. A dataset is stored as a pandas DataFrame plus a profile. Upload validation requires exactly the seven expected columns, validates timestamps and states, accepts blank numeric cells, and reports those blanks in the profile. Timestamps are normalized to UTC.

The model receives the profile in `instructions`, never the raw rows. It has three deliberately narrow tools:

- `query_runs` filters rows for exact events and timestamps, with a hard maximum of 50 returned rows.
- `aggregate_runs` calculates grouped sums, averages, extrema, and row counts for totals and comparisons.
- `get_weather` optionally geocodes a user-supplied location and retrieves historical weather or soil-moisture reanalysis from Open-Meteo. It never infers coordinates from ambiguous CSV field names; without a location it asks the user for one.

Both robot-run tools support server-side `exclude_states`, `exclude_robot_ids`, and `exclude_fields` filters. Requests such as “everything except idle” therefore use one auditable aggregate instead of subtracting model-generated values.

The five-minute sampling interval is explicitly described to the model so a charging row count can be converted to minutes. Keeping aggregation on the server prevents the full CSV from entering the prompt and makes answers reproducible.

Environmental data is deliberately marked as external evidence. The uploaded CSV remains the source of truth for robot metrics, while Open-Meteo historical weather and soil-moisture reanalysis are used only when the user supplies a location or coordinates. Evidence includes the resolved location, date range, units, and source in the trace; soil moisture is described as modeled reanalysis rather than an on-farm sensor reading.

Each chat turn gets a unique trace ID. A trace contains the complete conversation, prompt version, model configuration, ordered model/tool steps, response IDs, arguments, returned results, errors, durations, usage, hard-coded cost, final reply, and outcome. It also records traced-step time, orchestration time outside those steps, and timing coverage so the waterfall can explain the entire turn. Model pricing is intentionally explicit (`$0.15 / 1M` input tokens and `$0.60 / 1M` output tokens) so the analytics are deterministic and easy to replace. Tool errors are returned as structured function output, allowing the model to recover; eight model calls is the turn cap.

The chat response returns the trace ID in both the JSON body (`trace_id`) and the `X-Trace-ID` response header. The optional `previous_trace_id` request field makes the preceding structured tool evidence available to a follow-up without creating server-side chat sessions. Reuse is limited to explicit evidence questions; a new date, entity, filter, or metric still requires a new tool call. The full trace is stored in the process-memory `TRACES` list in `fleet_buddy/state.py`, and can be retrieved with `GET /traces/{trace_id}`. `GET /traces` returns newest-first summaries. This is intentionally not persistent: traces disappear when the process restarts because the assignment requests an in-memory implementation. In a production version, this list would be replaced with a trace store or OpenTelemetry backend.

Evaluation uses the same chat path and creates ordinary traces. The scorer is intentionally local rather than another model call: it requires all numeric/robot identifier facts, keeps robot IDs paired with their expected values, and applies a threshold to expected content words. This avoids contaminating analytics with judge calls and makes pass/fail reproducible. It is a lightweight smoke evaluator, not a substitute for human review.

For questions that require fleet data, the server enforces a grounding postcondition after the model loop. A turn only succeeds if a tool returned usable evidence or the reply clearly states that the available data cannot answer the question. An unsupported fleet answer is replaced with a refusal to guess. The trace and UI expose this grounding status, along with whether a deterministic fallback was used.

The chat loop also has a capped, data-driven fallback for a few high-value comparison shapes (lowest average battery, named-robot efficiency comparison, and lowest fleet-efficiency date). If the model spends its call budget exploring redundant filters, the fallback runs the corresponding aggregate against the uploaded dataset rather than inventing values; the fallback is recorded as a trace tool step and drives the visualization.

The frontend is a single static page served by FastAPI. It includes upload/profile, chat, a live trace list, selectable step waterfall, analytics cards, two canvas charts, and the evaluation runner. Chat and evaluation requests use NDJSON streaming endpoints, so model passes, tool calls (with their arguments, shown as soon as each tool starts), errors, durations, and scoring appear as they happen, with an elapsed-time counter during long model calls, rather than behind a static “thinking” state. Traces are refreshed after each completed turn without a page reload.

Each successful data-backed chat answer also includes an evidence disclosure in the UI. It shows the tools used, the effective timeframe, the filters passed to those tools, the number of tool-result rows, and whether a result was capped. The adjacent bar chart is generated from structured tool output, not from text invented by the model. The response exposes both `trace_id` and `X-Trace-ID` so a caller can correlate the answer with the full trace.

The model is prompted to use one grouped aggregation for fleet-wide rankings and trends. This keeps latency and cost lower than making one query per robot while still allowing separate calls for explicitly named comparisons.

## API overview

- `GET /health` — process health and in-memory dataset/trace counts.
- `POST /datasets` — multipart CSV upload and profile.
- `POST /datasets/{id}/chat` — full conversation in, reply and trace ID out.
- `POST /datasets/{id}/chat/stream` — the same chat path with live NDJSON progress events.
- `GET /traces` and `GET /traces/{id}` — trace summaries/full traces.
- `GET /analytics` — aggregate timing, token, cost, outcome, grounding, fallback, and tool metrics.
- `GET /evals` — bundled `evals.json` cases.
- `POST /datasets/{id}/evals` — evaluate a supplied list of cases.
- `POST /datasets/{id}/evals/stream` — evaluate cases with live case/model/tool/scoring events.

The custom tracing here is intentional. The assignment explicitly prohibits OpenTelemetry and third-party tracing frameworks, so this project records the trace lifecycle directly and keeps it in memory.

## Next steps

For production I would add persistent storage, authentication, request size/rate limits, dataset expiry, structured logging, richer evaluator assertions, and a background job for large evaluation batches. I would also make model pricing a versioned configuration instead of a source constant.
