# Fleet Buddy

Fleet Buddy is a small FastAPI application for uploading robot-run CSV data, asking questions through OpenAI Responses API function calling, and inspecting the cost and timing of every turn.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Windows PowerShell
$env:OPENAI_API_KEY = "your-key"
uvicorn main:app --reload
```

Open http://127.0.0.1:8000, upload `robot_runs.csv`, and use the chat and dashboard. The default model is `gpt-4o-mini`; set `OPENAI_MODEL` to override it.

Run the API-key-free tests with:

```bash
pytest -q
```

## Design decisions

The server keeps datasets and traces in process memory, as requested. A dataset is stored as a pandas DataFrame plus a profile. Upload validation requires exactly the seven expected columns, validates timestamps and states, accepts blank numeric cells, and reports those blanks in the profile. Timestamps are normalized to UTC.

The model receives the profile in `instructions`, never the raw rows. It has two deliberately narrow tools:

- `query_runs` filters rows for exact events and timestamps, with a hard maximum of 50 returned rows.
- `aggregate_runs` calculates grouped sums, averages, extrema, and row counts for totals and comparisons.

The five-minute sampling interval is explicitly described to the model so a charging row count can be converted to minutes. Keeping aggregation on the server prevents the full CSV from entering the prompt and makes answers reproducible.

Each chat turn gets a unique trace ID. A trace contains the question, ordered model/tool steps, arguments, returned results, errors, durations, usage, hard-coded cost, final reply, and outcome. Model pricing is intentionally explicit (`$0.15 / 1M` input tokens and `$0.60 / 1M` output tokens) so the analytics are deterministic and easy to replace. Tool errors are returned as structured function output, allowing the model to recover; eight model calls is the turn cap.

The chat response returns the trace ID in both the JSON body (`trace_id`) and the `X-Trace-ID` response header. The full trace is stored in the process-memory `TRACES` list in `main.py`, and can be retrieved with `GET /traces/{trace_id}`. `GET /traces` returns newest-first summaries. This is intentionally not persistent: traces disappear when the process restarts because the assignment requests an in-memory implementation. In a production version, this list would be replaced with a trace store or OpenTelemetry backend.

Evaluation uses the same chat path and creates ordinary traces. The scorer is intentionally local rather than another model call: it requires all numeric/robot identifier facts and a threshold of expected content words. This avoids contaminating analytics with judge calls and makes pass/fail reproducible. It is a lightweight smoke evaluator, not a substitute for human review.

The frontend is a single static page served by FastAPI. It includes upload/profile, chat, a live trace list, selectable step waterfall, analytics cards, two canvas charts, and the evaluation runner. Traces are refreshed after each turn without a page reload.

## API overview

- `POST /datasets` — multipart CSV upload and profile.
- `POST /datasets/{id}/chat` — full conversation in, reply and trace ID out.
- `GET /traces` and `GET /traces/{id}` — trace summaries/full traces.
- `GET /analytics` — aggregate timing, token, cost, outcome, and tool metrics.
- `GET /evals` — bundled `evals.json` cases.
- `POST /datasets/{id}/evals` — evaluate a supplied list of cases.

## Next steps

For production I would add persistent storage, authentication, request size/rate limits, dataset expiry, structured logging, richer evaluator assertions, and a background job for large evaluation batches. I would also make model pricing a versioned configuration instead of a source constant.
