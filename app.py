"""
app.py
======
PURPOSE: Flask web server for NBA Stats Explorer.

ROUTES:
  GET  /                  — serves the single-page app
  POST /api/query         — submit a new stat query → returns job_id
  GET  /api/stream/<id>   — SSE stream of agent progress + final result
  GET  /api/status/<id>   — poll job status (JSON)
  GET  /api/health        — health check

SSE EVENT TYPES (consumed by frontend JS):
  thinking    — agent step description
  tool_start  — tool being called
  tool_result — tool returned data
  message     — Claude's partial text commentary
  result      — final structured JSON data
  done        — stream complete
  error       — error message
"""

import atexit
import json
import logging
import os
import threading
import uuid
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT
from agent import NBAStatsAgent

# ── Gemini 2.5 Flash pricing (USD per 1M tokens, as of 2025) ─────────────────
PRICE_INPUT_PER_M  = 0.075   # $0.075 / 1M input tokens
PRICE_OUTPUT_PER_M = 0.30    # $0.30  / 1M output tokens

# ── Session usage store ───────────────────────────────────────────────────────
SESSION_START    = datetime.utcnow()
SESSION_START_ISO = SESSION_START.isoformat()

# usage_data/ folder — one JSON file per server session
USAGE_DIR = "usage_data"
os.makedirs(USAGE_DIR, exist_ok=True)

# File name: usage_data/session_2025-03-21T14-30-00.json  (colons → hyphens)
_safe_ts    = SESSION_START.strftime("%Y-%m-%dT%H-%M-%S")
SESSION_FILE = os.path.join(USAGE_DIR, f"session_{_safe_ts}.json")

# In-memory query map: { query_text: { ...cost fields... } }
QUERY_MAP:  dict = {}
USAGE_LOCK       = threading.Lock()


def _compute_cost(input_tok: int, output_tok: int) -> tuple[float, float]:
    """Return (input_cost, output_cost) in USD."""
    return (
        input_tok  / 1_000_000 * PRICE_INPUT_PER_M,
        output_tok / 1_000_000 * PRICE_OUTPUT_PER_M,
    )


def _write_json():
    """Flush the current session data to usage_data/<session>.json."""
    with USAGE_LOCK:
        queries   = dict(QUERY_MAP)

    now          = datetime.utcnow()
    elapsed      = round((now - SESSION_START).total_seconds(), 2)
    total_calls  = sum(q["api_calls"]     for q in queries.values())
    total_cost   = sum(q["query_cost_usd"] for q in queries.values())

    payload = {
        "session_start":         SESSION_START_ISO,
        "session_end":           now.isoformat(),
        "total_duration_seconds": elapsed,
        "total_api_calls":       total_calls,
        "estimated_cost_usd":    round(total_cost, 8),
        "model":                 "gemini-2.5-flash",
        "pricing": {
            "input_per_1m_tokens":  PRICE_INPUT_PER_M,
            "output_per_1m_tokens": PRICE_OUTPUT_PER_M,
        },
        "queries": queries,
    }

    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            f"Usage report → {SESSION_FILE} "
            f"({len(queries)} queries, ${total_cost:.6f} total, {elapsed}s elapsed)"
        )
    except Exception as e:
        logger.warning(f"Could not write usage JSON: {e}")


def _record_usage(query: str, usage: dict):
    """Record one query's usage in the in-memory map and flush to JSON."""
    input_tok  = usage.get("input_tokens",  0)
    output_tok = usage.get("output_tokens", 0)
    api_calls  = usage.get("api_calls",     0)
    in_cost, out_cost = _compute_cost(input_tok, output_tok)

    entry = {
        "timestamp":       datetime.utcnow().isoformat(),
        "api_calls":       api_calls,
        "input_tokens":    input_tok,
        "output_tokens":   output_tok,
        "input_cost_usd":  round(in_cost,             8),
        "output_cost_usd": round(out_cost,            8),
        "query_cost_usd":  round(in_cost + out_cost,  8),
    }

    with USAGE_LOCK:
        # If the same query is run multiple times, accumulate rather than overwrite
        if query in QUERY_MAP:
            prev = QUERY_MAP[query]
            entry["api_calls"]       += prev["api_calls"]
            entry["input_tokens"]    += prev["input_tokens"]
            entry["output_tokens"]   += prev["output_tokens"]
            entry["input_cost_usd"]   = round(entry["input_cost_usd"]  + prev["input_cost_usd"],  8)
            entry["output_cost_usd"]  = round(entry["output_cost_usd"] + prev["output_cost_usd"], 8)
            entry["query_cost_usd"]   = round(entry["query_cost_usd"]  + prev["query_cost_usd"],  8)
        QUERY_MAP[query] = entry

    _write_json()


atexit.register(_write_json)  # final flush when Python process exits

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── In-memory job store ───────────────────────────────────────────────────────
# Maps job_id → {"status", "events": [...], "result": {...}, "lock": Lock}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


def _make_job(query: str) -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "query":    query,
        "status":   "running",
        "events":   [],
        "result":   None,
        "error":    None,
        "created":  datetime.utcnow().isoformat(),
        "lock":     threading.Lock(),
    }
    return job_id


def _push_event(job_id: str, event_type: str, data: str):
    job = JOBS.get(job_id)
    if job:
        with job["lock"]:
            job["events"].append({"type": event_type, "data": data})


def _run_agent(job_id: str, query: str):
    """Background thread: runs the agent and pushes SSE events."""
    def cb(event_type: str, message: str):
        _push_event(job_id, event_type, message)

    try:
        agent  = NBAStatsAgent()
        output = agent.run(query, progress_cb=cb)

        # Record token usage / cost for this query
        _record_usage(query, output.get("usage") or {})

        job = JOBS[job_id]
        with job["lock"]:
            if output["success"]:
                job["result"] = output
                job["status"] = "done"
                job["events"].append({
                    "type": "result",
                    "data": json.dumps(output["result"]),
                })
                job["events"].append({
                    "type": "narrative",
                    "data": output.get("narrative", ""),
                })
            else:
                job["status"] = "error"
                job["error"]  = output.get("error", "Unknown error")
                job["events"].append({
                    "type": "error",
                    "data": job["error"],
                })
            job["events"].append({"type": "done", "data": "complete"})

    except Exception as e:
        logger.exception(f"Agent thread error for job {job_id}: {e}")
        job = JOBS.get(job_id)
        if job:
            with job["lock"]:
                job["status"] = "error"
                job["error"]  = str(e)
                job["events"].append({"type": "error",  "data": str(e)})
                job["events"].append({"type": "done",   "data": "complete"})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def submit_query():
    body  = request.get_json(force=True, silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    job_id = _make_job(query)
    logger.info(f"New job {job_id[:8]}: {query}")

    t = threading.Thread(target=_run_agent, args=(job_id, query),
                         name=f"agent-{job_id[:8]}", daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404

    @stream_with_context
    def generate():
        sent = 0
        import time
        while True:
            job = JOBS.get(job_id)
            if not job:
                break
            with job["lock"]:
                events = job["events"]
                new_events = events[sent:]
                sent = len(events)
                status = job["status"]

            for ev in new_events:
                payload = json.dumps({"type": ev["type"], "data": ev["data"]})
                yield f"data: {payload}\n\n"

            if status in ("done", "error") and sent >= len(JOBS[job_id]["events"]):
                break

            time.sleep(0.15)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/status/<job_id>")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status":  job["status"],
        "query":   job["query"],
        "created": job["created"],
        "error":   job.get("error"),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "jobs": len(JOBS)})


@app.route("/api/usage_report")
def usage_report():
    """Return the current session's usage JSON as a file download."""
    _write_json()          # ensure latest data is flushed
    if not os.path.exists(SESSION_FILE):
        return jsonify({"error": "No usage data yet."}), 404
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    filename = os.path.basename(SESSION_FILE)
    return Response(
        content,
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/api/usage_summary")
def usage_summary():
    """Return a live JSON summary of session usage and cost."""
    with USAGE_LOCK:
        queries = dict(QUERY_MAP)

    now         = datetime.utcnow()
    elapsed     = round((now - SESSION_START).total_seconds(), 2)
    total_calls = sum(q["api_calls"]      for q in queries.values())
    total_in    = sum(q["input_tokens"]   for q in queries.values())
    total_out   = sum(q["output_tokens"]  for q in queries.values())
    total_cost  = sum(q["query_cost_usd"] for q in queries.values())

    return jsonify({
        "session_start":          SESSION_START_ISO,
        "total_duration_seconds": elapsed,
        "total_api_calls":        total_calls,
        "total_input_tokens":     total_in,
        "total_output_tokens":    total_out,
        "estimated_cost_usd":     round(total_cost, 6),
        "model":                  "gemini-2.5-flash",
        "session_file":           SESSION_FILE,
        "pricing": {
            "input_per_1m_tokens":  PRICE_INPUT_PER_M,
            "output_per_1m_tokens": PRICE_OUTPUT_PER_M,
        },
        "queries": queries,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"NBA Stats Explorer → http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
