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
import csv
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
SESSION_START   = datetime.utcnow().isoformat()
USAGE_LOG: list = []          # list of per-query dicts
USAGE_LOCK      = threading.Lock()
REPORT_PATH     = "usage_report.csv"

CSV_FIELDS = [
    "timestamp", "query", "api_calls",
    "input_tokens", "output_tokens",
    "input_cost_usd", "output_cost_usd", "query_cost_usd",
    "session_total_cost_usd",
]


def _compute_cost(input_tok: int, output_tok: int) -> tuple[float, float]:
    """Return (input_cost, output_cost) in USD."""
    return (
        input_tok  / 1_000_000 * PRICE_INPUT_PER_M,
        output_tok / 1_000_000 * PRICE_OUTPUT_PER_M,
    )


def _write_csv():
    """Write the full session usage log to CSV (called after every query + on exit)."""
    with USAGE_LOCK:
        rows = list(USAGE_LOG)

    running_total = 0.0
    try:
        with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                running_total += row["query_cost_usd"]
                writer.writerow({**row, "session_total_cost_usd": round(running_total, 8)})
        logger.info(f"Usage report written → {REPORT_PATH} ({len(rows)} queries, ${running_total:.6f} total)")
    except Exception as e:
        logger.warning(f"Could not write usage report: {e}")


def _record_usage(query: str, usage: dict):
    """Append one query's usage to the in-memory log and flush to CSV."""
    input_tok  = usage.get("input_tokens",  0)
    output_tok = usage.get("output_tokens", 0)
    api_calls  = usage.get("api_calls",     0)
    in_cost, out_cost = _compute_cost(input_tok, output_tok)
    row = {
        "timestamp":        datetime.utcnow().isoformat(),
        "query":            query,
        "api_calls":        api_calls,
        "input_tokens":     input_tok,
        "output_tokens":    output_tok,
        "input_cost_usd":   round(in_cost,  8),
        "output_cost_usd":  round(out_cost, 8),
        "query_cost_usd":   round(in_cost + out_cost, 8),
        # session_total_cost_usd filled in _write_csv
        "session_total_cost_usd": 0.0,
    }
    with USAGE_LOCK:
        USAGE_LOG.append(row)
    _write_csv()


atexit.register(_write_csv)  # final flush when Python process exits

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
    """Return the current session's usage/cost CSV as a file download."""
    _write_csv()          # ensure latest data is flushed
    if not os.path.exists(REPORT_PATH):
        return jsonify({"error": "No usage data yet."}), 404
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        csv_content = f.read()
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{REPORT_PATH}"'},
    )


@app.route("/api/usage_summary")
def usage_summary():
    """Return a JSON summary of session usage and cost."""
    with USAGE_LOCK:
        rows = list(USAGE_LOG)
    total_input  = sum(r["input_tokens"]  for r in rows)
    total_output = sum(r["output_tokens"] for r in rows)
    total_calls  = sum(r["api_calls"]     for r in rows)
    total_cost   = sum(r["query_cost_usd"] for r in rows)
    return jsonify({
        "session_start":       SESSION_START,
        "queries":             len(rows),
        "total_api_calls":     total_calls,
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "total_cost_usd":      round(total_cost, 6),
        "model":               "gemini-2.5-flash",
        "pricing": {
            "input_per_1m_tokens":  PRICE_INPUT_PER_M,
            "output_per_1m_tokens": PRICE_OUTPUT_PER_M,
        },
        "log": rows,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"NBA Stats Explorer → http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
