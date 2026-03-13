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

import json
import logging
import threading
import uuid
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT
from agent import NBAStatsAgent

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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"NBA Stats Explorer → http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
