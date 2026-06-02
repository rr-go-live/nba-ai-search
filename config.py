"""
config.py
=========
Central configuration for NBA Stats Explorer.

This replaces the old video-highlight config entirely.
No video directories, no ffmpeg — only stats fetching settings.
"""

# ── NBA API ───────────────────────────────────────────────────────────────────
NBA_API_TIMEOUT = 30
NBA_API_DELAY   = 0.6          # polite delay between API calls (seconds)
NBA_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":  "https://www.nba.com/",
    "Origin":   "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "Accept": "application/json",
}

# ── Season strings ────────────────────────────────────────────────────────────
DEFAULT_SEASON  = "2025-26"
MIN_YEAR        = 1996
MAX_YEAR        = 2026

# Build a lookup so "2018" → "2017-18", "2017-18" → "2017-18", etc.
YEAR_TO_SEASON = {}
for _y in range(MIN_YEAR, MAX_YEAR + 1):
    _s = f"{_y}-{str(_y + 1)[-2:]}"
    YEAR_TO_SEASON[str(_y)]       = _s   # start year  e.g. "2017"
    YEAR_TO_SEASON[_s]            = _s   # already canonical
    YEAR_TO_SEASON[str(_y + 1)]  = _s   # end year    e.g. "2018"

# ── Gemini agent ──────────────────────────────────────────────────────────────
# Two supported backends — auto-detected from environment variables:
#
# 1. Google AI Studio (default, free tier)
#    Geo-restricted: not available in all countries.
#    Set env var: GOOGLE_API_KEY=<key from https://aistudio.google.com/apikey>
#
# 2. Vertex AI (globally available, requires GCP project + billing)
#    Bypasses geo-restrictions. Uses Application Default Credentials (ADC).
#    Local setup:  gcloud auth application-default login
#    Render setup: set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON.
#    Set env vars: GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
#                  GOOGLE_CLOUD_LOCATION=us-central1   (optional, defaults below)
#
# If GOOGLE_CLOUD_PROJECT is set, Vertex AI is used automatically.
# Otherwise falls back to AI Studio via GOOGLE_API_KEY.
GEMINI_MODEL            = "gemini-2.5-flash"
VERTEX_LOCATION_DEFAULT = "us-central1"
MAX_AGENT_ITERS         = 12
AGENT_MAX_TOKENS        = 16384

# ── Flask ─────────────────────────────────────────────────────────────────────
# PORT env var lets you override without touching code (e.g. PORT=8080 python3 app.py).
# app.py will auto-increment from this value if the port is already in use.
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(__import__('os').environ.get('PORT', 5000))

# ── Data limits ───────────────────────────────────────────────────────────────
MAX_GAME_LOG_ROWS = 250    # rows returned to frontend in game log
MAX_SEASONS_SPAN  = 30     # max seasons fetched; 30 covers full supported range (1996–2026)