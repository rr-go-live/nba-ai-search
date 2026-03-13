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

# ── Claude agent ──────────────────────────────────────────────────────────────
CLAUDE_MODEL     = "claude-sonnet-4-20250514"
MAX_AGENT_ITERS  = 12
AGENT_MAX_TOKENS = 4096

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# ── Data limits ───────────────────────────────────────────────────────────────
MAX_GAME_LOG_ROWS = 250    # rows returned to frontend in game log
MAX_SEASONS_SPAN  = 15     # max seasons fetched in multi-season range queries