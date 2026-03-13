"""
nba_http.py
===========
PURPOSE: Cookie-harvested HTTP session for stats.nba.com.

WHY THIS EXISTS:
  stats.nba.com is protected by Akamai Bot Manager, which checks:
    1. TLS fingerprint — Python's ssl library looks different from Chrome
    2. Cookies — specifically _abck / ak_bmsc / bm_sz set by nba.com

  nba_api uses Python's built-in urllib/requests under the hood, which has
  a non-browser TLS fingerprint. Akamai detects this and hangs the connection
  until timeout (30–90 seconds) — which is exactly what we see in the logs:
    "Read timed out. (read timeout=30)"

THE FIX:
  Visit https://www.nba.com/ first to harvest Akamai cookies, then reuse that
  session (with those cookies) for all stats.nba.com calls. Akamai sees a
  request that came from a browser-like session with valid cookies and lets it
  through — typically in 1–3 seconds.

SESSION LIFECYCLE:
  A single Session is created once per process and reused for all calls.
  It re-harvests cookies every COOKIE_TTL seconds (default: 10 minutes) so
  the session stays fresh for long-running queries.

PARSING:
  All nba_api endpoints return JSON with this structure:
    {"resultSets": [{"name": "...", "headers": [...], "rowSet": [[...]]}]}
  parse_response() converts that into a list of dicts (one per result set),
  matching what nba_api's get_normalized_dict() does.
"""

import time
import logging
import threading
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NBA_HOME  = "https://www.nba.com/"
NBA_STATS = "https://stats.nba.com/stats/"
COOKIE_TTL = 600  # seconds before re-harvesting cookies (10 min)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":             "application/json, text/plain, */*",
    "Accept-Language":    "en-US,en;q=0.9",
    "Accept-Encoding":    "gzip, deflate, br",
    "Origin":             "https://www.nba.com",
    "Referer":            "https://www.nba.com/",
    "Connection":         "keep-alive",
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-site",
    "Sec-Ch-Ua":          '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}

# ── Module-level session (shared across all calls) ────────────────────────────
_session:         requests.Session = None
_session_lock:    threading.Lock   = threading.Lock()
_last_cookie_time: float           = 0.0


def _get_session(timeout_home: int = 15) -> requests.Session:
    """
    Return a requests.Session with fresh Akamai cookies from nba.com.

    Thread-safe: only one thread harvests cookies at a time.
    Cookies are refreshed every COOKIE_TTL seconds.
    """
    global _session, _last_cookie_time

    with _session_lock:
        now = time.time()
        if _session is None or (now - _last_cookie_time) > COOKIE_TTL:
            logger.info("Harvesting fresh nba.com cookies...")
            sess = requests.Session()
            sess.headers.update(HEADERS)
            try:
                resp = sess.get(NBA_HOME, timeout=timeout_home, allow_redirects=True)
                cookies = dict(resp.cookies)
                logger.info(f"Got {len(cookies)} cookies: {list(cookies.keys())}")
                time.sleep(1.0)  # brief pause after homepage visit
            except Exception as e:
                logger.warning(f"Cookie harvest failed (will still try): {e}")
            _session          = sess
            _last_cookie_time = now

        return _session


def get(endpoint: str, params: dict, timeout: int = 30) -> dict:
    """
    Make a GET request to stats.nba.com/{endpoint}?{params}.

    Uses the cookie-harvested session so Akamai lets it through.
    Returns the parsed JSON dict.

    Args:
        endpoint: e.g. "playergamelog", "playercareerstats"
        params:   query-string parameters dict
        timeout:  request timeout in seconds

    Raises:
        requests.exceptions.RequestException on network error
        ValueError if response is not valid JSON or non-200
    """
    url = NBA_STATS + endpoint + "?" + urlencode(params)
    logger.info(f"GET {url[:120]}...")

    sess = _get_session()
    resp = sess.get(url, timeout=timeout)

    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code} from {url}")

    return resp.json()


def parse_response(data: dict) -> dict:
    """
    Parse a standard nba_api stats JSON response into a dict of lists-of-dicts.

    Input format (from stats.nba.com):
      {"resultSets": [
        {"name": "PlayerGameLog", "headers": ["COL1","COL2",...], "rowSet": [[v1,v2,...], ...]},
        ...
      ]}

    Output format:
      {"PlayerGameLog": [{"COL1": v1, "COL2": v2, ...}, ...], ...}

    This is equivalent to what nba_api's get_normalized_dict() returns,
    but applied to data we fetched ourselves with the cookie session.
    """
    result = {}
    for rs in data.get("resultSets", []):
        name    = rs.get("name", "unknown")
        headers = rs.get("headers", [])
        row_set = rs.get("rowSet",  [])
        result[name] = [dict(zip(headers, row)) for row in row_set]
    return result


def get_parsed(endpoint: str, params: dict, timeout: int = 30) -> dict:
    """Convenience wrapper: fetch + parse in one call."""
    return parse_response(get(endpoint, params, timeout))