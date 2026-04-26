"""
nba_http.py
===========
PURPOSE: HTTP session for stats.nba.com with Chrome TLS fingerprint spoofing.

WHY THIS EXISTS:
  stats.nba.com is protected by Akamai Bot Manager, which checks two things:
    1. TLS fingerprint — Python's ssl/requests library has a non-browser
       TLS ClientHello that Akamai instantly flags as bot traffic, especially
       from cloud server IPs (AWS, GCP, Render, etc.).
    2. Cookies — specifically _abck / ak_bmsc / bm_sz set by nba.com.

  On cloud servers, even with valid cookies, standard requests/urllib gets
  hung at the TCP layer until timeout (30–90 s) — exactly the pattern seen
  in the logs: "Read timed out. (read timeout=30)"

THE FIX — curl_cffi:
  curl_cffi is a Python binding to curl-impersonate, which patches libcurl
  to replicate Chrome's exact TLS ClientHello (cipher suite order, extensions,
  elliptic curves, ALPN, etc.). Akamai cannot distinguish it from real Chrome.

  Key change vs standard requests:
    requests.Session()                     ← flagged by Akamai
    curl_cffi.requests.Session(impersonate="chrome124")  ← passes as Chrome

  The cookie harvest still runs to satisfy Akamai's cookie checks. Together,
  TLS impersonation + valid cookies = reliable access from any IP.

SESSION LIFECYCLE:
  A single Session is created once per process and reused for all calls.
  It re-harvests cookies every COOKIE_TTL seconds (default: 10 minutes).

RETRY LOGIC:
  Timeout and connection errors are retried up to MAX_RETRIES times with
  exponential backoff (2 s → 4 s → 8 s). A new cookie harvest is forced on
  retry so stale cookies don't compound the problem.

PARSING:
  All nba_api endpoints return JSON with this structure:
    {"resultSets": [{"name": "...", "headers": [...], "rowSet": [[...]]}]}
  parse_response() converts that into a list of dicts (one per result set).
"""

import time
import logging
import threading
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
NBA_HOME   = "https://www.nba.com/"
NBA_STATS  = "https://stats.nba.com/stats/"
COOKIE_TTL = 600   # seconds before re-harvesting cookies (10 min)
MAX_RETRIES = 3    # attempts before giving up on a single request

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

# ── Module-level session (shared across all calls) ─────────────────────────────
_session:          object         = None
_session_lock:     threading.Lock = threading.Lock()
_last_cookie_time: float          = 0.0


def _make_session():
    """
    _make_session
    -------------
    Create a curl_cffi Session impersonating Chrome 124.

    Falls back to a standard requests.Session if curl_cffi is not installed,
    logging a warning so the issue is visible in logs.

    Returns:
        Session object (curl_cffi or requests)
    """
    try:
        from curl_cffi import requests as cffi_requests
        sess = cffi_requests.Session(impersonate="chrome124")
        sess.headers.update(HEADERS)
        logger.info("Using curl_cffi Chrome-impersonation session (Akamai bypass active)")
        return sess
    except ImportError:
        import requests as std_requests
        sess = std_requests.Session()
        sess.headers.update(HEADERS)
        logger.warning(
            "curl_cffi not installed — falling back to standard requests. "
            "stats.nba.com may timeout on cloud server IPs. "
            "Run: pip install curl_cffi"
        )
        return sess


def _get_session(force_refresh: bool = False, timeout_home: int = 15) -> object:
    """
    _get_session
    ------------
    Return a session with fresh Akamai cookies from nba.com.

    Thread-safe: only one thread harvests cookies at a time. Cookies are
    refreshed every COOKIE_TTL seconds, or immediately if force_refresh=True.

    Args:
        force_refresh (bool): Ignore TTL and re-harvest cookies right now.
        timeout_home (int): Timeout in seconds for the nba.com homepage visit.

    Returns:
        Session object ready for stats.nba.com requests.
    """
    global _session, _last_cookie_time

    with _session_lock:
        now = time.time()
        needs_refresh = (
            _session is None
            or force_refresh
            or (now - _last_cookie_time) > COOKIE_TTL
        )

        if needs_refresh:
            logger.info("Harvesting fresh nba.com cookies (Chrome TLS impersonation)...")
            sess = _make_session()
            try:
                resp = sess.get(NBA_HOME, timeout=timeout_home, allow_redirects=True)
                cookies = dict(resp.cookies)
                logger.info(f"Got {len(cookies)} cookies: {list(cookies.keys())}")
                time.sleep(1.0)   # brief pause after homepage visit
            except Exception as e:
                logger.warning(f"Cookie harvest failed (will still try stats calls): {e}")
            _session          = sess
            _last_cookie_time = now

        return _session


def get(endpoint: str, params: dict, timeout: int = 30) -> dict:
    """
    get
    ---
    Make a GET request to stats.nba.com/{endpoint}?{params}.

    Retries up to MAX_RETRIES times on timeout or connection errors, forcing
    a cookie refresh on each retry so stale cookies don't compound failures.

    Args:
        endpoint (str): e.g. "playergamelog", "playercareerstats"
        params (dict):  Query-string parameters.
        timeout (int):  Per-attempt timeout in seconds.

    Returns:
        dict: Parsed JSON response body.

    Raises:
        Exception: If all retries are exhausted or a non-timeout error occurs.
    """
    url = NBA_STATS + endpoint + "?" + urlencode(params)
    logger.info(f"GET {url[:120]}...")

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        force_refresh = attempt > 1    # re-harvest cookies on retries
        sess = _get_session(force_refresh=force_refresh)
        try:
            resp = sess.get(url, timeout=timeout)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code} from {url}")
            return resp.json()

        except Exception as exc:
            last_exc = exc
            exc_str  = str(exc)
            is_timeout = (
                "timed out" in exc_str.lower()
                or "timeout" in exc_str.lower()
                or "connectionerror" in type(exc).__name__.lower()
                or "connecttimeout" in type(exc).__name__.lower()
            )
            if is_timeout and attempt < MAX_RETRIES:
                wait = 2 ** attempt   # 2 s → 4 s → 8 s
                logger.warning(
                    f"[nba_http] timeout on attempt {attempt}/{MAX_RETRIES} "
                    f"— retrying in {wait}s (fresh cookies on retry)"
                )
                time.sleep(wait)
                continue
            # Non-timeout error or final attempt — re-raise
            raise


def parse_response(data: dict) -> dict:
    """
    parse_response
    --------------
    Parse a standard nba_api stats JSON response into a dict of lists-of-dicts.

    Input format (from stats.nba.com):
      {"resultSets": [
        {"name": "PlayerGameLog", "headers": ["COL1","COL2",...], "rowSet": [[v1,v2,...], ...]},
        ...
      ]}

    Output format:
      {"PlayerGameLog": [{"COL1": v1, "COL2": v2, ...}, ...], ...}

    Args:
        data (dict): Raw JSON dict from stats.nba.com.

    Returns:
        dict: Result-set name → list of row dicts.
    """
    result = {}
    for rs in data.get("resultSets", []):
        name    = rs.get("name", "unknown")
        headers = rs.get("headers", [])
        row_set = rs.get("rowSet",  [])
        result[name] = [dict(zip(headers, row)) for row in row_set]
    return result


def get_parsed(endpoint: str, params: dict, timeout: int = 30) -> dict:
    """
    get_parsed
    ----------
    Convenience wrapper: fetch + parse in one call.

    Args:
        endpoint (str): stats.nba.com endpoint name.
        params (dict):  Query parameters.
        timeout (int):  Request timeout in seconds.

    Returns:
        dict: Parsed result sets.
    """
    return parse_response(get(endpoint, params, timeout))
