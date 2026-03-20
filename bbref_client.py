"""
bbref_client.py
===============
Basketball Reference scraper — fallback for current-season game logs
when stats.nba.com is unavailable or returns empty data.

Strategy:
  1. Search BBRef for the player by name to resolve their BBRef player ID.
  2. Fetch the player's game log page and parse the #pgl_basic table.
  3. Return rows in the same key format as nba_http.get_parsed("playergamelog"),
     so the existing _to_row() and _avg() functions work without modification.

Rate limiting:
  BBRef enforces rate limits. BBREF_DELAY seconds are added between requests.
  On HTTP 429, we log a warning and return [] rather than retrying.
"""

import re
import time
import logging
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

BBREF_BASE  = "https://www.basketball-reference.com"
BBREF_DELAY = 4.0  # seconds between requests — BBRef is strict

_bbref_session: requests.Session = None
_id_cache: dict = {}  # player_name_lower → bbref_id

BBREF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.basketball-reference.com/",
}


def _get_session() -> requests.Session:
    global _bbref_session
    if _bbref_session is None:
        _bbref_session = requests.Session()
        _bbref_session.headers.update(BBREF_HEADERS)
    return _bbref_session


def find_bbref_id(player_name: str) -> str | None:
    """
    Resolve a player's Basketball Reference ID by name.
    Returns e.g. 'jamesle01' for LeBron James, None on failure.

    Hits BBRef's search endpoint; follows a direct redirect when there is only
    one match, otherwise scrapes the first result from the search results page.
    Results are cached in-process so repeat calls don't re-fetch.
    """
    key = player_name.lower().strip()
    if key in _id_cache:
        return _id_cache[key]

    url = f"{BBREF_BASE}/search/search.fcgi?search={quote_plus(player_name)}"
    try:
        time.sleep(BBREF_DELAY)
        resp = _get_session().get(url, timeout=15, allow_redirects=True)

        if resp.status_code == 429:
            logger.warning("BBRef rate-limited during player search")
            return None

        # Direct redirect → player page
        if "/players/" in resp.url and resp.url.rstrip("/").endswith(".html"):
            m = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", resp.url)
            if m:
                bbref_id = m.group(1)
                _id_cache[key] = bbref_id
                logger.info(f"BBRef ID for '{player_name}': {bbref_id} (redirect)")
                return bbref_id

        # Search results page — grab first player link
        m = re.search(r'href="/players/[a-z]/([a-z0-9]+)\.html"', resp.text)
        if m:
            bbref_id = m.group(1)
            _id_cache[key] = bbref_id
            logger.info(f"BBRef ID for '{player_name}': {bbref_id} (search result)")
            return bbref_id

        logger.warning(f"BBRef: no player ID found for '{player_name}'")

    except Exception as e:
        logger.warning(f"BBRef player search failed for '{player_name}': {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def _safe_int(v: str, default: int = 0) -> int:
    try:
        s = v.strip() if v else ""
        return int(s) if s else default
    except (ValueError, TypeError):
        return default


def _parse_bbref_table(html: str) -> list[dict]:
    """
    Parse the #pgl_basic tbody from a BBRef game log page.
    Each cell is keyed by its data-stat attribute value.
    Skips repeated header rows (class="thead") and non-game rows
    (Did Not Play / Did Not Dress / Inactive — these have a 'reason' cell
    and no 'pts' value).
    """
    rows = []

    tbody_m = re.search(
        r'<table[^>]+id="pgl_basic"[^>]*>.*?<tbody>(.*?)</tbody>',
        html, re.DOTALL | re.IGNORECASE,
    )
    if not tbody_m:
        logger.warning("BBRef: #pgl_basic table not found in response")
        return rows

    tbody = tbody_m.group(1)

    for tr_m in re.finditer(r'<tr\b([^>]*)>(.*?)</tr>', tbody, re.DOTALL):
        tr_attrs = tr_m.group(1)
        tr_body  = tr_m.group(2)

        if "thead" in tr_attrs:          # repeated column-header rows
            continue

        row: dict[str, str] = {}
        for td_m in re.finditer(
            r'<td\b[^>]+data-stat="([^"]+)"[^>]*>(.*?)</td>',
            tr_body, re.DOTALL,
        ):
            stat  = td_m.group(1)
            inner = td_m.group(2)
            text  = re.sub(r"<[^>]+>", "", inner).strip()
            row[stat] = text

        if not row:
            continue

        # Skip DNP / inactive / no-pts rows
        if row.get("reason") or not row.get("pts"):
            continue

        rows.append(row)

    return rows


def _bbref_row_to_nba_format(r: dict, season: str) -> dict:
    """
    Convert one BBRef game-log row into the key/value format returned by
    nba_http.get_parsed("playergamelog"), so _to_row() and _avg() are unchanged.

    BBRef stores FG/3P/FT percentages as 0–1 decimals, matching NBA API format.
    The 'game_result' cell looks like "W (+15)" or "L (-3)".
    """
    team     = r.get("team_id", "")
    opp      = r.get("opp_id", "")
    location = r.get("game_location", "")   # empty = home, "@" = away
    matchup  = f"{team} @ {opp}" if location == "@" else f"{team} vs. {opp}"

    result_raw = r.get("game_result", "")
    wl = "W" if result_raw.startswith("W") else ("L" if result_raw.startswith("L") else "")

    return {
        "SEASON":     season,
        "GAME_DATE":  r.get("date_game", ""),
        "MATCHUP":    matchup,
        "WL":         wl,
        "MIN":        r.get("mp", ""),
        "PTS":        _safe_int(r.get("pts")),
        "REB":        _safe_int(r.get("trb")),
        "AST":        _safe_int(r.get("ast")),
        "STL":        _safe_int(r.get("stl")),
        "BLK":        _safe_int(r.get("blk")),
        "TOV":        _safe_int(r.get("tov")),
        "FGM":        _safe_int(r.get("fg")),
        "FGA":        _safe_int(r.get("fga")),
        "FG_PCT":     _safe_float(r.get("fg_pct")),
        "FG3M":       _safe_int(r.get("fg3")),
        "FG3A":       _safe_int(r.get("fg3a")),
        "FG3_PCT":    _safe_float(r.get("fg3_pct")),
        "FTM":        _safe_int(r.get("ft")),
        "FTA":        _safe_int(r.get("fta")),
        "FT_PCT":     _safe_float(r.get("ft_pct")),
        "PLUS_MINUS": _safe_int(r.get("plus_minus")),
        "Game_ID":    "",
        "_source":    "bbref",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_game_log(bbref_id: str, season_end_year: int) -> list:
    """
    Fetch a player's regular-season game log from Basketball Reference.

    Args:
        bbref_id:        e.g. 'jamesle01'
        season_end_year: e.g. 2026 for the 2025-26 season

    Returns a list of row dicts in NBA API game-log format.
    Returns [] on network error, rate-limit, or missing table.
    """
    initial    = bbref_id[0]
    season_str = f"{season_end_year - 1}-{str(season_end_year)[-2:]}"
    url = f"{BBREF_BASE}/players/{initial}/{bbref_id}/gamelog/{season_end_year}/"

    try:
        time.sleep(BBREF_DELAY)
        resp = _get_session().get(url, timeout=20, allow_redirects=True)

        if resp.status_code == 429:
            logger.warning(f"BBRef rate-limited fetching game log for {bbref_id}")
            return []
        if resp.status_code != 200:
            logger.warning(f"BBRef returned HTTP {resp.status_code} for {url}")
            return []

        raw_rows = _parse_bbref_table(resp.text)
        rows = [_bbref_row_to_nba_format(r, season_str) for r in raw_rows]
        logger.info(f"BBRef game log {bbref_id} {season_str}: {len(rows)} games")
        return rows

    except Exception as e:
        logger.warning(f"BBRef game log fetch failed ({bbref_id} {season_end_year}): {e}")
        return []
