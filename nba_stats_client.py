"""
nba_stats_client.py
===================
PURPOSE: NBA statistics data layer.

NETWORK: All stats.nba.com calls go through nba_http (cookie session).
STATIC:  nba_api.stats.static for player/team lookups (zero network).

KEY CHANGES vs previous version:
  - get_player_conditional_stats() replaces get_player_with_without_stats()
    → supports a LIST of condition player IDs (multi-player inactive splits)
    → all_active=True  → H2H: games where ALL condition players were active
    → all_active=False → games where NONE of the condition players played
  - Career stats now include fg3a (three-point attempt volume per game)
  - Career stats now return playoff data (SeasonTotalsPostSeason)
  - BLK bug fixed (was incorrectly multiplied by 100)
  - headshot_url() added: https://cdn.nba.com/headshots/nba/latest/1040x760/{id}.png
"""

import re
import time
import logging
import unicodedata
from typing import Optional
from difflib import SequenceMatcher

from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams   as static_teams

import nba_http
from config import (
    NBA_API_DELAY, NBA_API_TIMEOUT,
    DEFAULT_SEASON, YEAR_TO_SEASON,
    MAX_GAME_LOG_ROWS, MAX_SEASONS_SPAN, MIN_YEAR, MAX_YEAR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Headshot
# ─────────────────────────────────────────────────────────────────────────────

def headshot_url(player_id: int) -> str:
    """Official NBA CDN headshot — always resolves (silhouette for unknowns)."""
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"


# ─────────────────────────────────────────────────────────────────────────────
# Lookup tables
# ─────────────────────────────────────────────────────────────────────────────

NICKNAME_MAP = {
    "lebron": "LeBron James", "king james": "LeBron James", "bron": "LeBron James",
    "steph": "Stephen Curry", "chef curry": "Stephen Curry", "steph curry": "Stephen Curry",
    "kd": "Kevin Durant", "slim reaper": "Kevin Durant",
    "giannis": "Giannis Antetokounmpo", "greek freak": "Giannis Antetokounmpo",
    "jokic": "Nikola Jokic", "joker": "Nikola Jokic",
    "ad": "Anthony Davis", "the brow": "Anthony Davis",
    "cp3": "Chris Paul", "point god": "Chris Paul",
    "kawhi": "Kawhi Leonard", "the klaw": "Kawhi Leonard",
    "luka": "Luka Doncic", "luka magic": "Luka Doncic",
    "tatum": "Jayson Tatum", "jt": "Jayson Tatum",
    "dame": "Damian Lillard", "dame dolla": "Damian Lillard",
    "ja": "Ja Morant", "book": "Devin Booker",
    "shai": "Shai Gilgeous-Alexander", "sga": "Shai Gilgeous-Alexander",
    "embiid": "Joel Embiid", "the process": "Joel Embiid",
    "kyrie": "Kyrie Irving", "uncle drew": "Kyrie Irving",
    "kobe": "Kobe Bryant", "mamba": "Kobe Bryant", "bean": "Kobe Bryant",
    "shaq": "Shaquille O'Neal", "diesel": "Shaquille O'Neal",
    "td": "Tim Duncan", "big fundamental": "Tim Duncan",
    "dirk": "Dirk Nowitzki", "wemby": "Victor Wembanyama",
    "ant": "Anthony Edwards", "ant-man": "Anthony Edwards",
    "pg": "Paul George", "pg13": "Paul George",
    "jaylen": "Jaylen Brown", "brown": "Jaylen Brown",
    "porzingis": "Kristaps Porzingis", "kp": "Kristaps Porzingis",
    "horford": "Al Horford",
}

TEAM_ALIASES = {
    "dubs": "Golden State Warriors", "warriors": "Golden State Warriors",
    "cavs": "Cleveland Cavaliers", "clips": "Los Angeles Clippers",
    "blazers": "Portland Trail Blazers", "trailblazers": "Portland Trail Blazers",
    "t-wolves": "Minnesota Timberwolves", "wolves": "Minnesota Timberwolves",
    "mavs": "Dallas Mavericks", "spurs": "San Antonio Spurs",
    "thunder": "Oklahoma City Thunder", "suns": "Phoenix Suns",
    "kings": "Sacramento Kings", "sixers": "Philadelphia 76ers",
    "knicks": "New York Knicks", "pelicans": "New Orleans Pelicans",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """NFD normalize + strip accents → ASCII lowercase. Jokić→jokic, Dončić→doncic."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Static lookups — zero network
# ─────────────────────────────────────────────────────────────────────────────

def search_player(name: str) -> Optional[dict]:
    """Find a player by name/nickname. Bundled JSON — no network. Unicode-safe."""
    name_clean = name.strip().lower()
    name_norm  = _norm(name)
    all_players = static_players.get_players()

    if name_clean in NICKNAME_MAP:
        canonical = NICKNAME_MAP[name_clean]
        for p in all_players:
            if _norm(p["full_name"]) == _norm(canonical):
                return p

    for p in all_players:
        if _norm(p["full_name"]) == name_norm:
            return p
    for p in all_players:
        if _norm(p["last_name"]) == name_norm:
            return p

    matches = [p for p in all_players if name_norm in _norm(p["full_name"])]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        active = [p for p in matches if p["is_active"]]
        return active[0] if active else matches[0]

    best, best_r = None, 0.0
    for p in all_players:
        r = SequenceMatcher(None, name_norm, _norm(p["full_name"])).ratio()
        if r > best_r:
            best_r, best = r, p
    return best if best_r >= 0.75 else None


def search_team(name: str) -> Optional[dict]:
    """Find a team by name/city/nickname/alias. Zero network."""
    all_teams = static_teams.get_teams()
    n = name.strip().lower()
    if n in TEAM_ALIASES:
        full = TEAM_ALIASES[n].lower()
        for t in all_teams:
            if t["full_name"].lower() == full:
                return t
    for t in all_teams:
        if any([
            t["nickname"].lower() == n,
            t["abbreviation"].lower() == n,
            t["full_name"].lower() == n,
            t["city"].lower() == n,
            n in t["full_name"].lower(),
            n in t["nickname"].lower(),
        ]):
            return t
    return None


def parse_season(raw: str) -> str:
    if not raw:
        return DEFAULT_SEASON
    cleaned = raw.strip()
    if cleaned in YEAR_TO_SEASON:
        return YEAR_TO_SEASON[cleaned]
    m = re.search(r"\b((?:19|20)\d{2})\b", cleaned)
    if m and m.group(1) in YEAR_TO_SEASON:
        return YEAR_TO_SEASON[m.group(1)]
    return DEFAULT_SEASON


def seasons_between(s_from: str, s_to: str) -> list:
    """
    Return a list of NBA season strings between s_from and s_to inclusive.

    ORDER: newest season first (descending).

    WHY DESCENDING:
    The NBA API is called once per season. If the cookie session throttles
    or errors partway through a long range (e.g. "since 2015" = 11 API calls),
    the calls that succeed are the ones fetched FIRST. Fetching newest seasons
    first means recent games always survive throttling — older historical data
    is less critical than the most recent matchups. The [:MAX_SEASONS_SPAN]
    slice then keeps the freshest data if we have to drop anything.
    """
    def start(s):
        try:
            return int(s.split("-")[0])
        except Exception:
            return 2024
    y0 = max(start(s_from), MIN_YEAR)
    y1 = min(start(s_to),   MAX_YEAR)
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(y0, y1 + 1)]
    seasons.reverse()   # newest first
    return seasons[:MAX_SEASONS_SPAN]


# ─────────────────────────────────────────────────────────────────────────────
# nba_http wrappers — all network calls through cookie session
# ─────────────────────────────────────────────────────────────────────────────

def _game_log(player_id: int, season: str,
              season_type: str = "Regular Season") -> list:
    """One season's game log via cookie session. Injects SEASON tag on each row."""
    time.sleep(NBA_API_DELAY)
    data = nba_http.get_parsed(
        "playergamelog",
        params={"PlayerID": player_id, "Season": season,
                "SeasonType": season_type, "LeagueID": "00",
                "DateFrom": "", "DateTo": ""},
        timeout=NBA_API_TIMEOUT,
    )
    rows = data.get("PlayerGameLog", [])
    for row in rows:
        row["SEASON"] = season
    return rows


def _career_stats_raw(player_id: int) -> dict:
    """Career stats all result sets via cookie session."""
    time.sleep(NBA_API_DELAY)
    return nba_http.get_parsed(
        "playercareerstats",
        params={"PlayerID": player_id, "PerMode": "PerGame", "LeagueID": "00"},
        timeout=NBA_API_TIMEOUT,
    )


def _player_info_http(player_id: int) -> dict:
    """Non-blocking player profile. 10s timeout — returns {} on failure."""
    try:
        time.sleep(NBA_API_DELAY)
        data = nba_http.get_parsed(
            "commonplayerinfo",
            params={"PlayerID": player_id, "LeagueID": "00"},
            timeout=10,
        )
        rows = data.get("CommonPlayerInfo", [])
        if rows:
            r = rows[0]
            return {
                "team":       r.get("TEAM_NAME", ""),
                "team_abbr":  r.get("TEAM_ABBREVIATION", ""),
                "position":   r.get("POSITION", ""),
                "jersey":     r.get("JERSEY", ""),
                "height":     r.get("HEIGHT", ""),
                "weight":     r.get("WEIGHT", ""),
            }
    except Exception as e:
        logger.info(f"commonplayerinfo {player_id}: {e}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_opponent(rows: list, opp_abbr: str) -> list:
    if not opp_abbr:
        return rows
    abbr = opp_abbr.upper()
    return [r for r in rows if abbr in (r.get("MATCHUP") or "").upper()]


def _avg(rows: list) -> dict:
    """Per-game averages from a game-log row list. All values guaranteed non-None."""
    if not rows:
        return {}
    n = len(rows)

    def mean(key):
        return round(sum(_safe_float(r.get(key)) for r in rows) / n, 1)

    def pct(key):
        # FG_PCT etc stored as 0–1 in game logs, convert to 0–100
        return round(sum(_safe_float(r.get(key)) for r in rows) / n * 100, 1)

    wins = sum(1 for r in rows if r.get("WL") == "W")
    return {
        "gp":      n,
        "pts":     mean("PTS"),
        "reb":     mean("REB"),
        "ast":     mean("AST"),
        "stl":     mean("STL"),
        "blk":     mean("BLK"),
        "tov":     mean("TOV"),
        "fg_pct":  pct("FG_PCT"),
        "fg3_pct": pct("FG3_PCT"),
        "fg3a":    mean("FG3A"),   # three-point attempt volume
        "ft_pct":  pct("FT_PCT"),
        "wins":    wins,
        "losses":  n - wins,
    }


def _parse_date_sort(raw_date: str) -> str:
    """
    Convert NBA API date string to YYYY-MM-DD for correct lexicographic sorting.

    WHY THIS EXISTS:
    The NBA API returns GAME_DATE as 'OCT 22, 2024' (all-caps month abbreviation).
    Sorting these as raw strings fails — 'APR' < 'FEB' alphabetically, so
    April games appear before February games. Cross-year comparisons are even
    worse: 'NOV 05, 2022' > 'OCT 25, 2023' alphabetically, which puts the
    2022 game after the 2023 game.

    Converting to 'YYYY-MM-DD' makes string sort identical to date sort.
    """
    if not raw_date:
        return ""
    try:
        from datetime import datetime
        return datetime.strptime(raw_date.strip().title(), "%b %d, %Y").strftime("%Y-%m-%d")
    except Exception:
        return raw_date   # fallback: keep original, better than crashing


def _to_row(r: dict) -> dict:
    """Convert raw game-log dict to frontend-safe dict. All values non-None."""
    raw_date = r.get("GAME_DATE") or ""
    return {
        "season":     r.get("SEASON") or "",
        "date":       raw_date,
        "date_sort":  _parse_date_sort(raw_date),   # YYYY-MM-DD for correct sort
        "matchup":    r.get("MATCHUP") or "",
        "result":     r.get("WL") or "",
        "min":        str(r.get("MIN") or ""),
        "pts":        _safe_int(r.get("PTS")),
        "reb":        _safe_int(r.get("REB")),
        "ast":        _safe_int(r.get("AST")),
        "stl":        _safe_int(r.get("STL")),
        "blk":        _safe_int(r.get("BLK")),
        "tov":        _safe_int(r.get("TOV")),
        "fgm":        _safe_int(r.get("FGM")),
        "fga":        _safe_int(r.get("FGA")),
        "fg_pct":     round(_safe_float(r.get("FG_PCT")) * 100, 1),
        "fg3m":       _safe_int(r.get("FG3M")),
        "fg3a":       _safe_int(r.get("FG3A")),
        "fg3_pct":    round(_safe_float(r.get("FG3_PCT")) * 100, 1),
        "ftm":        _safe_int(r.get("FTM")),
        "fta":        _safe_int(r.get("FTA")),
        "ft_pct":     round(_safe_float(r.get("FT_PCT")) * 100, 1),
        "plus_minus": _safe_int(r.get("PLUS_MINUS")),
        "game_id":    str(r.get("Game_ID") or ""),
    }


def _career_row(r: dict, is_playoffs: bool = False) -> dict:
    """
    Convert a career-stats row dict to frontend-safe format.
    Includes fg3a (three-point attempt volume per game) so the insight layer
    can distinguish high-efficiency/low-volume from truly elite shooting.
    team fallback: 'TOT' means player was traded that season.
    """
    team = str(r.get("TEAM_ABBREVIATION") or "")
    if not team:
        team = "TOT"   # "total" row for traded players; prevents "undefined" display
    return {
        "season":   str(r.get("SEASON_ID") or ""),
        "team":     team,
        "gp":       _safe_int(r.get("GP")),
        "pts":      round(_safe_float(r.get("PTS")), 1),
        "reb":      round(_safe_float(r.get("REB")), 1),
        "ast":      round(_safe_float(r.get("AST")), 1),
        "stl":      round(_safe_float(r.get("STL")), 1),
        "blk":      round(_safe_float(r.get("BLK")), 1),      # NOT * 100
        "fg_pct":   round(_safe_float(r.get("FG_PCT"))  * 100, 1),
        "fg3_pct":  round(_safe_float(r.get("FG3_PCT")) * 100, 1),
        "fg3a":     round(_safe_float(r.get("FG3A")), 1),     # volume context
        "ft_pct":   round(_safe_float(r.get("FT_PCT"))  * 100, 1),
        "playoffs": is_playoffs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public stat functions (called by tools.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_player_info(player_id: int) -> dict:
    """Non-blocking player profile. Always includes headshot_url."""
    info = _player_info_http(player_id)
    info["headshot_url"] = headshot_url(player_id)
    return info


def get_player_vs_team_stats(
    player_id: int,
    opponent_abbr: str,
    season_from: str = "2015-16",
    season_to: str   = DEFAULT_SEASON,
    season_type: str = "Regular Season",
) -> dict:
    """Player averages + game log vs specific opponent across multiple seasons."""
    all_rows, splits = [], []
    for season in seasons_between(season_from, season_to):
        try:
            rows     = _game_log(player_id, season, season_type)
            filtered = _filter_opponent(rows, opponent_abbr)
            if filtered:
                s = _avg(filtered)
                s["season"] = season
                splits.append(s)
                all_rows.extend(filtered)
        except Exception as e:
            logger.warning(f"GameLog {season}: {e}")
    # Sort newest games first so [:MAX_GAME_LOG_ROWS] always keeps the most
    # recent matchups. SEASON is already a string like "2024-25" so it sorts
    # lexicographically correctly. GAME_DATE within a season comes back
    # newest-first from the API so no secondary sort needed.
    all_rows.sort(key=lambda r: r.get("SEASON", ""), reverse=True)

    return {
        "averages":      _avg(all_rows),
        "season_splits": splits,
        "game_log":      [_to_row(r) for r in all_rows][:MAX_GAME_LOG_ROWS],
        "total_games":   len(all_rows),
    }


def get_player_career_stats(player_id: int) -> dict:
    """
    Career season-by-season averages + career totals.

    Returns both regular season and playoff data.
    Each season row includes fg3a (three-point attempts) for volume context.
    team='TOT' means the player was traded that season (prevents undefined display).
    """
    data = _career_stats_raw(player_id)

    regular_seasons = [
        _career_row(r, is_playoffs=False)
        for r in data.get("SeasonTotalsRegularSeason", [])
    ]
    playoff_seasons = [
        _career_row(r, is_playoffs=True)
        for r in data.get("SeasonTotalsPostSeason", [])
    ]

    def _totals(key: str) -> dict:
        rows = data.get(key, [])
        if not rows:
            return {}
        r = rows[-1]
        return {
            "gp":      _safe_int(r.get("GP")),
            "pts":     round(_safe_float(r.get("PTS")), 1),
            "reb":     round(_safe_float(r.get("REB")), 1),
            "ast":     round(_safe_float(r.get("AST")), 1),
            "stl":     round(_safe_float(r.get("STL")), 1),
            "blk":     round(_safe_float(r.get("BLK")), 1),
            "fg_pct":  round(_safe_float(r.get("FG_PCT"))  * 100, 1),
            "fg3_pct": round(_safe_float(r.get("FG3_PCT")) * 100, 1),
            "fg3a":    round(_safe_float(r.get("FG3A")), 1),
            "ft_pct":  round(_safe_float(r.get("FT_PCT"))  * 100, 1),
        }

    return {
        "seasons":         regular_seasons,
        "playoff_seasons": playoff_seasons,
        "career_totals":   _totals("CareerTotalsRegularSeason"),
        "playoff_totals":  _totals("CareerTotalsPostSeason"),
    }


def get_player_conditional_stats(
    player_id: int,
    condition_player_ids: list,
    all_active: bool        = True,
    season_from: str        = DEFAULT_SEASON,
    season_to: str          = DEFAULT_SEASON,
    opponent_abbr: str      = "",
    season_type: str        = "Regular Season",
) -> dict:
    """
    Primary player's stats split by condition players' active/inactive status.

    THREE USE CASES:

    1. H2H (head-to-head) — all_active=True, opposing players:
       "LeBron vs Curry H2H since 2015"
       → LeBron's game log filtered to games where Curry also played
       → opponent_abbr should be the other player's team (e.g. "GSW")
       → condition_player_ids=[curry_id], all_active=True

    2. Single inactive teammate — all_active=False:
       "Tatum stats when Brown is inactive"
       → Games where Brown's Game_ID is NOT in Tatum's log
       → condition_player_ids=[brown_id], all_active=False

    3. Multiple inactive teammates — all_active=False, multiple IDs:
       "Tatum stats when Brown AND Porzingis are both inactive"
       → Games where NEITHER Brown NOR Porzingis appeared
       → condition_player_ids=[brown_id, porzingis_id], all_active=False
       → Uses complement of UNION: game_id NOT IN (brown_ids ∪ porzingis_ids)

    ALGORITHM:
      1. Fetch primary player's game log (+ optional opponent filter).
      2. For each condition player, fetch their game IDs.
      3. all_active=True:  primary_rows WHERE game_id IN intersection of all condition sets
         all_active=False: primary_rows WHERE game_id NOT IN union of all condition sets
    """
    all_seasons  = seasons_between(season_from, season_to)
    primary_rows = []
    cond_id_sets = {pid: set() for pid in condition_player_ids}

    def _norm_game_id(raw) -> str:
        """
        Normalize a Game_ID to a consistent string for set comparisons.

        WHY THIS EXISTS:
        The NBA API sometimes returns Game_ID as a JSON string ("0022401012")
        and sometimes as a JSON integer (22401012).  When cast naively with
        str(), the string form keeps its leading zero while the integer form
        loses it — so "0022401012" != "22401012" and the intersection is empty.

        Fix: always convert to int (drops any leading zeros) then back to str.
        Both sides of the comparison use this function so they always agree.
        """
        if not raw:
            return ""
        try:
            return str(int(raw))
        except (ValueError, TypeError):
            return str(raw).lstrip("0") or "0"

    for season in all_seasons:
        try:
            rows = _game_log(player_id, season, season_type)
            if opponent_abbr:
                rows = _filter_opponent(rows, opponent_abbr)
            primary_rows.extend(rows)
        except Exception as e:
            logger.warning(f"Primary log {season}: {e}")

        for cid in condition_player_ids:
            try:
                crows = _game_log(cid, season, season_type)
                cond_id_sets[cid].update(
                    _norm_game_id(r.get("Game_ID")) for r in crows
                )
            except Exception as e:
                logger.warning(f"Condition player {cid} log {season}: {e}")

    if not primary_rows:
        return {"error": "No game log data found."}

    if all_active:
        # Intersection: games where ALL condition players appeared
        matched_ids = set(_norm_game_id(r.get("Game_ID")) for r in primary_rows)
        for cid_set in cond_id_sets.values():
            matched_ids &= cid_set
        matched  = [r for r in primary_rows if _norm_game_id(r.get("Game_ID")) in matched_ids]
        opposite = [r for r in primary_rows if _norm_game_id(r.get("Game_ID")) not in matched_ids]
    else:
        # Complement of union: games where NONE of the condition players appeared
        union_ids = set()
        for cid_set in cond_id_sets.values():
            union_ids |= cid_set
        matched  = [r for r in primary_rows if _norm_game_id(r.get("Game_ID")) not in union_ids]
        opposite = [r for r in primary_rows if _norm_game_id(r.get("Game_ID")) in union_ids]

    # Sort both logs newest-first before slicing
    matched.sort( key=lambda r: r.get("SEASON", ""), reverse=True)
    opposite.sort(key=lambda r: r.get("SEASON", ""), reverse=True)

    return {
        "primary_averages":    _avg(matched),
        "comparison_averages": _avg(opposite),
        "primary_games":       len(matched),
        "comparison_games":    len(opposite),
        "primary_log":         [_to_row(r) for r in matched][:MAX_GAME_LOG_ROWS],
        "comparison_log":      [_to_row(r) for r in opposite][:MAX_GAME_LOG_ROWS],
        "all_active":          all_active,
        "total_games":         len(primary_rows),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Playoff series breakdown
# ─────────────────────────────────────────────────────────────────────────────

ROUND_LABELS = ['R1', 'R2', 'CF', 'Finals']

def _extract_opp(matchup: str) -> str:
    """
    Extract opponent abbreviation from a MATCHUP string.
    'LAL vs. BOS' → 'BOS'    'LAL @ BOS' → 'BOS'
    """
    if not matchup:
        return ''
    parts = matchup.upper().replace(' VS. ', ' VS ').replace(' @ ', ' @ ').split()
    # Format is always: TEAM_ABB (vs.|@) OPP_ABB
    if len(parts) >= 3:
        return parts[-1]
    return ''


def get_playoff_series_breakdown(
    player_id:   int,
    season:      str,
) -> dict:
    """
    Fetch a player's playoff game log for a single season and group it
    into per-series stats with individual game logs per series.

    WHY THIS EXISTS:
    A flat playoff game log gives no sense of which round each game belongs to.
    This function groups games by opponent abbreviation in chronological order —
    each unique opponent = one series. The first series = R1, second = R2, etc.
    up to a maximum of 4 rounds (Finals).

    RETURNS:
      {
        "series": [
          {
            "round_label": "R1",        # R1 / R2 / CF / Finals
            "opponent":    "OKC",       # opponent abbreviation
            "games":       6,           # games played in series
            "result":      "W 4-2",     # series result
            "averages":    { pts, reb, ast, ... },
            "game_log":    [ { ...row... }, ... ]
          },
          ...
        ],
        "overall_averages": { ... },
        "total_games": N
      }
    """
    try:
        rows = _game_log(player_id, season, "Playoffs")
    except Exception as e:
        logger.warning(f"playoff game log {player_id} {season}: {e}")
        return {"series": [], "overall_averages": {}, "total_games": 0}

    if not rows:
        return {"series": [], "overall_averages": {}, "total_games": 0}

    # Sort oldest → newest so series appear in round order
    # GAME_DATE is 'OCT 22, 2024' format — sort by season+date
    from datetime import datetime as dt
    def _sort_key(r):
        raw = (r.get("GAME_DATE") or "").strip().title()
        try:
            return dt.strptime(raw, "%b %d, %Y")
        except Exception:
            return dt(2000, 1, 1)

    rows_sorted = sorted(rows, key=_sort_key)

    # Group into series by opponent — preserve insertion order (Python 3.7+)
    series_map: dict = {}   # opp_abbr -> list of rows
    for r in rows_sorted:
        opp = _extract_opp(r.get("MATCHUP", ""))
        if opp not in series_map:
            series_map[opp] = []
        series_map[opp].append(r)

    series_list = []
    for idx, (opp, s_rows) in enumerate(series_map.items()):
        avgs = _avg(s_rows)
        wins   = sum(1 for r in s_rows if r.get("WL") == "W")
        losses = len(s_rows) - wins
        result = f"W {wins}-{losses}" if wins > losses else f"L {wins}-{losses}"
        round_label = ROUND_LABELS[idx] if idx < len(ROUND_LABELS) else f"R{idx+1}"
        series_list.append({
            "round_label": round_label,
            "opponent":    opp,
            "games":       len(s_rows),
            "wins":        wins,
            "losses":      losses,
            "result":      result,
            "averages":    avgs,
            "game_log":    [_to_row(r) for r in s_rows],
        })

    return {
        "series":           series_list,
        "overall_averages": _avg(rows_sorted),
        "total_games":      len(rows_sorted),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-player comparison
# ─────────────────────────────────────────────────────────────────────────────

def get_player_season_averages(
    player_id:   int,
    season_from: str = DEFAULT_SEASON,
    season_to:   str = DEFAULT_SEASON,
    season_type: str = "Regular Season",
) -> dict:
    """
    Per-game averages for a single player across a season range.

    HOW IT WORKS:
    Uses the career stats endpoint to get all season rows, then filters to
    the requested range and aggregates. This is one API call per player
    regardless of how many seasons are requested, making it fast for
    multi-player comparisons.

    For multi-season ranges the rows are GP-weighted so a 50-game season
    contributes proportionally more than a 10-game injury season.

    HISTORICAL PLAYERS:
    Works identically for retired players (Kobe, Shaq, Jordan etc.) —
    the nba_api static player list and career stats endpoint include all
    historical players going back to the 1940s.
    """
    data = _career_stats_raw(player_id)
    key  = "SeasonTotalsPostSeason" if season_type == "Playoffs" else "SeasonTotalsRegularSeason"
    rows = data.get(key, [])

    def _season_year(s: str) -> int:
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return 0

    y0 = _season_year(season_from)
    y1 = _season_year(season_to)
    filtered = [r for r in rows if y0 <= _season_year(r.get("SEASON_ID", "")) <= y1]

    if not filtered:
        return {"error": f"No {season_type} data for seasons {season_from}–{season_to}"}

    # GP-weighted average across multiple seasons
    total_gp = sum(_safe_int(r.get("GP")) for r in filtered)
    if total_gp == 0:
        return {"error": "No games played in range"}

    # Extract team_abbr from the career stats row for this season range.
    # This is the authoritative source for historical players (Kobe, Kawhi etc.)
    # where _player_info_http returns empty string because they are retired.
    # For multi-team seasons, the last row's abbreviation is used.
    hist_team_abbr = str(filtered[-1].get("TEAM_ABBREVIATION") or "")

    def wp(key):   # weighted per-game value
        return round(
            sum(_safe_float(r.get(key)) * _safe_int(r.get("GP")) for r in filtered) / total_gp, 1
        )

    def wpct(key): # weighted pct (stored as 0-1 in career stats)
        return round(
            sum(_safe_float(r.get(key)) * _safe_int(r.get("GP")) for r in filtered) / total_gp * 100, 1
        )

    wins   = sum(_safe_int(r.get("W",  0)) for r in filtered)
    losses = sum(_safe_int(r.get("L",  0)) for r in filtered)

    return {
        "gp":        total_gp,
        "pts":       wp("PTS"),
        "reb":       wp("REB"),
        "ast":       wp("AST"),
        "stl":       wp("STL"),
        "blk":       wp("BLK"),
        "tov":       wp("TOV"),
        "fg_pct":    wpct("FG_PCT"),
        "fg3_pct":   wpct("FG3_PCT"),
        "fg3a":      wp("FG3A"),
        "ft_pct":    wpct("FT_PCT"),
        "wins":      wins,
        "losses":    losses,
        "team_abbr": hist_team_abbr,   # from career stats row for this season
        "seasons_found": [str(r.get("SEASON_ID", "")) for r in filtered],
    }


def get_multi_player_stats(
    players: list,     # [{player_id, season_from, season_to, season_type, label}]
) -> dict:
    """
    Fetch per-game averages for multiple players in parallel configurations.

    USAGE:
      "Compare Kobe 2010 playoffs vs Tatum 2024 playoffs":
        players = [
          {player_id: 977,     season_from: "2009-10", season_to: "2009-10", season_type: "Playoffs", label: "Kobe 2010 Finals"},
          {player_id: 1628369, season_from: "2023-24", season_to: "2023-24", season_type: "Playoffs", label: "Tatum 2024 Finals"},
        ]

      "Compare LeBron, Curry, Durant career stats":
        players = [
          {player_id: 2544,   season_from: "2003-04", season_to: "2025-26", season_type: "Regular Season", label: "LeBron Career"},
          ...
        ]

    Returns a list of player result objects ready for the compare renderer.
    Each includes name, headshot, team, averages, and the label.
    The caller (agent) is responsible for resolving player IDs and labels.
    """
    results = []
    for cfg in players:
        pid          = cfg.get("player_id")
        season_from  = cfg.get("season_from", DEFAULT_SEASON)
        season_to    = cfg.get("season_to",   DEFAULT_SEASON)
        season_type  = cfg.get("season_type", "Regular Season")
        label        = cfg.get("label", "")

        try:
            avgs = get_player_season_averages(pid, season_from, season_to, season_type)
            info = get_player_info(pid)
            # For historical/retired players, _player_info_http returns empty
            # team_abbr. Fall back to the team_abbr embedded in the career
            # stats row for the requested season — this is always accurate.
            hist_abbr = avgs.get("team_abbr", "") if isinstance(avgs, dict) else ""
            team_abbr = info.get("team_abbr", "") or hist_abbr

            # For playoff comparisons, fetch series-level breakdown so the
            # frontend can show per-round stats and per-series game logs.
            # Only fetch if a single season is requested (playoff runs are
            # season-specific — multi-season playoff aggregates don't have
            # meaningful per-series breakdown).
            series_data = None
            if season_type == "Playoffs" and season_from == season_to:
                series_data = get_playoff_series_breakdown(pid, season_from)

            results.append({
                "player_id":    pid,
                "name":         info.get("full_name", label or f"Player {pid}"),
                "team":         info.get("team", ""),
                "team_abbr":    team_abbr,
                "position":     info.get("position", ""),
                "headshot_url": info.get("headshot_url", headshot_url(pid)),
                "label":        label,
                "season_from":  season_from,
                "season_to":    season_to,
                "season_type":  season_type,
                "averages":     avgs,
                "series":       series_data.get("series", []) if series_data else [],
                "playoff_games": series_data.get("total_games", 0) if series_data else 0,
            })
        except Exception as e:
            logger.warning(f"get_multi_player_stats player {pid}: {e}")
            results.append({"player_id": pid, "label": label, "error": str(e)})

    return {"players": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

# Maps natural language stat names → NBA API StatCategory param
STAT_CATEGORY_MAP = {
    # Points
    "pts": "PTS", "points": "PTS", "scoring": "PTS", "scorers": "PTS",
    "ppg": "PTS", "point": "PTS",
    # Rebounds
    "reb": "REB", "rebounds": "REB", "rebounders": "REB", "rpg": "REB",
    "rebound": "REB",
    # Assists
    "ast": "AST", "assists": "AST", "playmakers": "AST", "apg": "AST",
    "assist": "AST",
    # Steals
    "stl": "STL", "steals": "STL", "spg": "STL", "steal": "STL",
    # Blocks
    "blk": "BLK", "blocks": "BLK", "bpg": "BLK", "block": "BLK",
    # 3-pointers made
    "fg3m": "FG3M", "3pm": "FG3M", "threes": "FG3M", "three-pointers": "FG3M",
    "3-pointers": "FG3M", "triples": "FG3M",
    # Efficiency / EFF
    "eff": "EFF", "efficiency": "EFF", "pir": "EFF",
    # Minutes
    "min": "MIN", "minutes": "MIN",
}


def get_leaderboard(
    stat:        str = "pts",
    season:      str = DEFAULT_SEASON,
    season_type: str = "Regular Season",
    per_mode:    str = "PerGame",
    top_n:       int = 10,
) -> dict:
    """
    League-wide stat leaderboard via the leagueleaders endpoint.

    PARAMETERS:
      stat         — natural language stat name OR raw category ('PTS', 'REB', etc.)
      season       — e.g. '2025-26'
      season_type  — 'Regular Season' or 'Playoffs'
      per_mode     — 'PerGame' (averages) or 'Totals' (cumulative)
      top_n        — number of players to return (max 25)

    EXAMPLE:
      "Top 10 scorers this season"
        → stat='pts', season='2025-26', season_type='Regular Season', top_n=10

      "Points leaders in 2009-10 playoffs"
        → stat='pts', season='2009-10', season_type='Playoffs', top_n=10

    NETWORK:
      One API call to stats.nba.com/stats/leagueleaders via the cookie session.
      Returns an empty leaderboard on network failure so the agent can still
      provide a graceful response.
    """
    # Resolve stat string to API category
    cat_key = stat.strip().lower()
    category = STAT_CATEGORY_MAP.get(cat_key, cat_key.upper())

    top_n = min(max(top_n, 1), 25)

    try:
        time.sleep(NBA_API_DELAY)
        data = nba_http.get_parsed(
            "leagueleaders",
            params={
                "LeagueID":    "00",
                "PerMode":     per_mode,
                "Scope":       "S",
                "Season":      season,
                "SeasonType":  season_type,
                "StatCategory": category,
            },
            timeout=NBA_API_TIMEOUT,
        )
    except Exception as e:
        logger.error(f"leagueleaders {category} {season}: {e}")
        return {"error": str(e), "leaders": []}

    raw_rows = data.get("LeagueLeaders", [])[:top_n]

    leaders = []
    for rank, r in enumerate(raw_rows, 1):
        pid = _safe_int(r.get("PLAYER_ID"))
        leaders.append({
            "rank":        rank,
            "player_id":   pid,
            "name":        str(r.get("PLAYER") or ""),
            "team":        str(r.get("TEAM") or ""),
            "headshot_url": headshot_url(pid),
            "gp":          _safe_int(r.get("GP")),
            "pts":         round(_safe_float(r.get("PTS")), 1),
            "reb":         round(_safe_float(r.get("REB")), 1),
            "ast":         round(_safe_float(r.get("AST")), 1),
            "stl":         round(_safe_float(r.get("STL")), 1),
            "blk":         round(_safe_float(r.get("BLK")), 1),
            "fg_pct":      round(_safe_float(r.get("FG_PCT")) * 100, 1),
            "fg3_pct":     round(_safe_float(r.get("FG3_PCT")) * 100, 1),
            "stat_value":  round(_safe_float(r.get(category, r.get("PTS"))), 1),
        })

    return {
        "stat_category": category,
        "stat_label":    stat,
        "season":        season,
        "season_type":   season_type,
        "per_mode":      per_mode,
        "leaders":       leaders,
        "count":         len(leaders),
    }
