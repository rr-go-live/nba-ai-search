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
    def start(s):
        try:
            return int(s.split("-")[0])
        except Exception:
            return 2024
    y0 = max(start(s_from), MIN_YEAR)
    y1 = min(start(s_to),   MAX_YEAR)
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(y0, y1 + 1)][:MAX_SEASONS_SPAN]


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


def _to_row(r: dict) -> dict:
    """Convert raw game-log dict to frontend-safe dict. All values non-None."""
    return {
        "season":     r.get("SEASON") or "",
        "date":       r.get("GAME_DATE") or "",
        "matchup":    r.get("MATCHUP") or "",
        "result":     r.get("WL") or "",
        "min":        str(r.get("MIN") or ""),
        "pts":        _safe_int(r.get("PTS")),
        "reb":        _safe_int(r.get("REB")),
        "ast":        _safe_int(r.get("AST")),
        "stl":        _safe_int(r.get("STL")),
        "blk":        _safe_int(r.get("BLK")),
        "tov":        _safe_int(r.get("TOV")),
        "fg_pct":     round(_safe_float(r.get("FG_PCT")) * 100, 1),
        "fg3_pct":    round(_safe_float(r.get("FG3_PCT")) * 100, 1),
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