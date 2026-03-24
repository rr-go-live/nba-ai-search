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


def _season_year(s: str) -> int:
    """Convert season string '2015-16' → 2015 (start year)."""
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return 0


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
    """
    One season's game log. Tries stats.nba.com first via the cookie session.
    Falls back to Basketball Reference for seasons >= 2025-26 when the NBA
    API returns empty data or raises an exception (e.g. season not yet live).
    """
    try:
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
        if rows:
            return rows
    except Exception as e:
        logger.warning(f"NBA API game log {player_id} {season}: {e}")

    # Fallback: try Basketball Reference for current/future seasons
    if season >= "2025-26" and season_type == "Regular Season":
        return _game_log_bbref(player_id, season)

    return []


def _game_log_bbref(player_id: int, season: str) -> list:
    """Fetch game log from Basketball Reference for the given player/season."""
    import bbref_client

    player = static_players.find_player_by_id(player_id)
    if not player:
        logger.warning(f"BBRef fallback: no static player found for ID {player_id}")
        return []

    name = player.get("full_name", "")
    bbref_id = bbref_client.find_bbref_id(name)
    if not bbref_id:
        logger.warning(f"BBRef fallback: could not resolve BBRef ID for '{name}'")
        return []

    season_end_year = int(season.split("-")[0]) + 1  # "2025-26" → 2026
    return bbref_client.get_game_log(bbref_id, season_end_year)


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
        "fga":     mean("FGA"),    # field goal attempts per game (for donut hover)
        "fg3_pct": pct("FG3_PCT"),
        "fg3a":    mean("FG3A"),   # three-point attempt volume
        "ft_pct":  pct("FT_PCT"),
        "fta":     mean("FTA"),    # free throw attempts per game (for donut hover)
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
    Includes fg3a/fga/fta (attempt volumes per game) for donut hover context.
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
        "blk":      round(_safe_float(r.get("BLK")), 1),
        "fg_pct":   round(_safe_float(r.get("FG_PCT"))  * 100, 1),
        "fga":      round(_safe_float(r.get("FGA")), 1),      # attempts/game for donut hover
        "fg3_pct":  round(_safe_float(r.get("FG3_PCT")) * 100, 1),
        "fg3a":     round(_safe_float(r.get("FG3A")), 1),     # volume context
        "ft_pct":   round(_safe_float(r.get("FT_PCT"))  * 100, 1),
        "fta":      round(_safe_float(r.get("FTA")), 1),      # attempts/game for donut hover
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
    """Player averages + game log vs specific opponent across multiple seasons.
    Always fetches both Regular Season and Playoffs; returns each separately.
    """
    def _fetch_type(stype: str):
        rows_all, splits = [], []
        for season in seasons_between(season_from, season_to):
            try:
                rows     = _game_log(player_id, season, stype)
                filtered = _filter_opponent(rows, opponent_abbr)
                if filtered:
                    s = _avg(filtered)
                    s["season"] = season
                    splits.append(s)
                    rows_all.extend(filtered)
            except Exception as e:
                logger.warning(f"GameLog {season} {stype}: {e}")
        rows_all.sort(key=lambda r: r.get("SEASON", ""), reverse=True)
        return rows_all, splits

    rs_rows, rs_splits = _fetch_type("Regular Season")
    po_rows, po_splits = _fetch_type("Playoffs")

    result = {
        "averages":        _avg(rs_rows),
        "season_splits":   rs_splits,
        "game_log":        [_to_row(r) for r in rs_rows][:MAX_GAME_LOG_ROWS],
        "total_games":     len(rs_rows),
    }

    if po_rows:
        result["playoff_averages"]   = _avg(po_rows)
        result["playoff_splits"]     = po_splits
        result["playoff_game_log"]   = [_to_row(r) for r in po_rows][:MAX_GAME_LOG_ROWS]
        result["playoff_total_games"] = len(po_rows)

    return result


def get_player_career_stats(
    player_id: int,
    season_from: str = None,
    season_to:   str = None,
) -> dict:
    """
    Career season-by-season averages + career totals.

    When season_from / season_to are provided the returned seasons and totals
    are restricted to that range — ideal for queries like "Tatum stats 2022-25".
    Without filters the full career is returned and the NBA API's pre-computed
    career totals are used (most accurate for all-time queries).

    Returns both regular season and playoff data.
    Each season row includes fga/fg3a/fta (attempt volumes) for donut hover.
    team='TOT' means the player was traded that season.
    """
    def _season_year(s: str) -> int:
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return 0

    data = _career_stats_raw(player_id)

    # ── Filter raw rows to the requested season range ─────────────────────────
    has_range = bool(season_from and season_to)
    y0 = _season_year(season_from) if season_from else 0
    y1 = _season_year(season_to)   if season_to   else 9999

    def _filter(rows):
        if not has_range:
            return rows
        return [r for r in rows if y0 <= _season_year(r.get("SEASON_ID", "")) <= y1]

    reg_raw  = _filter(data.get("SeasonTotalsRegularSeason", []))
    post_raw = _filter(data.get("SeasonTotalsPostSeason",    []))

    regular_seasons = [_career_row(r, is_playoffs=False) for r in reg_raw]
    playoff_seasons = [_career_row(r, is_playoffs=True)  for r in post_raw]

    # ── Totals: GP-weighted averages from raw rows ────────────────────────────
    # For full-career queries we could use the API's CareerTotals row, but we
    # always compute from raw rows so the per-game averages are consistent with
    # the per-season rows shown in the table, and fga/fta are always present.
    def _totals_from_raw(rows: list) -> dict:
        if not rows:
            return {}
        total_gp = sum(_safe_int(r.get("GP")) for r in rows)
        if total_gp == 0:
            return {}

        def wp(key):     # weighted per-game
            return round(
                sum(_safe_float(r.get(key)) * _safe_int(r.get("GP")) for r in rows)
                / total_gp, 1
            )
        def wpct(key):   # weighted pct (NBA API stores as 0-1; multiply to 0-100)
            return round(
                sum(_safe_float(r.get(key)) * _safe_int(r.get("GP")) for r in rows)
                / total_gp * 100, 1
            )

        return {
            "gp":      total_gp,
            "pts":     wp("PTS"),
            "reb":     wp("REB"),
            "ast":     wp("AST"),
            "stl":     wp("STL"),
            "blk":     wp("BLK"),
            "fg_pct":  wpct("FG_PCT"),
            "fga":     wp("FGA"),      # attempts/game — drives donut hover
            "fg3_pct": wpct("FG3_PCT"),
            "fg3a":    wp("FG3A"),
            "fg3m":    wp("FG3M"),     # made 3s/game — career high cards
            "ft_pct":  wpct("FT_PCT"),
            "fta":     wp("FTA"),      # attempts/game — drives donut hover
            "ftm":     wp("FTM"),      # made FTs/game — career high cards
        }

    # ── Regular-Season Career Highs — embedded in playercareerstats response ───
    # Each row: STAT, STAT_VALUE, GAME_DATE, VS_TEAM_ABBREVIATION
    # Multiple rows may share the same STAT (tied highs) — keep first occurrence.
    _WANTED_HIGHS = {"PTS", "REB", "AST", "STL", "BLK", "FTM", "FTA", "FG3M"}
    ch_rows = data.get("CareerHighs", [])
    career_highs: dict = {}
    for row in ch_rows:
        stat = (row.get("STAT") or "").upper()
        if stat not in _WANTED_HIGHS or stat in career_highs:
            continue
        career_highs[stat.lower()] = {
            "value":    _safe_int(row.get("STAT_VALUE")),
            "date":     str(row.get("GAME_DATE") or ""),
            "opponent": str(row.get("VS_TEAM_ABBREVIATION") or ""),
        }

    # ── Regular-Season Best (best single-season RS average per stat) ──────────
    # Used for the "Best Regular Season Averages" card on career pages.
    _RS_STAT_MAP = [
        ("pts",  "PTS"),  ("reb",  "REB"),  ("ast",  "AST"),
        ("stl",  "STL"),  ("blk",  "BLK"),  ("ftm",  "FTM"),
        ("fta",  "FTA"),  ("fg3m", "FG3M"),
    ]
    reg_all_raw = data.get("SeasonTotalsRegularSeason", [])
    regular_season_best: dict = {}
    for lk, uk in _RS_STAT_MAP:
        best = max(reg_all_raw, key=lambda r, k=uk: _safe_float(r.get(k, 0)), default=None)
        if best and _safe_float(best.get(uk, 0)) > 0:
            regular_season_best[lk] = {
                "value":  round(_safe_float(best.get(uk)), 1),
                "season": str(best.get("SEASON_ID") or ""),
                "team":   str(best.get("TEAM_ABBREVIATION") or ""),
            }

    # ── Playoff Career Best (best single-season playoff average per stat) ─────
    # SeasonTotalsPostSeason has per-season per-game averages.  We find the
    # peak season for each stat across the full career (not range-filtered)
    # so the "Playoff Best Season" card always shows the true career peak.
    _PO_STAT_MAP = [
        ("pts",  "PTS"),  ("reb",  "REB"),  ("ast",  "AST"),
        ("stl",  "STL"),  ("blk",  "BLK"),  ("ftm",  "FTM"),
        ("fta",  "FTA"),  ("fg3m", "FG3M"),
    ]
    po_all_raw = data.get("SeasonTotalsPostSeason", [])
    playoff_best: dict = {}
    for lk, uk in _PO_STAT_MAP:
        best = max(po_all_raw, key=lambda r, k=uk: _safe_float(r.get(k, 0)), default=None)
        if best and _safe_float(best.get(uk, 0)) > 0:
            raw_season = str(best.get("SEASON_ID") or "")
            # SEASON_ID is like "22009-10" (2 prefix chars) → strip to "2009-10"
            season_label = raw_season[2:] if len(raw_season) > 7 else raw_season
            playoff_best[lk] = {
                "value":  round(_safe_float(best.get(uk)), 1),
                "season": season_label,
                "team":   str(best.get("TEAM_ABBREVIATION") or ""),
            }

    # ── Playoff Career Highs (single-game bests) ──────────────────────────────
    # Uses leaguegamefinder to fetch all career playoff games in ONE API call,
    # then finds the single-game max for each stat — mirrors how RS career_highs
    # work but for playoff games.
    _CH_STAT_MAP = [
        ("pts",  "PTS"),  ("reb",  "REB"),  ("ast",  "AST"),
        ("stl",  "STL"),  ("blk",  "BLK"),  ("ftm",  "FTM"),
        ("fta",  "FTA"),  ("fg3m", "FG3M"),
    ]
    playoff_career_highs: dict = {}
    if po_all_raw:  # only fetch if player has playoff appearances
        try:
            po_games = _fetch_all_player_games(player_id, "Playoffs")
            for lk, uk in _CH_STAT_MAP:
                best_game = max(po_games, key=lambda r, k=uk: _safe_float(r.get(k, 0)), default=None)
                if best_game and _safe_float(best_game.get(uk, 0)) > 0:
                    opp = _extract_opp(best_game.get("MATCHUP", ""))
                    playoff_career_highs[lk] = {
                        "value":    _safe_int(best_game.get(uk)),
                        "date":     str(best_game.get("GAME_DATE") or ""),
                        "opponent": opp,
                    }
        except Exception as e:
            logger.warning(f"playoff career highs {player_id}: {e}")

    result = {
        "seasons":               regular_seasons,
        "playoff_seasons":       playoff_seasons,
        "career_totals":         _totals_from_raw(reg_raw),
        "playoff_totals":        _totals_from_raw(post_raw),
        "career_highs":          career_highs,
        "regular_season_best":   regular_season_best,
        "playoff_best":          playoff_best,
        "playoff_career_highs":  playoff_career_highs,
    }
    # Pass the range back so the frontend can display it correctly
    if season_from:
        result["season_from"] = season_from
    if season_to:
        result["season_to"] = season_to

    return result


def get_player_conditional_stats(
    player_id: int,
    condition_player_ids: list,
    all_active: bool        = True,
    require_opponent: bool  = False,
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
            # For H2H (all_active=True) do NOT pre-filter by opponent_abbr —
            # the game-ID intersection with condition players already captures
            # all matchups correctly, including when the condition player changes
            # teams mid-career (e.g. Jimmy Butler MIA→GSW → filtering by "GSW"
            # for LeBron would yield only 1 game instead of 31).
            # Only apply the opponent filter for inactive splits (all_active=False).
            if opponent_abbr and not all_active:
                rows = _filter_opponent(rows, opponent_abbr)
            # Tag each row with the season so we can build per-season cutoffs later
            for r in rows:
                r["_season"] = season
            primary_rows.extend(rows)
        except Exception as e:
            logger.warning(f"Primary log {season}: {e}")

        for cid in condition_player_ids:
            try:
                crows = _game_log(cid, season, season_type)
                for r in crows:
                    gid = _norm_game_id(r.get("Game_ID"))
                    # Store (game_id, team_abbr, season) so we can later restrict
                    # the inactive split to seasons/periods where they were teammates.
                    matchup = (r.get("MATCHUP") or "").upper()
                    team_abbr = matchup[:3] if len(matchup) >= 3 else ""
                    cond_id_sets[cid].add((gid, team_abbr, season))
            except Exception as e:
                logger.warning(f"Condition player {cid} log {season}: {e}")

    if not primary_rows:
        return {"error": "No game log data found."}

    def _is_teammate(primary_row, cid_set):
        """
        Return True only when the condition player appeared in the same game
        AND was on the same team as the primary player (not an opponent).
        Uses the MATCHUP field to determine each player's team abbreviation.
        Primary MATCHUP: "LAL vs. WAS"  → primary team = "LAL", opp = "WAS"
        Condition stored as (game_id, team_abbr, season) tuples.
        """
        gid = _norm_game_id(primary_row.get("Game_ID"))
        # Extract primary player's opponent abbreviation from their MATCHUP
        primary_matchup = (primary_row.get("MATCHUP") or "").upper()
        # MATCHUP is "TEAM vs. OPP" or "TEAM @ OPP" — opponent is last 3 chars
        opp_abbr = primary_matchup[-3:] if len(primary_matchup) >= 3 else ""
        # A condition entry (gid, team, season) is a teammate if same game AND their team != opponent
        return any(g == gid and t != opp_abbr for g, t, _s in cid_set)

    # Build flat game_id-only sets for efficient "played that day" checks
    flat_sets = {cid: {g for g, _, _s in s} for cid, s in cond_id_sets.items()}

    if all_active:
        def _all_played(row):
            gid = _norm_game_id(row.get("Game_ID"))
            return all(gid in flat_sets[cid] for cid in condition_player_ids)

        if require_opponent:
            # H2H mode: condition players must be on the OPPOSING team.
            # Excludes shared-team games (e.g. Simmons/Embiid on PHI, Klay/Curry on GSW)
            # so only true opponent matchups are counted.
            def _match_fn(row):
                return _all_played(row) and all(
                    not _is_teammate(row, cond_id_sets[cid])
                    for cid in condition_player_ids
                )
        else:
            # Teammate "with" mode: condition players just need to have played that game
            # (same team or not — caller is asking for games where both were active).
            _match_fn = _all_played

        matched  = [r for r in primary_rows if _match_fn(r)]
        opposite = [r for r in primary_rows if not _all_played(r)]
    else:
        # Complement of union: games where NONE of the condition players played at all
        # (inactive means didn't play, period).
        #
        # BUT we must restrict to seasons/periods when the condition player(s) were
        # actually on the primary player's team.  After a trade the condition player
        # gets entirely new game_ids (for their new team), so their former-team games
        # are never in union_ids — causing the primary player's post-trade games to
        # appear falsely as "without <condition player>".
        #
        # Fix: for each condition player compute
        #   • valid_seasons   — seasons with ≥1 teammate game
        #   • last_tm_gid[s]  — last game_id in season s where they were a teammate
        # Then restrict primary_rows to those seasons, and within mid-season-trade
        # seasons, only games up to that cutoff game_id.

        # Detect seasons where the condition player played for MULTIPLE teams
        # (i.e. a mid-season trade).  Only in those seasons do we apply date
        # cutoffs — an injury absence at the start/end of a season must NOT
        # shrink the "without" window.
        #
        # Example:
        #   Pau Gasol 2007-08 → played for MEM then LAL (multi-team) → apply first/last
        #   AD 2025-26        → played for LAL then WAS (multi-team) → apply first/last
        #   KD 2014-15        → played for OKC only (single team, just injured)
        #                       → NO cutoff; Westbrook's full 2014-15 season is valid
        cond_multi_team: dict = {}   # cid → set of seasons with a mid-season team change
        # cond_season_teams[cid][season] = set of team abbrs the condition player
        # played for in that season.  Used to confirm the PRIMARY player was on the
        # same team (handles cases like Luka on DAL pre-trade in 2024-25 — those
        # DAL games must NOT appear as "LeBron inactive" since they weren't teammates).
        cond_season_teams: dict = {cid: {} for cid in condition_player_ids}
        for cid in condition_player_ids:
            season_teams: dict = {}
            for _g, t, s in cond_id_sets[cid]:
                season_teams.setdefault(s, set()).add(t)
                cond_season_teams[cid].setdefault(s, set()).add(t)
            cond_multi_team[cid] = {s for s, teams in season_teams.items() if len(teams) > 1}

        # Per-condition-player: first AND last ISO date per season where they
        # were a teammate.  Only relevant for multi-team seasons (trades).
        cond_first_tm = {cid: {} for cid in condition_player_ids}
        cond_last_tm  = {cid: {} for cid in condition_player_ids}
        for cid in condition_player_ids:
            for row in primary_rows:
                if _is_teammate(row, cond_id_sets[cid]):
                    s    = row.get("_season", "")
                    # Skip seasons where condition player stayed on one team —
                    # we don't need date bounds there.
                    if s not in cond_multi_team[cid]:
                        continue
                    iso  = _parse_date_sort(row.get("GAME_DATE", ""))
                    prev_first = cond_first_tm[cid].get(s)
                    prev_last  = cond_last_tm[cid].get(s)
                    if not prev_first or iso < prev_first:
                        cond_first_tm[cid][s] = iso
                    if not prev_last or iso > prev_last:
                        cond_last_tm[cid][s] = iso

        # Valid seasons = seasons where ALL condition players were ever teammates.
        # Build a per-season set of seasons where _is_teammate fired at least once.
        cond_valid_seasons: dict = {cid: set() for cid in condition_player_ids}
        for row in primary_rows:
            s = row.get("_season", "")
            for cid in condition_player_ids:
                if s not in cond_valid_seasons[cid] and _is_teammate(row, cond_id_sets[cid]):
                    cond_valid_seasons[cid].add(s)

        all_valid_seasons: set = set()
        for i, cid in enumerate(condition_player_ids):
            all_valid_seasons = cond_valid_seasons[cid] if i == 0 else all_valid_seasons & cond_valid_seasons[cid]

        def _in_valid_range(row) -> bool:
            s = row.get("_season", "")
            if s not in all_valid_seasons:
                return False
            # Guard: primary player must have been on the same team as the condition
            # player in this game.  This catches the case where the PRIMARY player
            # changed teams mid-season (e.g. Luka DAL→LAL in 2024-25): his pre-trade
            # DAL games must not appear as "LeBron inactive" splits even though 2024-25
            # is in all_valid_seasons (because LAL-era games exist that season).
            primary_team = (row.get("MATCHUP") or "")[:3].upper().strip()
            if primary_team:
                for cid in condition_player_ids:
                    cid_teams = cond_season_teams[cid].get(s, set())
                    if primary_team not in cid_teams:
                        return False  # different teams — not a valid teammate split
            iso = _parse_date_sort(row.get("GAME_DATE", ""))
            for cid in condition_player_ids:
                # Only enforce date bounds in mid-season-trade seasons
                if s not in cond_multi_team[cid]:
                    continue
                first = cond_first_tm[cid].get(s)
                last  = cond_last_tm[cid].get(s)
                if first and iso < first:
                    return False
                if last and iso > last:
                    return False
            return True

        primary_rows_valid = [r for r in primary_rows if _in_valid_range(r)]

        # Build union of condition-player game_ids that fall in the valid window.
        # For multi-team seasons, restrict to the teammate portion only.
        union_ids: set = set()
        for cid in condition_player_ids:
            for g, _t, s in cond_id_sets[cid]:
                if s not in all_valid_seasons:
                    continue
                if s in cond_multi_team[cid]:
                    # Only include teammate-era games (before cutoff)
                    first = cond_first_tm[cid].get(s)
                    last  = cond_last_tm[cid].get(s)
                    if not first or not last:
                        continue
                    # Approximate: include any game_id from this season from
                    # cond_id_sets — non-matching game_ids won't appear in
                    # primary_rows_valid anyway (date-filtered out).
                union_ids.add(g)

        matched  = [r for r in primary_rows_valid if _norm_game_id(r.get("Game_ID")) not in union_ids]
        opposite = [r for r in primary_rows_valid if _norm_game_id(r.get("Game_ID")) in union_ids]

    # Sort both logs newest-first before slicing.
    # GAME_DATE from NBA API is "OCT 22, 2024" — must parse to YYYY-MM-DD first.
    def _sort_key(r):
        return _parse_date_sort(r.get("GAME_DATE", "") or "")
    matched.sort( key=_sort_key, reverse=True)
    opposite.sort(key=_sort_key, reverse=True)

    # Compute season_range from the actual valid seasons in the game data,
    # not from the LLM's guess — e.g. "2012-13 TO 2024-25" for Giannis/Middleton.
    all_seasons_present = sorted({
        r.get("_season", "") for r in (matched + opposite) if r.get("_season")
    })
    if all_seasons_present:
        computed_range = (all_seasons_present[0] + " TO " + all_seasons_present[-1]
                         if all_seasons_present[0] != all_seasons_present[-1]
                         else all_seasons_present[0])
    else:
        computed_range = None

    return {
        "primary_averages":    _avg(matched),
        "comparison_averages": _avg(opposite),
        "primary_games":       len(matched),
        "comparison_games":    len(opposite),
        "primary_log":         [_to_row(r) for r in matched][:MAX_GAME_LOG_ROWS],
        "comparison_log":      [_to_row(r) for r in opposite][:MAX_GAME_LOG_ROWS],
        "all_active":          all_active,
        "total_games":         len(primary_rows),
        "computed_season_range": computed_range,
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
        series_round = (cfg.get("series_round") or "").strip().lower()

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

            # If a specific round is requested (e.g. "finals"), filter
            # series data to only that round and override the averages.
            all_series = series_data.get("series", []) if series_data else []
            total_games = series_data.get("total_games", 0) if series_data else 0
            if series_round and all_series:
                # Normalise round label for matching: "finals"→"Finals", "r1"→"R1", etc.
                label_map = {"finals": "Finals", "cf": "CF", "r1": "R1", "r2": "R2",
                             "conf finals": "CF", "conference finals": "CF"}
                target = label_map.get(series_round, series_round.title())
                filtered_series = [s for s in all_series if s.get("round_label") == target]
                if filtered_series:
                    all_series = filtered_series
                    total_games = sum(s.get("games", 0) for s in filtered_series)
                    # Override averages with the specific series' stats
                    if len(filtered_series) == 1:
                        avgs = dict(filtered_series[0].get("averages", avgs))
                        avgs["gp"] = total_games
                    else:
                        # Multiple matching series (edge case) — GP-weighted avg of series averages
                        stat_keys_avg = ["pts", "reb", "ast", "stl", "blk", "tov",
                                         "fg_pct", "fga", "fg3_pct", "fg3a", "ft_pct", "fta"]
                        merged: dict = {"gp": total_games}
                        if total_games > 0:
                            for k in stat_keys_avg:
                                merged[k] = round(
                                    sum(s["averages"].get(k, 0) * s.get("games", 0)
                                        for s in filtered_series if "averages" in s)
                                    / total_games, 1
                                )
                        avgs = merged

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
                "series":       all_series,
                "playoff_games": total_games,
            })
        except Exception as e:
            logger.warning(f"get_multi_player_stats player {pid}: {e}")
            results.append({"player_id": pid, "label": label, "error": str(e)})

    return {"players": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# H2H matrix
# ─────────────────────────────────────────────────────────────────────────────

def _season_id_to_year(season_id) -> int:
    """Convert NBA season_id to start year. Handles '22015', '2015-16', 2015."""
    s = str(season_id).strip()
    if "-" in s:
        try:
            return int(s.split("-")[0])
        except Exception:
            return 0
    # Numeric format: "22015" (regular season) or "42015" (playoffs)
    # Last 4 digits are the start year.
    if len(s) >= 4:
        try:
            return int(s[-4:])
        except Exception:
            return 0
    return 0


def _fetch_all_player_games(player_id: int, season_type: str = "Regular Season") -> list:
    """
    Fetch every career game for a player in one API call via leaguegamefinder.
    Returns list of dicts with at least: SEASON_ID, TEAM_ABBREVIATION, MATCHUP, WL.
    """
    st_param = "Playoffs" if season_type == "Playoffs" else "Regular Season"
    try:
        data = nba_http.get_parsed("leaguegamefinder", {
            "PlayerOrTeam": "P",
            "PlayerID":     player_id,
            "LeagueID":     "00",
            "SeasonType":   st_param,
        })
        return data.get("LeagueGameFinderResults", [])
    except Exception as e:
        logger.warning(f"leaguegamefinder player {player_id}: {e}")
        return []


def get_h2h_matrix(players: list) -> dict:
    """
    Compute head-to-head records for every pair of players.

    players — list of dicts, each with:
        player_id, name, season_from, season_to, season_type

    Algorithm (one LeagueGameFinder call per player):
      1. Fetch all career games for every player (MATCHUP, WL, SEASON_ID).
      2. For each season in player B's games, note which team B was on.
      3. For each game in player A's log that season, if B's team is in the
         MATCHUP string, record the WL as an H2H result for (A, B).
      4. B's record vs A is the mirror (B wins = A losses).

    Returns:
        {"pairs": [
            {"player_a_id": .., "player_a_name": .., "player_b_id": .., "player_b_name": ..,
             "a_wins": .., "a_losses": .., "b_wins": .., "b_losses": .., "gp": ..},
            ...
        ]}
    """
    if len(players) < 2:
        return {"pairs": []}

    # Determine season_type from first player (assume homogeneous for compare query)
    season_type = players[0].get("season_type", "Regular Season")

    # ── Step 1: fetch all games per player (one API call each) ───────────────
    all_games:       dict[int, list] = {}   # player_id → list of game dicts
    team_by_season:  dict[int, dict] = {}   # player_id → {season_id → team_abbr}

    for cfg in players:
        pid = int(cfg["player_id"])
        time.sleep(NBA_API_DELAY)
        games = _fetch_all_player_games(pid, season_type)
        all_games[pid] = games

        # Build season_id → team_abbr map (last team seen wins for TOT seasons)
        tmap: dict = {}
        for g in games:
            sid  = str(g.get("SEASON_ID", ""))
            team = str(g.get("TEAM_ABBREVIATION", ""))
            if sid and team and team != "TOT":
                tmap[sid] = team
        team_by_season[pid] = tmap

    # ── Step 2: compute H2H for each unique pair ─────────────────────────────
    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            cfg_a = players[i]
            cfg_b = players[j]
            pid_a = int(cfg_a["player_id"])
            pid_b = int(cfg_b["player_id"])

            y_from = _season_year(cfg_a.get("season_from", "1996-97"))
            y_to   = _season_year(cfg_a.get("season_to",   DEFAULT_SEASON))

            tmap_b = team_by_season.get(pid_b, {})

            a_wins = a_losses = 0
            # For series tracking: group games by season_id → {wins, losses}
            series_by_season: dict = {}
            for g in all_games.get(pid_a, []):
                sid    = str(g.get("SEASON_ID", ""))
                s_year = _season_id_to_year(sid)
                if not (y_from <= s_year <= y_to):
                    continue
                b_team = tmap_b.get(sid, "")
                if not b_team:
                    continue
                # Skip games where A and B were teammates — if A's team for this
                # specific game matches B's season team, they played together, not
                # against each other (e.g. Curry+Durant both on GSW 2016-19).
                a_game_team = str(g.get("TEAM_ABBREVIATION", ""))
                if a_game_team == b_team:
                    continue
                if b_team in str(g.get("MATCHUP", "")):
                    wl = str(g.get("WL", ""))
                    if wl == "W":
                        a_wins += 1
                        series_by_season.setdefault(sid, {"w": 0, "l": 0})["w"] += 1
                    elif wl == "L":
                        a_losses += 1
                        series_by_season.setdefault(sid, {"w": 0, "l": 0})["l"] += 1

            # Series records (only meaningful for playoffs — each season = one series)
            a_series_wins = a_series_losses = 0
            for rec in series_by_season.values():
                if rec["w"] > rec["l"]:
                    a_series_wins += 1
                else:
                    a_series_losses += 1

            pairs.append({
                "player_a_id":   pid_a,
                "player_a_name": cfg_a.get("name", ""),
                "player_b_id":   pid_b,
                "player_b_name": cfg_b.get("name", ""),
                "a_wins":          a_wins,
                "a_losses":        a_losses,
                "b_wins":          a_losses,   # mirror: B won the games A lost
                "b_losses":        a_wins,
                "gp":              a_wins + a_losses,
                "a_series_wins":   a_series_wins,
                "a_series_losses": a_series_losses,
                "b_series_wins":   a_series_losses,
                "b_series_losses": a_series_wins,
                "is_playoffs":     season_type == "Playoffs",
            })

    return {"pairs": pairs}


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


def _season_fallback_chain(season: str, depth: int = 4) -> list:
    """Return [requested, previous...] canonical seasons for fallback queries."""
    canon = parse_season(season)
    try:
        start = int(canon.split('-')[0])
    except Exception:
        start = int(DEFAULT_SEASON.split('-')[0])
    out = []
    for i in range(max(depth, 1)):
        y = start - i
        if y < MIN_YEAR:
            break
        out.append(f"{y}-{str(y + 1)[-2:]}")
    return out


def _get_leaderboard_leaguedash(
    season: str,
    season_type: str,
    per_mode: str,
    category: str,
    top_n: int,
    min_gp: int = 0,
) -> list:
    """
    Fallback leaderboard using LeagueDashPlayerStats (nba_api endpoint).
    Returns rows in leagueleaders column format so get_leaderboard() can
    process them without branching.  Used when leagueleaders returns nothing
    for recent seasons (endpoint often lags behind the live season).
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats

        per_mode_map  = {"PerGame": "PerGame", "Totals": "Totals"}
        nba_per_mode  = per_mode_map.get(per_mode, "PerGame")

        season_type_map = {"Regular Season": "Regular Season", "Playoffs": "Playoffs"}
        nba_season_type = season_type_map.get(season_type, "Regular Season")

        time.sleep(NBA_API_DELAY)
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=nba_season_type,
            per_mode_detailed=nba_per_mode,
        ).get_data_frames()[0]

        if df.empty:
            return []

        col = category
        if category == "EFF":
            df["EFF"] = (
                df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"]
                - (df["FGA"] - df["FGM"]) - (df["FTA"] - df["FTM"]) - df["TOV"]
            )
        elif col not in df.columns:
            logger.warning(f"LeagueDash fallback: column '{col}' not in dataframe")
            return []

        # Apply GP qualifier for per-game stats to avoid small-sample outliers
        if min_gp > 0 and "GP" in df.columns:
            df = df[df["GP"] >= min_gp]
            if df.empty:
                logger.warning(f"LeagueDash fallback: no players with GP >= {min_gp} for {season} {category}")
                return []

        # Minimum attempts qualifier for percentage stats (mirrors NBA official rules)
        # FG3_PCT: ≥4.0 three-point attempts per game (PerGame) or ≥328 total (Totals)
        # FG_PCT:  ≥3.0 field-goal attempts per game or ≥246 total
        # FT_PCT:  ≥1.5 free-throw attempts per game or ≥82 total
        PCT_ATTEMPTS = {
            "FG3_PCT": ("FG3A", 4.0, 328),
            "FG_PCT":  ("FGA",  3.0, 246),
            "FT_PCT":  ("FTA",  1.5,  82),
        }
        if category in PCT_ATTEMPTS:
            att_col, pg_thresh, tot_thresh = PCT_ATTEMPTS[category]
            thresh = pg_thresh if per_mode == "PerGame" else tot_thresh
            if att_col in df.columns:
                before = len(df)
                df = df[df[att_col] >= thresh]
                logger.info(f"LeagueDash {category}: filtered {before - len(df)} players below {thresh} {att_col}")
                if df.empty:
                    logger.warning(f"LeagueDash fallback: no qualified players for {category} {season}")
                    return []

        df_sorted = df.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)

        # Lazy import for name-based player ID lookup
        try:
            from nba_api.stats.static import players as nba_players_static
            _player_name_to_id = {
                p["full_name"].lower(): p["id"]
                for p in nba_players_static.get_players()
            }
        except Exception:
            _player_name_to_id = {}

        rows = []
        for _, r in df_sorted.iterrows():
            raw_id = r.get("PLAYER_ID")
            # pandas NaN → 0 via _safe_int; try name lookup as fallback
            pid = int(raw_id) if (raw_id is not None and raw_id == raw_id) else 0
            if not pid:
                name = str(r.get("PLAYER_NAME", "") or "").lower()
                pid = _player_name_to_id.get(name, 0)
            rows.append({
                "PLAYER_ID": pid,
                "PLAYER":    r.get("PLAYER_NAME", ""),
                "TEAM":      r.get("TEAM_ABBREVIATION", ""),
                "GP":        r.get("GP"),
                "PTS":       r.get("PTS"),
                "REB":       r.get("REB"),
                "AST":       r.get("AST"),
                "STL":       r.get("STL"),
                "BLK":       r.get("BLK"),
                "FG_PCT":    r.get("FG_PCT"),
                "FG3_PCT":   r.get("FG3_PCT"),
                "FG3A":      r.get("FG3A"),
                col:         r.get(col),
            })
        logger.info(f"LeagueDash fallback: {len(rows)} rows for {season} {category}")
        return rows

    except Exception as e:
        logger.warning(f"LeagueDash leaderboard fallback failed ({season}): {e}")
        return []


def _leaderboard_bbref(category: str, season: str, per_mode: str, top_n: int) -> dict:
    """
    BBRef fallback for get_leaderboard() when both the NBA API and
    LeagueDashPlayerStats return no data for the current season.
    """
    import bbref_client
    season_end_year = int(season.split("-")[0]) + 1   # "2025-26" → 2026

    logger.info(f"Falling back to BBRef for leaderboard {category} {season}")
    bbref_rows = bbref_client.get_leaderboard(category, season_end_year, per_mode, top_n)

    if not bbref_rows:
        return {
            "error": f"No leaderboard data available for {season} (NBA API and Basketball Reference both returned empty).",
            "leaders": [],
            "season": season,
            "season_requested": season,
        }

    leaders = []
    for rank, r in enumerate(bbref_rows, 1):
        name = r["name"]
        pid  = 0
        matches = static_players.find_players_by_full_name(name)
        if matches:
            pid = matches[0]["id"]
        leaders.append({
            "rank":         rank,
            "player_id":    pid,
            "name":         name,
            "team":         r.get("team", ""),
            "headshot_url": headshot_url(pid) if pid else "",
            "gp":           r.get("gp", 0),
            "pts":          round(r.get("pts", 0.0), 1),
            "reb":          round(r.get("reb", 0.0), 1),
            "ast":          round(r.get("ast", 0.0), 1),
            "stl":          round(r.get("stl", 0.0), 1),
            "blk":          round(r.get("blk", 0.0), 1),
            "fg_pct":       r.get("fg_pct", 0.0),
            "fg3_pct":      r.get("fg3_pct", 0.0),
            "stat_value":   round(r.get("stat_value", 0.0), 1),
        })

    return {
        "stat_category":       category,
        "stat_label":          category.lower(),
        "season":              season,
        "season_requested":    season,
        "season_fallback_used": False,
        "season_type":         "Regular Season",
        "per_mode":            per_mode,
        "leaders":             leaders,
        "count":               len(leaders),
        "_source":             "bbref",
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

    # Percentage stats (0.0–1.0 from the API) are multiplied ×100 before
    # being stored in stat_value so the frontend always receives 0–100 values.
    PCT_STATS = {"FG_PCT", "FG3_PCT", "FT_PCT"}
    is_percentage = category in PCT_STATS

    top_n = min(max(top_n, 1), 25)

    # GP qualifier: for per-game stats enforce a minimum so small-sample
    # players don't top the list.  NBA's official threshold is ~58 games
    # (70 % of 82) for Regular Season.  Playoffs have no GP minimum because
    # the max possible games in a full run is only ~28, and the leagueleaders
    # endpoint already applies its own qualifier for Playoffs.
    if per_mode == "PerGame" and season_type != "Playoffs":
        min_gp = 58
    else:
        min_gp = 0

    requested_season = parse_season(season)
    season_chain = _season_fallback_chain(requested_season, depth=5)
    used_season = requested_season
    raw_rows = []

    # leagueleaders with SeasonType=Playoffs uses an internal qualifier that
    # often excludes deep-playoff teams (e.g. the eventual champion).  Skip it
    # entirely for Playoffs and go straight to LeagueDash which is accurate.
    if season_type != "Playoffs":
        for cand in season_chain:
            try:
                time.sleep(NBA_API_DELAY)
                data = nba_http.get_parsed(
                    "leagueleaders",
                    params={
                        "LeagueID":    "00",
                        "PerMode":     per_mode,
                        "Scope":       "S",
                        "Season":      cand,
                        "SeasonType":  season_type,
                        "StatCategory": category,
                    },
                    timeout=NBA_API_TIMEOUT,
                )
            except Exception as e:
                logger.error(f"leagueleaders {category} {cand}: {e}")
                continue

            rows = data.get("LeagueLeaders", [])
            if rows:
                # Apply GP qualifier for per-game stats
                if min_gp > 0:
                    rows = [r for r in rows if _safe_int(r.get("GP")) >= min_gp]
                raw_rows = rows[:top_n]
                if raw_rows:
                    used_season = cand
                    break

    # LeagueDash fallback: fires any time leagueleaders returns empty/filtered-out
    # data, including historical seasons where the GP filter may have removed all
    # rows due to key mismatches, plus Playoffs (always) and current seasons.
    if not raw_rows:
        logger.info(f"Trying LeagueDash fallback for {requested_season} {category} {season_type} leaderboard")
        raw_rows = _get_leaderboard_leaguedash(
            season=requested_season,
            season_type=season_type,
            per_mode=per_mode,
            category=category,
            top_n=top_n,
            min_gp=min_gp,
        )
        if raw_rows:
            used_season = requested_season

    if not raw_rows:
        if season_type == "Regular Season":
            return _leaderboard_bbref(category, requested_season, per_mode, top_n)
        return {
            "error": f"No leaderboard data available for {requested_season} or fallback seasons.",
            "leaders": [],
            "season_requested": requested_season,
            "season": requested_season,
            "season_fallback_used": False,
        }

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
            "fg3a":        round(_safe_float(r.get("FG3A")), 1),
            # Percentage stats arrive as 0.0–1.0; multiply ×100 so the frontend
            # always receives a 0–100 value and can append "%" directly.
            "stat_value":  round(
                _safe_float(r.get(category, r.get("PTS"))) * (100 if is_percentage else 1),
                1
            ),
        })

    return {
        "stat_category":  category,
        "stat_label":     stat,
        "season":         used_season,
        "season_requested": requested_season,
        "season_fallback_used": used_season != requested_season,
        "season_options": season_chain,
        "season_type":    season_type,
        "per_mode":       per_mode,
        "min_gp":         min_gp,
        "is_percentage":  is_percentage,
        "leaders":        leaders,
        "count":          len(leaders),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Player Awards / Accolades
# ─────────────────────────────────────────────────────────────────────────────

def get_player_awards(player_id: int) -> list:
    """Fetch and aggregate player career awards into a badge list.

    Returns a list of dicts ordered by prestige:
        [{"label": "4× MVP", "tier": "gold"}, ...]

    tier values:
        'gold'   — MVP, FMVP, Champion, 1st All-NBA
        'silver' — 2nd All-NBA
        'bronze' — 3rd All-NBA
        'blue'   — All-Star, DPOY, All-Defensive
        'green'  — ROY
        'orange' — 6MOY, MIP
    """
    try:
        time.sleep(NBA_API_DELAY)
        data = nba_http.get_parsed(
            "playerawards",
            params={"PlayerID": player_id},
            timeout=10,
        )
        rows = data.get("PlayerAwards", [])
    except Exception as e:
        logger.warning(f"playerawards {player_id}: {e}")
        return []

    if not rows:
        return []

    mvp = fmvp = champ = allnba1 = allnba2 = allnba3 = 0
    allstar = dpoy = alldef1 = alldef2 = roy = smoy = mip = 0

    for row in rows:
        desc     = (row.get("DESCRIPTION") or "").strip()
        team_num = str(row.get("ALL_NBA_TEAM_NUMBER") or "").strip()

        if desc == "NBA Most Valuable Player":
            mvp += 1
        elif desc == "NBA Finals Most Valuable Player":
            fmvp += 1
        elif desc == "NBA Champion":
            champ += 1
        elif desc == "All-NBA":
            if   team_num == "1": allnba1 += 1
            elif team_num == "2": allnba2 += 1
            elif team_num == "3": allnba3 += 1
        elif desc == "NBA All-Star":
            allstar += 1
        elif "All-Defensive" in desc:
            if team_num == "1": alldef1 += 1
            else:               alldef2 += 1
        elif "Defensive Player" in desc:
            dpoy += 1
        elif "Rookie of the Year" in desc:
            roy += 1
        elif "Sixth Man" in desc or "6th Man" in desc:
            smoy += 1
        elif "Most Improved" in desc:
            mip += 1

    badges = []
    def _add(label, tier):
        badges.append({"label": label, "tier": tier})

    def _n(n, singular, plural=None):
        """'4× MVP' or '1× MVP'."""
        return f"{n}× {plural or singular}"

    if mvp:     _add(_n(mvp, "MVP"),            "gold")
    if fmvp:    _add(_n(fmvp, "FMVP"),          "gold")
    if champ:   _add(_n(champ, "Champ"),         "gold")
    if allnba1: _add(_n(allnba1, "1st All-NBA"), "gold")
    if allnba2: _add(_n(allnba2, "2nd All-NBA"), "silver")
    if allnba3: _add(_n(allnba3, "3rd All-NBA"), "bronze")
    if allstar: _add(_n(allstar, "All-Star"),    "blue")
    if dpoy:    _add(_n(dpoy, "DPOY"),           "blue")
    if alldef1: _add(_n(alldef1, "All-Def 1st"), "blue")
    if alldef2: _add(_n(alldef2, "All-Def 2nd"), "blue")
    if roy:     _add("ROY",                      "green")
    if smoy:    _add(_n(smoy, "6MOY"),           "orange")
    if mip:     _add(_n(mip, "MIP"),             "orange")

    return badges
