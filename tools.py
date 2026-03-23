"""
tools.py
========
PURPOSE: Claude tool schemas + executor for the NBA Stats Explorer.

TOOLS:
  search_player        — resolve player name/nickname → player_id + headshot
  search_team          — resolve team name → abbreviation
  get_stats_vs_team    — player stats against a specific opponent across seasons
  get_conditional_stats — stats split by condition players active/inactive
                          covers: H2H, single inactive, multi-player inactive
  get_career_stats     — full career season-by-season + playoff breakdown

TOOL ROUTING GUIDE (for Claude):
  ┌─────────────────────────────────┬────────────────────────────┐
  │ Query pattern                   │ Tool                       │
  ├─────────────────────────────────┼────────────────────────────┤
  │ "X vs [team] since YYYY"        │ get_stats_vs_team          │
  │ "X H2H with Y" / "X vs Y H2H"  │ get_conditional_stats      │
  │                                 │   all_active=True          │
  │ "X when Y inactive"             │ get_conditional_stats      │
  │                                 │   all_active=False, 1 id   │
  │ "X when Y and Z inactive"       │ get_conditional_stats      │
  │                                 │   all_active=False, 2 ids  │
  │ "X career stats"                │ get_career_stats           │
  └─────────────────────────────────┴────────────────────────────┘
"""

import logging
import nba_stats_client as nba

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_player",
        "description": (
            "Look up an NBA player by name or nickname → returns player_id and headshot_url. "
            "Call this FIRST before any stat query involving a player. "
            "Handles nicknames: LeBron, Steph, KD, Giannis, Joker, Wemby, Tatum, Brown, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Player name or nickname"}
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_team",
        "description": (
            "Look up an NBA team → returns 3-letter abbreviation used for opponent filtering. "
            "Handles: Celtics, Dubs, Mavs, Cavs, Clips, Sixers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Team name, city, nickname, or abbreviation"}
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_stats_vs_team",
        "description": (
            "Player per-game averages + game log against a specific opponent across seasons.\n"
            "Use for: 'LeBron vs Celtics since 2015', 'Curry career vs Clippers', "
            "'Giannis vs Heat in playoffs'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id":     {"type": "integer", "description": "NBA player ID"},
                "opponent_abbr": {"type": "string",  "description": "3-letter team abbr e.g. 'BOS'"},
                "season_from":   {"type": "string",  "description": "Start season e.g. '2015-16'"},
                "season_to":     {"type": "string",  "description": "End season e.g. '2024-25'"},
                "season_type":   {"type": "string",  "description": "'Regular Season' or 'Playoffs'"},
            },
            "required": ["player_id", "opponent_abbr"],
        },
    },
    {
        "name": "get_conditional_stats",
        "description": (
            "Stats split by whether condition players are ACTIVE or INACTIVE in the same game.\n\n"
            "H2H (Head-to-Head) — all_active=true:\n"
            "  'LeBron vs Curry H2H' or 'LeBron head to head with Curry'\n"
            "  → LeBron's stats in games where Curry ALSO played.\n"
            "  → Set condition_player_ids=[curry_id], all_active=true.\n"
            "  → Also set opponent_abbr to the other player's team.\n"
            "  → Call this TWICE (once per player) to get both sides of H2H.\n\n"
            "Single inactive teammate — all_active=false:\n"
            "  'Tatum stats when Jaylen Brown is inactive'\n"
            "  → condition_player_ids=[brown_id], all_active=false.\n\n"
            "Multiple inactive teammates — all_active=false, multiple IDs:\n"
            "  'Tatum stats when Brown AND Porzingis are both inactive'\n"
            "  → condition_player_ids=[brown_id, porzingis_id], all_active=false.\n"
            "  → Finds games where NEITHER player appeared (complement of union).\n\n"
            "Returns primary_averages (the queried condition) and comparison_averages "
            "(the opposite condition) for side-by-side display."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id":            {"type": "integer", "description": "Primary player ID"},
                "condition_player_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of condition player IDs (1 for single, 2+ for multi-inactive)"
                },
                "all_active":           {"type": "boolean", "description": "true=H2H/all active, false=inactive split"},
                "season_from":          {"type": "string",  "description": "Start season"},
                "season_to":            {"type": "string",  "description": "End season"},
                "opponent_abbr":        {"type": "string",  "description": "Optional opponent team filter — do NOT use for H2H (all_active=true); only use for inactive splits (all_active=false)."},
                "season_type":          {"type": "string",  "description": "'Regular Season' or 'Playoffs'"},
            },
            "required": ["player_id", "condition_player_ids", "all_active"],
        },
    },
    {
        "name": "get_career_stats",
        "description": (
            "Season-by-season regular season + playoff averages for a player.\n"
            "Each season row includes fga/fg3a/fta (attempts per game) for donut hover.\n"
            "Pass season_from + season_to to restrict to a specific range "
            "(e.g. 'Tatum stats 2022-2025' → season_from='2022-23', season_to='2024-25').\n"
            "Omit both for full career. "
            "Use for: 'Tatum career stats', 'Curry season-by-season', "
            "'Tatum 2022 to 2025', 'LeBron last 3 seasons'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id":   {"type": "integer", "description": "NBA player ID"},
                "season_from": {"type": "string",  "description": "First season of range e.g. '2022-23'. Omit for full career."},
                "season_to":   {"type": "string",  "description": "Last season of range e.g. '2024-25'. Omit for full career."},
                "season_type": {"type": "string",  "description": "'Regular Season' (default) or 'Playoffs'"},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "get_multi_player_stats",
        "description": (
            "Compare 2–6 players side by side, each with their own season range and type.\n\n"
            "Use for:\n"
            "  • 'Kobe 2010 playoffs vs Tatum 2024 playoffs'\n"
            "  • 'Compare LeBron, Curry, Durant, Giannis career stats'\n"
            "  • 'Jordan vs LeBron prime years comparison'\n"
            "  • Any query asking to compare multiple players\n\n"
            "Each player config specifies their own season_from/season_to/season_type so "
            "historical eras can be compared directly (e.g. Kobe 2009-10 Playoffs vs Tatum 2023-24 Playoffs).\n"
            "Works for ALL players including retired legends (Jordan, Kobe, Shaq, Bird, Magic, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "players": {
                    "type": "array",
                    "description": "List of player configs, one per player (2-6 players). Each entry MUST include player_id (integer), season_from (string e.g. '2003-04'), season_to (string), season_type (string), label (string).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_id":   {"type": "integer", "description": "NBA player ID — must be a plain integer"},
                            "season_from": {"type": "string",  "description": "Start season e.g. '2003-04'"},
                            "season_to":   {"type": "string",  "description": "End season e.g. '2025-26'"},
                            "season_type": {"type": "string",  "description": "'Regular Season' or 'Playoffs'"},
                            "label":       {"type": "string",  "description": "Display label e.g. 'LeBron Career'"},
                        },
                        # No nested required — Gemini rejects calls with required inside items.
                        # The description above and system prompt enforce the fields instead.
                    },
                },
            },
            "required": ["players"],
        },
    },
    {
        "name": "get_leaderboard",
        "description": (
            "League-wide stat leaderboard — top N players ranked by a stat category.\n\n"
            "Use for:\n"
            "  • 'Who are the top scorers this season?'\n"
            "  • 'Leaderboard of players with most active points'\n"
            "  • 'Most rebounds per game leaders'\n"
            "  • 'Best FG% shooters this year'\n"
            "  • '2009-10 season scoring leaders' (historical seasons work too)\n\n"
            "Stat values: pts, reb, ast, stl, blk, fg3m, eff, min\n"
            "Use per_mode='PerGame' for averages, 'Totals' for cumulative season totals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stat":        {"type": "string",  "description": "Stat to rank by: pts, reb, ast, stl, blk, fg3m, eff, min"},
                "season":      {"type": "string",  "description": "Season e.g. '2025-26'"},
                "season_type": {"type": "string",  "description": "'Regular Season' or 'Playoffs'"},
                "per_mode":    {"type": "string",  "description": "'PerGame' (default) or 'Totals'"},
                "top_n":       {"type": "integer", "description": "How many players to return (default 10, max 25)"},
            },
            "required": ["stat"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> dict:
    logger.info(f"Tool: {tool_name} | {tool_input}")

    if tool_name == "search_player":
        player = nba.search_player(tool_input["name"])
        if player:
            info = nba.get_player_info(player["id"])
            return {
                "found":        True,
                "player_id":    player["id"],
                "full_name":    player["full_name"],
                "is_active":    player["is_active"],
                "team":         info.get("team", ""),
                "team_abbr":    info.get("team_abbr", ""),
                "position":     info.get("position", ""),
                "height":       info.get("height", ""),
                "jersey":       info.get("jersey", ""),
                "headshot_url": info.get("headshot_url", nba.headshot_url(player["id"])),
            }
        return {"found": False, "error": f"Player '{tool_input['name']}' not found."}

    elif tool_name == "search_team":
        team = nba.search_team(tool_input["name"])
        if team:
            return {
                "found":        True,
                "team_id":      team["id"],
                "full_name":    team["full_name"],
                "abbreviation": team["abbreviation"],
                "nickname":     team["nickname"],
                "city":         team["city"],
            }
        return {"found": False, "error": f"Team '{tool_input['name']}' not found."}

    elif tool_name == "get_stats_vs_team":
        return nba.get_player_vs_team_stats(
            player_id     = tool_input["player_id"],
            opponent_abbr = tool_input["opponent_abbr"],
            season_from   = tool_input.get("season_from", "2015-16"),
            season_to     = tool_input.get("season_to",   nba.DEFAULT_SEASON),
            season_type   = tool_input.get("season_type", "Regular Season"),
        )

    elif tool_name == "get_conditional_stats":
        # Default season_from to 2012-13 (wide range) so H2H queries never
        # silently truncate to the current season when Claude omits the range.
        # Claude can always pass a narrower range explicitly if the user asked for it.
        return nba.get_player_conditional_stats(
            player_id            = tool_input["player_id"],
            condition_player_ids = tool_input["condition_player_ids"],
            all_active           = tool_input["all_active"],
            season_from          = tool_input.get("season_from", "2012-13"),
            season_to            = tool_input.get("season_to",   nba.DEFAULT_SEASON),
            opponent_abbr        = tool_input.get("opponent_abbr", ""),
            season_type          = tool_input.get("season_type", "Regular Season"),
        )

    elif tool_name == "get_career_stats":
        return nba.get_player_career_stats(
            tool_input["player_id"],
            season_from = tool_input.get("season_from"),
            season_to   = tool_input.get("season_to"),
        )

    elif tool_name == "get_multi_player_stats":
        # Fill in defaults and coerce types — Gemini sometimes sends player_id
        # as a float (e.g. 2544.0) or omits season_to; normalise here so the
        # backend never receives bad types.
        players = tool_input.get("players", [])
        for cfg in players:
            if "player_id" in cfg:
                cfg["player_id"] = int(cfg["player_id"])
            if not cfg.get("season_to"):
                cfg["season_to"] = cfg.get("season_from", nba.DEFAULT_SEASON)
            if not cfg.get("season_type"):
                cfg["season_type"] = "Regular Season"
        return nba.get_multi_player_stats(players)

    elif tool_name == "get_leaderboard":
        return nba.get_leaderboard(
            stat        = tool_input.get("stat",        "pts"),
            season      = tool_input.get("season",      nba.DEFAULT_SEASON),
            season_type = tool_input.get("season_type", "Regular Season"),
            per_mode    = tool_input.get("per_mode",    "PerGame"),
            top_n       = tool_input.get("top_n",       10),
        )

    else:
        return {"error": f"Unknown tool: {tool_name}"}