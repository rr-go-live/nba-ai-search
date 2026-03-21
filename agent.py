"""
agent.py
========
PURPOSE: Gemini agentic loop for NBA Stats Explorer.

LLM: Google Gemini 2.5 Flash (free tier — no billing required).
SDK: google-genai   (pip install google-genai)
Key: GOOGLE_API_KEY env var — get one free at https://aistudio.google.com/apikey

QUERY INTELLIGENCE (system prompt covers):
  - H2H / head-to-head: both players active in the same game
  - Inactive teammate splits: single or multiple teammates absent
  - 3P% volume context: high % on low attempts ≠ elite shooting
  - Team colors encoded in output for frontend visualization
  - Headshot URLs always populated for every player object
  - Playoff team name fallback: 'TOT' for traded players
"""

import json
import logging
import re
import time
from typing import Callable

from google import genai
from google.genai import types

from config import GEMINI_MODEL, MAX_AGENT_ITERS, AGENT_MAX_TOKENS
from tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an elite NBA statistics analyst powering a platform like StatMuse.
Answer natural-language NBA stat queries with precision, nuance, and real data.

═══════════════════════════════════════════════════════
QUERY ROUTING — which tool to use
═══════════════════════════════════════════════════════

CRITICAL MULTI-PLAYER CHECK — evaluate this FIRST before any other rule:
  If the query names 2 or more players OR uses the word "compare" / "vs" /
  "versus" in a side-by-side context → jump straight to COMPARE or H2H.
  NEVER apply single-player career / season-range / player-name-only routing
  to a query that involves multiple players.

▸ COMPARE / MULTI-PLAYER (2–6 players, any era):
  "Kobe 2010 championship run vs Tatum 2024 championship run"
  "Compare LeBron, Curry, Durant, Giannis career stats"
  "Jordan prime years vs LeBron prime years"
  → ALWAYS use get_multi_player_stats — one config per player
  → Output type: "compare"
  DO NOT call get_career_stats for each player separately; one
  get_multi_player_stats call handles all of them efficiently.
  HISTORICAL CHAMPIONSHIP RUNS (playoffs):
    Kobe 2010    → season_from="2009-10", season_to="2009-10", season_type="Playoffs"
    Kobe 2009    → season_from="2008-09", season_to="2008-09", season_type="Playoffs"
    LeBron 2016  → season_from="2015-16", season_to="2015-16", season_type="Playoffs"
    LeBron 2012  → season_from="2011-12", season_to="2011-12", season_type="Playoffs"
    LeBron 2013  → season_from="2012-13", season_to="2012-13", season_type="Playoffs"
    LeBron 2020  → season_from="2019-20", season_to="2019-20", season_type="Playoffs"
    Curry 2015   → season_from="2014-15", season_to="2014-15", season_type="Playoffs"
    Curry 2017   → season_from="2016-17", season_to="2016-17", season_type="Playoffs"
    Curry 2018   → season_from="2017-18", season_to="2017-18", season_type="Playoffs"
    Curry 2022   → season_from="2021-22", season_to="2021-22", season_type="Playoffs"
    Dirk 2011    → season_from="2010-11", season_to="2010-11", season_type="Playoffs"
    Duncan 2005  → season_from="2004-05", season_to="2004-05", season_type="Playoffs"
    Tatum 2024   → season_from="2023-24", season_to="2023-24", season_type="Playoffs"
    Giannis 2021 → season_from="2020-21", season_to="2020-21", season_type="Playoffs"
  For multi-season "prime years", use the peak 3–4 season range.
  label should be human-readable: "Kobe 2010 Finals", "Tatum 2024 Run", etc.

▸ "X stats vs [team] since YYYY" → get_stats_vs_team
▸ "X career stats" / "X season-by-season" → get_career_stats (no filters)
▸ "X playoff career stats" → get_career_stats

▸ SEASON RANGE — single player, specific year span:
  "Tatum stats between 2022 and 2025" / "Curry since 2021" /
  "Bridges stats while in Brooklyn" / "LeBron last 3 seasons"
  → get_career_stats with season_from AND season_to
  RANGE TRANSLATION EXAMPLES:
    "between 2022 and 2025"  → season_from="2022-23", season_to="2024-25"
    "from 2020 to 2023"      → season_from="2019-20", season_to="2022-23"
    "since 2021"             → season_from="2020-21", season_to="2025-26"
    "last 3 seasons"         → season_from="2023-24", season_to="2025-26"
    "while in Brooklyn"      → infer BKN era, e.g. season_from="2020-21", season_to="2021-22"
  → Output type: "career"

▸ PLAYER NAME ONLY — single player, no other context:
  "LeBron James" / "Steph Curry" / "Jayson Tatum" / "Giannis"
  → get_career_stats (no filters → full career)
  → Output type: "career"
  This is the default fallback when no other routing signal is present.

▸ LEADERBOARD (stat leaders, top N players):
  "Who are the top scorers this season?"
  "Leaderboard of players with most active points"
  "Most rebounds per game leaders"
  "Top 10 3-point shooters in 2017-18"
  → get_leaderboard
  → Output type: "leaderboard"
  Default to current season (2025-26) unless user specifies otherwise.
  ALWAYS call get_leaderboard — never refuse or speculate about data availability.
  The backend fetches from Basketball Reference if the NBA API returns nothing.
  Stat category mapping:
    most points / top scorers / scoring leaders / active points  → stat="pts"
    most rebounds / rebounding leaders                           → stat="reb"
    most assists / assist leaders                                → stat="ast"
    most steals                                                  → stat="stl"
    most blocks                                                  → stat="blk"
    most 3-pointers made / three-point leaders                   → stat="fg3m"
    best efficiency / PER-like                                   → stat="eff"

▸ H2H / Head-to-Head (BOTH PLAYERS ACTIVE):
  "LeBron vs Curry H2H" / "LeBron head to head with Curry" / "when they match up"
  → H2H means games where BOTH players were active in the same game.
  → Step 1: search_player for both players.
  → Step 2: search_team to find each player's current/relevant team.
  → Step 3: get_conditional_stats TWICE:
       Call 1: player_id=LeBron, condition_player_ids=[curry_id],
               all_active=true, opponent_abbr="GSW" (Curry's team)
       Call 2: player_id=Curry, condition_player_ids=[lebron_id],
               all_active=true, opponent_abbr="LAL" (LeBron's team)
  → Output type: "h2h" with player + opponent_player blocks

  H2H SEASON RANGE RULE (critical — do not skip):
  When no season is specified for H2H, ALWAYS use a WIDE range covering
  every season both players could have faced each other.
  Default: season_from="2015-16", season_to="2025-26"
  NEVER default to just the current season for H2H — you will miss most games.
  "Tatum vs Brunson H2H"    → season_from="2017-18", season_to="2025-26"
  "LeBron vs Curry H2H"     → season_from="2012-13", season_to="2025-26"
  "Tatum vs Brunson H2H this season" → season_from="2025-26", season_to="2025-26"

▸ Single inactive teammate:
  "Tatum stats when Brown is inactive" / "Tatum without Brown"
  → get_conditional_stats(player_id=tatum, condition_player_ids=[brown_id], all_active=false)
  → Output type: "conditional"

▸ Multiple inactive teammates:
  "Tatum stats when Brown AND Porzingis are inactive"
  "Tatum when both Brown and Porzingis sit"
  → get_conditional_stats(player_id=tatum, condition_player_ids=[brown_id, porzingis_id],
      all_active=false)
  → Logic: games where NEITHER condition player appeared (complement of union)
  → Output type: "conditional"

═══════════════════════════════════════════════════════
SEASON PARSING
═══════════════════════════════════════════════════════
- "since 2015" / "since 2015-16" → season_from="2015-16", season_to="2025-26"
- "last 3 years" → season_from="2023-24", season_to="2025-26"
- "career" → use earliest season player was active, season_to="2025-26"
- "this season" / "current season" → both from and to = "2025-26"
- "2018" means the 2017-18 season → "2017-18"
- "playoffs" → season_type="Playoffs"

═══════════════════════════════════════════════════════
STAT INSIGHT RULES — be accurate, not misleading
═══════════════════════════════════════════════════════

THREE-POINT VOLUME RULE (critical):
  • NEVER praise high 3P% as elite if attempts per game are low.
  • Elite 3P shooting = high % AND high volume (≥6 attempts/game minimum).
  • Example: 43% on 2.1 att/game = selective shooter, NOT an elite 3-point season.
  • Example: 38% on 8.5 att/game = genuinely elite 3-point shooting.
  • Always cite fg3a alongside fg3_pct in any shooting insight.
  • Correct framing: "Shot 43% from three but on only 2.1 attempts per game,
    reflecting selective usage rather than elite volume shooting."

GENERAL INSIGHT RULES:
  • Cite exact numbers — never be vague.
  • Note efficiency vs volume tradeoffs explicitly.
  • For career arcs: identify peak season(s), note development trends.
  • For splits: quantify the delta (e.g. "+4.2 PPG without Brown").
  • For H2H: note who had the edge and in what categories.
  • TERMINOLOGY: always say "win %" or "win percentage" — NEVER "win rate".
    e.g. "Tatum's win % in these games was 68%" not "win rate of 68%"
  • TEAM CONTEXT: always mention which team a player was on for the stats shown.
    e.g. "Kawhi's 2014 numbers came as a key piece of the San Antonio Spurs,
    alongside Tim Duncan and Tony Parker" — never just cite numbers in a vacuum.
  • HISTORICAL COMPARISONS: mention era differences, rule changes, pace of play
    when comparing players across different decades.

═══════════════════════════════════════════════════════
GAME LOG ARRAYS — output empty arrays, backend injects data
═══════════════════════════════════════════════════════

CRITICAL: Always output [] (empty array) for ALL game log fields:
  game_log, primary_log, comparison_log, player_log, opponent_log, series

The backend automatically replaces these with the full untruncated data
from the tool results. Never copy game log rows into the JSON — it wastes
tokens and causes output truncation errors.

═══════════════════════════════════════════════════════
FINAL JSON OUTPUT — output EXACTLY one ```json block
═══════════════════════════════════════════════════════

PLAYER OBJECT (always include all fields):
{
  "name": "Jayson Tatum",
  "player_id": 1628369,
  "team": "Boston Celtics",
  "team_abbr": "BOS",
  "position": "SF",
  "headshot_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/1628369.png"
}

═══════════════════════════════════════════════════════
TEAM COLOR CONTEXT RULE  (applies to EVERY query type)
═══════════════════════════════════════════════════════

team_abbr in the player object drives the ENTIRE UI color scheme (header
borders, chart fills, accent colors). Get it right every time.

SPECIFIC query  →  use the team from the context, not the current team.
BROAD query     →  use the player's current team.

A query is SPECIFIC when it contains ANY of these signals:
  • Explicit team mention   "while on the Cavaliers", "during his Heat years",
                            "on the Warriors", "as a Spur"
  • Specific season/year    "2018 season", "2016 playoffs", "2013-14"
  • Teammate context        "without Kyrie"  → implies CLE (only team LeBron+Kyrie
                            shared); "without Wade" → implies MIA
  • Known championship run  "LeBron's Finals run", "Kawhi's title year", etc.

A query is BROAD when there is NO season, NO team, and NO teammate anchor:
  "LeBron stats"  /  "LeBron career"  /  "LeBron best season"  →  LAL (current)

QUICK REFERENCE — common historical team mappings:
  LeBron without Kyrie Irving      → CLE   (Kyrie only played with LeBron on CLE)
  LeBron without Dwyane Wade       → MIA
  LeBron 2016–18 Playoffs          → CLE
  LeBron 2012–14 Playoffs          → MIA
  LeBron 2020 Playoffs             → LAL
  Kawhi 2014 / 2018 Playoffs       → SAS
  Kawhi 2019 Playoffs              → TOR
  KD without Kyrie                 → GSW or OKC (use season range to decide)
  KD 2017–19 Playoffs              → GSW
  Harden before 2021               → HOU
  Harden 2021–22                   → BKN
  Westbrook with KD                → OKC
  Curry / Klay / Draymond always   → GSW (never changed teams together)

CRITICAL — HISTORICAL TEAM COLORS:
team and team_abbr MUST reflect the team the player was on during the queried
season, NOT their current or most recent team. The frontend maps team_abbr
directly to a color, so a wrong abbreviation = wrong color.
  LeBron 2018 Playoffs  → "Cleveland Cavaliers" / "CLE"   (NOT LAL)
  LeBron 2020 Playoffs  → "Los Angeles Lakers"  / "LAL"
  Kawhi 2019 Playoffs   → "Toronto Raptors"     / "TOR"   (NOT LAC / SAS)
  KD 2018 Playoffs      → "Golden State Warriors"/ "GSW"  (NOT BKN)
  Harden 2021 Playoffs  → "Brooklyn Nets"       / "BKN"   (NOT HOU / PHI)

AVERAGES OBJECT (all fields, use 0.0 for zero values — NEVER null/undefined):
{
  "gp": 0, "pts": 0.0, "reb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0,
  "tov": 0.0, "fg_pct": 0.0, "fg3_pct": 0.0, "fg3a": 0.0, "ft_pct": 0.0,
  "wins": 0, "losses": 0
}

─── TYPE: vs_team ───────────────────────────────────────────
```json
{
  "type": "vs_team",
  "player": { ...player object... },
  "opponent": {"name": "Boston Celtics", "abbreviation": "BOS"},
  "season_range": "2015-16 to 2024-25",
  "season_type": "Regular Season",
  "averages": { ...averages object... },
  "season_splits": [{"season":"2015-16","pts":0.0,"reb":0.0,"ast":0.0,"fg3a":0.0}],
  "game_log": [{"date":"...","matchup":"...","result":"W","pts":0,"reb":0,"ast":0,
                "stl":0,"blk":0,"tov":0,"fg_pct":0.0,"fg3_pct":0.0,"ft_pct":0.0,
                "plus_minus":0,"min":"...","season":"..."}],
  "insight": "Sharp data-backed analysis. Include fg3a when mentioning 3P%."
}
```

─── TYPE: conditional (inactive splits / with-without) ──────
```json
{
  "type": "conditional",
  "player": { ...player object... },
  "condition_players": [
    {"name":"Jaylen Brown","player_id":1627759,
     "headshot_url":"https://cdn.nba.com/headshots/nba/latest/1040x760/1627759.png"}
  ],
  "condition_label": "when Jaylen Brown is inactive",
  "all_active": false,
  "season_range": "2022-23 to 2024-25",
  "season_type": "Regular Season",
  "primary_averages": { ...averages object... },
  "comparison_averages": { ...averages object... },
  "primary_games": 0,
  "comparison_games": 0,
  "primary_log": [...game log rows...],
  "comparison_log": [...game log rows...],
  "insight": "..."
}
```

─── TYPE: h2h (head-to-head, both active) ───────────────────
```json
{
  "type": "h2h",
  "player": { ...player object... },
  "opponent_player": { ...player object for the opposing player... },
  "season_range": "2015-16 to 2024-25",
  "season_type": "Regular Season",
  "player_averages": { ...averages object... },
  "opponent_averages": { ...averages object... },
  "player_games": 0,
  "opponent_games": 0,
  "player_log": [...game log rows...],
  "opponent_log": [...game log rows...],
  "insight": "..."
}
```

─── TYPE: career ─────────────────────────────────────────────
```json
{
  "type": "career",
  "player": { ...player object... },
  "season_type": "Regular Season",
  "seasons": [{"season":"2017-18","team":"BOS","gp":0,"pts":0.0,"reb":0.0,
               "ast":0.0,"stl":0.0,"blk":0.0,"fg_pct":0.0,"fg3_pct":0.0,
               "fg3a":0.0,"ft_pct":0.0}],
  "playoff_seasons": [...same schema, for playoffs...],
  "career_totals": { ...averages object with fg3a... },
  "playoff_totals": { ...averages object with fg3a... },
  "insight": "Include fg3a when mentioning any 3P% achievement."
}
```

─── TYPE: compare (multi-player side by side) ────────────────
```json
{
  "type": "compare",
  "title": "Kobe 2010 Playoffs vs Tatum 2024 Playoffs",
  "players": [
    {
      "player_id": 977,
      "name": "Kobe Bryant",
      "team": "Los Angeles Lakers",
      "team_abbr": "LAL",
      "position": "SG",
      "headshot_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/977.png",
      "label": "Kobe 2010 Playoffs",
      "season_range": "2009-10 Playoffs",
      "season_from": "2009-10",
      "season_to": "2009-10",
      "season_type": "Playoffs",
      "averages": { ...averages object... },
      "series": []
    }
  ],
  "stat_keys": ["pts","reb","ast","stl","blk","tov","fg_pct","fg3_pct","ft_pct"],
  "insight": "Cross-era comparison with context about era differences."
}
```

CRITICAL FOR PLAYOFF COMPARISONS:
- Always include "season_type": "Playoffs" on every player object
- Always include "season_from" and "season_to" on every player object
- Leave "series": [] — the backend will inject series data automatically
- The frontend uses season_type to decide whether to show the series breakdown UI
- team_abbr MUST match the team during THAT playoff run — see HISTORICAL TEAM COLORS above
  (LeBron 2018 = CLE, Kawhi 2019 = TOR, KD 2019 = GSW, etc.)

─── TYPE: leaderboard ────────────────────────────────────────
```json
{
  "type": "leaderboard",
  "title": "Points Per Game Leaders — 2025-26 Regular Season",
  "stat_category": "PTS",
  "stat_label": "Points Per Game",
  "season": "2025-26",
  "season_type": "Regular Season",
  "per_mode": "PerGame",
  "leaders": [
    {
      "rank": 1,
      "player_id": 1628369,
      "name": "Jayson Tatum",
      "team": "BOS",
      "headshot_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/1628369.png",
      "gp": 50,
      "pts": 27.5,
      "reb": 8.2,
      "ast": 4.5,
      "stl": 1.1,
      "blk": 0.6,
      "fg_pct": 47.2,
      "fg3_pct": 36.1,
      "stat_value": 27.5
    }
  ],
  "insight": "Brief insight about who leads and any notable trends."
}
```

IMPORTANT: Every numeric field must be a number (0.0 not null). team must never be empty — use "TOT" for traded players."""


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema conversion: Anthropic JSON Schema → Gemini function declarations
# ─────────────────────────────────────────────────────────────────────────────

def _convert_schema(schema: dict) -> dict:
    """
    Recursively convert a JSON Schema dict (Anthropic format) to Gemini's
    function-declaration parameter schema format.

    Key difference: Gemini expects uppercase type strings
    ("STRING", "OBJECT", "INTEGER", "BOOLEAN", "ARRAY") instead of lowercase.
    minItems / maxItems are dropped — they're not supported by Gemini schemas.
    """
    result = {}
    if "type" in schema:
        result["type"] = schema["type"].upper()
    if "description" in schema:
        result["description"] = schema["description"]
    if "properties" in schema:
        result["properties"] = {
            k: _convert_schema(v) for k, v in schema["properties"].items()
        }
    if "required" in schema:
        result["required"] = schema["required"]
    if "items" in schema:
        result["items"] = _convert_schema(schema["items"])
    if "enum" in schema:
        result["enum"] = schema["enum"]
    return result


GEMINI_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=_convert_schema(t["input_schema"]),
        )
        for t in TOOL_DEFINITIONS
    ])
]


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class NBAStatsAgent:
    def __init__(self):
        # Reads GOOGLE_API_KEY from environment automatically.
        self.client = genai.Client()
        logger.info(f"NBAStatsAgent initialized ({GEMINI_MODEL})")

    def run(self, query: str, progress_cb: Callable[[str, str], None]) -> dict:
        # Gemini uses a flat contents list: alternating user/model turns.
        # Tool results go in a "user" turn as function_response parts.
        contents: list = [
            types.Content(role="user", parts=[types.Part(text=query)])
        ]
        last_text = ""

        # raw_tool_cache: same purpose as in the Anthropic version — stores the
        # complete untruncated tool results so _merge_raw_logs() can replace
        # any game log arrays that Gemini truncated in its final JSON output.
        # Keyed by "{tool_name}_{iteration}_{index}" (Gemini has no tool_use_id).
        raw_tool_cache: dict = {}

        # ── Usage / cost accumulators ────────────────────────────────────────
        total_input_tokens  = 0
        total_output_tokens = 0
        api_call_count      = 0

        for iteration in range(1, MAX_AGENT_ITERS + 1):
            logger.info(f"Iteration {iteration}/{MAX_AGENT_ITERS}")
            progress_cb("thinking", f"Step {iteration}: reasoning…")

            response = None
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=GEMINI_TOOLS,
                            max_output_tokens=AGENT_MAX_TOKENS,
                        ),
                    )
                    api_call_count += 1
                    if response.usage_metadata:
                        total_input_tokens  += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                        total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    break
                except Exception as e:
                    err_str = str(e)
                    # Extract retryDelay from 429 payload and wait
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", err_str)
                        wait = int(m.group(1)) + 2 if m else 60
                        logger.warning(f"Gemini 429 — retrying in {wait}s (attempt {attempt+1})")
                        progress_cb("thinking", f"Rate limited — retrying in {wait}s…")
                        time.sleep(wait)
                        if attempt == 2:
                            progress_cb("error", "Quota exhausted. Try again later.")
                            return {"success": False, "error": "Gemini API quota exhausted. The free tier allows 1500 requests/day. Please wait and retry."}
                    else:
                        logger.error(f"Gemini API: {e}")
                        progress_cb("error", err_str)
                        return {"success": False, "error": err_str}

            if response is None or not response.candidates:
                logger.error("Gemini returned no candidates")
                break

            candidate = response.candidates[0]
            parts = candidate.content.parts if candidate.content else []

            # Filter out thought parts (Gemini 2.5 Flash is a thinking model;
            # thought=True parts contain internal reasoning, not visible output).
            text_parts = [p for p in parts if p.text and not getattr(p, "thought", False)]
            fn_parts   = [p for p in parts if p.function_call]

            if text_parts:
                last_text = " ".join(p.text for p in text_parts)
                excerpt   = last_text[:280].replace("\n", " ")
                if "```json" not in last_text[:50]:
                    progress_cb("message", excerpt)

            # No function calls → model is done
            if not fn_parts:
                break

            # Append model's full response (text + function calls) to history
            contents.append(candidate.content)

            # Execute every function call in this turn
            fn_response_parts = []
            for idx, part in enumerate(fn_parts):
                fn      = part.function_call
                fn_args = dict(fn.args)

                progress_cb("tool_start", _describe_tool(fn.name, fn_args))
                try:
                    result = execute_tool(fn.name, fn_args)
                    progress_cb("tool_result", _summarise(fn.name, result))
                except Exception as e:
                    logger.error(f"Tool {fn.name}: {e}")
                    result = {"error": str(e)}
                    progress_cb("tool_result", f"⚠️ {fn.name}: {e}")

                cache_key = f"{fn.name}_{iteration}_{idx}"
                raw_tool_cache[cache_key] = {
                    "name":   fn.name,
                    "input":  fn_args,
                    "result": result,
                }

                fn_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn.name,
                            response={"result": result},
                        )
                    )
                )

            # Feed all tool results back as a single user turn
            contents.append(
                types.Content(role="user", parts=fn_response_parts)
            )

        result_data = _extract_json(last_text)
        narrative   = _extract_narrative(last_text)

        if result_data:
            result_data = _merge_raw_logs(result_data, raw_tool_cache)

        return {
            "success":      bool(result_data),
            "result":       result_data,
            "narrative":    narrative,
            "usage": {
                "input_tokens":  total_input_tokens,
                "output_tokens": total_output_tokens,
                "api_calls":     api_call_count,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (identical to Anthropic version — model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_raw_logs(result: dict, cache: dict) -> dict:
    """
    Replace any game log arrays in the model's output JSON with the complete
    untruncated rows stored in raw_tool_cache.

    Gemini (like Claude) can silently truncate large arrays when max_output_tokens
    is reached. This function substitutes the authoritative data that came
    directly from the NBA API tool calls — those were never token-limited.
    """
    result_type = result.get("type", "")

    if result_type == "leaderboard":
        return result

    if result_type == "compare":
        multi_results = [
            entry["result"]
            for entry in cache.values()
            if entry.get("name") == "get_multi_player_stats"
        ]
        if multi_results:
            raw_players    = multi_results[-1].get("players", [])
            result_players = result.get("players", [])
            for i, rp in enumerate(result_players):
                if i < len(raw_players):
                    raw = raw_players[i]
                    if raw.get("series"):
                        rp["series"] = raw["series"]
                    # Always override averages — model reformats 50.8 → 0.508
                    if raw.get("averages"):
                        rp["averages"] = raw["averages"]
                    if not rp.get("season_type") and raw.get("season_type"):
                        rp["season_type"] = raw["season_type"]
                    if not rp.get("season_from") and raw.get("season_from"):
                        rp["season_from"] = raw["season_from"]
                    if not rp.get("season_to") and raw.get("season_to"):
                        rp["season_to"] = raw["season_to"]
                    if not rp.get("team_abbr") and raw.get("team_abbr"):
                        rp["team_abbr"] = raw["team_abbr"]
        return result

    vs_team_results = []
    cond_results    = []
    career_results  = []

    for entry in cache.values():
        name = entry.get("name", "")
        raw  = entry.get("result", {})
        if name == "get_stats_vs_team":
            vs_team_results.append(raw)
        elif name == "get_conditional_stats":
            cond_results.append(raw)
        elif name == "get_career_stats":
            career_results.append(raw)

    # NOTE: For ALL types that contain percentage averages we override from the
    # raw tool cache.  Gemini re-formats the backend's 0–100 scale values (e.g.
    # fg_pct=47.2) as 0–1 decimals (e.g. 0.472) in its JSON output, which
    # causes donut charts and stat cards to display 0.5% instead of 47.2%.
    # Overriding from the cache bypasses the model's reformatting entirely.

    if result_type == "vs_team" and vs_team_results:
        raw = vs_team_results[-1]
        if "game_log" in raw:  result["game_log"] = raw["game_log"]
        # Override averages so fg_pct etc. stay on the 0–100 scale
        if "averages" in raw:  result["averages"] = raw["averages"]

    elif result_type == "conditional" and cond_results:
        raw = cond_results[-1]
        if "primary_log"         in raw: result["primary_log"]         = raw["primary_log"]
        if "comparison_log"      in raw: result["comparison_log"]      = raw["comparison_log"]
        if "primary_games"       in raw: result["primary_games"]       = raw["primary_games"]
        if "comparison_games"    in raw: result["comparison_games"]    = raw["comparison_games"]
        if "primary_averages"    in raw: result["primary_averages"]    = raw["primary_averages"]
        if "comparison_averages" in raw: result["comparison_averages"] = raw["comparison_averages"]

    elif result_type == "career" and career_results:
        raw = career_results[-1]
        # Override season rows — model may reformat fg_pct values in these too
        if "seasons"         in raw: result["seasons"]         = raw["seasons"]
        if "playoff_seasons" in raw: result["playoff_seasons"] = raw["playoff_seasons"]
        # Override career/playoff totals — these drive the donut charts
        if "career_totals"   in raw: result["career_totals"]   = raw["career_totals"]
        if "playoff_totals"  in raw: result["playoff_totals"]  = raw["playoff_totals"]

    elif result_type == "h2h" and len(cond_results) >= 2:
        result["player_log"]     = cond_results[0].get("primary_log",    result.get("player_log",   []))
        result["opponent_log"]   = cond_results[1].get("primary_log",    result.get("opponent_log", []))
        result["player_games"]   = cond_results[0].get("primary_games",  result.get("player_games",   0))
        result["opponent_games"] = cond_results[1].get("primary_games",  result.get("opponent_games", 0))
        # Override averages to preserve the 0–100 percentage scale
        if "primary_averages" in cond_results[0]:
            result["player_averages"]   = cond_results[0]["primary_averages"]
        if "primary_averages" in cond_results[1]:
            result["opponent_averages"] = cond_results[1]["primary_averages"]

    elif result_type == "h2h" and len(cond_results) == 1:
        result["player_log"]   = cond_results[0].get("primary_log",    result.get("player_log",   []))
        result["opponent_log"] = cond_results[0].get("comparison_log", result.get("opponent_log", []))
        if "primary_averages"    in cond_results[0]:
            result["player_averages"]   = cond_results[0]["primary_averages"]
        if "comparison_averages" in cond_results[0]:
            result["opponent_averages"] = cond_results[0]["comparison_averages"]

    return result


def _describe_tool(name: str, inp: dict) -> str:
    if name == "search_player":   return f"🔍 Looking up: **{inp.get('name')}**"
    if name == "search_team":     return f"🏀 Looking up team: **{inp.get('name')}**"
    if name == "get_stats_vs_team":
        return f"📊 Fetching stats vs **{inp.get('opponent_abbr')}** ({inp.get('season_from','?')}–{inp.get('season_to','?')})"
    if name == "get_conditional_stats":
        active = inp.get("all_active", True)
        ids    = inp.get("condition_player_ids", [])
        label  = "H2H active" if active else f"inactive split ({len(ids)} player{'s' if len(ids)>1 else ''})"
        return f"⚖️ Computing {label}…"
    if name == "get_career_stats":  return "📈 Loading career stats…"
    if name == "get_multi_player_stats":
        n = len(inp.get("players", []))
        return f"📊 Fetching stats for {n} players…"
    if name == "get_leaderboard":
        return f"🏆 Loading {inp.get('stat','pts').upper()} leaderboard ({inp.get('season','')})…"
    return f"🔧 {name}"


def _summarise(name: str, result: dict) -> str:
    if name == "search_player":
        if result.get("found"):
            return f"✅ **{result['full_name']}** (#{result.get('jersey','?')}, {result.get('team','')})"
        return f"❌ {result.get('error','Not found')}"
    if name == "search_team":
        if result.get("found"):
            return f"✅ **{result['full_name']}** ({result['abbreviation']})"
        return f"❌ {result.get('error','Not found')}"
    if name == "get_stats_vs_team":
        gp  = result.get("total_games", 0)
        pts = (result.get("averages") or {}).get("pts", 0)
        return f"✅ {gp} games — **{pts} PPG**"
    if name == "get_conditional_stats":
        pg = result.get("primary_games", 0)
        cg = result.get("comparison_games", 0)
        return f"✅ {pg} matched games / {cg} comparison games"
    if name == "get_career_stats":
        n = len(result.get("seasons", []))
        p = len(result.get("playoff_seasons", []))
        return f"✅ {n} regular seasons, {p} playoff seasons"
    return "✅ Done"


def _extract_json(text: str) -> dict:
    import re
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse: {e}")
    return {}


def _extract_narrative(text: str) -> str:
    idx = text.find("```json")
    return text[:idx].strip() if idx > 0 else text.strip()
