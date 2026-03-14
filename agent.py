"""
agent.py
========
PURPOSE: Claude agentic loop for NBA Stats Explorer.

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
from typing import Callable

import anthropic

from config import CLAUDE_MODEL, MAX_AGENT_ITERS, AGENT_MAX_TOKENS
from tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an elite NBA statistics analyst powering a platform like StatMuse.
Answer natural-language NBA stat queries with precision, nuance, and real data.

═══════════════════════════════════════════════════════
QUERY ROUTING — which tool to use
═══════════════════════════════════════════════════════

▸ "X stats vs [team] since YYYY" → get_stats_vs_team
▸ "X career stats" / "X season-by-season" → get_career_stats
▸ "X playoff career stats" → get_career_stats (season_type implied in JSON output)

▸ COMPARE / MULTI-PLAYER (2–6 players, any era):
  "Kobe 2010 championship run vs Tatum 2024 championship run"
  "Compare LeBron, Curry, Durant, Giannis career stats"
  "Jordan prime years vs LeBron prime years"
  → get_multi_player_stats with one config per player
  → Output type: "compare"
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

▸ LEADERBOARD (stat leaders, top N players):
  "Who are the top scorers this season?"
  "Leaderboard of players with most active points"
  "Most rebounds per game leaders"
  "Top 10 3-point shooters in 2017-18"
  → get_leaderboard
  → Output type: "leaderboard"
  Default to current season (2025-26) unless user specifies otherwise.
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


class NBAStatsAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        logger.info(f"NBAStatsAgent initialized ({CLAUDE_MODEL})")

    def run(self, query: str, progress_cb: Callable[[str, str], None]) -> dict:
        messages       = [{"role": "user", "content": query}]
        last_text      = ""
        # raw_tool_cache stores the complete untruncated result of every tool
        # call, keyed by tool_use_id.
        #
        # WHY THIS EXISTS:
        # Claude output is capped at AGENT_MAX_TOKENS. A full multi-season game
        # log (40-80 rows x ~150 tokens each) can exceed that budget — Claude
        # silently truncates the game_log array in its final JSON to fit.
        # After we parse Claude's JSON we call _merge_raw_logs() which replaces
        # the truncated arrays with the full rows from this cache, so the
        # frontend always receives every game regardless of output length.
        raw_tool_cache: dict = {}

        for iteration in range(1, MAX_AGENT_ITERS + 1):
            logger.info(f"Iteration {iteration}/{MAX_AGENT_ITERS}")
            progress_cb("thinking", f"Step {iteration}: reasoning…")

            try:
                response = self.client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=AGENT_MAX_TOKENS,
                    system=SYSTEM_PROMPT, tools=TOOL_DEFINITIONS, messages=messages,
                )
            except Exception as e:
                logger.error(f"Claude API: {e}")
                progress_cb("error", str(e))
                return {"success": False, "error": str(e)}

            text_blocks     = [b for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if text_blocks:
                last_text = " ".join(b.text for b in text_blocks)
                excerpt   = last_text[:280].replace("\n", " ")
                if "```json" not in last_text[:50]:
                    progress_cb("message", excerpt)

            if response.stop_reason == "end_turn" and not tool_use_blocks:
                break

            if tool_use_blocks:
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in tool_use_blocks:
                    progress_cb("tool_start", _describe_tool(block.name, block.input))
                    try:
                        result = execute_tool(block.name, block.input)
                        progress_cb("tool_result", _summarise(block.name, result))
                    except Exception as e:
                        logger.error(f"Tool {block.name}: {e}")
                        result = {"error": str(e)}
                        progress_cb("tool_result", f"⚠️ {block.name}: {e}")

                    # Cache by tool_use_id so two calls to the same tool
                    # (e.g. both H2H get_conditional_stats calls) are kept
                    # separately and can each be merged back correctly.
                    raw_tool_cache[block.id] = {
                        "name":   block.name,
                        "input":  block.input,
                        "result": result,
                    }

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps(result),
                    })
                messages.append({"role": "user", "content": tool_results})
            elif response.stop_reason != "end_turn":
                break

        result_data = _extract_json(last_text)
        narrative   = _extract_narrative(last_text)

        # Overwrite any truncated game logs with the full rows from the cache.
        if result_data:
            result_data = _merge_raw_logs(result_data, raw_tool_cache)

        return {"success": bool(result_data), "result": result_data, "narrative": narrative}


def _merge_raw_logs(result: dict, cache: dict) -> dict:
    """
    _merge_raw_logs(result, cache) -> dict
    ======================================
    Replace any game log arrays in Claude's output JSON with the complete
    untruncated rows stored in raw_tool_cache.

    WHY:
    Claude writes its final answer as JSON inside its text output, which is
    capped at AGENT_MAX_TOKENS tokens. A multi-season game log (e.g. LeBron
    vs Celtics since 2015 = ~44 games) can take 6,000+ tokens to serialise.
    When that pushes Claude over budget it silently truncates the array,
    leaving only the most recent N games in its output.

    This function ignores whatever Claude wrote for game log fields and
    substitutes the authoritative rows that came directly from the NBA API
    via the tool call result — those were never token-limited.

    MAPPING:
      vs_team     → game_log       from get_stats_vs_team result
      conditional → primary_log    from get_conditional_stats result (1st call)
                    comparison_log from get_conditional_stats result
      h2h         → player_log     from 1st  get_conditional_stats call
                    opponent_log   from 2nd get_conditional_stats call
    """
    result_type = result.get("type", "")

    # leaderboard has no arrays to merge — pass through unchanged
    if result_type == "leaderboard":
        return result

    # compare: inject series + season_type from raw tool cache into each player object.
    # Claude writes "series": [] in its JSON (as instructed) — we fill it here from
    # the actual get_multi_player_stats result which contains the full series breakdown.
    if result_type == "compare":
        multi_results = [
            entry["result"]
            for entry in cache.values()
            if entry.get("name") == "get_multi_player_stats"
        ]
        if multi_results:
            raw_players = multi_results[-1].get("players", [])
            result_players = result.get("players", [])
            for i, rp in enumerate(result_players):
                if i < len(raw_players):
                    raw = raw_players[i]
                    # Inject series breakdown (the main thing Claude can't write)
                    if raw.get("series"):
                        rp["series"] = raw["series"]
                    # Ensure season_type is present — frontend uses it to show series UI
                    if not rp.get("season_type") and raw.get("season_type"):
                        rp["season_type"] = raw["season_type"]
                    if not rp.get("season_from") and raw.get("season_from"):
                        rp["season_from"] = raw["season_from"]
                    if not rp.get("season_to") and raw.get("season_to"):
                        rp["season_to"] = raw["season_to"]
                    # Backfill team_abbr from raw if Claude left it empty
                    if not rp.get("team_abbr") and raw.get("team_abbr"):
                        rp["team_abbr"] = raw["team_abbr"]
        return result

    # Collect all tool results by name (preserving order for H2H dual calls)
    vs_team_results   = []
    cond_results      = []

    for entry in cache.values():
        name = entry.get("name", "")
        raw  = entry.get("result", {})
        if name == "get_stats_vs_team":
            vs_team_results.append(raw)
        elif name == "get_conditional_stats":
            cond_results.append(raw)

    if result_type == "vs_team" and vs_team_results:
        raw = vs_team_results[-1]
        if "game_log" in raw:
            result["game_log"] = raw["game_log"]

    elif result_type == "conditional" and cond_results:
        raw = cond_results[-1]
        if "primary_log"    in raw: result["primary_log"]    = raw["primary_log"]
        if "comparison_log" in raw: result["comparison_log"] = raw["comparison_log"]
        # Also update game counts so the UI badge is accurate
        if "primary_games"    in raw: result["primary_games"]    = raw["primary_games"]
        if "comparison_games" in raw: result["comparison_games"] = raw["comparison_games"]

    elif result_type == "h2h" and len(cond_results) >= 2:
        # H2H makes two get_conditional_stats calls: Call 1 = primary player,
        # Call 2 = opponent player. The matched rows ARE the H2H games.
        result["player_log"]   = cond_results[0].get("primary_log",   result.get("player_log",   []))
        result["opponent_log"] = cond_results[1].get("primary_log",   result.get("opponent_log", []))
        result["player_games"]   = cond_results[0].get("primary_games",   result.get("player_games",   0))
        result["opponent_games"] = cond_results[1].get("primary_games",   result.get("opponent_games", 0))

    elif result_type == "h2h" and len(cond_results) == 1:
        # Fallback: only one cond call found
        result["player_log"]   = cond_results[0].get("primary_log",    result.get("player_log",   []))
        result["opponent_log"] = cond_results[0].get("comparison_log", result.get("opponent_log", []))

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