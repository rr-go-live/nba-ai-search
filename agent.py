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
- "since 2015" / "since 2015-16" → season_from="2015-16", season_to="2024-25"
- "last 3 years" → season_from="2022-23", season_to="2024-25"
- "career" → use earliest season player was active, season_to="2024-25"
- "this season" / "2024-25" → both from and to = "2024-25"
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

IMPORTANT: Every numeric field must be a number (0.0 not null). team must never be empty — use "TOT" for traded players."""


class NBAStatsAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        logger.info(f"NBAStatsAgent initialized ({CLAUDE_MODEL})")

    def run(self, query: str, progress_cb: Callable[[str, str], None]) -> dict:
        messages  = [{"role": "user", "content": query}]
        last_text = ""

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
        return {"success": bool(result_data), "result": result_data, "narrative": narrative}


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
    if name == "get_career_stats": return "📈 Loading career stats…"
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