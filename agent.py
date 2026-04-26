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

from config import GEMINI_MODEL, MAX_AGENT_ITERS, AGENT_MAX_TOKENS, YEAR_TO_SEASON, DEFAULT_SEASON
from tools import TOOL_DEFINITIONS, execute_tool
import nba_stats_client as nba

logger = logging.getLogger(__name__)

# NBA team abbreviation → nickname, used to label the era team on H2H cards
# when a player changed teams (e.g. Klay played Iguodala as a Warrior, not a Maverick).
_H2H_TEAM_NAMES: dict = {
    "ATL": "Hawks",        "BOS": "Celtics",       "BKN": "Nets",
    "CHA": "Hornets",      "CHI": "Bulls",          "CLE": "Cavaliers",
    "DAL": "Mavericks",    "DEN": "Nuggets",        "DET": "Pistons",
    "GSW": "Warriors",     "HOU": "Rockets",        "IND": "Pacers",
    "LAC": "Clippers",     "LAL": "Lakers",         "MEM": "Grizzlies",
    "MIA": "Heat",         "MIL": "Bucks",          "MIN": "Timberwolves",
    "NOP": "Pelicans",     "NYK": "Knicks",         "OKC": "Thunder",
    "ORL": "Magic",        "PHI": "76ers",          "PHX": "Suns",
    "POR": "Trail Blazers","SAC": "Kings",          "SAS": "Spurs",
    "TOR": "Raptors",      "UTA": "Jazz",           "WAS": "Wizards",
    # Historical
    "SEA": "SuperSonics",  "NJN": "Nets",           "NOH": "Hornets",
    "VAN": "Grizzlies",    "NOK": "Hornets",        "WSB": "Bullets",
}

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

  FINALS / SPECIFIC ROUND — series_round field:
  When the query explicitly asks for a specific series or round (Finals, Conference Finals, etc.),
  add series_round to each player config to filter stats to that round only:
    "LeBron 2016 Finals"  → season_from="2015-16", season_to="2015-16", season_type="Playoffs", series_round="finals"
    "KD 2017 Finals"      → season_from="2016-17", season_to="2016-17", season_type="Playoffs", series_round="finals"
    "Kobe 2010 Finals"    → season_from="2009-10", season_to="2009-10", season_type="Playoffs", series_round="finals"
  Valid series_round values: "finals", "cf" (Conference Finals), "r2", "r1"
  Do NOT add series_round for full "championship run" or "playoff run" queries — only when a
  specific round is explicitly mentioned.

  SEASON PARALLELISM RULE (critical):
  When one player has an explicit season/type and the other(s) do not, apply
  the SAME season and type to all players — do not default the unspecified
  player to their full career or current season.
  EXAMPLES:
    "Maxey Playoffs vs Brunson 2024 Playoffs"
        → both: season_from="2023-24", season_to="2023-24", season_type="Playoffs"
    "Tatum 2024 Playoffs vs Jaylen Brown"
        → both: season_from="2023-24", season_to="2023-24", season_type="Playoffs"
    "LeBron vs Curry 2022 championship run"
        → both: season_from="2021-22", season_to="2021-22", season_type="Playoffs"
  Exception: only use different seasons when the query explicitly contrasts eras
    (e.g. "Kobe 2010 playoffs vs Tatum 2024 playoffs").

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

▸ LAST / FIRST N GAMES or DATE-RANGE GAMES:
  *** get_last_n_games WORKS FOR ANY SEASON — past seasons are fully supported. ***
  *** NEVER mark a past-season or first-N-games query as invalid. ***

  "order" controls direction:
    order="last"  → most recent N games (default)
    order="first" → earliest N games from the season start

  Examples (ALL valid):
    "LeBron last 10 games"                    → n=10, order="last", season="2025-26"
    "Curry last 10 games in 2022-23"          → n=10, order="last", season="2022-23"
    "Giannis last 5 games in 2021"            → n=5,  order="last", season="2020-21"
    "LeBron last 5 playoff games 2020"        → n=5,  order="last", season="2019-20", season_type="Playoffs"
    "Tatum last 10 games this season"         → n=10, order="last", season="2025-26"
    "Giannis first 10 games of 2025-26"       → n=10, order="first", season="2025-26"
    "Curry first 5 games this season"         → n=5,  order="first", season="2025-26"
    "LeBron games between 1/1/26 and 3/1/26"  → date_from="1/1/26", date_to="3/1/26"
    "Tatum stats Jan 1 to Mar 1 2026"         → date_from="2026-01-01", date_to="2026-03-01"

  DATE-RANGE RULES:
    • When date_from/date_to are provided, season is auto-inferred from dates.
    • n is optional for date-range queries — omit it to return all games in the window.
    • Dates accepted as "YYYY-MM-DD" or "M/D/YY" (e.g. "1/1/26").

  → Step 1: search_player to get player_id
  → Step 2: get_last_n_games(player_id, n, season, season_type, order, date_from, date_to)
  → Output type: "last_n_games"
  Default n=10 when the user doesn't specify a number (skip n for date-range queries).
  Default season="2025-26" only when the user does NOT mention any season or year.
  Pass season_type="Playoffs" when the user asks about playoff games.
  Always pass the exact season the user named — never assume current.

▸ H2H / Head-to-Head (BOTH PLAYERS ACTIVE):
  "LeBron vs Curry H2H" / "LeBron head to head with Curry" / "when they match up"
  → H2H means games where BOTH players were active in the same game.
  → Step 1: search_player for both players.
  → Step 2: search_team to find each player's current/relevant team.
  → Step 3: get_conditional_stats TWICE — always set require_opponent=true for H2H.
       This ensures shared-team games (e.g. Klay+Curry on GSW) are excluded,
       leaving only genuine opponent matchups.
       Call 1: player_id=LeBron, condition_player_ids=[curry_id], all_active=true, require_opponent=true
       Call 2: player_id=Curry,  condition_player_ids=[lebron_id], all_active=true, require_opponent=true
  → Output type: "h2h" with player + opponent_player blocks

  H2H SEASON RANGE RULE (critical — do not skip):
  When no season is specified, ALWAYS cover EVERY season both players could
  have faced each other — from when the YOUNGER player entered the league.
  NEVER default to just the current season — you will miss most games.

  *** FORMER-TEAMMATE RULE — CRITICAL ***
  For players who were TEAMMATES at some point but also played as OPPONENTS before
  or after, you MUST STILL use the full career range. DO NOT narrow season_from to
  avoid teammate years. The backend automatically excludes all same-team games.
  Narrowing the range SILENTLY DROPS early opponent-era games.

  EXAMPLE: Iguodala vs Curry
    ✗ WRONG: season_from="2019-20"  ← misses 2009-2013 PHI/DEN era vs GSW
    ✓ RIGHT: season_from="2009-10"  ← backend excludes 2013-2019 GSW teammate games

  EXAMPLE: KAT vs Edwards, Klay vs Curry, Bam vs Butler, Gobert vs Mitchell
    ✓ RIGHT: use full career range — backend excludes all shared-team games

  FULL-CAREER DEFAULT: season_from = rookie year of the younger player,
                       season_to   = "2025-26"

  You MUST look up both players with search_player first, then set the range
  based on the younger player's first NBA season. Common rookie years:
    LeBron James          → 2003-04
    Jimmy Butler          → 2011-12
    Steph Curry           → 2009-10
    Kevin Durant          → 2007-08
    Giannis               → 2013-14
    Jayson Tatum          → 2017-18
    Jalen Brunson         → 2018-19
    Luka Doncic           → 2018-19
    Nikola Jokic          → 2015-16
    Kawhi Leonard         → 2011-12
    Andre Iguodala        → 2004-05
    Rudy Gobert           → 2013-14
    Donovan Mitchell      → 2017-18
    Bam Adebayo           → 2017-18

  EXAMPLES — no season qualifier → full career overlap:
    "LeBron vs Curry H2H"          → season_from="2009-10", season_to="2025-26"
    "LeBron vs Jimmy Butler H2H"   → season_from="2011-12", season_to="2025-26"
    "LeBron vs Tatum H2H"          → season_from="2017-18", season_to="2025-26"
    "Tatum vs Brunson H2H"         → season_from="2018-19", season_to="2025-26"
    "KD vs Giannis H2H"            → season_from="2013-14", season_to="2025-26"
    "Iguodala vs Curry H2H"        → season_from="2009-10", season_to="2025-26"
    "Gobert vs Mitchell H2H"       → season_from="2017-18", season_to="2025-26"
  EXAMPLES — with season qualifier → use the specified range only:
    "Tatum vs Brunson H2H this season" → season_from="2025-26", season_to="2025-26"
    "LeBron vs Butler H2H since 2020"  → season_from="2019-20", season_to="2025-26"

▸ IMPOSSIBLE / OUT-OF-SCOPE QUERIES — INVALID (output immediately, no tools):
  If the query is fundamentally impossible or out of scope, do NOT fetch any stats.
  Output ONLY this JSON and nothing else:
    { "type": "invalid", "reason": "<Clear, specific explanation of why this query can't be answered.>" }

  Cases that are always invalid — return immediately:
  • Future season not yet played:
      "Curry 2040 Playoffs", "LeBron 2031-32 stats"
      → reason: "The [YEAR] season hasn't happened yet. Curry's last active season was [YEAR]."

  *** NEVER MARK THESE AS INVALID — they are always valid: ***
  • "last N games in [past season]" — past seasons are fully supported by get_last_n_games.
      "Curry last 10 games in 2022-23" → VALID → use get_last_n_games(season="2022-23")
      "LeBron last 5 games 2020"       → VALID → use get_last_n_games(season="2019-20")
  • Non-NBA athlete or fictional entity:
      "Patrick Mahomes career stats", "LeBron vs Chad Basketball"
      → reason: "[Name] is not an NBA player."
  • Player who retired before the opponent entered the league:
      "Bill Russell vs LeBron H2H", "Wilt Chamberlain vs Jokic H2H"
      → reason: "Russell retired in 1969; LeBron entered the league in 2003 — they never overlapped."
  • Season that predates the stat-tracking era (three-point line didn't exist until 1979-80):
      "Three-point leaderboard for the 1975-76 season"
      → reason: "The NBA three-point line wasn't introduced until the 1979-80 season."
  • Player playing against their own current team:
      "Tatum career stats vs Celtics", "Embiid vs 76ers career"
      → reason: "A player cannot have head-to-head stats against their own team."
  • Self-referential queries:
      "Tatum stats when Tatum is inactive", "Jokic vs Jokic H2H"
      → reason: "A player cannot be compared against themselves."

▸ SAME-TEAM H2H — REQUIRES VERIFICATION BEFORE MARKING INVALID:
  *** NEVER output invalid for same-team H2H without first calling search_player
  for BOTH players to confirm their current teams. Your training data may be stale
  — a player may have been traded recently. ***

  Workflow:
    1. Call search_player for Player A → get current team_abbr
    2. Call search_player for Player B → get current team_abbr
    3. If current teams DIFFER → proceed with H2H stats (they are now opponents)
    4. If current teams MATCH AND they have NEVER been opponents in any era → invalid

  Only output invalid after confirming both players share the same current team:
    { "type": "invalid", "reason": "<Player A> and <Player B> have been <Team> teammates since <year> and have never faced each other as opponents in the NBA." }

  Examples of always-same-team pairs (verify with search_player first):
    Jayson Tatum vs Jaylen Brown       → BOS teammates since 2017, never opponents
    Joel Embiid vs Tyrese Maxey        → PHI teammates since 2020, never opponents
    Nikola Jokic vs Jamal Murray       → DEN teammates since 2016, never opponents
    Steph Curry vs Draymond Green      → GSW teammates, never opponents

  Do NOT output "invalid" for players who WERE teammates but later became opponents:
    KAT & Edwards    — KAT traded to NYK 2024, now opponents ✓
    Bam & Butler     — Butler traded to GSW Feb 2025, now opponents ✓
    Iguodala & Curry — Iguodala on MEM/MIA 2019-21 (opponent era) ✓
    Gobert & Mitchell — both traded from UTA 2022, now opponents ✓
    Klay & Curry     — Klay signed with DAL 2024, now opponents ✓
    Al Horford       — traded from BOS to GSW; former BOS teammates now opponents ✓

▸ INVALID "WHEN INACTIVE" — NEVER TEAMMATES:
  *** Call search_player for BOTH players before marking invalid. A recent trade
  may have made them teammates. Only mark invalid after confirming they have never
  shared a roster in any season. ***

  If confirmed never teammates, output:
    { "type": "invalid", "reason": "<Primary> and <Condition> have never been teammates. 'Stats when X is inactive' only applies to teammates whose absence affects team dynamics." }

  Examples of never-teammate inactive splits (confirm with search_player first):
    Curry stats when LeBron inactive   → never shared a roster
    Jokic stats when Giannis inactive  → never teammates
    Tatum stats when KD inactive       → KD was OKC/GSW/BKN/PHX/HOU, never BOS
    Embiid stats when Jokic inactive   → PHI vs DEN, never teammates
    SGA stats when Luka inactive       → OKC vs DAL/LAL, never teammates

  EXCEPTION — if they became teammates mid-season (e.g. Luka traded to LAL Feb 2025),
  still pass the PRIMARY player's FULL career range (e.g. season_from="2018-19").
  The backend's same-team guard automatically excludes pre-trade games from the wrong
  team, so you will NOT accidentally include Luka's DAL games in a Luka/LeBron split.
  Never narrow season_from to just the trade season or later — that silently drops
  all games from the trade season itself (e.g. the Feb–Apr 2025 LAL window).

▸ Stats WITH a teammate (both active, same team):
  "Luka stats with Brunson" / "Tatum stats when Brown plays" / "LeBron stats with AD"
  → get_conditional_stats(player_id=luka, condition_player_ids=[brunson_id],
      all_active=true, require_opponent=false)
  *** DO NOT set require_opponent=true here — that would exclude all shared-team games
  and return only games when they faced each other as opponents, which is wrong. ***
  → Output type: "conditional"

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

▸ SEASON RANGE FOR CONDITIONAL STATS — CRITICAL:
  ALWAYS pass the PRIMARY player's FULL career range, NOT your guess of when they
  became teammates. The backend automatically filters to valid teammate seasons.
  Passing a narrow range WILL silently drop early seasons and show wrong date headers.

  season_from = primary player's rookie season
  season_to   = "2025-26"

  EXAMPLES:
    "Westbrook stats without KD"   → season_from="2008-09", season_to="2025-26"
    "KD stats without Westbrook"   → season_from="2007-08", season_to="2025-26"
    "LeBron stats without AD"      → season_from="2003-04", season_to="2025-26"
    "Giannis stats without Middleton" → season_from="2013-14", season_to="2025-26"
    "Booker stats without KD"      → season_from="2015-16", season_to="2025-26"
    "Embiid stats without Harden"  → season_from="2016-17", season_to="2025-26"

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
  • Write insight as plain prose — NO markdown. No **bold**, no *italic*, no bullet points
    starting with * or -. Use plain sentences and paragraph breaks only.
  • For H2H: note who had the edge and in what categories.
  • H2H TEAM ERA CONTEXT (critical for accuracy):
    - ALWAYS state which teams each player was on during the H2H games, not just their
      current or most well-known team.
    - If the H2H games span multiple eras where a player was on DIFFERENT teams
      (e.g. Iguodala: PHI → DEN → [GSW teammate gap] → MEM → MIA), list every team
      they represented in these matchups.
    - If the two players were TEAMMATES at some point, explicitly say so and clarify
      that the stats exclude their shared tenure. e.g. "These 10 games exclude their
      6-year run as Golden State Warriors teammates from 2013 to 2019."
    - If a player had MULTIPLE SEPARATE STINTS as opponents (e.g. opposed before AND
      after a teammate era), name both windows explicitly. e.g. "Iguodala faced Curry
      as a 76er and Nugget from 2009-13, and again as a Heat player in 2020-21 —
      bookending their championship run together."
    - Never say "before he joined X" if the player also returned to X afterward.
  • TERMINOLOGY: always say "win %" or "win percentage" — NEVER "win rate".
    e.g. "Tatum's win % in these games was 68%" not "win rate of 68%"
  • TEAM CONTEXT: always mention which team a player was on for the stats shown.
    e.g. "Kawhi's 2014 numbers came as a key piece of the San Antonio Spurs,
    alongside Tim Duncan and Tony Parker" — never just cite numbers in a vacuum.
  • TRADE CONTEXT: if a player has been traded and the stats shown are entirely from
    their previous team, say so explicitly. e.g. "All of these games were played while
    Davis was with the Los Angeles Lakers; he has not faced the Celtics since being
    traded to the Washington Wizards in February 2026." This explains why the player's
    current team color differs from the team in the stats.
  • LEADERBOARD QUALIFIER: when presenting a per-game leaderboard, note the minimum
    games played threshold used (available as min_gp in the tool result). e.g. "Rankings
    require a minimum of 58 games played to qualify." This helps users understand why
    some players with high per-game averages over small samples may not appear.
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
  "season_range": "2017-18 to 2025-26",  // ALWAYS use primary player's full career range — backend will override with real computed range

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

COMPARE TITLE FORMAT:
- All players SAME season+type → mention the season ONCE at the end, not per player:
    "Stephen Curry vs LeBron James · 2018 Playoffs"   ← CORRECT
    "Stephen Curry 2018 Playoffs vs LeBron James 2018 Playoffs"  ← WRONG
- Different seasons/types → include the season per player:
    "Kobe 2010 Playoffs vs Tatum 2024 Playoffs"  ← CORRECT

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
  "is_percentage": false,
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
      "fg3a": 6.2,
      "stat_value": 27.5
    }
  ],
  "insight": "Brief insight about who leads and any notable trends."
}
```

─── TYPE: last_n_games ───────────────────────────────────────
```json
{
  "type": "last_n_games",
  "player": { ...player object... },
  "n_requested": 10,
  "n_returned": 10,
  "season": "2025-26",          // use whatever season the user asked for
  "season_type": "Regular Season",
  "order": "last",              // "last" or "first"
  "period_label": "Last 10 games of 2025-26",   // human-readable label, e.g. "First 10 games of 2025-26" or "Jan 1, 2026 – Mar 1, 2026"
  "date_from": "",              // ISO date if a start date was requested, else ""
  "date_to": "",                // ISO date if an end date was requested, else ""
  "last_game_date": "2026-04-10",
  "averages": {
    "gp": 10, "pts": 0.0, "reb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0,
    "tov": 0.0, "fg_pct": 0.0, "fg3_pct": 0.0, "fg3a": 0.0, "ft_pct": 0.0,
    "wins": 0, "losses": 0
  },
  "game_log": [],
  "insight": "Concise analysis of the player's recent form and standout performances."
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
# Pre-flight interceptor for last-N-games queries
# ─────────────────────────────────────────────────────────────────────────────
# Gemini has a strong training-data prior that 'last N games' only works for
# the current season, and it ignores system-prompt instructions to the contrary.
# To guarantee past-season queries work, we detect the pattern here and bypass
# the LLM routing entirely, calling the NBA API directly.

# ── Regex for "last N games in <season>" ─────────────────────────────────────
# Matches "Curry last 10 games in 2022-23" and "LeBron last 5 playoff games 2020"
_LAST_N_RE = re.compile(
    r'^(.+?)\s+last\s+(\d+)\s+(playoff\s+)?games?'   # player + N + optional "playoff"
    r'(?:\s+(?:in|during|of(?:\s+the)?|from))?\s*'   # optional preposition
    r'((?:19|20)\d{2}(?:-\d{2})?)',                   # year or season string
    re.IGNORECASE,
)

# ── Regex for "first N games [of <season>]" ──────────────────────────────────
# Matches "Giannis first 10 games of 2025-26" and "Curry first 5 games this season"
_FIRST_N_RE = re.compile(
    r'^(.+?)\s+first\s+(\d+)\s+(playoff\s+)?games?'  # player + N + optional "playoff"
    r'(?:\s+(?:in|during|of(?:\s+the)?|from|this))?\s*'   # optional preposition
    r'((?:19|20)\d{2}(?:-\d{2})?|season)?',           # year/season or bare "season"
    re.IGNORECASE,
)


def _parse_last_n_query(query: str) -> dict | None:
    """
    _parse_last_n_query
    -------------------
    Detect and parse a 'last N games in <past season>' or 'first N games [of <season>]'
    query pattern, bypassing the Gemini router.

    Gemini has strong training-data priors that break both of these patterns:
      • "last N games" → it assumes the current season only, ignoring the past-season signal
      • "first N games" → it returns the last N games (wrong direction)

    This interceptor catches both patterns and routes them directly to the NBA API.

    Matches queries like:
        "Curry last 10 games in 2022-23"        → order='last', season='2022-23'
        "LeBron last 5 playoff games 2020"      → order='last', season='2019-20', playoffs
        "Giannis first 10 games of 2025-26"     → order='first', season='2025-26'
        "Tatum first 5 games this season"       → order='first', season=DEFAULT_SEASON
        "Giannis first 5 games in 2021"         → order='first', season='2020-21'

    Note on bare year resolution: a bare year like "2020" is treated as a season
    start year → "2020-21". This mirrors the YEAR_TO_SEASON convention used
    throughout the app.

    Args:
        query (str): Raw user query string.

    Returns:
        dict | None: Parsed params {"player_name", "n", "season", "season_type", "order"}
                     or None if the pattern didn't match.
    """
    # Check "last N games in <past season>" first
    m = _LAST_N_RE.search(query)
    if m:
        player_name = m.group(1).strip()
        n           = int(m.group(2))
        is_playoff  = bool(m.group(3))
        year_str    = m.group(4)

        season = YEAR_TO_SEASON.get(year_str)
        if not season or not player_name:
            return None

        # Current season "last N games" is fine for Gemini to handle
        if season == DEFAULT_SEASON:
            return None

        return {
            "player_name": player_name,
            "n":           n,
            "season":      season,
            "season_type": "Playoffs" if is_playoff else "Regular Season",
            "order":       "last",
        }

    # Check "first N games [of <season>]"
    m = _FIRST_N_RE.search(query)
    if m:
        player_name = m.group(1).strip()
        n           = int(m.group(2))
        is_playoff  = bool(m.group(3))
        year_str    = (m.group(4) or "").strip().lower()

        if not player_name:
            return None

        # Resolve season: bare "season" keyword or absent → current season
        if not year_str or year_str in ("season", "this", "the"):
            season = DEFAULT_SEASON
        else:
            season = YEAR_TO_SEASON.get(year_str)
            if not season:
                return None

        return {
            "player_name": player_name,
            "n":           n,
            "season":      season,
            "season_type": "Playoffs" if is_playoff else "Regular Season",
            "order":       "first",
        }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class NBAStatsAgent:
    def __init__(self):
        # Reads GOOGLE_API_KEY from environment automatically.
        self.client = genai.Client()
        logger.info(f"NBAStatsAgent initialized ({GEMINI_MODEL})")

    def _run_last_n_direct(
        self,
        params: dict,
        progress_cb: Callable[[str, str], None],
    ) -> dict:
        """
        _run_last_n_direct
        ------------------
        Handle a 'last N games in <past season>' query without Gemini routing.

        Calls search_player and get_player_last_n_games directly from Python,
        then returns a result dict in the same structure as the Gemini-produced
        last_n_games output so the frontend renders identically.

        Args:
            params (dict): Parsed query params from _parse_last_n_query —
                           {"player_name", "n", "season", "season_type"}.
            progress_cb:   SSE progress callback (event_type, message).

        Returns:
            dict | None: {"success": bool, "result": dict, "narrative": str, "usage": dict},
                         or None if the player name was too ambiguous (signals run() to
                         fall through to the Gemini loop for proper disambiguation).
        """
        player_name = params["player_name"]
        n           = params["n"]
        season      = params["season"]
        season_type = params["season_type"]
        order       = params.get("order", "last")
        date_from   = params.get("date_from", "")
        date_to     = params.get("date_to", "")
        label       = "Playoffs" if season_type == "Playoffs" else "Regular Season"
        order_word  = "first" if order == "first" else "last"

        progress_cb("thinking", "Step 1: reasoning…")

        # ── Step 1: resolve player ────────────────────────────────────────────
        progress_cb("tool_start", f"🔍 Looking up: **{player_name}**")
        player = nba.search_player(player_name)
        if not player:
            err = f"Player '{player_name}' not found."
            progress_cb("error", err)
            return {"success": False, "error": err}

        info   = nba.get_player_info(player["id"])
        awards = nba.get_player_awards(player["id"])
        progress_cb(
            "tool_result",
            f"✅ **{player['full_name']}** (#{info.get('jersey','?')}, {info.get('team','')})",
        )

        progress_cb("thinking", "Step 2: reasoning…")

        # ── Step 2: fetch game log ────────────────────────────────────────────
        progress_cb(
            "tool_start",
            f"📅 Fetching {order_word} **{n} games** ({season} {label})…",
        )
        raw = nba.get_player_last_n_games(
            player["id"], n, season, season_type,
            order=order, date_from=date_from, date_to=date_to,
        )

        n_ret     = raw.get("n_returned", 0)
        pts       = (raw.get("averages") or {}).get("pts", 0)
        log_label = " Playoffs" if season_type == "Playoffs" else ""
        progress_cb("tool_result", f"✅ {n_ret} games ({season}{log_label}) — **{pts} PPG** average")

        if raw.get("error") and not raw.get("game_log"):
            # If the player resolved correctly but just has no games in this season,
            # it's a real result. But if a bare name like "Curry" matched the wrong
            # player (e.g. Dell Curry instead of Stephen Curry), n_returned will be
            # 0 because the retired player has no games. Fall back to Gemini in that
            # case so it can disambiguate the name properly from context.
            if not player.get("is_active", True):
                logger.info(
                    f"[last-n-direct] inactive player '{player['full_name']}' returned "
                    f"0 games for {season} — falling back to Gemini for name disambiguation."
                )
                return None   # signals run() to fall through to Gemini
            err = raw["error"]
            progress_cb("error", err)
            return {"success": False, "error": err}

        # ── Build result ──────────────────────────────────────────────────────
        avgs = raw.get("averages") or {}
        reb  = avgs.get("reb", 0)
        ast  = avgs.get("ast", 0)

        insight = (
            f"{player['full_name']} averaged {pts} PPG, {reb} RPG, and {ast} APG "
            f"in {n_ret} {season} {label.lower()} games."
        )

        result = {
            "type":           "last_n_games",
            "player": {
                "name":        player["full_name"],
                "player_id":   player["id"],
                "team":        info.get("team", ""),
                "team_abbr":   info.get("team_abbr", ""),
                "position":    info.get("position", ""),
                "headshot_url": info.get("headshot_url", nba.headshot_url(player["id"])),
                "awards":      awards,
            },
            "season":         raw.get("season",       season),
            "season_type":    raw.get("season_type",  season_type),
            "order":          raw.get("order",        order),
            "n_requested":    raw.get("n_requested",  n),
            "n_returned":     n_ret,
            "last_game_date": raw.get("last_game_date", ""),
            "period_label":   raw.get("period_label",   ""),
            "date_from":      raw.get("date_from",      date_from),
            "date_to":        raw.get("date_to",        date_to),
            "averages":       avgs,
            "game_log":       raw.get("game_log", []),
            "insight":        insight,
        }

        logger.info(
            f"[last-n-direct] {player['full_name']} {order_word} {n_ret} games "
            f"{season} {season_type} — {pts} PPG"
        )
        return {
            "success":   True,
            "result":    result,
            "narrative": "",
            "usage":     {"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
        }

    def run(self, query: str, progress_cb: Callable[[str, str], None]) -> dict:
        # Pre-flight: past-season 'last N games' queries bypass the Gemini router
        # entirely. Gemini has a strong training-data prior that this feature only
        # works for the current season, and it outputs a false "invalid" response
        # even when the system prompt explicitly permits past seasons. Intercepting
        # here ensures reliable results regardless of model behaviour.
        last_n_params = _parse_last_n_query(query)
        if last_n_params:
            logger.info(f"[pre-flight] last-n-games intercepted: {last_n_params}")
            direct_result = self._run_last_n_direct(last_n_params, progress_cb)
            if direct_result is not None:
                return direct_result
            # None → player name was ambiguous; fall through to Gemini for resolution

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
        empty_stop_retries  = 0

        # ── Usage / cost accumulators ────────────────────────────────────────
        total_input_tokens  = 0
        total_output_tokens = 0
        api_call_count      = 0

        logger.info(f"[DEBUG] Agent run — query: {query!r}")

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
                        in_tok  = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                        out_tok = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                        total_input_tokens  += in_tok
                        total_output_tokens += out_tok
                        logger.info(f"[DEBUG] iter={iteration} tokens in={in_tok} out={out_tok} total_out={total_output_tokens}")
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
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        # Transient server overload — retry with a short backoff
                        wait = 8 * (attempt + 1)
                        logger.warning(f"Gemini 503 — retrying in {wait}s (attempt {attempt+1})")
                        progress_cb("thinking", f"Server busy — retrying in {wait}s…")
                        time.sleep(wait)
                        if attempt == 2:
                            logger.error("[DEBUG] Gemini 503 after 3 attempts — returning error")
                            progress_cb("error", err_str)
                            return {"success": False, "error": "Gemini is currently overloaded. Please try again in a few minutes."}
                    else:
                        logger.error(f"[DEBUG] Gemini API error (iter={iteration} attempt={attempt+1}): {type(e).__name__}: {e}")
                        progress_cb("error", err_str)
                        return {"success": False, "error": err_str}

            if response is None or not response.candidates:
                # Check if the prompt was blocked by safety filters
                feedback = getattr(response, "prompt_feedback", None) if response else None
                block_reason = getattr(feedback, "block_reason", None) if feedback else None
                if block_reason:
                    err = f"Query blocked by content filter ({block_reason}). Try rephrasing."
                    logger.error(f"[DEBUG] Prompt blocked: {block_reason}")
                    progress_cb("error", err)
                    return {"success": False, "error": err}
                logger.error("[DEBUG] Gemini returned no candidates — breaking loop")
                break

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            parts = (candidate.content and candidate.content.parts) or []

            # Empty response (0 tokens, STOP) — can be a transient model glitch or
            # silent content filter. Retry up to 2 times with a small delay.
            if not parts and str(finish_reason) in ("FinishReason.STOP", "STOP", "1"):
                if empty_stop_retries < 2:
                    empty_stop_retries += 1
                    logger.warning(f"[DEBUG] Empty STOP response at iter={iteration} — retrying ({empty_stop_retries}/2)")
                    time.sleep(1)
                    continue
                else:
                    err = "The AI model returned an empty response for this query. Try rephrasing or searching again."
                    logger.error("[DEBUG] Empty STOP response after 2 retries — returning error")
                    progress_cb("error", err)
                    return {"success": False, "error": err}

            # Filter out thought parts (Gemini 2.5 Flash is a thinking model;
            # thought=True parts contain internal reasoning, not visible output).
            text_parts = [p for p in parts if p.text and not getattr(p, "thought", False)]
            fn_parts   = [p for p in parts if p.function_call]

            logger.info(
                f"[DEBUG] iter={iteration} finish_reason={finish_reason} "
                f"text_parts={len(text_parts)} fn_parts={len(fn_parts)} "
                f"total_parts={len(parts)}"
            )

            # Warn loudly if the model was cut off by the token limit
            if finish_reason is not None and str(finish_reason) in ("MAX_TOKENS", "2"):
                logger.warning(
                    f"[DEBUG] *** MAX_TOKENS hit at iter={iteration} — "
                    f"output TRUNCATED. Increase AGENT_MAX_TOKENS (currently {AGENT_MAX_TOKENS}). ***"
                )

            # MALFORMED_FUNCTION_CALL: Gemini generated a tool call that failed
            # its own schema validation (commonly happens with get_multi_player_stats
            # when the nested players array has 4+ entries). Inject a retry message
            # with explicit formatting constraints so Gemini can self-correct.
            if finish_reason is not None and "MALFORMED_FUNCTION_CALL" in str(finish_reason):
                logger.warning(
                    f"[DEBUG] MALFORMED_FUNCTION_CALL at iter={iteration} — injecting retry message"
                )
                contents.append(types.Content(role="user", parts=[types.Part(text=(
                    "Your last function call was malformed and could not be executed. "
                    "Please call get_multi_player_stats again with strictly valid arguments:\n"
                    "• players must be an array of objects\n"
                    "• Each object MUST have: player_id (integer), season_from (string), "
                    "season_to (string)\n"
                    "• Optional fields: season_type (string), label (string)\n"
                    "• Do NOT include any extra fields or null values\n"
                    "• player_id must be a plain integer — no quotes, no decimals\n"
                    "Example: {\"player_id\": 2544, \"season_from\": \"2003-04\", "
                    "\"season_to\": \"2025-26\", \"season_type\": \"Regular Season\", "
                    "\"label\": \"LeBron Career\"}"
                ))]))
                continue

            if text_parts:
                last_text = " ".join(p.text for p in text_parts)
                logger.info(f"[DEBUG] iter={iteration} last_text length={len(last_text)} has_json_block={'```json' in last_text}")
                excerpt   = last_text[:280].replace("\n", " ")
                if "```json" not in last_text[:50]:
                    progress_cb("message", excerpt)

            # No function calls → model is done
            if not fn_parts:
                logger.info(f"[DEBUG] iter={iteration} — no fn_parts, model finished")
                break

            # Append model's full response (text + function calls) to history
            contents.append(candidate.content)

            # Execute every function call in this turn
            fn_response_parts = []
            for idx, part in enumerate(fn_parts):
                fn      = part.function_call
                fn_args = dict(fn.args)

                logger.info(f"[DEBUG] calling tool={fn.name} args_keys={list(fn_args.keys())}")
                if fn.name == "get_multi_player_stats":
                    players_cfg = fn_args.get("players", [])
                    logger.info(f"[DEBUG] get_multi_player_stats: {len(players_cfg)} players: "
                                + str([{"pid": p.get("player_id"), "from": p.get("season_from"), "to": p.get("season_to")} for p in players_cfg]))

                progress_cb("tool_start", _describe_tool(fn.name, fn_args))
                try:
                    result = execute_tool(fn.name, fn_args)
                    # Log tool errors (the tool returned {"error": ...})
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"[DEBUG] tool={fn.name} returned error: {result['error']}")
                    else:
                        logger.info(f"[DEBUG] tool={fn.name} success — top-level keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
                    progress_cb("tool_result", _summarise(fn.name, result))
                except Exception as e:
                    import traceback
                    logger.error(f"[DEBUG] tool={fn.name} EXCEPTION: {type(e).__name__}: {e}")
                    logger.error(f"[DEBUG] traceback:\n{traceback.format_exc()}")
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

        logger.info(f"[DEBUG] Loop ended — last_text length={len(last_text)} iters_used={iteration}")
        result_data = _extract_json(last_text)
        narrative   = _extract_narrative(last_text)

        if not result_data:
            logger.error(f"[DEBUG] _extract_json returned empty — success will be False")
            logger.error(f"[DEBUG] last_text[:800]:\n{last_text[:800]}")
            logger.error(f"[DEBUG] last_text[-400:]:\n{last_text[-400:]}")

        if result_data:
            logger.info(f"[DEBUG] result_data type={result_data.get('type')} keys={list(result_data.keys())}")
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
        # Override with authoritative raw tool data so the model can't reformat
        # percentages (0.459 → 45.9) or drop fields like is_percentage / fg3a.
        raw_lb = next(
            (e["result"] for e in cache.values() if e.get("name") == "get_leaderboard"),
            None,
        )
        if raw_lb and raw_lb.get("leaders"):
            # Carry over all metadata fields the model template doesn't include
            for field in ("stat_category", "stat_label", "season", "season_type",
                          "per_mode", "min_gp", "is_percentage",
                          "season_fallback_used", "season_requested"):
                if field in raw_lb:
                    result[field] = raw_lb[field]
            # Replace the leaders array entirely with the authoritative data,
            # but preserve the model's insight by keeping result["insight"].
            model_leaders = {r.get("rank"): r for r in result.get("leaders", [])}
            merged = []
            for raw_row in raw_lb["leaders"]:
                row = dict(raw_row)
                rank = row.get("rank")
                if rank in model_leaders:
                    # Use model name/team if they differ (model may have better formatting)
                    pass  # raw data is authoritative for all numeric fields
                merged.append(row)
            result["leaders"] = merged
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

    search_player_results = []
    for entry in cache.values():
        name = entry.get("name", "")
        raw  = entry.get("result", {})
        if name == "get_stats_vs_team":
            vs_team_results.append(raw)
        elif name == "get_conditional_stats":
            cond_results.append(raw)
        elif name == "get_career_stats":
            career_results.append(raw)
        elif name == "search_player" and raw.get("found"):
            search_player_results.append(raw)

    # NOTE: For ALL types that contain percentage averages we override from the
    # raw tool cache.  Gemini re-formats the backend's 0–100 scale values (e.g.
    # fg_pct=47.2) as 0–1 decimals (e.g. 0.472) in its JSON output, which
    # causes donut charts and stat cards to display 0.5% instead of 47.2%.
    # Overriding from the cache bypasses the model's reformatting entirely.

    # Always override player.team / player.team_abbr from the search_player tool
    # result — the model frequently uses stale team data from its training weights
    # (e.g. showing "Lakers" for Anthony Davis after he was traded to Washington).
    # Match by player_id so we don't accidentally apply AD's team to LeBron when
    # both are searched in the same query.
    if search_player_results and result.get("player"):
        primary_pid = result["player"].get("player_id") or result["player"].get("id")
        sp = next(
            (s for s in reversed(search_player_results) if s.get("player_id") == primary_pid),
            None,
        )
        if sp is None:
            sp = search_player_results[-1]   # fallback: single-player queries
        # is_active=False is the reliable retired signal — retired players often
        # still have a last-team entry in the NBA API (e.g. Iguodala → Warriors),
        # so checking for empty team alone is insufficient.
        if sp.get("is_active") is False:
            result["player"]["retired"] = True
        if sp.get("team"):
            result["player"]["team"]      = sp["team"]
            result["player"]["team_abbr"] = sp["team_abbr"]
        elif not result["player"].get("retired"):
            result["player"]["retired"] = True  # no team and not flagged yet

    # Apply the same current-team override to opponent_player (H2H queries).
    # Without this the LLM uses stale team data (e.g. Luka showing DAL after
    # his trade to LAL).
    if search_player_results and result.get("opponent_player"):
        opp_pid = result["opponent_player"].get("player_id") or result["opponent_player"].get("id")
        osp = next(
            (s for s in reversed(search_player_results) if s.get("player_id") == opp_pid),
            None,
        )
        if osp:
            if osp.get("is_active") is False:
                result["opponent_player"]["retired"] = True
            if osp.get("team"):
                result["opponent_player"]["team"]      = osp["team"]
                result["opponent_player"]["team_abbr"] = osp["team_abbr"]
            elif not result["opponent_player"].get("retired"):
                result["opponent_player"]["retired"] = True

    if result_type == "vs_team" and vs_team_results:
        raw = vs_team_results[-1]
        if "game_log" in raw:  result["game_log"] = raw["game_log"]
        # Override averages so fg_pct etc. stay on the 0–100 scale
        if "averages" in raw:  result["averages"] = raw["averages"]
        # Pass through playoff data (model may omit or reformat these)
        for key in ("playoff_averages", "playoff_game_log", "playoff_splits", "playoff_total_games"):
            if key in raw:
                result[key] = raw[key]

    elif result_type == "conditional" and cond_results:
        raw = cond_results[-1]
        if "primary_log"         in raw: result["primary_log"]         = raw["primary_log"]
        if "comparison_log"      in raw: result["comparison_log"]      = raw["comparison_log"]
        if "primary_games"       in raw: result["primary_games"]       = raw["primary_games"]
        if "comparison_games"    in raw: result["comparison_games"]    = raw["comparison_games"]
        if "primary_averages"    in raw: result["primary_averages"]    = raw["primary_averages"]
        if "comparison_averages" in raw: result["comparison_averages"] = raw["comparison_averages"]
        # Override LLM's season_range with backend-computed value from actual game data
        if raw.get("computed_season_range"):
            result["season_range"] = raw["computed_season_range"]

        # Derive historical team_abbr from game log matchup data so the color
        # reflects the era (e.g. Westbrook on OKC, not current Kings).
        # Use the most common team abbreviation across all rows (matchup[:3]).
        # Save the current team so the UI can show both when they differ.
        if result.get("player"):
            all_rows = list(raw.get("primary_log") or []) + list(raw.get("comparison_log") or [])
            if all_rows:
                from collections import Counter
                abbr_counts = Counter(
                    r.get("matchup", "")[:3].upper()
                    for r in all_rows
                    if r.get("matchup", "")[:3].strip()
                )
                if abbr_counts:
                    era_abbr = abbr_counts.most_common(1)[0][0]
                    current_abbr = result["player"].get("team_abbr", "")
                    if era_abbr and era_abbr != current_abbr:
                        # Preserve current team for UI display, override color team
                        result["player"]["current_team_abbr"] = current_abbr
                        result["player"]["current_team"] = result["player"].get("team", "")
                    result["player"]["team_abbr"] = era_abbr

    elif result_type == "career" and career_results:
        raw = career_results[-1]
        # Override season rows — model may reformat fg_pct values in these too
        if "seasons"         in raw: result["seasons"]         = raw["seasons"]
        if "playoff_seasons" in raw: result["playoff_seasons"] = raw["playoff_seasons"]
        # Override career/playoff totals — these drive the donut charts
        if "career_totals"   in raw: result["career_totals"]   = raw["career_totals"]
        if "playoff_totals"  in raw: result["playoff_totals"]  = raw["playoff_totals"]
        # Career highs — the model never outputs this, always pull from cache
        if "career_highs"         in raw: result["career_highs"]         = raw["career_highs"]
        if "regular_season_best"  in raw: result["regular_season_best"]  = raw["regular_season_best"]
        if "playoff_best"         in raw: result["playoff_best"]         = raw["playoff_best"]
        if "playoff_career_highs" in raw: result["playoff_career_highs"] = raw["playoff_career_highs"]

    elif result_type == "career" and not career_results:
        # Gemini skipped the get_career_stats tool call and output the empty JSON
        # template directly. Auto-fetch career data from the NBA API using the
        # player_id already resolved by search_player in this same query.
        player_id = (
            result.get("player", {}).get("player_id")
            or result.get("player", {}).get("id")
            or (search_player_results[-1].get("player_id") if search_player_results else None)
        )
        if player_id:
            logger.warning(
                f"[career-fallback] Gemini skipped get_career_stats for player_id={player_id}; "
                "fetching directly from NBA API."
            )
            raw = nba.get_player_career_stats(int(player_id))
            if "seasons"              in raw: result["seasons"]              = raw["seasons"]
            if "playoff_seasons"      in raw: result["playoff_seasons"]      = raw["playoff_seasons"]
            if "career_totals"        in raw: result["career_totals"]        = raw["career_totals"]
            if "playoff_totals"       in raw: result["playoff_totals"]       = raw["playoff_totals"]
            if "career_highs"         in raw: result["career_highs"]         = raw["career_highs"]
            if "regular_season_best"  in raw: result["regular_season_best"]  = raw["regular_season_best"]
            if "playoff_best"         in raw: result["playoff_best"]         = raw["playoff_best"]
            if "playoff_career_highs" in raw: result["playoff_career_highs"] = raw["playoff_career_highs"]

    elif result_type == "last_n_games":
        # Pull the authoritative game log and averages from the tool cache so the
        # model cannot reformat or truncate the data.
        last_n_results = [
            entry["result"]
            for entry in cache.values()
            if entry.get("name") == "get_last_n_games"
        ]
        if last_n_results:
            raw = last_n_results[-1]
            if "game_log"     in raw: result["game_log"]     = raw["game_log"]
            if "averages"     in raw: result["averages"]     = raw["averages"]
            if "n_returned"   in raw: result["n_returned"]   = raw["n_returned"]
            if "n_requested"  in raw: result["n_requested"]  = raw["n_requested"]
            if "season"       in raw: result["season"]       = raw["season"]
            if "season_type"  in raw: result["season_type"]  = raw["season_type"]
            if "order"        in raw: result["order"]        = raw["order"]
            if "period_label" in raw: result["period_label"] = raw["period_label"]
            if "date_from"    in raw: result["date_from"]    = raw["date_from"]
            if "date_to"      in raw: result["date_to"]      = raw["date_to"]
            if raw.get("last_game_date"):
                result["last_game_date"] = raw["last_game_date"]

    elif result_type == "h2h":
        if len(cond_results) >= 2:
            result["player_log"]     = cond_results[0].get("primary_log",    result.get("player_log",   []))
            result["opponent_log"]   = cond_results[1].get("primary_log",    result.get("opponent_log", []))
            result["player_games"]   = cond_results[0].get("primary_games",  result.get("player_games",   0))
            result["opponent_games"] = cond_results[1].get("primary_games",  result.get("opponent_games", 0))
            # Override averages to preserve the 0–100 percentage scale
            if "primary_averages" in cond_results[0]:
                result["player_averages"]   = cond_results[0]["primary_averages"]
            if "primary_averages" in cond_results[1]:
                result["opponent_averages"] = cond_results[1]["primary_averages"]
        elif len(cond_results) == 1:
            result["player_log"]   = cond_results[0].get("primary_log",    result.get("player_log",   []))
            result["opponent_log"] = cond_results[0].get("comparison_log", result.get("opponent_log", []))
            if "primary_averages"    in cond_results[0]:
                result["player_averages"]   = cond_results[0]["primary_averages"]
            if "comparison_averages" in cond_results[0]:
                result["opponent_averages"] = cond_results[0]["comparison_averages"]

        # Era team detection for H2H: collect ALL distinct teams a player was on
        # during the H2H games (in chronological order), and use the most recent
        # for the card color. When a player had multiple opponent eras (e.g. Iguodala
        # on PHI → DEN → [GSW teammate gap] → MEM → MIA), all teams are stored so
        # the UI can display them as pill tags.
        def _h2h_era_team(player_obj: dict, game_log: list) -> None:
            if not player_obj or not game_log:
                return
            # date_sort is YYYY-MM-DD — chronological sort
            sorted_log = sorted(game_log, key=lambda r: r.get("date_sort", ""))
            # Collect unique teams in chronological order (first appearance per team)
            seen: set  = set()
            era_abbrs: list = []
            for row in sorted_log:
                abbr = ((row.get("matchup") or "")[:3].upper().strip())
                if abbr and abbr not in seen:
                    seen.add(abbr)
                    era_abbrs.append(abbr)
            if not era_abbrs:
                return
            recent_abbr  = era_abbrs[-1]   # most recent era → card color
            current_abbr = player_obj.get("team_abbr", "")
            # Store all opponent-era teams for the UI pill tags
            player_obj["era_teams"] = [
                {"abbr": a, "name": _H2H_TEAM_NAMES.get(a, a)} for a in era_abbrs
            ]
            if recent_abbr != current_abbr:
                # Era differs from current: save current as secondary label
                player_obj["current_team_abbr"] = current_abbr
                player_obj["current_team"]      = player_obj.get("team", "")
            player_obj["team_abbr"] = recent_abbr
            player_obj["team"]      = _H2H_TEAM_NAMES.get(recent_abbr, recent_abbr)

        _h2h_era_team(result.get("player"),          result.get("player_log",   []))
        _h2h_era_team(result.get("opponent_player"), result.get("opponent_log", []))

    # ── Attach awards/accolades to every player object in the result ──────────
    # Collect all player objects that have a player_id, then fetch awards once
    # per unique player_id so we never double-fetch (H2H has two players).
    player_objects: list = []
    if result.get("player") and result["player"].get("player_id"):
        player_objects.append(result["player"])
    if result.get("opponent_player") and result["opponent_player"].get("player_id"):
        player_objects.append(result["opponent_player"])
    # compare type: players list
    for rp in result.get("players", []):
        if rp.get("player_id"):
            player_objects.append(rp)

    seen_pids: set = set()
    awards_cache: dict = {}
    for pobj in player_objects:
        pid = pobj.get("player_id")
        if not pid or pid in seen_pids:
            continue
        seen_pids.add(pid)
        try:
            awards = nba.get_player_awards(int(pid))
            awards_cache[pid] = awards
        except Exception as e:
            logger.warning(f"awards fetch failed pid={pid}: {e}")
            awards_cache[pid] = []

    for pobj in player_objects:
        pid = pobj.get("player_id")
        if pid in awards_cache:
            pobj["awards"] = awards_cache[pid]

    return result


def _season_range_label(inp: dict) -> str:
    """Return a readable season range string like '2018-19 → 2025-26'."""
    s_from = inp.get("season_from", "")
    s_to   = inp.get("season_to", "")
    if s_from and s_to and s_from != s_to:
        return f"{s_from} → {s_to}"
    if s_from:
        return s_from
    return ""


def _describe_tool(name: str, inp: dict) -> str:
    if name == "search_player":   return f"🔍 Looking up: **{inp.get('name')}**"
    if name == "search_team":     return f"🏀 Looking up team: **{inp.get('name')}**"
    if name == "get_stats_vs_team":
        rng = _season_range_label(inp)
        return f"📊 Fetching stats vs **{inp.get('opponent_abbr')}**" + (f" ({rng})" if rng else "")
    if name == "get_conditional_stats":
        active = inp.get("all_active", True)
        rng    = _season_range_label(inp)
        ids    = inp.get("condition_player_ids", [])
        if active and inp.get("require_opponent"):
            label = "H2H matchups"
        elif active:
            label = "shared games (both active)"
        else:
            label = f"inactive split ({len(ids)} player{'s' if len(ids)>1 else ''})"
        return f"⚖️ Computing {label}" + (f" · **{rng}**" if rng else "") + "…"
    if name == "get_career_stats":
        rng  = _season_range_label(inp)
        s_type = inp.get("season_type", "")
        parts = ["📈 Loading career stats"]
        if rng:   parts.append(f"**{rng}**")
        if s_type and s_type != "Regular Season": parts.append(s_type)
        return " · ".join(parts) + "…"
    if name == "get_multi_player_stats":
        players = inp.get("players", [])
        n = len(players)
        # Show season range from first player config if available
        rng = _season_range_label(players[0]) if players else ""
        return f"📊 Fetching stats for {n} player{'s' if n != 1 else ''}" + (f" · **{rng}**" if rng else "") + "…"
    if name == "get_last_n_games":
        n           = inp.get("n")
        season      = inp.get("season", "2025-26")
        season_type = inp.get("season_type", "Regular Season")
        order       = inp.get("order", "last")
        date_from   = inp.get("date_from", "")
        date_to     = inp.get("date_to", "")
        type_label  = "Playoffs" if season_type == "Playoffs" else "Regular Season"
        if date_from or date_to:
            d_label = f"{date_from or '?'} → {date_to or '?'}"
            return f"📅 Fetching games **{d_label}** ({season} {type_label})…"
        order_word = "first" if order == "first" else "last"
        n_label    = f"**{n} games**" if n else "**all games**"
        return f"📅 Fetching {order_word} {n_label} ({season} {type_label})…"
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
    if name == "get_last_n_games":
        n_ret       = result.get("n_returned", 0)
        pts         = (result.get("averages") or {}).get("pts", 0)
        season      = result.get("season", "")
        season_type = result.get("season_type", "Regular Season")
        label       = " Playoffs" if season_type == "Playoffs" else ""
        return f"✅ {n_ret} games ({season}{label}) — **{pts} PPG** average"
    return "✅ Done"


def _extract_json(text: str) -> dict:
    import re

    def _try_parse(raw: str) -> dict:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"[DEBUG] JSON parse error: {e}")
            char = getattr(e, "pos", 0)
            snippet = raw[max(0, char - 80): char + 80]
            logger.error(f"[DEBUG] JSON error near char {char}: ...{snippet!r}...")
            logger.error(f"[DEBUG] Full JSON block length={len(raw)} first300={raw[:300]!r}")
        return {}

    # 1. Prefer a properly closed ```json ... ``` fence
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        result = _try_parse(m.group(1))
        if result:
            return result

    # 2. Gemini sometimes emits ```json ... { } without the closing fence.
    #    Find the opening fence and grab everything from the first { to the last }.
    fence_idx = text.find("```json")
    if fence_idx != -1:
        after_fence = text[fence_idx + 7:]  # skip "```json"
        brace_start = after_fence.find("{")
        brace_end   = after_fence.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            result = _try_parse(after_fence[brace_start: brace_end + 1])
            if result:
                logger.warning("[DEBUG] Recovered JSON from unclosed ```json fence")
                return result

    # 3. Last resort: bare JSON object in the text
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        result = _try_parse(text[brace_start: brace_end + 1])
        if result:
            logger.warning("[DEBUG] Recovered bare JSON object from model output")
            return result

    logger.error(f"[DEBUG] No JSON found in model output (text length={len(text)})")
    if text:
        logger.error(f"[DEBUG] Output head: {text[:300]!r}")
        logger.error(f"[DEBUG] Output tail: {text[-300:]!r}")
    return {}


def _extract_narrative(text: str) -> str:
    idx = text.find("```json")
    return text[:idx].strip() if idx > 0 else text.strip()
