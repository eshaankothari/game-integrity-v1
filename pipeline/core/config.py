"""Foundation config for game-integrity v1.

SUMMARY: This file imports all previous caches (from Eshaan's V0 - disregard), environment variables, and sets up the global
variables.

Standalone project: all logic is COPIED in (no imports from the v0 repo). We only
REUSE the v0 caches on disk, so OddsAPI/nba calls already paid for are never re-billed.
"""
import os
from pathlib import Path

# pipeline/core/config.py -> pipeline/core -> gi -> the repository root. If this file ever moves
# again, this is the ONE line in the backend that has to move with it: cache/, out/
# and game_integrity.duckdb are all resolved from it.
ROOT = Path(__file__).resolve().parents[2]                 # game-integrity-v1/
V0 = Path("/Users/eshaankothari/Desktop/Game Integrity Product v0")   # source of paid caches only

# --- paid caches (reused in place; we read, never rewrite v0) ---
SNAPSHOT_CACHE = V0 / "snapshot_cache"                     # OddsAPI historical responses
NBA_CACHE = V0 / "nba_cache"                               # box / advanced / hustle / track responses
EVENTS_CACHE = V0 / "isolation_test" / "data" / "events_cache"   # historical-events by date

# --- v1's own cache (everything we fetch lands here, so v0 stays read-only) ---
CACHE = ROOT / "cache"
SNAPSHOT_CACHE_V1 = CACHE / "snapshots"                    # new OddsAPI responses
NBA_CACHE_V1 = CACHE / "nba"                               # new nba_api responses
EVENTS_CACHE_V1 = CACHE / "events"                         # historical-events, 1 per date
BBREF_CACHE_V1 = CACHE / "bbref"                           # basketball-reference salary tables
LLM_CACHE_V1 = CACHE / "llm"                               # one file per (model, prompt, game)


def _load_env(path):
    """Copy KEY=value lines from a .env file into the environment (a real export still wins)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


_load_env(ROOT / ".env")                                   # our own .env first
_load_env(V0 / ".env")                                     # fall back to v0's (holds ODDSAPI_KEY)
ODDS_API_KEY = os.environ.get("ODDSAPI_KEY")

# L9 reviewer. Keys absent by default -- the layer reports what it WOULD do and writes
# nothing, so a missing key is never a silent no-op.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Default: Gemini's cheapest tier. The reviewer is structured extraction against a fixed
# taxonomy with every fact handed to it in the packet -- the job small models hold up on.
# Coverage matters more here than per-call quality: reviewing all 4,810 shortlisted games
# beats reviewing the top 200 with a model ten times the price.
#
# Gemini also enforces the output SCHEMA server-side (responseSchema), so a malformed
# JSON answer is impossible rather than something to parse defensively around.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")

# Provider is inferred from the model name, so switching is one env var.
LLM_PROVIDER = os.environ.get(
    "LLM_PROVIDER", "gemini" if LLM_MODEL.startswith("gemini") else "anthropic")

# $ per MILLION tokens (input, output). Used ONLY for the --dry estimate; nothing in the
# pipeline depends on these being right.
#
# THESE GO STALE. gemini-2.5-flash-lite was the default here until the API answered
# "no longer available to new users" -- a model name is not a constant. Run
# `python -m pipeline.llm_review.review --models` for what your key can actually call, and check the
# provider's pricing page before repeating any number below to anyone.
LLM_PRICES = {
    "gemini-2.5-flash-lite":     (0.10, 0.40),
    "gemini-2.0-flash-lite":     (0.075, 0.30),
    "gemini-2.5-flash":          (0.30, 2.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5":           (3.00, 15.00),
    # gemini-3.5-flash / -flash-lite: no entry, because I do not know their published
    # prices. Add them and --dry will estimate; without them it prints the token counts
    # and says so rather than inventing a dollar figure.
}

# --- scope for v1: league-wide, 2023-24 regular season, points props ---
SEASON = "2023-24"
SEASON_TYPE = "regular"                                    # game_id prefix '002'
MARKET = "player_points"
REGIONS = "us"                                             # returns ALL US books for
                                                           # the SAME credit -- so we
                                                           # ingest every book...
BOOK = "fanduel"                                           # ...and prefer this one at
                                                           # QUERY time, with the rest
                                                           # as free fallback coverage

# --- database ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///game_integrity_v1")

# THREE READ BACKENDS, POSTGRES BY DEFAULT. The WRITE path is always Postgres.
#
#   postgres   the source of truth -- every loader writes here, all 21 tables
#   duckdb     a read-only export of the 17 scored tables, one file, no server
#   csv        the same 17 tables as plain text files, no database at all
#
# The last two are EXPORTS, produced by pipeline/tools/to_duckdb.py and to_csv.py.
# They exist so the dashboard can be handed to someone who has neither a Postgres
# server nor a reason to install one. Nothing in the pipeline reads them.
#
# GI_DB picks the backend by shape, so there is no second switch to forget:
#
#   unset                        -> postgres
#   GI_DB=game_integrity.duckdb  -> duckdb   (a .duckdb file)
#   GI_DB=data                   -> csv      (a directory of .csv)
GI_DB = os.environ.get("GI_DB")


def _backend(gi_db):
    if not gi_db:
        return "postgres"
    if gi_db.endswith(".duckdb"):
        return "duckdb"
    p = Path(gi_db)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir() or gi_db.endswith("/"):
        return "csv"
    return "postgres"


BACKEND = _backend(GI_DB)

# Where the CSV export lives, as an absolute path. None unless BACKEND == 'csv'.
CSV_DIR = (Path(GI_DB) if GI_DB and Path(GI_DB).is_absolute() else ROOT / (GI_DB or "")) \
          if BACKEND == "csv" else None