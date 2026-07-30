"""Foundation config for game-integrity v1.

SUMMARY: This file imports all previous caches, environment variables, and sets up the global
variables.

Standalone project: all logic is COPIED in (no imports from the v0 repo). We only
REUSE the v0 caches on disk, so OddsAPI/nba calls already paid for are never re-billed.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent                     # game-integrity-v1/
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