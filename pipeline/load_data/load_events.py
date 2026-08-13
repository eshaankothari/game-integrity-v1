"""L2: map every game to its OddsAPI event_id -> `odds_events`, and backfill tipoff.

SUMMARY: Uses the tipoff time from GAMES db to get the event id and commence 
time from OddsAPI - costs 1 token and loaded in ODDS_EVENTS db.
Live tracker will pass in all games from one day.

Cheapest-source-first, in three passes:

  1. SEED from v0's bulk event files -- 1,235 distinct events already on disk from
     the season dumps and the 69 per-date cache files. Costs 0 credits and covers
     1,209 of 1,230 games.
  2. ENDPOINT for whatever is left, batched ONE CALL PER DATE (each response holds
     every game that day). 21 games remain, spread over 5 dates -> 5 credits.
  3. BACKFILL games.tipoff_utc from the matched commence_time.

commence_time is the payoff here. Every open/close probe offset in L3 is measured
from it, and v0's cached snapshot filenames were built from it -- so a wrong tip
time does not just skew a feature, it misses the cache and re-bills all of L3.

Matching is EXACT on both axes -- team ids, and the event's US/Eastern date. v0
matched loosely (nickname substrings, plus a +/-1 day window for UTC spillover) and
that silently mis-mapped 6 games league-wide. See `match()` for what went wrong.

    python -m pipeline.load_data.load_events            # DRY: exact credit count, writes nothing
    python -m pipeline.load_data.load_events run        # fetch missing dates + write
"""
import json
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from nba_api.stats.static import teams as static_teams

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db

EVENTS_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events"
ET = ZoneInfo("America/New_York")               # NBA's game_date is the US/Eastern date

# Exact full-name -> team_id. All 30 OddsAPI team names match NBA's full_name
# byte-for-byte, so matching is an ID comparison, never a string search.
NAME2ID = {t["full_name"]: t["id"] for t in static_teams.get_teams()}
ABBR2ID = {t["abbreviation"]: t["id"] for t in static_teams.get_teams()}

# v0 bulk event dumps: local files, not API responses -- reading them is free and
# needs no cache layer. Only the per-date ENDPOINT calls go through cache.get().
SEEDS = [
    (config.V0 / "OddsAPI" / "Key Figures" / "events_2023-2024_full_season.json", "v0_season_file"),
    (config.V0 / "OddsAPI" / "event_ids" / "events_2023-2024_season.json", "v0_season_file"),
]


def _events_of(payload):
    """Unwrap either a bare list or the {'data': [...]} envelope."""
    if isinstance(payload, dict):
        return payload.get("data") or []
    return payload or []


def seed_pool():
    """Every event v0 already has on disk -> {event_id: (event, source)}. 0 credits."""
    pool = {}
    for path, src in SEEDS:
        if path.exists():
            for e in _events_of(json.loads(path.read_text())):
                pool.setdefault(e["id"], (e, src))
    if config.EVENTS_CACHE.exists():
        for path in sorted(config.EVENTS_CACHE.glob("events_*.json")):
            for e in _events_of(json.loads(path.read_text())):
                pool.setdefault(e["id"], (e, "v0_events_cache"))
    return pool


def et_date(commence_iso):
    """UTC commence_time -> US/Eastern calendar date, which IS NBA's game_date.

    A 7:10pm PT tip on Dec 28 is 2023-12-29T03:10:00Z -- next day in UTC, but still
    Dec 28 in Eastern, and Dec 28 is what NBA calls it. Converting to ET makes the
    date an exact key and removes the need for a +/-1 day window entirely.
    """
    return datetime.fromisoformat(commence_iso.replace("Z", "+00:00")).astimezone(ET).date()


def by_date(pool):
    """Index events by ET commence date, so matching is not O(games x events)."""
    idx = {}
    for eid, (e, src) in pool.items():
        idx.setdefault(et_date(e["commence_time"]), []).append((eid, e, src))
    return idx


def team_ids(matchup):
    """'TOR @ HOU' -> (away_id, home_id) via exact abbreviation lookup."""
    a, h = [s.strip() for s in matchup.replace(" vs. ", " @ ").split(" @ ")]
    return ABBR2ID.get(a), ABBR2ID.get(h)


def match(idx, game_date, matchup):
    """Find the event for this game -> (event_id, event, source) or (None, None, None).

    Exact on both axes:
      DATE  -- event's ET commence date == game_date (no fuzzy window)
      TEAMS -- (away_id, home_id) equality, not substring search

    Both are deliberate. A substring test matched 'nets' inside 'hornets', quietly
    mapping Brooklyn games onto Charlotte events; and a +/-1 day window made two
    games between the same teams on consecutive days indistinguishable, so both
    claimed one event and the loser silently lost its odds entirely.
    """
    away_id, home_id = team_ids(matchup)
    if not (away_id and home_id):
        return None, None, None
    for eid, e, src in idx.get(game_date, ()):
        e_home, e_away = NAME2ID.get(e["home_team"]), NAME2ID.get(e["away_team"])
        if (e_away, e_home) == (away_id, home_id):
            return eid, e, src
    return None, None, None


def _call_events(day):
    """One historical-events call covering `day` (a date). -> (payload, meta).

    Params mirror v0's so the response window lines up with what it cached.
    """
    r = requests.get(EVENTS_URL, params={
        "apiKey": config.ODDS_API_KEY, "dateFormat": "iso",
        "commenceTimeFrom": f"{day}T00:00:00Z",
        "commenceTimeTo": f"{day + timedelta(days=2)}T00:00:00Z",
        "date": f"{day}T18:00:00Z"})
    meta = {"http_status": r.status_code,
            "requests_used": r.headers.get("x-requests-used"),
            "requests_remaining": r.headers.get("x-requests-remaining"),
            "credits_delta": None}   # measured from headers, not assumed
    if r.status_code == 404:
        return {"data": []}, meta            # negative-cached like any other response
    r.raise_for_status()
    return r.json(), meta


def resolve(games, idx):
    """Match every game against the pool -> (matched, unmatched)."""
    matched, unmatched = [], []
    for gid, gdate, matchup in games:
        eid, e, src = match(idx, gdate, matchup)
        if eid:
            matched.append((gid, eid, e, src))
        else:
            unmatched.append((gid, gdate, matchup))
    return matched, unmatched


def collisions(matched):
    """event_ids claimed by more than one game -> [(event_id, [game_id, ...])].

    Must always be empty. odds_events.event_id is the primary key, so two games
    claiming one event does not error -- the upsert just overwrites, and the losing
    game ends up with no odds at all. That is silent data loss, so it is checked
    explicitly rather than left to the database to (not) complain about.
    """
    seen = {}
    for gid, eid, _e, _src in matched:
        seen.setdefault(eid, []).append(gid)
    return [(eid, gids) for eid, gids in seen.items() if len(gids) > 1]


def main(dry=True, date=None, game=None):
    with db.connect() as conn:
        with conn.cursor() as cur:
            # NULL scope = no filter, so one query serves both the full season and
            # a single night. %s is passed twice per predicate because psycopg
            # binds positionally, not by name.
            cur.execute("""
                SELECT game_id, game_date, matchup FROM games
                WHERE (%s::date IS NULL OR game_date = %s::date)
                  AND (%s::text IS NULL OR game_id   = %s::text)
                ORDER BY game_id""", (date, date, game, game))
            games = cur.fetchall()
        if not games:
            sys.exit(f"!! no games match {db.scope_label(date, game)}")

        pool = seed_pool()
        idx = by_date(pool)
        matched, unmatched = resolve(games, idx)
        need_dates = sorted({g[1] for g in unmatched})
        uncached = [d for d in need_dates
                    if cache.status("events", cache.events_key(d.isoformat())) is None]

        print(f"scope                    : {db.scope_label(date, game)}")
        print(f"games                    : {len(games)}")
        print(f"v0 event pool (0 credits): {len(pool)} events across {len(idx)} ET dates")
        print(f"matched from v0 files    : {len(matched)}  <- free")
        print(f"still unmatched          : {len(unmatched)} games on {len(need_dates)} dates")
        print(f"  of those dates, cached : {len(need_dates) - len(uncached)}")
        print(f"==> PLANNED billed calls : {len(uncached)}  (1 credit per date)")
        before = cache.report_plan(len(uncached), "historical_events", conn)

        if dry:
            for d in need_dates:
                n = sum(1 for u in unmatched if u[1] == d)
                print(f"     {d}: {n} games")
            dupes = collisions(matched)
            print(f"event_id collisions      : {len(dupes)}  (must be 0)")
            for eid, gids in dupes[:5]:
                print(f"   !! {eid} claimed by {gids}")
            print(cache.summary())
            db.dry_notice()
            return

        if not config.ODDS_API_KEY and uncached:
            sys.exit("!! ODDSAPI_KEY not set -- cannot fetch the missing dates")

        # pass 2: one call per still-missing date, then re-match
        for d in need_dates:
            key = cache.events_key(d.isoformat())
            payload, source = cache.get(
                "events", key, lambda d=d: _call_events(d),
                api="oddsapi", endpoint="historical_events", conn=conn,
                params={"date": d.isoformat()})
            # Label by where the DATA came from, not which cache tier served it this
            # run -- these responses are endpoint pulls whether billed now or cached
            # from an earlier run.
            for e in _events_of(payload):
                pool.setdefault(e["id"], (e, "events_endpoint"))
        idx = by_date(pool)
        matched, unmatched = resolve(games, idx)
        print(f"\nafter endpoint pass: matched {len(matched)}/{len(games)}"
              + (f", {len(unmatched)} STILL unmatched" if unmatched else ""))
        for gid, gdate, matchup in unmatched[:10]:
            print(f"   !! {gid} {gdate} {matchup}")

        dupes = collisions(matched)
        if dupes:
            for eid, gids in dupes[:10]:
                print(f"   !! event {eid} claimed by {len(gids)} games: {gids}")
            sys.exit(f"!! {len(dupes)} event_id collisions -- refusing to write. "
                     "Each would silently drop a game's odds.")

        # write matched events, then backfill tipoff from the authoritative commence_time
        rows = [{"event_id": eid, "game_id": gid, "commence_time": e["commence_time"],
                 "home_team_name": e["home_team"], "away_team_name": e["away_team"],
                 "matched_by": src, "match_note": None}
                for gid, eid, e, src in matched]
        n = db.upsert(conn, "odds_events", rows, conflict=["event_id"],
                      update=["game_id", "commence_time", "home_team_name",
                              "away_team_name", "matched_by"])
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE games g SET tipoff_utc = e.commence_time
                FROM odds_events e
                WHERE e.game_id = g.game_id AND g.tipoff_utc IS DISTINCT FROM e.commence_time""")
            filled = cur.rowcount

        print(f"\nupserted {n} rows -> odds_events ({db.count(conn, 'odds_events')} total)")
        print(f"backfilled tipoff_utc on {filled} games")
        print(cache.summary())
        cache.report_spend(before)


if __name__ == "__main__":
    _date, _game = db.scope()
    main(dry=db.is_dry(), date=_date, game=_game)
