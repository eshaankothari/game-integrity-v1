"""L4: box scores for every player-game -> `player_games`.

SUMMARY: Pulls from four box-score endpoints (per game which is pulled from GAMES db)
from the NBA API. It runs using a ThreadPoolExecutor for efficiency and merges all
four endpoint pulls into one row per player per game into the PLAYER_GAMES db.

Four endpoints per game, all keyed by `personId`, so unlike L3 there is NO name
matching here -- the join is an integer.

    box     BoxScoreTraditionalV3   minutes, points, fga, rebounds, assists, +/-
    adv     BoxScoreAdvancedV3      usage_pct, turnover_ratio, ratings, pace
    hustle  BoxScoreHustleV2        contested shots, deflections, loose balls
    track   BoxScorePlayerTrackV3   distance, touches, speed, passes

FREE, BUT SLOW. nba_api has no built-in rate limiting, so wall clock is purely
`calls x latency / workers`. The 4,920 calls are INDEPENDENT, so they are flattened
into one queue rather than nested per game the way v0 did it (6 games in parallel,
each doing its 4 endpoints in series). A flat queue means a slow endpoint never
blocks the other three of its own game.

EVERY PLAYER IS INGESTED, not just the ~431 who had props. A box-score call returns
the whole roster whether you want one player or fifteen, so league-wide coverage is
free -- and L5's baseline REQUIRES it. Standardising "relative to the league"
against only propped players would compute the mean over a systematically
higher-usage population and bias every z-score.

A FAILED ENDPOINT DOES NOT ABORT THE RUN. With ~4,100 calls some will stall. The
row is written with that source's has_* flag false and the run continues; a later
pass fills the gap (see `player_games_incomplete_idx`). Those flags are what
separate "the endpoint failed" from "this stat is genuinely absent" -- without them
a timeout is indistinguishable from a real zero and quietly biases L5.

DNPs are kept. `minutes IS NULL` with has_traditional TRUE means the player was on
the roster and did not appear. L5 must exclude them from the baseline, or a pile of
zeros drags every league mean down.

    python -m pipeline.load_data.load_boxscores                          # DRY: what it would fetch
    python -m pipeline.load_data.load_boxscores cached                   # build from cache only, 0 calls
    python -m pipeline.load_data.load_boxscores run                      # fetch + write
    python -m pipeline.load_data.load_boxscores run --date 2023-11-08    # one game day
    python -m pipeline.load_data.load_boxscores run --workers 1          # tune concurrency
    python -m pipeline.load_data.load_boxscores run force                # skip the preflight probe
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from nba_api.stats.endpoints import (boxscoreadvancedv3, boxscorehustlev2,
                                     boxscoreplayertrackv3, boxscoretraditionalv3)

from pipeline.core import cache
from pipeline.core import db

TIMEOUT = 20
RETRIES = 3
WORKERS = 3              # override with --workers N

# CIRCUIT BREAKER. stats.nba.com throttles in bursts: it serves a run of requests,
# then blackholes the IP for a while. The pre-flight probe only checks ONCE, at the
# start, so a mid-run block goes undetected and every remaining task independently
# rediscovers it -- 66s of retry ladder each, producing nothing. Observed: 33 files
# in 69s, then a block that would have burned ~6 hours of timeouts.
#
# So: after this many CONSECUTIVE failures, stop the whole run. Cached responses
# persist, so aborting costs nothing and re-running resumes from the gap.
BREAKER_LIMIT = 12
_consecutive_failures = 0
_tripped = False

# kind -> (endpoint class, {source column: our column})
KINDS = {
    "box": (boxscoretraditionalv3.BoxScoreTraditionalV3, {
        "points": "points", "fieldGoalsAttempted": "fga", "fieldGoalsMade": "fgm",
        "threePointersAttempted": "fg3a", "threePointersMade": "fg3m",
        "freeThrowsAttempted": "fta", "freeThrowsMade": "ftm",
        "reboundsTotal": "rebounds", "reboundsOffensive": "rebounds_off",
        "assists": "assists", "steals": "steals", "blocks": "blocks",
        "turnovers": "turnovers", "foulsPersonal": "fouls",
        "plusMinusPoints": "plus_minus"}),
    "adv": (boxscoreadvancedv3.BoxScoreAdvancedV3, {
        "usagePercentage": "usage_pct", "turnoverRatio": "turnover_ratio",
        "trueShootingPercentage": "true_shooting_pct",
        "effectiveFieldGoalPercentage": "efg_pct",
        "assistPercentage": "assist_pct", "reboundPercentage": "rebound_pct",
        "offensiveRating": "offensive_rating", "defensiveRating": "defensive_rating",
        "netRating": "net_rating", "pace": "pace"}),
    "hustle": (boxscorehustlev2.BoxScoreHustleV2, {
        "contestedShots": "contested_shots", "deflections": "deflections",
        "looseBallsRecoveredTotal": "loose_balls", "boxOuts": "box_outs",
        "chargesDrawn": "charges_drawn", "screenAssists": "screen_assists"}),
    "track": (boxscoreplayertrackv3.BoxScorePlayerTrackV3, {
        "speed": "speed", "distance": "distance", "touches": "touches",
        "passes": "passes", "contestedFieldGoalsAttempted": "contested_fga",
        "uncontestedFieldGoalsAttempted": "uncontested_fga"}),
}
FLAG = {"box": "has_traditional", "adv": "has_advanced",
        "hustle": "has_hustle", "track": "has_track"}

# Rate stats: every one of these is a per-possession or per-48-minute quantity, so
# each divides by playing time somewhere.
#
# WHY THEY NEED NULLING. A player who dressed and logged '0:00' is NOT a DNP -- the
# minutes field is populated, just zero -- so he reaches the merge with a real row and
# the API hands back the degenerate limit of the division rather than a blank. Bol Bol
# in 0022300162 came through with pace = 28,800, which is 48 x 60 x 10 and not a
# measurement of anything. That single row overflowed NUMERIC(7,3) and aborted the
# write of all 32,385.
#
# Widening the column would have stored the garbage instead of rejecting it. These are
# undefined at zero minutes, so they are recorded as undefined. Counting stats
# (points, rebounds, distance) are left alone: zero really is zero for those.
RATE_STATS = ["usage_pct", "turnover_ratio", "true_shooting_pct", "efg_pct",
              "assist_pct", "rebound_pct", "offensive_rating", "defensive_rating",
              "net_rating", "pace", "speed"]

# Player-tracking fields, and the outage they need protecting from.
#
# The optical tracking system goes down. On 2024-03-09 it failed for SEVEN games and
# the endpoint returned 0.0 for every player rather than an error or a blank -- 89
# players logged 20+ minutes with distance 0.00, against a league average of 2.38
# miles. Six more games across five other dates did the same.
#
# A ZERO FROM AN OUTAGE IS INDISTINGUISHABLE FROM A REAL ZERO at the row level, and
# downstream it is worse than useless: Kevin Durant's 45-point night on 2024-03-09
# came out at distance_z -6.77, so Isolation Forest ranked one of the best games of
# his season as its third most anomalous.
#
# Detected PER FIELD and PER GAME: if every player who appeared shows exactly 0 for a
# field, the field did not record. One player can genuinely have 0 touches; twenty
# cannot. Done per field rather than per game because outages are sometimes partial --
# 0022300489 lost distance for all 24 players but kept touches for 14 of them.
TRACK_FIELDS = ["distance", "touches", "speed", "passes",
                "contested_fga", "uncontested_fga"]
MIN_FOR_OUTAGE = 5          # too few players to tell an outage from a quiet night

# PARTIAL outages, which the all-zero rule cannot see. In 0022300650 the system came
# up late and recorded roughly a tenth of the game: LeBron played 47.7 minutes and was
# credited with 0.34 miles and 10 touches, against the ~2.8 miles and ~70 touches that
# workload actually involves. Nothing is zero, so every value looks legitimate.
#
# Distance per minute is the giveaway because it is near-constant across the league --
# basketball players run at roughly the same rate whoever they are. League median is
# 0.0761 mi/min. The three bad games sit at 0.0075, 0.0189 and 0.0396; the next lowest
# game is 0.0555 and the bulk begins at 0.0687, so a cut at 0.045 separates them with
# a 40 percent gap on either side rather than slicing into a continuum.
MIN_DIST_PER_MIN = 0.045
MIN_MIN_FOR_RATE = 10       # short stints have unstable rates, so exclude them


def _call(kind, game_id):
    """One box-score call. -> (DataFrame, meta). Retries only on transient stalls."""
    endpoint = KINDS[kind][0]
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            b = endpoint(game_id=game_id, timeout=TIMEOUT)
            return b.player_stats.get_data_frame(), {"http_status": 200, "credits_delta": 0}
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last = e
            if attempt == RETRIES:
                break
            time.sleep(2 * attempt)
    raise last


def fetch_one(kind, game_id, conn=None, allow_fetch=False):
    """Cached box score -> DataFrame, or None if absent / failed / breaker tripped."""
    global _consecutive_failures, _tripped
    key = cache.nba_key(kind, game_id)
    if not allow_fetch:
        return cache.read_cached("nba", key, fmt="csv")

    # Once tripped, remaining tasks return immediately instead of each spending
    # ~66s in the retry ladder to rediscover the same block.
    if _tripped:
        cached = cache.read_cached("nba", key, fmt="csv")
        return cached

    try:
        df, _ = cache.get("nba", key, lambda: _call(kind, game_id),
                          api="nba", endpoint=f"boxscore_{kind}", conn=conn,
                          params={"game_id": game_id, "kind": kind}, fmt="csv")
        _consecutive_failures = 0             # any success resets the run
        return df
    except Exception as e:                    # one bad call must not kill the pass
        _consecutive_failures += 1
        print(f"   !! {kind} {game_id}: {type(e).__name__} "
              f"({_consecutive_failures} in a row)")
        if _consecutive_failures >= BREAKER_LIMIT and not _tripped:
            _tripped = True
            print(f"\n!! CIRCUIT BREAKER: {BREAKER_LIMIT} consecutive failures -- "
                  "stats.nba.com has stopped responding.")
            print("   Abandoning the remaining fetches. Everything already fetched is")
            print("   cached; re-run later and it resumes from the gap.")
        return None


PREFLIGHT_TRIES = 3


def preflight():
    """Probe before committing to thousands of calls. Any single success passes.

    Deliberately several attempts, not one. stats.nba.com throttles intermittently
    rather than blocking outright -- most calls time out while some get through
    (76 files landed in one session while single probes were failing). A one-shot
    probe is therefore a single sample of a very noisy signal, and treating its
    failure as fatal refuses runs that would have made real progress.

    So: succeed on ANY attempt, and if all fail, say so without pretending it is
    certain. The circuit breaker is what handles a genuine block once running.
    """
    for i in range(1, PREFLIGHT_TRIES + 1):
        try:
            boxscoretraditionalv3.BoxScoreTraditionalV3(game_id="0022300001", timeout=10)
            if i > 1:
                print(f"   preflight ok on attempt {i}/{PREFLIGHT_TRIES}")
            return True
        except Exception as e:
            print(f"   preflight attempt {i}/{PREFLIGHT_TRIES}: {type(e).__name__}")
    print("!! stats.nba.com did not answer any preflight probe.")
    print("   It throttles intermittently, so this is not proof the run would fail --")
    print("   pass 'force' to try anyway; the circuit breaker will stop it if it is")
    print("   genuinely blocked. Cached games build with: load_boxscores.py cached")
    return False


def _minutes(v):
    """'29:24' -> 29.4 decimal minutes. NaN/blank -> None, meaning DID NOT PLAY."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan"):
        return None
    s = str(v)
    if ":" in s:
        mm, ss = s.split(":")[:2]
        try:
            return round(int(mm) + int(ss) / 60, 3)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _num(v):
    return None if v is None or pd.isna(v) else v


def _text(v):
    """Trimmed string, or None for blank/NaN -- so 'played' is NULL, not ''."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def build_game(game_id, frames):
    """The four frames for one game -> player_games rows, keyed by personId."""
    box = frames.get("box")
    if box is None or box.empty:
        return [], set()                       # no roster -> nothing to anchor on

    # Every row must carry EVERY column, even when a supplementary endpoint has no
    # entry for that player. db.upsert derives its column list from rows[0], so a
    # ragged row set would silently take the first row's shape -- and the hustle
    # frame in particular only lists players who actually appeared (19 of 29 in a
    # typical game), so raggedness is the norm here, not an edge case.
    ALL_COLS = [dst for _, m in KINDS.values() for dst in m.values()]

    rows, seen = {}, set()
    for _, r in box.iterrows():
        pid = int(r["personId"])
        seen.add(pid)
        row = {"game_id": game_id, "player_id": pid,
               "team_id": int(r["teamId"]) if not pd.isna(r["teamId"]) else None,
               # `position` is populated for starters only -- 10 of ~27 rows a game
               # -- so it is a did-they-start flag, not a player attribute. Real
               # position comes from load_rosters.py.
               "started": bool(str(r.get("position") or "").strip()),
               "minutes": _minutes(r.get("minutes")),
               "dnp_reason": _text(r.get("comment"))}
        row.update({c: None for c in ALL_COLS})
        for src, dst in KINDS["box"][1].items():
            row[dst] = _num(r.get(src))
        for f in FLAG.values():
            row[f] = False
        row["has_traditional"] = True
        rows[pid] = row

    for kind in ("adv", "hustle", "track"):
        df = frames.get(kind)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            pid = int(r["personId"])
            if pid not in rows:                # in a supplement but not the box score
                continue
            for src, dst in KINDS[kind][1].items():
                rows[pid][dst] = _num(r.get(src))
            rows[pid][FLAG[kind]] = True

    # A rate stat is undefined without playing time. Applied after the merge so it
    # catches whichever endpoint supplied the value.
    for row in rows.values():
        if not row["minutes"]:                 # None (DNP) or 0.0 (dressed, never in)
            for c in RATE_STATS:
                row[c] = None

    # Tracking outage: a field that is exactly 0 for EVERY player who appeared did not
    # record. Null it for the game rather than let a fabricated zero reach the z-scores.
    played = [r for r in rows.values() if r["minutes"]]
    if len(played) >= MIN_FOR_OUTAGE:
        for c in TRACK_FIELDS:
            vals = [r[c] for r in played]
            if all(v is not None and float(v) == 0.0 for v in vals):
                for r in rows.values():
                    r[c] = None

        # Partial outage: the values are non-zero but the whole game ran too slowly to
        # be physical. One field is enough to condemn the rest -- the tracking either
        # recorded the game or it did not.
        rates = sorted(float(r["distance"]) / float(r["minutes"])
                       for r in played
                       if r["minutes"] >= MIN_MIN_FOR_RATE and r["distance"] is not None)
        if rates and rates[len(rates) // 2] < MIN_DIST_PER_MIN:
            for r in rows.values():
                for c in TRACK_FIELDS:
                    r[c] = None

    return list(rows.values()), seen


def new_players(conn, seen, frames_by_game):
    """personIds absent from `players` -> rows to insert, so the FK never fails.

    nba_api's static player table is a snapshot frozen in the installed package, so
    a player it predates would otherwise break the foreign key. Names come from the
    box score itself, which is authoritative anyway.
    """
    known = db.existing_ids(conn, "players", "player_id")
    missing = seen - known
    if not missing:
        return []
    from load_players import ascii_name
    out, done = [], set()
    for frames in frames_by_game.values():
        box = frames.get("box")
        if box is None:
            continue
        for _, r in box.iterrows():
            pid = int(r["personId"])
            if pid in missing and pid not in done:
                full = f"{r.get('firstName') or ''} {r.get('familyName') or ''}".strip()
                out.append({"player_id": pid, "full_name": full,
                            "canonical_ascii": ascii_name(full)})
                done.add(pid)
    return out


def main(mode="dry", date=None, game=None, workers=WORKERS, force=False):
    allow_fetch = (mode == "run")
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT game_id FROM games
                WHERE (%s::date IS NULL OR game_date = %s::date)
                  AND (%s::text IS NULL OR game_id   = %s::text)
                ORDER BY game_id""", (date, date, game, game))
            gids = [r[0] for r in cur.fetchall()]
        if not gids:
            sys.exit(f"!! no games match {db.scope_label(date, game)}")

        tasks = [(k, g) for g in gids for k in KINDS]
        cached_n = sum(1 for k, g in tasks if cache.status("nba", cache.nba_key(k, g)))
        missing_n = len(tasks) - cached_n

        print(f"scope                 : {db.scope_label(date, game)}")
        print(f"games                 : {len(gids)}")
        print(f"box-score calls needed: {len(tasks)}  ({len(KINDS)} endpoints x {len(gids)} games)")
        print(f"  already cached      : {cached_n}  <- free")
        print(f"  to fetch            : {missing_n}   (nba_api is FREE -- costs time, not credits)")
        if missing_n:
            print(f"  est. wall clock     : ~{missing_n * 1.5 / max(workers, 1) / 60:.0f} min "
                  f"at {workers} workers")

        if mode == "dry":
            db.dry_notice()
            return
        # `force` skips the probe entirely: on an intermittently throttled host the
        # probe can fail while the run would still make progress, and the circuit
        # breaker is the real safety net once fetching starts.
        if allow_fetch and missing_n and not force and not preflight():
            sys.exit(1)

        # One flat queue over all (kind, game) pairs -- see module docstring.
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(
                lambda t: (t[0], t[1], fetch_one(t[0], t[1], conn, allow_fetch)), tasks))
        elapsed = time.time() - t0

        frames_by_game = {}
        for kind, gid, df in results:
            frames_by_game.setdefault(gid, {})[kind] = df

        rows, seen, no_box = [], set(), 0
        for gid in gids:
            r, s = build_game(gid, frames_by_game.get(gid, {}))
            if not r:
                no_box += 1
            rows.extend(r)
            seen |= s

        print(f"\nfetched in {elapsed:.0f}s")
        if _tripped:
            still = sum(1 for k, g in tasks if not cache.status("nba", cache.nba_key(k, g)))
            print(f"!! run ABORTED by the circuit breaker -- {still:,} calls still missing.")
            print("   Rows below cover only what was cached at the time of the abort.")
        print(f"games with no box score: {no_box}")
        print(f"player_games rows      : {len(rows):,}")
        if rows:
            dnp = sum(1 for r in rows if r["minutes"] is None)
            print(f"  did not play (DNP)   : {dnp:,}")
            for kind, flag in FLAG.items():
                print(f"  {flag:16}: {sum(1 for r in rows if r[flag]):,}")

        added = new_players(conn, seen, frames_by_game)
        if added:
            db.upsert(conn, "players", added, conflict=["player_id"])
            print(f"players added from box scores: {len(added)}")

        n = db.upsert(conn, "player_games", rows, conflict=["game_id", "player_id"])
        print(f"\nupserted {n:,} rows -> player_games ({db.count(conn, 'player_games'):,} total)")
        print(cache.summary())


if __name__ == "__main__":
    args = sys.argv[1:]
    _mode = next((a for a in args if a in ("run", "cached")), "dry")
    _w = next((int(args[i + 1]) for i, a in enumerate(args)
               if a == "--workers" and i + 1 < len(args)), WORKERS)
    _date, _game = db.scope()
    main(mode=_mode, date=_date, game=_game, workers=_w,
         force="force" in args)
