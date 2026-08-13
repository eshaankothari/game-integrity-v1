"""L4b: play-by-play -> game/player context that box scores cannot express.

Built as TWO separate commands on purpose:

    python -m pipeline.load_data.load_pbp fetch     slow, network-bound, run overnight
    python -m pipeline.load_data.load_pbp derive    fast, offline, safe to re-run and refine

The split matters. Fetching 1,230 games is the expensive, irreversible part; parsing
is where the judgement calls live (what counts as garbage time, how an ejection is
worded) and those will want revising. Keeping them apart means you can rewrite the
parsing a dozen times without re-fetching anything.

WHAT PBP GIVES US THAT BOX SCORES CANNOT
----------------------------------------
GARBAGE TIME, as a TIMESTAMP rather than a game-level flag. scoreHome/scoreAway plus
clock/period give the margin at every moment, so we can find the instant a game
stopped being competitive and split every stat around it. This is the difference
between dropping blowout games (loses information) and stripping garbage-time
possessions from the features (keeps the game, removes the confound).

EJECTIONS, as a fact. Klay Thompson's 1.7-minute game on 2023-11-14 was an ejection.
Nothing in the box score says so, so every ranking we have puts it near the top. With
PBP it becomes `ejected = TRUE` and leaves the population with a stated reason.

STINT STRUCTURE. Substitution events give every on/off transition, so "left early and
never came back while the game was still live" -- the signature of a mid-game injury --
becomes distinguishable from a normal rotation or a planned rest.

SHOT PROFILE. shotDistance and shotResult separate "took his usual shots and missed"
from "took nothing but contested 28-footers", which is invisible in a box score and is
among the harder things to fake convincingly.

    nohup python3 load_pbp.py fetch > pbp.log 2>&1 &
    tail -f pbp.log
    python3 load_pbp.py derive
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from nba_api.stats.endpoints import playbyplayv3

from pipeline.core import cache
from pipeline.core import db

TIMEOUT = 30
RETRIES = 3
WORKERS = 1                  # override with --workers N; 1 is deliberate
BREAKER_LIMIT = 15           # consecutive failures before abandoning the run
PROGRESS_EVERY = 25          # print a heartbeat this often, for tailing a log

_consecutive_failures = 0
_tripped = False
_done = 0

# --- garbage-time rule -----------------------------------------------------
# A game is "decided" once the margin is large enough with little enough time left.
# Once it fires it STAYS fired: garbage time does not un-happen.
#
# These numbers are a convention, not a discovery. 20 points inside the last 6
# minutes is the common heuristic. They are constants so the rule is visible and
# tunable rather than buried, and `derive` can be re-run for free after changing them.
GT_MARGIN = 20
GT_SECONDS_LEFT = 360        # 6 minutes
GT_MIN_PERIOD = 4            # only from the fourth quarter onward

DDL = """
CREATE TABLE IF NOT EXISTS game_pbp_context (
    game_id             TEXT PRIMARY KEY REFERENCES games (game_id),
    n_events            INTEGER,
    n_periods           INTEGER,
    -- seconds elapsed from tip-off at which the game became decided; NULL = never
    garbage_start_sec   INTEGER,
    competitive_sec     INTEGER,          -- seconds of genuinely contested basketball
    final_margin        INTEGER,
    max_margin          INTEGER,          -- biggest lead either side held
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS player_game_pbp (
    game_id        TEXT    NOT NULL REFERENCES games (game_id),
    player_id      INTEGER NOT NULL REFERENCES players (player_id),

    -- the fact that resolves the Klay Thompson problem
    ejected        BOOLEAN NOT NULL DEFAULT false,
    ejection_sec   INTEGER,

    -- stint structure: how his night was shaped, not just how long it was
    n_stints       INTEGER,
    first_in_sec   INTEGER,
    last_out_sec   INTEGER,
    returned_after_last_exit BOOLEAN,

    -- production split around the garbage-time boundary
    points_competitive INTEGER,
    points_garbage     INTEGER,
    fga_competitive    INTEGER,
    fga_garbage        INTEGER,

    -- shot profile: took his usual shots, or something else?
    n_shots        INTEGER,
    avg_shot_dist  NUMERIC(6,2),
    max_shot_dist  NUMERIC(6,2),

    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS player_game_pbp_ejected_idx
    ON player_game_pbp (game_id) WHERE ejected;
"""


# ===========================================================================
# FETCH
# ===========================================================================

def _call(game_id):
    """One play-by-play call. -> (DataFrame, meta)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=TIMEOUT)
            return pbp.play_by_play.get_data_frame(), {"http_status": 200,
                                                       "credits_delta": 0}
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            last = e
            if attempt == RETRIES:
                break
            time.sleep(2 * attempt)
    raise last


def fetch_one(game_id, conn=None, allow_fetch=False):
    """Cached play-by-play for one game, or None."""
    global _consecutive_failures, _tripped, _done
    key = cache.nba_key("pbp", game_id)
    if not allow_fetch or _tripped:
        return cache.read_cached("nba", key, fmt="csv")
    try:
        df, src = cache.get("nba", key, lambda: _call(game_id),
                            api="nba", endpoint="playbyplayv3", conn=conn,
                            params={"game_id": game_id}, fmt="csv")
        _consecutive_failures = 0
        _done += 1
        if _done % PROGRESS_EVERY == 0:
            print(f"   ... {_done} fetched (latest {game_id}, {src})", flush=True)
        return df
    except Exception as e:
        _consecutive_failures += 1
        print(f"   !! {game_id}: {type(e).__name__} "
              f"({_consecutive_failures} in a row)", flush=True)
        if _consecutive_failures >= BREAKER_LIMIT and not _tripped:
            _tripped = True
            print(f"\n!! CIRCUIT BREAKER: {BREAKER_LIMIT} consecutive failures. "
                  "stats.nba.com has stopped answering.", flush=True)
            print("   Abandoning the rest. Everything fetched is cached; "
                  "re-run to resume from the gap.", flush=True)
        return None


def cmd_fetch(workers=WORKERS, limit=None):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT game_id FROM games ORDER BY game_id")
            gids = [r[0] for r in cur.fetchall()]
    if limit:
        gids = gids[:limit]

    todo = [g for g in gids if cache.status("nba", cache.nba_key("pbp", g)) is None]
    print(f"games            : {len(gids):,}")
    print(f"already cached   : {len(gids) - len(todo):,}")
    print(f"to fetch         : {len(todo):,}")
    print(f"workers          : {workers}   (1 is deliberate; concurrency throttles)")
    if not todo:
        print("\nnothing to fetch. Run:  python -m pipeline.load_data.load_pbp derive")
        return
    print(f"est. wall clock  : ~{len(todo) * 2.0 / max(workers,1) / 60:.0f} min "
          f"if unthrottled\n", flush=True)

    t0 = time.time()
    with db.connect() as conn:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda g: fetch_one(g, conn, allow_fetch=True), todo))
    still = sum(1 for g in gids if cache.status("nba", cache.nba_key("pbp", g)) is None)
    print(f"\nfetched in {(time.time()-t0)/60:.1f} min")
    print(f"cached now : {len(gids) - still:,} / {len(gids):,}")
    if still:
        print(f"still missing {still:,} -- re-run to resume")
    print(cache.summary())
    print("\nNext:  python -m pipeline.load_data.load_pbp derive")


# ===========================================================================
# DERIVE  (offline, re-runnable, this is where the judgement lives)
# ===========================================================================

_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def clock_to_sec_left(clock):
    """'PT11M47.00S' -> 707.0 seconds remaining in the period.

    PBP v3 uses ISO-8601 durations. Anything unparseable returns None rather than
    guessing, so a format change shows up as missing data instead of wrong data.
    """
    if not isinstance(clock, str):
        return None
    m = _CLOCK.match(clock)
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def elapsed_sec(period, sec_left):
    """Seconds since tip-off. Periods 1-4 are 12 minutes, overtimes are 5."""
    if period is None or sec_left is None:
        return None
    p = int(period)
    before = (p - 1) * 720 if p <= 4 else 4 * 720 + (p - 5) * 300
    length = 720 if p <= 4 else 300
    return int(before + (length - sec_left))


def find_garbage_start(df):
    """First moment the game was decided, in seconds from tip. None if never.

    Deliberately monotone: the first qualifying event wins, and the state never
    reverts. A brief 20-point lead in the first quarter does NOT count -- the rule
    requires the fourth period and little time left.
    """
    for _, r in df.iterrows():
        p = r.get("period")
        if p is None or pd.isna(p) or int(p) < GT_MIN_PERIOD:
            continue
        sl = clock_to_sec_left(r.get("clock"))
        if sl is None or sl > GT_SECONDS_LEFT:
            continue
        try:
            margin = abs(int(r.get("scoreHome")) - int(r.get("scoreAway")))
        except (TypeError, ValueError):
            continue
        if margin >= GT_MARGIN:
            return elapsed_sec(p, sl)
    return None


def _is_ejection(row):
    """Ejections are worded inconsistently, so check every text field we have."""
    blob = " ".join(str(row.get(c) or "") for c in
                    ("actionType", "subType", "description")).lower()
    return "eject" in blob


def _is_sub(row):
    blob = f"{row.get('actionType') or ''} {row.get('subType') or ''}".lower()
    return "substitution" in blob or blob.strip() == "sub"


def derive_game(game_id, df):
    """One game's events -> (game_context_row, [player_rows]).

    Returns whatever it can. Missing pieces stay NULL rather than being guessed at.
    """
    df = df.copy()
    df["sec_left"] = df["clock"].map(clock_to_sec_left)
    df["elapsed"] = [elapsed_sec(p, s) for p, s in zip(df.get("period"), df["sec_left"])]

    gt = find_garbage_start(df)
    periods = int(pd.to_numeric(df["period"], errors="coerce").max() or 0)
    total_sec = 4 * 720 + max(0, periods - 4) * 300
    home = pd.to_numeric(df["scoreHome"], errors="coerce")
    away = pd.to_numeric(df["scoreAway"], errors="coerce")

    gctx = {"game_id": game_id,
            "n_events": len(df),
            "n_periods": periods or None,
            "garbage_start_sec": gt,
            "competitive_sec": gt if gt is not None else total_sec,
            "final_margin": (int(abs(home.iloc[-1] - away.iloc[-1]))
                             if home.notna().any() and away.notna().any() else None),
            "max_margin": (int((home - away).abs().max())
                           if home.notna().any() else None)}

    boundary = gt if gt is not None else 10 ** 9
    rows = {}
    for _, r in df.iterrows():
        pid = r.get("personId")
        if pid is None or pd.isna(pid) or int(pid) == 0:
            continue                       # team-level or administrative event
        pid = int(pid)
        row = rows.setdefault(pid, {
            "game_id": game_id, "player_id": pid, "ejected": False,
            "ejection_sec": None, "n_stints": 0, "first_in_sec": None,
            "last_out_sec": None, "returned_after_last_exit": None,
            "points_competitive": 0, "points_garbage": 0,
            "fga_competitive": 0, "fga_garbage": 0,
            "n_shots": 0, "avg_shot_dist": None, "max_shot_dist": None,
            "_dists": []})

        el = r.get("elapsed")
        if _is_ejection(r):
            row["ejected"] = True
            row["ejection_sec"] = el

        # shots, split around the garbage-time boundary
        if str(r.get("isFieldGoal") or "") in ("1", "1.0", "True", "true"):
            side = "competitive" if (el is not None and el < boundary) else "garbage"
            row[f"fga_{side}"] += 1
            row["n_shots"] += 1
            d = pd.to_numeric(r.get("shotDistance"), errors="coerce")
            if pd.notna(d):
                row["_dists"].append(float(d))
            if str(r.get("shotResult") or "").lower() == "made":
                pts = 3 if (pd.notna(d) and d >= 22) else 2
                row[f"points_{side}"] += pts

        if _is_sub(r) and el is not None:
            row["n_stints"] += 1
            if row["first_in_sec"] is None:
                row["first_in_sec"] = el
            row["last_out_sec"] = el

    out = []
    for row in rows.values():
        d = row.pop("_dists")
        if d:
            row["avg_shot_dist"] = round(sum(d) / len(d), 2)
            row["max_shot_dist"] = round(max(d), 2)
        out.append(row)
    return gctx, out


def cmd_derive(dry=True):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute("SELECT game_id FROM games ORDER BY game_id")
            gids = [r[0] for r in cur.fetchall()]
            known = db.existing_ids(conn, "players", "player_id")

        gctx_rows, prows, seen_actions, parsed = [], [], {}, 0
        for gid in gids:
            df = cache.read_cached("nba", cache.nba_key("pbp", gid), fmt="csv")
            if df is None or df.empty:
                continue
            parsed += 1
            for a in df.get("actionType", pd.Series(dtype=str)).dropna().unique():
                seen_actions[a] = seen_actions.get(a, 0) + 1
            g, ps = derive_game(gid, df)
            gctx_rows.append(g)
            # Foreign key: a personId absent from `players` would fail the insert.
            prows.extend(p for p in ps if p["player_id"] in known)

        print(f"cached play-by-play games parsed : {parsed:,} of {len(gids):,}")
        if not parsed:
            sys.exit("!! nothing cached. Run:  python -m pipeline.load_data.load_pbp fetch")
        gt = sum(1 for g in gctx_rows if g["garbage_start_sec"] is not None)
        ej = sum(1 for p in prows if p["ejected"])
        print(f"  games reaching garbage time    : {gt:,} "
              f"({100*gt/max(parsed,1):.0f}%)")
        print(f"  player-games with an ejection  : {ej:,}")
        print(f"  player rows                    : {len(prows):,}")
        print(f"\n  distinct actionType values seen: {len(seen_actions)}")
        for a, n in sorted(seen_actions.items(), key=lambda x: -x[1])[:12]:
            print(f"     {a:24} {n:,}")

        if dry:
            db.dry_notice()
            return

        n1 = db.upsert(conn, "game_pbp_context", gctx_rows, conflict=["game_id"])
        n2 = db.upsert(conn, "player_game_pbp", prows,
                       conflict=["game_id", "player_id"])
        print(f"\nupserted {n1:,} -> game_pbp_context")
        print(f"upserted {n2:,} -> player_game_pbp")


if __name__ == "__main__":
    args = sys.argv[1:]
    w = next((int(args[i + 1]) for i, a in enumerate(args)
              if a == "--workers" and i + 1 < len(args)), WORKERS)
    lim = next((int(args[i + 1]) for i, a in enumerate(args)
                if a == "--limit" and i + 1 < len(args)), None)
    if "fetch" in args:
        cmd_fetch(workers=w, limit=lim)
    elif "derive" in args:
        cmd_derive(dry="run" not in args)
    else:
        print(__doc__.strip().split("\n\n")[-1])
