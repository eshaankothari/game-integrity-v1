"""L3b: intra-day line/price CURVES -> `prop_quotes` as snapshot_role='poll'.

SUMMARY: pulls the same historical-odds endpoint as L3, but at a LADDER of offsets
before tip instead of one, so a player-game has a movement series rather than two
endpoints. Written for the case-view graph.

COSMETIC BY CONSTRUCTION. Every row lands with snapshot_role='poll'. standardize.py
selects `FILTER (WHERE snapshot_role = 'close')` and `'open'` by name, so nothing here
can reach the score, the blocks or the funnel. Confirm with `python -m pipeline.load_data.load_line_history
--verify`, which re-reads score_100 and fails loudly if a single row moved.

  - 'poll' is already permitted by prop_quotes_snapshot_role_check -- no migration.
  - The primary key already carries snapshot_requested, so many timestamps per
    (event, player, book) coexist. No migration there either.
  - offset_from_tip_sec is measured from the timestamp the API RETURNED, not the one
    requested: OddsAPI snaps to a ~5-minute grid, so the real window is a few minutes
    off the nominal rung and the graph must plot the real one.

COST IS PER (EVENT, TIMESTAMP), NOT PER PLAYER. One call returns every player and all
~10 US books for that event, so a ladder over N events costs N x rungs x 10 credits
regardless of how many player-games you are charting.

    python -m pipeline.load_data.load_line_history --plan probe            # DRY: exact credit count
    python -m pipeline.load_data.load_line_history --plan probe run        # fetch + write
    python -m pipeline.load_data.load_line_history --plan probe cached     # write only what is on disk, 0 credits
    python -m pipeline.load_data.load_line_history --verify                # scores untouched?
"""
import sys
from datetime import datetime, timedelta, timezone

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db
from pipeline.load_data.load_players import make_resolver
from pipeline.load_data.load_props import FMT, quotes, snapshot

# THE LADDERS, in hours before tip.
#
# `probe` is the starting-line question and nothing else: one rung at T-3h across the
# events whose flagged player had no line at T-12h. It is also the first rung of every
# other ladder, so no credit spent here is wasted whatever is decided next.
#
# `hourly` is for events already priced at T-12h -- 11 new rungs, T-12 and T-0 cached.
# `half`   is for events priced late: 30 minutes apart across the last three hours.
#          15-minute spacing was costed and rejected: lines are quoted in half points,
#          so most adjacent 15-minute pairs are identical and you pay to plot flat
#          segments. Densify a single interesting event afterwards for 60 credits.
PLANS = {
    "probe":  [3.0],
    "probe1": [1.0],
    "hourly": [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    "half":   [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
    # Five-minute ladder across the final hour. T-1h is the probe rung and T-0 is the
    # cached close, so this is the 11 rungs strictly between them.
    "min5":   [55 / 60, 50 / 60, 45 / 60, 40 / 60, 35 / 60, 30 / 60,
               25 / 60, 20 / 60, 15 / 60, 10 / 60, 5 / 60],
    # GRID TEST. Three adjacent 5-minute rungs on a couple of events, to find out
    # whether the API's historical snapshots are actually 5 minutes apart for a
    # 2023-24 date. If snapshot_actual repeats across them the grid is coarser and a
    # 5-minute ladder would bill twice for identical data -- 40 credits to protect
    # roughly ten thousand.
    "grid":   [50 / 60, 55 / 60, 1.0],
}

# The demo set: the top N of the shortlist. Cost scales with the DISTINCT EVENTS these
# land in, which is materially fewer than N -- several events carry two flagged players.
TOP_N = 150
ROLE = "poll"
TOL_H = 0.12          # how close a cached snapshot must be to count as that rung


def _iso(commence, hours):
    return (commence.astimezone(timezone.utc)
            - timedelta(hours=hours)).strftime(FMT)


def _cached(event_id, iso):
    return any(cache.status("snapshot", cache.snapshot_key(event_id, iso, config.MARKET, b))
               for b in ("all", "fanduel"))


def targets(conn, group):
    """(event_id, commence_time) for the demo set, split by whether the FLAGGED player
    already had a line at T-12h.

    The split is per EVENT because that is the billing unit. An event is 'early' if any
    of its flagged players was priced at T-12h; those get the hourly ladder. The rest
    are 'late' and get the probe first.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT e.event_id, e.commence_time,
                   EXISTS (SELECT 1 FROM prop_quotes q
                            WHERE q.event_id  = e.event_id
                              AND q.player_id = s.player_id
                              AND q.snapshot_role = 'open') AS early
              FROM player_game_scores s
              JOIN odds_events e ON e.game_id = s.game_id
             WHERE s.in_shortlist AND s.rank <= %s
             ORDER BY e.commence_time""", (TOP_N,))
        rows = cur.fetchall()
    # An event counts as early if ANY flagged player in it was priced at T-12h.
    early = {e for e, _, ok in rows if ok}
    # 'unpriced': the flagged player still has NO poll quote at any rung pulled so far.
    # This is the set each successive probe narrows, so a ladder is only ever bought
    # for events where the line has actually been located.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT e.event_id
              FROM player_game_scores s
              JOIN odds_events e ON e.game_id = s.game_id
             WHERE s.in_shortlist AND s.rank <= %s
               AND NOT EXISTS (SELECT 1 FROM prop_quotes q
                                WHERE q.event_id = e.event_id
                                  AND q.player_id = s.player_id
                                  AND q.snapshot_role = 'poll')""", (TOP_N,))
        unpriced = {r[0] for r in cur.fetchall()}
    keep = {"early": lambda e: e in early, "late": lambda e: e not in early,
            "unpriced": lambda e: e in unpriced, "priced": lambda e: e not in unpriced,
            "all": lambda e: True}[group]
    seen, out = set(), []
    for eid, ct, _ in rows:
        if eid in seen:
            continue
        seen.add(eid)
        if keep(eid):
            out.append((eid, ct))
    return out, early


def rows_for(event_id, commence, payload, hours, resolve, unresolved, methods):
    """One (event, rung) payload -> prop_quotes rows tagged 'poll'."""
    snap_req = _iso(commence, hours)
    actual = (payload or {}).get("timestamp") or snap_req
    offset = int((commence - datetime.strptime(actual, FMT).replace(tzinfo=timezone.utc))
                 .total_seconds())
    out = []
    for q in quotes(payload):
        pid, method = resolve(q["player_name_raw"])
        methods[method or "UNRESOLVED"] = methods.get(method or "UNRESOLVED", 0) + 1
        if pid is None:
            unresolved[q["player_name_raw"]] = unresolved.get(q["player_name_raw"], 0) + 1
        out.append({
            "event_id": event_id, "player_name_raw": q["player_name_raw"],
            "player_id": pid, "market": config.MARKET, "book": q["book"],
            "snapshot_requested": snap_req, "snapshot_actual": actual,
            "line_last_update": q["line_last_update"], "snapshot_role": ROLE,
            "offset_from_tip_sec": offset, "line": q["line"],
            "over_price": q["over_price"], "under_price": q["under_price"]})
    return out


def verify(conn):
    """score_100 must be byte-identical after any poll write. If this ever fails, a
    'poll' row has leaked into the scoring path and the run must be treated as a
    methodology change, not a cosmetic one."""
    with conn.cursor() as cur:
        cur.execute("""SELECT count(*),
              count(*) FILTER (WHERE s.score_100 IS DISTINCT FROM z.score_100),
              count(*) FILTER (WHERE s.score IS DISTINCT FROM z.score)
            FROM player_game_scores s
            JOIN player_game_z z USING (player_id, game_id)""")
        n, d100, draw = cur.fetchone()
        cur.execute("SELECT count(*) FROM prop_quotes WHERE snapshot_role = 'poll'")
        polls = cur.fetchone()[0]
    print(f"rows compared            : {n:,}")
    print(f"score_100 differing      : {d100}")
    print(f"score differing          : {draw}")
    print(f"prop_quotes 'poll' rows  : {polls:,}")
    if d100 or draw:
        sys.exit("!! SCORES MOVED -- a poll row reached the scoring path")
    print("OK: the poll rows are invisible to the model.")


def main(mode, plan, group, limit=None):
    offsets = PLANS[plan]
    with db.connect() as conn:
        evs, early = targets(conn, group)
        # --limit N keeps the first N events, so a ladder can be rehearsed on a couple
        # of games before it is bought for all of them.
        if limit:
            evs = evs[:limit]
        rungs = [(e, ct, h) for e, ct in evs for h in offsets]
        have = [(e, ct, h) for e, ct, h in rungs if _cached(e, _iso(ct, h))]
        need = [r for r in rungs if r not in set(have)]

        print(f"plan                     : {plan}  rungs {offsets}")
        print(f"group                    : {group}   "
              f"(early={len(early)} of {len(evs) if group=='all' else '...'} events)")
        print(f"events                   : {len(evs)}")
        print(f"event-rungs total        : {len(rungs)}")
        print(f"  already cached         : {len(have)}   <- free")
        print(f"  to fetch               : {len(need)}")
        n_calls = 0 if mode == "cached" else len(need)
        print(f"==> {'PLANNED' if mode == 'run' else 'WOULD COST'} billed calls : "
              f"{n_calls}   ({n_calls * 10:,} credits at 10/call)")
        before = cache.report_plan(n_calls, "historical_odds", conn)

        resolve = make_resolver(conn)
        todo = rungs if mode == "run" else have
        rows, unresolved, methods, empty = [], {}, {}, 0
        for eid, ct, h in todo:
            payload, _ = snapshot(eid, _iso(ct, h), conn=conn,
                                  allow_fetch=(mode == "run"))
            if payload is None:
                continue
            r = rows_for(eid, ct, payload, h, resolve, unresolved, methods)
            if not r:
                empty += 1
            rows.extend(r)

        print(f"\nevent-rungs processed    : {len(todo)}")
        print(f"  returning no line      : {empty}")
        print(f"prop_quotes rows built   : {len(rows):,}")
        print(f"unresolved player names  : {len(unresolved)}")
        for nm, k in sorted(unresolved.items(), key=lambda x: -x[1])[:8]:
            print(f"   !! {nm}  ({k} rows)")

        if mode == "dry":
            print(cache.summary())
            db.dry_notice()
            return

        n = db.upsert(conn, "prop_quotes", rows,
                      conflict=["event_id", "player_name_raw", "market", "book",
                                "snapshot_requested", "line"])
        print(f"\nupserted {n:,} rows -> prop_quotes")
        print(cache.summary())
        cache.report_spend(before)
    with db.connect() as conn:
        print()
        verify(conn)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--verify" in a:
        with db.connect() as conn:
            verify(conn)
        sys.exit()
    _plan = next((a[i + 1] for i, t in enumerate(a) if t == "--plan" and i + 1 < len(a)),
                 "probe")
    _group = next((a[i + 1] for i, t in enumerate(a) if t == "--group" and i + 1 < len(a)),
                  {"probe": "late", "probe1": "unpriced", "half": "late",
                   "hourly": "early", "min5": "unpriced", "grid": "unpriced"}[_plan])
    _limit = next((int(a[i + 1]) for i, t in enumerate(a)
                   if t == "--limit" and i + 1 < len(a)), None)
    main(next((t for t in a if t in ("run", "cached")), "dry"), _plan, _group, _limit)
