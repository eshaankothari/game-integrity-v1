"""L3: closing points props -> `prop_quotes`. The layer that costs real money.

SUMMARY: This file fetches all player prop bets from OddsAPI and using the event_id and commencement time
in odds_events db. It stores line, over and under price for points  in the prop_quotes db
and gets information from the top 3 sporting books in the US (fanduel, draftkings, and Ceasar).
Right now, we are only fetching closing line information for API cost purposes.


ONE probe per event, at odds_events.commence_time exactly -- the closing line.
v0's cache shows the close ladder never fired past its first rung (zero snapshots
at tip-20min or tip-40min), so a single tip-exact probe is enough.

FETCH ALL BOOKS, WRITE THREE. The `bookmakers` filter is free but it changes the
cache key, so requesting a subset would forfeit v0's 395 already-paid closing
snapshots. We request regions=us (key stays `_all_`, reuse intact) and filter to
BOOKS at write time -- same credits, one third the rows, and the untouched 10-book
JSON stays on disk if it is ever wanted.

Opening lines are deliberately NOT fetched yet. They are `open` ROWS in this same
table, not extra columns, so adding them later is an INSERT and nothing downstream
changes. Historical snapshots never expire, so deferring costs nothing.

    python -m pipeline.load_data.load_props             # DRY: exact credit count, writes nothing
    python -m pipeline.load_data.load_props cached      # write ONLY what is already cached, 0 credits
    python -m pipeline.load_data.load_props run         # fetch missing closes + write
"""
import sys
from datetime import datetime, timedelta, timezone

import requests

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db
from pipeline.load_data.load_players import make_resolver

ODDS_URL = ("https://api.the-odds-api.com/v4/historical/sports/basketball_nba/"
            "events/{event_id}/odds")
FMT = "%Y-%m-%dT%H:%M:%SZ"

# Probe offsets before tip, by snapshot_role.
#
# close = tip exactly. v0's cache shows the close ladder never fired past its first
#         rung (zero snapshots at tip-20min or tip-40min), so one probe suffices.
# open  = T-12h. NOT the true first tick of the line -- books post progressively and
#         only ~55% of players who eventually have a line have one this early. It is
#         chosen because 219 of the 303 currently-analysable events ALREADY have a
#         T-12h snapshot cached from v0, and no other offset comes close (T-3h has 27
#         cached across all 1,230). Free reuse beats marginal coverage here.
#
#         Players whose line does not exist 12h out keep open_line NULL, and
#         offset_from_tip_sec records the real window on every row, so heterogeneous
#         windows are labelled rather than hidden.
ROLE_OFFSET = {"close": timedelta(0), "open": timedelta(hours=12)}
BOOKS = ("fanduel", "draftkings", "williamhill_us")     # written; all 10 are fetched


def _call_odds(event_id, snapshot_iso):
    """One historical-odds call. -> (payload, meta). No bookmakers filter."""
    r = requests.get(ODDS_URL.format(event_id=event_id), params={
        "apiKey": config.ODDS_API_KEY, "regions": config.REGIONS,
        "markets": config.MARKET, "oddsFormat": "decimal", "dateFormat": "iso",
        "date": snapshot_iso})
    meta = {"http_status": r.status_code,
            "requests_used": r.headers.get("x-requests-used"),
            "requests_remaining": r.headers.get("x-requests-remaining"),
            "credits_delta": None}   # real cost measured from headers in cache.measured_delta
    if r.status_code == 404:
        return {"data": None}, meta          # negative-cached, never re-billed
    r.raise_for_status()
    return r.json(), meta


def snapshot(event_id, snapshot_iso, conn=None, allow_fetch=False):
    """Closing snapshot for one event -> (payload, source) or (None, None).

    Prefers the all-books cache over the fanduel-only one: when both exist they
    cost the same (nothing) and all-books carries nine more books.
    """
    for book in ("all", "fanduel"):
        key = cache.snapshot_key(event_id, snapshot_iso, config.MARKET, book)
        payload = cache.read_cached("snapshot", key)
        if payload is not None:
            return payload, f"cache:{book}"
    if not allow_fetch:
        return None, None
    key = cache.snapshot_key(event_id, snapshot_iso, config.MARKET, "all")
    payload, src = cache.get(
        "snapshot", key, lambda: _call_odds(event_id, snapshot_iso),
        api="oddsapi", endpoint="historical_odds", conn=conn,
        params={"event_id": event_id, "date": snapshot_iso, "market": config.MARKET})
    return payload, src


def quotes(payload):
    """Flatten a response into one dict per (book, player, line).

    Outcomes arrive as separate Over and Under entries; they are grouped by
    (player, point) so alternate-line books collapse correctly instead of the
    last Over silently overwriting the first.
    """
    out = []
    data = (payload or {}).get("data") or {}
    for bk in data.get("bookmakers", []):
        if bk["key"] not in BOOKS:
            continue
        for mk in bk.get("markets", []):
            if mk["key"] != config.MARKET:
                continue
            acc = {}
            for o in mk.get("outcomes", []):
                if o.get("point") is None:
                    continue
                acc.setdefault((o["description"], o["point"]), {})[o["name"].lower()] = o.get("price")
            for (name, point), prices in acc.items():
                out.append({"book": bk["key"], "player_name_raw": name, "line": point,
                            "over_price": prices.get("over"),
                            "under_price": prices.get("under"),
                            "line_last_update": mk.get("last_update")})
    return out


def rows_for(event_id, commence, payload, resolve, unresolved, methods,
             role="close"):
    """One event's payload -> prop_quotes row dicts."""
    snap_req = (commence.astimezone(timezone.utc) - ROLE_OFFSET[role]).strftime(FMT)
    actual = (payload or {}).get("timestamp") or snap_req
    # Measured from what the API actually RETURNED, not what we asked for: it snaps
    # back to a 5-minute grid, so a "tip-exact" probe really lands ~4-5 min early.
    offset = int((commence - datetime.strptime(actual, FMT).replace(tzinfo=timezone.utc))
                 .total_seconds())
    rows = []
    for q in quotes(payload):
        pid, method = resolve(q["player_name_raw"])
        methods[method or "UNRESOLVED"] = methods.get(method or "UNRESOLVED", 0) + 1
        if pid is None:
            unresolved[q["player_name_raw"]] = unresolved.get(q["player_name_raw"], 0) + 1
        rows.append({
            "event_id": event_id, "player_name_raw": q["player_name_raw"],
            "player_id": pid, "market": config.MARKET, "book": q["book"],
            "snapshot_requested": snap_req, "snapshot_actual": actual,
            "line_last_update": q["line_last_update"], "snapshot_role": role,
            "offset_from_tip_sec": offset, "line": q["line"],
            "over_price": q["over_price"], "under_price": q["under_price"]})
    return rows


def main(mode="dry", date=None, game=None, role="close", analysable=False):
    dry, allow_fetch = (mode == "dry"), (mode == "run")
    with db.connect() as conn:
        with conn.cursor() as cur:
            # Joined to `games` so a --date scope filters on the NBA game date
            # (US/Eastern) rather than commence_time's UTC date -- a 10pm tip would
            # otherwise land on the wrong night.
            cur.execute("""
                SELECT e.event_id, e.commence_time
                FROM odds_events e JOIN games g ON g.game_id = e.game_id
                WHERE (%s::date IS NULL OR g.game_date = %s::date)
                  AND (%s::text IS NULL OR g.game_id   = %s::text)
                  -- `analysable` restricts to events we can actually USE: ones with
                  -- props AND box scores. Fetching an opening line for a game with no
                  -- box score buys nothing, because there is no performance to join it
                  -- to. With L4 barely a third done that is 303 events instead of
                  -- 1,230 -- a tenfold cut in spend, deferred rather than avoided.
                  -- Re-run as L4 fills in and it picks up newly-joinable events.
                  -- NB keep percent signs out of this comment entirely: psycopg
                  -- parses one as a placeholder and the whole query fails.
                  AND (NOT %s OR (
                        EXISTS (SELECT 1 FROM prop_quotes q
                                WHERE q.event_id = e.event_id AND q.book = %s)
                    AND EXISTS (SELECT 1 FROM player_games pg
                                WHERE pg.game_id = g.game_id)))
                ORDER BY e.event_id""", (date, date, game, game, analysable, BOOKS[0]))
            events = cur.fetchall()
        if not events:
            sys.exit(f"!! no events match {db.scope_label(date, game)}")
        resolve = make_resolver(conn)

        def snap_iso(ct):
            return (ct.astimezone(timezone.utc) - ROLE_OFFSET[role]).strftime(FMT)

        cached = [(e, c) for e, c in events
                  if any(cache.status("snapshot",
                                      cache.snapshot_key(e, snap_iso(c), config.MARKET, b))
                         for b in ("all", "fanduel"))]
        missing = [(e, c) for e, c in events if (e, c) not in set(cached)]

        print(f"role                     : {role}  (T-{int(ROLE_OFFSET[role].total_seconds()//3600)}h before tip)")
        print(f"scope                    : {db.scope_label(date, game)}"
              + ("  [analysable only]" if analysable else ""))
        print(f"events                   : {len(events)}")
        print(f"closing snapshot cached  : {len(cached)}  <- free")
        print(f"closing snapshot missing : {len(missing)}")
        # DRY must quote what `run` WOULD cost -- that is its entire purpose. Only
        # `cached` is genuinely 0, because it can never reach the network.
        n_calls = 0 if mode == "cached" else len(missing)
        print(f"==> {'PLANNED' if mode == 'run' else 'WOULD COST'} billed calls : {n_calls}")
        before = cache.report_plan(n_calls, "historical_odds", conn)
        print(f"books written            : {', '.join(BOOKS)}  (all 10 fetched, 3 stored)")

        todo = events if allow_fetch else cached
        unresolved, methods, rows, no_line = {}, {}, [], 0
        for eid, ct in todo:
            iso = snap_iso(ct)
            payload, src = snapshot(eid, iso, conn=conn, allow_fetch=allow_fetch)
            if payload is None:
                continue
            r = rows_for(eid, ct, payload, resolve, unresolved, methods, role)
            if not r:
                no_line += 1
            rows.extend(r)

        print(f"\nevents processed         : {len(todo)}")
        print(f"  with no line in BOOKS  : {no_line}")
        print(f"prop_quotes rows built   : {len(rows):,}")
        print(f"name resolution          : "
              + ", ".join(f"{k}={v:,}" for k, v in sorted(methods.items())))
        print(f"unresolved player names  : {len(unresolved)}")
        for name, n in sorted(unresolved.items(), key=lambda x: -x[1])[:10]:
            print(f"   !! {name}  ({n} rows)")

        if dry:
            print(cache.summary())
            db.dry_notice()
            return

        n = db.upsert(conn, "prop_quotes", rows,
                      conflict=["event_id", "player_name_raw", "market", "book",
                                "snapshot_requested", "line"])
        print(f"\nupserted {n:,} rows -> prop_quotes ({db.count(conn, 'prop_quotes'):,} total)")
        print(cache.summary())
        cache.report_spend(before)


if __name__ == "__main__":
    _args = sys.argv[1:]
    _mode = next((a for a in _args if a in ("run", "cached")), "dry")
    _role = next((_args[i + 1] for i, a in enumerate(_args)
                  if a == "--role" and i + 1 < len(_args)), "close")
    _date, _game = db.scope()
    main(mode=_mode, date=_date, game=_game, role=_role,
         analysable="--analysable" in _args)
