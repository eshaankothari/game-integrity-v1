"""L3b: line withdrawal -> `player_line_pulls`. Reads cache only, 0 API calls.

WHY THIS EXISTS. Jontay Porter's 2024-01-26 game -- one of two the NBA later banned
him over -- has ZERO rows in prop_quotes, so no cut can ever reach it. Not because the
market ignored him but because of what the market did:

    T-4h  draftkings, fanduel, williamhill_us, betmgm, +4
    T-3h  fanduel, williamhill_us, betmgm, +4          <- draftkings gone
    T-2h  fanduel, williamhill_us, betmgm, +4
    T-1h  pointsbetus ONLY
    tip   pointsbetus ONLY

Every book we store had pulled his points line before tip. load_props.py probes at
tip exactly and keeps three books, so it correctly recorded that nothing was there --
and threw away the fact that something had been.

A BOOK WITHDRAWING A LINE IS THE STRONGEST SIGNAL IN THE MARKET. It is what happens
when a trader concludes he cannot price the player safely: one-way money, an injury
whisper, or action he does not want. Absence in the closing snapshot is the shape of
that evidence, and absence is invisible to every downstream cut.

MEASURED ACROSS ALL TEN BOOKS, not the three we store. The withdrawal is the event of
interest and it happens book by book, so counting only three would have missed that
four others also dropped him. A line surviving on one offshore book while the majors
leave is a different situation from a line nobody wanted.

FREE, AND ALREADY PAID FOR. 454 of 1,230 events have two or more cached snapshots
from v0's laddered pulls. No new call is made or needed; this is re-reading what is
on disk. Events with a single snapshot get no row -- one observation cannot show a
change, and recording them as 'not pulled' would be a false negative.

    python -m pipeline.load_data.load_line_pulls            # DRY
    python -m pipeline.load_data.load_line_pulls run        # write
"""
import glob
import json
import re
import sys
from collections import defaultdict

from pipeline.core import config
from pipeline.core import db
from pipeline.load_data.load_players import make_resolver

NAME_RE = re.compile(r"([0-9a-f]{32})_(\d{4}-\d{2}-\d{2}T\d{6}Z)_player_points_([a-zA-Z]+)_")

DDL = """
CREATE TABLE IF NOT EXISTS player_line_pulls (
    player_id       INTEGER NOT NULL REFERENCES players (player_id),
    event_id        TEXT    NOT NULL,
    game_id         TEXT    REFERENCES games (game_id),
    player_name_raw TEXT,
    n_snapshots     INTEGER,      -- distinct timestamps seen for this event
    first_seen      TEXT,         -- earliest snapshot the player had ANY line in
    last_seen       TEXT,         -- latest snapshot he had one
    close_snapshot  TEXT,         -- the latest snapshot cached for the event
    books_first     INTEGER,      -- books quoting him at first_seen
    books_close     INTEGER,      -- books quoting him in the closing snapshot
    books_dropped   INTEGER,
    line_pulled     BOOLEAN,      -- quoted earlier, by NOBODY at close
    major_pulled    BOOLEAN,      -- quoted earlier by a major, by no major at close
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, event_id)
);
"""

# The three we store in prop_quotes. `major_pulled` tracks these specifically because
# a line surviving only on an offshore book is, for our purposes, no line at all --
# that is exactly the Porter case.
MAJORS = {"fanduel", "draftkings", "williamhill_us"}


def snapshots():
    """{event_id: {timestamp: path}} over every cached points snapshot.

    Prefers the all-books file when a timestamp has both, since the fanduel-only
    variant cannot show what the other nine did.
    """
    out = defaultdict(dict)
    for d in (config.SNAPSHOT_CACHE, config.SNAPSHOT_CACHE_V1):
        for path in glob.glob(f"{d}/*player_points*"):
            m = NAME_RE.match(path.split("/")[-1])
            if not m:
                continue
            eid, ts, book = m.groups()
            prev = out[eid].get(ts)
            if prev is None or book.lower().startswith("all"):
                out[eid][ts] = path
    return out


def books_by_player(path):
    """One snapshot file -> {player name: {book keys quoting him}}."""
    try:
        payload = json.load(open(path))
    except Exception:
        return {}
    data = (payload or {}).get("data") or {}
    out = defaultdict(set)
    for bk in data.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != config.MARKET:
                continue
            for o in mk.get("outcomes", []):
                if o.get("point") is not None and o.get("description"):
                    out[o["description"]].add(bk["key"])
    return out


def build(events, resolve, game_of, unresolved):
    rows = []
    for eid, by_ts in events.items():
        if len(by_ts) < 2:            # one observation cannot show a change
            continue
        order = sorted(by_ts)
        close_ts = order[-1]
        seen = {ts: books_by_player(by_ts[ts]) for ts in order}
        names = {n for m in seen.values() for n in m}
        for name in names:
            present = [ts for ts in order if seen[ts].get(name)]
            if not present:
                continue
            pid, _ = resolve(name)
            if pid is None:
                unresolved[name] = unresolved.get(name, 0) + 1
                continue
            first, last = present[0], present[-1]
            b_first = seen[first].get(name, set())
            b_close = seen[close_ts].get(name, set())
            rows.append({
                "player_id": pid, "event_id": eid, "game_id": game_of.get(eid),
                "player_name_raw": name, "n_snapshots": len(order),
                "first_seen": first, "last_seen": last, "close_snapshot": close_ts,
                "books_first": len(b_first), "books_close": len(b_close),
                "books_dropped": len(b_first) - len(b_close),
                "line_pulled": len(b_close) == 0,
                "major_pulled": bool(b_first & MAJORS) and not (b_close & MAJORS),
            })
    return rows


def main(dry=True):
    events = snapshots()
    multi = {e: v for e, v in events.items() if len(v) >= 2}
    print(f"events with a cached points snapshot : {len(events):,}")
    print(f"  with 2 or more timestamps          : {len(multi):,}  <- usable")
    print(f"  single snapshot, skipped           : {len(events) - len(multi):,}")
    print("==> API calls: 0 (cache only)")

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute("SELECT event_id, game_id FROM odds_events")
            game_of = dict(cur.fetchall())
        resolve = make_resolver(conn)
        unresolved = {}
        rows = build(multi, resolve, game_of, unresolved)

        pulled = [r for r in rows if r["line_pulled"]]
        major = [r for r in rows if r["major_pulled"]]
        print(f"\nplayer-events built    : {len(rows):,}")
        print(f"  line_pulled (all books gone)   : {len(pulled):,}")
        print(f"  major_pulled (FD/DK/Caesars)   : {len(major):,}")
        print(f"unresolved names       : {len(unresolved)}")

        if dry:
            db.dry_notice()
            return
        n = db.upsert(conn, "player_line_pulls", rows,
                      conflict=["player_id", "event_id"])
        print(f"\nupserted {n:,} rows -> player_line_pulls")


if __name__ == "__main__":
    main(dry=db.is_dry())
