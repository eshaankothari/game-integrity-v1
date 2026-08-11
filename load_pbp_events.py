"""L4c: per-event play-by-play -> `player_game_events`. Reads cache only, 0 API calls.

WHY THIS EXISTS. The shot chart and the play timeline used to read the cached pbp CSV
for a game at REQUEST time -- server/app.py opened a file off disk inside the endpoint.
That works on the machine that ran the ingest and nowhere else: the cache is 117 MB of
paid API responses, deliberately not in the repo, so those two panels rendered empty for
anyone who cloned it. Everything else on the case page came from the database and worked.

Moving the events into a table fixes that asymmetry. The data ships wherever the database
ships -- including inside the single-file DuckDB export -- and the API loses its last
filesystem dependency.

SCOPE: only the (player, game) pairs the case view can actually open, i.e. the propped
player-games in `player_game_scores`. The full pbp is ~610,000 rows across 1,230 games;
restricting to what is reachable is a large cut for no loss of function, and an unopenable
row is a row nobody can see.

COLUMNS: exactly the fifteen the two endpoints read, no more. The raw CSVs stay in the
cache for anything that later wants the rest.

    python load_pbp_events.py          # DRY: what it would write
    python load_pbp_events.py run      # write
"""
import sys

import pandas as pd

import cache
import config
import db

DDL = """
CREATE TABLE IF NOT EXISTS player_game_events (
    game_id         TEXT    NOT NULL,
    player_id       INTEGER NOT NULL,
    action_number   INTEGER NOT NULL,
    period          INTEGER,
    clock           TEXT,
    description     TEXT,
    action_type     TEXT,
    -- tri-state at the API: 'Made'/'Missed' for shots, NULL for everything else
    shot_result     TEXT,
    is_field_goal   BOOLEAN,
    x_legacy        REAL,
    y_legacy        REAL,
    shot_distance   REAL,
    shot_value      REAL,
    video_available BOOLEAN,
    score_away      INTEGER,
    score_home      INTEGER,
    PRIMARY KEY (game_id, player_id, action_number)
);
CREATE INDEX IF NOT EXISTS player_game_events_lookup
    ON player_game_events (player_id, game_id);
"""

COLS = {"actionNumber": "action_number", "period": "period", "clock": "clock",
        "description": "description", "actionType": "action_type",
        "shotResult": "shot_result", "isFieldGoal": "is_field_goal",
        "xLegacy": "x_legacy", "yLegacy": "y_legacy",
        "shotDistance": "shot_distance", "shotValue": "shot_value",
        "videoAvailable": "video_available",
        "scoreAway": "score_away", "scoreHome": "score_home"}


def _clean(v):
    """NaN -> None. pandas keeps missing numerics as float NaN, which psycopg would
    happily store as the float NaN rather than SQL NULL."""
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def main(mode="dry"):
    with db.connect() as conn:
        # The pairs the case view can reach. Anything else is unopenable.
        want = pd.read_sql(
            "SELECT player_id, game_id FROM player_game_scores", conn)
        by_game = want.groupby("game_id")["player_id"].apply(set).to_dict()
        print(f"propped player-games : {len(want):,} across {len(by_game):,} games")

        rows, missing, scanned = [], 0, 0
        for game_id, players in sorted(by_game.items()):
            df = cache.read_cached("nba", cache.nba_key("pbp", game_id), fmt="csv")
            if df is None:
                missing += 1
                continue
            scanned += 1
            d = df[df["personId"].isin(players)]
            for _, e in d.iterrows():
                r = {"game_id": game_id, "player_id": int(e["personId"])}
                for src, dst in COLS.items():
                    r[dst] = _clean(e.get(src))
                r["action_number"] = int(r["action_number"])
                # The CSV carries these as 0/1, which Postgres will not accept into a
                # BOOLEAN column. NULL stays NULL -- absent is not the same as False.
                for b in ("is_field_goal", "video_available"):
                    r[b] = None if r[b] is None else bool(r[b])
                for i in ("period", "score_away", "score_home"):
                    r[i] = None if r[i] is None else int(r[i])
                rows.append(r)

        print(f"pbp files read       : {scanned:,}")
        if missing:
            print(f"  !! not cached      : {missing:,} games -- their panels stay empty")
        print(f"events to write      : {len(rows):,}")
        if rows:
            n_pairs = len({(r['game_id'], r['player_id']) for r in rows})
            print(f"  covering           : {n_pairs:,} player-games")

        if mode != "run":
            db.dry_notice()
            return

        with conn.cursor() as cur:
            cur.execute(DDL)
        n = db.upsert(conn, "player_game_events", rows,
                      conflict=["game_id", "player_id", "action_number"])
        print(f"\nupserted {n:,} rows -> player_game_events "
              f"({db.count(conn, 'player_game_events'):,} total)")


if __name__ == "__main__":
    main("run" if "run" in sys.argv[1:] else "dry")
