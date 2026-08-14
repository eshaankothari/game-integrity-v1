"""L1d: career team history for all propped players -> `player_career`.

SUMMARY: One PlayerCareerStats call per propped player, derive "journeyman"
metrics (stints, franchises, recent moves, gap seasons), and upsert one row per
player.

WHY THIS EXISTS. The salary side of the motive axis knows what a player is paid
but not how precarious his career has BEEN. A player on his fourth organisation
in three seasons has a different exposure profile from a same-salary player who
has sat on one bench his whole career. standardize.py folds `instability_last3`
into the motive block as a percentile (see MOTIVE_W there).

POPULATION. Every distinct player_id in `player_game_scores` -- the full propped
population (~415 players), NOT just the shortlist. Motive is scored over all
15,498 propped rows, so the instability percentile must be computed over all of
them; a shortlist-only table would leave most of the ranking population NULL.

SHAPE OF THE SOURCE. SeasonTotalsRegularSeason has one row per SEASON-STINT: a
mid-season trade produces two-plus rows in one season, and seasons with multiple
teams also carry a TEAM_ABBREVIATION='TOT' aggregate row, which is excluded from
every count here. Only seasons <= config.SEASON are counted, so a 2024-25 row
fetched today cannot leak into metrics about the scored season.

MISSING != FAILED (invariant 1). A player whose fetch returns no usable rows gets
NULLs, not zeros -- zero stints is a claim, NULL is an absence.

    python -m pipeline.load_data.load_career            # DRY
    python -m pipeline.load_data.load_career run        # ~415 calls total, free
"""
import sys
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import playercareerstats

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db

TIMEOUT = 20
RETRIES = 3
INTERVAL = 0.7          # seconds between NETWORK calls; nba_api is free but rate-limited

# The 3 most recent seasons up to and including the scored season define the
# "recent" window for moves_last3.
LAST_N = 3

DDL = """
CREATE TABLE IF NOT EXISTS player_career (
    player_id        INTEGER PRIMARY KEY REFERENCES players (player_id),
    seasons_played   INTEGER,
    career_stints    INTEGER,
    career_teams     INTEGER,
    moves_last3      INTEGER,
    gap_seasons      INTEGER,
    instability_last3 INTEGER,
    instability_career NUMERIC(6,3),
    moves_per_season NUMERIC(6,3),
    first_season     TEXT,
    last_season      TEXT,
    loaded_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# CREATE TABLE IF NOT EXISTS is a no-op on the existing 188-row table, so newly
# added columns need an explicit ALTER too (same lesson as standardize.py's Z_ALTER).
ALTER = {"gap_seasons": "INTEGER", "instability_last3": "INTEGER",
         "instability_career": "NUMERIC(6,3)"}

CSV_TYPES = {                       # for data/_types.json, CSV-backend fallback only
    "player_id": "INTEGER", "seasons_played": "INTEGER", "career_stints": "INTEGER",
    "career_teams": "INTEGER", "moves_last3": "INTEGER",
    "gap_seasons": "INTEGER", "instability_last3": "INTEGER",
    "instability_career": "DOUBLE", "moves_per_season": "DOUBLE", "first_season": "VARCHAR", "last_season": "VARCHAR",
}

# The stint-level companion to player_career: the same cached frames kept as ROWS
# rather than summarised into a rate, so the case view can DRAW the movement the
# instability number counts. seq is the 0-based position of the stint across the
# whole career (within-season API order preserved), so ORDER BY seq replays the
# path exactly. Nothing here feeds a score -- display only.
STINTS_DDL = """
CREATE TABLE IF NOT EXISTS player_career_stints (
    player_id  INTEGER NOT NULL REFERENCES players (player_id),
    season     TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    team_abbr  TEXT NOT NULL,
    PRIMARY KEY (player_id, seq)
);
"""

STINTS_CSV_TYPES = {"player_id": "INTEGER", "season": "VARCHAR",
                    "seq": "INTEGER", "team_abbr": "VARCHAR"}


def _season_range(first_year, last_year):
    """['2020-21', '2021-22', ...] inclusive. Handles the '1999-00' rollover."""
    return [f"{y}-{str(y + 1)[-2:].zfill(2)}" for y in range(first_year, last_year + 1)]


def _call(player_id):
    """One PlayerCareerStats call -> (SeasonTotalsRegularSeason frame, meta)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = playercareerstats.PlayerCareerStats(player_id=player_id,
                                                    timeout=TIMEOUT)
            return (r.season_totals_regular_season.get_data_frame(),
                    {"http_status": 200, "credits_delta": 0})
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last = e
            if attempt == RETRIES:
                break
            time.sleep(2 * attempt)
    raise last


def fetch_player(player_id, conn=None, allow_fetch=False):
    key = cache.nba_key("career", player_id)
    if not allow_fetch:
        return cache.read_cached("nba", key, fmt="csv")
    try:
        df, src = cache.get("nba", key, lambda: _call(player_id),
                            api="nba", endpoint="playercareerstats", conn=conn,
                            params={"player_id": player_id}, fmt="csv")
        if src == "network":
            time.sleep(INTERVAL)
        return df
    except Exception as e:
        print(f"   !! career {player_id}: {type(e).__name__}")
        return None


def derive(player_id, df):
    """One career frame -> one player_career row. NULLs when there is nothing to count.

    Stints are kept in API row order within a season (chronological for traded
    players), sorted stably by SEASON_ID across seasons. moves_last3 counts every
    stint->stint team change whose LATER stint falls in the player's LAST_N most
    recent seasons -- so both a mid-season trade inside the window and the change
    of teams that brought him INTO the window count as moves.

    gap_seasons counts seasons strictly inside [first_season, last_season] with no
    NBA row at all -- a season the league didn't want him. G-League/overseas years
    BEFORE the debut or AFTER the last season are invisible here and deliberately
    not counted: outside the span there is no evidence he was trying to be in the
    league.

    instability_last3 = moves_last3 + the gap seasons that fall in the LAST_N most
    recent LEAGUE seasons anchored at config.SEASON (for 2023-24: 2021-22, 2022-23,
    2023-24). The gap window is league-anchored where the moves window is
    player-anchored, because a gap is only observable against the league calendar --
    and a gap season counts only if it lies within the player's own career span.

    instability_career = (career team changes + 2*gap_seasons) / (span_seasons + 1),
    where span_seasons counts every league season in [first_season, last_season]
    inclusive -- gaps live inside the same span the denominator measures. This is
    the RATE form, and the one standardize.py scores. Two deliberate departures
    from the plain rate, both measured before adoption (see
    instability_variants.py in the analysis scratch dir):

      GAPS COUNT DOUBLE. A trade means some team wanted him; a gap season means
      no team did. Weighting gaps 2x puts the chronic-fringe profile (Jontay
      Porter: (1 move + 2*2 gaps) / (4+1) = 1.000) above the successful-vet
      journeyman (Seth Curry/Ish Smith profile) whose moves came with a contract
      every year.

      THE DENOMINATOR CARRIES ONE PSEUDO-SEASON. Without it, a one-season player
      with a single mid-season trade posts the most extreme rate in the league
      (1.000) off one observation. span+1 shrinks short careers toward calm and
      lets long careers speak for themselves; heavier shrinkage (EB alpha=2) was
      tested and starts pulling down real fringe players, so one season is the
      dose.
    """
    row = {"player_id": player_id, "seasons_played": None, "career_stints": None,
           "career_teams": None, "moves_last3": None, "gap_seasons": None,
           "instability_last3": None, "instability_career": None,
           "moves_per_season": None, "first_season": None, "last_season": None}
    if df is None or df.empty:
        return row
    d = df.copy()
    d["SEASON_ID"] = d["SEASON_ID"].astype(str)
    d = d[d["SEASON_ID"] <= config.SEASON]                     # nothing post-dates 2023-24
    d = d[d["TEAM_ABBREVIATION"].astype(str) != "TOT"]         # aggregates are not stints
    if d.empty:
        return row
    d = d.sort_values("SEASON_ID", kind="stable")              # keeps within-season order

    seasons = sorted(d["SEASON_ID"].unique())
    teams = d["TEAM_ID"].tolist()
    stint_seasons = d["SEASON_ID"].tolist()
    moves = [teams[i] != teams[i - 1] for i in range(1, len(teams))]
    window = set(seasons[-LAST_N:])
    moves_last3 = sum(m for m, s in zip(moves, stint_seasons[1:]) if s in window)

    # Gap seasons: the career span year-by-year, minus the seasons with a row.
    span = _season_range(int(seasons[0][:4]), int(seasons[-1][:4]))
    gaps = [s for s in span if s not in set(seasons)]

    # League-anchored recent window: the LAST_N seasons up to config.SEASON.
    anchor = int(config.SEASON[:4])
    league_window = set(_season_range(anchor - LAST_N + 1, anchor))
    gaps_recent = sum(1 for s in gaps if s in league_window)

    row.update({
        "seasons_played": len(seasons),
        "career_stints": len(d),
        "career_teams": int(d["TEAM_ID"].nunique()),
        "moves_last3": int(moves_last3),
        "gap_seasons": len(gaps),
        "instability_last3": int(moves_last3) + gaps_recent,
        "instability_career": round((sum(moves) + 2 * len(gaps)) / (len(span) + 1), 3),
        "moves_per_season": round(sum(moves) / len(seasons), 3),
        "first_season": seasons[0],
        "last_season": seasons[-1],
    })
    return row


def derive_stints(player_id, df):
    """One career frame -> player_career_stints rows, [] when there is nothing.

    Filters and ordering are IDENTICAL to derive() -- seasons <= config.SEASON,
    TOT aggregates excluded, stable sort by SEASON_ID so within-season API order
    (chronological for traded players) survives. A player with no usable rows
    contributes no rows at all: absent history stays absent (invariant 1), and
    the API 404s rather than serving an empty career that looks like data.
    """
    if df is None or df.empty:
        return []
    d = df.copy()
    d["SEASON_ID"] = d["SEASON_ID"].astype(str)
    d = d[d["SEASON_ID"] <= config.SEASON]                     # nothing post-dates 2023-24
    d = d[d["TEAM_ABBREVIATION"].astype(str) != "TOT"]         # aggregates are not stints
    if d.empty:
        return []
    d = d.sort_values("SEASON_ID", kind="stable")              # keeps within-season order
    return [{"player_id": player_id, "season": s, "seq": i, "team_abbr": str(t)}
            for i, (s, t) in enumerate(zip(d["SEASON_ID"].tolist(),
                                           d["TEAM_ABBREVIATION"].tolist()))]


def population():
    """ALL distinct propped player_ids, Postgres first, data/*.csv if unreachable.

    The FULL population of player_game_scores, not just the shortlist: motive is
    scored for every propped row, so the instability percentile in standardize.py
    must rank over everyone who gets a motive value.

    -> (sorted ids, 'postgres'|'csv'). The CSV path exists so the fetch+derive can
    still run (and cache) on a machine with no server; the WRITE stays Postgres-only.
    """
    try:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT player_id FROM player_game_scores "
                        "ORDER BY player_id")
            return [r[0] for r in cur.fetchall()], "postgres"
    except (SystemExit, Exception):
        f = config.ROOT / "data" / "player_game_scores.csv"
        if not f.exists():
            sys.exit("!! no Postgres and no data/player_game_scores.csv -- nothing to target")
        d = pd.read_csv(f, usecols=["player_id"])
        ids = sorted(d["player_id"].astype(int).unique())
        return ids, "csv"


def write_csv_fallback(rows, stint_rows):
    """Postgres unreachable: land the derived tables in data/ so the CSV backend sees them."""
    import json
    manifest = config.ROOT / "data" / "_types.json"
    types = json.loads(manifest.read_text()) if manifest.exists() else {}
    for table, table_rows, table_types in (
            ("player_career", rows, CSV_TYPES),
            ("player_career_stints", stint_rows, STINTS_CSV_TYPES)):
        out = config.ROOT / "data" / f"{table}.csv"
        pd.DataFrame(table_rows)[list(table_types)].to_csv(out, index=False)
        types[table] = table_types
        print(f"-> {out}  {len(table_rows)} rows  [Postgres was unreachable]")
    manifest.write_text(json.dumps(types, indent=1))
    print("   (+ _types.json updated)")


def main(dry=True):
    ids, source = population()
    cached_n = sum(1 for p in ids if cache.status("nba", cache.nba_key("career", p)))
    print(f"propped players    : {len(ids)}  (from {source})")
    print(f"careers cached     : {cached_n}  <- free")
    print(f"==> calls to make  : {len(ids) - cached_n}  (nba_api is FREE, "
          f"~{INTERVAL}s apart)")
    print("writes             : player_career (1 row/player) + "
          "player_career_stints (1 row/season-stint, for the case view's career strip)")

    if dry:
        db.dry_notice()
        return

    conn_ctx = None
    try:
        conn_ctx = db.connect()
        conn = conn_ctx.__enter__()
    except (SystemExit, Exception) as e:
        conn = None
        print(f"!! Postgres unreachable ({type(e).__name__}) -- fetching anyway, "
              f"will fall back to data/player_career.csv")

    try:
        rows, stint_rows = [], []
        for p in ids:                        # ONE fetch per player feeds both tables
            df = fetch_player(p, conn, allow_fetch=True)
            rows.append(derive(p, df))
            stint_rows.extend(derive_stints(p, df))
        empty = sum(1 for r in rows if r["seasons_played"] is None)
        print(f"\nderived {len(rows)} rows ({empty} empty -> NULLs, per invariant 1)"
              f" + {len(stint_rows)} season-stints")

        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(DDL)
                for col, typ in ALTER.items():
                    cur.execute(f"ALTER TABLE player_career "
                                f"ADD COLUMN IF NOT EXISTS {col} {typ}")
                cur.execute(STINTS_DDL)
            n = db.upsert(conn, "player_career", rows, conflict=["player_id"])
            print(f"upserted {n} rows -> player_career "
                  f"({db.count(conn, 'player_career')} total)")
            ns = db.upsert(conn, "player_career_stints", stint_rows,
                           conflict=["player_id", "seq"])
            print(f"upserted {ns} rows -> player_career_stints "
                  f"({db.count(conn, 'player_career_stints')} total)")
        else:
            write_csv_fallback(rows, stint_rows)
        print(cache.summary())
    finally:
        if conn_ctx is not None:
            conn_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main(dry=db.is_dry())
