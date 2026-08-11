"""Export the scored database to a single DuckDB file, for handing the demo off.

WHY. Postgres stays the source of truth here -- every loader writes to it and nothing
about this script changes that. But a recipient who only wants to RUN the dashboard
should not have to install a database server, create a role, match a major version and
restore a dump. DuckDB is a single file: copy it next to the code and the API reads it.

    postgres  146 MB, 21 tables, needs a server
    duckdb     ~9 MB, 13 tables, needs a file

The size drop is DuckDB's columnar storage, not a loss of rows.

WHAT IS LEFT BEHIND, and why each one:

    player_game_residuals   L5c was built and rejected -- context explains 1-23% of
    resid_onestage          variance and corr(score, score_residualized) = 0.995. The
    residual_models         one column still used, the role `tier`, is already carried
    residual_coefs          on player_game_features, so nothing downstream notices.
    api_calls               the spend ledger. Real cost history, but it is a record of
                            THIS machine's billing and means nothing on another.
    player_aliases          name-resolution scratch, only read during ingest.

COPY FROM DATABASE (the one-liner in DuckDB's docs) FAILS on this database: api_calls
carries an index type DuckDB cannot bind, and the whole copy aborts. Copying table by
table avoids that, and lets the skip list above exist at all.

    python to_duckdb.py            # DRY: what it would copy, writes nothing
    python to_duckdb.py run        # write game_integrity.duckdb
    python to_duckdb.py run --all  # include the skipped tables too
"""
import pathlib
import sys
import time

import config

OUT = pathlib.Path("game_integrity.duckdb")

# Everything the API, packet.py and summarize.py actually read.
KEEP = [
    "player_game_scores",      # the shortlist: rank, in_shortlist, cut_failed
    "player_game_features",    # raw box score, lines, prices, movement
    "player_game_z",           # standardised components and the three blocks
    "player_games",            # every player-game, for season strips and baselines
    "player_baselines",        # his own season, precomputed
    "player_salaries",         # motive, and the salary shown on the case page
    "players", "teams", "games",
    "odds_events",
    "prop_quotes",             # includes the 'poll' curve rows the case chart draws
    "player_game_pbp",         # stints, ejections, garbage time
    # L4c. The shot chart and play timeline read this instead of the pbp cache, so
    # both panels work wherever the database goes -- which is the whole point.
    "player_game_events",
    "game_pbp_context",
    "game_context",            # rest, altitude, pace, margin
    "player_game_review",      # L9 output, empty unless the reviewer has been run
    # L5c was REJECTED as a scoring input, but the case view still displays its
    # per-stat residual z-scores, so the API left-joins it. Dropping it 500s every
    # case page -- the reason this list is checked against the serving layer rather
    # than against which layers "matter".
    "player_game_residuals",
]
SKIP_REASON = {
    "resid_onestage": "L5c control arm, not read by the API",
    "residual_models": "L5c fit diagnostics",
    "residual_coefs": "L5c coefficients",
    "api_calls": "spend ledger -- specific to the machine that paid",
    "player_aliases": "ingest-time name resolution only",
}


def main(mode="dry", keep_all=False):
    try:
        import duckdb
    except ImportError:
        sys.exit("!! pip install duckdb")

    if OUT.exists() and mode == "run":
        print(f"removing existing {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
        OUT.unlink()

    # An in-memory connection for the dry run, so a plan costs nothing on disk.
    con = duckdb.connect(str(OUT) if mode == "run" else ":memory:")
    catalog = OUT.stem if mode == "run" else "memory"
    con.execute("INSTALL postgres; LOAD postgres;")
    # DATABASE_URL is a libpq URI; the postgres extension takes the same string.
    con.execute(f"ATTACH '{config.DATABASE_URL}' AS pg (TYPE POSTGRES, READ_ONLY);")

    found = [r[0] for r in con.execute(
        "SELECT table_name FROM pg.information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY 1").fetchall()]
    wanted = found if keep_all else [t for t in found if t in KEEP]
    skipped = [t for t in found if t not in wanted]

    print(f"source   : {config.DATABASE_URL}")
    print(f"target   : {OUT}  ({'WRITING' if mode == 'run' else 'dry run'})")
    print(f"tables   : {len(wanted)} of {len(found)}\n")

    t0, total = time.time(), 0
    for t in wanted:
        n = con.execute(f'SELECT count(*) FROM pg."{t}"').fetchone()[0]
        total += n
        if mode == "run":
            con.execute(f'CREATE OR REPLACE TABLE "{t}" AS SELECT * FROM pg."{t}"')
            got = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            # A silent short copy would look like a successful export.
            if got != n:
                sys.exit(f"!! {t}: copied {got:,} of {n:,} rows")
        print(f"  {t:<24} {n:>9,}")

    if skipped:
        print("\nskipped:")
        for t in skipped:
            print(f"  {t:<24} {SKIP_REASON.get(t, 'not read by the dashboard')}")

    con.execute("DETACH pg;")
    if mode != "run":
        con.close()
        print(f"\n{total:,} rows would be copied. Re-run with 'run' to write {OUT}.")
        return

    # A missing index is invisible until a case page takes two seconds to open.
    for tbl, cols in (("player_game_scores", "player_id, game_id"),
                      ("player_game_features", "player_id, game_id"),
                      ("player_game_z", "player_id, game_id"),
                      ("prop_quotes", "event_id, player_id"),
                      ("player_games", "player_id, game_id")):
        con.execute(f'CREATE INDEX IF NOT EXISTS {tbl}_lookup ON "{tbl}" ({cols})')
    con.close()

    print(f"\ncopied {total:,} rows in {time.time()-t0:.1f}s")
    print(f"-> {OUT}  {OUT.stat().st_size/1e6:.1f} MB")
    print(f"\nPoint the API at it with:  export GI_DB={OUT}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main("run" if "run" in a else "dry", keep_all="--all" in a)
