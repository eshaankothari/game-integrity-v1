"""Load the shipped DuckDB file into a Postgres server. The reverse of to_duckdb.py.

WHY. The repo ships `game_integrity.duckdb` so the dashboard runs with no server at all.
But a recipient who already has Postgres -- because they want to re-run the pipeline, or
point their own tools at the data -- should not have to ask for a dump. The file they
already cloned is a complete copy of every table the project serves, and DuckDB's postgres
extension writes as well as reads, so it can seed a server directly.

    duckdb file  ->  17 tables, ~300k rows  ->  their Postgres, in about four seconds

NOTE ON WHAT THIS DOES NOT CARRY. It copies TABLES AND ROWS, not the schema's constraints:
primary keys, foreign keys, CHECKs and defaults live in schema.sql and are not reproduced
by a CREATE TABLE AS. That is fine for reading and for the dashboard, and NOT fine if you
intend to re-run the loaders, whose upserts need the ON CONFLICT targets. Run schema.sql
first if you plan to write:

    psql -d game_integrity_v1 -f schema.sql     # then this script, which fills the tables

The five tables to_duckdb.py leaves out (the rejected L5c models, the spend ledger, the
alias scratch) are not here either -- they are not in the file to begin with.

    python to_postgres.py                       # DRY: what it would write
    python to_postgres.py run                   # write to config.DATABASE_URL
    python to_postgres.py run --db mydb         # ...or to a database you name
"""
import pathlib
import sys
import time

import config

SRC = pathlib.Path("game_integrity.duckdb")


def main(mode="dry", target=None):
    try:
        import duckdb
    except ImportError:
        sys.exit("!! pip install duckdb")
    if not SRC.exists():
        sys.exit(f"!! {SRC} not found -- it ships with the repo; re-clone or run "
                 f"`python to_duckdb.py run` if you have the Postgres source.")

    dsn = f"dbname={target}" if target else config.DATABASE_URL

    # NOT read_only: the flag is per-SESSION in DuckDB, so opening the source read-only
    # would also make the ATTACHed Postgres read-only and every CREATE would fail with
    # "attached in read-only mode". The source is still only ever read FROM.
    con = duckdb.connect(str(SRC))
    con.execute("INSTALL postgres; LOAD postgres;")
    try:
        con.execute(f"ATTACH '{dsn}' AS pg (TYPE POSTGRES);")
    except Exception as e:
        sys.exit(f"!! could not reach Postgres at {dsn}\n   {e}\n\n"
                 f"   Create it first:  createdb {target or 'game_integrity_v1'}")

    tabs = [r[0] for r in con.execute(
        "SELECT table_name FROM duckdb_tables() "
        f"WHERE database_name = '{SRC.stem}' ORDER BY 1").fetchall()]

    print(f"source : {SRC}")
    print(f"target : {dsn}  ({'WRITING' if mode == 'run' else 'dry run'})")
    print(f"tables : {len(tabs)}\n")

    t0, total = time.time(), 0
    for t in tabs:
        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        total += n
        if mode == "run":
            con.execute(f'CREATE OR REPLACE TABLE pg."{t}" AS SELECT * FROM "{t}"')
            got = con.execute(f'SELECT count(*) FROM pg."{t}"').fetchone()[0]
            if got != n:                      # a short copy would look like success
                sys.exit(f"!! {t}: wrote {got:,} of {n:,} rows")
        print(f"  {t:<24} {n:>9,}")

    con.execute("DETACH pg;")
    con.close()

    if mode != "run":
        print(f"\n{total:,} rows would be written. Re-run with 'run'.")
        return
    print(f"\nwrote {total:,} rows in {time.time()-t0:.1f}s")
    print(f"\nPoint the API at it with:  DATABASE_URL='{dsn.replace('dbname=', 'postgresql:///')}' make demo")
    print("Constraints are NOT copied -- run schema.sql first if you intend to re-run "
          "the loaders.")


if __name__ == "__main__":
    a = sys.argv[1:]
    db = next((a[i + 1] for i, t in enumerate(a) if t == "--db" and i + 1 < len(a)), None)
    main("run" if "run" in a else "dry", db)
