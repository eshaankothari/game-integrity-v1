"""Export the tables the dashboard reads to plain CSV, for running the demo with no database.

WHY. `to_duckdb.py` already removes the need for a database SERVER. This removes the need
for a database FILE: the demo runs off a directory of CSVs you can open in Excel, diff in
git, or hand to someone who wants to see the numbers rather than run the code.

    postgres   146 MB   21 tables, needs a server
    duckdb      33 MB   17 tables, needs a file
    csv        ~90 MB   17 files,  needs nothing

THE PIPELINES ARE UNAFFECTED. This is a read-side export only. Loaders still write to
Postgres through db.upsert, which is psycopg and stays that way.

TYPES ARE THE WHOLE PROBLEM, so they are exported alongside the data. CSV carries no
schema, and DuckDB's read_csv_auto would infer `game_id` '0022300016' as the INTEGER
22300016 -- the '002' regular-season prefix gone, every join silently broken. The same
class of bug bit the pbp loader once already (0/1 vs BOOLEAN). So every column's declared
type is written to _types.json and replayed on read. Nothing is inferred.

    python -m pipeline.tools.to_csv            # DRY: what it would write
    python -m pipeline.tools.to_csv run        # write data/
    python -m pipeline.tools.to_csv run --out somewhere_else
"""
import json
import sys
import time

from pipeline.core import config
from pipeline.tools.to_duckdb import KEEP

OUT = config.ROOT / "data"
TYPES = "_types.json"


def _source(con):
    """ATTACH whichever backend holds the data, and return its catalog name.

    Postgres is the source of truth, so it wins when it is reachable. The shipped
    .duckdb file is the fallback, which means a recipient who never had Postgres can
    still produce the CSV export from what they cloned.
    """
    try:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{config.DATABASE_URL}' AS src (TYPE POSTGRES, READ_ONLY);")
        # ATTACH is LAZY -- it returns fine against a database that does not exist and
        # only fails on first use. Probe here so the fallback below actually triggers,
        # rather than surfacing as a confusing error 40 lines later.
        con.execute("SELECT 1 FROM src.information_schema.tables LIMIT 1").fetchall()
        return "src", config.DATABASE_URL
    except Exception as e:
        try:
            con.execute("DETACH src;")
        except Exception:
            pass
        duck = config.ROOT / "game_integrity.duckdb"
        if not duck.exists():
            sys.exit(f"!! no Postgres ({config.DATABASE_URL}) and no game_integrity.duckdb "
                     f"-- nothing to export from.\n   {type(e).__name__}: {e}")
        print(f"(Postgres unreachable, exporting from {duck.name} instead)")
        con.execute(f"ATTACH '{duck}' AS src (READ_ONLY);")
        return "src", str(duck)


def main(mode="dry", out=OUT):
    try:
        import duckdb
    except ImportError:
        sys.exit("!! pip install duckdb")

    con = duckdb.connect(":memory:")
    cat, origin = _source(con)

    # Probe KEEP directly rather than reading information_schema. A catalog attached
    # over the postgres extension and one attached from a .duckdb file expose their
    # metadata differently, and this tool has to work against both -- but the list of
    # tables it wants is fixed either way, so asking each one is simpler and portable.
    wanted, missing = [], []
    for t in sorted(KEEP):
        try:
            con.execute(f'SELECT 1 FROM {cat}."{t}" LIMIT 0')
            wanted.append(t)
        except Exception:
            missing.append(t)

    print(f"source : {origin}")
    print(f"target : {out}  ({'WRITING' if mode == 'run' else 'dry run'})")
    print(f"tables : {len(wanted)} of {len(KEEP)} wanted"
          + (f"   (absent: {', '.join(missing)})" if missing else "") + "\n")

    if mode == "run":
        out.mkdir(parents=True, exist_ok=True)

    t0, total, types = time.time(), 0, {}
    for t in wanted:
        cols = con.execute(f'DESCRIBE {cat}."{t}"').fetchall()
        types[t] = {c[0]: c[1] for c in cols}
        n = con.execute(f'SELECT count(*) FROM {cat}."{t}"').fetchone()[0]
        total += n
        if mode == "run":
            f = out / f"{t}.csv"
            # ORDER BY ALL, so the export is DETERMINISTIC. Neither Postgres nor DuckDB
            # promises row order without it, so two runs -- or a run against each backend
            # -- would produce files that differ everywhere while holding identical data.
            # That would make `git diff` useless, which is one of the reasons to want CSV.
            con.execute(f'COPY (SELECT * FROM {cat}."{t}" ORDER BY ALL) '
                        f"TO '{f}' (HEADER, DELIMITER ',')")
            mb = f.stat().st_size / 1e6
            print(f"  {t:<24} {n:>9,} rows  {mb:>7.1f} MB")
        else:
            print(f"  {t:<24} {n:>9,} rows  {len(cols)} cols")

    if mode != "run":
        print(f"\n{total:,} rows would be written. Re-run with 'run'.")
        return

    # NOT sort_keys: read_csv(columns={...}) matches by POSITION, not by name, so the
    # manifest must preserve DESCRIBE order. Sorting it alphabetically pairs every
    # column with the wrong type -- which fails loudly on the first DATE, but would
    # quietly succeed for any two same-typed neighbours.
    (out / TYPES).write_text(json.dumps(types, indent=1))
    size = sum(f.stat().st_size for f in out.glob("*.csv")) / 1e6
    print(f"\nwrote {len(wanted)} files + {TYPES}, {total:,} rows, {size:.1f} MB "
          f"in {time.time()-t0:.1f}s")
    print(f"\nRun the demo on it with:  GI_DB={out.name} make demo")


if __name__ == "__main__":
    a = sys.argv[1:]
    dest = OUT
    if "--out" in a:
        dest = config.ROOT / a[a.index("--out") + 1]
    main("run" if "run" in a else "dry", dest)
