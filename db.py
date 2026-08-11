"""Postgres access for game-integrity v1.

SUMMARY: Helps write all tables into database and writes helper functions to load all data
when being fetched from APIs; using driver/adapter psycopg to write Python that is converted to SQL

Two rules every loader follows:

  IDEMPOTENT -- writes go through `upsert`, so re-running a loader fills gaps
  instead of duplicating rows. A run that dies halfway is resumed by re-running it.

  DRY BY DEFAULT -- `is_dry(sys.argv)` is true unless you pass 'run'. Loaders report
  exactly what they would fetch and write, and touch nothing, until you say so.
  Carried over from v0, where it is the reason the paid caches exist at all.
"""
import sys
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

import config


@contextmanager
def connect(autocommit=False):
    """Connection to DATABASE_URL; commits on clean exit, rolls back on exception."""
    try:
        with psycopg.connect(config.DATABASE_URL, autocommit=autocommit) as conn:
            yield conn
    except psycopg.OperationalError as e:
        # The most likely person to hit this is someone who just cloned the repo and
        # has no Postgres at all. The shipped DuckDB file is right there and answers
        # every read the dashboard makes, so say so rather than leaving them with a
        # libpq socket error to interpret.
        duck = config.ROOT / "game_integrity.duckdb"
        if duck.exists() and config.BACKEND == "postgres":
            raise SystemExit(
                f"Could not reach Postgres ({config.DATABASE_URL}).\n\n"
                f"This repo ships a database that needs no server. Run:\n\n"
                f"    GI_DB={duck.name} make demo\n\n"
                f"Set DATABASE_URL only if you actually want the Postgres source of "
                f"truth.\n\noriginal error: {e}") from None
        raise


# --- read path, backend-agnostic -------------------------------------------
#
# The WRITE path above is Postgres-only and stays that way: loaders, upserts and the
# spend ledger all target the source of truth. Only READS need to work against the
# DuckDB export, so only reads are abstracted here.
#
# DuckDB speaks a different parameter dialect, so the SQL the API already contains is
# translated rather than rewritten -- %(name)s -> $name, %s -> ?. Doing it here means
# app.py has one query helper and no idea which engine answered it.

def _to_duck_sql(sql):
    import re
    return re.sub(r"%\(([a-zA-Z_][a-zA-Z0-9_]*)\)s", r"$\1", sql).replace("%s", "?")


def rows(sql, params=None, one=False):
    """Run a read query -> list of dicts (or one dict / None when `one`).

    Works on either backend. Callers pass psycopg-style SQL and a dict (named) or
    sequence (positional) of parameters, exactly as before.
    """
    if config.BACKEND == "duckdb":
        import re

        import duckdb
        duck_sql = _to_duck_sql(sql)
        # psycopg IGNORES dict keys the query does not use; DuckDB raises. The API
        # builds one params dict and reuses it across a COUNT and a paged SELECT, so
        # the count query legitimately never mentions limit/offset. Subsetting here
        # keeps that pattern working rather than making every caller trim its own dict.
        if isinstance(params, dict):
            used = set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", duck_sql))
            params = {k: v for k, v in params.items() if k in used}
        con = duckdb.connect(config.GI_DB, read_only=True)
        try:
            cur = con.execute(duck_sql, params or [])
            cols = [d[0] for d in cur.description]
            out = [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            con.close()
    else:
        from psycopg.rows import dict_row
        with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params or ())
            out = [dict(r) for r in cur.fetchall()]
    return (out[0] if out else None) if one else out


def upsert(conn, table, rows, conflict, update=None):
    """Bulk INSERT ... ON CONFLICT (conflict) DO UPDATE. Returns rows sent.

    `update` limits which columns a conflict overwrites (default: all non-key
    columns). Pass [] to make an existing row immutable -- DO NOTHING.
    """
    if not rows:
        return 0
    cols = list(rows[0])
    # The column list comes from rows[0], so every row must have the same keys.
    # A ragged row set would otherwise take the first row's shape and either raise
    # a bare KeyError deep in executemany or, worse, silently drop columns that
    # only later rows carry. Check once, name the offender.
    if any(r.keys() != rows[0].keys() for r in rows):
        odd = next(r for r in rows if r.keys() != rows[0].keys())
        raise ValueError(
            f"upsert into {table!r}: rows are not homogeneous. "
            f"missing={sorted(set(cols) - set(odd))} extra={sorted(set(odd) - set(cols))}")
    upd = [c for c in (cols if update is None else update) if c not in conflict]
    action = (sql.SQL("DO UPDATE SET ") + sql.SQL(", ").join(
                  sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in upd)
              if upd else sql.SQL("DO NOTHING"))
    stmt = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({ph}) ON CONFLICT ({conf}) {act}").format(
        t=sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        ph=sql.SQL(", ").join([sql.Placeholder()] * len(cols)),
        conf=sql.SQL(", ").join(map(sql.Identifier, conflict)),
        act=action)
    with conn.cursor() as cur:
        cur.executemany(stmt, [[r[c] for c in cols] for r in rows])
    return len(rows)


def update(conn, table, rows, key):
    """Bulk UPDATE of existing rows. Returns rows actually matched.

    Use this, NOT upsert(), when `rows` carries only a SUBSET of the table's
    columns. ON CONFLICT DO UPDATE builds the proposed INSERT tuple and checks
    NOT NULL on it *before* resolving the conflict, so an upsert that omits a
    NOT NULL column fails even when the row already exists and the UPDATE branch
    would have been taken. A plain UPDATE never forms that tuple.

    Rows whose key is absent are silently skipped -- this only ever enriches.
    """
    if not rows:
        return 0
    cols = [c for c in rows[0] if c not in key]
    if not cols:
        return 0
    if any(r.keys() != rows[0].keys() for r in rows):
        raise ValueError(f"update on {table!r}: rows are not homogeneous")
    stmt = sql.SQL("UPDATE {t} SET {sets} WHERE {where}").format(
        t=sql.Identifier(table),
        sets=sql.SQL(", ").join(
            sql.SQL("{c} = {p}").format(c=sql.Identifier(c), p=sql.Placeholder())
            for c in cols),
        where=sql.SQL(" AND ").join(
            sql.SQL("{c} = {p}").format(c=sql.Identifier(c), p=sql.Placeholder())
            for c in key))
    with conn.cursor() as cur:
        cur.executemany(stmt, [[r[c] for c in cols] + [r[c] for c in key] for r in rows])
        return cur.rowcount


def count(conn, table, where=None, params=()):
    """Row count, optionally filtered (`where` is trusted SQL -- callers only)."""
    stmt = sql.SQL("SELECT count(*) FROM {t}").format(t=sql.Identifier(table))
    if where:
        stmt = stmt + sql.SQL(" WHERE ") + sql.SQL(where)
    with conn.cursor() as cur:
        cur.execute(stmt, params)
        return cur.fetchone()[0]


def existing_ids(conn, table, col):
    """Set of values already present in `table.col` -- for planning a DRY run."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT {c} FROM {t}").format(
            c=sql.Identifier(col), t=sql.Identifier(table)))
        return {r[0] for r in cur.fetchall()}


def log_call(conn, *, api, endpoint, params=None, http_status=None, cache_hit,
             cache_source=None, cache_path=None, requests_used=None,
             requests_remaining=None, credits_delta=None, error=None):
    """Record one API call (or cache hit) in the spend ledger.

    OddsAPI credits come from the response's own x-requests-used header rather
    than an assumed billing formula, so cost is measured, not inferred.

    WRITTEN ON ITS OWN AUTOCOMMIT CONNECTION, never on `conn`. A billed call is a
    fact about the world: the credit is gone whether or not the caller's
    transaction goes on to succeed. Logging it inside that transaction meant a
    later failure rolled the ledger row back while the cached response file
    survived on disk -- so the credit was spent, never re-billed, and never
    recorded. `conn` is still accepted so callers need not know any of this.
    """
    with connect(autocommit=True) as ledger:
        return upsert(ledger, "api_calls", [{
            "api": api, "endpoint": endpoint,
            "params": Jsonb(params) if params is not None else None,
            "http_status": http_status, "cache_hit": cache_hit,
            "cache_source": cache_source, "cache_path": cache_path,
            "requests_used": requests_used, "requests_remaining": requests_remaining,
            "credits_delta": credits_delta, "error": error,
        }], conflict=["id"], update=[])


def scope(argv=None):
    """Optional single-day / single-game scope -> (game_date, game_id), None = no filter.

        python load_props.py run --date 2024-01-15     one game day
        python load_props.py run --game 0022300427     one game

    Every layer from L2 down is driven by what is in the tables rather than by a
    season constant, so narrowing the query is all it takes to run the identical
    pipeline over one night instead of 1,230 games. That makes a cheap end-to-end
    rehearsal possible (~10 events) before committing to a full season, and it is
    the same mechanism a nightly live run would use.
    """
    a = sys.argv[1:] if argv is None else argv
    date = game = None
    for i, tok in enumerate(a):
        if tok == "--date" and i + 1 < len(a):
            date = a[i + 1]
        elif tok == "--game" and i + 1 < len(a):
            game = a[i + 1]
    return date, game


def scope_label(date, game):
    return f"date={date}" if date else (f"game={game}" if game else "ALL games")


def is_dry(argv=None):
    """True unless the caller passed 'run'. Every loader defaults to DRY."""
    return "run" not in (sys.argv[1:] if argv is None else argv)


def dry_notice():
    print("\nDRY RUN: nothing fetched, nothing written. Re-run with 'run' to execute.")
