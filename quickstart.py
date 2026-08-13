"""A five-minute tour of the helpers. Costs nothing, writes nothing, fetches nothing.

    python quickstart.py                              # Postgres
    GI_DB=game_integrity.duckdb python quickstart.py  # the shipped file

Everything here uses the same three helpers the real pipeline uses, so whatever you
learn from this transfers directly to writing a loader.
"""
from pipeline.core import cache, config, db

print(f"backend: {config.BACKEND}     root: {config.ROOT.name}\n")


# --- 1. db.rows() -- the ONE read helper --------------------------------------
#
# Write psycopg-style SQL (%(name)s placeholders). db.rows translates it for DuckDB,
# so this identical call works against either backend. Returns a list of dicts.

top = db.rows("""
    SELECT rank, player, game_date, points, close_line, score_100
      FROM player_game_scores
     WHERE in_shortlist AND score_100 > %(floor)s
     ORDER BY rank
     LIMIT 5
""", {"floor": 70})

print("TOP 5 OF THE SHORTLIST")
for r in top:
    print(f"  #{r['rank']:<4} {r['player']:<18} {r['game_date']}  "
          f"{r['points']:>2} pts on a {r['close_line']} line   score {r['score_100']}")


# --- 2. one row instead of a list ---------------------------------------------
#
# `one=True` returns a single dict (or None). Same helper, no unpacking.

n = db.rows("SELECT count(*) AS n FROM player_game_scores WHERE in_shortlist",
            one=True)["n"]
print(f"\n{n:,} games survived all seven cuts.")


# --- 3. cache.status() -- ask what is on disk WITHOUT fetching -----------------
#
# This is the primitive every DRY run is built on: a loader counts what it is
# missing, prints the credit cost, and only then asks whether to spend it.
# 'v0' = paid for in the old project, 'v1' = fetched by this one, None = not cached.

game_id = "0022300999"                      # Jontay Porter, 2024-03-20
key = cache.nba_key("box", game_id)         # -> 'box_0022300999.csv'
print(f"\ncache.status('nba', {key!r}) -> {cache.status('nba', key)!r}")

# read_cached() NEVER touches the network. If it is not on disk you get None.
box = cache.read_cached("nba", key, fmt="csv")
print(f"  that cached response holds {len(box)} rows, {len(box.columns)} columns")


# --- 4. how a loader would spend, if it had to --------------------------------
#
# `get()` calls the fetcher ONLY on a miss. Here everything is cached, so the
# fetcher below never runs -- which is the whole point.

def never_called():
    raise AssertionError("we should not be here: the response is already cached")

payload, source = cache.get("nba", key, never_called, api="nba", endpoint="boxscore",
                            fmt="csv")
# Expect 'mem', not 'v0': step 3 already pulled this key off disk, and cache.get
# checks the in-process memo BEFORE the filesystem. Three layers, cheapest first --
# memo, then disk, then the wire.
print(f"  cache.get(...) served it from {source!r} -- 0 credits, fetcher not called")


# --- 5. explain one game, in English, with no model ---------------------------
#
# packet.build() gathers the evidence; summarize() turns it into four sentences
# using RULES, so it cannot invent a claim the data does not support.
#
# NOTE: packet.py goes through db.connect(), which is psycopg-only -- so this
# section needs Postgres. It is the clearest example of where the read/write
# split actually bites.

if config.BACKEND == "postgres":
    from pipeline.llm_review import packet, summarize

    p = packet.build(1629007, game_id=game_id, blind=True, pbp_mode="none")
    print(f"\nCASE: {p['identity']['player']} · {p['identity']['game_date']}")
    print("  " + summarize.summarize(p).replace("\n", "\n  "))
    print(f"\n  innocent explanations the data supports: {summarize.flags(p) or 'none'}")
else:
    print("\n(skipping the case summary: packet.py needs Postgres)")
