# Architecture — what every file and key function does

[README.md](README.md) says how to *run* this. [METHODOLOGY.md](METHODOLOGY.md) says why the
statistics are what they are. **This file is for the person who has to change something.**

It is organised the way the data moves: shared plumbing, then the pipeline layer by layer,
then the API, then the dashboard. The last two sections — [Invariants](#invariants-dont-break-these)
and [Recipes](#recipes) — are the ones to read first if you are here to make an edit.

## The layout

The backend is one package, `pipeline/`, with one subpackage per layer. Nothing loose at the root.

```
game-integrity-v1/
├── pipeline/                     THE BACKEND
│   ├── core/               config · db · cache · schema.sql
│   ├── ingest/     L0-L4   the 13 loaders
│   ├── score/      L5-L6   standardize · export_candidates · residualize
│   ├── explain/    L8-L9   optional · packet · summarize · LLM review
│   └── tools/              to_duckdb · to_csv
├── server/app.py   L7      read-only FastAPI
├── frontend/               React dashboard
├── tests/                  experiments + figures (not a test suite)
├── Makefile                every command, Python and Node alike
└── game_integrity.duckdb   the shipped database — no server needed
```

**Everything runs as a module, from the repository root:**

```bash
python -m pipeline.load_data.load_props           # DRY: reports, writes nothing
python -m pipeline.load_data.load_props run       # actually executes
python -m pipeline.score.standardize run
python -m pipeline.load_data.load_props run --date 2024-01-15    # one game day
```
---

## THE CORE FILES 

Three files that everything else imports. Changing them changes all 13 loaders at once.

### `pipeline/core/config.py` — 102 lines, no logic

Constants and secrets, nothing else. `_load_env()` reads `.env` into `os.environ` without
a dependency on python-dotenv.

| name | is | change it when |
|---|---|---|
| `SEASON`, `SEASON_TYPE` | `"2023-24"`, `"regular"` | running another season |
| `MARKET` | `"player_points"` | adding rebounds/assists props |
| `REGIONS` / `BOOK` | `"us"` / `"fanduel"` | `REGIONS` decides what is *fetched* (all ~10 US books, same price); `BOOK` decides which one is *scored* |
| `DATABASE_URL` | `postgresql:///game_integrity_v1` | pointing at another Postgres |
| `GI_DB` + `BACKEND` | unset → `"postgres"` | `GI_DB=game_integrity.duckdb` → the file; `GI_DB=data` → the CSV export |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_PRICES` | Gemini by default | only `pipeline/llm_review/review.py` reads these |

`BACKEND` is derived from the SHAPE of `GI_DB`, not configured separately — a `.duckdb`
suffix means the file, a directory means the CSV export, unset means Postgres. One switch,
so there is no second one to forget.

### `pipeline/core/db.py` — the only file that talks to a database

**READ path (works on both backends):**

- **`rows(sql, params, one=False)`** — the one function the API/frontend calls, on all
  three backends. For DuckDB and CSV it translates the dialect via `_to_duck_sql()`
  (`%(name)s` → `$name`, `%s` → `?`) and subsets the params dict, because DuckDB raises
  on unused keys where psycopg ignores them. Returns a list of dicts, or one dict /
  `None` when `one=True`.
- **`_csv_connect()`** — builds the CSV backend once per process: an in-memory DuckDB
  loaded from `data/*.csv` using the types in `_types.json`. **Loaded as tables, not
  views** — a view re-parses 422,884 pbp rows on every shot-chart request. Costs ~2s on
  the first query and nothing after. It refuses to run without the manifest, because
  inferring types reads `game_id` `'0022300016'` as an integer and breaks every join.
- **`connect(autocommit=False)`** — psycopg context manager, commits on clean exit. If
  Postgres is unreachable *and* the DuckDB file exists, it exits with the `GI_DB=` hint
  rather than a raw socket error.

**WRITE path (Postgres only — this is the reason pipelines don't run on DuckDB):**

- **`upsert(conn, table, rows, conflict, update=None)`** — bulk
  `INSERT … ON CONFLICT DO UPDATE`. **This is how every loader writes**, and it is why
  re-running a loader fills gaps instead of duplicating. Pass `update=[]` to make existing
  rows immutable. Raises if `rows` is ragged, naming the offending row.
- **`update(conn, table, rows, key)`** — plain `UPDATE`. Use this, *not* `upsert`, when
  your rows carry only a subset of columns: `ON CONFLICT DO UPDATE` builds the proposed
  INSERT tuple and checks `NOT NULL` on it *before* resolving the conflict, so an upsert
  that omits a NOT NULL column fails even when the row already exists.


### `pipeline/core/cache.py` — the only file that touches the network

- **`get(kind, key, fetcher, *, api, endpoint, conn=…)`** — Checks existing cache and writes new API calls to cache →
  v0 cache → v1 cache → network, in that order. Returns `(payload, source)` where source is
  `mem|v0|v1|network`. **Writes every response back, including empty ones** (negative
  caching), so a re-run costs nothing.
  If no cache is found, then it reads from fetcher() function, which is passed in. **this is the actual API call** (often _call function in the loader file) that is cached as well!

---

## L0–L1 · Who and when *(cheap, just run once)*

| file | key functions | writes |
|---|---|---|
| **`pipeline/load_data/load_teams.py`** | `build()` | `teams` — 30 rows, **zero network calls** (`nba_api` ships a static table) |
| **`pipeline/load_data/load_games.py`** | `_call()` → `fetch()` → `build(df)` → `report()` | `games` — the season spine, 1 API call for all 1,230 |
| **`pipeline/load_data/load_players.py`** | `ascii_name` / `depunct_name` / `desuffix_name`, `make_resolver(conn)`, `MANUAL_ALIASES` | `players` |
| **`pipeline/load_data/load_rosters.py`** | `fetch_team(team_id)`, `_height_in`, `_exp`, `build(frames)` | *enriches* `players` with position, height, weight, birth date, **experience** |
| **`pipeline/load_data/load_salaries.py`** | `fetch_team`, `parse(page)`, `_money`, `team_rosters(conn)`, `build(...)` | `player_salaries` — 30 free scrapes at 3.5s intervals |

---

## L2–L3 · The market *(this is where money is spent)*

| file | key functions | writes |
|---|---|---|
| **`pipeline/load_data/load_events.py`** | `seed_pool()`, `by_date(pool)`, `match(idx, date, matchup)`, `resolve()`, `collisions()` | `odds_events` — game_id ↔ OddsAPI event_id, 1 credit each |
| **`pipeline/load_data/load_props.py`** | `snapshot(event_id, iso)`, `quotes(payload)`, `rows_for(...)` | `prop_quotes` at `close` and `open` |
| **`pipeline/load_data/load_line_history.py`** | `targets(conn, group)`, `rows_for(...)`, **`verify(conn)`** | `prop_quotes` with `snapshot_role='poll'` |

**`ROLE_OFFSET = {"close": 0h, "open": 12h}`** in `pipeline/load_data/load_props.py` is the whole definition of
open vs close: the same endpoint, called at two times before tip.

**`pipeline/load_data/load_line_history.py` is display-first, with one scored use.** It calls the same endpoint at a
*ladder* of times (`PLANS`) for the top 150 games so a case page can draw a movement curve
instead of two endpoints. Its rows are tagged `snapshot_role='poll'`, and `pipeline/score/standardize.py`
selects `'open'`/`'close'` by name — plus the *earliest* poll as a fallback effective open
for player-events with no T-12 `'open'` row (see invariant 2). Rows with a true open are
untouched by polls.

> **Cost model:** one OddsAPI call = **10 credits** and returns *all* players and *all* ~10
> US books for one event. So per-book and per-player granularity is free; **per-timestamp is
> what costs money.**

---

## L4–L5 · What happened on the floor

| file | key functions | writes |
|---|---|---|
| **`pipeline/load_data/load_boxscores.py`** | `fetch_one(kind, game_id)`, `preflight()`, `build_game(game_id, frames)`, `new_players(...)` | `player_games` |
| **`pipeline/load_data/load_pbp.py`** | `cmd_fetch()` / `cmd_derive()`, `clock_to_sec_left`, `elapsed_sec`, `find_garbage_start(df)`, `_is_ejection`, `_is_sub`, `derive_game()` | `player_game_pbp`, `game_pbp_context` |
| **`pipeline/load_data/load_pbp_events.py`** | `main()`, `_clean(v)` | `player_game_events` — 422,884 rows, **cache only, 0 API calls** |
| **`pipeline/load_data/load_context.py`** | `build(conn)`, `ALTITUDE_FT` | `game_context` — rest, B2B, altitude, pace, margin. No API calls |

---

## L6–L7 · Score, rank, and display

### `pipeline/score/standardize.py` — the scoring model (621 lines, the file that matters most)

`load(conn)` → `build(d)` → write. `build()` calls four adders in order:

| function | produces | how |
|---|---|---|
| **`add_movement(d)`** | `line_move_pct`, `under_move_pct`, `price_only_move` | open→close as fractional moves. **NaN where no opening observation exists** — not zero. The effective open is the T-12 `'open'`, else the earliest `'poll'` (provenance in `open_source`/`open_offset_hours`) |
| **`add_performance(d)`** | `game_z`, `effort_z`, `shortfall_z` → `performance` | `game_score(d)` is Hollinger Game Score (11 box inputs). `effort_z` averages the nine `EFFORT` stats. Each is z-scored against *his own season* and against his `tier` |
| **`add_market(d)`** | `p_price`, `p_line`, `mk_line_mv`, `mk_price_mv` → `market` | four components, equal weight, all oriented so a **low raw value scores high** |
| **`add_motive(d)`** | `motive` | `0.75·(1 − salary percentile) + 0.25·(instability_career percentile)`, then z-scored. **Two-way contracts (no listed salary) score maximum on the salary term.** Instability (team changes + gap seasons per season of career span, from `player_career`) is NEUTRAL 0.5 where history is missing — absence, not evidence |

The four weight dicts are the model's dials:

```python
PERF_W   = {"game_z": 1/3, "effort_z": 1/3, "shortfall_z": 1/3}
MARKET_W = {"p_price": .25, "p_line": .25, "line_mv": .25, "price_mv": .25}
MOTIVE_W = {"salary": .75, "instability": .25}
BLOCK_W  = {"performance": .45, "market": .30, "motive": .25}
```

See METHODOLOGY.md.

### `pipeline/score/export_candidates.py` → `player_game_scores`

`load(conn)` pulls features + z + pbp + experience, then `main()` applies **seven cuts in
order** and ranks the survivors.

```
15,498  propped player-games
11,595  1  game_z   top 25% removed      (he had a good game)
 9,660  2  effort_z top 25% removed      (he was more involved than usual)
 7,349  3  market   bottom 25% removed   (the market leaned over)
 6,881  4  no upward line move
 5,841  5  no upward price-only move
 4,630  6  salary <= $20M or unlisted
 3,077  7  experience > 2 seasons
```

Three things about this block are load-bearing:

1. **Percentiles are taken on the full population *before* any cut.** Computing them
   sequentially would make cut 2 a quartile of whatever cut 1 left — a different and much
   harsher filter.
2. **`~(x > 0)`, never `x <= 0`.** A row with nothing to test survives. Cuts 4, 5 and 7 all
   use this form; Jontay Porter is one of 12 propped players with no experience value, so
   writing cut 7 the natural way would delete a labelled case.
3. **Rank on raw `score`, display `score_100`.** `score_100` is stored rounded to 2dp, which
   collapses 15,494 distinct values into ~9,700 and invents ties. `shortfall` breaks ties —
   it is the only input no cut range-restricts.

`cut_failed` records the **first** cut a row failed, so the UI can answer "why isn't this
game here?". All 15,498 rows are kept, with `in_shortlist` as a flag.

### `server/app.py` — read-only FastAPI, 1,095 lines

| helper | does |
|---|---|
| **`q(sql, params, one)`** | delegates to `db.rows()`, then scrubs. **All 12 endpoints go through this** — which is why the backend swap needed no endpoint changes |

| endpoint | feeds |
|---|---|
| `GET /api/summary` | header counts, landing page |
| `GET /api/watchlist` | the ranked rail (`q`, `sort`, paging) |
| `GET /api/case/{pid}/{gid}` | the case file |
| `…/shots` · `…/plays` | shot chart, play timeline (from `player_game_events`) |
| `…/line-history` | the movement chart (`BOOKS_LIVE` / `BOOKS_SOON`) |
| `…/report.pdf` | a printable case packet |
| `/api/funnel` | the seven-cut scrollytelling story |
| `/api/calendar` · `/api/cloud` · `/api/players` · `/api/isolation` | season heatmap, 3D scatter, player leaderboard, isolation panel |

Every list endpoint carries an explicit total-order `ORDER BY` (…`, player_id, game_id`).
Without it, Postgres and DuckDB return ties in different orders and pagination drops rows.

### L8–L9 · Explaining one case *(optional, not in the main flow)*

| file | key functions | note |
|---|---|---|
| `pipeline/llm_review/packet.py` | `build(player_id, …, blind=True, pbp_mode=…)`, `_pbp()` | ~3,300-token JSON evidence packet for one player-game. `blind=True` withholds the score |
| `pipeline/llm_review/summarize.py` | `flags(p)`, `summarize(p)`, `_context(g)` | plain-English summary from **rules, not a model**. Free, instant, cannot invent a claim |
| `pipeline/llm_review/review.py` | `review_one(...)`, `_post`, `_parse`, `SCHEMA`, `probe()` | optional LLM reviewer. **Reads and explains; never scores or ranks** |

---

## The frontend

React + TypeScript + Vite. `npm run dev` on **:5173**, proxying `/api` → **:8000**.

### `App.tsx` — 101 lines, all the app-level state

Three pieces of state and nothing else: `selected` (the open player-game), `view`
(`"home" | "ledger"`), `dark`, plus `railOpen`. `summary` is fetched once here and shared.

**`openCase(s)` is the single funnel** — rail row, calendar day, 3D node, and search all land
there. The ledger layout is three columns: `RankedList` → `CaseView` → `Methodology`.

### `api.ts` — 390 lines, the contract with the backend

One fetch helper, ~13 typed endpoint functions (`fetchWatchlist`, `fetchCase`,
`fetchLineHistory`, `fetchShots`, `fetchPlays`, `fetchSummary`, `fetchFunnel`,
`fetchCalendar`, `fetchCloud`, `fetchPlayers`, `fetchIsolation`, `fetchPlayerFlags`) and the
row interfaces. **If you change a payload in `server/app.py`, the matching interface here is
the only other place to edit.**

### `severity.ts` — 56 lines, the shared thresholds

`RED_SCORE = 73.7`, `scoreLevel(score)`, `fmtScore()`, and the `CONFIRMED` set with
`isConfirmed(playerId, gameId)`. Any component that colours by severity imports from here so
the thresholds cannot drift apart.

### Components

| file | lines | what it is |
|---|---|---|
| **`CaseView.tsx`** | 659 | The case file. `ScoreRing`, `ScoreCard`, `ZBar`, `EqTable` (the performance equation), `SeasonStrip`, `takeaway(c)` — the generated one-paragraph verdict — and `badness(z, flip)`, the shared severity mapping |
| **`RankedList.tsx`** | 375 | The rail. `parseTerm` / `liveTerms` / `hay()` implement the search grammar; `HAY` is a `WeakMap` search-string cache. `LOAD = 5000` fetches the whole shortlist once, `RENDER_CAP = 150` limits DOM nodes |
| **`LineHistory.tsx`** | 297 | Line + price movement, **two stacked panels on one time axis — deliberately not dual-axis.** `extent()` returns a `flat` flag so a never-moving line doesn't print fake tick values; `timeTicks()` adapts step to window; width is measured via `ResizeObserver` |
| **`FunnelStory.tsx`** | 263 | Scroll-driven seven-cut ribbon. `bandH()`, `NARRATION`, `PAN_END = 0.78` |
| **`Cloud3D.tsx`** | 222 | react-three-fiber scatter on `performance × market × motive`. `normalize()`, `Axes`, `Points` |
| **`PlayerSeason.tsx`** | 108 | One player's season strip |
| **`Landing.tsx`** | 105 | Home view — headline numbers and entry points |
| **`ShotChart.tsx`** | 98 | Half-court shot locations. `ARC` is the 3-point line, 89 sampled points |
| **`SeasonCalendar.tsx`** | 95 | Season heatmap. `level(d)` buckets a day's worst score |
| **`PlayTimeline.tsx`** | 56 | Ordered play events with NBA video links |
| **`Methodology.tsx`** | 44 | The third column — method notes, live-linked to the open case |
| **`Logoman.tsx`** | 13 | Inline SVG wordmark |

`tokens.css` holds the design tokens; `app.css` the layout. Both themes are token swaps.

---

## Invariants — don't break these

Each of these was a bug once.

1. **Missing ≠ failed.** Cuts use `~(x > 0)`. A row with nothing to test survives. Two-way
   contracts have no salary and are exactly the population the motive axis exists to find.
2. **Poll rows reach the score ONLY as a fallback opening observation where no 'open'
   exists; display-only otherwise.** `pipeline/score/standardize.py` selects
   `snapshot_role` by name and uses the earliest `'poll'` snapshot as the effective open
   only for player-events with no T-12 `'open'` row (provenance in
   `player_game_features.open_source` / `open_offset_hours`). Rows that have a true
   `'open'` must be byte-identical whatever polls exist — no poll predates T-12, and
   `verify()` proves every `'open'` is used verbatim. Anything you add to `prop_quotes`
   for display still gets a new role.
3. **Percentiles come before cuts, never between them.**
4. **Rank on raw `score`; display `score_100`.**
5. **Every list query needs a total-order `ORDER BY`,** or Postgres and DuckDB disagree.
6. **The write path is Postgres-only.** `db.upsert` is psycopg. DuckDB is a read-only export.
7. **Loaders are dry by default.** `python -m pipeline.load_data.load_props` is safe; `run` spends money.
8. **Cache keys must be deterministic,** or `cache.get()` raises rather than double-paying.
9. **`raw` and `z` tables stay separate.** Re-standardising must not rewrite its own inputs.

## Known rough edges

Real, documented, and not fixed — so nobody rediscovers them as surprises:

- **`score_100` is a linear min-max rescale, not a percentile.** The stated block weights are
  45/30/25 but the *effective* weights are **42.3 / 25.0 / 32.7**, because the three blocks
  have unequal spread (sd 0.718 / 0.636 / 1.000). A comment in `server/app.py:_decorate`
  still calls it a percentile — the `scale_note` the API returns is correct.
- **No injury data.** `dnp_reason` is blank for every played game, so a hamstring looks
  identical to a withdrawal. This is the single largest source of false positives.
- **Three label sets exist** — `FLAG` in [tests/experiments/weight_audit.py:70](tests/experiments/weight_audit.py#L70),
  `CONFIRMED` in [severity.ts:32](frontend/src/severity.ts#L32), and the set used in
  presentations. They disagree. Reconcile before quoting a hit rate.
- **Two of six canonical labels are OVERS** (Porter 2024‑01‑20, Beasley 2024‑03‑10, both
  `shortfall = 0.000`) and are structurally undetectable by a one-sided pipeline.
- **`computed_at` never refreshes on upsert.**
- **A closing line is required,** excluding 41% of played games — including Porter's
  2024‑01‑26, because no book ever posted a line on him that night.

## Recipes

**Change a model weight** → `pipeline/score/standardize.py` (`PERF_W` / `MARKET_W` / `BLOCK_W`), then
`python -m pipeline.score.standardize run && python -m pipeline.score.export_candidates run`. Nothing upstream re-runs.

**Add or change a cut** → the `cuts` list at [export_candidates.py:217](pipeline/score/export_candidates.py#L217). Use the `~(x > …)`
form. Re-run `python -m pipeline.score.export_candidates run` only.

**Add a stat to the effort block** → add the column in `pipeline/load_data/load_boxscores.py`'s `TRACK_FIELDS`,
add it to `EFFORT` in `pipeline/score/standardize.py`, add it to `FEATURE_COLS`. Re-run L4 → L5 → L6.

**Add an API endpoint** → write it with `q()` (never raw psycopg) so it works on both
backends, give it a total-order `ORDER BY`, then add the typed function and interface in
`api.ts`.

**Add a season** → set `config.SEASON`, then run L0 → L6 in order. Every layer below L2 is
driven by what's in the tables, not by the constant, so scope flags work throughout.

**Rehearse cheaply before spending** → `--date 2024-01-15` on any paid loader runs the
identical pipeline over ~10 events.

**Regenerate the handoff artifacts** → `make duckdb`, `make dump`, `make csv` — a second or
two each. `make csv` works with no Postgres; it falls back to the committed `.duckdb`.
