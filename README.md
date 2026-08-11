# Game Integrity — NBA player-prop screening

Finds NBA player-games where a player may have deliberately underperformed against his
points prop, for the 2023-24 regular season.

The output is a **ranked shortlist for human review**, not a verdict. Nothing here
establishes intent; it establishes that a game is unusual on several independent axes at
once. Full statistical reasoning lives in [METHODOLOGY.md](METHODOLOGY.md).

```
32,385 player-games  ->  15,498 with a betting line  ->  3,207 shortlisted  ->  ranked
```

---

## Prerequisites

Installed however your machine prefers (`brew`, `apt`, installer). Nothing below is pinned
by the dependency files — those install *libraries*, not runtimes.

| | version | needed for |
|---|---|---|
| **Python** | 3.11+ (built on 3.13) | pipeline and API |
| **Node** | 20+ (built on 24) | dashboard build |
| PostgreSQL | 14+ (built on 18) | **optional** — only to re-run the pipeline |

**You do not need PostgreSQL to run the demo.** The repo ships a DuckDB file that needs no
server; see "Two database options" below.

Dependencies are declared per language — `requirements.txt` is **pip only** and installs
nothing for the frontend — but you never have to run the two separately:

| file | installs |
|---|---|
| `requirements.txt` | Python packages |
| `frontend/package.json` + `package-lock.json` | Node packages |

## Quick start

Every command lives in the `Makefile`. Run `make` on its own to see them all.

```bash
make setup      # pip install + npm ci, both languages
make demo       # API on :8000 and dashboard on :5173, together
```

Then open the **localhost** server link.

**You do not need API keys for the demo, just for the data pipelines** 
The database ships already scored; keys are only for re-running ingest, which is already done. `cp .env.example .env` if you plan to.

| command | does |
|---|---|
| `make setup` | install everything |
| `make demo` | run API + dashboard together (prefix with `GI_DB=game_integrity.duckdb`) |
| `make api` / `make ui` | run one of them alone |
| `make check` | confirm the database is present and scored |
| `make restore` | load `game_integrity.dump` into a Postgres database |
| `make dump` | export Postgres → `game_integrity.dump` (all 22 tables) |
| `make duckdb` | export Postgres → `game_integrity.duckdb` (dashboard tables) |

**Check it worked:** `GI_DB=game_integrity.duckdb make check` prints `shortlist: 3207`,
and the landing page reports **15,494 player-games**.

## Two database options - Postgres or Duckdb

The API reads either backend — same code, same responses, verified endpoint by endpoint.

**DuckDB — start here.** A single 33 MB file, already in this repo and already scored.
No server, no restore, no version matching. It carries all 17 tables the dashboard reads,
including the 422,884 play-by-play events behind the shot chart:

```bash
GI_DB=game_integrity.duckdb make demo
```

**Postgres (the source of truth).** While the demo can run on duckdb, the data pipelines 
currently load to postgres making it the source of truth and the duckdb a static snapshot
of the postgres, especially when `GI_DB` is unset (meaning postgres by default)

`game_integrity.dump` ships with the repo — 17 MB, **all 22 tables with their constraints
and indexes**, including the four the DuckDB export leaves out (the spend ledger and the
rejected L5c models).

```bash
make restore        # createdb + pg_restore, ~30s
make demo           # GI_DB unset, so this is Postgres
```

Restore under another name with `PGDB=your_name make restore`, or point at a server you
already have with `DATABASE_URL=postgresql:///your_db make demo`.

| | `game_integrity.duckdb` | `game_integrity.dump` |
|---|---|---|
| size | 33 MB | 17 MB |
| needs a server | no | yes |
| tables | 17 — what the dashboard reads | **22 — everything** |
| constraints / indexes | lookup indexes only | **150 constraints, 44 indexes** |
| can re-run the pipeline | no | **yes** |

Both are regenerated from a live Postgres: `make dump` and `make duckdb`, a second or two
each.

---

## How the data flows

Each layer reads what the layer above it wrote. Nothing skips ahead, and every layer is
**idempotent** — re-running fills gaps instead of duplicating.

```
        NBA API            OddsAPI          basketball-reference
           |                  |                      |
   L0/L1 dimensions      L2 events              L1d salary
   teams, games,              |                      |
   players, rosters      L3 prop_quotes  <-- the layer that costs money
           |                  |                      |
           +--------- L4 box scores + play-by-play ---+
                              |
                     L5 standardize.py      the three scoring blocks
                              |
                     L6 export_candidates   cuts, then rank
                              |
                     L7 server/app.py       read-only API
                              |
                        frontend/           React dashboard
```

### L0–L1 · Who and when *(cheap, run once)*

| file | writes | what it does |
|---|---|---|
| `load_teams.py` | `teams` | The 30 NBA teams. Zero network calls — `nba_api` ships a static table. |
| `load_games.py` | `games` | The season's game spine. One API call for all 1,230 games. |
| `load_players.py` | `players` | Every player who has ever played, plus nicknames. Run once a season. |
| `load_rosters.py` | `players` | *Enriches* existing rows: position, height, weight, birth date, experience. |
| `load_salaries.py` | `player_salaries` | Scrapes each team's salary table, resolves names to `player_id`, stores rank and percentile. 30 free calls. |

### L2–L3 · The market *(this is where money is spent)*

| file | writes | what it does |
|---|---|---|
| `load_events.py` | `odds_events` | Maps each game to its OddsAPI `event_id`. 1 credit each. |
| `load_props.py` | `prop_quotes` | Points props from FanDuel, DraftKings and Caesars — line, over price, under price — at tip-off (`close`) and T‑12h (`open`). **10 credits per call.** |
| `load_line_history.py` | `prop_quotes` | The same endpoint at a *ladder* of times before tip, so a game has a movement **curve** rather than two endpoints. Rows are tagged `snapshot_role='poll'` and are **invisible to the score** — display only. |

> Every network call in the project goes through `cache.py`. It checks the cache before
> calling and writes every response back, so a re-run costs nothing. `db.log_call()`
> records the real credit cost from the API's own response headers.

### L4–L5 · What happened on the floor

| file | writes | what it does |
|---|---|---|
| `load_boxscores.py` | `player_games` | Merges four NBA box-score endpoints (traditional, advanced, hustle, tracking) into one row per player-game. |
| `load_pbp.py` | `player_game_pbp`, `game_pbp_context` | Play-by-play: substitution stints, ejections, garbage-time split. Two commands — `fetch` (slow, network) and `derive` (fast, offline). |
| `load_pbp_events.py` | `player_game_events` | The individual pbp events for every propped player-game — 422,884 rows — so the shot chart and play timeline read the **database** rather than a cache file. Cache only, 0 API calls. |
| `load_context.py` | `game_context` | Rest days, back-to-backs, altitude, pace, final margin — the circumstances that legitimately depress production. No API calls. |
| **`standardize.py`** | `player_game_features`, `player_game_z` | **The scoring model.** Turns a player-game into the three blocks below. |

### L6–L7 · Rank and serve

| file | writes | what it does |
|---|---|---|
| `export_candidates.py` | `player_game_scores` | Applies seven cuts, then ranks the survivors. Keeps **all 15,498 rows** with `in_shortlist` and `cut_failed`, so the UI can answer *"why isn't this game here?"* |
| `server/app.py` | — | Read-only FastAPI over the three tables. Postgres is the only source; no CSVs, so it can never serve a stale run. |
| `frontend/` | — | React dashboard: ranked list, case file, season calendar, 3D anomaly cloud. |

### L8–L9 · Explaining a case *(optional)*

| file | what it does |
|---|---|
| `packet.py` | Gathers everything known about one player-game into a single JSON evidence packet (~3,300 tokens). |
| `summarize.py` | Plain-English summary of a packet using **rules, not a model**. Free, instant, and cannot invent a claim the data doesn't support. |
| `review.py` | Optional LLM reviewer (Gemini or Anthropic) over the same packet. It reads and explains; **it never scores or ranks.** |

### Shared plumbing

| file | what it does |
|---|---|
| `config.py` | Paths, season, book, API keys, model prices. Reads `.env`. |
| `db.py` | Database access. Every write is an upsert; every loader is **dry by default** — pass `run` to actually execute. |
| `cache.py` | The one place the project touches the network. |
| `schema.sql` | Every table definition. |

---

## The score

```
score = 0.45 × performance  +  0.30 × market  +  0.25 × motive
```

| block | asks | built from |
|---|---|---|
| **performance** | Was this a bad game *for him*? | `game_z` (Hollinger Game Score vs his own season), `effort_z` (nine involvement stats — touches, passes, distance, deflections…), `shortfall_z` (how far under the line he finished) |
| **market** | Did money move toward the under? | closing under price, line level, line movement, price-only movement |
| **motive** | What did he stand to lose? | `1 − salary percentile`; two-way contracts score maximum |

The blocks are **averaged, not multiplied**, and kept separate because they are nearly
independent (performance ↔ market correlate at just **+0.07**) — which is what makes
"both point the same way" a meaningful question.

Then seven cuts remove the clean end of each axis before ranking:

```
15,498  propped player-games
11,595  1  game_z   top 25% removed   (he had a good game)
 9,660  2  effort_z top 25% removed   (he was more involved than usual)
 7,346  3  market   bottom 25% removed (the market leaned over)
 6,936  4  no upward line move
 6,033  5  no upward price-only move
 4,810  6  salary <= $20M or unlisted
 3,207  7  experience > 2 seasons
```

**Does it work?** Across all 15,498 games, the rate at which the player finished *under*
his line rises monotonically with the score — from **13.6%** in the bottom decile to
**95.9%** in the top. That check uses no labels at all.

---

## The three tables the dashboard reads

Split by *what they mean*, so re-standardising never rewrites the evidence it came from.

| table | holds |
|---|---|
| `player_game_features` | **Raw.** Box score, lines, prices, movement. Nothing standardised. Check this when you distrust a number. |
| `player_game_z` | **Standardised.** The performance and market components, and the three blocks. |
| `player_game_scores` | **The shortlist.** Rank, `in_shortlist`, `cut_failed` — one row per propped player-game. |

---

## Conventions worth knowing

- **Dry by default.** Every loader reports what it *would* fetch and writes nothing until
  you pass `run`. `python load_props.py` is safe; `python load_props.py run` spends money.
- **Missing ≠ failed.** Cuts use `~(x > 0)` rather than `x <= 0`, so a row with nothing to
  test survives instead of being deleted. Two-way contracts have no published salary, and
  they are exactly the population the motive axis exists to find.
- **Caches are not in this repo, and nothing at runtime needs them.** They are large and
  already paid for. Without them the *loaders* would re-fetch, but the API reads only the
  database — including the shot chart and play timeline, which used to open a cache file
  inside the request and so rendered empty for anyone but the machine that ran the ingest.

## Known limitations

- **No injury data.** A player who left with a hamstring looks identical to one who
  withdrew. `dnp_reason` is blank for every played game.
- **One-sided by construction.** The pipeline cannot see over-side manipulation.
- **Points props only.** Rebounds snapshots are cached but unused.
- **A closing line is required**, which excludes 41% of played games — including one of
  the two games Jontay Porter was banned over, because no book ever posted a line on him
  that night.

## Not part of the flow

`residualize.py` (context-adjusted residuals) was built and **rejected** — context explains
only 1–23% of variance and `corr(score, score_residualized) = 0.995`. It is kept because
`standardize.py` still uses one column from it, the role `tier`. `load_line_pulls.py`
detects withdrawn lines and has not been run. `tests/` holds exploratory work whose
conclusions are recorded in METHODOLOGY.md.
