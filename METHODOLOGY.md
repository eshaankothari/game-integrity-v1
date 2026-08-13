# Methodology

Detecting NBA player-games where a player may have deliberately underperformed against
his points prop, 2023-24 regular season.

The output is a **ranked shortlist for human review**, not a verdict. Nothing here
establishes intent; it establishes that a game is unusual on several independent axes at
once.

---

## 1. Data

| table | rows | what it is |
|---|---:|---|
| `games` | 1,230 | full regular season |
| `player_games` | 32,385 | box + advanced + hustle + tracking, 26,393 with minutes > 0 |
| `prop_quotes` | 58,338 | FanDuel points props, opening and closing snapshots |
| `player_game_features` | 15,498 | propped player-games with line/price movement |
| `player_game_residuals` | 26,401 | context-adjusted stats and role `tier` |
| `player_game_pbp` | 26,709 | stints, ejections, garbage-time split |
| `player_salaries` | 611 | 525 listed, 86 two-way/10-day with no published figure |

### Population, stated precisely

```
played player-games (minutes > 0)      26,393
  with a FanDuel CLOSING line          15,498    <- the population
  with an OPENING line                  8,037
```

**A closing line is required.** `shortfall` divides by it and `p_price` is derived from
it, so a row without one has no `performance` and no `market` block. That excludes 10,895
played games (41%), including **47 player-games whose line was opened and later pulled**.
Pulled lines are a deliberate exclusion, not an oversight — a withdrawn market is a
different question from a market that priced through to tip-off.

**An opening line is not required, and its absence is never a penalty.** 48% of rows have
none. The market block divides by the weight *present*, so those rows are judged on price
and line alone rather than scored as though the movement terms had been observed and
found neutral. Cuts 4 and 5 use `~(x > 0)` rather than `x <= 0` for the same reason: a
NaN comparison is False in pandas, so the direct form would delete every row with nothing
to test.

**Baselines are computed over every game a player played**; the propped filter is applied
last. Filtering first would make each player's "normal" the mean of his propped games —
systematically his higher-profile nights. For Jontay Porter that is 7 games of 26.

Three data-corruption classes were found and fixed during ingest, each of which would
have produced confident nonsense:

- **Rate stats at zero minutes.** A dressed-but-unused player returned `pace = 28,800`
  (the degenerate division limit), which overflowed the column and rolled back a whole
  batch. Rate stats are now nulled when `minutes` is falsy — `0.0` counts, not just
  `None`.
- **Tracking outages.** 2024-03-09 had 89 players with 20+ minutes and `distance = 0`
  across 7 games. Detected per-field by all-zero test, plus a rate-based check for
  *partial* outages (2024-01-27 recorded ~10% of a game).
- **Stale `game_context.abs_margin`** — populated for 678 of 2,460 rows, silently
  capping residualization at 7,321 rows.

---

## 2. The score

```
score      = 0.45 · performance  +  0.30 · market  +  0.25 · motive     (raw z-scale)
score_100  = linear rescale of score onto 0–100 over all propped games
```

**The blocks are combined on the RAW z-scale.** `score_100` is a min–max rescale applied
afterwards, for display only. It is *not* a percentile: 73.7 does not mean "worse than
73.7% of games". The rescale is strictly monotone, so it changes how a number reads and
never which games surface.

### The stated weights are not the effective weights

A weight only means what it says when the things it weights have the same spread, and
these do not:

| block | sd | stated weight | **effective share** |
|---|---:|---:|---:|
| performance | 0.718 | 0.45 | **42.3%** |
| market | 0.636 | 0.30 | **25.0%** |
| motive | 1.000 | 0.25 | **32.7%** |

`motive` is a z-score by construction, so its sd is exactly 1; the other two are *means* of
z-scores, which shrinks their spread. **Motive therefore acts at roughly a third of the
score rather than the quarter it is nominally held to** — and holding it to a quarter was
the entire point of the number, so that the score could not become a salary sort.

This is a known, unfixed discrepancy, stated here rather than quietly corrected. Converting
each block to a percentile before combining would make the shares exactly 45/30/25; it was
tested and **not adopted**, because a percentile is uniform over the population and
therefore flattest exactly at the tail anyone reads — across the top 200 it spanned 1.28
points, so rank 1 printed 100.00 and rank 10 printed 99.94. The linear rescale preserves
the raw score's spacing, and spacing is what a reviewer reads.

The raw `score` is kept alongside and is the number to compare across runs: the rescale's
endpoints are set by the two most extreme games, so every 0–100 value shifts if a season is
added.

Three blocks, deliberately kept separate. `performance` and `market` correlate at just
**+0.072**, and `performance` with `motive` at **−0.014** — they carry genuinely different
information, and collapsing them early would destroy the ability to ask "did *both* point
the same way", which is the interesting question.

Blocks are **averaged, not multiplied**. Multiplying correlated inputs inflates apparent
rarity roughly tenfold on this data, and a product collapses to zero whenever any block
is missing.

### 2.1 `performance` — three components, equal weight

| component | baseline | what it asks |
|---|---|---|
| `game_z` | his own season | was this unlike **him**? |
| `effort_z` | his own season | was he less involved than **he** usually is? |
| `shortfall_z` | league-wide | how far short of the **market's forecast** did he fall? |

`game_z_tier` and `effort_z_tier` — the same quantities standardised against everyone in
the player's role rather than against himself — are still computed and stored on
`player_game_z`, but carry **zero weight**. See "Why two baselines" below for why they were
adopted and then withdrawn.

`game_z` standardises **Hollinger Game Score**:

```
PTS + .4·FGM − .7·FGA − .4·(FTA−FTM) + .7·ORB + .3·DRB + STL + .7·AST + .7·BLK − .4·PF − TOV
```

`effort_z` is the mean of nine separately-standardised involvement stats: `touches`,
`passes`, `usage_pct`, `distance`, `contested_shots`, `deflections`, `loose_balls`,
`box_outs`, `screen_assists`. `fga` is excluded — Game Score already charges −0.7 per
attempt.

#### Why two baselines

They fail in **opposite directions**, so both get a vote:

- **own** is soft for a player with many quiet games. Beasley 2024-01-06 is `game_z`
  −1.28 against himself but −1.68 against starters — 3 points is worse than his own
  season makes it look.
- **tier** is soft in reverse. The bench tier is full of six-minute end-of-rotation
  cameos, so Porter 2024-01-20 reads `game_z` −0.17 own but **+0.27** tier — "above
  average for a bench player", which is true and useless.

Six ways of combining them were tested, and adding both as extra components was **adopted
on a bad measurement and then reverted**. This is worth recording in full, because the
error is the kind that flatters a result:

The supporting per-player split indexed a **sorted** rank list as though it were game
order. It reported Beasley −24.7% and Porter −2.3%, i.e. a clear gain for one player at
almost no cost to the other. Recomputed correctly, the same blend moved Porter's 2024‑03‑20
game from rank **90 to 176 — 46.8% worse**, while Beasley improved 38.6%. The blend was not
a free gain; it was a transfer from the player with two labels to the player with four.

`PERF_W` is back to three equal components. The tier columns remain in the table for
inspection, and the role `tier` itself is still used — for the "as a bench player he
averages…" baselines shown on the case page, and by the tier trims. Neither feeds the
score.

Tier-only was separately tested and also **rejected**, for the same underlying reason: it
scores best overall on the labels, but its gain is reweighting toward the player who
supplies four of six of them, not more signal.

#### Why `shortfall` exists — the floor effect

```
shortfall = clip(1 − points / close_line, 0, 1)
```

Every z-score is bounded below by the player's own mean. A 4-point-per-game player
scoring zero is only **−0.97 sd** from himself; a star scoring zero is **−3.23**. So any
threshold on a z-score is secretly a threshold on how much a player normally scores.

`shortfall` has no floor: a zero-point game is **1.000 for anyone**, because the
denominator is the market's player-specific, game-specific forecast rather than the
player's own variance.

**It is standardised league-wide, not within player.** Standardising an
already-player-relative quantity per-player puts the floor effect straight back. Measured
across all 349 zero-point games, raw `shortfall` is constant at 1.000 while the
within-player z ran from +0.50 to +4.87 and correlated **+0.788** with scoring level:

| player | his mean sf | his sd | shortfall | within-player z |
|---|---:|---:|---:|---:|
| Jontay Porter | 0.456 | 0.453 | 1.000 | 1.199 |
| Malik Beasley | 0.229 | 0.295 | 1.000 | 2.616 |
| Ayo Dosunmu | 0.127 | 0.216 | 1.000 | 4.040 |

All three scored **zero**. This was a live bug, found and fixed.

### 2.2 `market` — four components, equal weight

All oriented so a **low raw value scores high**, since short price / small line /
downward drift is what under-side money produces.

| component | coverage |
|---|---:|
| `p_price` percentile of the closing under price | 100% |
| `p_line` percentile of the line | 100% |
| `line_mv` −`line_move_pct` | 52% |
| `price_mv` −`price_only_move` | 28% |

**Normalised by the weight *present*, not the full total**, so a row with no opening line
is judged on price and line alone rather than penalised for two components nobody could
observe.

`p_price` carries the block — it is the only component that ever passed a test on its
own: de-vigged closing lean was monotonic across five buckets at **z = 3.81** with 100%
coverage. `line_move_pct` tested **backwards** (47.1% under-hit vs a 52.7% baseline), and
`price_only_move`, cross-book divergence and delta-`p_under` were all null.

### 2.3 `motive` — salary

```
motive = z(1 − salary percentile);   unlisted (two-way / 10-day) scores maximum
```

The only axis about what a player **risked** rather than what he did. What is forfeited
by throwing a game spans two orders of magnitude on one roster — roughly $50M for a max
contract against $560K for a two-way.

**Percentile, not raw dollars.** Salary is skewed +1.78 with 67% of players below the
mean, so a z-score spends 92% of its range on the top half of earners: the entire bottom
half — two-way deal through the $4.3M median — is squeezed into a 0.38-wide band. That is
exactly backwards for an axis meant to discriminate among the underpaid. Percentile is
uniform by construction, and `rank(pct=True)` gives the 37 players on the exact $2.02M
minimum one identical value rather than spreading them arbitrarily.

**Unlisted salary scores maximum, not missing.** Two-way and 10-day contracts have no
published figure, and they are the lowest-paid players on any roster. `salary <= X`
evaluates False for them in pandas and NULL in SQL, so the naive form deletes exactly the
population the axis exists to surface. Jontay Porter is one of these rows.

### 2.4 Weights

**Within-block weights are `1/n`** — not humility, a measurement. `tests/experiments/weight_audit.py`:

- **One-at-a-time.** Setting each weight to 0 and to 2×, the cost (mean log₁₀ rank of the
  six flagged games) moved **0.02 – 0.09** for every within-block weight, against **0.58**
  for the motive block weight. Zeroing `shortfall_z` entirely cost 0.027.
- **Random ensemble.** Over 20,000 Dirichlet draws, **34.3% of random block-weight vectors beat
  the previously tuned ones**, whose geometric-mean flagged rank (858) sat almost exactly
  on the random median (897).
- **The fits themselves.** Held-out AUC was 0.661 and 0.600 against a 0.50 baseline, and
  the four behavioural coefficients correlated **−0.607** across the two folds — `game_z`
  swung 17×, `shortfall_z` flipped sign.

**Block weights are a stated prior, not a fit.** They are the only weights that measurably
change the ranking. `motive` is held at 0.25 — below both other blocks — precisely so the
score cannot become a salary sort. Raising it always improves the flagged ranks and means
nothing, because "cheap" is what the two label players have in common.

---

## 3. The funnel

Applied **before** ranking. Seven cuts, 15,498 → **3,207**.

```
start                                        15,498
1  game_z   top 25%  (he had a good game)    11,595   (−3,903)
2  effort_z top 25%  (more involved)          9,660   (−1,935)
3  market   bottom 25% (leaned over)          7,346   (−2,314)
4  no upward line move        (NaN kept)      6,936   (−410)
5  no upward price-only move  (NaN kept)      6,033   (−903)
6  salary <= $20M or unlisted                 4,810   (−1,223)
7  experience > 2 seasons     (NaN kept)      3,207   (−1,603)
```

**Cut 7 removes players in their first three seasons.** It is written
`~(experience <= 2)` rather than `experience > 2`, so the **12 propped players the roster
endpoint returned no experience for are kept**. That branch is load-bearing rather than
defensive: Jontay Porter is one of the twelve, and the written form would delete a labelled
case. He survives on a missing value, not on merit — 2023‑24 was effectively his first NBA
season, so backfilling `experience` would make this cut remove him.

Measured like the others: **0 order inversions**, and the shortlist's top 500 goes from
100% under-hits to 99.6% with average points 1.66 → 2.13 — the top 500 now reaches deeper
into a smaller pool. That is the trade, stated rather than buried.

### Why trims, not half-plane gates

The previous funnel used `game_z < 0 AND effort_z < 0` plus `market > 0`. That deleted
10,518 rows and — the real problem — **79 rows from the score's own top 500**, of which
**52 sat within 0.25 of the boundary**. A game at `effort_z = +0.05` was deleted while one
at `−0.05` survived, and no strength on the other axes could compensate.

It was also double-counting: `game_z`, `effort_z` and `market` are all inputs to the
score, so the funnel judged them twice — once with a hard edge, once smoothly. The old cut
pool and the plain top-N-by-score overlapped only **57%**, and the plain ranking was the
*cleaner* pool (94.4% under-hits against 84.5%).

Trimming only the clean quartile fixes both. Verified: **0 inversions** — the trims remove
rows without reordering anything.

### Orientation, which is easy to get backwards

`game_z` and `effort_z` are raw z-scores, so **high** is a good night and the top quartile
goes. `market` is already oriented so **high** means more under-lean — more suspicious —
so it is the **bottom** quartile that goes.

### `~(x > 0)`, not `x <= 0`

A NaN comparison is False in pandas, so the direct form deletes every row whose opening
line was never posted — rows with **nothing to test**, which is not the same as rows that
**failed** the test.

### What is deliberately *not* gated

- **Ejections.** 56 rows, one in the score's top 100. An ejection is a *mechanism* as well
  as an innocent explanation — two quick technicals guarantee a low total. `ejected` and
  `ejected_alone` are carried as columns to filter on. 52 of 72 ejections are solo, which
  is the interesting shape.
- **Low minutes.** The strongest benign explanation *and* a plausible signature of
  withdrawal. Nothing in the data separates them, so it stays evidence rather than a
  nuisance parameter.
- **Salary as the sole motive mechanism.** Swept as a gate over seven thresholds and it
  lost at every one: as a share of the pool it ranked, the flagged games went from **10.9%
  ungated to 15.9%** at a 20th-percentile gate. A binary gate destroys every distinction
  *within* the eligible group and only ever removes rows sitting above the targets. The
  $20M gate is set loose (86th percentile) to remove only the implausible population; the
  ordering is left to `motive` in the score.

---

## 4. Ranking

Primary key `score`, ties broken by `shortfall`.

The cuts already selected on performance, market and salary, so those inputs are
range-restricted among survivors — they retain 73%, 66% and 71% of their full-population
spread. `shortfall` retains **120%**, which is the argument for it as the tiebreak: it is
the one quantity no cut touches.

---

## 5. What was tested and rejected

Recorded because the negative results constrain what the pipeline is allowed to claim.

**Supervised learning on the flagged games.** Six positives from two players. Leave-one-
**player**-out is the only honest split — a random split leaks identity through every
player-level feature. Held-out AUC 0.661 / 0.600. With salary included, test AUC **0.087**:
the model learned "earns $2.0M with 79 games", i.e. Malik Beasley. `motive`'s coefficient
ran **3.9× and 7.0×** the sum of every other coefficient.

**Isolation Forest**, four configurations — pooled with motive, pooled without, per line
band, and on the cut survivors. It lost to a weighted sum every time, for a structural
reason: it scores **rarity**, and the target class here is a **mode**. 349 zero-point games
form a dense cluster, so the criterion systematically discounts the region where the signal
lives. It also inverts `motive` — being underpaid is not rare among games that fail badly,
so a *well-paid* player having such a night is what reads as isolated (Batum at $11.7M
ranked 1st).

**Peer-matched z** (role × minutes band). Works as designed —
`corr(game_z, minutes)` collapses from +0.337 to +0.052 — but it conditions away the
withdrawal itself. Porter 03-20 goes from rank 206 to 541 because, among bench players
under 8 minutes, scoring zero is unremarkable.

**Context residualization.** `corr(score, score_residualized) = 0.995`. Almost nothing
changes.

**Market efficiency.** The closing line is well-calibrated to ~1 percentage point. There
is no betting edge here; price is expectation, not anomaly.

**Effort-stat reliability.** `corr(ICC, |corr with performance|) = +0.829` — reliability
and independence are in direct conflict. Only `contested_shots` passes both.

---

## 6. Known limitations

- **Injury detection is absent.** `dnp_reason` is blank for every played game and
  `returned_after_last_exit` is 100% NULL. A player who left with a hamstring looks
  identical to one who withdrew.
- **One-sided by construction.** The pipeline cannot see over-side manipulation.
- **Points props only.** 447 rebounds-market snapshots are cached and unused.
- **`p_line` is partly an identity proxy** — a two-way draws 5.5 and a star draws 28.5, so
  it encodes player quality as much as market lean.
- **`close_line` as a level may capture mean reversion**, not integrity: books post a high
  line after a hot streak and the player regresses.
- **Six labelled games from two players.** Every number in section 2.4 and section 5 rests
  on that. It is enough to falsify a weighting scheme; it is not enough to fit one.

---

## 7. Where the code lives

```
load_pbp_events.py     L4c  -> player_game_events      one row per pbp event, so the
                            shot chart and timeline need no cache file at request time
standardize.py         L5   -> player_game_features   raw: box score, lines, movement
                            -> player_game_z          standardised: components + blocks
export_candidates.py   L6   -> player_game_scores     the frontend table
                            -> out/candidates.csv     the same shortlist, as a file
server/app.py          L7   read-only FastAPI over those three tables
tests/experiments/          everything else, verdicts in tests/experiments/README.md
tests/analysis/             figures.py and exploratory.py regenerate every number
                            and figure in the write-up from the database
```

L7 queries Postgres only — no CSV reads, so the dashboard can never serve a stale run —
and imports the weights from `pipeline/score/standardize.py` rather than copying them, so a re-weighting
re-labels the UI on restart. Filtering, sorting and pagination are done in SQL.

**Three tables, one row per player-game each, split by what they mean.** Raw values are
wrong only if the ingest was wrong; a z is wrong if the *baseline population* changed,
which happens every time a game is added. Keeping them apart means re-standardising never
rewrites the evidence it was computed from.

`player_game_scores` carries **all 15,498 propped games**, not just the 3,207 survivors.
`rank` is NULL for eliminated rows — they have no position in the shortlist, which is
different from being last in it — and `cut_failed` records the first cut that removed
them, so the UI can answer "why isn't this game here?" from the same query:

```sql
SELECT rank, player, game_date, points, close_line, shortfall,
       score, performance, market, motive
  FROM player_game_scores
 WHERE in_shortlist
 ORDER BY rank;
```

```
cut_failed breakdown            n
(passed all)                 3,207
1  game_z   top 25%          3,903
3  market   bottom 25%       2,314
2  effort_z top 25%          1,935
7  experience > 2 seasons    1,603
6  salary <= $20M            1,223
5  no upward price-only move   903
4  no upward line move         410
```

L5 owns the weights. `tests/experiments/weight_audit.py` reads them out of `pipeline/score/standardize.py` by
AST rather than copying them, so the audit cannot silently disagree with the pipeline it
is auditing.

Run order:

```
python -m pipeline.score.standardize run && python -m pipeline.score.export_candidates
```

**Verification L6 prints on every run:** `order inversions vs the ungated ranking: 0`.
The trims must remove rows without reordering them. A non-zero count means a cut is doing
ranking work it is not supposed to be doing.
