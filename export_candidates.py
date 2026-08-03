"""L6: trim the clean end of each axis, then rank by score -> `out/candidates.csv`.

    1  game_z   top 25%      he had a GOOD game
    2  effort_z top 25%      he was MORE involved than usual
    3  market   bottom 25%   the market leaned OVER, against the thesis
    4  no upward line move            (NaN kept)
    5  no upward price-only move      (NaN kept)
    6  salary <= $20M or unlisted
    -> rank by score_100 (0-100), shortfall breaking ties

Reads `player_game_features`, which L5 has already scored. This layer decides WHO IS
ELIGIBLE and in WHAT ORDER; it computes no new evidence.

WHY TRIMS RATHER THAN HALF-PLANE GATES. An earlier funnel used
`game_z < 0 AND effort_z < 0` plus `market > 0`. That deleted 10,518 of 15,498 rows and,
more to the point, 79 rows from the score's OWN top 500 -- of which 52 sat within 0.25 of
the boundary. A game at effort_z +0.05 was deleted while one at -0.05 survived, and no
strength on the other axes could compensate, because the gate is binary.

It was also double-counting. game_z, effort_z and market are all inputs to the score, so
the funnel judged them twice: once with a hard edge, once smoothly. Measured, the old cut
pool and the plain top-N-by-score overlapped only 57%, and the plain ranking was the
CLEANER pool -- 94.4% under-hits against 84.5%.

Trimming only the clean quartile fixes both, and the result is verifiable: the survivors
appear in exactly the order the ungated ranking put them in (0 inversions). The trims
remove rows; they never reorder.

ORIENTATION, which is easy to get backwards. game_z and effort_z are raw z-scores, so a
HIGH value is a good night and the top quartile goes. market is already oriented so HIGH
means more under-lean -- more suspicious -- so it is the BOTTOM quartile that goes.
Trimming market's top would delete exactly the rows this is looking for.

`~(x > 0)` NOT `x <= 0` ON CUTS 4 AND 5. A NaN comparison is False in pandas, so the
direct form would delete every row whose opening line was never posted -- rows with
NOTHING TO TEST, which is not the same as rows that FAILED the test. 48 percent of the
population has no opening line and must not be penalised for it.

THE isna() BRANCH IN CUT 6 IS LOAD-BEARING. basketball-reference lists no figure for
two-way and 10-day contracts, so those rows arrive NULL -- and they are the lowest-paid,
most exposed players on any roster. `salary <= MAX` evaluates False for them in pandas
and NULL in SQL, so the naive form deletes exactly the population the cut exists to
preserve. Jontay Porter is one of these rows.

WHAT IS DELIBERATELY NOT GATED.

    SALARY AS THE SOLE MECHANISM. Cut 6 is loose ($20M, the 86th percentile) and motive
    stays in the score. Salary as a gate ALONE was swept over seven thresholds and lost
    at every one: as a share of the pool it ranked, the flagged games went from 10.9%
    ungated to 15.9% at a 20th-percentile gate. A binary gate destroys every distinction
    WITHIN the eligible group and only ever removes rows sitting above the targets.

    EJECTIONS. 56 rows, one in the score's top 100. An ejection is a MECHANISM as well as
    an innocent explanation -- two quick technicals guarantee a low total -- so `ejected`
    and `ejected_alone` are exported as columns to filter on rather than removed here.
    52 of the 72 ejections are solo, which is the interesting shape.

    LOW MINUTES. The strongest benign explanation AND a plausible signature of the thing
    being looked for. Nothing in the data separates them, so it stays evidence.

    python standardize.py run && python export_candidates.py
"""
import pathlib
import warnings

import numpy as np
import pandas as pd

import db

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

OUT = pathlib.Path("out/candidates.csv")

TRIM = 0.25
MAX_SALARY = 20_000_000

# THE FRONTEND TABLE. Every propped player-game, not just the survivors.
#
# A shortlist-only table forces the UI into two queries and makes "why is this game not
# here?" unanswerable. Carrying all 15,498 rows with `in_shortlist` and `cut_failed`
# means one query serves both the ranked list and the per-game explanation, and a game
# that was eliminated can say WHICH cut eliminated it.
#
# `rank` is NULL for eliminated rows rather than 0 or a large sentinel -- they have no
# position in the shortlist, which is different from being last in it. `rank_all` is
# always populated, so the two can be compared to see what the cuts cost each row.
SCORES_DDL = """
CREATE TABLE IF NOT EXISTS player_game_scores (
    player_id     INTEGER NOT NULL REFERENCES players (player_id),
    game_id       TEXT    NOT NULL REFERENCES games (game_id),
    game_date     DATE    NOT NULL,
    player        TEXT    NOT NULL,
    matchup       TEXT,
    tier          TEXT,

    minutes       NUMERIC(6,2),
    points        INTEGER,
    close_line    NUMERIC(6,2),
    close_under   NUMERIC(8,3),
    shortfall     NUMERIC(6,3),
    margin_vs_line NUMERIC(6,2),

    performance   NUMERIC(8,3),
    market        NUMERIC(8,3),
    motive        NUMERIC(8,3),
    score         NUMERIC(8,3),

    -- 0-100 percentile of each, over ALL propped games (not over the shortlist).
    score_100        NUMERIC(6,2),
    performance_100  NUMERIC(6,2),
    market_100       NUMERIC(6,2),
    motive_100       NUMERIC(6,2),

    rank_all      INTEGER,          -- position among ALL propped games
    rank          INTEGER,          -- position in the shortlist; NULL if eliminated
    in_shortlist  BOOLEAN NOT NULL,
    cut_failed    TEXT,             -- the FIRST cut this row failed; NULL if it passed

    under_hit     BOOLEAN,
    ejected       BOOLEAN,
    ejected_alone BOOLEAN,
    salary        BIGINT,
    has_listed_salary BOOLEAN,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS player_game_scores_rank_idx
    ON player_game_scores (rank) WHERE rank IS NOT NULL;
CREATE INDEX IF NOT EXISTS player_game_scores_score_idx
    ON player_game_scores (score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS player_game_scores_date_idx
    ON player_game_scores (game_date);
"""

SCORE_COLS = ["player_id", "game_id", "game_date", "player", "matchup", "tier",
              "minutes", "points", "close_line", "close_under", "shortfall",
              "margin_vs_line", "performance", "market", "motive", "score",
              "score_100", "performance_100", "market_100", "motive_100",
              "rank_all", "rank", "in_shortlist", "cut_failed", "under_hit",
              "ejected", "ejected_alone", "salary", "has_listed_salary"]

SQL = """
SELECT p.full_name AS player, g.matchup, f.*,
       z.game_z, z.effort_z, z.game_z_tier, z.effort_z_tier, z.shortfall_z,
       z.p_price, z.p_line, z.performance, z.market, z.motive, z.score,
       z.score_100, z.performance_100, z.market_100, z.motive_100, z.n_market,
       pb.ejected, pb.ejection_sec, pb.n_stints, pb.last_out_sec,
       pb.points_competitive, pb.points_garbage,
       count(*) FILTER (WHERE pb.ejected)
           OVER (PARTITION BY pb.game_id, pb.ejection_sec) AS n_ejected_together
  FROM player_game_features f
  JOIN players p ON p.player_id = f.player_id
  JOIN games   g ON g.game_id   = f.game_id
  -- INNER on the z table: a row with no standardised components cannot be cut on or
  -- ranked, so it has no business in the shortlist. If this ever drops rows, L5 and L6
  -- have gone out of sync and the count printed below will say so.
  JOIN player_game_z z ON z.player_id = f.player_id AND z.game_id = f.game_id
  LEFT JOIN player_game_pbp pb
         ON pb.player_id = f.player_id AND pb.game_id = f.game_id
"""

EXPORT = ["rank", "rank_all", "player", "game_date", "matchup", "tier",
          "minutes", "points", "close_line", "shortfall", "margin_vs_line",
          "score", "score_100", "performance", "performance_100",
          "market", "market_100", "motive", "motive_100",
          "game_z", "effort_z", "game_z_tier", "effort_z_tier", "shortfall_z",
          "close_under", "open_line", "open_under", "line_move_pct",
          "under_move_pct", "price_only_move", "p_price", "p_line",
          "salary", "has_listed_salary", "under_hit",
          "ejected", "ejected_alone", "n_stints", "last_out_sec",
          "points_competitive", "points_garbage",
          "n_market", "n_player_games", "player_id", "game_id"]


def load(conn):
    d = pd.read_sql(SQL, conn)
    for c in ("minutes", "points", "close_line", "close_under", "open_line",
              "open_under", "line_move_pct", "under_move_pct", "price_only_move",
              "shortfall", "margin_vs_line", "game_z", "effort_z", "game_z_tier",
              "effort_z_tier", "shortfall_z", "p_price", "p_line",
              "performance", "market", "motive", "score", "score_100",
              "performance_100", "market_100", "motive_100", "salary"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ejected"] = d["ejected"].astype("boolean").fillna(False).astype(bool)
    d["ejected_alone"] = d["ejected"] & (d["n_ejected_together"] == 1)
    d["under_hit"] = d["points"] < d["close_line"]
    return d


def main():
    with db.connect() as conn:
        d = load(conn)

    if d.empty:
        raise SystemExit("player_game_features is empty -- run standardize.py first")

    # The ungated ranking, kept so the cost of every trim stays visible per row.
    d["rank_all"] = d["score"].rank(ascending=False, method="min").astype("Int64")

    print(f"start: propped player-games with a score        {len(d):>8,}\n")

    # Percentiles are taken on the FULL propped population BEFORE any cut, so each
    # threshold means what it says. Computing them sequentially would make cut 2 a
    # quartile of whatever cut 1 left, which is a different and much harsher filter.
    gz_hi = d["game_z"].quantile(1 - TRIM)
    ez_hi = d["effort_z"].quantile(1 - TRIM)
    mk_lo = d["market"].quantile(TRIM)

    cuts = [
        (f"1  game_z   top {TRIM:.0%} (good game)", d["game_z"] < gz_hi),
        (f"2  effort_z top {TRIM:.0%} (more involved)", d["effort_z"] < ez_hi),
        (f"3  market   bottom {TRIM:.0%} (leaned over)", d["market"] > mk_lo),
        ("4  no upward line move  (NaN kept)", ~(d["line_move_pct"] > 0)),
        ("5  no upward price-only move  (NaN kept)", ~(d["price_only_move"] > 0)),
        (f"6  salary <= ${MAX_SALARY/1e6:.0f}M or unlisted",
         d["salary"].isna() | (d["salary"] <= MAX_SALARY)),
    ]
    # `cut_failed` records the FIRST cut a row failed, so an eliminated game can explain
    # itself in the UI. First, not all -- the funnel is sequential and the first failure
    # is the one that actually removed it.
    keep = pd.Series(True, index=d.index)
    d["cut_failed"] = pd.Series(pd.NA, index=d.index, dtype="object")
    for label, mask in cuts:
        before = int(keep.sum())
        ok = mask.fillna(False) if mask.dtype == bool else mask
        newly_failed = keep & ~ok
        d.loc[newly_failed & d["cut_failed"].isna(), "cut_failed"] = label.strip()
        keep &= ok
        print(f"{label:<48}{int(keep.sum()):>8,}   (-{before - int(keep.sum()):,})")

    c = d[keep & d["score_100"].notna()].copy()

    # RANK BY score, shortfall BREAKING TIES.
    #
    # The cuts already selected on performance, market and salary, so those inputs are
    # range-restricted among survivors -- they retain 73, 66 and 71 percent of their
    # full-population spread. shortfall retains 120 percent, which is the argument for
    # it as the tiebreak: it is the one quantity no cut touches. It also saturates --
    # every zero-point game ties at 1.000 -- so ties are common and need breaking.
    # RANK ON THE RAW score, DISPLAY score_100.
    #
    # score_100 is a strictly monotone transform, so it carries the same ordering in
    # principle -- but it is stored rounded to 2dp, which collapses 15,494 distinct
    # values into ~9,700 and invents ties the raw score does not have. Ranking on the
    # rounded number would let the shortfall tiebreak reorder rows that are genuinely
    # separated, purely because the display precision hid the gap.
    #
    # Two rows can therefore show the same score_100 and hold different ranks. That is
    # correct: they differ, just below the precision shown.
    c = c.sort_values(["score", "shortfall"], ascending=[False, False])
    c = c.reset_index(drop=True)
    c.insert(0, "rank", range(1, len(c) + 1))

    inversions = int((np.diff(c["rank_all"].astype(float).values) < 0).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c[[x for x in EXPORT if x in c.columns]].to_csv(OUT, index=False)

    # ---- the frontend table: ALL propped games, shortlist flagged ----------------
    d["in_shortlist"] = keep & d["score_100"].notna()
    d = d.merge(c[["player_id", "game_id", "rank"]], on=["player_id", "game_id"],
                how="left")
    d["rank"] = d["rank"].astype("Int64")        # NULL for eliminated rows, not 0
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCORES_DDL)
            # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so the DDL
            # above cannot add a column to one that already exists. Derived from
            # SCORE_COLS so a new column cannot be silently dropped on the floor.
            for col in SCORE_COLS:
                if col in ("player_id", "game_id", "game_date", "player"):
                    continue
                typ = ("NUMERIC(6,2)" if col.endswith("_100") else
                       "INTEGER" if col in ("rank", "rank_all", "points") else
                       "BOOLEAN" if col in ("in_shortlist", "under_hit", "ejected",
                                            "ejected_alone", "has_listed_salary") else
                       "BIGINT" if col == "salary" else
                       "TEXT" if col in ("matchup", "tier", "cut_failed") else
                       "NUMERIC(8,3)")
                cur.execute(f"ALTER TABLE player_game_scores "
                            f"ADD COLUMN IF NOT EXISTS {col} {typ}")
            # Fail loudly rather than silently dropping a column. The type above is
            # inferred from the column NAME, that heuristic can be wrong, and
            # ADD COLUMN IF NOT EXISTS is silent when it no-ops.
            cur.execute("""SELECT column_name FROM information_schema.columns
                            WHERE table_name = 'player_game_scores'""")
            missing = set(SCORE_COLS) - {r[0] for r in cur.fetchall()}
            if missing:
                raise SystemExit(f"player_game_scores: declared columns never reached "
                                 f"the database: {sorted(missing)}")
        conn.commit()
        rows = d[SCORE_COLS].replace({np.nan: None, pd.NA: None}).to_dict("records")
        n = db.upsert(conn, "player_game_scores", rows,
                      conflict=["player_id", "game_id"])
        total = db.count(conn, "player_game_scores")
        listed = db.count(conn, "player_game_scores", "in_shortlist")
    print(f"\n-> player_game_scores   {n:,} rows upserted "
          f"({total:,} total, {listed:,} in the shortlist)")

    print(f"\n-> {OUT}   {len(c):,} rows, {c.player.nunique()} players, "
          f"{c.game_id.nunique()} games")
    print(f"   under hit {100*c.under_hit.mean():.1f}%   "
          f"median salary ${c.salary.median()/1e6:.1f}M   "
          f"ejections retained {int(c.ejected.sum())}")
    # Should be 0. A non-zero count means a trim reordered rather than merely removing,
    # which would mean a cut is doing ranking work it is not supposed to be doing.
    print(f"   order inversions vs the ungated ranking: {inversions}")

    fmt = lambda x: x.assign(salary=x.salary.map(
        lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M"))
    show = ["rank", "rank_all", "player", "game_date", "minutes", "points",
            "close_line", "shortfall", "score_100", "performance_100",
            "market_100", "motive_100", "salary"]
    print(f"\nTOP 25")
    print(fmt(c.head(25))[show].to_string(index=False))


if __name__ == "__main__":
    main()
