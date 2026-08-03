"""CURRENT METHODOLOGY: trim the clean end of each axis, then rank by score.

    1  game_z   top 25%      he had a GOOD game
    2  effort_z top 25%      he was MORE involved than usual
    3  market   bottom 25%   the market leaned OVER, against the thesis
    4  no upward line move            (NaN kept)
    5  no upward price-only move      (NaN kept)
    6  salary <= $20M or unlisted
    -> rank by score, shortfall breaking ties

WHY TRIMS RATHER THAN HALF-PLANE GATES. The previous funnel used
`game_z < 0 AND effort_z < 0` plus `market > 0`. That deleted 10,518 of 15,498 rows and,
more to the point, 79 rows from the score's OWN top 500 -- of which 52 sat within 0.25
of the boundary. A game at effort_z +0.05 was deleted while one at -0.05 survived, and
no strength on the other axes could compensate, because the gate is binary.

It was also double-counting: game_z, effort_z and market are all inputs to the score, so
the funnel judged them twice, once with a hard edge and once smoothly. Measured, the old
cut pool and the plain top-N-by-score overlapped only 57%, and the plain ranking was the
CLEANER pool -- 94.4% under-hits against 84.5%.

Trimming only the clean quartile fixes both. The head of the ranking is now IDENTICAL to
the ungated ranking, because the trims remove tail rather than reorder.

ORIENTATION, which is easy to get backwards. game_z and effort_z are raw z-scores, so a
HIGH value is a good night and the top quartile goes. market is already oriented so HIGH
means more under-lean -- more suspicious -- so it is the BOTTOM quartile that goes.
Trimming market's top would delete exactly the rows this is looking for.

CUT 6 IS A GATE WHILE MOTIVE IS ALSO IN THE SCORE, and that is deliberate rather than
sloppy. Salary as a gate ALONE was swept over seven thresholds and lost at every one:
as a share of the pool it ranked, the flagged games went from 10.9% ungated to 15.9% at
a 20th-percentile gate, because a binary gate destroys every distinction WITHIN the
eligible group and only ever removes rows sitting above the targets. The gate here is
set loose ($20M, the 86th percentile) so it removes only the population for whom the
motive argument is implausible, and leaves the ordering to motive in the score.

THE isna() BRANCH IN CUT 6 IS LOAD-BEARING. basketball-reference lists no figure for
two-way and 10-day contracts, so those rows arrive NULL -- and they are the lowest-paid,
most exposed players on any roster. `salary <= MAX` evaluates False for them in pandas
and NULL in SQL, so the naive form deletes exactly the population the cut exists to
preserve. Jontay Porter is one of these rows.

EJECTIONS ARE NOT GATED. 56 rows, one of them in the score's top 100. An ejection is a
MECHANISM as well as an innocent explanation -- two quick technicals guarantee a low
total -- so `ejected` and `ejected_alone` are carried in the output to filter on rather
than removed here. 52 of the 72 ejections are solo, which is the interesting shape.

`~(x > 0)` NOT `x <= 0` on cuts 4 and 5. A NaN comparison is False in pandas, so the
direct form deletes every row whose opening line was never posted -- rows with nothing
to test, which is not the same as rows that failed the test.

    python analysis/two_metrics.py && python analysis/combined_cut.py
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import db
from weight_audit import live_weights

OUT = HERE / "out"
BLOCK_W = live_weights()[2]          # read from two_metrics.py, never copied
TRIM = 0.25
MAX_SALARY = 20_000_000

FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

# SOURCE IS two_metrics.csv. Its z-baselines are computed on EVERY game a player played
# and only then filtered to propped rows, so every cut and the ranking read from one
# consistent set of definitions rather than two files that could drift.
d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
d = d.drop(columns=["rank", "shortfall", "line_move_pct", "under_move_pct",
                    "price_only_move"], errors="ignore")

with db.connect() as c:
    mv = pd.read_sql("""SELECT player_id, game_id, open_line, line_move_pct,
                               under_move_pct, price_only_move
                          FROM player_game_features""", c)
    # `ejected_alone` matters more than `ejected`. A multi-player altercation is hard to
    # stage; a solo ejection is one person's decision.
    pbp = pd.read_sql("""SELECT game_id, player_id, ejected, ejection_sec, n_stints,
                                last_out_sec, points_competitive, points_garbage,
                                count(*) FILTER (WHERE ejected)
                                    OVER (PARTITION BY game_id, ejection_sec)
                                    AS n_ejected_together
                           FROM player_game_pbp""", c)
mv["game_id"] = mv.game_id.astype(str)
pbp["game_id"] = pbp.game_id.astype(str)
d = (d.merge(mv, on=["player_id", "game_id"], how="left")
       .merge(pbp, on=["player_id", "game_id"], how="left"))
d["ejected"] = d.ejected.fillna(False).astype(bool)
d["ejected_alone"] = d.ejected & (d.n_ejected_together == 1)

# shortfall against the market's own forecast. The one measure immune to the floor
# effect: a player averaging 5 points who scores 0 is barely -1 sd from himself while a
# star is -3, so any z-threshold is secretly a threshold on how much a player normally
# scores. `1 - points/line` is 1.00 for a zero-point game whoever produced it, because
# the denominator is the market's player-specific expectation rather than his own
# variance. It saturates, so margin_vs_line records the absolute size of the miss.
d["shortfall"] = (1 - d.points / d.close_line.replace(0, np.nan)).clip(0, 1).round(3)
d["margin_vs_line"] = (d.points - d.close_line).round(1)

print(f"start: propped player-games with a score        {len(d):>8,}\n")


def step(mask, label):
    global d
    before = len(d)
    d = d[mask].copy()
    print(f"{label:<48}{len(d):>8,}   (-{before-len(d):,})")


# Percentiles are taken on the FULL propped population before any cut, so each threshold
# means what it says. Computing them sequentially would make cut 2 a quartile of what
# cut 1 left, which is a different and much harsher filter.
gz_hi = d.game_z.quantile(1 - TRIM)
ez_hi = d.effort_z.quantile(1 - TRIM)
mk_lo = d.market.quantile(TRIM)

step(d.game_z < gz_hi, f"1  game_z   top {TRIM:.0%} (good game)")
step(d.effort_z < ez_hi, f"2  effort_z top {TRIM:.0%} (more involved)")
step(d.market > mk_lo, f"3  market   bottom {TRIM:.0%} (leaned over)")
step(~(d.line_move_pct > 0), "4  no upward line move  (NaN kept)")
step(~(d.price_only_move > 0), "5  no upward price-only move  (NaN kept)")
step(d.salary.isna() | (d.salary <= MAX_SALARY),
     f"6  salary <= ${MAX_SALARY/1e6:.0f}M or unlisted")

# RANK BY THE COMBINED SCORE, shortfall breaking ties.
#
#     score = 0.45*performance + 0.30*market + 0.25*motive
#
# Weights are equal WITHIN each block -- analysis/weight_audit.py showed unequal ones do
# nothing (one-at-a-time swing 0.02-0.09 against 0.58 for the motive block, and 48% of
# random weight vectors beat the previously tuned ones). The block weights are a stated
# prior, not a fit.
#
# shortfall as tiebreaker is nearly clean: it enters score at 0.333*0.45 = 0.15 of the
# total, and among rows the score ties it is usually the thing separating them.
d["score"] = (BLOCK_W["performance"] * d.performance
              + BLOCK_W["market"] * d.market
              + BLOCK_W["motive"] * d.motive).round(3)
d = d[d.score.notna()].sort_values(["score", "shortfall"], ascending=[False, False])
d = d.reset_index(drop=True)
d.insert(0, "rank", range(1, len(d) + 1))

# Rank in the ungated population, so the cost of the trims stays visible per row.
full = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
full["rank_all"] = full.score.rank(ascending=False, method="min")
d = d.merge(full[["player_id", "game_id", "rank_all"]],
            on=["player_id", "game_id"], how="left")

d.to_csv(OUT / "combined_cut.csv", index=False)
print(f"\n-> {OUT/'combined_cut.csv'}   {len(d):,} rows, "
      f"{d.player.nunique()} players, {d.game_id.nunique()} games")
print(f"   under hit {100*d.under_hit.mean():.1f}%   "
      f"median salary ${d.salary.median()/1e6:.1f}M   "
      f"ejections retained {int(d.ejected.sum())}")

fmt = lambda x: x.assign(salary=x.salary.map(
    lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M"))
cols = ["rank", "rank_all", "player", "gd", "minutes", "points", "close_line",
        "shortfall", "score", "performance", "market", "motive", "salary",
        "under_hit"]

f = d[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in d.iterrows()]]
print(f"\nFLAGGED  ({len(f)} of 6 survive)")
if len(f):
    print(fmt(f[cols]).to_string(index=False))
miss = [(p, g) for p, gs in FLAG.items() for g in gs
        if not ((f.player == p) & (f.gd == g)).any()]
if miss:
    print(f"   trimmed out: {miss}")

print(f"\nTOP 25")
print(fmt(d.head(25)[cols]).to_string(index=False))
