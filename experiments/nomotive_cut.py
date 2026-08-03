"""Motive as a GATE instead of a score component.

THE ARGUMENT FOR THIS. motive is the most contaminated feature in the pipeline. In the
leave-one-player-out logistic its coefficient ran 3.9x and 7.0x the SUM of every other
coefficient, and including salary drove held-out AUC to 0.087 -- because "cheap" is
what the two label players have in common, so weighting it up always flatters them.
The weight audit found it was also the largest single lever on the ranking (one-at-a-time
swing 0.580 against <=0.09 for anything else).

A gate removes that leverage entirely. Salary decides WHO is eligible; nothing about
salary decides the ORDER. A player either could plausibly have been approached or he
could not, which is closer to what the axis actually asserts -- it was never a claim
that $1.9M is 30 percent more suspicious than $2.6M.

    score_nm = 0.60 * performance + 0.40 * market

0.60/0.40 is 0.45/0.30 renormalised, so the surviving blocks keep their relative
standing rather than silently becoming equal.

THRESHOLDS ARE SWEPT, not chosen here. The percentile is over PLAYERS, not
player-games, so "bottom 20%" means one fifth of the men on rosters rather than one
fifth of the games -- the two differ a lot, because expensive players play more.

UNLISTED SALARY ALWAYS PASSES. basketball-reference publishes no figure for two-way and
10-day contracts, and those are the lowest-paid players in the league. `salary <= X`
evaluates False for them in pandas, so the naive form deletes exactly the population
the gate exists to keep. Jontay Porter is one of these rows.

    python analysis/nomotive_cut.py
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
from weight_audit import live_weights

OUT = HERE / "out"
_, _, BLOCK_W = live_weights()
USE = ["performance", "market"]
W = {k: BLOCK_W[k] / sum(BLOCK_W[j] for j in USE) for k in USE}

PCTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.86]
FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
d["score_nm"] = (W["performance"] * d.performance + W["market"] * d.market).round(3)

# PLAYER-level salary percentile. One row per man, so the threshold means what it says.
pl = (d[["player_id", "salary", "has_listed_salary"]].drop_duplicates("player_id")
        .reset_index(drop=True))
pl["sal_pct"] = pl.salary.rank(pct=True)
d = d.drop(columns=["sal_pct"], errors="ignore").merge(
    pl[["player_id", "sal_pct"]], on="player_id", how="left")

g_ = lambda s: 10 ** np.mean(np.log10(pd.to_numeric(s)))
print(f"score_nm = {W['performance']:.2f}*performance + {W['market']:.2f}*market"
      f"   (motive removed from the average)")
print(f"players {pl.player_id.nunique()}   listed {int(pl.salary.notna().sum())}   "
      f"unlisted/two-way {int(pl.salary.isna().sum())}")

print(f"\nTHRESHOLD SWEEP -- keep salary at or below the Nth percentile, or unlisted")
print(f"   {'cut':>6}{'$ threshold':>14}{'players':>9}{'games':>8}"
      f"{'flagged kept':>14}{'geo-mean rank':>15}")
rows = []
for p in PCTS:
    thr = pl.salary.quantile(p)
    keep = d.sal_pct.isna() | (d.sal_pct <= p)
    s = d[keep].copy()
    # Int64: 41 rows have no market block and therefore no score_nm.
    s["r"] = s.score_nm.rank(ascending=False, method="min").astype("Int64")
    f = s[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in s.iterrows()]]
    gm = g_(f.r) if len(f) else np.nan
    rows.append((p, thr, keep.sum(), len(f), gm))
    npl = int((pl.sal_pct.isna() | (pl.sal_pct <= p)).sum())
    gs = f"{gm:,.0f}" if len(f) else "--"
    print(f"   {p:>5.0%}{'$'+format(thr/1e6,'.1f')+'M':>14}{npl:>9}"
          f"{int(keep.sum()):>8,}{str(len(f))+' of 6':>14}{gs:>15}")

# Reference points that use no gate at all.
d["r_all"] = d.score_nm.rank(ascending=False, method="min").astype("Int64")
d["r_prod"] = d.score.rank(ascending=False, method="min").astype("Int64")
fa = d[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in d.iterrows()]]
print(f"\n   {'no gate, motive dropped':<34}{len(d):>8,} games   "
      f"geo-mean {g_(fa.r_all):>6,.0f}")
print(f"   {'no gate, motive in the score':<34}{len(d):>8,} games   "
      f"geo-mean {g_(fa.r_prod):>6,.0f}   <- production")

# ---- build the chosen output ------------------------------------------------
CHOSEN = 0.20
keep = d.sal_pct.isna() | (d.sal_pct <= CHOSEN)
s = d[keep & d.score_nm.notna()].sort_values("score_nm", ascending=False)
s = s.drop(columns=["rank"], errors="ignore").reset_index(drop=True)   # `rank` came
s.insert(0, "rank", range(1, len(s) + 1))                              # from the CSV
thr = pl.salary.quantile(CHOSEN)

print(f"\n{'='*78}\nGATE AT THE {CHOSEN:.0%} PERCENTILE  (salary <= ${thr/1e6:.2f}M, "
      f"or unlisted)\n{'='*78}")
print(f"   {len(s):,} player-games   {s.player.nunique()} players   "
      f"under-hit {100*s.under_hit.mean():.1f}%")

f = s[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in s.iterrows()]]
print(f"\nFLAGGED  ({len(f)} of 6 pass the gate)")
c = ["rank", "player", "gd", "minutes", "points", "close_line", "shortfall",
     "score_nm", "performance", "market", "salary", "r_all", "r_prod"]
if len(f):
    y = f[c].copy()
    y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
    print(y.to_string(index=False))
miss = [(p, g) for p, gs in FLAG.items() for g in gs
        if not ((f.player == p) & (f.gd == g)).any()]
if miss:
    print(f"   gated out: {miss}")

print(f"\nTOP 25")
y = s.head(25)[c].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))

s[[x for x in c if x in s.columns] + ["player_id", "game_id", "under_hit"]] \
    .to_csv(OUT / "nomotive_cut.csv", index=False)
print(f"\n-> {OUT/'nomotive_cut.csv'}")
