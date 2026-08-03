"""Loosened funnel: trim only the CLEAN END of each axis, then rank by score.

WHY THE OLD CUT 1 WAS TOO MUCH. `game_z < 0 AND effort_z < 0` deletes 10,518 of 15,498
rows -- and 79 of the score's own top 500. Of those 79, fifty-two sat within 0.25 of the
boundary: a game at effort_z +0.05 was deleted while one at -0.05 survived, and no
amount of evidence on the other axes could compensate, because the gate is binary.

Worse, it double-counts. game_z and effort_z are already IN the score, so the old funnel
judged them twice -- once with a hard edge, once smoothly -- while cut 2 did the same
for market. Measured: the cut pool and the plain top-N-by-score overlap only 57%, and
the plain ranking is the CLEANER pool (94.4% under-hits against 84.5%).

WHAT A CUT SHOULD DO instead is encode facts the score cannot express -- eligibility,
not degree. Those are cuts 3-6, and they are nearly free: between them they cost 16 rows
of the score's top 500.

SO THE PERFORMANCE GATES BECOME TRIMS. Rather than keeping the bad half, drop only the
unambiguously clean tail of each axis:

    game_z   >= its (1-TRIM) percentile   he had a GOOD game
    effort_z >= its (1-TRIM) percentile   he was MORE involved than usual
    market   <= its TRIM percentile       the market leaned OVER, against the thesis

ORIENTATION. game_z and effort_z are raw z-scores, so HIGH is a good night and the top
is what goes. market is already oriented so HIGH means more under-lean -- more
suspicious -- so it is the BOTTOM that goes. Trimming market's top would delete exactly
the rows the whole exercise is looking for.

TRIM is swept rather than chosen. At TRIM=0.50 the game_z and effort_z trims reproduce
the old cut 1 as an OR rather than an AND, which is why even the loosest end of the
sweep is more permissive than what it replaces.

    python analysis/loose_cut.py
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

OUT = HERE / "out"
TRIMS = [0.10, 0.20, 0.25, 0.33, 0.50]
CHOSEN = 0.25
MAX_SALARY = 20_000_000
FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
with db.connect() as c:
    mv = pd.read_sql("""SELECT player_id, game_id, open_line, line_move_pct,
                               under_move_pct, price_only_move
                          FROM player_game_features""", c)
    pbp = pd.read_sql("""SELECT player_id, game_id, ejected, n_stints, last_out_sec
                           FROM player_game_pbp""", c)
mv["game_id"] = mv.game_id.astype(str)
pbp["game_id"] = pbp.game_id.astype(str)
d = (d.drop(columns=["line_move_pct", "under_move_pct", "price_only_move"],
            errors="ignore")
       .merge(mv, on=["player_id", "game_id"], how="left")
       .merge(pbp, on=["player_id", "game_id"], how="left"))
d["ejected"] = d.ejected.fillna(False).astype(bool)

isf = lambda x: [r.player in FLAG and r.gd in FLAG[r.player] for _, r in x.iterrows()]
g_ = lambda s: 10 ** np.mean(np.log10(pd.to_numeric(s)))

# Cuts 3-6 are unchanged. `~(x > 0)` not `x <= 0`: a NaN comparison is False in pandas,
# so the direct form deletes every row whose opening line was never posted -- rows with
# nothing to test, which is not the same as rows that failed the test.
ELIG = (~(d.line_move_pct > 0)
        & ~(d.price_only_move > 0)
        & (d.salary.isna() | (d.salary <= MAX_SALARY))
        & ~d.ejected)


def trims(t):
    """Clean-end trims. Percentiles, not fixed z-values, so the share removed is the
    same on each axis regardless of how each is distributed."""
    return {
        f"game_z   top {t:.0%} (good game)":   d.game_z < d.game_z.quantile(1 - t),
        f"effort_z top {t:.0%} (more involved)": d.effort_z < d.effort_z.quantile(1 - t),
        f"market   bottom {t:.0%} (leaned over)": d.market > d.market.quantile(t),
    }


print(f"pool {len(d):,}\n")
print("TRIM SWEEP  (cuts 3-6 held fixed, performance/market gates loosened)")
print(f"   {'trim':>6}{'pool':>8}{'flagged':>10}{'geo rank':>11}{'% of pool':>12}"
      f"{'under-hit':>11}{'med salary':>12}")
for t in TRIMS:
    m = ELIG.copy()
    for v in trims(t).values():
        m &= v
    s = d[m & d.score.notna()].copy()
    s["r"] = s.score.rank(ascending=False, method="min")
    f = s[isf(s)]
    print(f"   {t:>5.0%}{len(s):>8,}{str(len(f))+' of 6':>10}{g_(f.r):>11,.0f}"
          f"{100*g_(f.r)/len(s):>11.1f}%{100*s.under_hit.mean():>10.1f}%"
          f"{'$'+format(s.salary.median()/1e6,'.1f')+'M':>12}")

# Reference rows: the funnel this replaces, and no gates at all.
old = ((d.game_z < 0) & (d.effort_z < 0) & (d.market > 0) & ELIG)
for nm, m in (("old funnel (cut 1 AND cut 2)", old),
              ("cuts 3-6 only, no trims", ELIG),
              ("no gates at all", pd.Series(True, index=d.index))):
    s = d[m & d.score.notna()].copy()
    s["r"] = s.score.rank(ascending=False, method="min")
    f = s[isf(s)]
    print(f"   {nm:<30}{len(s):>6,}{str(len(f))+' of 6':>10}{g_(f.r):>11,.0f}"
          f"{100*g_(f.r)/len(s):>11.1f}%{100*s.under_hit.mean():>10.1f}%"
          f"{'$'+format(s.salary.median()/1e6,'.1f')+'M':>12}")

# ---- build the chosen funnel, printing the cost of each step ----------------
print(f"\n{'='*80}\nFUNNEL AT TRIM = {CHOSEN:.0%}\n{'='*80}")
cur = pd.Series(True, index=d.index)
n0 = int(cur.sum())
print(f"   {'start':<44}{n0:>7,}")
steps = list(trims(CHOSEN).items()) + [
    ("no upward line move (NaN kept)", ~(d.line_move_pct > 0)),
    ("no upward price-only move (NaN kept)", ~(d.price_only_move > 0)),
    (f"salary <= ${MAX_SALARY/1e6:.0f}M or unlisted",
     d.salary.isna() | (d.salary <= MAX_SALARY)),
    ("not ejected", ~d.ejected),
]
for i, (nm, m) in enumerate(steps, 1):
    before = int(cur.sum())
    cur &= m
    print(f"   {str(i)+'  '+nm:<44}{int(cur.sum()):>7,}   (-{before-int(cur.sum()):,})")

s = d[cur & d.score.notna()].sort_values("score", ascending=False)
s = s.drop(columns=["rank"], errors="ignore").reset_index(drop=True)
s.insert(0, "rank", range(1, len(s) + 1))
s["rank_all"] = d.score.rank(ascending=False, method="min") \
                 .reindex(s.index).values if False else np.nan
allr = d.assign(ra=d.score.rank(ascending=False, method="min"))[
    ["player_id", "game_id", "ra"]]
s = s.drop(columns=["rank_all"]).merge(allr, on=["player_id", "game_id"], how="left") \
     .rename(columns={"ra": "rank_all"})

print(f"\n   {len(s):,} games   {s.player.nunique()} players   "
      f"under-hit {100*s.under_hit.mean():.1f}%   median salary "
      f"${s.salary.median()/1e6:.1f}M")

f = s[isf(s)]
print(f"\nFLAGGED  ({len(f)} of 6 survive)")
cols = ["rank", "rank_all", "player", "gd", "minutes", "points", "close_line",
        "shortfall", "score", "performance", "market", "motive", "game_z",
        "effort_z", "salary"]
fmt = lambda x: x.assign(salary=x.salary.map(
    lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M"))
if len(f):
    print(fmt(f[cols]).to_string(index=False))
miss = [(p, gg) for p, gs in FLAG.items() for gg in gs
        if not ((f.player == p) & (f.gd == gg)).any()]
if miss:
    print(f"   trimmed out: {miss}")

print(f"\nTOP 25")
print(fmt(s.head(25)[cols]).to_string(index=False))

s.to_csv(OUT / "loose_cut.csv", index=False)
print(f"\n-> {OUT/'loose_cut.csv'}")
