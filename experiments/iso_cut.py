"""Isolation Forest run ONLY on the survivors of the cut funnel.

WHY THIS IS THE RIGHT PLACE FOR IT, when three earlier attempts were not.

Isolation Forest scores rarity in ANY direction, which has been the problem every
time: fitted on all 15,498 propped games it spent half its budget on career games,
and the one-sided clip that fixed the direction introduced a sparse `(0, x)` edge
that promoted games players actually WON.

The cut funnel removes both failure modes before the forest ever runs. Every survivor
has already cleared:

    game_z < 0 AND effort_z < 0        the good half of performance is gone
    market > 0                         the over-leaning half is gone
    no upward line or price-only move  contrary movement is gone
    salary <= $20M or unlisted         the no-motive population is gone
    not ejected

So the good side is not clipped to a degenerate point mass -- it is absent. Within
what remains, "far from the bulk" and "worse than the bulk" point the same way,
because the bulk is now games that BARELY cleared the gates and the sparse tail is
games that cleared them by a mile. That is the only configuration in which IF's
criterion and the question being asked coincide.

WHAT IT CAN ADD OVER THE LINEAR SCORE. The score is a weighted sum, so it ranks by
distance along one direction and a game must be bad on the AVERAGE. IF can surface a
row that is unremarkable on the average but sits in a sparse CORNER -- saturated
shortfall with an ordinary game_z, say, or an extreme market lean with a mild
performance drop. Post-cut those corners are exactly the rows a sum buries.

FEATURES ARE STANDARDISED WITHIN THE SURVIVORS, not inherited from the full league.
The gates compressed every input -- performance, market and motive retain 73, 66 and
71 percent of their full-population spread -- so league-scaled features would present
the forest with four nearly-constant axes. Re-standardising restores the contrast
that actually exists among the 2,000 rows.

Motive is excluded from the features (see iso_three.py) and doubly so here: cut 5
already selected on salary, so it is the most range-restricted axis of all. It is
carried through to the output so its effect stays visible.

    python analysis/combined_cut.py && python analysis/iso_cut.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

OUT = HERE / "out"
FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

# Oriented so HIGHER = more suspicious, matching the score's convention.
FEATS = {"shortfall":  lambda x: x.shortfall,
         "game_z":     lambda x: -x.game_z,
         "effort_z":   lambda x: -x.effort_z,
         "market":     lambda x: x.market}

d = pd.read_csv(OUT / "combined_cut.csv", dtype={"game_id": str}).reset_index(drop=True)
full = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})

X = pd.DataFrame({k: fn(d) for k, fn in FEATS.items()})
X = ((X - X.mean()) / X.std()).fillna(0)

print(f"survivors {len(d):,}   players {d.player.nunique()}   "
      f"games {d.game_id.nunique()}")
print(f"\nHOW MUCH SPREAD SURVIVED THE GATES (sd among survivors / sd league-wide):")
for k, fn in FEATS.items():
    a, b = fn(d).std(), fn(full).std()
    print(f"   {k:<12}{a:.3f} / {b:.3f}   = {100*a/b:>5.0f}%")

m = IsolationForest(n_estimators=400, contamination=0.05,
                    random_state=42, n_jobs=-1).fit(X)
d["iso"] = -m.decision_function(X)
d["iso_rank"] = d.iso.rank(ascending=False, method="min").astype(int)
d["score_rank"] = d["rank"]                      # from combined_cut, score-then-shortfall

print(f"\ncorr(iso, score) {d.iso.corr(d.score):+.3f}   "
      f"spearman {d.iso.corr(d.score, method='spearman'):+.3f}")
top = lambda c, k=100: set(d.nsmallest(k, c).index)
for k in (50, 100, 200):
    print(f"   top-{k} overlap with the score ranking: "
          f"{len(top('iso_rank', k) & top('score_rank', k))} of {k}")

print(f"\nWHAT THE FOREST IS SEPARATING ON (mean among its top 100 vs the rest):")
t = d.nsmallest(100, "iso_rank")
rest = d[~d.index.isin(t.index)]
for k, fn in FEATS.items():
    print(f"   {k:<12} top100 {fn(t).mean():+.3f}   rest {fn(rest).mean():+.3f}")
print(f"   {'minutes':<12} top100 {t.minutes.mean():+.3f}   rest {rest.minutes.mean():+.3f}")
print(f"   {'points':<12} top100 {t.points.mean():+.3f}   rest {rest.points.mean():+.3f}")

f = d[[r.gd in FLAG.get(r.player, []) for _, r in d.iterrows()]]
g_ = lambda s: 10 ** np.mean(np.log10(pd.to_numeric(s)))
print(f"\nFLAGGED  ({len(f)} of 6 survived the cuts)")
cols = ["player", "gd", "minutes", "points", "line", "shortfall", "game_z",
        "effort_z", "market", "iso_rank", "score_rank"]
print(f.sort_values("iso_rank")[cols].to_string(index=False))
if len(f):
    print(f"\n   geometric-mean rank among the {len(d):,} survivors:"
          f"   isolation {g_(f.iso_rank):,.0f}   score {g_(f.score_rank):,.0f}")

print(f"\nTOP 15 BY ISOLATION")
c2 = ["iso_rank", "score_rank", "player", "gd", "minutes", "points", "line",
      "shortfall", "game_z", "effort_z", "market", "salary"]
y = d.nsmallest(15, "iso_rank")[c2].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))

print(f"\nWHAT ISOLATION ADDS -- high by forest, low by the score")
d["gap"] = d.score_rank - d.iso_rank
y = d.nlargest(10, "gap")[c2].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
a = ax[0]
sc = a.scatter(d.market, -d.game_z, c=d.iso, s=14, cmap="viridis", alpha=.75)
a.scatter(f.market, -f.game_z, s=130, facecolor="none", edgecolor="crimson", lw=1.8)
for _, r in f.iterrows():
    a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}", (r.market, -r.game_z),
               textcoords="offset points", xytext=(8, 5), color="crimson", fontsize=8)
plt.colorbar(sc, ax=a, label="isolation")
a.set_xlabel("market"); a.set_ylabel("-game_z  (higher = worse night)")
a.set_title(f"Survivors of the cuts, coloured by isolation  (n={len(d):,})")
a.grid(alpha=.2)

a = ax[1]
a.scatter(d.score_rank, d.iso_rank, s=10, alpha=.35, color="#4C78A8")
a.scatter(f.score_rank, f.iso_rank, s=130, color="crimson", edgecolor="white", zorder=5)
lim = [0, len(d)]
a.plot(lim, lim, color="black", ls="--", lw=1)
a.set_xlabel("rank by score"); a.set_ylabel("rank by isolation")
a.set_title(f"Agreement  (spearman {d.iso.corr(d.score, method='spearman'):+.3f})")
a.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT / "iso_cut.png", dpi=140)

d.sort_values("iso_rank").to_csv(OUT / "iso_cut.csv", index=False)
print(f"\n-> {OUT/'iso_cut.csv'}\n-> {OUT/'iso_cut.png'}")
