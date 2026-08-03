"""Isolation Forest fitted SEPARATELY WITHIN BANDS OF SIMILAR GAMES.

THE PROBLEM THIS ADDRESSES. Fitted on everything, Isolation Forest lost to a plain
weighted sum three times running, for a structural reason: it scores RARITY, and the
target class here is not rare. 349 zero-point games form a dense cluster, so the
forest -- whose only criterion is how few neighbours a row has -- systematically
discounts the crowded region where the signal lives.

CONDITIONING TURNS THE MODE INTO A TAIL. How ordinary a zero-point game is depends
almost entirely on what was expected of the player beforehand:

    close_line      n      share scoring ZERO
    <=6           481          13.3%
    6.5-8       2,091           6.3%
    8.5-10      2,175           3.9%
    10.5-13     3,266           1.4%
    13.5-17     2,788           0.6%
    17+         4,697           0.1%

A zero on a 5.5 line is a Tuesday. A zero on an 18.5 line happens 5 times in 4,697
games. Pooled, the first swamps the second and the forest sees one dense blob. Fitted
per band, the second is genuinely isolated -- which is what IF is actually good at.

WHY THE LINE IS THE RIGHT BAND KEY, and minutes is not. The closing line is the
market's player-specific, game-specific forecast, fixed BEFORE tip-off. Conditioning
on it removes "how good is this player" without touching anything he did. Minutes is
a mediator -- disengaged, benched, fewer minutes -- so banding on it conditions away
the very evidence, which is what cost Porter 03-20 in the peer_z run.

THE CONTROL THAT MATTERS. Banding might help on its own, regardless of the forest.
So this also ranks the LINEAR score within the same bands. If per-band linear matches
per-band isolation, the gain came from conditioning and IF is still contributing
nothing.

Motive is excluded from the features -- see iso_three.py for why -- and reported
alongside so its effect stays visible.

    python analysis/iso_peer.py
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
from sklearn.ensemble import IsolationForest

from weight_audit import live_weights

OUT = HERE / "out"
BLOCK_W = live_weights()[2]
USE = ["performance", "market"]
W = {k: v / sum(BLOCK_W[j] for j in USE) for k, v in BLOCK_W.items() if k in USE}

EDGES = [0, 6, 8, 10, 13, 17, 50]
LABELS = ["<=6", "6.5-8", "8.5-10", "10.5-13", "13.5-17", "17+"]

FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
d = d.dropna(subset=USE).reset_index(drop=True)
d["band"] = pd.cut(d.close_line, EDGES, labels=LABELS)

# One-sided clip, as in iso_three.py: a good game collapses to exactly (0, 0), the
# most crowded point in the space, so the forest cannot spend its budget on the wrong
# tail. Applied BEFORE banding so the clip point is the same in every band.
for k in USE:
    d[f"{k}_pos"] = d[k].clip(lower=0)

print(f"rows {len(d):,}   bands {d.band.nunique()}")
print(f"forest features: " + "  ".join(f"{k} {v:.2f}" for k, v in W.items()))


# ---------------------------------------------------------------- per-band fit
d["iso_raw"] = np.nan
d["iso_p"] = np.nan          # within-band percentile of the isolation score
d["lin_p"] = np.nan          # within-band percentile of the linear score -- the control

for b, idx in d.groupby("band", observed=True).groups.items():
    g = d.loc[idx]
    X = pd.DataFrame({k: g[f"{k}_pos"] * w for k, w in W.items()})
    m = IsolationForest(n_estimators=400, contamination=0.05,
                        random_state=42, n_jobs=-1).fit(X)
    iso = pd.Series(-m.decision_function(X), index=idx)
    lin = sum(W[k] * g[f"{k}_pos"] for k in USE)
    d.loc[idx, "iso_raw"] = iso
    d.loc[idx, "iso_p"] = iso.rank(pct=True)
    d.loc[idx, "lin_p"] = lin.rank(pct=True)

# Within-band percentile is the only comparable currency across bands -- a raw
# isolation score means different things in a 481-row band and a 4,697-row one.
d["iso_band_rank"] = d.iso_p.rank(ascending=False, method="min").astype(int)
d["lin_band_rank"] = d.lin_p.rank(ascending=False, method="min").astype(int)

# Global (unbanded) baselines, recomputed here so all four live in one table.
d["lin_nm"] = sum(W[k] * d[k] for k in USE)
d["lin_rank_nm"] = d.lin_nm.rank(ascending=False, method="min").astype(int)
d["lin_rank"] = d.score.rank(ascending=False, method="min").astype("Int64")

f = d[[r.gd in FLAG.get(r.player, []) for _, r in d.iterrows()]]
g_ = lambda s: 10 ** np.mean(np.log10(pd.to_numeric(s)))

print(f"\nGEOMETRIC-MEAN RANK OF THE 6 FLAGGED GAMES")
for lab, col in (("isolation, per band", "iso_band_rank"),
                 ("linear,    per band", "lin_band_rank"),
                 ("isolation, global   ", None),
                 ("linear,    global   ", "lin_rank_nm"),
                 ("linear + motive (production)", "lin_rank")):
    if col is None:
        continue
    print(f"   {lab:<30}{g_(f[col]):>8,.0f}")

print(f"\n   corr(iso_p, lin_p) within bands "
      f"{d.iso_p.corr(d.lin_p):+.3f}   spearman {d.iso_p.corr(d.lin_p, method='spearman'):+.3f}")
top = lambda c, k=200: set(d.nsmallest(k, c).index)
print(f"   top-200 overlap, per-band isolation vs per-band linear: "
      f"{len(top('iso_band_rank') & top('lin_band_rank'))} of 200")

print(f"\nWHERE THE TOP 200 COMES FROM, BY BAND")
print(f"   {'band':<10}{'n':>7}{'pooled linear':>15}{'per-band iso':>14}")
for b in LABELS:
    n = int((d.band == b).sum())
    a = int((d.nsmallest(200, "lin_rank_nm").band == b).sum())
    c = int((d.nsmallest(200, "iso_band_rank").band == b).sum())
    print(f"   {b:<10}{n:>7,}{a:>15}{c:>14}")

print(f"\nFLAGGED")
cols = ["player", "gd", "band", "minutes", "points", "close_line",
        "performance", "market", "iso_band_rank", "lin_band_rank",
        "lin_rank_nm", "lin_rank"]
print(f.sort_values("iso_band_rank")[cols].to_string(index=False))

print(f"\nTOP 15 BY PER-BAND ISOLATION")
c2 = ["iso_band_rank", "lin_band_rank", "lin_rank_nm", "player", "gd", "band",
      "minutes", "points", "close_line", "performance", "market", "salary"]
y = d.nsmallest(15, "iso_band_rank")[c2].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))

print(f"\nWHAT BANDING SURFACES THAT THE POOLED RANKING BURIES")
d["gap"] = d.lin_rank_nm - d.iso_band_rank
y = d.nlargest(10, "gap")[c2].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))

d.sort_values("iso_band_rank").to_csv(OUT / "iso_peer.csv", index=False)
print(f"\n-> {OUT/'iso_peer.csv'}")
