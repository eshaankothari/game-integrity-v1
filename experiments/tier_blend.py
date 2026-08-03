"""Does adding tier-relative z to the performance block improve the ranking?

THE TWO BASELINES MEASURE DIFFERENT THINGS.

    own    (x - his mean) / his sd            "unlike HIM"
    tier   (x - tier mean) / tier sd          "bad for a player in his ROLE"

own has a soft-baseline problem: a player with many quiet games sets a low bar for
himself, so a bad night barely registers. Malik Beasley is exactly that player -- against
starters generally his 3-point games look far worse than they do against his own season.

tier has the opposite problem: the bench tier contains a lot of 6-minute end-of-rotation
appearances, so a bench player who normally does more gets graded against a population
that mostly does less. Jontay Porter 01-20 goes POSITIVE (+0.27) on tier, meaning "above
average for a bench player", which is true and useless.

Neither dominates, which is the argument for blending rather than choosing. Measured
earlier as standalone baselines:

    game_z      Beasley 01-26   01-06   02-27   03-10  |  Porter 03-20   01-20
    own              -1.25   -1.28   -0.20   +0.75  |     -0.81   -0.17
    tier             -1.66   -1.68   -0.91   -0.23  |     -0.56   +0.27

TIER DOES NOT FIX THE MINUTES BIAS -- corr(game_z, minutes) was +0.337 own and +0.351
tier, because the mean within-group spread of minutes is 6.94 for tier against 7.75
league-wide. Grouping by role barely constrains minutes at all. That is not what this is
for; it is for removing the self-set curve.

BLENDS TESTED. Each keeps the block structure and only changes what feeds `performance`,
so any difference is attributable to the baseline and nothing else.

    python analysis/tier_blend.py
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

from weight_audit import live_weights

OUT = HERE / "out"
_, _, BLOCK_W = live_weights()
TRIM = 0.25
MAX_SALARY = 20_000_000
FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
pz = pd.read_csv(OUT / "peer_z.csv", dtype={"game_id": str})[
    ["player_id", "game_id", "game_z_own", "effort_z_own",
     "game_z_tier", "effort_z_tier", "tier"]]
d = d.drop(columns=["tier"], errors="ignore").merge(
    pz, on=["player_id", "game_id"], how="left")

isf = lambda x: [r.player in FLAG and r.gd in FLAG[r.player] for _, r in x.iterrows()]
g_ = lambda s: 10 ** np.mean(np.log10(pd.to_numeric(s)))

# `own` here must reproduce two_metrics exactly, or the comparison is against a straw
# man. peer_z recomputes the same quantity, so this asserts they agree.
chk = (d.game_z - d.game_z_own).abs().max()
print(f"sanity: max |game_z(two_metrics) - game_z_own(peer_z)| = {chk:.4f}\n")

# Each blend defines the PERFORMANCE block's inputs, oriented so higher = worse night.
# shortfall_z is untouched throughout -- it has no player or tier baseline by design,
# being already player-relative through the line.
BLENDS = {
    "A own only (current)":     lambda x: [-x.game_z_own, -x.effort_z_own],
    "B own + tier game_z":      lambda x: [-x.game_z_own, -x.effort_z_own,
                                           -x.game_z_tier],
    "C own + tier both":        lambda x: [-x.game_z_own, -x.effort_z_own,
                                           -x.game_z_tier, -x.effort_z_tier],
    "D 50/50 game_z, own effort": lambda x: [-(x.game_z_own + x.game_z_tier) / 2,
                                             -x.effort_z_own],
    "E 50/50 both":             lambda x: [-(x.game_z_own + x.game_z_tier) / 2,
                                           -(x.effort_z_own + x.effort_z_tier) / 2],
    "F tier only":              lambda x: [-x.game_z_tier, -x.effort_z_tier],
}


def build(x, parts_fn):
    """performance = equal-weight mean of the blend's parts plus shortfall_z, using the
    present-weight form so a missing component does not drag the block toward zero."""
    parts = parts_fn(x) + [x.shortfall_z]
    num = sum(p.fillna(0) for p in parts)
    den = sum(p.notna().astype(float) for p in parts)
    perf = num / den.replace(0, np.nan)
    return (BLOCK_W["performance"] * perf
            + BLOCK_W["market"] * x.market
            + BLOCK_W["motive"] * x.motive)


def funnel(x, sc):
    """The 6-cut methodology, with the blend's own game_z/effort_z driving cuts 1-2."""
    gz, ez = -sc["gz"], -sc["ez"]          # back to raw orientation for the trims
    m = ((gz < gz.quantile(1 - TRIM))
         & (ez < ez.quantile(1 - TRIM))
         & (x.market > x.market.quantile(TRIM))
         & ~(x.line_move_pct > 0)
         & ~(x.price_only_move > 0)
         & (x.salary.isna() | (x.salary <= MAX_SALARY)))
    return m


mv = pd.read_csv(OUT / "combined_cut.csv", dtype={"game_id": str})[
    ["player_id", "game_id"]].assign(in_cut=True)
full = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
for c_ in ("line_move_pct", "price_only_move"):
    if c_ not in d.columns:
        d[c_] = full[c_].values if c_ in full.columns else np.nan

print("FULL POOL  (no cuts)")
print(f"   {'blend':<28}{'geo rank':>10}{'flagged ranks':>44}")
res = {}
for nm, fn in BLENDS.items():
    s = build(d, fn)
    r = s.rank(ascending=False, method="min")
    f = r[isf(d)]
    res[nm] = (s, r)
    print(f"   {nm:<28}{g_(f):>10,.0f}   "
          f"{str(sorted(int(v) for v in f)):>41}")

print("\nAFTER THE 6 CUTS  (each blend runs its own funnel)")
print(f"   {'blend':<28}{'pool':>8}{'kept':>9}{'geo rank':>10}{'% of pool':>11}")
for nm, fn in BLENDS.items():
    parts = fn(d)
    sc = {"gz": parts[0], "ez": parts[1]}
    m = funnel(d, sc)
    s = res[nm][0][m & res[nm][0].notna()]
    r = s.rank(ascending=False, method="min")
    sub = d.loc[s.index]
    f = r[isf(sub)]
    print(f"   {nm:<28}{len(s):>8,}{str(len(f))+' of 6':>9}{g_(f):>10,.0f}"
          f"{100*g_(f)/len(s):>10.1f}%")

# ---- per-game detail on the flagged set ------------------------------------
print(f"\nPER-GAME RANK IN THE FULL POOL")
fl = d[isf(d)].copy()
tab = fl[["player", "gd", "tier", "minutes", "points",
          "game_z_own", "game_z_tier"]].copy()
for nm in BLENDS:
    tab[nm.split()[0]] = res[nm][1][fl.index].astype(int).values
print(tab.sort_values("A").to_string(index=False))

# ---- does tier change WHO is at the top? -----------------------------------
print(f"\nTOP-200 OVERLAP WITH THE CURRENT BLEND (A)")
topA = set(res["A own only (current)"][1].nsmallest(200).index)
for nm in BLENDS:
    if nm.startswith("A"):
        continue
    t = set(res[nm][1].nsmallest(200).index)
    sub = d.loc[list(t)]
    print(f"   {nm:<28}{len(topA & t):>4} of 200   "
          f"median salary ${sub.salary.median()/1e6:>5.1f}M   "
          f"median line {sub.close_line.median():>4.1f}   "
          f"avg min {sub.minutes.mean():>4.1f}")
subA = d.loc[list(topA)]
print(f"   {'A own only (current)':<28}{200:>4} of 200   "
      f"median salary ${subA.salary.median()/1e6:>5.1f}M   "
      f"median line {subA.close_line.median():>4.1f}   "
      f"avg min {subA.minutes.mean():>4.1f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
a = ax[0]
a.scatter(d.game_z_own, d.game_z_tier, s=5, alpha=.12, color="#4C78A8",
          rasterized=True)
a.scatter(fl.game_z_own, fl.game_z_tier, s=120, color="crimson",
          edgecolor="white", zorder=5)
for _, r in fl.iterrows():
    a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}", (r.game_z_own, r.game_z_tier),
               textcoords="offset points", xytext=(8, 5), color="crimson", fontsize=8)
lim = [d.game_z_own.min(), d.game_z_own.max()]
a.plot(lim, lim, color="black", ls="--", lw=1)
a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
a.set_xlabel("game_z  own baseline"); a.set_ylabel("game_z  tier baseline")
a.set_title(f"The two baselines  (corr {d.game_z_own.corr(d.game_z_tier):+.3f})")
a.grid(alpha=.2)

a = ax[1]
names = list(BLENDS)
vals = [g_(res[n][1][isf(d)]) for n in names]
a.barh(range(len(names)), vals, color=["#C44E52" if n.startswith("A") else "#4C78A8"
                                       for n in names])
a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=9)
a.invert_yaxis()
a.set_xlabel("geometric-mean rank of the 6 flagged  (lower = better)")
a.set_title("Full pool")
for i, v in enumerate(vals):
    a.text(v, i, f" {v:,.0f}", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "tier_blend.png", dpi=140)
print(f"\n-> {OUT/'tier_blend.png'}")
