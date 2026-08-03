"""League-wide distribution of Hollinger Game Score.

    PTS + .4*FGM - .7*FGA - .4*(FTA-FTM) + .7*ORB + .3*DRB
        + STL + .7*AST + .7*BLK - .4*PF - TOV

All 11 inputs are present on all 26,393 played games -- they come from the traditional
box score, which never suffered the tracking-feed outages that hit distance and touches.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import db

OUT = pathlib.Path(__file__).resolve().parent / "out"
q = """SELECT p.full_name AS player, pg.minutes, pg.points, pg.fgm, pg.fga, pg.fta,
              pg.ftm, pg.rebounds, pg.rebounds_off, pg.assists, pg.steals,
              pg.blocks, pg.turnovers, pg.fouls, r.tier, pg.player_id
       FROM player_games pg JOIN players p USING (player_id)
       LEFT JOIN player_game_residuals r USING (player_id, game_id)
       WHERE pg.minutes > 0 AND pg.points IS NOT NULL"""
with db.connect() as c: d = pd.read_sql(q, c)
for c_ in d.columns:
    if c_ not in ("player", "tier"): d[c_] = pd.to_numeric(d[c_], errors="coerce")
drb = d.rebounds - d.rebounds_off.fillna(0)
d["gs"] = (d.points + .4*d.fgm - .7*d.fga - .4*(d.fta-d.ftm)
           + .7*d.rebounds_off.fillna(0) + .3*drb + d.steals
           + .7*d.assists + .7*d.blocks - .4*d.fouls - d.turnovers)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
gs = d.gs

a = ax[0, 0]
a.hist(gs, bins=90, color="#4C72B0", edgecolor="white")
xs = np.linspace(gs.min(), gs.max(), 400)
a.plot(xs, stats.norm.pdf(xs, gs.mean(), gs.std()) * len(gs) * (gs.max()-gs.min())/90,
       color="crimson", lw=1.6, label=f"normal({gs.mean():.1f}, {gs.std():.1f})")
a.axvline(gs.median(), color="black", ls="--", lw=1.2, label=f"median {gs.median():.1f}")
a.set_xlabel("game score"); a.set_ylabel("player-games")
a.set_title(f"League-wide  (n={len(gs):,}, skew {gs.skew():+.2f}, "
            f"kurt {gs.kurt():+.2f})")
a.legend(fontsize=8)

# Split by role tier -- the reason a league-wide z is not usable across players.
a = ax[0, 1]
for t, c in (("bench", "#55A868"), ("rotation", "#4C72B0"), ("starter", "#C44E52")):
    s = d[d.tier == t].gs
    if len(s):
        a.hist(s, bins=70, histtype="step", lw=1.7, color=c, density=True,
               label=f"{t}  n={len(s):,}  med {s.median():.1f}")
a.axvline(0, color="grey", lw=.8)
a.set_xlabel("game score"); a.set_ylabel("density")
a.set_title("By role tier"); a.legend(fontsize=8)

# Game score against minutes -- how much of it is just playing time.
a = ax[1, 0]
a.scatter(d.minutes, gs, s=3, alpha=.05, color="#4C72B0", rasterized=True)
b = np.polyfit(d.minutes, gs, 1)
xs = np.linspace(0, d.minutes.max(), 50)
a.plot(xs, np.polyval(b, xs), color="crimson", lw=1.8,
       label=f"slope {b[0]:+.3f}/min   r = {d.minutes.corr(gs):+.3f}")
a.axhline(0, color="grey", lw=.8)
a.set_xlabel("minutes"); a.set_ylabel("game score")
a.set_title("Game score vs minutes"); a.legend(fontsize=8)

# The left tail, which is what the whole project is looking at.
a = ax[1, 1]
lo = gs[gs <= gs.quantile(.10)]
a.hist(lo, bins=50, color="#C44E52", edgecolor="white")
for q_, col in ((.01, "black"), (.05, "darkorange")):
    a.axvline(gs.quantile(q_), color=col, ls="--", lw=1.3,
              label=f"{q_:.0%} = {gs.quantile(q_):.1f}")
a.set_xlabel("game score"); a.set_ylabel("player-games")
a.set_title(f"The bottom decile  (n={len(lo):,})"); a.legend(fontsize=8)

fig.tight_layout(); fig.savefig(OUT / "gs_league.png", dpi=140)
print(f"-> {OUT/'gs_league.png'}")
print(f"\nn {len(gs):,}   mean {gs.mean():.2f}   median {gs.median():.2f}   "
      f"sd {gs.std():.2f}")
print(f"skew {gs.skew():+.2f}   kurtosis {gs.kurt():+.2f}")
print(f"min {gs.min():.1f}   max {gs.max():.1f}   negative: "
      f"{int((gs<0).sum()):,} ({100*(gs<0).mean():.1f}%)")
print("\npercentiles:")
print("  " + "  ".join(f"{int(q*100)}%={gs.quantile(q):.1f}"
                       for q in (.01,.05,.10,.25,.50,.75,.90,.95,.99)))
print("\nby tier:")
print(d.groupby("tier").gs.agg(["size","mean","std","min","median","max"]).round(2).to_string())
