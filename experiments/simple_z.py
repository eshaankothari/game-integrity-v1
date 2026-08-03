import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import db
from sklearn.linear_model import LinearRegression

OUT = pathlib.Path("/Users/eshaankothari/Desktop/game-integrity-v1/experiments/out")
q = """SELECT p.full_name AS player, g.game_date, pg.minutes, pg.points,
              pg.fgm, pg.fga, pg.fta, pg.ftm, pg.rebounds, pg.rebounds_off,
              pg.assists, pg.steals, pg.blocks, pg.turnovers, pg.fouls,
              pg.usage_pct, pg.touches, pg.passes, pg.distance,
              pg.contested_shots, pg.deflections, pg.loose_balls,
              pg.box_outs, pg.screen_assists, pg.charges_drawn,
              pg.player_id, pg.game_id
       FROM player_games pg JOIN players p USING (player_id)
       JOIN games g ON g.game_id = pg.game_id
       WHERE pg.minutes > 0 AND pg.points IS NOT NULL"""
with db.connect() as c: d = pd.read_sql(q, c)
for c_ in d.columns:
    if c_ not in ("player","game_date","game_id"): d[c_] = pd.to_numeric(d[c_], errors="coerce")
d["gd"] = d.game_date.astype(str).str[:10]

drb = d.rebounds - d.rebounds_off.fillna(0)
d["game_score"] = (d.points + .4*d.fgm - .7*d.fga - .4*(d.fta-d.ftm)
                   + .7*d.rebounds_off.fillna(0) + .3*drb + d.steals
                   + .7*d.assists + .7*d.blocks - .4*d.fouls - d.turnovers)

# fga is DELIBERATELY absent: Game Score already charges -0.7 per attempt, so
# including it here would count the same behaviour on both axes. Same reasoning
# excludes points, rebounds, assists, steals, blocks, turnovers and fouls.
#
# EFFORT is now everything about INVOLVEMENT and EXERTION -- how much of the offense
# ran through him and how much work he did -- with nothing that scores.
EFFORT = ["touches","passes","usage_pct","distance",
          "contested_shots","deflections","loose_balls","box_outs","screen_assists"]

# MINUTES ADJUSTMENT, FITTED SEPARATELY FOR EACH STAT.
#
# Not once on the finished composite. Minutes explain wildly different shares of each
# stat -- distance R2 = 0.98, touches 0.74, box_outs 0.03, screen_assists 0.03 -- so a
# single correction applied to the average would under-correct the stats that are
# nearly all minutes and over-correct the ones that are not. Averaging first destroys
# the information about which is which.
#
# Measured: correlation of the effort score with production falls from 0.562 raw to
# 0.140 adjusting per stat, but only to 0.464 adjusting the composite once.
#
# The intercept is KEPT. Forcing the line through the origin is physically tidier --
# zero minutes should predict zero touches -- but it over-steepens the fit and leaves
# corr(effort, minutes) at -0.183 against -0.083 with the intercept. Nobody plays zero
# minutes, so the fit is better serving the range the data occupies.
# MINUTES IS A MEDIATOR, NOT A CONFOUNDER, which is why this is OFF by default.
#
#     blowout    -> fewer minutes -> fewer touches      confound, worth removing
#     disengaged -> benched       -> fewer touches      SIGNAL, must be kept
#
# Both paths run through minutes, so adjusting for it removes the behaviour being
# looked for along with the innocent explanation. If a player withdrew from the game,
# low minutes IS part of the evidence rather than a nuisance to strip.
#
# The cost of leaving it off is real and should be stated: effort_z then correlates
# with production at 0.562 instead of 0.140, because both are partly measuring
# playing time. A short appearance scores low on BOTH axes by construction, so a
# genuine injury and a deliberate withdrawal look identical here.
ADJUST_MINUTES = False

_MIN = d[["minutes"]].fillna(d.minutes.median()).values

def minutes_adj(col):
    ok = d[[col, "minutes"]].dropna()
    m = LinearRegression().fit(ok[["minutes"]].values, ok[col].values)
    return d[col] - m.predict(_MIN)

# z against the PLAYER'S OWN season, applied to the minutes-adjusted value.
def own_z(col, adjust=False):
    v = minutes_adj(col) if adjust else d[col]
    g = v.groupby(d["player_id"])
    return (v - g.transform("mean")) / g.transform("std").replace(0, np.nan)

d["game_z"]   = own_z("game_score").round(3)
# MISSING COMPONENTS BECOME 0, not skipped.
#
# The five hustle stats vanish as a block -- BoxScoreHustleV2 either returned a game
# or it did not -- and those 1,397 rows average 4.2 minutes against 24.1 for complete
# ones. Skipping them left the mean over only touches/passes/usage/distance, all of
# which scale hard with minutes, so incomplete rows scored -0.75 against +0.02 and
# supplied 330 of the 500 worst effort scores from 5.3 percent of the data.
#
# A z of 0 means "average for him", which is the honest reading of an absent
# measurement. It shrinks an incomplete row toward the middle instead of letting four
# minutes-driven components speak for nine. Contamination of the worst 500 falls from
# 330 to 16.
_ez = pd.concat([own_z(c, adjust=ADJUST_MINUTES) for c in EFFORT], axis=1)
d["n_effort"] = _ez.notna().sum(axis=1)
d["effort_z"] = _ez.fillna(0).mean(axis=1).round(3)

n = d.groupby("player_id")["points"].transform("size")
d.loc[n < 15, ["game_z","effort_z"]] = np.nan
m = d.dropna(subset=["game_z","effort_z"])
m.to_csv(OUT / "simple_z.csv", index=False)

FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
f = m[m.apply(lambda r: r.gd in FLAG.get(r.player, []), axis=1)]

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for a, (yc, yl) in zip(ax, [("effort_z", "effort z  (touches, passes, usage, distance, hustle)"),
                            ("n_effort", "components present")]):
    if yc == "n_effort":
        a.hist(m.n_effort, bins=range(0, 11), color="#55A868", edgecolor="white")
        a.set_xlabel("components present of 9"); a.set_ylabel("player-games")
        a.set_title("How many effort components each row had")
        continue
    a.scatter(m.game_z, m[yc], s=4, alpha=.08, color="#4C72B0", rasterized=True)
    a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
    r = m.game_z.corr(m[yc])
    b = np.polyfit(m.game_z, m[yc], 1)
    xs = np.linspace(m.game_z.min(), m.game_z.max(), 50)
    a.plot(xs, np.polyval(b, xs), color="black", ls="--", lw=1.2, label=f"r = {r:+.3f}")
    a.scatter(f.game_z, f[yc], s=95, color="crimson", edgecolor="white", zorder=5)
    for _, row in f.iterrows():
        a.annotate(f"{row.player.split()[1][:3]} {row.gd[5:]}", (row.game_z, row[yc]),
                   textcoords="offset points", xytext=(7, 4), color="crimson", fontsize=8)
    a.axhspan(a.get_ylim()[0], 0, xmin=0, xmax=.5, color="crimson", alpha=.04)
    a.set_xlabel("game score z  (production)"); a.set_ylabel(yl)
    a.set_title(f"PRODUCTION vs EFFORT   (n={len(m):,})")
    a.legend(fontsize=9); a.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT / "simple_z.png", dpi=140)
print(f"-> {OUT/'simple_z.csv'}  ({len(m):,} rows)")
print(f"-> {OUT/'simple_z.png'}")
print(f"\ncorr(game_z, effort_z) = {m.game_z.corr(m.effort_z):+.3f}")
print(f"\nFLAGGED:")
print(f[["player","gd","minutes","points","game_score","game_z","effort_z","n_effort"]].to_string(index=False))
