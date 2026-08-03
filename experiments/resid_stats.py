"""Raw vs context-adjusted, one panel per stat.

Each panel: the raw stat's own-season z against its context-residualised z. Points on
the diagonal are unchanged by the adjustment; vertical spread is what context removed.
The context R2 in each title is the honest measure of how much the schedule explains.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import db

OUT = pathlib.Path(__file__).resolve().parent / "out"
S = ["game_score","touches","passes","distance","usage_pct","contested_shots",
     "deflections","loose_balls","box_outs","screen_assists","minutes","points"]
rz = ", ".join(f"r.{c}_resid_z" for c in S)
q = f"""SELECT p.full_name AS player, g.game_date, pg.minutes AS min_played,
               r.tier, {rz},
               pg.points, pg.fgm, pg.fga, pg.fta, pg.ftm, pg.rebounds,
               pg.rebounds_off, pg.assists, pg.steals, pg.blocks, pg.turnovers,
               pg.fouls, pg.touches, pg.passes, pg.distance, pg.usage_pct,
               pg.contested_shots, pg.deflections, pg.loose_balls, pg.box_outs,
               pg.screen_assists, pg.player_id, pg.game_id
        FROM player_games pg
        JOIN players p USING (player_id)
        JOIN games g ON g.game_id = pg.game_id
        JOIN player_game_residuals r
             ON r.player_id = pg.player_id AND r.game_id = pg.game_id
        WHERE pg.minutes > 0 AND pg.points IS NOT NULL"""
with db.connect() as c:
    d = pd.read_sql(q, c)
    mods = pd.read_sql("""SELECT stat, tier, n_rows, explained_context
                          FROM residual_models""", c)
for c_ in d.columns:
    if c_ not in ("player","game_date","game_id","tier"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
d["gd"] = d.game_date.astype(str).str[:10]
drb = d.rebounds - d.rebounds_off.fillna(0)
d["game_score"] = (d.points + .4*d.fgm - .7*d.fga - .4*(d.fta-d.ftm)
                   + .7*d.rebounds_off.fillna(0) + .3*drb + d.steals
                   + .7*d.assists + .7*d.blocks - .4*d.fouls - d.turnovers)
d["minutes"] = d.min_played

def own_z(col):
    g = d.groupby("player_id")[col]
    return (d[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
d["flag"] = d.apply(lambda r: r.gd in FLAG.get(r.player, []), axis=1)
r2 = (mods.groupby("stat")
          .apply(lambda x: np.average(x.explained_context.astype(float),
                                      weights=x.n_rows)).to_dict())

fig, ax = plt.subplots(3, 4, figsize=(16, 11))
for a, c_ in zip(ax.ravel(), S):
    raw, res = own_z(c_), d[f"{c_}_resid_z"]
    m = pd.DataFrame({"raw": raw, "res": res, "flag": d.flag}).dropna()
    a.scatter(m.raw, m.res, s=3, alpha=.05, color="#4C72B0", rasterized=True)
    lim = [min(m.raw.min(), m.res.min()), max(m.raw.max(), m.res.max())]
    a.plot(lim, lim, color="grey", ls=":", lw=1)
    a.axhline(0, color="grey", lw=.6); a.axvline(0, color="grey", lw=.6)
    fm = m[m.flag]
    a.scatter(fm.raw, fm.res, s=45, color="crimson", edgecolor="white", zorder=5)
    a.set_title(f"{c_}\ncontext R2 {100*r2.get(c_,np.nan):.1f}%   "
                f"corr {m.raw.corr(m.res):+.3f}", fontsize=9)
    a.set_xlabel("raw own-season z", fontsize=8)
    a.set_ylabel("context-adjusted z", fontsize=8)
    a.tick_params(labelsize=7)
fig.suptitle("Raw vs context-adjusted, per stat.  Dotted line = no change.\n"
             "Red = the six flagged games.  Vertical spread is what context removed.",
             fontsize=12)
fig.tight_layout(rect=[0,0,1,0.945])
fig.savefig(OUT / "resid_stats.png", dpi=130)
print(f"-> {OUT/'resid_stats.png'}")
print(f"\n{'stat':<18}{'context R2':>12}{'corr raw~adj':>14}")
for c_ in S:
    m = pd.DataFrame({"a": own_z(c_), "b": d[f"{c_}_resid_z"]}).dropna()
    print(f"{c_:<18}{100*r2.get(c_,np.nan):>11.1f}%{m.a.corr(m.b):>14.3f}")
