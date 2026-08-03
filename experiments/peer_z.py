"""Peer-matched z: compare a game to similar players in a similar minutes range.

THREE BASELINES NOW EXIST, answering different questions:

    own     vs the player's own season      "unlike HIM"
    league  vs all 26,393 games             "unlike anyone"
    PEER    vs same tier x same minutes     "unlike a player in his role having a
                                             night of this length"

WHY PEER. The score correlates -0.568 with minutes, and mean score runs monotonically
from +0.842 below 5 minutes to -0.258 above 30. Own-season z cannot fix that because a
short night IS unlike the player; league z cannot because it compares a 12-minute
bench appearance to a 36-minute starter's. Matching on both removes the comparison
that was never fair in the first place.

WHAT IT COSTS. Minutes stop being evidence. If a player withdrew, low minutes is part
of the behaviour -- the mediator argument -- and this baseline explicitly conditions
that away. It answers "was he bad FOR A GAME THIS LENGTH", which is a narrower and
cleaner question than the one the raw score asks.

THIN CELLS fall back to the minutes band alone. tier is defined by season-average
minutes, so the two are entangled by construction: starters have 21 games under 8
minutes and bench players 71 above 32.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import db, config

OUT = pathlib.Path(__file__).resolve().parent / "out"
MIN_CELL = 100
E = ["touches","passes","usage_pct","distance","contested_shots","deflections",
     "loose_balls","box_outs","screen_assists"]

q = f"""SELECT p.full_name AS player, g.game_date, pg.minutes, pg.points,
        pg.fgm, pg.fga, pg.fta, pg.ftm, pg.rebounds, pg.rebounds_off,
        pg.assists, pg.steals, pg.blocks, pg.turnovers, pg.fouls,
        {','.join('pg.'+c for c in E)}, r.tier, pg.player_id, pg.game_id
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        JOIN games   g ON g.game_id  = pg.game_id
        LEFT JOIN player_game_residuals r
             ON r.player_id = pg.player_id AND r.game_id = pg.game_id
        WHERE pg.minutes>0 AND pg.points IS NOT NULL"""
with db.connect() as c: d = pd.read_sql(q, c)
for c_ in d.columns:
    if c_ not in ("player","game_date","game_id","tier"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
d["gd"] = d.game_date.astype(str).str[:10]
drb = d.rebounds - d.rebounds_off.fillna(0)
d["game_score"] = (d.points + .4*d.fgm - .7*d.fga - .4*(d.fta-d.ftm)
                   + .7*d.rebounds_off.fillna(0) + .3*drb + d.steals
                   + .7*d.assists + .7*d.blocks - .4*d.fouls - d.turnovers)
d["mband"] = pd.cut(d.minutes, [0,8,12,16,20,24,28,32,60],
                    labels=["<8","8-12","12-16","16-20","20-24","24-28","28-32","32+"])
d["cell"] = d.tier.astype(str) + "|" + d.mband.astype(str)
big = d.cell.value_counts(); d["cell_n"] = d.cell.map(big)
d["peer"] = np.where(d.cell_n >= MIN_CELL, d.cell, "band|" + d.mband.astype(str))

def zby(col, key):
    g = d.groupby(key)[col]
    return (d[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

# THREE peer definitions, each conditioning on a different thing:
#   own   the player himself          keeps minutes as evidence, but a short night IS
#                                     unlike him so cameos score high automatically
#   tier  his ROLE only               removes the bench-vs-star comparison while
#                                     leaving minutes free to carry signal
#   peer  role x minutes band         also removes minutes, which fixes the cameo
#                                     problem and conditions away withdrawal
for c_ in ["game_score"] + E:
    d[f"{c_}_own"]  = zby(c_, "player_id")
    d[f"{c_}_tier"] = zby(c_, "tier")
    d[f"{c_}_peer"] = zby(c_, "peer")

n = d.groupby("player_id").points.transform("size")
for suf in ("own","tier","peer"):
    d[f"game_z_{suf}"] = d[f"game_score_{suf}"]
    d[f"effort_z_{suf}"] = d[[f"{c_}_{suf}" for c_ in E]].fillna(0).mean(axis=1)
    d.loc[n < 15, f"game_z_{suf}"] = np.where(suf=="own", np.nan, d[f"game_z_{suf}"])[n<15]

tm = pd.read_csv(OUT/"two_metrics.csv", dtype={"game_id":str})[
    ["player_id","game_id","shortfall","shortfall_z","market","motive",
     "close_line","salary","has_listed_salary","score"]]
d = d.merge(tm, on=["player_id","game_id"], how="inner")

for suf in ("own","tier","peer"):
    d[f"perf_{suf}"] = (0.45*(-d[f"game_z_{suf}"]) + 0.20*(-d[f"effort_z_{suf}"])
                        + 0.35*d.shortfall_z)
    d[f"score_{suf}"] = 0.45*d[f"perf_{suf}"] + 0.30*d.market + 0.25*d.motive
    d[f"rank_{suf}"] = d[f"score_{suf}"].rank(ascending=False, method="min")

print(f"propped games {len(d):,}   peer cells {d.peer.nunique()}"
      f"   fell back to band-only: {int((d.cell_n < MIN_CELL).sum()):,}\n")
print("MINUTES BIAS -- the thing this is meant to fix:")
for suf in ("own","tier","peer"):
    print(f"   {suf:<5} corr(score, minutes) {d[f'score_{suf}'].corr(d.minutes):+.3f}"
          f"   corr(game_z, minutes) {d[f'game_z_{suf}'].corr(d.minutes):+.3f}"
          f"   corr(effort_z, minutes) {d[f'effort_z_{suf}'].corr(d.minutes):+.3f}")
print(f"\n   corr(score_own, score_peer) {d.score_own.corr(d.score_peer):+.3f}")
print(f"   corr(game_z_own, game_z_peer) {d.game_z_own.corr(d.game_z_peer):+.3f}")

print(f"\nTOP 200 PROFILE:")
for suf in ("own","tier","peer"):
    t = d.nlargest(200, f"score_{suf}")
    print(f"   {suf:<5} avg min {t.minutes.mean():>5.1f}   <10min {100*(t.minutes<10).mean():>4.0f}%"
          f"   median salary ${t.salary.median()/1e6:>5.1f}M   avg line {t.close_line.mean():>5.1f}")

FLAG={"Malik Beasley":["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
      "Jontay Porter":["2024-01-20","2024-03-20"]}
f = d[d.apply(lambda r: r.gd in FLAG.get(r.player,[]),axis=1)]
print(f"\nFLAGGED:")
print(f"{'game':<20}{'min':>6}{'tier':>9}{'gz_own':>8}{'gz_tier':>9}{'gz_peer':>9}"
      f"{'r_own':>8}{'r_tier':>8}{'r_peer':>8}")
for _, x in f.sort_values("rank_own").iterrows():
    print(f"{x.player.split()[1]+' '+x.gd[5:]:<20}{x.minutes:>6.1f}{x.tier:>9}"
          f"{x.game_z_own:>8.2f}{x.game_z_tier:>9.2f}{x.game_z_peer:>9.2f}"
          f"{int(x.rank_own):>8,}{int(x.rank_tier):>8,}{int(x.rank_peer):>8,}")
d.to_csv(OUT/"peer_z.csv", index=False)
print(f"\n-> {OUT/'peer_z.csv'}")
