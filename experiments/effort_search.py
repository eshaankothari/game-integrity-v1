"""Exhaustive search: which subset of the 9 effort stats ranks the flagged games best?

511 non-empty subsets against 6 labels from 2 players. Some subset will look excellent
by chance, so the number that matters is not the winner's score but whether the SAME
subset wins when a label is removed -- the leave-one-out block at the bottom.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import db, config

OUT = pathlib.Path(__file__).resolve().parent / "out"
E = ["touches","passes","usage_pct","distance","contested_shots","deflections",
     "loose_balls","box_outs","screen_assists"]
tm = pd.read_csv(OUT/"two_metrics.csv", dtype={"game_id":str}).reset_index(drop=True)

q = f"""SELECT pg.player_id, pg.game_id, {','.join('pg.'+c for c in E)}
        FROM player_games pg WHERE pg.minutes>0 AND pg.points IS NOT NULL"""
with db.connect() as c: raw = pd.read_sql(q, c)
raw["game_id"] = raw.game_id.astype(str)
for c_ in E: raw[c_] = pd.to_numeric(raw[c_], errors="coerce")

# own-season z for each stat, computed on ALL played games, then aligned to the
# propped population -- same baseline discipline as everywhere else.
Zfull = {}
for c_ in E:
    g = raw.groupby("player_id")[c_]
    Zfull[c_] = (raw[c_]-g.transform("mean"))/g.transform("std").replace(0, np.nan)
Z = pd.DataFrame(Zfull); Z["player_id"]=raw.player_id; Z["game_id"]=raw.game_id
d = tm[["player_id","game_id","player","gd","game_z","shortfall_z","market","motive"]] \
        .merge(Z, on=["player_id","game_id"], how="left")

FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
d["flag"] = d.apply(lambda r: r.gd in FLAG.get(r.player, []), axis=1)
idx = np.flatnonzero(d.flag.values)
labels = [f"{d.player[i].split()[1][:3]} {d.gd[i][5:]}" for i in idx]

def ranks_for(subset, drop=None):
    eff = d[list(subset)].fillna(0).mean(axis=1)
    perf = 0.45*(-d.game_z) + 0.20*(-eff) + 0.35*d.shortfall_z
    score = 0.45*perf + 0.30*d.market + 0.25*d.motive
    r = score.rank(ascending=False, method="min")
    keep = [i for i in idx if i != drop]
    return r.values[keep]

best = []
for k in range(1, len(E)+1):
    for sub in itertools.combinations(E, k):
        r = ranks_for(sub)
        best.append((np.exp(np.mean(np.log(r))), np.median(r), sub))   # geometric mean
best.sort()
print(f"searched {len(best)} subsets against {len(idx)} labels\n")
print(f"{'geo-mean rank':>14}{'median':>9}  subset")
print("-"*78)
for g_, m_, sub in best[:8]:
    print(f"{g_:>14,.0f}{m_:>9,.0f}  {', '.join(sub)}")
print("   ...")
for g_, m_, sub in best[-3:]:
    print(f"{g_:>14,.0f}{m_:>9,.0f}  {', '.join(sub)}")
cur = tuple(E)
cg = [b for b in best if b[2]==cur][0]
print(f"\ncurrent 9-stat set: geo-mean {cg[0]:,.0f}  "
      f"(rank {best.index(cg)+1} of {len(best)})")

print(f"\nLEAVE-ONE-LABEL-OUT: hide one flagged game, re-run the whole search.")
print(f"If the winner is real, the same subset keeps winning.\n")
from collections import Counter
wins = []
for i in idx:
    res = []
    for k in range(1, len(E)+1):
        for sub in itertools.combinations(E, k):
            r = ranks_for(sub, drop=i)
            res.append((np.exp(np.mean(np.log(r))), sub))
    res.sort()
    wins.append(res[0][1])
    lab = f"{d.player[i].split()[1][:3]} {d.gd[i][5:]}"
    print(f"   hide {lab:<12} -> {', '.join(res[0][1])}")
cnt = Counter(s for w in wins for s in w)
print(f"\n   stat appears in the winning subset, out of {len(wins)} folds:")
for s_, n_ in cnt.most_common():
    print(f"      {s_:<18}{n_}/{len(wins)}")
