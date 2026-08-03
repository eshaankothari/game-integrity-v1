"""Early-exit indicator from play-by-play stints. Approximate, and honest about it.

WHAT THIS REPLACES. `player_game_pbp.returned_after_last_exit` is NULL on all 26,709
rows -- the field was declared but never computed -- so the direct "left and did not
come back" answer does not exist. This reconstructs an approximation from
`last_out_sec`, the moment of the player's final substitution off the floor.

THE MEASURE IS RELATIVE TO HIMSELF, not to a fixed clock. A bench player finishing
his rotation at minute 30 and a starter limping off at minute 30 look identical in
absolute terms; what separates them is whether THIS exit was early FOR HIM.

    exit_frac  = last_out_sec / game_length      share of the game elapsed at his
                                                 final exit, so overtime is handled
    exit_z     = (exit_frac - his mean) / his sd
    left_live  = last_out_sec < competitive_sec  off the floor while the game still
                                                 mattered

WHAT IT CANNOT DO. It cannot tell an injury from a benching from a withdrawal -- all
three produce an early final exit. It narrows the population; it does not explain it.
`last_out_sec` is also NULL for 3,064 of 25,841 played games (11.9 percent), which are
left as NaN rather than imputed.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import db, config

OUT = pathlib.Path(__file__).resolve().parent / "out"
q = """
SELECT p.full_name AS player, g.game_date, pg.minutes, pg.points,
       b.last_out_sec, b.first_in_sec, b.n_stints, b.ejected,
       c.n_periods, c.competitive_sec, c.garbage_start_sec, c.final_margin,
       f.close_line, s.salary,
       pg.player_id, pg.game_id
FROM player_games pg
JOIN players p ON p.player_id = pg.player_id
JOIN games   g ON g.game_id  = pg.game_id
LEFT JOIN player_game_pbp   b ON b.player_id = pg.player_id AND b.game_id = pg.game_id
LEFT JOIN game_pbp_context  c ON c.game_id = pg.game_id
LEFT JOIN player_game_features f ON f.player_id = pg.player_id AND f.game_id = pg.game_id
LEFT JOIN player_salaries s ON s.player_id = pg.player_id AND s.season = %(season)s
WHERE pg.minutes > 0
"""
with db.connect() as c: d = pd.read_sql(q, c, params={"season": config.SEASON})
for c_ in ("minutes","points","last_out_sec","first_in_sec","n_stints",
           "n_periods","competitive_sec","garbage_start_sec","close_line"):
    d[c_] = pd.to_numeric(d[c_], errors="coerce")
d["gd"] = d.game_date.astype(str).str[:10]

# Regulation is 2,880 seconds; each overtime adds 300.
d["game_len"] = 2880 + 300 * (d.n_periods - 4)
d["exit_frac"] = (d.last_out_sec / d.game_len).round(4)
d["left_live"] = d.last_out_sec < d.competitive_sec

g = d.groupby("player_id")["exit_frac"]
d["exit_frac_mean"] = g.transform("mean").round(4)
d["exit_z"] = ((d.exit_frac - g.transform("mean"))
               / g.transform("std").replace(0, np.nan)).round(3)
n = d.groupby("player_id")["points"].transform("size")
d.loc[n < 15, "exit_z"] = np.nan
d["n_games"] = n

print(f"played games                     : {len(d):,}")
print(f"  with last_out_sec              : {int(d.last_out_sec.notna().sum()):,}"
      f"   ({100*d.last_out_sec.notna().mean():.1f}%)")
print(f"  left while game still live     : {int(d.left_live.sum()):,}")
print(f"\nexit_frac distribution: median {d.exit_frac.median():.3f}   "
      f"p10 {d.exit_frac.quantile(.1):.3f}   p90 {d.exit_frac.quantile(.9):.3f}")
print(f"exit_z  distribution : sd {d.exit_z.std():.3f}   "
      f"p5 {d.exit_z.quantile(.05):.2f}   p95 {d.exit_z.quantile(.95):.2f}")

b = pd.cut(d.exit_z, [-9,-2,-1,1,9], labels=["<-2 (very early)","-2..-1","-1..+1",">+1"])
t = d.groupby(b).agg(rows=("minutes","size"), avg_min=("minutes","mean"),
                     avg_pts=("points","mean"), avg_stints=("n_stints","mean"),
                     left_live=("left_live","mean"))
t["avg_min"]=t.avg_min.round(1); t["avg_pts"]=t.avg_pts.round(1)
t["avg_stints"]=t.avg_stints.round(2); t["left_live"]=(100*t.left_live).round(1)
print(f"\n{t.to_string()}")
d.to_csv(OUT / "early_exit.csv", index=False)
print(f"\n-> {OUT/'early_exit.csv'}")
