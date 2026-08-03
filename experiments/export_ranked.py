"""Clean ranked export of the equal-weight score.

    score = 0.45*performance + 0.30*market + 0.25*motive

with EQUAL weights inside each block:

    performance = mean(-game_z, -effort_z, shortfall_z)
    market      = mean(-p_price, -p_line, -line_move, -price_only_move)   present-weighted
    motive      = z(1 - salary percentile);  unlisted (two-way) scores maximum

Within-block weights are 1/n because analysis/weight_audit.py showed unequal ones do
nothing: zeroing or doubling any of them moved the flagged games by 0.02-0.09 in mean
log10 rank against 0.58 for the motive block weight, and 48% of random weight vectors
beat the previously tuned ones.

Two rankings are written, and they answer different questions:

    ranked_all.csv   all 15,498 propped player-games, ranked by score.
    ranked_cut.csv   the 2,000 survivors of the cut funnel, ranked the same way.
                     `rank_all` is carried across so the effect of the gates is visible.

    python analysis/export_ranked.py
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from weight_audit import live_weights

PERF_W, MARKET_W, BLOCK_W = live_weights()

COLS = ["rank", "player", "gd", "minutes", "points", "close_line", "shortfall",
        "score", "score_pct", "performance", "market", "motive",
        "game_z", "effort_z", "shortfall_z", "under_price", "line_move_pct",
        "price_only_move", "salary", "has_listed_salary", "tier",
        "under_hit", "player_id", "game_id"]

FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}


def fmt(df):
    x = df.copy()
    x["salary"] = x.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
    return x


d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
d = d[[c for c in COLS if c in d.columns]].sort_values("rank")
d.to_csv(OUT / "ranked_all.csv", index=False)

c = pd.read_csv(OUT / "combined_cut.csv", dtype={"game_id": str})
c = c.merge(d[["player_id", "game_id", "rank"]].rename(columns={"rank": "rank_all"}),
            on=["player_id", "game_id"], how="left")
keep = ["rank", "rank_all", "player", "gd", "minutes", "points", "line", "shortfall",
        "score", "performance", "market", "motive", "game_z", "effort_z",
        "salary", "under_hit", "player_id", "game_id"]
c = c[[k for k in keep if k in c.columns]].sort_values("rank")
c.to_csv(OUT / "ranked_cut.csv", index=False)

print("EQUAL-WEIGHT SCORE")
print(f"   block weights   " + "   ".join(f"{k} {v}" for k, v in BLOCK_W.items()))
print(f"   performance     " + "   ".join(f"{k} {v:.3f}" for k, v in PERF_W.items()))
print(f"   market          " + "   ".join(f"{k} {v:.2f}" for k, v in MARKET_W.items()))
print(f"\n   ranked_all.csv  {len(d):,} propped player-games")
print(f"   ranked_cut.csv  {len(c):,} survivors of the cut funnel")

show = ["rank", "player", "gd", "minutes", "points", "close_line", "shortfall",
        "score", "performance", "market", "motive", "salary"]
print(f"\nTOP 40 -- ALL PROPPED GAMES")
print(fmt(d.head(40))[show].to_string(index=False))

show_c = ["rank", "rank_all", "player", "gd", "minutes", "points", "line",
          "shortfall", "score", "performance", "market", "motive", "salary"]
print(f"\nTOP 25 -- AFTER THE CUTS")
print(fmt(c.head(25))[show_c].to_string(index=False))

f = d[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in d.iterrows()]]
fc = c[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in c.iterrows()]]
print(f"\nFLAGGED GAMES")
j = f.merge(fc[["player_id", "game_id", "rank"]].rename(columns={"rank": "rank_cut"}),
            on=["player_id", "game_id"], how="left")
j["rank_cut"] = j.rank_cut.map(lambda v: "CUT" if pd.isna(v) else f"{int(v):,}")
print(fmt(j.sort_values("rank"))[
    ["rank", "rank_cut", "player", "gd", "minutes", "points", "close_line",
     "shortfall", "score", "performance", "market", "motive"]].to_string(index=False))

print(f"\n-> {OUT/'ranked_all.csv'}\n-> {OUT/'ranked_cut.csv'}")
