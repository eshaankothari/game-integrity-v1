"""game_z, effort_z and a MARKET composite on one figure.

MARKET COMPOSITE. Two market facts, combined:

    p_under   de-vigged closing probability the under hits,
              (1/under) / (1/under + 1/over), z-scored league-wide so it is on the
              same scale as the two performance axes. Positive = market leaned under.
    p_line    percentile of the line among all closing lines. Small line = low.

    market_z  = mean of  z(p_under)  and  z(smallness of the line)

RAW PRICE WOULD NOT WORK HERE. FanDuel's overround is a flat 1.049 at every line size,
so the raw under price is mostly margin; de-vigged, the market prices unders at
0.499-0.502 whatever the line. The de-vigged version is also the strongest signal
measured in this project -- monotonic across five buckets, z = 3.81, against 2.11 for
under_move_pct -- and needs only the two closing prices, so coverage is 100 percent.

MEAN, NOT PRODUCT, for the two market components. A product assumes independence and
compounds; these two are mechanically linked by the half-point grid, since a book that
cannot shade a 5.5 line to 5.75 must express its view in the price instead.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import db, config

OUT = pathlib.Path(__file__).resolve().parent / "out"
sz = pd.read_csv(OUT / "simple_z.csv", dtype={"game_id": str})
jl = pd.read_csv(OUT / "joint_low.csv", dtype={"game_id": str})[
    ["player_id", "game_id", "line", "p_under", "p_line", "under_price"]]
d = sz.merge(jl, on=["player_id", "game_id"], how="inner")

# ---- MARKET COMPOSITE -------------------------------------------------------
# Four market facts, each z-scored and oriented so HIGHER = more under-leaning,
# then combined with explicit weights.
#
#   p_price   percentile of the closing under price. Low price = heavily backed
#             under, so it is flipped. No de-vigging: FanDuel's overround is a flat
#             1.049 everywhere, so at the extremes the raw and de-vigged orderings
#             agree and only the crowded middle differs.
#   p_line    percentile of the line. Small line flipped to positive.
#   line_mv   -line_move_pct. Positive = the line drifted DOWN.
#   price_mv  -price_only_move. Positive = the under price SHORTENED while the line
#             held, i.e. the book repriced without conceding a new number.
#
# WEIGHTS REFLECT WHAT WAS MEASURED, not what sounds plausible:
#
#   closing price   monotonic across five buckets, z = 3.81   -> 3
#   line size       real but partly a proxy for player type   -> 1
#   line movement   NULL and in fact backwards: a line falling >10 pct went under
#                   47.1 pct against a 52.7 pct flat baseline -> 1
#   price-only move NULL, non-monotonic (54.0/52.4/51.9/53.8) -> 1
#
# The two movement terms are missing for roughly half the rows -- no opening line was
# posted -- so they are filled with 0, which after z-scoring means "no opinion" and
# neither promotes nor demotes the row.
# EQUAL WEIGHTS. The earlier 3/1/1/1 was justified by which components predicted
# `under_hit`, and that referee was wrong: under_hit is the OUTCOME the market was
# pricing, so agreeing with it measures market efficiency, not money flow. The thing
# being detected -- volume arriving on the under -- is unobservable here, so there is
# no evidence basis for unequal weights. Equal is the honest default.
WEIGHTS = {"p_price": 1.0, "p_line": 1.0, "line_mv": 1.0, "price_mv": 1.0}

with db.connect() as c:
    mv = pd.read_sql("""SELECT player_id, game_id, line_move_pct, price_only_move
                        FROM player_game_features""", c)
mv["game_id"] = mv.game_id.astype(str)
d = d.merge(mv, on=["player_id", "game_id"], how="left")

z = lambda s: (s - s.mean()) / s.std()
d["p_price"] = d.under_price.rank(pct=True).round(4)
parts = {
    "p_price":  z(1 - d.p_price),
    "p_line":   z(1 - d.p_line),
    "line_mv":  z(-pd.to_numeric(d.line_move_pct, errors="coerce")).fillna(0),
    "price_mv": z(-pd.to_numeric(d.price_only_move, errors="coerce")).fillna(0),
}
for k, v in parts.items():
    d[f"mz_{k}"] = v.round(3)
# NORMALISE BY THE WEIGHT ACTUALLY PRESENT, not the full total. line_mv is real on
# 52 percent of rows and price_mv on 28 percent; dividing by the full 6 anyway
# compressed every incomplete row toward zero by a third for a reason that has
# nothing to do with that game -- it shrank the observed range from +/-3.0 to +/-1.6.
present = {k: (d.line_move_pct.notna() if k == "line_mv" else
               d.price_only_move.notna() if k == "price_mv" else
               pd.Series(True, index=d.index)) for k in WEIGHTS}
num = sum(WEIGHTS[k] * parts[k] * present[k] for k in WEIGHTS)
den = sum(WEIGHTS[k] * present[k].astype(float) for k in WEIGHTS)
d["market_z"] = (num / den.replace(0, np.nan)).round(3)
d["n_market"] = sum(present[k].astype(int) for k in WEIGHTS)

# ---- COMBINED SCORE ---------------------------------------------------------
# game_z and effort_z correlate at +0.53 -- they are two views of the same night --
# so they are AVERAGED into one performance term rather than counted twice. market_z
# is independent of both (-0.01, -0.02), so it enters as its own term.
#
# Averaging within a correlated block and combining across independent blocks is the
# rule that avoids the 10x rarity inflation a straight product would produce on
# correlated inputs.
d["perf_z"] = (-(d.game_z + d.effort_z) / 2).round(3)     # higher = worse night

# ---- SHORTFALL, the third block --------------------------------------------
#     shortfall = 1 - points/line,  clipped to [0, 1]
#
# It measures the same night as perf_z but against a DIFFERENT reference: the market's
# forecast rather than the player's own history. That distinction is why it earns its
# own term. Every z-score here is bounded by how much a player normally scores -- a
# 4-point-per-game player who scores 0 is only -0.97 sd from himself while a star is
# -3.23 -- so a z-threshold is secretly a threshold on scoring level. Dividing by the
# line has no such floor: a zero-point game is 1.00 for anyone.
#
# Z-SCORED BEFORE COMBINING. shortfall lives on [0,1] with mean ~0.35, while perf_z and
# market_z are already standardised; adding it raw would let its smaller spread be
# outvoted regardless of how extreme a row was.
d["shortfall"] = (1 - d.points / d.line.replace(0, np.nan)).clip(0, 1).round(3)
d["shortfall_z"] = z(d.shortfall).round(3)

# corr(perf_z, shortfall_z) is checked at run time -- they measure the same night, so
# if it is high they belong in one block rather than two.
d["score"] = ((d.perf_z + d.shortfall_z + d.market_z) / 3).round(3)
d["score_pct"] = d.score.rank(pct=True).round(4)

d["under_hit"] = d.points < d.line
d.to_csv(OUT / "three_axis.csv", index=False)

FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
f = d[d.apply(lambda r: r.gd in FLAG.get(r.player, []), axis=1)]

fig = plt.figure(figsize=(19, 10.5))
# 1: the performance plane, coloured by the market axis
a = fig.add_subplot(2, 4, 1)
sc = a.scatter(d.game_z, d.effort_z, c=d.market_z, s=5, alpha=.35, cmap="coolwarm",
               vmin=-2, vmax=2, rasterized=True)
plt.colorbar(sc, ax=a, label="market_z  (red = low price, small line, both moving down)")
a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
a.scatter(f.game_z, f.effort_z, s=110, facecolor="none", edgecolor="black", lw=1.8, zorder=5)
for _, r in f.iterrows():
    a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}", (r.game_z, r.effort_z),
               textcoords="offset points", xytext=(7,4), fontsize=8)
a.set_xlabel("game_z (production)"); a.set_ylabel("effort_z")
a.set_title(f"Performance plane, coloured by market  (n={len(d):,})")

for i, (yc, yl) in enumerate([("game_z","game_z (production)"), ("effort_z","effort_z")], 2):
    a = fig.add_subplot(2, 4, i)
    a.scatter(d.market_z, d[yc], s=4, alpha=.08, color="#4C72B0", rasterized=True)
    a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
    r_ = d.market_z.corr(d[yc])
    b = np.polyfit(d.market_z, d[yc], 1)
    xs = np.linspace(d.market_z.min(), d.market_z.max(), 50)
    a.plot(xs, np.polyval(b, xs), color="black", ls="--", lw=1.2, label=f"r = {r_:+.3f}")
    a.scatter(f.market_z, f[yc], s=95, color="crimson", edgecolor="white", zorder=5)
    for _, r in f.iterrows():
        a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}", (r.market_z, r[yc]),
                   textcoords="offset points", xytext=(7,4), color="crimson", fontsize=8)
    a.set_xlabel("market_z  (price, line, line move, price move)"); a.set_ylabel(yl)
    a.set_title(f"MARKET vs {yl.split(' ')[0].upper()}")
    a.legend(fontsize=9); a.grid(alpha=.2)
# 4: the combined score
a = fig.add_subplot(2, 4, 4)
a.hist(d.score.dropna(), bins=70, color="#4C72B0", edgecolor="white")
a.axvline(d.score.quantile(.95), color="darkorange", ls="--", lw=1.2,
          label=f"95th pct = {d.score.quantile(.95):.2f}")
for _, r in f.iterrows():
    a.axvline(r.score, color="crimson", ls="--", lw=1.3)
    a.text(r.score, a.get_ylim()[1]*.97, f" {r.player.split()[1][:3]} {r.gd[5:]}",
           color="crimson", fontsize=7.5, rotation=90, va="top")
a.set_xlabel("score = mean(perf_z, shortfall_z, market_z)"); a.set_ylabel("player-games")
a.set_title("COMBINED SCORE"); a.legend(fontsize=8)

# ---- second row: shortfall against each of the other blocks ----------------
for j, (xc, xl) in enumerate([("perf_z", "perf_z  (production + effort, flipped)"),
                              ("market_z", "market_z"),
                              ("game_z", "game_z (production)")], 5):
    a = fig.add_subplot(2, 4, j)
    a.scatter(d[xc], d.shortfall_z, s=4, alpha=.08, color="#55A868", rasterized=True)
    a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
    r_ = d[xc].corr(d.shortfall_z)
    b = np.polyfit(d[xc].fillna(0), d.shortfall_z.fillna(0), 1)
    xs = np.linspace(d[xc].min(), d[xc].max(), 50)
    a.plot(xs, np.polyval(b, xs), color="black", ls="--", lw=1.2, label=f"r = {r_:+.3f}")
    a.scatter(f[xc], f.shortfall_z, s=95, color="crimson", edgecolor="white", zorder=5)
    for _, r in f.iterrows():
        a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}", (r[xc], r.shortfall_z),
                   textcoords="offset points", xytext=(7,4), color="crimson", fontsize=8)
    a.set_xlabel(xl); a.set_ylabel("shortfall_z  (1 - points/line)")
    a.set_title(f"SHORTFALL vs {xl.split(' ')[0].upper()}")
    a.legend(fontsize=9); a.grid(alpha=.2)

# score vs its own percentile, with the flagged games placed
a = fig.add_subplot(2, 4, 8)
a.scatter(d.score, d.score_pct, s=4, alpha=.08, color="#4C72B0", rasterized=True)
a.scatter(f.score, f.score_pct, s=95, color="crimson", edgecolor="white", zorder=5)
for _, r in f.iterrows():
    a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}  {r.score_pct:.0%}",
               (r.score, r.score_pct), textcoords="offset points", xytext=(-70,0),
               color="crimson", fontsize=8)
a.axhline(.95, color="darkorange", ls="--", lw=1.2, label="95th pct")
a.set_xlabel("score"); a.set_ylabel("percentile"); a.set_title("WHERE THE FLAGGED GAMES LAND")
a.legend(fontsize=8); a.grid(alpha=.2)

fig.tight_layout(); fig.savefig(OUT / "three_axis.png", dpi=140)
print(f"-> {OUT/'three_axis.csv'}  ({len(d):,} rows)")
print(f"-> {OUT/'three_axis.png'}")
print(f"\nweights: " + "  ".join(f"{k}={v:g}" for k,v in WEIGHTS.items()))
print(f"component coverage: line_mv {int(d.line_move_pct.notna().sum()):,}   "
      f"price_mv {int(d.price_only_move.notna().sum()):,}   of {len(d):,}")
print(f"\ncorrelations:")
for a_,b_ in (("market_z","game_z"),("market_z","effort_z"),("game_z","effort_z"),
              ("shortfall_z","perf_z"),("shortfall_z","market_z"),
              ("shortfall_z","game_z")):
    print(f"   {a_:<9} ~ {b_:<9} {d[a_].corr(d[b_]):+.3f}")
print(f"\nscore range {d.score.min():+.2f} to {d.score.max():+.2f}   "
      f"95th pct {d.score.quantile(.95):+.2f}")
print(f"\nFLAGGED (sorted by score):")
print(f.sort_values("score", ascending=False)[
    ["player","gd","minutes","points","line","shortfall",
     "perf_z","shortfall_z","market_z","score","score_pct"]
    ].to_string(index=False))
