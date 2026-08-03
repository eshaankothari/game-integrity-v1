"""Two independent metrics: PERFORMANCE (what he did) and MARKET (what prices did).

Kept separate on purpose. They correlate at roughly -0.03, so they carry genuinely
different information and blending them into one number destroys the ability to ask
"did BOTH point the same way", which is the interesting question.

PERFORMANCE -- five components: two measures against TWO baselines, plus shortfall.

    game_z       Hollinger Game Score. 11 box-score inputs:
                 PTS + .4*FGM - .7*FGA - .4*(FTA-FTM) + .7*ORB + .3*DRB
                     + STL + .7*AST + .7*BLK - .4*PF - TOV
    effort_z     mean of 9 involvement/exertion stats, each z-scored separately:
                 touches, passes, usage_pct, distance,
                 contested_shots, deflections, loose_balls, box_outs, screen_assists
                 fga is excluded -- Game Score already charges -0.7 per attempt.
    shortfall_z  1 - points/line, clipped [0,1], then standardised. The only
                 component with no floor: a 0-point game is 1.00 for anyone, while
                 game_z gives a 4-pt/game player -0.97 and a star -3.23 for the same
                 event, so any z-threshold is secretly a threshold on scoring level.

    game_z_tier     the same Game Score, z-scored against everyone in his ROLE
    effort_z_tier   the same 9 stats, z-scored against everyone in his ROLE

    The own and tier baselines fail in OPPOSITE directions -- own is soft for a player
    with many quiet games, tier is soft for a bench player graded against cameos -- so
    both are kept rather than one chosen. See the WEIGHTS block below for the test.

    They all measure the same night -- pairwise 0.53 to 0.68 -- so they are AVERAGED,
    not multiplied. Multiplying correlated inputs inflates apparent rarity roughly
    10-fold, measured on this data.

MARKET -- four components, each z-scored league-wide, all oriented so that a LOW raw
value gives a HIGH score, since low price / small line / downward movement is what
under-side money produces.

    p_price     percentile of the closing under price      100 pct coverage
    p_line      percentile of the line                     100 pct
    line_mv     -line_move_pct                              52 pct
    price_mv    -price_only_move                            28 pct

    Normalised by the weight PRESENT, not the full total, so a missing movement term
    does not compress the score toward zero for a reason unrelated to the game.

WEIGHTS ARE EQUAL WITHIN EACH BLOCK, and now the code matches that claim. There is no
valid outcome to fit to: `under_hit` is what the market was pricing, so agreeing with it
measures market efficiency rather than manipulation, and 6 labelled games across 2
players cannot support fitted weights -- a logistic fit on them learned "Malik Beasley",
not the phenomenon. Unequal weights would be an asserted prior wearing empirical
clothes. See analysis/weight_audit.py for the measurements that settled this.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import db, config

OUT = pathlib.Path(__file__).resolve().parent / "out"
MIN_GAMES = 15

# ---- WEIGHTS ----------------------------------------------------------------
# WITHIN-BLOCK WEIGHTS ARE EQUAL BECAUSE UNEQUAL ONES WERE TESTED AND DID NOTHING.
#
# An earlier version carried tuned splits (game_z .45 / effort_z .20 / shortfall_z .35,
# and p_price .50 / p_line .15 / line_mv .15 / price_mv .20) justified by leave-one-
# player-out logistic coefficients. analysis/weight_audit.py falsified that:
#
#   ONE-AT-A-TIME.  Setting each weight to 0 and to 2x, the cost (mean log10 rank of
#   the 6 flagged games) moved by 0.02 to 0.09 for every WITHIN-block weight, against
#   0.58 for block.motive. Zeroing shortfall_z entirely cost 0.027. Seven numbers,
#   none of which changed the answer.
#
#   RANDOM ENSEMBLE.  Over 20,000 Dirichlet draws, 48.1% of RANDOM weight vectors beat
#   the tuned ones, whose geometric-mean flagged rank (858) sat almost exactly on the
#   random median (897). Equal weights scored 593 -- better than the tuning.
#
#   THE FITS THEMSELVES.  Held-out AUC was 0.661 and 0.600 against a 0.50 baseline,
#   and the four behavioural coefficients CORRELATED -0.607 across the two folds --
#   game_z swung 17x, shortfall_z flipped sign. The apparent +0.972 agreement was one
#   term: motive, which ran 3.9x and 7.0x the sum of all other |coefficients| because
#   both label players are cheap. The fit learns identity, not behaviour.
#
# So the within-block weights are 1/n. This is not humility for its own sake -- it is
# that any other choice is an unfalsifiable assertion, and the equal one is at least
# as good on the only evidence available.
# FIVE components: the same two performance measures against TWO baselines, plus
# shortfall. The two baselines fail in opposite directions, so both get a vote.
#
#   own    (x - his mean) / his sd        "unlike HIM"
#   tier   (x - tier mean) / tier sd      "bad for a player in his ROLE"
#
# own has a SOFT-BASELINE problem: a player with many quiet games sets a low bar for
# himself. Malik Beasley 01-06 is game_z_own -1.28 but game_z_tier -1.68 -- against
# starters generally, 3 points is worse than it looks against his own season.
#
# tier is soft in reverse: the bench tier is full of 6-minute end-of-rotation cameos, so
# a bench player who normally does more gets graded against a population that does less.
# Jontay Porter 01-20 is game_z_own -0.17 but game_z_tier +0.27 -- "above average for a
# bench player", which is true and useless.
#
# Averaging both means neither soft baseline gets the last word. analysis/tier_blend.py
# tested six ways of combining them; this one (blend C) was the ONLY one that improved
# the Beasley games without paying for it with the Porter games:
#
#     blend                 Beasley 4   Porter 2
#     own only (was)             868        468
#     own + tier both            654        457     -24.7%   -2.3%
#     tier only                  539        595     -37.9%  +27.2%
#
# tier-only scores best overall and is rejected: its gain is reweighting toward the
# player who supplies four of the six labels, not more signal.
PERF_W   = {"game_z": 1/5, "effort_z": 1/5,
            "game_z_tier": 1/5, "effort_z_tier": 1/5, "shortfall_z": 1/5}
MARKET_W = {"p_price": 0.25, "p_line": 0.25, "line_mv": 0.25, "price_mv": 0.25}

# THE BLOCK WEIGHTS ARE A PRIOR, NOT A FIT, and they are the only weights in this file
# that measurably change the ranking (swings 0.14 / 0.25 / 0.58 vs <=0.09 within-block).
#
# motive is the largest lever and the one least safe to fit: turning it up always helps
# the flagged games, because "cheap player" is what the label set has in common. It is
# held at 0.25 -- below both other blocks -- precisely so the score cannot become a
# salary sort. Raising it would improve the flagged ranks and mean nothing.
#
# Sensitivity is published rather than buried: run analysis/weight_audit.py.
BLOCK_W  = {"performance": 0.45, "market": 0.30, "motive": 0.25}

# MOTIVE -- salary, the only axis about what a player RISKED rather than what he did.
#
# Kept out of the logistic fits deliberately: with salary included it took the largest
# coefficient in every run (-3.76, then -5.96) and drove test AUC to 0.087, because
# "earns $2.0M with 79 games" identifies Malik Beasley rather than a behaviour. A
# fitted weight on salary is a fitted weight on which players supplied the labels.
#
# It belongs in the score anyway, on grounds that owe nothing to the fit: what a
# player forfeits by throwing a game spans two orders of magnitude on one roster --
# roughly $50M for a max contract against $560K for a two-way. Weighted below
# performance because it is a PRIOR about who could plausibly be approached, not
# evidence about what happened.
#
# UNLISTED SALARY SCORES MAXIMUM, not missing. basketball-reference publishes no
# figure for two-way and 10-day contracts, so those rows arrive NULL -- and they are
# the lowest-paid players on any roster. Treating NULL as unknown would drop exactly
# the population this axis exists to surface. Jontay Porter is one of these rows.

EFFORT = ["touches", "passes", "usage_pct", "distance", "contested_shots",
          "deflections", "loose_balls", "box_outs", "screen_assists"]

q = """
SELECT p.full_name AS player, g.game_date, pg.minutes, pg.points,
       pg.fgm, pg.fga, pg.fta, pg.ftm, pg.rebounds, pg.rebounds_off,
       pg.assists, pg.steals, pg.blocks, pg.turnovers, pg.fouls,
       pg.touches, pg.passes, pg.usage_pct, pg.distance,
       pg.contested_shots, pg.deflections, pg.loose_balls,
       pg.box_outs, pg.screen_assists,
       f.close_line, f.line_move_pct, f.price_only_move,
       q2.under_price, q2.over_price,
       r.tier, s.salary, s.has_listed_salary,
       pg.player_id, pg.game_id
FROM player_games pg
JOIN players p ON p.player_id = pg.player_id
JOIN games   g ON g.game_id  = pg.game_id
LEFT JOIN player_game_features f ON f.player_id = pg.player_id AND f.game_id = pg.game_id
LEFT JOIN (SELECT q.player_id, e.game_id, max(q.under_price) under_price,
                  max(q.over_price) over_price
           FROM prop_quotes q JOIN odds_events e ON e.event_id = q.event_id
           WHERE q.snapshot_role='close' AND q.book=%(book)s GROUP BY 1,2) q2
     ON q2.player_id = pg.player_id AND q2.game_id = pg.game_id
LEFT JOIN player_game_residuals r ON r.player_id=pg.player_id AND r.game_id=pg.game_id
LEFT JOIN player_salaries s ON s.player_id=pg.player_id AND s.season=%(season)s
WHERE pg.minutes > 0 AND pg.points IS NOT NULL
"""
with db.connect() as c:
    d = pd.read_sql(q, c, params={"season": config.SEASON, "book": config.BOOK})
for c_ in d.columns:
    if c_ not in ("player", "game_date", "game_id", "tier", "has_listed_salary"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
d["gd"] = d.game_date.astype(str).str[:10]

z = lambda s: (s - s.mean()) / s.std()

# ---- PERFORMANCE ------------------------------------------------------------
drb = d.rebounds - d.rebounds_off.fillna(0)
d["game_score"] = (d.points + .4*d.fgm - .7*d.fga - .4*(d.fta-d.ftm)
                   + .7*d.rebounds_off.fillna(0) + .3*drb + d.steals
                   + .7*d.assists + .7*d.blocks - .4*d.fouls - d.turnovers).round(2)

def own_z(s):
    g = s.groupby(d.player_id)
    return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)

def tier_z(s):
    """Against everyone in the same role, not against himself. tier comes from
    player_game_residuals and is the season-average-minutes bucket
    (bench / rotation / starter), so it is a property of the PLAYER, not of the night."""
    g = s.groupby(d.tier)
    return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)


d["game_z"] = own_z(d.game_score).round(3)
_e = pd.concat([own_z(d[c]) for c in EFFORT], axis=1)
d["n_effort"] = _e.notna().sum(axis=1)
d["effort_z"] = _e.fillna(0).mean(axis=1).round(3)      # missing -> 0 = "average for him"

d["game_z_tier"] = tier_z(d.game_score).round(3)
_et = pd.concat([tier_z(d[c]) for c in EFFORT], axis=1)
d["effort_z_tier"] = _et.fillna(0).mean(axis=1).round(3)
d["shortfall"] = (1 - d.points / d.close_line.replace(0, np.nan)).clip(0, 1).round(3)

# STANDARDISED LEAGUE-WIDE, NOT WITHIN PLAYER. This is the fix for a bug that undid
# the entire reason shortfall exists.
#
# The raw quantity is already player-relative -- the denominator is the market's
# forecast for THAT player in THAT game -- so a zero-point night is 1.000 for anyone.
# Z-scoring it against the player's own season then divided by his own sd and put the
# floor effect straight back:
#
#     player            his mean sf   his sd    shortfall   shortfall_z
#     Jontay Porter        0.456       0.453      1.000        1.199
#     Malik Beasley        0.229       0.295      1.000        2.616
#     Ayo Dosunmu          0.127       0.216      1.000        4.040
#
# All three scored ZERO. A low-usage player falls short of his line often, so his mean
# shortfall is high and his sd large, and dividing by it compressed his worst possible
# game to a third of what a high scorer got for the same event. Across all 349
# zero-point games, raw shortfall is constant at 1.000 while the within-player z ran
# from +0.50 to +4.87, correlating +0.788 with the player's scoring level.
#
# League-wide standardisation puts it on the same scale as game_z and effort_z -- which
# is the only reason to standardise at all -- without any per-player reference.
d["shortfall_z"] = z(d.shortfall).round(3)

# MIN_GAMES blanks only the OWN-baseline components. A thin season makes a player's own
# mean and sd unreliable, which is what that guard is for -- but it says nothing about
# the tier baseline, which is estimated from every player in the role and is exactly the
# thing that still works when a player has too few games of his own. Blanking the tier
# columns here would discard the one measure that survives the problem.
n = d.groupby("player_id")["points"].transform("size")
d["n_games"] = n
for c_ in ("game_z", "effort_z", "shortfall_z"):
    d.loc[n < MIN_GAMES, c_] = np.nan

# oriented so HIGHER = worse night
perf = {"game_z": -d.game_z, "effort_z": -d.effort_z,
        "game_z_tier": -d.game_z_tier, "effort_z_tier": -d.effort_z_tier,
        "shortfall_z": d.shortfall_z}
pn = sum(PERF_W[k]*v.fillna(0) for k, v in perf.items())
pd_ = sum(PERF_W[k]*v.notna().astype(float) for k, v in perf.items())
d["performance"] = (pn / pd_.replace(0, np.nan)).round(3)

# ---- MARKET -----------------------------------------------------------------
d["p_price"] = d.under_price.rank(pct=True).round(4)
d["p_line"] = d.close_line.rank(pct=True).round(4)
mkt = {"p_price":  z(1 - d.p_price),
       "p_line":   z(1 - d.p_line),
       "line_mv":  z(-d.line_move_pct),
       "price_mv": z(-d.price_only_move)}
for k, v in mkt.items():
    d[f"mk_{k}"] = v.round(3)
mn = sum(MARKET_W[k]*v.fillna(0) for k, v in mkt.items())
md = sum(MARKET_W[k]*v.notna().astype(float) for k, v in mkt.items())
d["market"] = (mn / md.replace(0, np.nan)).round(3)
d["n_market"] = sum(v.notna().astype(int) for v in mkt.values())

# PROPPED ONLY, and only NOW. The z-scores above are built on every game the player
# played; filtering first would have made each baseline the mean of his PROPPED games,
# which for Jontay Porter is 7 of 26 -- few enough that the MIN_GAMES guard blanked him
# entirely. The baseline must be his season; the output is the propped subset of it.
d = d[d.close_line.notna()].copy()

# ---- COMBINED SCORE ---------------------------------------------------------
# The two blocks correlate at +0.06, so they are genuinely separate evidence. Averaged
# rather than multiplied: a product would be dominated by whichever block is more
# extreme, and would go to zero whenever either is missing.
sal = pd.to_numeric(d.salary, errors="coerce")
# pd.Series, not the bare np.where result: numpy's mean/std PROPAGATE NaN while
# pandas skips it, and the handful of players with no salary row at all would
# otherwise turn the entire column into NaN.
_mot = pd.Series(np.where(d.has_listed_salary == False, 1.0,
                          1 - sal.rank(pct=True)), index=d.index)
d["motive"] = z(_mot).round(3)

d["score"] = (BLOCK_W["performance"] * d.performance
              + BLOCK_W["market"] * d.market
              + BLOCK_W["motive"] * d.motive).round(3)
d["score_pct"] = d.score.rank(pct=True).round(4)
d["perf_pct"] = d.performance.rank(pct=True).round(4)
d["market_pct"] = d.market.rank(pct=True).round(4)
d["under_hit"] = d.points < d.close_line
d = d.sort_values("score", ascending=False).reset_index(drop=True)
d.insert(0, "rank", range(1, len(d) + 1))
d.to_csv(OUT / "two_metrics.csv", index=False)

m = d.dropna(subset=["performance", "market"])
print(f"rows {len(d):,}   with both metrics {len(m):,}\n")
print("WITHIN-BLOCK correlations (why they are averaged, not multiplied):")
for a_, b_ in (("game_z","effort_z"),("game_z","shortfall_z"),("effort_z","shortfall_z"),
               ("game_z","game_z_tier"),("effort_z","effort_z_tier")):
    print(f"   {a_:<12} ~ {b_:<12} {d[a_].corr(d[b_]):+.3f}")
for a_, b_ in (("mk_p_price","mk_p_line"),("mk_p_price","mk_line_mv"),
               ("mk_p_price","mk_price_mv"),("mk_line_mv","mk_price_mv")):
    print(f"   {a_:<12} ~ {b_:<12} {d[a_].corr(d[b_]):+.3f}")
print(f"\nBETWEEN blocks:  performance ~ market  {m.performance.corr(m.market):+.3f}")
print(f"\ncoverage: p_price {int(d.mk_p_price.notna().sum()):,}  "
      f"p_line {int(d.mk_p_line.notna().sum()):,}  "
      f"line_mv {int(d.mk_line_mv.notna().sum()):,}  "
      f"price_mv {int(d.mk_price_mv.notna().sum()):,}   of {len(d):,}")

# ---- PLOT -------------------------------------------------------------------
FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
f = d[d.apply(lambda r: r.gd in FLAG.get(r.player, []), axis=1)]
print(f"\nFLAGGED:")
print(f.sort_values("score", ascending=False)[
    ["rank","player","gd","minutes","points","close_line","game_z","game_z_tier",
     "effort_z","effort_z_tier","shortfall_z","performance","market","motive",
     "score","score_pct"]].to_string(index=False))
print(f"\nTOP 20 BY SCORE:")
cols = ["rank","player","gd","minutes","points","close_line",
        "performance","market","motive","score","salary","under_hit"]
t = d.head(20)[cols].copy()
t["salary"] = t.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(t.to_string(index=False))


m = d.dropna(subset=["performance", "market"])
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))

# Main: the two independent blocks against each other.
a = ax[0]
a.scatter(m.market, m.performance, s=5, alpha=.10, color="#4C72B0", rasterized=True)
a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
r_ = m.market.corr(m.performance)
a.plot(np.linspace(m.market.min(), m.market.max(), 50),
       np.polyval(np.polyfit(m.market, m.performance, 1),
                  np.linspace(m.market.min(), m.market.max(), 50)),
       color="black", ls="--", lw=1.2, label=f"r = {r_:+.3f}")
# iso-score lines: everything on one line has the same (perf+market)/2
for sv in (0.5, 1.0, 1.5):
    xs = np.linspace(m.market.min(), m.market.max(), 50)
    a.plot(xs, 2*sv - xs, color="darkorange", lw=1, alpha=.7,
           label=f"score = {sv}" if sv == 0.5 else None)
a.scatter(f.market, f.performance, s=110, color="crimson", edgecolor="white", zorder=5)
for _, r in f.iterrows():
    a.annotate(f"{r.player.split()[1][:3]} {r.gd[5:]}  ({r.score_pct:.0%})",
               (r.market, r.performance), textcoords="offset points",
               xytext=(8, 5), color="crimson", fontsize=8)
a.set_xlabel("MARKET   (price, line, line move, price move)")
a.set_ylabel("PERFORMANCE   (game_z, effort_z, shortfall_z -- higher = worse)")
a.set_title(f"The two blocks are independent  (n={len(m):,})")
a.legend(fontsize=9, loc="lower left"); a.grid(alpha=.2)

# Same plane, coloured by whether the under actually hit.
a = ax[1]
for hit, col, lab in ((False, "#4C72B0", "over"), (True, "#C44E52", "under hit")):
    s_ = m[m.under_hit == hit]
    a.scatter(s_.market, s_.performance, s=5, alpha=.10, color=col,
              label=f"{lab}  ({len(s_):,})", rasterized=True)
a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
a.scatter(f.market, f.performance, s=110, facecolor="none", edgecolor="black",
          lw=1.8, zorder=5)
a.set_xlabel("MARKET"); a.set_ylabel("PERFORMANCE")
a.set_title("Same plane, coloured by outcome")
leg = a.legend(fontsize=9, markerscale=3); 
for lh in leg.legend_handles: lh.set_alpha(1)
a.grid(alpha=.2)

fig.tight_layout(); fig.savefig(OUT / "two_metrics.png", dpi=140)
print(f"\n-> {OUT/'two_metrics.png'}")
q = m.assign(pq=pd.qcut(m.performance,4,labels=["P1","P2","P3","P4"]),
             mq=pd.qcut(m.market,4,labels=["M1","M2","M3","M4"]))
print("\nunder-hit % by quartile of each block:")
print((100*q.pivot_table(index="pq", columns="mq", values="under_hit",
                         aggfunc="mean")).round(1).to_string())
