"""L7: composite scoring -> out/scored.csv. No API calls.

WHY THIS REPLACES THE SEQUENTIAL CUTS
--------------------------------------
A chain of cuts has one property that disqualifies it here: ANY single cut can
delete a true positive permanently, and no later evidence can bring it back. That is
not hypothetical. Both known-bad games died to the SAME cut:

    Jontay Porter  2024-03-20   2.7 min, 0 pts on 7.5   killed by game_margin 34
    Malik Beasley  2024-04-14   4 pts on 8.5            killed by game_margin 25

The blowout filter was defensible in isolation and still wrong, because a blowout is
simultaneously the best innocent explanation and the exact circumstance where a
player can disappear unnoticed. A score cannot make that mistake: a weak axis lowers
a row, it does not erase it.

So the design is ONE MINIMAL GATE plus a weighted score over independent axes.

THE GATE, deliberately tiny -- only things that are logically necessary:
    played (minutes > 0)   a DNP has no performance to judge
    under hit             if he cleared the line nobody was paid, so there is no
                          payoff to explain. Lines are .5 so a push is impossible.

WHAT IS SCORED, AND WHY EACH EARNS ITS PLACE
--------------------------------------------
1. SHORTFALL VS THE MARKET, the heaviest weight, because it is the one measure
   immune to the floor effect that cripples every z-score here. A player averaging 5
   points who scores 0 is only -1 sd from himself; a star averaging 25 who scores 0 is
   -3. Ranking on z therefore systematically finds STARS and misses the fringe --
   precisely the population most exposed to an approach. Jontay Porter's 0-point game
   scored prod_z -0.91 despite being a total no-show.
   `1 - points/close_line` has no such floor: 0 of 7.5 is 1.00, maximally bad, whoever
   you are. The line is also the market's own player-specific forecast, so this is
   normalised by construction.

2. PRODUCTION RESIDUAL and 3. INVOLVEMENT RESIDUAL, context already regressed out per
   role tier. These are the corroboration -- missing shots is common and mostly
   innocent, but doing it while also touching the ball less and running less is not.
   Residual rather than raw so a starter pulled in a rout is not punished for the
   scoreline. THIS IS WHY NO BLOWOUT CUT IS NEEDED: the adjustment already happened.

4. MOTIVE, from salary. The only variable in the project that speaks to what a player
   stood to LOSE. A two-way contract is ~$560K against a max at $50M, and an unlisted
   salary IS the two-way marker, so those rows score maximum motive rather than being
   dropped as missing.

5. LINE PULLED AND HE PLAYED ANYWAY, a small bonus on a rare event. Of 136 withdrawn
   lines, 89 are DNPs -- the books were right and the player was out. The 47 who
   played went under 59.6 percent against a 52.3 percent base. n is small and z is
   about 1.0, so this is NOT significant and is weighted accordingly; it earns a place
   because it is highly specific and is the shape of the Porter 2024-01-26 case.

WHAT IS DELIBERATELY EXCLUDED, having been tested and found empty:
    price_only_move   54.0 vs 51.8 pct under across buckets, non-monotonic, n=4,304
    line_move_pct     flat across buckets at n=2,169 and again now
    close_under level mean reversion, near-zero correlation with anything
Carrying a null signal at any weight only adds variance, so they are exported as
columns for inspection and given weight zero.

    python score_candidates.py            -> out/scored.csv
"""
import pathlib
import warnings

import numpy as np
import pandas as pd

import config
import db

warnings.filterwarnings("ignore")

OUT = pathlib.Path("out/scored.csv")

# Weights. Shortfall carries the most because it is the only floor-free measure and
# the most direct evidence of the thing being detected; the two residual families
# corroborate; motive is a prior, not evidence; the pull bonus is rare and unproven.
WEIGHTS = {"shortfall": 3.0, "production": 2.0, "involvement": 2.0,
           "motive": 2.0, "pulled": 1.0, "under_money": 1.0}

# Salary ceiling for the gate. A max contract is ~$50M a season against a two-way's
# ~$560K, so what a player risks by throwing a game spans two orders of magnitude on
# one roster. This is not a claim that highly paid players are honest -- it is a claim
# about where to look first when the list is longer than anyone can read.
#
# Salary appears TWICE and deliberately: as this hard gate, and as the continuous
# `motive` axis in the score. The gate removes the implausible; the axis orders what
# is left, so a minimum-contract player outranks a $19M one rather than tying with him.
MAX_SALARY = 20_000_000

PROD = ["points_resid_z", "assists_resid_z", "rebounds_resid_z", "fga_resid_z"]
EFFORT = ["usage_pct_resid_z", "turnover_ratio_resid_z", "distance_resid_z",
          "touches_resid_z", "minutes_resid_z"]
FLIP = {"turnover_ratio_resid_z"}      # low turnovers is GOOD, so invert

# BUILT FROM prop_quotes, NOT player_game_features.
#
# player_game_features only has a row where a CLOSING line exists -- which means a
# withdrawn line is structurally invisible to it. The 47 games where the books pulled
# a player and he played anyway, the single most specific signal available, could
# never appear in any output derived from that table.
#
# So the line is taken as: the closing quote if there was one, otherwise the opening
# quote, with `line_source` recording which. A pulled line still has a number to judge
# the performance against -- the one the market last published.
SQL = """
WITH ctx AS (
    SELECT game_id, max(pts) - min(pts) AS game_margin
    FROM (SELECT game_id, team_id, sum(points) AS pts
          FROM player_games WHERE points IS NOT NULL GROUP BY 1, 2) t
    GROUP BY game_id HAVING count(*) = 2),
lines AS (
    SELECT q.player_id, e.game_id,
           max(q.line)  FILTER (WHERE q.snapshot_role = 'close') AS close_line,
           max(q.under_price) FILTER (WHERE q.snapshot_role = 'close') AS close_under,
           max(q.line)  FILTER (WHERE q.snapshot_role = 'open')  AS open_line,
           max(q.under_price) FILTER (WHERE q.snapshot_role = 'open')  AS open_under
    FROM prop_quotes q JOIN odds_events e ON e.event_id = q.event_id
    WHERE q.player_id IS NOT NULL AND q.book = '{book}'
    GROUP BY 1, 2)
SELECT p.full_name AS player, p.position, p.experience, g.game_date, g.matchup,
       pg.minutes, pg.points, pg.fga, pg.rebounds, pg.assists,
       pg.usage_pct, pg.turnover_ratio, pg.distance, pg.touches,
       l.close_line, l.close_under, l.open_line, l.open_under,
       coalesce(l.close_line, l.open_line) AS line,
       CASE WHEN l.close_line IS NOT NULL THEN 'close' ELSE 'open_pulled' END
            AS line_source,
       (l.close_line IS NULL AND l.open_line IS NOT NULL) AS line_pulled,
       f.line_move_pct, f.under_move_pct, f.price_only_move, f.margin_vs_line_z,
       r.tier, r.points_resid_z, r.assists_resid_z, r.rebounds_resid_z,
       r.fga_resid_z, r.usage_pct_resid_z, r.turnover_ratio_resid_z,
       r.distance_resid_z, r.touches_resid_z, r.minutes_resid_z,
       s.salary, s.has_listed_salary, s.salary_pct,
       ctx.game_margin, pg.plus_minus, pg.fouls, pg.started,
       coalesce(b.ejected, FALSE) AS ejected, b.ejection_sec,
       b.points_competitive, b.points_garbage, b.n_stints,
       gc.competitive_sec, gc.max_margin,
       r.n_player_games, pg.player_id, pg.game_id
FROM player_games pg
JOIN games   g ON g.game_id = pg.game_id
JOIN players p ON p.player_id = pg.player_id
JOIN lines   l ON l.player_id = pg.player_id AND l.game_id = pg.game_id
LEFT JOIN ctx ON ctx.game_id = pg.game_id
LEFT JOIN player_game_features f
       ON f.player_id = pg.player_id AND f.game_id = pg.game_id
LEFT JOIN player_game_residuals r
       ON r.player_id = pg.player_id AND r.game_id = pg.game_id
LEFT JOIN player_salaries s
       ON s.player_id = pg.player_id AND s.season = '{season}'
LEFT JOIN player_game_pbp b
       ON b.player_id = pg.player_id AND b.game_id = pg.game_id
LEFT JOIN game_pbp_context gc ON gc.game_id = pg.game_id
"""


def oriented_mean(df, cols):
    """Mean of sign-oriented z-scores, skipping missing. More negative = worse."""
    present = [c for c in cols if c in df.columns]
    block = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") * (-1 if c in FLIP else 1)
                          for c in present})
    return block.mean(axis=1, skipna=True), block.notna().sum(axis=1)


def pct_rank(s):
    """0-1 percentile, HIGHER = more suspicious. NaN -> 0.5, i.e. no opinion.

    Imputing the median rather than 0 matters: a missing component should neither
    promote nor demote a row, and 0 would silently punish every player whose tracking
    data is absent.
    """
    return s.rank(pct=True, na_option="keep").fillna(0.5)


def main():
    with db.connect() as conn:
        df = pd.read_sql(SQL.format(season=config.SEASON, book=config.BOOK), conn)
    start = len(df)

    # ---- GATES --------------------------------------------------------------
    # Sequential, each reported with its cost, so the funnel stays legible. What a
    # gate must earn is that its converse is IMPLAUSIBLE, not merely unlikely -- a
    # gate deletes permanently and no later evidence brings the row back.
    df["prod_z"], df["n_prod"] = oriented_mean(df, PROD)
    df["effort_z"], df["n_effort"] = oriented_mean(df, EFFORT)

    # `~(x > 0)` NOT `x <= 0`. A NaN comparison is False in pandas, so the negated
    # form KEEPS missing values while the direct form would silently delete them --
    # and movement is missing for 48 percent of rows because the opening line simply
    # was not posted 12 hours out. Deleting those would throw away half the data for
    # a reason that has nothing to do with the player.
    gates = [
        ("played (minutes > 0)",           df["minutes"] > 0),
        # EJECTED, from play-by-play. The one benign explanation for a zero-point
        # cameo that the box score cannot express: Klay Thompson, Draymond Green and
        # Jaden McDaniels all logged 1.7 minutes and 0 points on 2023-11-14, and all
        # three were thrown out of the same altercation 103 seconds in. Every ranking
        # this project has produced put at least one of them near the top.
        #
        # Safe as a GATE where `game_margin < 20` never was, because being ejected is
        # genuinely disqualifying -- he did not choose to stop playing -- whereas a
        # blowout is simultaneously the innocent explanation and the opportunity.
        ("not ejected",                    ~df["ejected"].fillna(False).astype(bool)),
        ("under hit (points < line)",      df["points"] < df["line"]),
        ("NOT a good game (prod < 0)",     df["prod_z"] < 0),
        ("low effort for him (effort < 0)", df["effort_z"] < 0),
        # NO LINE-MOVEMENT GATE. Tested on 7,990 rows with movement known and the
        # effect is non-monotonic and BACKWARDS: a line falling more than 10 percent
        # goes under 47.1 percent of the time against a 52.7 percent flat baseline,
        # and those players averaged 1.108x the closing line. A sharply dropping line
        # is a book repricing correctly, after which the lowered number is HARDER to
        # go under -- an efficient market, not money arriving. Gating on it deleted
        # 376 rows for nothing.
        ("no upward PRICE-only move",      ~(df["price_only_move"] > 0)),
        # Same isna() branch as everywhere else, and for the same reason: an unlisted
        # salary marks a two-way or 10-day contract, the lowest-paid and most exposed
        # players on any roster. `salary <= MAX` alone evaluates False for them and
        # deletes exactly the population this gate exists to preserve.
        (f"salary <= ${MAX_SALARY/1e6:.0f}M or unlisted",
         df["salary"].isna() | (df["salary"] <= MAX_SALARY)),
    ]
    print(f"player-games with a line  : {start:,}\n")
    keep = pd.Series(True, index=df.index)
    for label, m in gates:
        before = int(keep.sum())
        keep &= m.fillna(True) if m.dtype == object else m
        print(f"GATE {label:<32} {int(keep.sum()):>7,}   "
              f"(-{before - int(keep.sum()):,})")
    df = df[keep].copy()
    print(f"\n   line from close        : {int((df.line_source == 'close').sum()):,}")
    print(f"   line from open, PULLED : {int((df.line_source == 'open_pulled').sum()):,}")
    print(f"   movement known         : {int(df.line_move_pct.notna().sum()):,}"
          f"   unknown (kept)        : {int(df.line_move_pct.isna().sum()):,}\n")

    # ---- AXES ---------------------------------------------------------------
    # 1. Shortfall as a FRACTION of the market's forecast. Clipped at 0 so a row can
    #    never be rewarded for overperforming; capped at 1 because scoring zero is as
    #    short as it is possible to be and negative points do not exist.
    df["shortfall"] = (1 - df["points"] / df["line"].replace(0, np.nan)).clip(0, 1)

    # 4. Motive. Unlisted salary IS the two-way marker, so it takes the maximum rather
    #    than being treated as unknown -- the whole reason the column exists.
    df["motive"] = np.where(df["has_listed_salary"] == False, 1.0,
                            1 - pd.to_numeric(df["salary_pct"], errors="coerce"))

    # 6. UNDER MONEY. under_move_pct is (close_under - open_under)/open_under, so a
    #    NEGATIVE value means the under price SHORTENED between T-12h and tip -- the
    #    book cut the payout because money was arriving on that side. Negated so more
    #    under money ranks higher.
    #
    #    Weight 1, not more. Across 7,990 rows the extreme buckets differ by 4.6
    #    points (54.2 vs 49.6 percent under, z = 2.11, p ~ 0.035) -- real, monotonic,
    #    and small. Several signals have now been tested on this same data, so a lone
    #    z of 2.1 deserves discounting for multiple comparisons rather than trust.
    #
    #    Missing for 48 percent of rows, which pct_rank sends to 0.5 -- no opinion.
    #    That is the whole reason this is an AXIS and not a gate: a gate would have to
    #    decide something about those rows, and there is nothing to decide.
    parts = {
        "shortfall":   pct_rank(df["shortfall"]),
        "production":  pct_rank(-df["prod_z"]),      # negate: lower z = worse = higher rank
        "involvement": pct_rank(-df["effort_z"]),
        "motive":      pct_rank(df["motive"]),
        "pulled":      df["line_pulled"].astype(float),   # already 0/1
        "under_money": pct_rank(-pd.to_numeric(df["under_move_pct"], errors="coerce")),
    }
    for k, v in parts.items():
        df[f"s_{k}"] = v.round(4)

    total_w = sum(WEIGHTS.values())
    df["score"] = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_w
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)

    print("WEIGHTS  " + "  ".join(f"{k}={v:g}" for k, v in WEIGHTS.items()))
    print("excluded at weight 0: price_only_move, line_move_pct, close_under "
          "(all tested null)\n")
    # Correlations computed BEFORE the sort, while `parts` and `df` still share an
    # index. Sorting first silently misaligns them and every figure comes out ~0.
    print(f"{'axis':<14} {'mean':>7} {'vs score':>9}   pairwise corr")
    keys = list(parts)
    for k in keys:
        pair = "  ".join(f"{j[:4]} {parts[k].corr(parts[j]):+.2f}"
                         for j in keys if j != k)
        print(f"  {k:<12} {parts[k].mean():>7.3f} {parts[k].corr(df['score']):>9.3f}   {pair}")

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\nline_pulled and played    : {int(df.line_pulled.sum()):,}")
    print(f"two-way / unlisted salary : {int((df.has_listed_salary == False).sum()):,}")
    sal = pd.to_numeric(df["salary"], errors="coerce")
    print(f"salary  median ${sal.median():,.0f}   "
          f"min ${sal.min():,.0f}   max ${sal.max():,.0f}")
    band = pd.cut(sal, [0, 2e6, 5e6, 10e6, 20e6],
                  labels=["<2M", "2-5M", "5-10M", "10-20M"])
    print("  " + "   ".join(f"{k} {v:,}" for k, v in band.value_counts().sort_index().items())
          + f"   unlisted {int(sal.isna().sum()):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    main()
