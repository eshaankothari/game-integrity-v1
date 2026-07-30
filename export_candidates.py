"""L6: narrow propped player-games to a reviewable shortlist, in three cuts.

    CUT 1  PRODUCTION   did he produce less than HE usually does?
                        mean(points_z, assists_z, rebounds_z, fga_z) < 0

    CUT 2  EFFORT       was he also less INVOLVED than usual?
                        mean(usage_z, -turnover_ratio_z, distance_z,
                             touches_z, minutes_z) < 0

    CUT 3  RANK         of what survives, order by the biggest shortfall
                        against the market's own forecast for him.

WHY THIS ORDER. Each cut asks a question the previous one cannot answer.
Production alone catches bad shooting nights, which are common and mostly innocent.
Effort is the corroboration: a player can miss shots honestly, but running less and
touching the ball less while getting his usual minutes is harder to explain. Only
then does the betting side get consulted -- so the ranking is applied to games that
already look wrong on the court, rather than fishing for large misses first.

EVERY z-SCORE IS PLAYER-RELATIVE (baseline = that player's own season), so every cut
reads "unlike himself", never "worse than an average NBA player". Against the league
a star's disaster looks ordinary -- Lillard scoring 6 was points_z -0.53 league-wide
but -2.23 against himself.

MEANS, NOT "ALL BELOW". Requiring every component to be negative lets one noisy
member veto the rest (rebounds and assists swing with game flow) and silently drops
the ~21% of rows lacking player-tracking data. The mean skips missing components and
records how many it used, in n_prod / n_effort.

turnover_ratio IS FLIPPED in cut 2. It is the one stat where LOW is good, so its raw
z points the wrong way; unflipped, a player with fewer turnovers than usual would
score as having played worse.

CONTEXT IS EXPORTED, NOT APPLIED. game_margin, fouls and plus_minus are columns, not
filters, so thresholds stay query-side. Deliberate: low minutes is simultaneously the
strongest BENIGN explanation for a bad game and a plausible signature of the thing
being looked for. Klay Thompson's 1.7-minute game on 2023-11-14 was an EJECTION, and
nothing in our data says so -- filtering on low minutes would delete genuine
candidates alongside the innocent ones.

    python export_candidates.py     -> out/candidates.csv  (+ the funnel on stdout)
"""
import pathlib
import warnings

import pandas as pd

import config
import db

warnings.filterwarnings("ignore")

OUT = pathlib.Path("out/candidates.csv")

# One file per cut, so each stage can be inspected on its own rather than only the
# survivors of all of them. Every file carries the same columns, so they diff cleanly.
OUT_CUT1 = pathlib.Path("out/cut1_production.csv")
OUT_CUT2 = pathlib.Path("out/cut2_effort.csv")
OUT_CUT3 = pathlib.Path("out/cut3_salary.csv")

# Cut-3 thresholds. Kept as constants so they are visible and tunable rather than
# buried in a query. 20 points is the conventional blowout line; 6 is a foul-out.
MAX_GAME_MARGIN = 20
FOUL_OUT = 6

# Cut-5 threshold: drop anyone earning more than this.
#
# Salary is the only variable in the project that speaks to MOTIVE rather than to
# what happened on the floor. What a player risks by throwing a game is roughly what
# he is paid, and that spans two orders of magnitude on one roster -- a max contract
# against a two-way's ~$560K. A $50M player forfeits more in one season than any
# plausible payment.
#
# NOT a claim that rich players are honest. It is a claim about where to LOOK first
# when the shortlist is longer than anyone can review by hand.
MAX_SALARY = 20_000_000

# Cuts 1 and 2 are ABSOLUTE thresholds on the player's own z-scores, not quantiles.
#
# A quantile keeps a fixed share of every player's games however ordinary they were,
# so it says "bad FOR HIM" rather than "bad". The threshold says the second, which is
# the claim actually wanted, and it lets a player who never underperformed contribute
# nothing while one who collapsed repeatedly contributes every collapse.
#
#   'mean' -- mean(PROD) < 0. One combined score, so a strong negative in one stat
#             can carry a flat one. Skips missing components and records how many it
#             used in n_prod.
#   'all'  -- every component < 0 independently. Much stricter, and it lets a single
#             noisy member veto the rest: rebounds and assists swing with game flow,
#             so a player who scored far below his norm survives on the mean rule but
#             is thrown out by 'all' for having grabbed one extra rebound.
#
# Both are reported at run time so the cost of switching is visible.
PROD_RULE = "mean"

# Thresholds in the player's OWN standard deviations. At 0 a cut is close to a coin
# flip -- half of anyone's games are below his average by definition -- so 0 keeps 15
# percent of all propped games and does not produce a shortlist. Measured on the full
# season, moving both together:
#
#     threshold    cut1     cut2    FINAL   players   Beasley
#        0.00     7,406    5,539    2,384      276       18
#       -0.25     5,287    3,569    1,617      250       13
#       -0.50     3,363    2,026      966      216        9
#       -0.75     1,872    1,032      507      181        4
#       -1.00       838      456      221      124        3
#
# -0.5 is the chosen default: a half-sigma shortfall on BOTH production and
# involvement, which is a real gap rather than noise, and it cuts the output by 60
# percent while still keeping 216 of 276 players represented. Past -0.75 the player
# count falls away faster than the row count, meaning the survivors concentrate into
# a few volatile players rather than staying league-wide.
PROD_THRESHOLD = -0.5
EFFORT_THRESHOLD = -0.5

# ---- CUT MODE --------------------------------------------------------------
#   'threshold' -- absolute, using the two constants above. "This game was bad."
#   'quantile'  -- WITHIN-PLAYER percentiles, using the two below. "This was bad
#                  FOR HIM", which keeps a fixed share of every player's games
#                  however ordinary they were.
#
# The quantile version balances the shortlist across players by construction, so a
# consistent low scorer is represented as fairly as a volatile star. Its cost is that
# it always keeps a quarter of everyone's games -- a player who never once
# underperformed still contributes rows -- and that a player needs at least 1/keep
# games for the fraction to reach one, so short-sample players drop out silently.
CUT_MODE = "quantile"
PROD_KEEP = 0.25            # keep each player's worst quarter
EFFORT_KEEP = 0.50

# What population cut 2's percentile is measured against.
#
#   'all'       -- the worst half of ALL propped player-games by effort, league-wide.
#                  One fixed cutoff, so surviving it means "low involvement compared
#                  with the league", and a player whose effort is always high can be
#                  eliminated entirely rather than being guaranteed a share.
#   'remaining' -- the worst half of what each player has LEFT after cut 1. Every
#                  player who reached cut 2 with two or more rows keeps at least one,
#                  so the shortlist stays balanced across players by construction.
#
# These answer different questions and the counts differ a lot; both are reported.
EFFORT_SCOPE = "all"

# ---- WHICH z-SCORES THE CUTS RUN ON ----------------------------------------
#   'raw'   -- player-relative z. "Unlike himself", context and all.
#   'resid' -- the same stats after regressing out abs_margin, rest, b2b, home,
#              altitude, pace and days_into_season PER ROLE TIER, then standardised
#              within the player.
#
# WHY RESID IS THE SHARPER QUESTION. A raw z asks "was this unlike him"; a residual z
# asks "was this unlike him AFTER allowing for the situation he was in". The two
# disagree most exactly where it matters -- a starter pulled early in a 30-point
# blowout has a large negative minutes_z and a near-zero minutes_resid_z, because the
# scoreline already explains him. Cut 4's blowout filter exists only because the raw
# version cannot make that distinction; on residuals it is largely redundant.
#
# The catch: residuals are quieter. Context is stripped, so the surviving spread is
# smaller and a fixed threshold bites differently -- which is why the quantile mode
# transfers across cleanly and the absolute thresholds do not.
FEATURE_SET = "resid"

PROD_RAW = ["points_z", "assists_z", "rebounds_z", "fga_z"]
EFFORT_RAW = ["usage_pct_z", "turnover_ratio_z", "distance_z", "touches_z", "minutes_z"]

PROD_RESID = ["points_resid_z", "assists_resid_z", "rebounds_resid_z", "fga_resid_z"]
EFFORT_RESID = ["usage_pct_resid_z", "turnover_ratio_resid_z", "distance_resid_z",
                "touches_resid_z", "minutes_resid_z"]

PROD = PROD_RESID if FEATURE_SET == "resid" else PROD_RAW
EFFORT = EFFORT_RESID if FEATURE_SET == "resid" else EFFORT_RAW

# Low turnovers is GOOD, so its z points the wrong way. Both spellings are listed so
# the set works whichever feature family is active.
FLIP = {"turnover_ratio_z", "turnover_ratio_resid_z"}

# Shared by the per-cut files and the final one, so they diff cleanly against each
# other. Anything absent from a given stage is simply skipped.
EXPORT_COLS = ["player", "position", "game_date", "matchup",
               "prod_z", "n_prod", "effort_z", "n_effort",
               "margin_vs_line_z", "margin_vs_line",
               "close_line", "close_under", "open_line", "line_move_pct",
               "salary", "has_listed_salary", "salary_pct", "experience",
               "minutes", "points", "fga", "rebounds", "assists",
               "usage_pct", "turnover_ratio", "distance", "touches",
               "points_z", "fga_z", "rebounds_z", "assists_z", "usage_pct_z",
               "turnover_ratio_z", "distance_z", "touches_z", "minutes_z",
               "tier",
               "points_resid_z", "fga_resid_z", "rebounds_resid_z", "assists_resid_z",
               "usage_pct_resid_z", "turnover_ratio_resid_z", "distance_resid_z",
               "touches_resid_z", "minutes_resid_z",
               "game_margin", "plus_minus", "fouls", "started",
               "n_player_games", "player_id", "game_id"]

SQL = """
WITH ctx AS (            -- final score margin per game, summed from the box scores
    SELECT game_id, max(pts) - min(pts) AS game_margin
    FROM (SELECT game_id, team_id, sum(points) AS pts
          FROM player_games WHERE points IS NOT NULL GROUP BY 1, 2) t
    GROUP BY game_id HAVING count(*) = 2)
SELECT p.full_name AS player, p.position, f.game_date, g.matchup,
       f.minutes, f.points, f.fga, f.rebounds, f.assists,
       f.usage_pct, f.turnover_ratio, f.distance, f.touches,
       f.points_z, f.fga_z, f.rebounds_z, f.assists_z, f.usage_pct_z,
       f.turnover_ratio_z, f.distance_z, f.touches_z, f.minutes_z,
       f.close_line, f.close_under, f.margin_vs_line, f.margin_vs_line_z,
       ctx.game_margin, pg.plus_minus, pg.fouls, pg.started,
       s.salary, s.has_listed_salary, s.salary_pct, p.experience,
       r.tier,
       r.points_resid_z, r.assists_resid_z, r.rebounds_resid_z, r.fga_resid_z,
       r.usage_pct_resid_z, r.turnover_ratio_resid_z, r.distance_resid_z,
       r.touches_resid_z, r.minutes_resid_z,
       f.n_player_games, f.player_id, f.game_id
FROM player_game_features f
JOIN players      p USING (player_id)
JOIN games        g  ON g.game_id    = f.game_id
JOIN player_games pg ON pg.player_id = f.player_id AND pg.game_id = f.game_id
LEFT JOIN ctx ON ctx.game_id = f.game_id
-- LEFT, not inner. Three players with games have no salary row at all (mid-season
-- waivers the end-of-season roster snapshot missed), and an inner join would delete
-- their games silently rather than leaving the salary NULL.
LEFT JOIN player_salaries s
       ON s.player_id = f.player_id AND s.season = '{season}'
LEFT JOIN player_game_residuals r
       ON r.player_id = f.player_id AND r.game_id = f.game_id
"""


def oriented_mean(df, cols):
    """Mean of sign-oriented z-scores, skipping missing ones.

    Returns (mean, n_used). n_used matters: a mean over 3 components has visibly
    more variance than one over 5, so the two are not directly comparable when
    ranking. It is carried through to the output for exactly that reason.
    """
    present = [c for c in cols if c in df.columns]
    block = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") * (-1 if c in FLIP else 1)
                          for c in present})
    return block.mean(axis=1, skipna=True), block.notna().sum(axis=1)


def worst_fraction(df, col, keep):
    """Mask keeping the worst `keep` fraction of each player's rows by `col`.

    More negative = worse, so an ascending percentile rank puts the worst first.

    method='first' rather than 'average': ties must still be ordered so the quantile
    lands somewhere definite. With 'average' a player whose values are all identical
    gets rank 0.5 on every row and either all of them survive or none do.

    A player needs at least 1/keep rows for the fraction to reach a single game -- at
    keep=0.25 that is four. Below it he contributes nothing and disappears without
    comment, which is why the caller reports how many players are still represented.
    """
    r = df.groupby("player_id")[col].rank(method="first", pct=True, ascending=True)
    return r <= keep


def dump(df, path, label):
    """Write one cut's survivors, ordered worst-production first."""
    cols = [c for c in EXPORT_COLS if c in df.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("prod_z")[cols].to_csv(path, index=False)
    print(f"   -> {path}  ({len(df):,} rows, {len(cols)} cols)  [{label}]")


def main():
    with db.connect() as conn:
        df = pd.read_sql(SQL.format(season=config.SEASON), conn)
        with conn.cursor() as cur:
            cur.execute("SELECT count(DISTINCT game_id) FROM player_games")
            games = cur.fetchone()[0]

    start = len(df)
    print(f"box scores so far        : {games:,} of 1,230 games")
    print(f"propped player-games     : {start:,}\n")

    # ---- CUT 1: PRODUCTION below the player's own norm ----------------------
    # ABSOLUTE, not a quantile. The claim is "this game was below his norm", full
    # stop -- so a player who never underperformed contributes nothing, and one who
    # collapsed repeatedly contributes every collapse. A within-player percentile
    # cannot say that: it keeps a fixed share of everyone's games however ordinary
    # they were.
    df["prod_z"], df["n_prod"] = oriented_mean(df, PROD)
    df["effort_z"], df["n_effort"] = oriented_mean(df, EFFORT)
    n_players_start = df["player_id"].nunique()
    quant = CUT_MODE == "quantile"

    if quant:
        c1 = df[worst_fraction(df, "prod_z", PROD_KEEP)].copy()
        rule = f"worst {PROD_KEEP:.0%} of EACH player's games by mean({', '.join(PROD)})"
    else:
        prod_cols = [c for c in PROD if c in df.columns]
        by_mean = df["prod_z"] < PROD_THRESHOLD
        by_all = pd.concat([pd.to_numeric(df[c], errors="coerce") < PROD_THRESHOLD
                            for c in prod_cols], axis=1).all(axis=1)
        c1 = df[by_mean if PROD_RULE == "mean" else by_all].copy()
        rule = (f"mean({', '.join(PROD)}) < {PROD_THRESHOLD:g}" if PROD_RULE == "mean"
                else f"EVERY one of {', '.join(PROD)} < {PROD_THRESHOLD:g}")

    print(f"CUT 1  production   [mode = {CUT_MODE!r}]")
    print(f"   {rule}")
    print(f"   kept      {len(c1):,}")
    print(f"   eliminated {start - len(c1):,}   ({100*(start-len(c1))/start:.0f}% of all)")
    print(f"   players    {c1['player_id'].nunique():,} of {n_players_start:,} "
          f"still represented")
    if quant:
        g = c1.groupby("player_id").size()
        print(f"   per player median {g.median():.0f} games, max {g.max():.0f}")
        print(f"   prod_z kept ranges {c1.prod_z.min():.2f} to {c1.prod_z.max():.2f}"
              f"   ({int((c1.prod_z > 0).sum()):,} kept rows sit ABOVE the player's "
              f"own mean -- the cost of a relative cut)")
    dump(c1, OUT_CUT1, "after production")
    print()

    # ---- CUT 2: EFFORT / INVOLVEMENT ---------------------------------------
    # The corroboration. A player can miss shots honestly -- that is most of cut 1 --
    # but running less and touching the ball less while getting his usual minutes is
    # harder to explain away.
    #
    # In quantile mode this ranks among cut-1 SURVIVORS, not against the player's whole
    # season, so the pair composes to roughly the worst 12.5 percent of each player's
    # games and everyone who reached cut 2 with two or more rows still contributes one.
    if quant and EFFORT_SCOPE == "all":
        # Cutoff measured on the WHOLE population, not on the survivors, so the bar
        # is one fixed effort level rather than one that moves with whoever is left.
        cutoff = df["effort_z"].quantile(EFFORT_KEEP)
        c2 = c1[c1["effort_z"] <= cutoff].copy()
        rule2 = (f"worst {EFFORT_KEEP:.0%} of ALL {len(df):,} propped games by "
                 f"mean(usage, -turnover_ratio, distance, touches, minutes)"
                 f"   -> effort_z <= {cutoff:.3f}")
    elif quant:
        c2 = c1[worst_fraction(c1, "effort_z", EFFORT_KEEP)].copy()
        rule2 = (f"worst {EFFORT_KEEP:.0%} of each player's REMAINING games by "
                 f"mean(usage, -turnover_ratio, distance, touches, minutes)")
    else:
        c2 = c1[c1["effort_z"] < EFFORT_THRESHOLD].copy()
        rule2 = (f"mean(usage, -turnover_ratio, distance, touches, minutes) "
                 f"< {EFFORT_THRESHOLD:g}")

    print(f"CUT 2  effort/involvement   [scope = {EFFORT_SCOPE!r}]")
    print(f"   {rule2}")
    print(f"   kept      {len(c2):,}")
    print(f"   eliminated {len(c1) - len(c2):,}   "
          f"({100*(len(c1)-len(c2))/max(len(c1),1):.0f}% of cut-1 survivors)")
    print(f"   players    {c2['player_id'].nunique():,} of {n_players_start:,} "
          f"still represented")
    if quant:
        alt = len(c1[worst_fraction(c1, "effort_z", EFFORT_KEEP)]) \
              if EFFORT_SCOPE == "all" \
              else int((c1["effort_z"] <= df["effort_z"].quantile(EFFORT_KEEP)).sum())
        other = "remaining" if EFFORT_SCOPE == "all" else "all"
        print(f"   the other scope ({other!r}) would keep: {alt:,}")
    dump(c2, OUT_CUT2, "after production + effort")
    print()

    # ---- CUT 3: SALARY -- the first cut about MOTIVE ------------------------
    # Everything above asks whether the game looked wrong. This asks whether the
    # player had anything to gain, which is a different question and the only one
    # that constrains WHO rather than WHICH GAME.
    #
    # NULL SALARY IS KEPT, AND THAT IS THE WHOLE POINT. basketball-reference lists no
    # figure for two-way and 10-day contracts, so those players arrive with salary
    # NULL -- and they are the LOWEST-paid, highest-exposure group on any roster.
    # Jontay Porter, the one confirmed case in NBA history, is exactly this row.
    #
    # A plain `salary <= MAX_SALARY` would drop every one of them: NaN comparisons
    # are False in pandas, NULL comparisons are NULL in SQL, and either way the
    # filter deletes the population it exists to find. The isna() branch is
    # load-bearing, not defensive.
    before3 = len(c2)
    dropped = c2[c2["salary"] > MAX_SALARY]
    c2 = c2[c2["salary"].isna() | (c2["salary"] <= MAX_SALARY)].copy()
    print(f"CUT 3  drop the highly paid (salary > ${MAX_SALARY:,})")
    print(f"   kept      {len(c2):,}")
    print(f"   eliminated {len(dropped):,}   "
          f"({100*len(dropped)/max(before3,1):.0f}% of cut-2 survivors, "
          f"{dropped['player'].nunique()} players)")
    print(f"      no listed salary, KEPT (two-way / 10-day): "
          f"{int(c2['salary'].isna().sum()):,}")
    for player, n in dropped["player"].value_counts().head(6).items():
        sal = dropped.loc[dropped["player"] == player, "salary"].iloc[0]
        print(f"      -- {player:<24} ${sal:>11,.0f}  ({n} game{'s' if n > 1 else ''})")
    print(f"   players    {c2['player_id'].nunique():,} of {n_players_start:,} "
          f"still represented")
    dump(c2, OUT_CUT3, "after production + effort + salary")
    print()

    # ---- CUT 4: REMOVE GAMES WITH AN ORDINARY EXPLANATION -------------------
    # These are the two benign causes our data CAN see. Garbage time explains a bad
    # night by itself, and a player who fouled out did not choose to stop playing.
    #
    # Note what is NOT here: low minutes. That is the strongest benign explanation
    # of all AND a plausible signature of the thing being looked for, and nothing in
    # our data separates the two -- Klay Thompson's 1.7-minute game on 2023-11-14 was
    # an ejection. Excluding on it would delete real candidates with the innocent ones.
    before4 = len(c2)
    c3 = c2[(c2["game_margin"] < MAX_GAME_MARGIN) & (c2["fouls"] < FOUL_OUT)].copy()
    print(f"CUT 4  drop games with an ordinary explanation   [retained from before]")
    print(f"   game_margin < {MAX_GAME_MARGIN} (not garbage time) and fouls < {FOUL_OUT}")
    print(f"   kept      {len(c3):,}")
    print(f"   eliminated {before4 - len(c3):,}   "
          f"({100*(before4-len(c3))/max(before4,1):.0f}% of cut-3 survivors)")
    print(f"      blowouts, game_margin >= {MAX_GAME_MARGIN} : "
          f"{(c2.game_margin >= MAX_GAME_MARGIN).sum():,}")
    print(f"      fouled out, fouls >= {FOUL_OUT}          : {(c2.fouls >= FOUL_OUT).sum():,}\n")

    # ---- CUT 4: THE UNDER ACTUALLY HIT --------------------------------------
    # points < close_line. Up to here every cut has been about the PERFORMANCE; this
    # is the first one about the BETTING OUTCOME. A player can play well below his own
    # norm and still clear a line that was set low, in which case nobody betting the
    # under won anything and there is no payoff to explain.
    #
    # Lines are always .5, so a push is impossible -- every row is a clean over or
    # under.
    #
    # This makes the tool one-sided by construction: it cannot see over-side
    # manipulation (a player inflating stats), which is a real pattern in prop
    # markets. That is the mirror filter, not a redesign.
    before5 = len(c3)
    c3 = c3[c3["margin_vs_line"] < 0].copy()
    print(f"CUT 5  the under hit (points < close_line)       [retained from before]")
    print(f"   kept      {len(c3):,}")
    print(f"   eliminated {before5 - len(c3):,}   "
          f"({100*(before5-len(c3))/max(before5,1):.0f}% of cut-4 survivors -- "
          f"played below their own norm but still beat the line)\n")

    # ---- RANK: two orderings, because they disagree ------------------------
    # margin_vs_line is points - close_line, so it is NEGATIVE on a miss.
    #
    # rank_abs  by margin_vs_line_z -- the miss in league-standardised points.
    #           Systematically favours STARS: a 30.5 line missed by 24 outranks a
    #           3.5 line missed by 3.5, though the second is proportionally total.
    # rank_pct  by margin_vs_line / close_line -- the miss as a FRACTION of what
    #           was expected. Puts small-line, low-usage players on equal footing,
    #           which is the profile most likely to matter.
    #
    # Both are exported. Where they disagree is where the interesting rows are.
    c3 = c3[c3["margin_vs_line"].notna()].copy()
    c3["margin_pct"] = c3["margin_vs_line"] / c3["close_line"].replace(0, pd.NA)
    c3["shortfall"] = -c3["margin_vs_line"]             # positive = missed by this much
    c3["rank_abs"] = c3["margin_vs_line_z"].rank(method="min").astype(int)
    c3["rank_pct"] = c3["margin_pct"].rank(method="min").astype(int)

    # PRIMARY RANKING: lowest close_under first.
    #
    # These are DECIMAL odds, so a LOW price means the under was heavily favoured --
    # 1.60 implies ~63% while 2.20 implies ~45%. A book shortens a price when money
    # arrives on that side, so a low under price is the market having already leaned
    # under before tip. Combined with the under then hitting, that is the "someone
    # knew" shape rather than "a player had a bad night".
    #
    # CAVEAT worth carrying: a short under price also just describes a player the
    # market expects to be quiet, so this may partly rank player TYPE rather than
    # unusual money. It is a LEVEL, not a movement -- it cannot separate "drifted down
    # on under money" from "opened there". line_move_pct / under_move_pct are the
    # direct measures and need the opening pass.
    c3["rank_under"] = c3["close_under"].rank(method="min").astype(int)
    c3 = c3.sort_values(["close_under", "margin_vs_line_z"]).reset_index(drop=True)

    print(f"RANK   {len(c3):,} candidates, ordered by close_under ASCENDING "
          f"(shortest price = most favoured under = rank 1)")
    print(f"   close_under range      : {c3['close_under'].min():.2f} "
          f"to {c3['close_under'].max():.2f}")
    print(f"   worst margin_vs_line_z : {c3['margin_vs_line_z'].min():.2f}")
    print(f"   worst margin_pct       : {c3['margin_pct'].min():.2f}")
    ov = len(set(c3.head(25).index) & set(c3.nsmallest(25, 'margin_vs_line_z').index))
    print(f"   top-25 overlap, price ranking vs absolute-miss ranking: {ov} of 25")

    cols = (["rank_under", "rank_pct", "rank_abs", "player", "position", "game_date", "matchup",
             "prod_z", "n_prod", "effort_z", "n_effort",
             "margin_vs_line_z", "margin_vs_line", "margin_pct", "shortfall",
             "close_line", "close_under",
             "salary", "has_listed_salary", "salary_pct", "experience",
             "minutes", "points", "fga", "rebounds", "assists",
             "usage_pct", "turnover_ratio", "distance", "touches",
             "points_z", "fga_z", "rebounds_z", "assists_z", "usage_pct_z",
             "turnover_ratio_z", "distance_z", "touches_z", "minutes_z",
             "game_margin", "plus_minus", "fouls", "started",
             "n_player_games", "player_id", "game_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c3[cols].to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(c3):,} rows, {len(cols)} columns)")


if __name__ == "__main__":
    main()
