"""L5: player-games -> the three scoring blocks -> `player_game_features`.

    performance   what he did on the floor, against two baselines plus the market's
    market        what the prices did before tip-off
    motive        what he stood to lose: salary (0.5) + career instability (0.5)

    score     = 0.45*performance + 0.30*market + 0.25*motive     (raw, z-scale)
    score_100 = (score - min) / (max - min) * 100                (what the UI shows)

score_100 is PRESENTATION. A linear rescale is strictly monotone, so it and `score` order
the population identically -- the 0-100 scale changes how a number reads, never which
games surface. Linear rather than a percentile because a percentile is flattest exactly
where the list is read: across the top 200 it spanned 1.28 points, this spans 15.07.

The raw `score` is kept because a percentile is defined against a POPULATION: add a season
and every 0-100 number shifts, while `score` does not. It is the number to compare across
runs.

No API calls. Pure computation over player_games, prop_quotes, player_salaries,
player_career and player_game_residuals.

POPULATION -- read this before anything else.

    played player-games (minutes > 0)      26,393
      with a FanDuel CLOSING line          15,498    <- the output
      with an OPENING observation           8,992    (8,037 T-12 'open',
                                                      955 earliest-'poll' fallback)

    BASELINES are computed over EVERY game a player played. The propped filter is
    applied at the END. Filtering first would make each player's "normal" the mean of
    his PROPPED games -- systematically his higher-profile nights -- inflating the mean
    and shrinking every z toward zero. For Jontay Porter that is 7 games of 26.

    A CLOSING LINE IS REQUIRED. shortfall divides by it and p_price is derived from it,
    so a row without one has no performance and no market block. 47 player-games opened
    a line that was later PULLED and never closed; those are excluded with the rest.

    AN OPENING OBSERVATION IS NOT REQUIRED, and its absence is never a penalty. The
    market block divides by the weight PRESENT, so those rows are judged on price and
    line alone rather than scored as though the movement terms had been observed and
    found neutral.

    THE EFFECTIVE OPEN. Where a T-12 'open' snapshot exists it is the open. Where none
    does, the EARLIEST 'poll' snapshot (the one furthest before tip) stands in as the
    opening observation -- used as-is, with the window recorded in open_offset_hours
    and the provenance in open_source ('open' | 'poll' | NULL). No poll predates T-12,
    so a poll can never displace a true open. Rows with neither stay NaN.

    python -m pipeline.score.standardize           # DRY: report, write nothing
    python -m pipeline.score.standardize run
"""
import warnings

import numpy as np
import pandas as pd

from pipeline.core import config
from pipeline.core import db

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

# ---- WEIGHTS ---------------------------------------------------------------
# WITHIN-BLOCK WEIGHTS ARE 1/n BECAUSE UNEQUAL ONES WERE TESTED AND DID NOTHING.
#
#   ONE-AT-A-TIME. Setting each weight to 0 and to 2x moved the cost (mean log10 rank
#   of the six flagged games) by 0.02-0.09 for every WITHIN-block weight, against 0.58
#   for the motive block weight. Zeroing shortfall_z entirely cost 0.027.
#
#   RANDOM ENSEMBLE. Over 20,000 Dirichlet draws, 48.1% of RANDOM weight vectors beat
#   the previously tuned ones, whose geometric-mean flagged rank (858) sat almost
#   exactly on the random median (897).
#
#   THE FITS. Held-out AUC 0.661 and 0.600 against a 0.50 baseline, and the four
#   behavioural coefficients correlated -0.607 across the two leave-one-PLAYER-out
#   folds -- game_z swung 17x, shortfall_z flipped sign.
#
# Any other choice is an unfalsifiable assertion, and the equal one is at least as good
# on the only evidence available. See experiments/weight_audit.py.
# THREE components, against the player's OWN season. The tier baseline was added and
# then REMOVED -- see the note below.
PERF_W = {"game_z": 1/3, "effort_z": 1/3, "shortfall_z": 1/3}

# WHY THE TIER BASELINE IS NOT IN HERE, having briefly been.
#
# game_z_tier and effort_z_tier standardise the same quantities against everyone in the
# player's ROLE rather than against himself. They were added on the strength of a
# per-player split that was computed WRONG -- a sorted rank list indexed as though it
# were in game order, which scrambled which player each number belonged to. Recomputed
# correctly, the blend does the opposite of what it was adopted for:
#
#     game          own only   own+tier
#     Por 03-20           90        176      Beasley  -38.6%
#     Bea 01-06          171         78      Porter   +46.8%
#     Bea 01-26          279        165
#     Bea 02-27        1,696      1,245
#     Por 01-20        2,432      2,679
#     Bea 03-10        7,028      5,052
#
# The mechanism is real and was described correctly at the time: the bench tier is full
# of short end-of-rotation appearances, so a 2.7-minute scoreless game is UNREMARKABLE
# for the role even though it is well outside what that player normally does. Porter
# 03-20 is game_z -0.81 against himself and only -0.56 against his tier, and averaging
# the two dilutes the one thing that made the game stand out.
#
# The columns are still COMPUTED and stored -- they are a useful diagnostic, and a
# reviewer should be able to see when a game is bad for the player but ordinary for the
# role. They are simply not inputs to the score.
MARKET_W = {"p_price": 0.25, "p_line": 0.25, "line_mv": 0.25, "price_mv": 0.25}

# THE BLOCK WEIGHTS ARE A PRIOR, NOT A FIT, and they are the only weights here that
# measurably change the ranking. motive is the largest lever and the one least safe to
# fit: turning it up always helps the flagged games, because "cheap" is what the two
# label players have in common. Held at 0.25 -- below both other blocks -- precisely so
# the score cannot become a salary sort.
BLOCK_W = {"performance": 0.45, "market": 0.30, "motive": 0.25}

# WITHIN the motive block: salary and career instability at 0.5 each. Instability is
# ((team changes + 2*gap seasons)/(span+1), from player_career). KNOWN TENSIONS with
# equal weighting, accepted deliberately: (a) it overlaps salary (r = -0.21 across the
# 406 listed-salary propped players), so some content is double-counted; (b) every
# labeled game improves when this weight rises BECAUSE both label players are
# journeymen -- the two-player-overfit axis; (c) at 0.5 well-paid traded stars
# (Porziņģis-shaped) rise ~3,000 rank_all places -- "well-traveled" partly outranking
# "nothing to lose". Measured at 0.25/0.35/0.50 before adoption.
MOTIVE_W = {"salary": 0.5, "instability": 0.5}

# The nine involvement stats, each standardised separately then averaged. fga is
# excluded: Game Score already charges -0.7 per attempt, so including it would count
# shot volume twice with opposite signs.
EFFORT = ["touches", "passes", "usage_pct", "distance", "contested_shots",
          "deflections", "loose_balls", "box_outs", "screen_assists"]

# Below this many games a player's own mean and sd are noise about noise, so the OWN
# baseline is withheld. The TIER baseline is not -- it is estimated from every player in
# the role, and is exactly the measure that still works when a player has few games.
MIN_GAMES = 15

# How close to zero counts as "the line did not move". Lines are quoted in half points,
# so anything under one percent is float noise rather than a real move.
LINE_FLAT_TOL = 0.01

SQL = """
SELECT pg.player_id, pg.game_id, g.game_date, p.position,
       pg.minutes, pg.points, pg.fgm, pg.fga, pg.fta, pg.ftm,
       pg.rebounds, pg.rebounds_off, pg.assists, pg.steals, pg.blocks,
       pg.turnovers, pg.fouls, pg.usage_pct, pg.turnover_ratio,
       pg.touches, pg.passes, pg.distance, pg.contested_shots,
       pg.deflections, pg.loose_balls, pg.box_outs, pg.screen_assists,
       r.tier,
       b.close_line, b.close_under, b.open_line, b.open_under, b.open_offset_sec,
       pl.poll_line, pl.poll_under, pl.poll_offset_sec,
       s.salary, s.has_listed_salary,
       pc.instability_career, pc.gap_seasons
  FROM player_games pg
  JOIN games   g ON g.game_id   = pg.game_id
  JOIN players p ON p.player_id = pg.player_id
  LEFT JOIN odds_events e ON e.game_id = pg.game_id
  LEFT JOIN LATERAL (
      SELECT max(line)        FILTER (WHERE snapshot_role = 'close') AS close_line,
             max(under_price) FILTER (WHERE snapshot_role = 'close') AS close_under,
             max(line)        FILTER (WHERE snapshot_role = 'open')  AS open_line,
             max(under_price) FILTER (WHERE snapshot_role = 'open')  AS open_under,
             max(offset_from_tip_sec)
                              FILTER (WHERE snapshot_role = 'open')  AS open_offset_sec
        FROM prop_quotes q
       WHERE q.event_id  = e.event_id
         AND q.player_id = pg.player_id
         AND q.book      = %(book)s
  ) b ON true
  -- FALLBACK OPEN: the EARLIEST 'poll' snapshot (largest offset before tip), used as
  -- the opening observation ONLY where no T-12 'open' row exists (the substitution
  -- happens in load(), which also records provenance). Line and under come from the
  -- SAME row -- a LIMIT 1 lateral, not per-column aggregates -- and the ORDER BY is a
  -- total order so ties resolve identically on every run. No poll predates T-12, so
  -- rows that already have an 'open' cannot be affected (verified by
  -- pipeline.load_data.load_line_history --verify).
  LEFT JOIN LATERAL (
      SELECT q.line                AS poll_line,
             q.under_price         AS poll_under,
             q.offset_from_tip_sec AS poll_offset_sec
        FROM prop_quotes q
       WHERE q.event_id  = e.event_id
         AND q.player_id = pg.player_id
         AND q.book      = %(book)s
         AND q.snapshot_role = 'poll'
       ORDER BY q.offset_from_tip_sec DESC, q.line_last_update DESC,
                q.snapshot_requested DESC, q.line DESC
       LIMIT 1
  ) pl ON true
  -- LEFT, not inner: a handful of players have no salary row at all (mid-season
  -- waivers the end-of-season roster snapshot missed). An inner join would delete
  -- their games silently rather than leaving the salary NULL.
  LEFT JOIN player_salaries s
         ON s.player_id = pg.player_id AND s.season = %(season)s
  -- LEFT: career history exists for the propped population; a player without a row
  -- (fetch failed / never propped) stays NULL, which add_motive maps to NEUTRAL --
  -- missing history is absence, not evidence (invariant 1).
  LEFT JOIN player_career pc
         ON pc.player_id = pg.player_id
  LEFT JOIN player_game_residuals r
         ON r.player_id = pg.player_id AND r.game_id = pg.game_id
 WHERE pg.minutes > 0 AND pg.points IS NOT NULL
 ORDER BY g.game_date, pg.game_id, pg.player_id
"""

NUMERIC = ["minutes", "points", "fgm", "fga", "fta", "ftm", "rebounds",
           "rebounds_off", "assists", "steals", "blocks", "turnovers", "fouls",
           "usage_pct", "turnover_ratio", "close_line", "close_under",
           "open_line", "open_under", "open_offset_sec",
           "poll_line", "poll_under", "poll_offset_sec", "salary",
           "instability_career", "gap_seasons"] + EFFORT


def load(conn):
    d = pd.read_sql(SQL, conn, params={"book": config.BOOK, "season": config.SEASON})
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # EFFECTIVE OPEN. Where no T-12 'open' snapshot exists, the earliest 'poll'
    # snapshot stands in as the opening observation -- used AS-IS, with the window
    # RECORDED: open_offset_hours says how far before tip the observation actually
    # was (~12.06h for a true open, typically 2-3h for a poll), and open_source says
    # which kind of row supplied it. Provenance only -- neither column feeds a z or
    # a weight. ABSENCE STAYS ABSENCE: a row with neither an open nor a poll keeps
    # open_line/open_under NaN, never zero, so add_movement() still yields NaN there.
    has_open = d["open_line"].notna()
    use_poll = ~has_open & d["poll_line"].notna()
    d["open_line"] = d["open_line"].where(~use_poll, d["poll_line"])
    d["open_under"] = d["open_under"].where(~use_poll, d["poll_under"])
    offset_sec = d["open_offset_sec"].where(~use_poll, d["poll_offset_sec"])
    d["open_offset_hours"] = (offset_sec / 3600).round(3)
    d["open_source"] = np.select([has_open, use_poll], ["open", "poll"], default=None)
    d = d.drop(columns=["open_offset_sec", "poll_line", "poll_under",
                        "poll_offset_sec"])
    return d


# --- market movement --------------------------------------------------------

def add_movement(d):
    """The open/close pair, as fractional moves. NaN where no opening line exists."""
    open_line = d["open_line"].replace(0, np.nan)
    open_under = d["open_under"].replace(0, np.nan)
    d["line_move_pct"] = (d["close_line"] - d["open_line"]) / open_line
    d["under_move_pct"] = (d["close_under"] - d["open_under"]) / open_under

    # PRICE-ONLY MOVE: the under price shifted while the LINE HELD. A book absorbs
    # action two ways -- move the line, or reprice it -- and repricing is the quieter
    # and, in this data, the more common one. NULL unless the line was flat, so the
    # feature never mixes the two mechanisms.
    flat = d["line_move_pct"].abs() <= LINE_FLAT_TOL
    d["price_only_move"] = d["under_move_pct"].where(flat)
    d["margin_vs_line"] = d["points"] - d["close_line"]
    return d


# --- performance ------------------------------------------------------------

def game_score(d):
    """Hollinger Game Score -- eleven box-score inputs on one scale."""
    drb = d["rebounds"] - d["rebounds_off"].fillna(0)
    return (d["points"] + .4*d["fgm"] - .7*d["fga"] - .4*(d["fta"] - d["ftm"])
            + .7*d["rebounds_off"].fillna(0) + .3*drb + d["steals"]
            + .7*d["assists"] + .7*d["blocks"] - .4*d["fouls"] - d["turnovers"])


def add_performance(d):
    """Five components, equal weight: two measures against TWO baselines, plus shortfall.

    THE TWO BASELINES FAIL IN OPPOSITE DIRECTIONS, which is why both are kept.

      own    is soft for a player with many quiet games. Beasley 2024-01-06 is game_z
             -1.28 against himself but -1.68 against starters -- 3 points is worse than
             his own season makes it look.
      tier   is soft in reverse. The bench tier is full of six-minute end-of-rotation
             cameos, so Porter 2024-01-20 reads -0.17 own but +0.27 tier: "above average
             for a bench player", which is true and useless.

    Six ways of combining them were tested (experiments/tier_blend.py). Adding both as
    extra components was the ONLY one that improved the Beasley games without paying for
    it with the Porter games (Beasley -24.7%, Porter -2.3%). Tier-only scored best
    overall and is REJECTED: its gain is reweighting toward the player who supplies four
    of the six labels, not more signal.
    """
    def own_z(s):
        g = s.groupby(d["player_id"])
        return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)

    def tier_z(s):
        # tier is the season-average-minutes bucket (bench / rotation / starter) from
        # player_game_residuals -- a property of the PLAYER, not of the night.
        g = s.groupby(d["tier"])
        return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)

    d["game_score"] = game_score(d).round(2)
    d["game_z"] = own_z(d["game_score"]).round(3)
    d["game_z_tier"] = tier_z(d["game_score"]).round(3)

    # Missing components -> 0 rather than dropping the row. Hustle and tracking coverage
    # is uneven, and demanding a complete set would silently delete the games where an
    # endpoint stalled. 0 reads as "average", which is the honest default.
    d["effort_z"] = pd.concat([own_z(d[c]) for c in EFFORT],
                              axis=1).fillna(0).mean(axis=1).round(3)
    d["effort_z_tier"] = pd.concat([tier_z(d[c]) for c in EFFORT],
                                   axis=1).fillna(0).mean(axis=1).round(3)

    # SHORTFALL -- the only component with no floor.
    #
    # Every z is bounded below by the player's own mean: a 4-point-per-game player
    # scoring zero is -0.97 sd from himself while a star is -3.23, so any threshold on a
    # z is secretly a threshold on how much a player normally scores. shortfall is 1.000
    # for a zero-point game whoever produced it, because the denominator is the market's
    # forecast for THAT player in THAT game.
    d["shortfall"] = (1 - d["points"] / d["close_line"].replace(0, np.nan)) \
        .clip(0, 1).round(3)

    # STANDARDISED LEAGUE-WIDE, NOT WITHIN PLAYER. The raw quantity is ALREADY
    # player-relative through the line, so standardising it per-player again is double
    # normalisation and the second one undoes the first. Measured across all 349
    # zero-point games: raw shortfall is constant at 1.000, the league z is constant at
    # +3.143, and the within-player z ran +0.50 to +4.87 correlating +0.780 with the
    # player's scoring level -- Porter +1.199 against Dosunmu +4.040 for the same event.
    d["shortfall_z"] = ((d["shortfall"] - d["shortfall"].mean())
                        / d["shortfall"].std()).round(3)

    # MIN_GAMES withholds the OWN baseline only. A thin season makes a player's own mean
    # and sd unreliable, which is what the guard is for -- it says nothing about the tier
    # baseline, and blanking those would discard the one measure that survives.
    n = d.groupby("player_id")["points"].transform("size")
    d["n_player_games"] = n
    for c in ("game_z", "effort_z"):
        d.loc[n < MIN_GAMES, c] = np.nan

    # Oriented so HIGHER = worse night, then averaged over whatever is present.
    parts = {"game_z": -d["game_z"], "effort_z": -d["effort_z"],
             "shortfall_z": d["shortfall_z"]}
    num = sum(PERF_W[k] * v.fillna(0) for k, v in parts.items())
    den = sum(PERF_W[k] * v.notna().astype(float) for k, v in parts.items())
    d["performance"] = (num / den.replace(0, np.nan)).round(3)
    return d


# --- market -----------------------------------------------------------------

def add_market(d):
    """Four components, equal weight, all oriented so a LOW raw value scores HIGH --
    short price, small line, line drifting down, price shortening while the line holds
    are what under-side money produces.

    p_price carries the block. It is the only component that ever passed a test on its
    own: the de-vigged closing lean was monotonic across five buckets at z = 3.81 with
    100 percent coverage. line_move_pct tested BACKWARDS on its own (47.1% under-hit
    against a 52.7% baseline), and price_only_move, cross-book divergence and delta
    p_under were all null. They are retained at equal weight because the weight audit
    showed within-block weights do not change the ranking either way.

    NORMALISED BY THE WEIGHT PRESENT. A row with no opening line is judged on price and
    line alone. Dividing by the full total instead would score it as though the two
    movement terms had been observed and found neutral, which is a claim about the
    market that nobody made.
    """
    z = lambda s: (s - s.mean()) / s.std()
    d["p_price"] = d["close_under"].rank(pct=True).round(4)
    d["p_line"] = d["close_line"].rank(pct=True).round(4)

    parts = {"p_price":  z(1 - d["p_price"]),
             "p_line":   z(1 - d["p_line"]),
             "line_mv":  z(-d["line_move_pct"]),
             "price_mv": z(-d["price_only_move"])}
    for k, v in parts.items():
        d[f"mk_{k}"] = v.round(3)
    num = sum(MARKET_W[k] * v.fillna(0) for k, v in parts.items())
    den = sum(MARKET_W[k] * v.notna().astype(float) for k, v in parts.items())
    d["market"] = (num / den.replace(0, np.nan)).round(3)
    d["n_market"] = sum(v.notna().astype(int) for v in parts.values())
    return d


# --- motive -----------------------------------------------------------------

def add_motive(d):
    """What a player RISKED rather than what he did: salary (0.75) plus career
    instability (0.25), blended per MOTIVE_W, then z-scored.

    SALARY COMPONENT -- percentile, not dollars. Salary is skewed +1.78 with 67 percent
    of players below the mean, so a z-score spends 92 percent of its range on the top
    half of earners: the entire bottom half, from a two-way deal to the $4.3M median,
    compresses into a 0.38-wide band. That is backwards for an axis meant to
    discriminate among the underpaid. rank(pct=True) is uniform by construction and
    gives the 37 players on the exact $2.02M minimum one identical value rather than
    spreading them arbitrarily.

    UNLISTED SALARY SCORES MAXIMUM, NOT MISSING. basketball-reference publishes no figure
    for two-way and 10-day contracts, and those are the lowest-paid players on any
    roster. Treating NULL as unknown would drop exactly the population this axis exists
    to surface. Jontay Porter is one of these rows.

    INSTABILITY COMPONENT -- instability_career from player_career: (career team
    changes + 2*gap seasons) / (span seasons + 1). The RATE form, so a chronic
    churner (Porter: 1.000) outranks a long-career journeyman whose many moves are
    spread over many seasons; gaps weigh double (a gap season means no team wanted
    him) and the +1 pseudo-season stops a single-season trade from posting the
    league's most extreme rate -- the full rationale lives in load_career.py. Percentiled over the full propped population, like
    salary; NaN mapped to NEUTRAL 0.5 AFTER ranking, never max and never zero --
    a player with no career row has missing history, not evidence of anything
    (invariant 1). Contrast with unlisted salary, where absence of a published
    figure IS evidence of a two-way deal.

    pd.Series around np.where, not the bare array: numpy's mean and std PROPAGATE NaN
    while pandas skips it, so the handful of players with no salary row at all would
    otherwise turn the entire column into NaN.
    """
    pct = d["salary"].rank(pct=True)
    salary_comp = pd.Series(np.where(d["has_listed_salary"] == False, 1.0, 1 - pct),
                            index=d.index)
    instability_pct = d["instability_career"].rank(pct=True).fillna(0.5)
    raw = MOTIVE_W["salary"] * salary_comp + MOTIVE_W["instability"] * instability_pct
    d["motive"] = ((raw - raw.mean()) / raw.std()).round(3)
    return d


# --- main -------------------------------------------------------------------

def build(d):
    d = add_movement(d)
    d = add_performance(d)

    # PROPPED ONLY, AND ONLY NOW. Every z above is built on each player's whole season;
    # filtering earlier would make his baseline the mean of his propped games, which are
    # systematically his higher-profile nights.
    d = d[d["close_line"].notna()].copy()

    d = add_market(d)
    d = add_motive(d)

    # THE BLOCKS ARE COMBINED RAW, then the RESULT is put on 0-100.
    #
    # THE 0-100 SCALE IS PRESENTATION ONLY. A percentile is a strictly monotone
    # transform, so `score_100` and `score` produce the IDENTICAL ordering -- rank 1 is
    # rank 1 either way, and nothing about which games surface changes.
    #
    # THE ALTERNATIVE WAS TESTED AND REJECTED. Percentiling each BLOCK before combining
    # them makes the stated weights literally true, because the blocks do not share a
    # spread: motive is a z-score (sd exactly 1) while performance and market are MEANS
    # of z-scores (sd 0.677 and 0.636), so their effective shares are 40.9 / 25.6 / 33.5
    # rather than the stated 45 / 30 / 25. That is a real discrepancy and it is recorded
    # here rather than hidden -- motive carries more of the score than the number says.
    #
    # It is not corrected, deliberately. Correcting it reordered the ranking (spearman
    # 0.9753, 84 of the top 100 retained) and moved the flagged games from a geometric
    # mean rank of 581 to 592. The 0-100 change was wanted as an aesthetic one, so it is
    # kept aesthetic; the weight-fidelity question is a separate decision about the
    # methodology and should be taken on its own merits, not smuggled in with a display
    # change. experiments/weight_audit.py is where that argument belongs.
    d["score"] = (BLOCK_W["performance"] * d["performance"]
                  + BLOCK_W["market"] * d["market"]
                  + BLOCK_W["motive"] * d["motive"]).round(3)


    # THE HEADLINE SCORE, 0-100. A weighted mean of three numbers that are each already
    # 0-100, so it lands on the same scale with no second rescaling -- and every weight
    # in it means exactly what it says.
    #
    # WHY PERCENTILE AND NOT A MIN-MAX RESCALE, for each block:
    #
    #   percentile    exactly "worse than X% of propped games". No distributional
    #                 assumption. THIS ONE.
    #   normal CDF    agrees with the percentile to within 1.70 points across the whole
    #                 range (corr 0.99972) because the score is near-normal -- skew
    #                 +0.13, kurtosis -0.31 -- so it buys nothing the percentile lacks.
    #   min-max       REJECTED. One extreme game owns the top of the scale: the 99th
    #                 percentile of the raw score reached only 86.2 and the 99.9th only
    #                 95.3, so 99% of the range went to the last 16 games. It also moves
    #                 for every row whenever a new extreme appears.
    #
    # Percentiles are taken over all 15,498 PROPPED games, not over the shortlist. A
    # percentile against the shortlist would read the same and mean something far
    # narrower: "worse than X% of games that already survived six cuts".
    # LINEAR rescale, not a percentile.
    #
    # A percentile is uniform over the POPULATION, which means it is densest where the
    # games are -- the middle -- and flattest at the tail, which is the only part anyone
    # reads. Across the top 200 the percentile spanned 1.28 points (98.72 to 100.00):
    # rank 1 printed 100.00 and rank 10 printed 99.94, so the list looked like it had no
    # opinion. A linear map preserves the raw score's actual shape, and the same top 200
    # spans 15.07 points.
    #
    # The cost, stated: the endpoints are set by the two most extreme games, so one
    # freak result compresses everyone else, and adding a season moves every number. The
    # raw `score` is kept for exactly that reason. A percentile has the opposite failure
    # -- stable ordering, useless spacing -- and spacing is what a reviewer reads.
    lo, hi = d["score"].min(), d["score"].max()
    d["score_100"] = ((d["score"] - lo) / (hi - lo) * 100).round(2)
    for c in ("performance", "market", "motive"):
        lo_, hi_ = d[c].min(), d[c].max()
        d[f"{c}_100"] = ((d[c] - lo_) / (hi_ - lo_) * 100).round(2)
    return d


# TWO TABLES, ONE ROW PER PLAYER-GAME IN EACH, SPLIT BY WHAT THEY MEAN.
#
#   player_game_features   RAW. What was measured -- box score, lines, prices, and the
#                          movement between open and close. Nothing standardised. This
#                          is the layer you check when you distrust a number.
#   player_game_z          STANDARDISED. The five performance components, the four
#                          market components, and the three blocks they combine into.
#                          Every column here is a comparison, not a measurement.
#
# The split matters because they go stale for different reasons. A raw value is wrong
# only if the ingest was wrong; a z is wrong if the BASELINE POPULATION changed, which
# happens every time a game is added. Keeping them apart means a re-standardisation
# never rewrites the evidence it was computed from.
FEATURE_COLS = ["player_id", "game_id", "game_date", "position", "tier",
                "minutes", "points", "fga", "usage_pct", "turnover_ratio",
                "distance", "touches", "passes", "rebounds", "assists",
                "close_line", "close_under", "open_line", "open_under",
                "open_offset_hours", "open_source",
                "line_move_pct", "under_move_pct", "price_only_move",
                "margin_vs_line", "game_score", "shortfall",
                "salary", "has_listed_salary", "n_player_games",
                "instability_career", "gap_seasons"]

Z_COLS = ["player_id", "game_id", "game_date",
          "game_z", "effort_z", "game_z_tier", "effort_z_tier", "shortfall_z",
          "p_price", "p_line", "mk_p_price", "mk_p_line", "mk_line_mv",
          "mk_price_mv", "performance", "market", "motive", "score",
          "score_100", "performance_100", "market_100", "motive_100",
          "n_market", "n_player_games"]

# ALTER, not just CREATE. `CREATE TABLE IF NOT EXISTS` is a NO-OP on an existing table,
# so a newly added metric column would silently never reach the database -- which has
# already happened once in this project (margin_vs_line_league_z was computed for weeks
# and never written).
FEATURE_DDL = {"tier": "TEXT", "position": "TEXT", "passes": "NUMERIC(8,2)",
               "game_score": "NUMERIC(8,2)", "shortfall": "NUMERIC(6,3)",
               "salary": "BIGINT", "has_listed_salary": "BOOLEAN",
               "n_player_games": "INTEGER",
               # PROVENANCE of the opening observation, not model inputs: how many
               # hours before tip it was taken, and whether it was a true T-12
               # 'open' or the earliest 'poll' standing in for one. NULL where no
               # opening observation exists at all.
               "open_offset_hours": "NUMERIC(8,3)", "open_source": "TEXT",
               # Career-instability inputs carried from player_career for provenance:
               # instability_career feeds add_motive (per MOTIVE_W) and NOTHING else --
               # no z, no cut. gap_seasons is its most opaque ingredient, kept
               # inspectable. NULL where the player has no career row.
               "instability_career": "NUMERIC(6,3)", "gap_seasons": "INTEGER"}

# Everything standardised, plus the old four-baseline sweep's per-stat z columns. All of
# it either moved to player_game_z or belongs to a layer that no longer exists.
MOVED_TO_Z = [c for c in Z_COLS if c not in ("player_id", "game_id", "game_date",
                                             "n_player_games")] + [
    "minutes_z", "points_z", "fga_z", "usage_pct_z", "turnover_ratio_z",
    "distance_z", "touches_z", "rebounds_z", "assists_z",
    "close_line_z", "close_under_z", "line_move_pct_z", "under_move_pct_z",
    "price_only_move_z", "margin_vs_line_z", "margin_vs_line_league_z",
    "combined_score", "n_features", "n_baseline", "n_league", "baseline_mode"]

Z_DDL = """
CREATE TABLE IF NOT EXISTS player_game_z (
    player_id      INTEGER NOT NULL REFERENCES players (player_id),
    game_id        TEXT    NOT NULL REFERENCES games (game_id),
    game_date      DATE    NOT NULL,

    -- PERFORMANCE: two measures against two baselines, plus the market's.
    game_z         NUMERIC(8,3),   -- Game Score vs his own season
    effort_z       NUMERIC(8,3),   -- 9 involvement stats vs his own season
    game_z_tier    NUMERIC(8,3),   -- Game Score vs everyone in his role
    effort_z_tier  NUMERIC(8,3),   -- the same 9 stats vs everyone in his role
    shortfall_z    NUMERIC(8,3),   -- LEAGUE-wide; the raw value is already
                                   -- player-relative through the line

    -- MARKET: percentiles, then the oriented z of each.
    p_price        NUMERIC(8,4),
    p_line         NUMERIC(8,4),
    mk_p_price     NUMERIC(8,3),
    mk_p_line      NUMERIC(8,3),
    mk_line_mv     NUMERIC(8,3),   -- NULL when no opening line was posted
    mk_price_mv    NUMERIC(8,3),   -- NULL unless the line held and the price moved

    -- THE THREE BLOCKS AND THE SCORE.
    performance    NUMERIC(8,3),
    market         NUMERIC(8,3),
    motive         NUMERIC(8,3),
    score          NUMERIC(8,3),

    -- 0-100 PERCENTILE of each of the four, over all propped games. "worse than X% of
    -- them". The raw values above are population-independent; these are not, and that
    -- is the trade for a number a human can read.
    score_100        NUMERIC(6,2),
    performance_100  NUMERIC(6,2),
    market_100       NUMERIC(6,2),
    motive_100       NUMERIC(6,2),

    n_market       INTEGER,        -- how many of the 4 market components were present
    n_player_games INTEGER,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS player_game_z_score_idx
    ON player_game_z (score DESC NULLS LAST);
"""


# CREATE TABLE IF NOT EXISTS is a NO-OP on an existing table, so Z_DDL alone can never
# add a column to a table that already exists. Every column in Z_COLS therefore needs an
# ALTER too. Built from Z_COLS rather than hand-listed, so adding a component to the
# scoring cannot silently fail to reach the database -- which is precisely how this
# broke the first time score_100 was added.
Z_ALTER = {c: ("INTEGER" if c.startswith("n_") else
               "NUMERIC(6,2)" if c.endswith("_100") else
               "NUMERIC(8,4)" if c.startswith("p_") else "NUMERIC(8,3)")
           for c in Z_COLS if c not in ("player_id", "game_id", "game_date")}


def ensure_tables(conn):
    """Idempotent DDL. Safe to run on every pass.

    player_game_z is REBUILT if it still carries `baseline_mode`. The old table held one
    row per (player, game, baseline) across four baselines -- 105,604 rows of a
    four-baseline sweep that this layer no longer computes. Its primary key has changed,
    so ALTER cannot migrate it and the stale rows cannot be left in place: they would
    mix a retired definition of `score` with the current one under the same column name.
    """
    with conn.cursor() as cur:
        for col, typ in FEATURE_DDL.items():
            cur.execute(f"ALTER TABLE player_game_features "
                        f"ADD COLUMN IF NOT EXISTS {col} {typ}")
        # Standardised columns belong in player_game_z. Dropped from the raw table
        # rather than left behind: a duplicated column that nothing writes any more is
        # a stale copy waiting to be read by mistake, and `SELECT f.*` downstream would
        # collide with the live version.
        for col in MOVED_TO_Z:
            cur.execute(f"ALTER TABLE player_game_features DROP COLUMN IF EXISTS {col}")
        cur.execute("""SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'player_game_z'
                          AND column_name = 'baseline_mode'""")
        if cur.fetchone():
            cur.execute("SELECT count(*) FROM player_game_z")
            print(f"   migrating player_game_z: dropping {cur.fetchone()[0]:,} rows "
                  f"of the retired four-baseline sweep")
            cur.execute("DROP TABLE player_game_z")
        cur.execute(Z_DDL)
        for col, typ in Z_ALTER.items():
            cur.execute(f"ALTER TABLE player_game_z "
                        f"ADD COLUMN IF NOT EXISTS {col} {typ}")
        _verify(cur, "player_game_features", FEATURE_COLS)
        _verify(cur, "player_game_z", Z_COLS)
    conn.commit()


def _verify(cur, table, declared):
    """Fail loudly if a declared column did not reach the table.

    The ALTER pass above infers a type from the column NAME. That heuristic can be
    wrong, and `ADD COLUMN IF NOT EXISTS` is silent when it is a no-op -- so without
    this check a new component could be computed, rounded, logged, and then quietly
    dropped on the way to the database. That failure has happened three times in this
    project. Raising here converts it into an error at the moment it is introduced.
    """
    cur.execute("""SELECT column_name FROM information_schema.columns
                    WHERE table_name = %s""", (table,))
    missing = set(declared) - {r[0] for r in cur.fetchall()}
    if missing:
        raise SystemExit(
            f"{table}: declared columns never reached the database: "
            f"{sorted(missing)}. The type inferred from the column name is probably "
            f"wrong -- add an explicit entry rather than widening the heuristic.")


def main(dry=True):
    with db.connect() as conn:
        raw = load(conn)
        print(f"played player-games (minutes > 0) : {len(raw):,}")
        print(f"  with a CLOSING line             : {raw['close_line'].notna().sum():,}"
              f"   <- the output population")
        n_src = raw["open_source"].value_counts()
        print(f"  with an OPENING observation     : {raw['open_line'].notna().sum():,}"
              f"   ('open' {n_src.get('open', 0):,}, poll fallback {n_src.get('poll', 0):,})")
        print(f"  distinct players                : {raw['player_id'].nunique():,}")

        d = build(raw)
        print(f"\npropped player-games              : {len(d):,}"
              f"   ({d['player_id'].nunique()} players)")
        print(f"  no opening line, KEPT and not penalised : "
              f"{int(d['line_move_pct'].isna().sum()):,} "
              f"({d['line_move_pct'].isna().mean():.0%})")
        print(f"  no price-only move, KEPT                : "
              f"{int(d['price_only_move'].isna().sum()):,} "
              f"({d['price_only_move'].isna().mean():.0%})")

        print(f"\nBLOCK COVERAGE")
        for c in ("performance", "market", "motive", "score"):
            print(f"   {c:<14}{d[c].notna().sum():>7,} / {len(d):,}"
                  f"   mean {d[c].mean():+.3f}  sd {d[c].std():.3f}")
        print(f"\n   performance ~ market  {d['performance'].corr(d['market']):+.3f}"
              f"   <- independent, which is why they are kept apart")

        if dry:
            db.dry_notice()
            return d

        print()
        ensure_tables(conn)
        for table, cols in (("player_game_features", FEATURE_COLS),
                            ("player_game_z", Z_COLS)):
            rows = d[cols].replace({np.nan: None}).to_dict("records")
            n = db.upsert(conn, table, rows, conflict=["player_id", "game_id"])
            print(f"   upserted {n:,} rows -> {table} "
                  f"({db.count(conn, table):,} total)")
        return d


if __name__ == "__main__":
    main(dry=db.is_dry())
