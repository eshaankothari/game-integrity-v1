"""L5: raw stats + closing lines -> `player_game_z`. Pure computation, no APIs.

SUMMARY: Turns each player-game into z-scores against a chosen baseline, adds the
miss against the market's own forecast, and reduces the lot to one signed score
where MORE NEGATIVE = worse.

Everything here is deliberately plain pandas, one small function per step, so any
single step can be changed without touching the others. The interesting decisions
are all in the constants at the top of the file.

THE CENTRAL FINDING THIS LAYER IS BUILT AROUND
----------------------------------------------
Damian Lillard, 2023-10-29, FanDuel line 30.5, scored 6 -- the worst miss in the
data. His league-wide z-scores say he was FINE:

    minutes_z +0.60   usage_z +0.69   touches_z +1.44   points_z -0.54

Because the league mean of 10.9 points includes bench players, a star's disaster is
somebody else's ordinary night. League-wide standardisation is close to useless for
exactly the players worth watching.

The same row against the market's forecast:  margin_vs_line_z = -3.81.

So `margin_vs_line` is the workhorse feature, and `player_to_date` -- "unusual FOR
HIM" -- matters more than any league-relative view. The nine stat z-scores describe
HOW a performance was unusual; the margin says WHETHER it was.

    python standardize.py                       # DRY: report, write nothing
    python standardize.py run                   # compute all three baselines
    python standardize.py run --mode player_to_date
"""
import sys
import warnings

import numpy as np
import pandas as pd

import config
import db

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

# --- the nine features, RAW ------------------------------------------------
# No per-36 rate adjustment, by design: minutes_z is itself a feature, so playing
# time stays available to the model instead of being divided out, and for an
# integrity question a low RAW count is the event of interest.
#
# Consequence to watch: the volume stats now correlate strongly with minutes, so an
# equal-weight mean counts playing time several times over. Check `correlations()`
# before trusting combined_score too far.
FEATURES = ["minutes", "points", "fga", "usage_pct", "turnover_ratio",
            "distance", "touches", "rebounds", "assists"]

# Sign toward UNDERPERFORMANCE. Multiply the z by this and every feature agrees
# that more negative = worse, which is what lets them be averaged at all.
# Only turnover_ratio is inverted: more turnovers is bad, so its raw z points the
# wrong way.
SIGNS = {f: 1.0 for f in FEATURES}
SIGNS["turnover_ratio"] = -1.0
SIGNS["margin_vs_line"] = 1.0        # already negative when a player fell short
SIGNS["line_move_pct"] = 1.0         # line drifting DOWN = money on the under
SIGNS["under_move_pct"] = 1.0        # under price shortening = same story
SIGNS["price_only_move"] = 1.0       # repriced without moving the line
SIGNS["close_under"] = 1.0           # cheap under = the market already leaned that way

# close_line: HIGH = suspicious. INVERTED relative to every other feature, and set
# from the data rather than from a mechanism.
#
# Measured on 2,767 player-games, bucketed by close_line_z under a player-relative
# baseline:
#
#     lowest 25%  (line below his norm)   avg margin +1.31   43.3% went under
#     highest 25% (line above his norm)   avg margin -0.97   58.8% went under
#
# Monotonic across all four buckets. A line set above a player's own norm precedes
# an under far more often than one set below it.
#
# CAVEAT, deliberately recorded. The most likely driver is MEAN REVERSION, not
# integrity: books post a high line after a hot streak or a soft matchup, and the
# player regresses. So this feature partly measures an ordinary market property, and
# a flag it contributes to may be regression dressed up as an anomaly. Treat rows
# whose score leans heavily on close_line_z with more suspicion than the number
# alone suggests.
#
# Only meaningful under a PLAYER-RELATIVE baseline -- league-wide, close_line just
# encodes player quality. And it is a LEVEL, so it cannot separate "drifted down on
# under money" from "opened low because he was cold". `line_move_pct` tests that
# directly and should be preferred once the opening pass runs.
SIGNS["close_line"] = -1.0

# EQUAL WEIGHTS, deliberately, for now.
#
# One caveat to keep in view. Averaging the nine stats alone gave Lillard's 6-point
# night on a 30.5 line a score of +0.317 -- POSITIVE -- because against the LEAGUE
# he still looked like a star: heavy minutes, high usage, plenty of touches. The
# league mean of ~11 points includes benchwarmers, so a star's disaster reads as an
# ordinary evening and the stat block cancels the one thing that went wrong.
#
# Measured against HIS OWN season the same row is points_z -2.23, minutes_z -1.44,
# distance_z -1.36 -- every feature negative. So the fix is choosing the right
# BASELINE, not re-weighting the features. Equal weights are fine once the baseline
# is player-relative. Revisit only if the distributions say otherwise.
WEIGHTS = {f: 1.0 for f in FEATURES}
WEIGHTS.update({f: 1.0 for f in
                ("margin_vs_line", "close_line", "close_under",
                 "line_move_pct", "under_move_pct", "price_only_move")})

# Betting-derived features. Only the first is computable today; the other two need
# `open` rows in prop_quotes, which the opening-line pass has not run yet. They are
# wired through anyway -- opens are ROWS, not new columns, so when that pass runs
# these populate with no code change here.
BET_FEATURES = ["margin_vs_line", "close_line", "close_under",
                "line_move_pct", "under_move_pct", "price_only_move"]
BOOK = config.BOOK          # primary book for the analysis; others are fallback

BASELINES = ("league_season", "league_to_date", "player_to_date", "player_season")

# THE PRIMARY OUTPUT USES A DIFFERENT BASELINE PER FEATURE. Not one mode applied
# uniformly -- the two families of feature ask different questions.
#
#   STATS -> that player's own season.  "Did HE play unlike himself tonight?"
#       League-wide is near-useless here: Lillard scoring 6 was points_z -0.53
#       against the league (which includes benchwarmers) but -2.23 against himself.
#
#   BETTING -> the whole league, EXCEPT margin_vs_line.
#
# WHY margin_vs_line MOVED TO player_season. The argument for the league was that the
# line already encodes what was expected tonight, so the miss is pre-normalised. That
# is true of the LEVEL and false of the SPREAD. A 28.5-point line and a 5.5-point line
# do not miss by comparable amounts: the star has 28 points of room to fall and the
# reserve has 5, so on a league scale the star's miss is always larger and stars
# monopolise the top of every ranking. Isolation Forest put Anthony Edwards, Jalen
# Brunson and Devin Booker in its top five on exactly this.
#
# Per-player, the question becomes "how big a miss is this FOR HIM, against how much
# he usually misses by", which is the comparison actually wanted and puts a reserve
# blanking a 5.5 line on equal footing with a star blanking 28.5.
#
# The league version is kept as margin_vs_line_league_z rather than dropped -- it is
# the honest answer to "how many points was this in absolute terms", just not the
# right ranking key.
FEATURE_BASELINE = {**{f: "player_season" for f in FEATURES},
                    **{f: "league_season" for f in BET_FEATURES},
                    "margin_vs_line": "player_season"}

# `player_season` is the batch default: a player's whole season, this game included.
# It leaks slightly -- the game sits in its own baseline -- but the distortion is
# ~z/(n-1), which at Lillard's 70 games moved points_z by only 0.10 sd. Below
# MIN_BASELINE games that self-masking stops being negligible, hence the floor.
# How close to zero counts as "the line did not move". Lines are quoted in half
# points, so anything under 1 percent is float noise, not a real move.
LINE_FLAT_TOL = 0.01

MIN_BASELINE = 5


# --- step 1: load -----------------------------------------------------------

def load(conn):
    """Every player-game that ACTUALLY PLAYED, with its FanDuel closing line.

    DNPs are dropped here. They are not missing data -- the advanced and track
    endpoints emit real zeros for a rostered player who did not appear, so leaving
    them in drags every league mean toward zero for ~20% of rows.

    The line is LEFT JOINed: a player-game with no prop still gets stat z-scores,
    it just has no margin. Dropping those rows would shrink the baseline population
    to only propped (i.e. higher-usage) players and bias every z-score.

    Betting signals are pivoted from LONG to WIDE here. prop_quotes stores one ROW
    per probe (`snapshot_role` = 'open' | 'close' | 'poll'), so open and close are
    two rows, not two columns. FILTER pivots them into one row per player-game --
    which is why adding opening lines later needs no change to this query: those
    rows simply start existing and the `open_*` columns stop being NULL.
    """
    sql = """
        SELECT pg.player_id, pg.game_id, g.game_date, p.position,
               pg.minutes, pg.points, pg.fga, pg.usage_pct, pg.turnover_ratio,
               pg.distance, pg.touches, pg.rebounds, pg.assists,
               b.close_line, b.close_under, b.open_line, b.open_under
        FROM player_games pg
        JOIN games   g ON g.game_id   = pg.game_id
        JOIN players p ON p.player_id = pg.player_id
        LEFT JOIN odds_events e ON e.game_id = pg.game_id
        LEFT JOIN LATERAL (
            SELECT max(line)        FILTER (WHERE snapshot_role = 'close') AS close_line,
                   max(under_price) FILTER (WHERE snapshot_role = 'close') AS close_under,
                   max(line)        FILTER (WHERE snapshot_role = 'open')  AS open_line,
                   max(under_price) FILTER (WHERE snapshot_role = 'open')  AS open_under
            FROM prop_quotes q
            WHERE q.event_id  = e.event_id
              AND q.player_id = pg.player_id
              AND q.book      = %(book)s
        ) b ON true
        WHERE pg.minutes IS NOT NULL
        ORDER BY g.game_date, pg.game_id, pg.player_id
    """
    df = pd.read_sql(sql, conn, params={"book": BOOK})
    for f in FEATURES + ["close_line", "close_under", "open_line", "open_under"]:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


# --- step 2: the market's forecast -----------------------------------------

def add_bet_features(df):
    """The five betting signals, derived from the open/close pair.

        pts_close_line       close_line          available now
        pts_close_under      close_under         available now
        pts_open_under       open_under          needs the opening pass
        pts_line_move_pct    (close-open)/open   needs the opening pass
        pts_under_move_pct   (close-open)/open   needs the opening pass

    plus margin_vs_line, which is the most informative of the lot: the closing line
    is the market's forecast for THIS player TONIGHT, already adjusted for opponent,
    role and expected minutes, so the miss measures performance against a
    personalised expectation instead of a league average that mixes stars with
    benchwarmers.

    The three open-dependent columns compute to NaN today and will populate the
    moment `open` rows land in prop_quotes -- no change needed here.
    """
    df["margin_vs_line"] = df["points"] - df["close_line"]

    # Guard against divide-by-zero: a 0 opening line would be nonsense anyway, but
    # it would propagate as inf rather than NaN and poison the z-scores downstream.
    open_line = df["open_line"].replace(0, np.nan)
    open_under = df["open_under"].replace(0, np.nan)
    df["line_move_pct"] = (df["close_line"] - df["open_line"]) / open_line
    df["under_move_pct"] = (df["close_under"] - df["open_under"]) / open_under

    # PRICE-ONLY MOVE: the under price shifted while the LINE HELD.
    #
    # A book absorbs incoming action two ways -- move the line, or reprice it. Moving
    # a line is the bigger decision, so repricing alone is the quieter signal, and in
    # this data it is also the MORE COMMON one: 913 player-book rows had a flat line
    # with a moved price, against 809 that moved the line at all.
    #
    # That matters because it means the null result on line_move_pct was partly a
    # measurement problem. Bucketing every flat-line row as "no movement" lumped
    # hundreds of repriced markets in with genuinely untouched ones. Separating them
    # gives the demand hypothesis a fair test.
    #
    # NULL unless the line was flat, so the feature never mixes the two mechanisms.
    # Sign matches under_move_pct: a FALLING under price means money on the under.
    line_flat = df["line_move_pct"].abs() <= LINE_FLAT_TOL
    df["price_only_move"] = df["under_move_pct"].where(line_flat)
    return df


# --- step 3: baselines ------------------------------------------------------
#
# Each returns (mean, sd, n) aligned to df's rows. Same shape, different
# population -- which is what lets batch and live share one code path.

def _stats_whole(series):
    """Whole-population mean/sd, broadcast to every row."""
    return (pd.Series(series.mean(), index=series.index),
            pd.Series(series.std(),  index=series.index),
            pd.Series(series.notna().sum(), index=series.index))


def _stats_prior_by_date(df, col):
    """Mean/sd over every player-game on a STRICTLY EARLIER date.

    Rows sharing a date must share a baseline, so this aggregates per date, takes a
    running total, then shifts by one date. Shifting by ROW instead would let games
    earlier in the same evening leak into games later that evening.
    """
    daily = df.groupby("game_date")[col].agg(
        n="count", s="sum", ss=lambda x: (x.astype(float) ** 2).sum())
    cum = daily.cumsum().shift(1)                      # strictly prior dates
    mean = cum["s"] / cum["n"]
    var = (cum["ss"] / cum["n"] - mean ** 2) * cum["n"] / (cum["n"] - 1)
    sd = np.sqrt(var.clip(lower=0))
    return (df["game_date"].map(mean), df["game_date"].map(sd),
            df["game_date"].map(cum["n"]))


def _stats_prior_by_player(df, col):
    """Mean/sd over that PLAYER's earlier games only -- 'unusual for him'.

    cumsum minus the current value gives the prior sum; cumcount gives the prior
    count (0 on a player's first game, which correctly yields no baseline).
    """
    v = df[col].astype(float)
    g = df.groupby("player_id")
    n = g.cumcount()
    s = g[col].cumsum() - v
    ss = v.pow(2).groupby(df["player_id"]).cumsum() - v.pow(2)
    mean = s / n
    var = (ss / n - mean ** 2) * n / (n - 1)
    return mean, np.sqrt(var.clip(lower=0)), n


def _stats_player_season(df, col):
    """Mean/sd over ALL of that player's games, including this one.

    The batch default, and by far the most informative of the four. Against the
    league, Lillard's 6-point night read minutes_z +0.61 / touches_z +1.40 -- a star
    having a normal evening. Against his own season it reads minutes_z -1.44,
    points_z -2.23, distance_z -1.36: every feature negative, which is the true
    picture. The nine stat features were never broken, they were being compared
    against the wrong population.

    It does leak -- the game is inside its own baseline -- so an extreme value pulls
    the mean toward itself and shrinks its own z. The bias is roughly z/(n-1): at 70
    games it moved points_z by 0.10 sd. MIN_BASELINE keeps it out of the range where
    that stops being negligible.

    Not usable live: it needs games that have not happened yet. `player_to_date` is
    its live-safe counterpart.
    """
    g = df.groupby("player_id")[col]
    return (g.transform("mean"), g.transform("std"), g.transform("count"))


def baseline_stats(df, col, mode):
    if mode == "league_season":
        return _stats_whole(df[col])
    if mode == "league_to_date":
        return _stats_prior_by_date(df, col)
    if mode == "player_to_date":
        return _stats_prior_by_player(df, col)
    if mode == "player_season":
        return _stats_player_season(df, col)
    raise ValueError(f"unknown baseline mode {mode!r}")


# --- step 4: z-scores -------------------------------------------------------

def zscore(df, mode):
    """Add <feature>_z for every feature, plus margin_vs_line_z, for one mode."""
    out = df.copy()
    ns = []
    for col in FEATURES + BET_FEATURES:
        mean, sd, n = baseline_stats(out, col, mode)
        # sd == 0 means a constant baseline; the z is undefined, not zero.
        z = (out[col] - mean) / sd.replace(0, np.nan)
        # Too few prior games and the "z" is just noise about noise.
        out[f"{col}_z"] = z.where(n >= MIN_BASELINE)
        ns.append(n)
    # Smallest baseline across features that HAVE data. Features with no data at
    # all (line_move_pct until the opening pass runs) are excluded, otherwise their
    # n=0 becomes the minimum and every row reports a baseline of zero.
    counts = pd.concat(ns, axis=1)
    counts = counts.loc[:, counts.max() > 0]
    out["n_baseline"] = (counts.min(axis=1).astype("Int64")
                         if not counts.empty else pd.NA)
    out["baseline_mode"] = mode
    return out


# --- step 5: one signed score ----------------------------------------------

def combine(df, features=None):
    """Mean of the sign-oriented z's. More negative = worse.

    Averaging whatever is present rather than requiring all nine, because hustle
    and track coverage is uneven -- demanding a complete set would silently drop
    the games where an endpoint stalled.
    """
    features = features or (FEATURES + BET_FEATURES)
    cols = [f"{f}_z" for f in features if f"{f}_z" in df]
    oriented = pd.DataFrame({c: df[c] * SIGNS[c[:-2]] * WEIGHTS[c[:-2]] for c in cols})
    df["combined_score"] = oriented.mean(axis=1, skipna=True)
    df["n_features"] = oriented.notna().sum(axis=1)
    return df


# --- the primary output: mixed baselines, raw AND z side by side ------------

RAW_COLS = (FEATURES
            + ["close_line", "close_under", "open_line", "open_under",
               "margin_vs_line", "line_move_pct", "under_move_pct",
               "price_only_move"])
Z_COLS = [f"{f}_z" for f in FEATURES + BET_FEATURES] + ["margin_vs_line_league_z"]


def build_features(df):
    """One row per player-game: raw values AND z-scores, each on its own baseline.

    Unlike `zscore`, which applies ONE mode to everything, this reads
    FEATURE_BASELINE per column -- stats against the player, betting against the
    league. The raw value sits next to its z so a row is readable without
    re-joining player_games.
    """
    out = df.copy()
    n_player = n_league = None

    for col in FEATURES + BET_FEATURES:
        mode = FEATURE_BASELINE[col]
        mean, sd, n = baseline_stats(out, col, mode)
        z = (out[col] - mean) / sd.replace(0, np.nan)
        out[f"{col}_z"] = z.where(n >= MIN_BASELINE)
        # Record one baseline size per FAMILY; every member of a family shares a
        # population, so the first non-empty one is representative.
        if mode == "player_season" and n_player is None:
            n_player = n
        if mode == "league_season" and n_league is None and n.max() > 0:
            n_league = n

    # The league-scaled miss, kept alongside the player-scaled one. Both are real
    # answers to different questions -- "how many points was this" versus "how big a
    # miss is this for him" -- and carrying both means the choice of ranking key stays
    # a query-side decision rather than being baked in here.
    mean, sd, n = baseline_stats(out, "margin_vs_line", "league_season")
    z = (out["margin_vs_line"] - mean) / sd.replace(0, np.nan)
    out["margin_vs_line_league_z"] = z.where(n >= MIN_BASELINE)

    out["n_player_games"] = (n_player.astype("Int64") if n_player is not None else pd.NA)
    out["n_league"] = (n_league.astype("Int64") if n_league is not None else pd.NA)
    return out


# --- reporting --------------------------------------------------------------

def correlations(df):
    """How much the features duplicate each other -- see the FEATURES note."""
    cols = [f"{f}_z" for f in FEATURES if f"{f}_z" in df]
    return df[cols].corr()


def main(dry=True, modes=BASELINES):
    with db.connect() as conn:
        df = load(conn)
        df = add_bet_features(df)

        print(f"player-games that played : {len(df):,}")
        print(f"  with a FanDuel line    : {df['close_line'].notna().sum():,}")
        print(f"  distinct players       : {df['player_id'].nunique():,}")
        print(f"  date range             : {df['game_date'].min()} -> {df['game_date'].max()}")

        frames = []
        for mode in modes:
            z = combine(zscore(df, mode))
            scored = z["combined_score"].notna().sum()
            print(f"\n[{mode}]")
            print(f"  rows with a combined_score : {scored:,} / {len(z):,}")
            print(f"  rows with margin_vs_line_z : {z['margin_vs_line_z'].notna().sum():,}")
            if scored:
                print(f"  combined_score  mean {z['combined_score'].mean():+.3f}"
                      f"  sd {z['combined_score'].std():.3f}"
                      f"  min {z['combined_score'].min():+.2f}")
            frames.append(z)

        if dry:
            db.dry_notice()
            return frames

        # PRIMARY OUTPUT: one row per player-game, mixed baselines, raw + z.
        feats = build_features(df)

        # ONLY propped player-games are written -- a game with no line has no
        # betting signal to analyse.
        #
        # The filter is HERE, after build_features, and that placement is the whole
        # point. Filtering in the SQL instead would shrink the BASELINE population
        # to propped games only, and those are systematically a player's
        # higher-profile nights. His "normal" would be computed from his best games,
        # inflating the mean and shrinking every z-score toward zero.
        #
        # So: baselines see every game he played; the output keeps only the ones
        # with money on them.
        before = len(feats)
        feats = feats[feats["close_line"].notna()]
        print(f"\nplayer-games with a prop : {len(feats):,} of {before:,} "
              f"({before - len(feats):,} unpropped dropped from OUTPUT, "
              f"kept in the baselines)")

        fcols = (["player_id", "game_id", "game_date", "position"]
                 + RAW_COLS + Z_COLS + ["n_player_games", "n_league"])
        frows = feats[fcols].replace({np.nan: None}).to_dict("records")
        n = db.upsert(conn, "player_game_features", frows,
                      conflict=["player_id", "game_id"])
        print(f"\nupserted {n:,} rows -> player_game_features "
              f"({db.count(conn, 'player_game_features'):,} total)")

        rows = []
        # Built from the constants rather than hand-listed, so adding a feature to
        # FEATURES or BET_FEATURES cannot silently fail to reach the table -- which
        # is exactly what happened when the betting signals were first added.
        cols = (["player_id", "game_id", "baseline_mode", "game_date",
                 "close_line", "close_under", "open_line", "open_under",
                 "line_move_pct", "under_move_pct", "margin_vs_line",
                 "combined_score", "n_features", "n_baseline"]
                + [f"{f}_z" for f in FEATURES + BET_FEATURES])
        for z in frames:
            keep = z[cols].replace({np.nan: None})
            rows.extend(keep.to_dict("records"))

        n = db.upsert(conn, "player_game_z", rows,
                      conflict=["player_id", "game_id", "baseline_mode"])
        print(f"\nupserted {n:,} rows -> player_game_z "
              f"({db.count(conn, 'player_game_z'):,} total)")


if __name__ == "__main__":
    args = sys.argv[1:]
    m = next((args[i + 1] for i, a in enumerate(args)
              if a == "--mode" and i + 1 < len(args)), None)
    main(dry=db.is_dry(), modes=(m,) if m else BASELINES)
