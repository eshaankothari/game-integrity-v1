"""Exploratory queries over the pipeline's tables. Read-only, no API calls.

Each query is a function decorated with @query that takes a live connection and
returns a DataFrame. Registering them in one dict means the runner, the CSV naming
and any future plotting all work without per-query wiring.

    python3 analysis/queries.py                     # list what is available
    python3 analysis/queries.py under_price          # run one, print + save CSV
    python3 analysis/queries.py under_price --n 40   # show more rows
    python3 analysis/queries.py under_price --plot   # also write a PNG

Output lands in analysis/out/<name>.csv so it never collides with the pipeline's
own out/ directory.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config                                    # noqa: E402
import db                                        # noqa: E402

warnings.filterwarnings("ignore")

OUT = pathlib.Path(__file__).resolve().parent / "out"

QUERIES = {}


def query(fn):
    """Register a function as a runnable query."""
    QUERIES[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------------------

@query
def under_price(conn, min_quotes=20, book=None):
    """Mean CLOSING under price per player, most-favoured first.

    These are DECIMAL odds, so LOW means the under was heavily backed -- 1.60
    implies about 63 percent, 2.20 about 45 percent. Sorting ascending therefore
    puts the players the market most expected to fall short at the top.

    ONE BOOK, not an average across three. Books disagree on price and each carries
    its own margin, so pooling them would blend different opinions into a number
    that is nobody's. config.BOOK is the primary book everywhere else in the project.

    min_quotes filters out players with a handful of lines, whose mean is dominated
    by which specific games happened to get a line rather than by how the market
    saw them.
    """
    book = book or config.BOOK
    sql = """
        SELECT p.full_name AS player,
               count(*)                      AS n_quotes,
               count(DISTINCT q.event_id)    AS n_games,
               round(avg(q.under_price), 4)  AS mean_under,
               round(stddev(q.under_price), 4) AS sd_under,
               round(min(q.under_price), 2)  AS min_under,
               round(max(q.under_price), 2)  AS max_under,
               round(avg(q.over_price), 4)   AS mean_over,
               round(avg(q.line), 2)         AS mean_line,
               s.salary
        FROM prop_quotes q
        JOIN players p ON p.player_id = q.player_id
        LEFT JOIN player_salaries s
               ON s.player_id = q.player_id AND s.season = %(season)s
        WHERE q.snapshot_role = 'close'
          AND q.book = %(book)s
          AND q.under_price IS NOT NULL
        GROUP BY p.full_name, s.salary
        ORDER BY mean_under ASC
    """
    df = pd.read_sql(sql, conn, params={"book": book, "season": config.SEASON,
                                        "min_quotes": min_quotes})
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------

@query
def under_lean(conn, book=None):
    """De-vigged closing under probability per player-game. The real price signal.

    RAW UNDER PRICE IS ALMOST ENTIRELY MARGIN. FanDuel's overround on points props is
    a flat 1.049 across every line size, and once removed the market prices unders at
    0.4990-0.5023 whatever the line -- so the few hundredths separating players in
    `under_price` are the book's cut, not its opinion.

        p_under = (1/under) / (1/under + 1/over)      de-vigged, sums to 1
        lean    = p_under - 0.5                       + means the market leans UNDER

    VALIDATED, and more strongly than anything else measured here:

        p_under >= .520   n=2,750   53.5 pct under
        .508-.520         n=3,019   53.2
        .492-.508         n=4,843   51.8
        .480-.492         n=2,615   49.1
        p_under <= .480   n=2,271   48.1        monotonic, z = 3.81

    Compare under_move_pct at z = 2.11, and note this needs only the two closing
    prices, so coverage is 100 percent rather than the 52 percent that anything
    involving an opening line is limited to.

    LOW LINES AMPLIFY IT. Splitting on line size, the spread is 5.3 points below a
    10-point line against 4.0 above it. The half-point grid is why: on a 5.5 line a
    book cannot shade to 5.75, so its view has to go into the price, while on a 25.5
    line it can move the line itself. So a short under on a small line carries more
    information than the same price on a big one -- `lean_x_lowline` scores that.
    """
    book = book or config.BOOK
    sql = """
        SELECT p.full_name AS player, g.game_date, g.matchup,
               q.line, q.under_price, q.over_price,
               (1.0/q.under_price)/((1.0/q.under_price)+(1.0/q.over_price))
                   AS p_under,
               pg.minutes, pg.points,
               (pg.points < q.line) AS under_hit,
               s.salary, s.has_listed_salary,
               q.player_id, e.game_id
        FROM prop_quotes q
        JOIN odds_events e ON e.event_id = q.event_id
        JOIN games g       ON g.game_id  = e.game_id
        JOIN players p     ON p.player_id = q.player_id
        JOIN player_games pg
             ON pg.player_id = q.player_id AND pg.game_id = e.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = q.player_id AND s.season = %(season)s
        WHERE q.snapshot_role = 'close' AND q.book = %(book)s
          AND q.under_price IS NOT NULL AND q.over_price IS NOT NULL
          AND pg.minutes > 0
    """
    df = pd.read_sql(sql, conn, params={"book": book, "season": config.SEASON})
    df["lean"] = (df["p_under"] - 0.5).round(4)
    df["p_under"] = df["p_under"].round(4)
    df["vig"] = (1 / df["under_price"] + 1 / df["over_price"]).round(4)

    # Both-low composite. lean is already comparable across lines; line size is not,
    # so it enters as a percentile. Multiplying keeps a row high only when BOTH hold,
    # which is the "extra suspicious" reading -- an average would let a big lean on a
    # 28.5 line score the same as a modest one on a 5.5.
    lean_pct = df["lean"].rank(pct=True)
    small_pct = 1 - df["line"].rank(pct=True)
    df["lean_x_lowline"] = (lean_pct * small_pct).round(4)

    return df.sort_values("lean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------

# Games believed compromised, marked on the per-player plots.
FLAGGED = {
    "Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
    "Jontay Porter": ["2024-01-20", "2024-01-26", "2024-03-20"],
}


@query
def usage_effort(conn, min_games=15):
    """Two composites -- offensive USAGE and physical EFFORT -- both minutes-adjusted.

    THE SPLIT IS EMPIRICAL, not editorial. Correlating each candidate with Game Score
    after stripping minutes from both gives how much NEW information it carries:

        fga             0.468      touches       0.370      usage_pct  0.341
        contested_fga   0.373      passes        0.224
        --- the above are largely production restated ---
        deflections     0.095      screen_assists 0.091     box_outs   0.086
        loose_balls     0.082      contested_sh   0.068     charges    0.002

    So the hustle family is close to orthogonal to production and the offensive family
    is not. Blending them would let a redundant axis outvote an independent one.

    ADJUSTED FOR MINUTES BY REGRESSION, not division. Every one of these scales with
    playing time -- distance correlates with minutes at 0.989, so it is a minutes
    proxy rather than an effort measure. Dividing (per-36) explodes for short stints:
    points-per-36 has sd 30.4 below five minutes against 8.4 above thirty. Subtracting
    a fitted line has no denominator and stays stable, so `*_adj` is the part of the
    stat that minutes do not explain.

    SPEED, NOT DISTANCE, carries the physical signal. Distance is minutes; speed is
    independent of it (r = -0.081) and is what changes when a player stops running.
    """
    from sklearn.linear_model import LinearRegression

    sql = """
        SELECT p.full_name AS player, g.game_date, pg.minutes, pg.points,
               pg.usage_pct, pg.touches, pg.passes, pg.fga,
               pg.speed, pg.distance,
               pg.contested_shots, pg.deflections, pg.loose_balls,
               pg.box_outs, pg.screen_assists, pg.charges_drawn,
               r.tier, p.position, f.close_line, s.salary, s.has_listed_salary,
               pg.player_id, pg.game_id
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        JOIN games   g ON g.game_id  = pg.game_id
        LEFT JOIN player_game_residuals r
             ON r.player_id = pg.player_id AND r.game_id = pg.game_id
        LEFT JOIN player_game_features f
             ON f.player_id = pg.player_id AND f.game_id = pg.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = pg.player_id AND s.season = %(season)s
        WHERE pg.minutes > 0 AND pg.points IS NOT NULL
    """
    d = pd.read_sql(sql, conn, params={"season": config.SEASON})

    USAGE = ["usage_pct", "touches", "passes", "fga"]
    EFFORT = ["contested_shots", "deflections", "loose_balls",
              "box_outs", "screen_assists", "charges_drawn", "speed"]
    for c in USAGE + EFFORT + ["minutes", "distance"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Strip the minutes component from each stat, then standardise what is left so
    # the members of a composite contribute equally rather than by unit size.
    for c in USAGE + EFFORT:
        ok = d[[c, "minutes"]].dropna()
        m = LinearRegression().fit(ok[["minutes"]], ok[c])
        adj = d[c] - m.predict(d[["minutes"]].fillna(d["minutes"].median()))
        d[f"{c}_adj"] = adj.where(d[c].notna())
        d[f"{c}_adj_z"] = ((d[f"{c}_adj"] - d[f"{c}_adj"].mean())
                           / d[f"{c}_adj"].std()).round(3)

    d["usage_score"] = d[[f"{c}_adj_z" for c in USAGE]].mean(axis=1, skipna=True).round(3)
    d["effort_score"] = d[[f"{c}_adj_z" for c in EFFORT]].mean(axis=1, skipna=True).round(3)
    d["n_usage"] = d[[f"{c}_adj_z" for c in USAGE]].notna().sum(axis=1)
    d["n_effort"] = d[[f"{c}_adj_z" for c in EFFORT]].notna().sum(axis=1)

    # THREE BASELINES, because they answer different questions and disagree:
    #
    #   _z_own    against his own season      "unlike HIM"
    #   _z_tier   against his role tier       "unlike a player in his role"
    #   _z_pos    against his listed position "unlike a player at his position"
    #
    # Own-season is the sharpest claim but is unavailable for anyone with few games
    # and, for a player whose normal is near zero, cannot flag anything -- his bad
    # night IS his normal night. Peer baselines have no such floor: a bench player
    # can be two sd below the bench distribution however erratic his own season was.
    #
    # The cost is the reverse. A peer z says nothing about whether THIS player
    # departed from his own pattern, so a low-usage specialist scores badly every
    # night simply for being who he is.
    n = d.groupby("player_id")["points"].transform("size")
    d["n_games"] = n
    for c in ("usage_score", "effort_score"):
        g = d.groupby("player_id")[c]
        d[f"{c}_z_own"] = ((d[c] - g.transform("mean"))
                           / g.transform("std").replace(0, np.nan)).round(3)
        d.loc[n < min_games, f"{c}_z_own"] = np.nan
        for key, suf in (("tier", "z_tier"), ("position", "z_pos")):
            if key not in d.columns:
                continue
            gk = d.groupby(key)[c]
            d[f"{c}_{suf}"] = ((d[c] - gk.transform("mean"))
                               / gk.transform("std").replace(0, np.nan)).round(3)
            d[f"{c}_pct_{key}"] = gk.rank(pct=True).round(4)
    return d.sort_values("effort_score").reset_index(drop=True)


@query
def player_cdf(conn, player="Malik Beasley", min_games=15):
    """One player's Game Scores as an empirical CDF, beside the z-score version.

    THE CALCULATION IS THREE LINES, and the whole point is that none of them assume
    anything about the distribution:

        s        = his game scores, one per played game
        cdf      = s.rank(pct=True)              share of his games at or below
        z        = (s - s.mean()) / s.std()      how many of HIS sds below his mean

    `cdf` is distribution-free. `z` is only a probability if his scores are normal,
    and Game Scores are right-skewed for 91 percent of players, so z systematically
    understates a bad game -- the left tail is short in sd units because a few huge
    games inflate the denominator.

    WHAT THE CDF COSTS. It saturates on ties and cannot go below 1/n, so a player with
    26 games can never score below 0.038 however badly he played, and every game at
    his modal value gets an identical percentile. That is fatal for a low-usage player
    -- Porter scored zero in 10 of 26 games, so all ten tie at 0.211 -- and harmless
    for someone with a wide spread like Beasley.

    Run with --player "Name" to switch subject.
    """
    d = output_stat(conn, min_games=min_games)
    s = d[d["player"] == player].copy()
    if s.empty:
        raise SystemExit(f"no games for {player!r}")
    s["gd"] = s["game_date"].astype(str).str[:10]

    gs = s["game_score"]
    s["cdf"] = gs.rank(pct=True).round(4)                       # empirical CDF
    s["z"] = ((gs - gs.mean()) / gs.std()).round(3)             # z-score
    s["cdf_if_normal"] = _norm_cdf(s["z"]).round(4)             # what z implies
    s["cdf_error"] = (s["cdf"] - s["cdf_if_normal"]).round(4)
    s["flagged"] = s["gd"].isin(FLAGGED.get(player, []))
    return s.sort_values("game_score").reset_index(drop=True)


def _norm_cdf(z):
    """Standard normal CDF, so the z column can be read as a probability."""
    from scipy import stats
    return pd.Series(stats.norm.cdf(z.values), index=z.index)


@query
def gs_z_propped(conn, min_games=15):
    """Propped player-games ranked by Game Score against the player's OWN season.

    PROPPED ONLY -- a game with no closing line has no market expectation to have
    fallen short of, and is not a candidate whatever the performance looked like.

    The z is computed on ALL of a player's played games, not just his propped ones,
    so the baseline is his real season rather than the subset that happened to get a
    line. Restricting first would shift every mean toward the games books thought
    worth pricing.

    READ z_own AND pct_own TOGETHER. Game Scores are right-skewed -- 91 percent of
    players have positive skew, mean +0.63 -- so the left tail is short in z units and
    z understates a bad game. pct_own makes no distributional assumption but saturates
    on ties and cannot go below 1/n.
    """
    d = output_stat(conn, min_games=min_games)
    d = d[d["close_line"].notna() & d["game_score_z_own"].notna()].copy()
    d["shortfall"] = (1 - d["points"] / d["close_line"]).clip(0, 1).round(3)
    d["under_hit"] = d["points"] < d["close_line"]

    # ABSOLUTE shortfall against his own typical game, in Game Score points.
    #
    # Chosen over both alternatives after measuring them. Dividing by sd (the z-score)
    # under-corrects: log(sd) regressed on log(mean output) has slope 0.376, so sd
    # grows roughly as the cube root of output and the coefficient of variation falls
    # from 0.80 to 0.33 across the range. What survives the division is still
    # proportional to output, which is why a star's zero-output game scores -3.0 and a
    # rotation player's scores -1.3 for the same event.
    #
    # Dividing by the median over-corrects and is unstable: Game Score can be
    # negative, 14 players have a median at or below 2, and the observed ratio ranges
    # to -190. Same division-by-a-small-number failure as per-36.
    #
    # Subtracting has no denominator, so neither problem arises. It IS scale-dependent
    # -- big producers can fall further -- and that is left to the salary axis to
    # handle rather than smuggled into the performance number.
    g = d.groupby("player_id")["game_score"]
    d["gs_median"] = g.transform("median").round(2)
    d["gs_vs_median"] = (d["game_score"] - d["gs_median"]).round(2)

    d = d.sort_values("gs_vs_median").reset_index(drop=True)
    d.insert(0, "rank", range(1, len(d) + 1))
    return d


@query
def game_score_by_player(conn, min_games=10):
    """Season Game Score per player, ranked. One row per player.

    All 11 Game Score inputs are present on all 26,393 played games -- they come from
    the traditional box score, which never suffered the tracking-feed outages that hit
    distance and touches.

    `worst_z` is the single worst game each player had against HIS OWN season, and
    `n_bad` counts how many fell below -1.5. Those two say something the mean cannot:
    a player can average well and still have collapses, and collapses are the thing
    being looked for.

    Sorted by mean ascending, so the lowest-output players come first. That is mostly
    a list of deep reserves rather than anything interesting -- the useful columns are
    `worst_z` and `n_bad`.
    """
    d = output_stat(conn, min_games=min_games)
    g = d.groupby(["player", "player_id"])
    out = g.agg(
        n_games=("game_score", "size"),
        tier=("tier", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        mean_gs=("game_score", "mean"),
        median_gs=("game_score", "median"),
        sd_gs=("game_score", "std"),
        min_gs=("game_score", "min"),
        max_gs=("game_score", "max"),
        mean_min=("minutes", "mean"),
        mean_pts=("points", "mean"),
        mean_usage=("usage_raw", "mean"),
        worst_z=("game_score_z_own", "min"),
        n_bad=("game_score_z_own", lambda s: int((s < -1.5).sum())),
        salary=("salary", "first"),
        has_listed_salary=("has_listed_salary", "first"),
    ).reset_index()
    out = out[out["n_games"] >= min_games].copy()
    for c in ("mean_gs", "median_gs", "sd_gs", "mean_min", "mean_pts", "mean_usage"):
        out[c] = out[c].round(2)
    out["bad_rate"] = (out["n_bad"] / out["n_games"]).round(3)
    out = out.sort_values("mean_gs").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


@query
def output_stat(conn, min_games=15):
    """Production, usage and efficiency per player-game, z-scored within PEER TIER.

    THREE NUMBERS, NOT ONE, because they fail differently:

        production   what he delivered      Hollinger Game Score
        usage        the chances he had     touches, usage_pct, fga, minutes
        efficiency   production per chance  game_score / usage_raw

    Normal usage with low production is a player who had the ball and did nothing.
    Low usage is a player who stayed out of the way. Blending them into one score
    cannot distinguish those, and they are different behaviours.

    GAME SCORE rather than a hand-rolled weighting. Hollinger's coefficients were
    fitted to approximate a player's contribution on a points scale, so a 10 is a
    solid game and a 40 is a great one. Inventing weights here would add a second set
    of arbitrary numbers on top of the ones the score already has:

        PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB
            + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

    PEER = TIER, not the whole league. Bench, rotation and starter tiers have
    genuinely different distributions -- a bench player's good game and a starter's
    are not the same size -- and tiers are already how residualize.py fits its
    context slopes, so the peer definition stays consistent across the project.

    BOTH z AND PERCENTILE are returned. Game Score is right-skewed like points, so z
    understates the low tail for the same reason it does there; pct_in_tier is the
    assumption-free version. Compare them before trusting either.
    """
    sql = """
        SELECT p.full_name AS player, g.game_date, pg.minutes,
               pg.points, pg.fgm, pg.fga, pg.fta, pg.ftm,
               pg.rebounds, pg.rebounds_off, pg.assists, pg.steals, pg.blocks,
               pg.turnovers, pg.fouls, pg.usage_pct, pg.touches, pg.distance,
               r.tier, f.close_line, s.salary, s.has_listed_salary,
               pg.player_id, pg.game_id
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        JOIN games   g ON g.game_id  = pg.game_id
        LEFT JOIN player_game_residuals r
             ON r.player_id = pg.player_id AND r.game_id = pg.game_id
        LEFT JOIN player_game_features f
             ON f.player_id = pg.player_id AND f.game_id = pg.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = pg.player_id AND s.season = %(season)s
        WHERE pg.minutes > 0 AND pg.points IS NOT NULL
    """
    df = pd.read_sql(sql, conn, params={"season": config.SEASON})
    num = ["points", "fgm", "fga", "fta", "ftm", "rebounds", "rebounds_off",
           "assists", "steals", "blocks", "turnovers", "fouls", "usage_pct",
           "touches", "minutes"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    drb = df["rebounds"] - df["rebounds_off"].fillna(0)

    df["game_score"] = (
        df["points"] + 0.4 * df["fgm"] - 0.7 * df["fga"]
        - 0.4 * (df["fta"] - df["ftm"]) + 0.7 * df["rebounds_off"].fillna(0)
        + 0.3 * drb + df["steals"] + 0.7 * df["assists"] + 0.7 * df["blocks"]
        - 0.4 * df["fouls"] - df["turnovers"]).round(2)

    # Usage as a single raw quantity: the three opportunity measures, each put on a
    # common scale by its league mean so none dominates by unit size alone.
    opp = ["touches", "fga", "minutes"]
    scaled = pd.DataFrame({c: df[c] / df[c].mean() for c in opp})
    df["usage_raw"] = scaled.mean(axis=1).round(3)
    df["efficiency"] = (df["game_score"] / df["usage_raw"].replace(0, np.nan)).round(2)

    # Drop thin players before standardising: a mean and sd from 5 games is noise.
    n = df.groupby("player_id")["points"].transform("size")
    df["n_games"] = n

    for col in ("game_score", "usage_raw", "efficiency"):
        g = df.groupby("tier")[col]
        df[f"{col}_z"] = ((df[col] - g.transform("mean"))
                          / g.transform("std")).round(3)
        df[f"{col}_pct"] = g.rank(pct=True).round(4)
        # And against the player himself, for the "unlike him" reading.
        gp = df.groupby("player_id")[col]
        df[f"{col}_z_own"] = ((df[col] - gp.transform("mean"))
                              / gp.transform("std").replace(0, np.nan)).round(3)
    df.loc[n < min_games, [c for c in df.columns if c.endswith("_z_own")]] = np.nan
    return df.sort_values("game_score_z").reset_index(drop=True)


@query
def points_dist(conn):
    """Every played player-game's points, with the player's own baseline attached.

    The shape matters more than the summary here. Points are COUNTS -- non-negative
    integers, right-skewed, with a spike at zero -- and every z-score in this project
    treats them as if they were symmetric. `z_own` is what that assumption produces;
    `pct_own` is the assumption-free version, and the two disagree most exactly where
    it matters.
    """
    sql = """
        SELECT p.full_name AS player, g.game_date, pg.points, pg.minutes,
               r.tier, s.salary, s.has_listed_salary,
               pg.player_id, pg.game_id
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        JOIN games   g ON g.game_id  = pg.game_id
        LEFT JOIN player_game_residuals r
             ON r.player_id = pg.player_id AND r.game_id = pg.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = pg.player_id AND s.season = %(season)s
        WHERE pg.minutes > 0 AND pg.points IS NOT NULL
    """
    df = pd.read_sql(sql, conn, params={"season": config.SEASON})
    g = df.groupby("player_id")["points"]
    df["player_mean"] = g.transform("mean").round(2)
    df["player_sd"] = g.transform("std").round(2)
    df["n_games"] = g.transform("size")
    df["z_own"] = ((df["points"] - df["player_mean"]) / df["player_sd"]).round(3)
    df["pct_own"] = g.rank(pct=True).round(4)
    # What z-score a ZERO would earn -- the floor effect, per player.
    df["z_if_zero"] = (-df["player_mean"] / df["player_sd"]).round(3)
    return df.sort_values(["player", "game_date"]).reset_index(drop=True)


@query
def price_only_move(conn, book=None):
    """Games where the PRICE moved while the LINE held flat -- both directions.

        price_only_move = under_move_pct, but only where |line_move_pct| <= 0.01

    A book absorbs incoming action two ways: move the line, or reprice it. Moving a
    line is the bigger decision, so a repriced-but-unmoved line is the quieter signal
    -- the book adjusting for money without conceding a new number.

    SIGN. under_move_pct = (close_under - open_under) / open_under, so POSITIVE means
    the under price LENGTHENED, i.e. it got less likely to be backed -- money on the
    OVER. Negative means the under shortened, which is the direction of interest here.
    The score gates out positive rows for exactly that reason.

    TESTED AND NULL. Across 4,304 rows with a flat line the buckets came out 54.0,
    52.4, 51.9, 53.8, 51.8 percent under -- non-monotonic, with "slight over" beating
    "slight under", which no mechanism explains. The continuous full-sample
    under_move_pct does carry a weak signal (z = 2.11) but restricting to flat lines
    throws most of it away.
    """
    sql = """
        SELECT p.full_name AS player, g.game_date, g.matchup,
               f.open_line, f.close_line, f.line_move_pct,
               f.open_under, f.close_under, f.under_move_pct, f.price_only_move,
               f.minutes, f.points, f.margin_vs_line,
               (f.points < f.close_line) AS under_hit,
               s.salary, s.has_listed_salary,
               f.player_id, f.game_id
        FROM player_game_features f
        JOIN players p ON p.player_id = f.player_id
        JOIN games   g ON g.game_id  = f.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = f.player_id AND s.season = %(season)s
        WHERE f.price_only_move IS NOT NULL AND f.minutes > 0
        ORDER BY f.price_only_move DESC
    """
    df = pd.read_sql(sql, conn, params={"season": config.SEASON})
    df["direction"] = np.where(df["price_only_move"] > 0, "under LENGTHENED (over money)",
                       np.where(df["price_only_move"] < 0, "under SHORTENED (under money)",
                                "no price move"))
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def no_up_move_low_joint(conn, book=None, max_joint=0.25):
    """No upward line move, and p_price x p_line below `max_joint`.

    `~(line_move_pct > 0)` NOT `line_move_pct <= 0`. The negated form is the whole
    point of the request: a NaN comparison is False in pandas, so `<= 0` would delete
    every row whose opening line was never posted -- 48 percent of the season, and
    disproportionately the low-salary fringe players, since books post stars first
    ($20.1M median with an open line against $10.6M without). Those rows have nothing
    to evaluate, which is not the same as failing the test.

    The product is the independence approximation of "both low", so it lets one
    extreme axis carry an ordinary one. `long_price` flags rows whose under price sits
    above the median -- the wrong direction entirely -- since the product cannot
    exclude them on its own.
    """
    df = joint_low(conn, book=book)

    with_move = pd.read_sql(
        "SELECT player_id, game_id, line_move_pct, under_move_pct, open_line "
        "FROM player_game_features", conn)
    df = df.merge(with_move, on=["player_id", "game_id"], how="left")

    n0 = len(df)
    keep_move = ~(df["line_move_pct"] > 0)          # NaN survives, by design
    df = df[keep_move]
    n1 = len(df)
    df = df[df["indep_pct"] < max_joint].copy()

    df["has_open_line"] = df["open_line"].notna()
    df["long_price"] = df["p_price"] > 0.5
    print(f"   {n0:,} -> no upward line move {n1:,} "
          f"-> product < {max_joint} {len(df):,}")
    print(f"   of those, {int(df.has_open_line.sum()):,} have an opening line, "
          f"{int((~df.has_open_line).sum()):,} never had one (kept)")
    print(f"   flagged long_price (p_price > .5): {int(df.long_price.sum()):,}")

    df = df.drop(columns=["rank"], errors="ignore")   # joint_low already ranked
    df = df.sort_values("indep_pct").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def line_pulled(conn, book=None):
    """Player-games that had an OPENING line and no CLOSING line -- withdrawals.

    A book pulling a prop is the strongest thing it can say. It is what happens when
    a trader decides he cannot price the player safely: one-way money, an injury
    whisper, or action he does not want. And because load_props probes at tip exactly,
    the pipeline records only that nothing was there -- never that something had been.

    This is the shape of Jontay Porter's 2024-01-26 game, which has ZERO rows in
    prop_quotes despite the market having quoted him all afternoon:

        T-4h  draftkings, fanduel, williamhill_us, betmgm, +4
        T-3h  fanduel, williamhill_us, betmgm, +4          <- draftkings gone
        T-1h  pointsbetus ONLY
        tip   pointsbetus ONLY

    MOST WITHDRAWALS ARE INJURIES, and the split is the whole story: roughly two
    thirds of pulled lines belong to players who then did not play at all. The book
    was right and the pull was routine. The interesting residue is the minority who
    PLAYED ANYWAY after every major had walked away.
    """
    book = book or config.BOOK
    sql = """
        WITH q AS (
            SELECT q.player_id, e.game_id,
                   max(q.line)  FILTER (WHERE q.snapshot_role = 'open')  AS open_line,
                   max(q.under_price) FILTER (WHERE q.snapshot_role = 'open') AS open_under,
                   bool_or(q.snapshot_role = 'close') AS has_close
            FROM prop_quotes q JOIN odds_events e ON e.event_id = q.event_id
            WHERE q.player_id IS NOT NULL AND q.book = %(book)s
            GROUP BY 1, 2)
        SELECT p.full_name AS player, g.game_date, g.matchup,
               q.open_line, q.open_under, q.has_close,
               pg.minutes, pg.points, pg.dnp_reason,
               (pg.minutes IS NULL OR pg.minutes = 0) AS did_not_play,
               (pg.minutes > 0 AND pg.points < q.open_line) AS under_hit,
               s.salary, s.has_listed_salary,
               q.player_id, q.game_id
        FROM q
        JOIN player_games pg ON pg.player_id = q.player_id AND pg.game_id = q.game_id
        JOIN players p ON p.player_id = q.player_id
        JOIN games   g ON g.game_id  = q.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = q.player_id AND s.season = %(season)s
        WHERE q.open_line IS NOT NULL AND NOT q.has_close
        ORDER BY q.open_line
    """
    df = pd.read_sql(sql, conn, params={"book": book, "season": config.SEASON})
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def line_moved_up(conn, book=None):
    """Player-games where the LINE rose between the T-12h open and the close.

        line_move_pct = (close_line - open_line) / open_line     positive = rose

    A rising line means the market revised EXPECTATIONS UPWARD -- money on the over,
    or news that the player would see more usage. It is the opposite direction from
    the under-side signal the rest of this project chases, which is why the score
    gates on `no upward PRICE-only move`.

    WORTH KNOWING BEFORE READING TOO MUCH INTO IT: line movement tested NULL and in
    fact slightly BACKWARDS as a predictor. Across 7,990 rows with movement known,
    lines that fell more than 10 percent went under 47.1 percent of the time against
    a 52.7 percent flat baseline, and those players averaged 1.108x the closing line.
    A sharply moving line is usually a book repricing correctly on public news --
    after which the corrected number is harder to beat -- rather than money arriving.

    Only rows with BOTH an open and a close can be measured, which is 52 percent of
    the season. The other 48 percent had no line posted 12 hours out; they are absent
    here rather than counted as unmoved.
    """
    book = book or config.BOOK
    sql = """
        SELECT p.full_name AS player, g.game_date, g.matchup,
               f.open_line, f.close_line,
               (f.close_line - f.open_line) AS line_move_abs,
               f.line_move_pct,
               f.open_under, f.close_under, f.under_move_pct,
               f.minutes, f.points, f.margin_vs_line,
               (f.points < f.close_line) AS under_hit,
               s.salary, s.has_listed_salary,
               f.player_id, f.game_id
        FROM player_game_features f
        JOIN players p ON p.player_id = f.player_id
        JOIN games   g ON g.game_id  = f.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = f.player_id AND s.season = %(season)s
        WHERE f.line_move_pct > 0
          AND f.minutes > 0
        ORDER BY f.line_move_pct DESC
    """
    df = pd.read_sql(sql, conn, params={"season": config.SEASON})
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def cuts_by_joint(conn, book=None):
    """export_candidates' first three cuts, then ranked by p_price x p_line.

    The cuts (production, effort, salary) decide WHO stays; the product decides the
    ORDER. They are answering different questions -- the cuts are about the player's
    night, the product is about what the market did -- so keeping them separate is
    the point rather than a compromise.

    PERCENTILES COME FROM ALL 15,498 GAMES, not from the survivors. A p_price of 0.02
    has to mean "shorter than 98 percent of the season" regardless of which cut
    configuration produced the shortlist, or the number changes meaning every time a
    threshold moves.

    A GUARD ON p_price. The product lets one extreme axis compensate for an ordinary
    one, and the failure case is severe: Pat Connaughton's 2.20 on a 3.5 line is the
    LONGEST under of the season -- the opposite of the signal -- yet ranks 123rd on
    the raw product because a 3.5 line is rare. `long_price` flags any row above the
    median price so those cannot reach the top on line rarity alone.
    """
    import export_candidates as EC

    jl = joint_low(conn, book=book)[
        ["player_id", "game_id", "line", "under_price", "p_price", "p_line",
         "indep_pct", "both_low", "p_under", "lean"]]

    df = pd.read_sql(EC.SQL.format(season=config.SEASON), conn)
    df["prod_z"], df["n_prod"] = EC.oriented_mean(df, EC.PROD)
    df["effort_z"], df["n_effort"] = EC.oriented_mean(df, EC.EFFORT)
    n0 = len(df)

    if EC.CUT_MODE == "quantile":
        c1 = df[EC.worst_fraction(df, "prod_z", EC.PROD_KEEP)]
        c2 = c1[c1["effort_z"] <= df["effort_z"].quantile(EC.EFFORT_KEEP)]
    else:
        c1 = df[df["prod_z"] < EC.PROD_THRESHOLD]
        c2 = c1[c1["effort_z"] < EC.EFFORT_THRESHOLD]
    c3 = c2[c2["salary"].isna() | (c2["salary"] <= EC.MAX_SALARY)].copy()
    print(f"   cuts [{EC.CUT_MODE}]: {n0:,} -> prod {len(c1):,} -> effort {len(c2):,}"
          f" -> salary {len(c3):,}")

    out = c3.merge(jl, on=["player_id", "game_id"], how="inner",
                   suffixes=("", "_q"))
    out["long_price"] = out["p_price"] > 0.5
    out = out.sort_values("indep_pct").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


@query
def joint_low(conn, book=None):
    """Rank by the JOINT probability of a price this short AND a line this small.

        p_price = P(price <= this)              marginal, the empirical CDF
        p_line  = P(line  <= this)              marginal
        joint   = P(price <= this AND line <= this)     the actual 2-D CDF
        indep   = p_price * p_line              what joint WOULD be if independent
        lift    = joint / indep                 how much they travel together

    `joint` is a real probability and reads directly: 0.0008 means eight games in ten
    thousand were at least this extreme on both. Sorting ascending puts the rarest
    corner of the price-line plane first, which is the "both low" question asked
    properly rather than as an average of two ranks.

    WHY NOT p_price * p_line. That product assumes independence and they are not
    independent -- the half-point grid forces books to express a view through PRICE on
    small lines and through the LINE itself on big ones, so short prices and small
    lines co-occur mechanically. `lift` measures the error: above 1 means the pair
    happens more often than independence predicts.

    Computed exactly rather than sampled: price and line take 38 and 37 distinct
    values, so a 38x37 count grid with a cumulative sum along both axes gives the
    joint CDF for every cell with no approximation.
    """
    df = low_price_low_line(conn, book=book).drop(columns=["rank", "combined"],
                                                  errors="ignore")
    n = len(df)

    grid = (df.groupby(["under_price", "line"]).size()
              .unstack(fill_value=0).sort_index().sort_index(axis=1))
    cum = grid.cumsum(axis=0).cumsum(axis=1) / n     # P(price <= p AND line <= l)

    idx = pd.MultiIndex.from_arrays([df["under_price"], df["line"]])
    df["joint_pct"] = cum.stack().reindex(idx).values.round(6)
    df["p_price"] = df["under_price"].rank(pct=True, method="max").round(4)
    df["p_line"] = df["line"].rank(pct=True, method="max").round(4)
    df["indep_pct"] = (df["p_price"] * df["p_line"]).round(6)
    df["lift"] = (df["joint_pct"] / df["indep_pct"].replace(0, np.nan)).round(3)

    # STRICT AND: the worse of the two, so nothing compensates. Small only when both
    # marginals are small. The product lets a 1-in-1000 price carry an ordinary line,
    # and the joint CDF is worse still -- it is dominated by whichever margin is
    # rarer, which put the season's LONGEST under price at rank 20.
    df["both_low"] = df[["p_price", "p_line"]].max(axis=1).round(4)

    df = df.sort_values(["both_low", "indep_pct"]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def by_line(conn, book=None):
    """Write one CSV per line value into analysis/out/by_line/, and return an index.

    Each file holds every player-game quoted at that line, sorted shortest under
    price first, so the top of each file is the most under-leaning game the market
    ever priced at that number. Within a file the line is constant, so `under_price`
    and `price_pct_in_line` order identically -- the half-point-grid confound that
    makes league-wide price ranking partly a ranking of line size cannot arise.

    Filenames are zero-padded (line_04.5.csv) so they sort numerically in a file
    listing rather than lexically, which would put 10.5 before 4.5.
    """
    df = price_within_line(conn, book=book)
    dest = OUT / "by_line"
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.glob("line_*.csv"):        # stale files if the line set changes
        f.unlink()

    rows = []
    for line, g in df.groupby("line"):
        g = g.sort_values("under_price").reset_index(drop=True)
        g.insert(0, "rank_in_line", range(1, len(g) + 1))
        path = dest / f"line_{line:05.1f}.csv"
        g.to_csv(path, index=False)
        rows.append({
            "line": line, "n_games": len(g),
            "min_price": g["under_price"].min(),
            "median_price": g["under_price"].median(),
            "max_price": g["under_price"].max(),
            "under_hit_pct": round(100 * g["under_hit"].mean(), 1),
            # Does a short price predict the under WITHIN this line? Negative
            # correlation is the expected direction: shorter price, more unders.
            "corr_price_vs_hit": round(g["under_price"].corr(
                g["under_hit"].astype(float)), 3),
            "file": path.name,
        })
    print(f"   wrote {len(rows)} files -> {dest}")
    return pd.DataFrame(rows)


@query
def price_within_line(conn, book=None, min_group=40):
    """Price percentile ranked WITHIN each line value, not against the league.

    WHY GROUP BY LINE. Line size mechanically drives price spread. Books quote lines
    on a half-point grid, so on a 5.5 line they cannot shade to 5.75 and must express
    their view in the PRICE, while on a 25.5 line they move the line itself and the
    price stays near 1.91. League-wide price ranking therefore partly ranks "this was
    a small line" -- and it does so in BOTH directions, since small lines produce the
    extreme prices at each end.

        price <= 1.75   avg line  6.9
        price 1.85-1.91 avg line 15.6
        price > 2.00    avg line  9.2

    Ranking within the line asks the cleaner question: short FOR A LINE OF THIS SIZE.
    It is the non-parametric version of residualising price on line.

    Lines are discrete with 37 distinct values and large groups -- the 7 rarest cover
    50 rows between them -- so exact grouping works without binning. Groups below
    min_group fall back to the league-wide rank rather than producing a percentile out
    of four observations.
    """
    df = low_price_low_line(conn, book=book).drop(columns=["rank", "combined"],
                                                  errors="ignore")
    grp = df.groupby("line")["under_price"]
    within = grp.rank(pct=True)
    sizes = df.groupby("line")["under_price"].transform("size")
    # Thin lines cannot support a percentile; use the league figure for those.
    df["price_pct_in_line"] = within.where(sizes >= min_group,
                                           df["under_price"].rank(pct=True)).round(4)
    df["n_in_line"] = sizes
    df["price_rank_shift"] = (df["price_pct"] - df["price_pct_in_line"]).round(4)

    df = df.sort_values("price_pct_in_line").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def price_only(conn, book=None):
    """Player-games ranked by closing under price alone, shortest first.

    `price_pct` is the percentile position in the season's distribution: 0.0001 means
    only 0.01 percent of the 15,498 closing prices were this short. It is there so
    price can be combined with quantities on other scales -- a price lives in
    1.61-2.20, a line in 2.5-35, and neither is comparable to the other raw.

    The distribution is tight. Median 1.91, and the middle half sits between 1.85 and
    1.96 -- an 11-cent band holding 7,700 games. Only 230 rows are at or below 1.75.
    That clustering is the vig; genuinely short unders are rare, which is what makes
    the tail worth reading.

    Price alone is monotonic against the outcome:

        <= 1.75    n=  230   56.5 pct under   avg line  6.9
        1.75-1.80  n=  934   53.7            avg line  9.8
        1.80-1.85  n=3,594   52.6            avg line 15.0
        1.85-1.91  n=4,419   52.3            avg line 15.6
        1.91-2.00  n=5,009   49.8            avg line 15.3
        > 2.00     n=1,312   47.9            avg line  9.2

    READ THE avg_line COLUMN. Both price extremes are LOW-line games while the middle
    of the price distribution holds the big lines. The half-point grid is why: on a
    5.5 line a book cannot shade to 5.75 so it moves the price instead, while on a
    25.5 line it moves the line and the price stays near 1.91. Ranking on price is
    therefore partly ranking on "this was a small line", in BOTH directions.
    """
    df = low_price_low_line(conn, book=book)
    df = df.drop(columns=["rank", "combined"], errors="ignore")
    df = df.sort_values("under_price").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@query
def low_price_low_line(conn, book=None):
    """Player-games ranked by BOTH the closing under price and the line being low.

    The two are on different scales -- a price lives in 1.6-2.2 and a line in 2.5-35 --
    so they are combined as percentile ranks rather than raw values. `combined` is the
    mean of the two, both ascending, so 0 would be the lowest price AND the lowest
    line in the dataset.

    MEAN, not product, unlike `under_lean`'s lean_x_lowline. A mean lets a very
    extreme price partly stand in for a merely lowish line, which is what "rank by
    both" ordinarily means. Swap to a product if you want strict conjunction.

    Raw price is kept here because that is what was asked for, but `p_under` and
    `lean` sit alongside it: FanDuel's overround is a flat 1.049, so most of the
    variation in the raw column is margin rather than opinion. Where the two orderings
    disagree, the de-vigged one is the one carrying information.
    """
    book = book or config.BOOK
    sql = """
        SELECT p.full_name AS player, g.game_date, g.matchup,
               q.line, q.under_price, q.over_price,
               (1.0/q.under_price)/((1.0/q.under_price)+(1.0/q.over_price)) AS p_under,
               pg.minutes, pg.points,
               (pg.points < q.line) AS under_hit,
               s.salary, s.has_listed_salary,
               q.player_id, e.game_id
        FROM prop_quotes q
        JOIN odds_events e ON e.event_id = q.event_id
        JOIN games g       ON g.game_id  = e.game_id
        JOIN players p     ON p.player_id = q.player_id
        JOIN player_games pg
             ON pg.player_id = q.player_id AND pg.game_id = e.game_id
        LEFT JOIN player_salaries s
             ON s.player_id = q.player_id AND s.season = %(season)s
        WHERE q.snapshot_role = 'close' AND q.book = %(book)s
          AND q.under_price IS NOT NULL AND q.over_price IS NOT NULL
          AND pg.minutes > 0
    """
    df = pd.read_sql(sql, conn, params={"book": book, "season": config.SEASON})
    df["lean"] = (df["p_under"] - 0.5).round(4)
    df["p_under"] = df["p_under"].round(4)

    df["price_pct"] = df["under_price"].rank(pct=True).round(4)
    df["line_pct"] = df["line"].rank(pct=True).round(4)
    df["combined"] = ((df["price_pct"] + df["line_pct"]) / 2).round(4)

    df = df.sort_values("combined").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------

def plot(name, df):
    """Best-effort chart for a query. Silently skipped if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib not installed -- skipping plot)")
        return
    if name == "usage_effort":
        import numpy as np
        o = pd.read_csv(OUT / "output_stat.csv", dtype={"game_id": str})
        m = df.merge(o[["player_id", "game_id", "game_score", "game_score_z_own"]],
                     on=["player_id", "game_id"], how="inner")
        m["gd"] = m["game_date"].astype(str).str[:10]
        # The cameo artefact: a league-wide linear minutes fit does not pass through
        # the origin, so a 1-minute appearance gets a large constant residual that has
        # nothing to do with effort. Floor the population before plotting.
        m = m[(m.minutes >= 5) & m.game_score_z_own.notna()
              & m.usage_score_z_own.notna()]
        f = m[m.apply(lambda r: r.gd in FLAGGED.get(r.player, []), axis=1)]

        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        for a, ycol, ylab in ((ax[0], "usage_score_z_own", "usage z vs own season"),
                              (ax[1], "effort_score_z_own", "effort z vs own season")):
            a.scatter(m["game_score_z_own"], m[ycol], s=4, alpha=.10,
                      color="#4C72B0", rasterized=True)
            a.axhline(0, color="grey", lw=.8); a.axvline(0, color="grey", lw=.8)
            r = m["game_score_z_own"].corr(m[ycol])
            b = np.polyfit(m["game_score_z_own"], m[ycol], 1)
            xs = np.linspace(m["game_score_z_own"].min(), m["game_score_z_own"].max(), 50)
            a.plot(xs, np.polyval(b, xs), color="black", lw=1.2, ls="--",
                   label=f"r = {r:+.3f}")
            a.scatter(f["game_score_z_own"], f[ycol], s=90, color="crimson",
                      edgecolor="white", zorder=5)
            for _, row in f.iterrows():
                a.annotate(f"{row.player.split()[1][:3]} {row.gd[5:]}",
                           (row.game_score_z_own, row[ycol]),
                           textcoords="offset points", xytext=(7, 4),
                           color="crimson", fontsize=8)
            # The quadrant that matters: produced little, was barely involved.
            a.axhspan(a.get_ylim()[0], 0, xmin=0, xmax=0.5, color="crimson", alpha=.04)
            a.set_xlabel("game score z vs own season")
            a.set_ylabel(ylab)
            a.set_title(f"{ylab.split(' z')[0].upper()} vs PRODUCTION   (n={len(m):,})")
            a.legend(fontsize=9); a.grid(alpha=.2)
        ax[0].text(0.02, 0.02, "low production\nlow usage", transform=ax[0].transAxes,
                   fontsize=8, color="crimson", va="bottom")
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    if name == "player_cdf":
        import numpy as np
        who = df["player"].iat[0]
        gs = df["game_score"].values
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        a = ax[0, 0]
        a.hist(gs, bins=25, color="#4C72B0", edgecolor="white")
        a.axvline(np.median(gs), color="black", lw=1.3, label=f"median {np.median(gs):.1f}")
        for _, r in df[df.flagged].iterrows():
            a.axvline(r.game_score, color="crimson", ls="--", lw=1.3)
            a.text(r.game_score, a.get_ylim()[1] * .97, f" {r.gd[5:]}", color="crimson",
                   fontsize=7.5, rotation=90, va="top")
        a.set_xlabel("game score"); a.set_ylabel("games")
        a.set_title(f"{who}  n={len(df)}  skew {pd.Series(gs).skew():+.2f}")
        a.legend(fontsize=8)

        # The CDF itself: a step function through his sorted scores.
        a = ax[0, 1]
        a.step(np.sort(gs), np.arange(1, len(gs) + 1) / len(gs), where="post",
               color="#4C72B0", lw=1.8, label="empirical CDF")
        xs = np.linspace(gs.min(), gs.max(), 300)
        from scipy import stats
        a.plot(xs, stats.norm.cdf(xs, gs.mean(), gs.std()), color="crimson", ls="--",
               lw=1.3, label="normal, i.e. what z assumes")
        for _, r in df[df.flagged].iterrows():
            a.plot([r.game_score], [r.cdf], "o", color="crimson", ms=7, zorder=5)
            a.annotate(f"{r.gd[5:]}  cdf={r.cdf:.2f}", (r.game_score, r.cdf),
                       textcoords="offset points", xytext=(8, -10),
                       color="crimson", fontsize=7.5)
        a.set_xlabel("game score"); a.set_ylabel("cumulative share of his games")
        a.set_title("Empirical CDF vs the normal z assumes"); a.legend(fontsize=8)
        a.grid(alpha=.25)

        # Where the two disagree, across his whole season.
        a = ax[1, 0]
        a.scatter(df["cdf_if_normal"], df["cdf"], s=22, alpha=.6, color="#4C72B0")
        a.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        f = df[df.flagged]
        a.scatter(f["cdf_if_normal"], f["cdf"], s=70, color="crimson", zorder=5)
        for _, r in f.iterrows():
            a.annotate(r.gd[5:], (r.cdf_if_normal, r.cdf), textcoords="offset points",
                       xytext=(6, 4), color="crimson", fontsize=7.5)
        a.set_xlabel("CDF implied by the z-score"); a.set_ylabel("true empirical CDF")
        a.set_title("Above the line = z UNDERSTATES how bad the game was")
        a.grid(alpha=.25)

        a = ax[1, 1]
        a.scatter(df["z"], df["cdf"], s=22, alpha=.6, color="#4C72B0")
        a.scatter(f["z"], f["cdf"], s=70, color="crimson", zorder=5)
        for _, r in f.iterrows():
            a.annotate(f"{r.gd[5:]}\nz={r.z:+.2f} cdf={r.cdf:.2f}", (r.z, r.cdf),
                       textcoords="offset points", xytext=(6, -14),
                       color="crimson", fontsize=7)
        a.set_xlabel("z vs his own season"); a.set_ylabel("empirical CDF")
        a.set_title("The two rankings are monotonic -- same order, different spacing")
        a.grid(alpha=.25)

        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    if name == "gs_z_propped":
        import numpy as np
        FLAG = [(1627736, "2024-01-06", "Bea 1/06"), (1627736, "2024-01-26", "Bea 1/26"),
                (1627736, "2024-02-27", "Bea 2/27"), (1627736, "2024-03-10", "Bea 3/10"),
                (1629007, "2024-01-20", "Por 1/20"), (1629007, "2024-03-20", "Por 3/20")]
        df = df.copy(); df["gd"] = df["game_date"].astype(str).str[:10]
        v = df["gs_vs_median"]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        a = ax[0, 0]
        a.hist(v, bins=80, color="#4C72B0", edgecolor="white")
        a.axvline(0, color="black", lw=1)
        a.axvline(v.quantile(.05), color="crimson", ls="--", lw=1.2,
                  label=f"5th pct = {v.quantile(.05):.1f}")
        a.set_xlabel("game_score - his median"); a.set_ylabel("player-games")
        a.set_title(f"Absolute shortfall vs own median  (n={len(v):,}, "
                    f"skew {v.skew():+.2f})")
        a.legend(fontsize=8)

        # By tier -- this is where the scale-dependence is visible.
        a = ax[0, 1]
        for t, c in (("bench", "#55A868"), ("rotation", "#4C72B0"),
                     ("starter", "#C44E52")):
            s = df[df.tier == t]["gs_vs_median"]
            if len(s):
                a.hist(s, bins=60, histtype="step", lw=1.6, color=c,
                       label=f"{t}  (5th pct {s.quantile(.05):.1f})", density=True)
        a.axvline(0, color="black", lw=1)
        a.set_xlabel("game_score - his median"); a.set_ylabel("density")
        a.set_title("By tier -- starters have more room to fall")
        a.legend(fontsize=8)

        a = ax[1, 0]
        a.hist(v, bins=80, color="#4C72B0", edgecolor="white")
        a.set_yscale("log")
        for pid, dt, lab in FLAG:
            m = df[(df.player_id == pid) & (df.gd == dt)]
            if len(m):
                x = m.gs_vs_median.iloc[0]
                a.axvline(x, color="crimson", ls="--", lw=1.2)
                a.text(x, a.get_ylim()[1] * 0.5, f" {lab}", color="crimson",
                       fontsize=7.5, rotation=90, va="top")
        a.set_xlabel("game_score - his median"); a.set_ylabel("player-games (log)")
        a.set_title("Log y, with the flagged games marked")

        a = ax[1, 1]
        b = pd.qcut(df["gs_vs_median"], 10, labels=False, duplicates="drop")
        gg = df.groupby(b)["under_hit"].agg(["mean", "size"])
        a.bar(gg.index, 100 * gg["mean"], color="#4C72B0", edgecolor="white")
        a.axhline(100 * df["under_hit"].mean(), color="crimson", ls="--", lw=1.2,
                  label=f"base {100*df['under_hit'].mean():.1f}%")
        a.set_xlabel("decile (0 = worst vs own median)")
        a.set_ylabel("under hit %"); a.set_title("Outcome by decile")
        a.legend(fontsize=8)

        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    if name == "points_dist":
        import numpy as np
        from scipy import stats
        # The flagged games, drawn on each player's own distribution.
        FOCUS = [("Malik Beasley", ["2024-01-06", "2024-01-26",
                                    "2024-02-27", "2024-03-10"]),
                 ("Jontay Porter", ["2024-01-20", "2024-01-26", "2024-03-20"])]
        df = df.copy()
        df["gd"] = df["game_date"].astype(str).str[:10]

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # League-wide, to establish the shape the z-score assumes away.
        a = ax[0, 0]
        a.hist(df["points"], bins=range(0, 62, 1), color="#4C72B0", edgecolor="white")
        mu, sd = df["points"].mean(), df["points"].std()
        xs = np.linspace(0, 60, 300)
        a.plot(xs, stats.norm.pdf(xs, mu, sd) * len(df), color="crimson", lw=1.6,
               label=f"normal({mu:.1f}, {sd:.1f}) -- what a z-score assumes")
        a.set_xlabel("points"); a.set_ylabel("player-games")
        a.set_title(f"League-wide  (n={len(df):,}, skew {df['points'].skew():+.2f})")
        a.legend(fontsize=8)

        # Per player, with the flagged games marked.
        for i, (who, dates) in enumerate(FOCUS):
            a = ax[0, 1] if i == 0 else ax[1, 0]
            s = df[df.player == who]
            if s.empty:
                continue
            a.hist(s["points"], bins=range(0, int(s.points.max()) + 2),
                   color="#55A868", edgecolor="white")
            a.axvline(s.points.mean(), color="black", ls="-", lw=1.2,
                      label=f"his mean {s.points.mean():.1f}")
            for d in dates:
                row = s[s.gd == d]
                if row.empty:
                    continue
                v = row.points.iloc[0]
                a.axvline(v, color="crimson", ls="--", lw=1.4)
                a.text(v, a.get_ylim()[1] * 0.95, f" {d[5:]}  z={row.z_own.iloc[0]:+.2f}",
                       color="crimson", fontsize=7.5, rotation=90, va="top")
            a.set_xlabel("points"); a.set_ylabel("games")
            a.set_title(f"{who}  (n={len(s)}, mean {s.points.mean():.1f}, "
                        f"sd {s.points.std():.1f})")
            a.legend(fontsize=8)

        # The floor effect: what z does a ZERO earn, by scoring level?
        a = ax[1, 1]
        pl = df.groupby("player").agg(mean=("points", "mean"),
                                      zif=("z_if_zero", "first"),
                                      n=("points", "size"))
        pl = pl[pl["n"] >= 20]
        a.scatter(pl["mean"], pl["zif"], s=10, alpha=0.4, color="#4C72B0")
        for who, _ in FOCUS:
            if who in pl.index:
                r = pl.loc[who]
                a.scatter([r["mean"]], [r["zif"]], s=70, color="crimson", zorder=5)
                a.annotate(who.split()[1], (r["mean"], r["zif"]),
                           textcoords="offset points", xytext=(6, 4),
                           color="crimson", fontsize=9)
        a.axhline(-2, color="grey", ls=":", lw=1, label="z = -2")
        a.set_xlabel("player's mean points"); a.set_ylabel("z-score a ZERO would earn")
        a.set_title("The floor effect: a 0-point game is not equally extreme")
        a.legend(fontsize=8)

        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    if name == "price_only_move":
        import numpy as np
        FLAG = {(1627736, "2024-01-06"): "Beasley 1/06",
                (1627736, "2024-01-26"): "Beasley 1/26"}
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        v = df["price_only_move"].dropna() * 100
        d2 = df.copy()
        d2["gd"] = d2["game_date"].astype(str).str[:10]

        a = ax[0, 0]
        a.hist(v, bins=70, color="#4C72B0", edgecolor="white")
        a.axvline(0, color="black", lw=1)
        for (pid, dt), lab in FLAG.items():
            m = d2[(d2.player_id == pid) & (d2.gd == dt)]
            if len(m):
                x = 100 * m.price_only_move.iloc[0]
                a.axvline(x, color="crimson", ls="--", lw=1.4)
                a.text(x, a.get_ylim()[1] * 0.9, f" {lab}", color="crimson",
                       fontsize=8, rotation=90, va="top")
        a.set_xlabel("price-only move, % change in under price")
        a.set_ylabel("player-games")
        a.set_title(f"Line FLAT, price moved  (n={len(v):,})")

        # ALL price movement, not just the flat-line subset. Restricting to a held
        # line throws away most of the sample and, as it turns out, most of the
        # signal -- under_move_pct reaches z = 2.11 on the full set while the
        # flat-line subset tests null.
        with db.connect() as c2:
            allmv = pd.read_sql(
                "SELECT f.under_move_pct, f.line_move_pct, f.points, f.close_line "
                "FROM player_game_features f "
                "WHERE f.under_move_pct IS NOT NULL AND f.minutes > 0", c2)
        allmv["under_hit"] = allmv.points < allmv.close_line

        a = ax[0, 1]
        a.hist(100 * allmv["under_move_pct"], bins=70, color="#55A868",
               edgecolor="white")
        a.axvline(0, color="black", lw=1)
        a.set_xlabel("under price move, %"); a.set_ylabel("player-games")
        a.set_title(f"ALL price movement, line held or not  (n={len(allmv):,})")

        # Outcome across the full price-move range, the version that does predict.
        a = ax[1, 0]
        b2 = pd.cut(allmv["under_move_pct"] * 100, [-40, -5, -1, 1, 5, 40],
                    labels=["<-5", "-5..-1", "flat", "+1..+5", ">+5"])
        g2 = allmv.groupby(b2)["under_hit"].agg(["mean", "size"])
        a.bar(range(len(g2)), 100 * g2["mean"], color="#55A868", edgecolor="white")
        a.set_xticks(range(len(g2)))
        a.set_xticklabels([f"{i}\nn={int(s):,}" for i, s in zip(g2.index, g2["size"])],
                          fontsize=8)
        a.axhline(100 * allmv["under_hit"].mean(), color="crimson", ls="--", lw=1.2,
                  label=f"base {100*allmv['under_hit'].mean():.1f}%")
        a.set_ylim(45, 60); a.set_ylabel("under hit %")
        a.set_title("ALL price movement (z = 2.11)"); a.legend(fontsize=8)

        # Outcome by size of the price-only move.
        a = ax[1, 1]
        b = pd.cut(df["price_only_move"] * 100,
                   [-25, -5, -2, 2, 5, 25],
                   labels=["<-5", "-5..-2", "flat", "+2..+5", ">+5"])
        g = df.groupby(b)["under_hit"].agg(["mean", "size"])
        a.bar(range(len(g)), 100 * g["mean"], color="#4C72B0", edgecolor="white")
        a.set_xticks(range(len(g)))
        a.set_xticklabels([f"{i}\nn={int(s):,}" for i, s in zip(g.index, g["size"])],
                          fontsize=8)
        a.axhline(100 * df["under_hit"].mean(), color="crimson", ls="--", lw=1.2,
                  label=f"base {100*df['under_hit'].mean():.1f}%")
        a.set_ylim(45, 60); a.set_ylabel("under hit %")
        a.set_title("FLAT-LINE subset only (tests null)"); a.legend(fontsize=8)

        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    if name in ("joint_low", "no_up_move_low_joint"):
        import numpy as np
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        v = df["indep_pct"].dropna()

        # Linear scale first. The product of two uniform-ish percentiles is heavily
        # right-skewed toward zero -- most rows are ordinary on at least one axis --
        # so a linear histogram shows the bulk but crushes the tail we care about.
        a = ax[0, 0]
        a.hist(v, bins=60, color="#4C72B0", edgecolor="white")
        for q, col in ((0.25, "darkorange"), (0.05, "crimson")):
            a.axvline(v.quantile(q), color=col, ls="--", lw=1.2,
                      label=f"{q:.0%} = {v.quantile(q):.4f}")
        a.set_xlabel("p_price x p_line"); a.set_ylabel("player-games")
        a.set_title(f"Linear scale  (n={len(v):,})"); a.legend(fontsize=8)

        # Log x, which is the honest view: the interesting rows span six orders of
        # magnitude, from 1e-6 to 1e-1, and are invisible on a linear axis.
        a = ax[0, 1]
        pos = v[v > 0]
        a.hist(np.log10(pos), bins=60, color="#55A868", edgecolor="white")
        a.set_xlabel("log10(p_price x p_line)"); a.set_ylabel("player-games")
        a.set_title("Log scale -- where the tail actually lives")

        # Is the product doing anything? Outcome by decile of the statistic.
        a = ax[1, 0]
        if "under_hit" in df.columns:
            hit = df["under_hit"].astype(float)
        else:
            hit = (df["points"] < df["line"]).astype(float)
        dec = pd.qcut(df["indep_pct"], 10, labels=False, duplicates="drop")
        g = hit.groupby(dec).agg(["mean", "size"])
        a.bar(g.index, 100 * g["mean"], color="#4C72B0", edgecolor="white")
        a.axhline(100 * hit.mean(), color="crimson", ls="--", lw=1.2,
                  label=f"base {100*hit.mean():.1f}%")
        a.set_xlabel("decile of p_price x p_line  (0 = most extreme)")
        a.set_ylabel("under hit %"); a.set_ylim(40, 70)
        a.set_title("Does the statistic predict?"); a.legend(fontsize=8)

        # The two marginals that build it, to show which one binds.
        a = ax[1, 1]
        a.scatter(df["p_price"], df["p_line"], s=4, alpha=0.15, color="#4C72B0")
        cut = v.quantile(0.25)
        xs = np.linspace(0.0005, 1, 400)
        a.plot(xs, np.clip(cut / xs, 0, 1), color="crimson", lw=1.5,
               label=f"product = {cut:.4f} (25th pct)")
        a.set_xlabel("p_price"); a.set_ylabel("p_line")
        a.set_title("The two marginals; curve is the threshold")
        a.legend(fontsize=8)

        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")

        # ---- cumulative view ------------------------------------------------
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        s = np.sort(v.values)
        cdf = np.arange(1, len(s) + 1) / len(s)

        a = ax[0, 0]
        a.plot(s, cdf, color="#4C72B0", lw=1.6, label="empirical")
        # Product of two INDEPENDENT uniforms has CDF t - t*ln(t). Comparing to it
        # tests the independence assumption the product silently makes.
        t = np.linspace(1e-6, 1, 2000)
        a.plot(t, t - t * np.log(t), color="crimson", ls="--", lw=1.2,
               label="if perfectly independent:  t - t ln t")
        a.set_xlabel("p_price x p_line"); a.set_ylabel("cumulative fraction")
        a.set_title("Empirical CDF vs the independence prediction")
        a.legend(fontsize=8); a.grid(alpha=0.25)

        a = ax[0, 1]
        pos = np.sort(v[v > 0].values)
        a.plot(pos, np.arange(1, len(pos) + 1) / len(pos), color="#55A868", lw=1.6)
        a.set_xscale("log")
        for q, col in ((0.05, "crimson"), (0.25, "darkorange")):
            a.axvline(v.quantile(q), color=col, ls="--", lw=1.1,
                      label=f"{q:.0%} = {v.quantile(q):.4f}")
        a.set_xlabel("p_price x p_line  (log)"); a.set_ylabel("cumulative fraction")
        a.set_title("Log-x CDF -- resolves the tail"); a.legend(fontsize=8)
        a.grid(alpha=0.25)

        # The practically useful one: take the top N, what rate do you get?
        a = ax[1, 0]
        order = df.sort_values("indep_pct")
        h = (order["under_hit"].astype(float) if "under_hit" in order.columns
             else (order["points"] < order["line"]).astype(float))
        run = h.expanding().mean().values
        a.plot(np.arange(1, len(run) + 1), 100 * run, color="#4C72B0", lw=1.4)
        a.axhline(100 * h.mean(), color="crimson", ls="--", lw=1.2,
                  label=f"base {100*h.mean():.1f}%")
        a.set_xscale("log")
        a.set_xlabel("top N by the statistic (log)")
        a.set_ylabel("cumulative under-hit %")
        a.set_title("Running under-hit rate of the top N")
        a.set_ylim(40, 75); a.legend(fontsize=8); a.grid(alpha=0.25)

        # Same, for each marginal, so it is clear which axis carries the signal.
        a = ax[1, 1]
        for col, lab, c in (("indep_pct", "product", "#4C72B0"),
                            ("p_price", "price only", "#55A868"),
                            ("p_line", "line only", "#C44E52")):
            o = df.sort_values(col)
            hh = (o["under_hit"].astype(float) if "under_hit" in o.columns
                  else (o["points"] < o["line"]).astype(float))
            a.plot(np.arange(1, len(o) + 1), 100 * hh.expanding().mean().values,
                   lw=1.4, color=c, label=lab)
        a.axhline(100 * h.mean(), color="grey", ls="--", lw=1)
        a.set_xscale("log"); a.set_xlim(10, len(df))
        a.set_xlabel("top N (log)"); a.set_ylabel("cumulative under-hit %")
        a.set_title("Which axis carries it?"); a.set_ylim(40, 75)
        a.legend(fontsize=8); a.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(OUT / f"{name}_cdf.png", dpi=140)
        print(f"   -> {OUT / f'{name}_cdf.png'}")
        return

    if name == "line_pulled":
        import numpy as np
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        played = df[~df.did_not_play]
        dnp = df[df.did_not_play]

        a = ax[0, 0]
        a.bar(["did NOT play", "played anyway"], [len(dnp), len(played)],
              color=["#C44E52", "#4C72B0"])
        for i, v in enumerate([len(dnp), len(played)]):
            a.text(i, v, f"{v}\n{100*v/len(df):.0f}%", ha="center", va="bottom")
        a.set_title(f"What happened after the line was pulled  (n={len(df)})")
        a.set_ylabel("player-games")
        a.margins(y=0.18)

        a = ax[0, 1]
        bins = np.arange(0, df.open_line.max() + 2, 2)
        a.hist([dnp.open_line, played.open_line], bins=bins, stacked=True,
               color=["#C44E52", "#4C72B0"], label=["did not play", "played"],
               edgecolor="white")
        a.set_title("Opening line at the moment of withdrawal")
        a.set_xlabel("opening line"); a.set_ylabel("player-games"); a.legend()

        a = ax[1, 0]
        hit = 100 * played.under_hit.mean()
        a.bar(["pulled,\nplayed anyway", "line held\nto close"], [hit, 52.3],
              color=["#4C72B0", "#999999"])
        a.axhline(52.3, color="grey", ls="--", lw=1)
        for i, v in enumerate([hit, 52.3]):
            a.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
        a.set_title(f"Under-hit rate  (n={len(played)} played)")
        a.set_ylabel("under hit %"); a.set_ylim(0, 75)

        a = ax[1, 1]
        s = df.dropna(subset=["salary"])
        a.hist([s[s.did_not_play].salary / 1e6, s[~s.did_not_play].salary / 1e6],
               bins=np.arange(0, 55, 5), stacked=True,
               color=["#C44E52", "#4C72B0"], label=["did not play", "played"],
               edgecolor="white")
        a.set_title(f"Salary  ({int((df.has_listed_salary == False).sum())} two-way, "
                    f"not shown)")
        a.set_xlabel("salary ($M)"); a.set_ylabel("player-games"); a.legend()

        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    if name == "price_only":
        # One bar per distinct price. There are only 38 of them across 15,498 games,
        # so binning would blur a discrete grid into a fake continuum -- books quote
        # on fixed increments and the spikes are real, not sampling noise.
        import numpy as np
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2]})
        vc = df["under_price"].value_counts().sort_index()
        a1.bar(vc.index, vc.values, width=0.008, color="#4C72B0")
        med = df["under_price"].median()
        a1.axvline(med, color="crimson", ls="--", lw=1.2, label=f"median {med:.2f}")
        a1.axvline(1.75, color="darkorange", ls=":", lw=1.2,
                   label=f"1.75  ({(df.under_price <= 1.75).sum()} games, "
                         f"{100*(df.under_price <= 1.75).mean():.1f}%)")
        a1.set_ylabel("player-games")
        a1.set_title(f"Closing under price, FanDuel points props  "
                     f"(n={len(df):,}, {df['under_price'].nunique()} distinct prices)")
        a1.legend(fontsize=9)

        # Outcome per price, so the shape and what it predicts sit on one axis.
        grp = df.groupby("under_price")["under_hit"].agg(["mean", "size"])
        grp = grp[grp["size"] >= 40]
        a2.scatter(grp.index, 100 * grp["mean"], s=grp["size"] / 8,
                   color="#55A868", alpha=0.8, edgecolor="white")
        a2.axhline(100 * df["under_hit"].mean(), color="grey", ls="--", lw=1,
                   label=f"base rate {100*df['under_hit'].mean():.1f}%")
        m = np.polyfit(grp.index, 100 * grp["mean"], 1)
        xs = np.linspace(grp.index.min(), grp.index.max(), 50)
        a2.plot(xs, np.polyval(m, xs), color="crimson", lw=1.2,
                label=f"slope {m[0]:.1f} pts per 1.00 of price")
        a2.set_xlabel("closing under price (decimal odds)")
        a2.set_ylabel("under hit %")
        a2.legend(fontsize=9)
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=140)
        print(f"   -> {OUT / f'{name}.png'}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if name == "under_price":
        ax.hist(df["mean_under"], bins=40, color="#4C72B0", edgecolor="white")
        ax.axvline(df["mean_under"].median(), color="crimson", ls="--",
                   label=f"median {df['mean_under'].median():.3f}")
        ax.set_xlabel("mean closing under price (decimal odds)")
        ax.set_ylabel("players")
        ax.set_title(f"Closing under price by player  (n={len(df)})")
        ax.legend()
    else:
        return
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=140)
    print(f"   -> {OUT / f'{name}.png'}")


def main():
    args = sys.argv[1:]
    name = next((a for a in args if not a.startswith("--")), None)
    if name not in QUERIES:
        print("available queries:")
        for k, fn in QUERIES.items():
            print(f"   {k:<24} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return
    n = int(args[args.index("--n") + 1]) if "--n" in args else 25
    with db.connect() as conn:
        df = QUERIES[name](conn)

    print(f"{name}: {len(df):,} rows\n")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.head(n).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n-> {OUT / f'{name}.csv'}  ({len(df):,} rows)")
    if "--plot" in args:
        plot(name, df)
    return df


if __name__ == "__main__":
    main()
