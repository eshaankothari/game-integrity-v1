"""L5c: context-adjusted residuals -> `player_game_residuals`. No API calls.

Instead of CUTTING games with an innocent explanation, ADJUST for it and look at what
is left over. Thresholds stop being arbitrary, no game is discarded, and the residual
is the anomaly.

    residual = actual - predicted(context)      in the stat's own units
    z        = residual standardised within the player

RESIDUAL FIRST, THEN Z-SCORE. Z-scoring first would divide by the player's raw
standard deviation, which is inflated by the very context variation we are removing --
so a player whose team plays lots of blowouts would have all his z-scores compressed.
Residualising first gives a tighter, context-free denominator.

FIT LEAGUE-WIDE WITH PLAYER FIXED EFFECTS, not per player. Seven covariates against
~15 games each would be overfit. `demean` subtracts each player's own mean, which is
algebraically the same as a dummy per player without a 500-column matrix, so a
coefficient means "what a back-to-back costs a typical player, holding his level fixed".

MINUTES IS NOT A COVARIATE, but it IS a target. Two different roles, and the
distinction is the whole point:

  - NOT a predictor of points. If minutes explained points, a short night would be
    "explained away" and a player who limited his own involvement would look normal.
  - IS a target of its own regression. Minutes genuinely respond to game context --
    a starter sits in a blowout -- and that response is innocent. Regressing minutes
    on abs_margin and the rest strips the garbage-time explanation and keeps only the
    part the score cannot account for, which is the part worth looking at.

So points_resid answers "did he score less than context predicts", and minutes_resid
answers "was he on the floor less than the situation explains". Neither absorbs the
other.

    python residualize.py            # DRY
    python residualize.py run        # write
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import db

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

STATS = ["points", "fga", "rebounds", "assists", "usage_pct",
         "turnover_ratio", "distance", "touches", "minutes"]

CONTEXT = ["abs_margin", "rest_days", "is_b2b", "is_home",
           "altitude_ft", "pace", "days_into_season"]

# ---- TWO-STAGE ------------------------------------------------------------
# MINUTES IS A MEDIATOR, NOT A CONFOUNDER, and that distinction decides the design.
# Two causal paths run through it:
#
#     blowout      -> fewer minutes -> fewer points          CONFOUND, remove it
#     disengaged   -> benched       -> fewer points          SIGNAL, keep it
#
# Putting ACTUAL minutes on the right-hand side of a points regression removes both.
# A player who got himself benched has his low scoring fully explained, which is
# precisely the behaviour being hunted.
#
# So stage 1 predicts minutes from EXOGENOUS situation only -- things the player did
# not choose. Stage 2 then offsets points by that PREDICTION rather than by what he
# actually played. The blowout path is closed, the behavioural path stays open, and a
# player benched for reasons the situation does not explain shows a negative residual
# TWICE: once in minutes, once in production.
EXOG = ["competitive_sec", "fouls", "rest_days", "is_b2b", "is_home",
        "altitude_ft", "pace", "days_into_season", "team_win_pct", "opp_win_pct"]

# `fouls` is the one debatable member -- a player can foul deliberately. It is kept
# because foul trouble genuinely constrains a coach's hand, and excluding it would
# push real foul-limited nights into the residual as though they were choices.
TWO_STAGE = True

MIN_GAMES = 8          # fewer than this and a residual sd is not a scale

# ROLE TIERS, by a player's season-average minutes.
#
# A single pooled model forces every player to share one slope, and abs_margin is
# where that breaks: in a blowout a STARTER gets pulled and scores less, while a
# BENCH player gets garbage time and scores MORE. Fitted separately:
#
#     bench (<15 min)     abs_margin  +0.0300      blowout helps him
#     rotation (15-28)                -0.0069      roughly nothing
#     starter (>28)                   -0.1571      blowout costs him
#     POOLED                          -0.0459      wrong for all three
#
# The pooled figure is a weighted average of opposing effects -- 3.4x too small for
# starters and the wrong SIGN for the bench. So each tier gets its own regression.
#
# Caveat: season-average minutes is slightly circular (minutes are partly caused by
# the same things being adjusted for) and gives a player one label for the season even
# if his role changed. `started` would be per-game but cannot separate a sixth man
# from a deep reserve.
TIER_EDGES = [0, 15, 28, 100]
TIER_NAMES = ["bench", "rotation", "starter"]

SQL = """
-- Pace as a property of the GAME, not of one player's stint.
--
-- pg.pace is possessions-per-48 while that player was on the floor, so it divides by
-- his minutes and detonates for a cameo: Jay Huff played 1 second in 0022300257 and
-- scored pace = 3,600 against a league median of 100.5. Sixty rows sit above 200. As
-- a regression covariate those are pure leverage -- one of them drags the slope for
-- everyone in the tier.
--
-- The median over players with real minutes is what "how fast was this game" actually
-- means, and a median ignores the cameos rather than being pulled by them. The 10-
-- minute floor is a rounding of "was he in the rotation tonight".
WITH game_pace AS (
    SELECT game_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY pace) AS pace
    FROM player_games
    WHERE minutes >= 10 AND pace IS NOT NULL
    GROUP BY game_id),
wins AS (        -- season win rate, a crude proxy for how much a team rests starters
    SELECT team_id, avg((margin > 0)::int)::float AS win_pct
    FROM game_context WHERE margin IS NOT NULL GROUP BY team_id)
SELECT pg.player_id, pg.game_id, g.game_date,
       pg.points, pg.fga, pg.rebounds, pg.assists,
       pg.usage_pct, pg.turnover_ratio, pg.distance, pg.touches, pg.minutes,
       gp.pace,
       avg(pg.minutes) OVER (PARTITION BY pg.player_id) AS season_min,
       c.abs_margin, c.rest_days, c.is_b2b, c.is_home, c.altitude_ft,
       pg.fouls, gc.competitive_sec,
       w.win_pct AS team_win_pct, wo.win_pct AS opp_win_pct,
       (g.game_date - (SELECT min(game_date) FROM games))::int AS days_into_season
FROM player_games pg
JOIN games        g ON g.game_id = pg.game_id
JOIN game_context c ON c.game_id = pg.game_id AND c.team_id = pg.team_id
LEFT JOIN game_pace gp ON gp.game_id = pg.game_id
LEFT JOIN game_pbp_context gc ON gc.game_id = pg.game_id
LEFT JOIN wins w  ON w.team_id  = pg.team_id
LEFT JOIN wins wo ON wo.team_id = c.opp_team_id
WHERE pg.minutes IS NOT NULL
ORDER BY pg.player_id, g.game_date
"""

DDL = """
CREATE TABLE IF NOT EXISTS player_game_residuals (
    player_id      INTEGER NOT NULL REFERENCES players (player_id),
    game_id        TEXT    NOT NULL REFERENCES games (game_id),
    game_date      DATE    NOT NULL,
    n_player_games INTEGER,
    tier           TEXT,
""" + ",\n".join(
    f"    {s}_resid NUMERIC(9,3),\n    {s}_resid_z NUMERIC(8,4)" for s in STATS
) + """,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, game_id)
);
"""

# THE MODEL ITSELF, so a fit is reproducible and a NEW game can be scored without
# refitting. To predict one player-game you need three things, and all three live here:
#
#     predicted = baseline(stat)                            <- player_baselines
#                 + SUM over c of  coef_c                   <- residual_coefs
#                                  * (context_c - baseline(context_c))
#
# residual_models carries the fit quality alongside, so a coefficient is never read
# without knowing how much it actually explained.
MODEL_DDL = """
CREATE TABLE IF NOT EXISTS residual_models (
    stat              TEXT NOT NULL,
    tier              TEXT NOT NULL,
    n_rows            INTEGER,
    n_players         INTEGER,
    raw_sd            NUMERIC(12,4),   -- spread across all players
    demeaned_sd       NUMERIC(12,4),   -- spread within a player (what we can explain)
    resid_sd          NUMERIC(12,4),   -- spread left over
    explained_total   NUMERIC(7,4),    -- vs raw_sd:      mostly WHO the player is
    explained_context NUMERIC(7,4),    -- vs demeaned_sd: what CONTEXT actually bought
    intercept         NUMERIC(14,6),   -- ~0 by construction; a check on the demeaning
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stat, tier)
);
CREATE TABLE IF NOT EXISTS residual_coefs (
    stat        TEXT NOT NULL,
    tier        TEXT NOT NULL,
    covariate   TEXT NOT NULL,
    coef        NUMERIC(14,6),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stat, tier, covariate)
);
CREATE TABLE IF NOT EXISTS player_baselines (
    player_id   INTEGER NOT NULL REFERENCES players (player_id),
    tier        TEXT,
    column_name TEXT    NOT NULL,
    mean_value  NUMERIC(14,6),
    n_games     INTEGER,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, column_name)
);
"""


def load(conn):
    df = pd.read_sql(SQL, conn)
    for c in STATS + ["pace", "abs_margin", "rest_days", "altitude_ft",
                      "days_into_season"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_b2b"] = df["is_b2b"].astype(float)
    df["is_home"] = df["is_home"].astype(float)
    # NULL only on a team's season opener -- undefined, not missing.
    df["rest_days"] = df["rest_days"].fillna(df["rest_days"].median())
    df["season_min"] = pd.to_numeric(df["season_min"], errors="coerce")
    df["tier"] = pd.cut(df["season_min"], TIER_EDGES, labels=TIER_NAMES)
    # NULL where L4 has not fetched the game; those rows cannot be adjusted at all.
    return df.dropna(subset=["abs_margin", "pace", "tier"]).reset_index(drop=True)


def demean(df, cols):
    """Subtract each player's own mean -- the fixed-effects trick."""
    g = df.groupby("player_id")
    return pd.DataFrame({c: df[c] - g[c].transform("mean") for c in cols},
                        index=df.index)


def stage1_minutes(df):
    """Predict minutes from EXOGENOUS situation only, per tier, player-demeaned.

    Returns (minutes_hat, minutes_resid) in demeaned units. minutes_hat is the offset
    stage 2 uses in place of actual minutes; minutes_resid is 'was he on the floor
    less than the situation explains' and is a FEATURE in its own right.
    """
    hat = pd.Series(np.nan, index=df.index)
    res = pd.Series(np.nan, index=df.index)
    cols = ["minutes"] + EXOG
    for tier in TIER_NAMES:
        sel = (df["tier"] == tier) & df[cols].notna().all(axis=1)
        if sel.sum() < 100:
            continue
        d = demean(df[sel], cols)
        model = LinearRegression().fit(d[EXOG], d["minutes"])
        pred = model.predict(d[EXOG])
        hat[sel] = pred
        res[sel] = d["minutes"] - pred
    return hat, res


def residual(df, stat):
    """actual - predicted(context), fitted SEPARATELY PER TIER.

    Returns (residual series, [fit dicts]). Each tier gets its own slopes because
    context does not act the same way on a starter and a reserve.
    """
    out = pd.Series(np.nan, index=df.index)
    fits = []
    # Two-stage: EXOG situation plus PREDICTED minutes. Never actual minutes -- see
    # the note on EXOG. Minutes itself is fitted in stage 1 and skips this path.
    if TWO_STAGE and stat != "minutes":
        CTX = EXOG + ["minutes_hat"]
    elif TWO_STAGE:
        CTX = EXOG
    else:
        CTX = CONTEXT
    cols = [stat] + CTX
    for tier in TIER_NAMES:
        sel = (df["tier"] == tier) & df[cols].notna().all(axis=1)
        if sel.sum() < 100:                # too few rows to fit the covariates
            continue
        d = demean(df[sel], cols)
        model = LinearRegression().fit(d[CTX], d[stat])
        r = d[stat] - model.predict(d[CTX])
        out[sel] = r
        raw, dem, res = df.loc[sel, stat].std(), d[stat].std(), r.std()
        fits.append({
            "stat": stat, "tier": tier,
            "n_rows": int(sel.sum()),
            "n_players": int(df.loc[sel, "player_id"].nunique()),
            "raw_sd": raw, "demeaned_sd": dem, "resid_sd": res,
            # Two very different numbers. Against raw_sd almost all the "explaining"
            # is just knowing WHO the player is; against demeaned_sd is the honest
            # measure of what the context covariates bought on top of that.
            "explained_total": 1 - (res / raw) ** 2 if raw else None,
            "explained_context": 1 - (res / dem) ** 2 if dem else None,
            "intercept": float(model.intercept_),
            "coefs": dict(zip(CTX, model.coef_)),
        })
    return out, fits


def baselines(df):
    """Each player's mean for every stat and covariate -- the fixed effects that
    `demean` removes. Without these a stored coefficient cannot score a new game."""
    cols = STATS + CONTEXT
    g = df.groupby("player_id")
    means, counts = g[cols].mean(), g[cols].count()
    tiers = g["tier"].first().astype(str)
    return [{"player_id": int(pid), "tier": tiers[pid], "column_name": c,
             "mean_value": None if pd.isna(means.at[pid, c])
                           else round(float(means.at[pid, c]), 6),
             "n_games": int(counts.at[pid, c])}
            for pid in means.index for c in cols]


def zscore(df, resid):
    """Standardise within the player, against the RESIDUAL spread."""
    g = resid.groupby(df["player_id"])
    n = g.transform("count")
    return ((resid - g.transform("mean")) / g.transform("std").replace(0, np.nan)
            ).where(n >= MIN_GAMES)


def main(dry=True):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(MODEL_DDL)
            # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so adding a
            # stat to STATS would silently fail to produce its columns and the upsert
            # would then fail on an unknown column. Add them explicitly.
            for s in STATS:
                cur.execute(
                    f"ALTER TABLE player_game_residuals "
                    f"ADD COLUMN IF NOT EXISTS {s}_resid NUMERIC(9,3), "
                    f"ADD COLUMN IF NOT EXISTS {s}_resid_z NUMERIC(8,4)")
        df = load(conn)

    out = df[["player_id", "game_id", "game_date"]].copy()
    out["n_player_games"] = df.groupby("player_id")["points"].transform("count")
    out["tier"] = df["tier"].astype(str)

    print(f"player-games : {len(df):,}   players : {df.player_id.nunique():,}")
    print(f"covariates   : {', '.join(EXOG if TWO_STAGE else CONTEXT)}"
          + ("  + minutes_hat" if TWO_STAGE else ""))
    print("tiers        : " + ", ".join(
        f"{t} n={(df.tier == t).sum():,}" for t in TIER_NAMES) + "\n")

    if TWO_STAGE:
        df["minutes_hat"], df["minutes_resid_raw"] = stage1_minutes(df)
        ok = df["minutes_hat"].notna().sum()
        print(f"stage 1      : minutes ~ {len(EXOG)} exogenous covariates"
              f"   fitted on {ok:,} of {len(df):,} rows\n")
    all_fits = []
    print(f"{'stat':16} {'raw sd':>8} {'resid sd':>9} {'expl':>6} {'ctx':>6}"
          f"   key coefficient by tier")
    for s in STATS:
        r, fits = residual(df, s)
        all_fits.extend(fits)
        out[f"{s}_resid"] = r.round(3)
        out[f"{s}_resid_z"] = zscore(df, r).round(4)
        raw, rs = df[s].std(), r.std()
        expl = 1 - (rs / raw) ** 2 if raw else np.nan
        # Rows-weighted mean of the per-tier context R-squared.
        n_tot = sum(f["n_rows"] for f in fits) or 1
        ctx = sum(f["explained_context"] * f["n_rows"] for f in fits) / n_tot
        # Show whichever covariate is the informative one for the active design:
        # minutes_hat under two-stage (how much of the stat the predicted playing
        # time accounts for), abs_margin under the one-stage model.
        key = next((k for k in ("minutes_hat", "abs_margin", "competitive_sec")
                    if fits and k in fits[0]["coefs"]), None)
        by_tier = "  ".join(f"{f['tier'][:4]} {f['coefs'][key]:+.3f}" for f in fits) \
                  if key else ""

        print(f"{s:16} {raw:8.2f} {rs:9.2f} {expl*100:5.0f}% {ctx*100:5.1f}%"
              f"   {by_tier}")

    base = baselines(df)
    print(f"\nmodels   : {len(all_fits)} (stat x tier)"
          f"   coefficients : {len(all_fits) * len(CONTEXT)}")
    print(f"baselines: {len(base):,} rows "
          f"({df.player_id.nunique():,} players x {len(STATS + CONTEXT)} columns)")

    if dry:
        db.dry_notice()
        return out

    models = [{k: v for k, v in f.items() if k != "coefs"} for f in all_fits]
    coefs = [{"stat": f["stat"], "tier": f["tier"], "covariate": c,
              "coef": round(v, 6)}
             for f in all_fits for c, v in f["coefs"].items()]

    rows = out.replace({np.nan: None}).to_dict("records")
    with db.connect() as conn:
        n = db.upsert(conn, "player_game_residuals", rows,
                      conflict=["player_id", "game_id"])
        print(f"\nupserted {n:,} rows -> player_game_residuals")
        for tbl, data, key in (
                ("residual_models", models, ["stat", "tier"]),
                ("residual_coefs", coefs, ["stat", "tier", "covariate"]),
                ("player_baselines", base, ["player_id", "column_name"])):
            print(f"upserted {db.upsert(conn, tbl, data, conflict=key):,} rows "
                  f"-> {tbl}")
    return out


if __name__ == "__main__":
    main(dry=db.is_dry())
