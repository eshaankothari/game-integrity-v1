"""A learned, MARKET-INDEPENDENT expectation for how many points a player should score.

WHAT PROBLEM THIS SOLVES. Every performance measure in the pipeline is anchored to
something borrowed:

    shortfall = 1 - points/close_line     anchored to the MARKET. Unscoreable without a
                                          posted line, and a manipulated line corrupts
                                          the metric that is meant to detect it.
    game_z    = (x - own mean) / own sd    anchored to the PLAYER. Bounded below by his
                                          own average, which is the floor effect: a
                                          4-pt/game player scoring zero is -0.97 sd
                                          while a star is -3.23, so any threshold on it
                                          is secretly a threshold on scoring level.

This learns the anchor instead. Given only what was knowable BEFORE tip-off, what
should he have scored -- and how surprising is what he actually did?

WHY THIS IS FITTABLE WHEN THE LOGISTIC WAS NOT. The logistic had 6 positives from 2
players and learned "Malik Beasley". This has 26,339 labels (points scored), and is
validated on held-out FUTURE games. There is no label scarcity and no identity to
memorise, because the target is a box-score number rather than a verdict.

STRICTLY PREGAME FEATURES. Anything produced by the game being predicted is excluded,
which rules out the obvious mistakes:

    minutes      EXCLUDED. It is an outcome and, worse, a mediator -- disengaged, then
                 benched, then fewer minutes. A model that knows minutes predicts
                 points nearly perfectly and explains away the entire signal.
    close_line   EXCLUDED. The whole point is an estimate the market did not supply, so
                 it can be COMPARED against the market rather than derived from it.
    box score    EXCLUDED. Every rolling feature is shifted one game back.

A DISTRIBUTION, NOT A POINT ESTIMATE. Quantile regression across a grid gives the full
predictive law, so the output is a calibrated probability:

    surprise = 1 - P(points <= observed | pregame context)

which is on a probability scale for everyone. It has no floor: a low-usage player who
scores zero gets whatever probability that event actually carries for a player like him
in a game like this, not a z-score bounded by his own mean.

THE HEADLINE TEST is against the closing line, the sharpest public forecast of this
quantity that exists. Matching it on held-out games means the model is genuinely
calibrated; losing badly to it means the features are too thin to trust.

    python analysis/expect_model.py
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

import config
import db

OUT = HERE / "out"
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)
TEST_FROM = "2024-02-01"          # time split: fit on the past, score the future
FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}

SQL = """
SELECT pg.player_id, pg.game_id, g.game_date, p.full_name AS player,
       pg.points, pg.minutes, pg.fga, pg.usage_pct, pg.started,
       pg.team_id, gc.opp_team_id, gc.is_home, gc.rest_days, gc.is_b2b,
       gc.altitude_ft, gc.month,
       f.close_line, s.salary, s.has_listed_salary
  FROM player_games pg
  JOIN players p ON p.player_id = pg.player_id
  JOIN games   g ON g.game_id  = pg.game_id
  LEFT JOIN game_context gc ON gc.game_id = pg.game_id AND gc.team_id = pg.team_id
  LEFT JOIN player_game_features f ON f.player_id = pg.player_id
                                  AND f.game_id  = pg.game_id
  LEFT JOIN player_salaries s ON s.player_id = pg.player_id AND s.season = %(season)s
 WHERE pg.points IS NOT NULL
"""


def load():
    with db.connect() as c:
        d = pd.read_sql(SQL, c, params={"season": config.SEASON})
    d["game_date"] = pd.to_datetime(d.game_date)
    d["gd"] = d.game_date.dt.strftime("%Y-%m-%d")
    for c_ in ("points", "minutes", "fga", "usage_pct", "close_line", "salary",
               "rest_days", "altitude_ft", "is_home", "is_b2b", "month"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    return d.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def prior(g, col, win=None):
    """Player history STRICTLY BEFORE the current game.

    shift(1) first, then the window. Without the shift the current game is inside its
    own feature and the model reads the answer off the input -- the single easiest way
    to build something that validates beautifully and means nothing."""
    s = g[col].shift(1)
    return s.rolling(win, min_periods=1).mean() if win else s.expanding().mean()


def build(d):
    gp = d.groupby("player_id", group_keys=False)
    # form: season-to-date and two shorter windows, so both the level and the trend
    # are available to the model
    for col in ("points", "minutes", "fga", "usage_pct"):
        d[f"{col}_std"] = gp.apply(lambda g: prior(g, col))
    for win in (3, 5, 10):
        d[f"points_r{win}"] = gp.apply(lambda g: prior(g, "points", win))
        d[f"minutes_r{win}"] = gp.apply(lambda g: prior(g, "minutes", win))
    d["points_sd"] = gp.apply(
        lambda g: g["points"].shift(1).expanding().std())
    d["started_std"] = gp.apply(lambda g: prior(g, "started"))
    d["n_prior"] = gp.cumcount()
    d["dnp_prior"] = gp.apply(
        lambda g: (g["minutes"].shift(1).fillna(0) == 0).expanding().mean())
    d["trend"] = d.points_r3 - d.points_std          # hot or cold relative to himself
    d["days_into"] = (d.game_date - d.game_date.min()).dt.days

    # opponent strength: points allowed per game BEFORE this date, so it is pregame.
    tg = (d.groupby(["opp_team_id", "game_date"])["points"].sum()
            .reset_index().sort_values(["opp_team_id", "game_date"]))
    tg["opp_pts_allowed"] = (tg.groupby("opp_team_id")["points"]
                               .transform(lambda s: s.shift(1).expanding().mean()))
    d = d.merge(tg[["opp_team_id", "game_date", "opp_pts_allowed"]],
                on=["opp_team_id", "game_date"], how="left")
    return d


FEATS = ["points_std", "minutes_std", "fga_std", "usage_pct_std",
         "points_r3", "points_r5", "points_r10",
         "minutes_r3", "minutes_r5", "minutes_r10",
         "points_sd", "started_std", "n_prior", "dnp_prior", "trend",
         "is_home", "rest_days", "is_b2b", "altitude_ft", "month", "days_into",
         "opp_pts_allowed"]


def main():
    d = build(load())
    # A player with no history has no features; those rows can be predicted but carry
    # no information, so they are held out of the FIT rather than fed as noise.
    d = d[d.n_prior >= 5].reset_index(drop=True)
    tr = d.game_date < TEST_FROM
    te = ~tr
    print(f"rows {len(d):,}   train {int(tr.sum()):,} (before {TEST_FROM})   "
          f"test {int(te.sum()):,}")
    print(f"features {len(FEATS)}   quantiles {len(QUANTILES)}\n")

    Xtr, ytr = d.loc[tr, FEATS], d.loc[tr, "points"]
    X = d[FEATS]

    base = dict(max_iter=300, learning_rate=0.06, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    mean_m = HistGradientBoostingRegressor(loss="poisson", **base).fit(Xtr, ytr)
    d["pred"] = mean_m.predict(X)

    # ---- quantile grid -> full predictive distribution ----------------------
    Q = np.empty((len(d), len(QUANTILES)))
    for j, q in enumerate(QUANTILES):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **base)
        Q[:, j] = m.fit(Xtr, ytr).predict(X)
    # Quantile models are fitted independently and can cross; a sort restores
    # monotonicity, which is the standard fix and changes nothing else.
    Q.sort(axis=1)
    for j, q in enumerate(QUANTILES):
        d[f"q{int(q*100)}"] = Q[:, j].round(2)

    obs = d.points.values
    # P(points <= observed): where the observation falls in the predicted quantile
    # grid, linearly interpolated between levels and clipped at the grid edges.
    p = np.array([np.interp(o, Q[i], QUANTILES, left=0.02, right=0.98)
                  for i, o in enumerate(obs)])
    d["p_model"] = p.round(4)
    d["surprise"] = (1 - p).round(4)
    d["resid"] = (d.points - d.pred).round(2)

    # ---- HEADLINE: model vs the closing line, held-out games only ------------
    m_ = d[te & d.close_line.notna()]
    print(f"HELD-OUT ACCURACY vs THE CLOSING LINE   ({len(m_):,} propped test games)")
    print(f"   {'':<22}{'MAE':>8}{'RMSE':>8}{'corr':>8}")
    for nm, pr in (("model (no line seen)", m_.pred), ("closing line", m_.close_line)):
        e = m_.points - pr
        print(f"   {nm:<22}{e.abs().mean():>8.3f}{np.sqrt((e**2).mean()):>8.3f}"
              f"{pr.corr(m_.points):>8.3f}")
    e0 = (m_.points - m_.pred).abs().mean()
    e1 = (m_.points - m_.close_line).abs().mean()
    print(f"\n   model is {100*(e0-e1)/e1:+.1f}% on MAE relative to the market.")
    print(f"   baseline (predict his season-to-date average): "
          f"{(m_.points - m_.points_std).abs().mean():.3f} MAE")

    # ---- calibration of the distribution ------------------------------------
    print(f"\nCALIBRATION on held-out games -- share of actuals below each quantile")
    print(f"   {'nominal':>9}{'actual':>9}")
    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        col = f"q{int(q*100)}"
        print(f"   {q:>9.2f}{(d.loc[te, 'points'] <= d.loc[te, col]).mean():>9.3f}")

    # ---- what it learned ----------------------------------------------------
    from sklearn.inspection import permutation_importance
    sub = d[te].sample(min(3000, int(te.sum())), random_state=0)
    imp = permutation_importance(mean_m, sub[FEATS], sub.points,
                                 n_repeats=5, random_state=0, n_jobs=-1)
    print(f"\nPERMUTATION IMPORTANCE (held-out, top 10)")
    for i in np.argsort(-imp.importances_mean)[:10]:
        print(f"   {FEATS[i]:<18}{imp.importances_mean[i]:>8.3f}")

    # ---- the flagged games --------------------------------------------------
    f = d[[r.player in FLAG and r.gd in FLAG[r.player] for _, r in d.iterrows()]]
    prop = d[d.close_line.notna()].copy()
    prop["sr"] = prop.surprise.rank(ascending=False, method="min").astype(int)
    print(f"\nFLAGGED -- model expectation vs what happened")
    j = f.merge(prop[["player_id", "game_id", "sr"]], on=["player_id", "game_id"],
                how="left")
    cols = ["player", "gd", "minutes", "points", "close_line", "pred", "q50",
            "surprise", "sr"]
    print(j.sort_values("surprise", ascending=False)[cols].to_string(index=False))

    print(f"\nTOP 15 BY SURPRISE  (propped games, whole season)")
    y = prop.nlargest(15, "surprise")[
        ["player", "gd", "minutes", "points", "close_line", "pred", "surprise"]]
    print(y.to_string(index=False))

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    a.scatter(m_.pred, m_.points, s=6, alpha=.15, color="#4C78A8", rasterized=True)
    lim = [0, m_.points.max()]
    a.plot(lim, lim, color="black", ls="--", lw=1)
    a.set_xlabel("model prediction"); a.set_ylabel("actual points")
    a.set_title(f"Held-out fit  (MAE {e0:.2f} vs market {e1:.2f})")
    a.grid(alpha=.2)

    a = ax[1]
    a.scatter(m_.close_line, m_.pred, s=6, alpha=.15, color="#54A24B", rasterized=True)
    a.plot(lim, lim, color="black", ls="--", lw=1)
    a.set_xlabel("closing line"); a.set_ylabel("model prediction")
    a.set_title(f"Model vs market  (corr {m_.pred.corr(m_.close_line):+.3f})")
    a.grid(alpha=.2)

    a = ax[2]
    a.hist(d.loc[te, "p_model"], bins=40, color="#B0B0B0", edgecolor="white")
    a.set_xlabel("P(points <= observed)")
    a.set_ylabel("held-out games")
    a.set_title("Calibration -- flat is well-calibrated")
    fig.tight_layout(); fig.savefig(OUT / "expect_model.png", dpi=140)

    keep = ["player_id", "game_id", "player", "gd", "minutes", "points", "close_line",
            "pred", "q05", "q25", "q50", "q75", "q95", "p_model", "surprise", "resid"]
    d[keep].to_csv(OUT / "expect_model.csv", index=False)
    print(f"\n-> {OUT/'expect_model.csv'}\n-> {OUT/'expect_model.png'}")
    return d


if __name__ == "__main__":
    main()
