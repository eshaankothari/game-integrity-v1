"""L6b: Isolation Forest over the z-scores, as a COMPLEMENT to the rule-based cuts.

Fitted on ALL propped player-games, not just the rule-based survivors. An outlier
detector needs to see the whole distribution to know what ordinary looks like; fitting
it on pre-filtered rows would only find outliers among outliers.

TWO THINGS TO KNOW BEFORE READING THE OUTPUT
--------------------------------------------
1. IT IS DIRECTION-BLIND. Isolation Forest finds points far from the bulk in ANY
   direction, so a 50-point explosion scores exactly as anomalous as a 4-point
   no-show. `direction` below records which one each row is; the underperformance
   half is the only half relevant here.

2. THE FEATURES ARE CORRELATED. Because the stats are raw rather than per-36,
   minutes/points/fga/touches all move together, and IF's axis-aligned random splits
   handle that badly -- it effectively re-discovers "played a lot vs played a little"
   over and over. Expect it to rank low-minute games highly for that reason alone.

Missing z-scores are IMPUTED WITH 0, which for a z-score means "exactly average".
That is the conservative choice: it makes a missing value look unremarkable rather
than extreme, so the ~21% of rows without tracking data are not flagged for absence
of data. n_present records how many were real.

    python isolation.py            -> out/isolation.csv
"""
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import db

warnings.filterwarnings("ignore")

OUT = pathlib.Path("out/isolation.csv")
OUT_CUT4 = pathlib.Path("out/isolation_cut4.csv")

# ---- THE FEATURE SET -------------------------------------------------------
# Three families, each answering a different question about the same player-game.

# 1. PERFORMANCE, baseline = the player's OWN season. "Unlike himself."
PERF = ["points_z", "fga_z", "rebounds_z", "assists_z", "usage_pct_z",
        "turnover_ratio_z", "distance_z", "touches_z", "minutes_z"]

# 2. BETTING, baseline = the LEAGUE. What the market thought and what it got.
BET = ["close_line_z", "close_under_z", "margin_vs_line_z"]

# 3. RESIDUAL, the same stats after regressing out game context per role tier, then
#    standardised within the player. Strictly better than PERF in principle -- a
#    blowout or a back-to-back is already removed -- so both are fed in and the
#    output shows which the model actually leans on.
RESID = ["points_resid_z", "fga_resid_z", "rebounds_resid_z", "assists_resid_z",
         "usage_pct_resid_z", "turnover_ratio_resid_z", "distance_resid_z",
         "touches_resid_z"]

# EXCLUDED, and the reason matters. These are the direct measures of line movement --
# conceptually the best betting signal available -- but they exist for only 2,169 and
# 1,237 of 15,498 rows, because the opening pass covered 297 events of 1,196.
#
# Missing z-scores are imputed as 0, so switching these on hands the model a column
# that is a literal constant on 86 percent of rows and real on the rest. IF splits on
# whichever axis separates points cheaply, and a near-constant column separates the
# 14 percent that HAVE data from everyone else -- so it would rank "this event got an
# opening snapshot" as an anomaly. That is an artefact of our fetch schedule, not of
# anything a player did.
SPARSE = ["line_move_pct_z", "under_move_pct_z", "price_only_move_z"]
USE_SPARSE = False

FEATURES = PERF + BET + RESID + (SPARSE if USE_SPARSE else [])

# Sign-oriented copy of the features, used ONLY to label direction -- not fed to the
# model. Sums to a negative number when the row is underperformance.
FLIP = {"turnover_ratio_z"}

CONTAMINATION = 0.05        # expected outlier share; only affects the binary label
RANDOM_STATE = 42           # reproducible trees

# Fit on underperformance rows ONLY. Isolation Forest cannot tell a collapse from a
# career night -- both are far from the bulk -- so left unrestricted it hands back a
# list that is half players who played BRILLIANTLY. See the block in main().
UNDER_ONLY = True

SQL = """
WITH ctx AS (
    SELECT game_id, max(pts) - min(pts) AS game_margin
    FROM (SELECT game_id, team_id, sum(points) AS pts
          FROM player_games WHERE points IS NOT NULL GROUP BY 1, 2) t
    GROUP BY game_id HAVING count(*) = 2)
SELECT p.full_name AS player, f.game_date, g.matchup,
       f.minutes, f.points, f.close_line, f.close_under, f.margin_vs_line,
       f.points_z, f.fga_z, f.rebounds_z, f.assists_z, f.usage_pct_z,
       f.turnover_ratio_z, f.distance_z, f.touches_z, f.minutes_z,
       f.close_line_z, f.close_under_z, f.margin_vs_line_z,
       f.line_move_pct_z, f.under_move_pct_z, f.price_only_move_z,
       r.tier,
       r.points_resid_z, r.fga_resid_z, r.rebounds_resid_z, r.assists_resid_z,
       r.usage_pct_resid_z, r.turnover_ratio_resid_z, r.distance_resid_z,
       r.touches_resid_z,
       ctx.game_margin, pg.plus_minus, pg.fouls,
       s.salary, s.has_listed_salary,
       f.n_player_games, f.player_id, f.game_id
FROM player_game_features f
JOIN players      p USING (player_id)
JOIN games        g  ON g.game_id    = f.game_id
JOIN player_games pg ON pg.player_id = f.player_id AND pg.game_id = f.game_id
LEFT JOIN ctx ON ctx.game_id = f.game_id
LEFT JOIN player_game_residuals r
       ON r.player_id = f.player_id AND r.game_id = f.game_id
LEFT JOIN player_salaries s
       ON s.player_id = f.player_id AND s.season = '2023-24'
"""


def apply_cuts(df):
    """export_candidates.py's FIRST THREE cuts -- production, effort, salary.

    Reads its constants from that module rather than restating them, so the two
    files cannot drift apart.

    Stops at three deliberately. Cuts 4 and 5 there (blowout/foul-out, under hit)
    are about the GAME and the BETTING OUTCOME, not the player's performance; leaving
    them off keeps the population defined purely by "played badly, had a motive" and
    lets the model speak to the rest.

    FITTING ON SURVIVORS ASKS A DIFFERENT QUESTION than fitting on everything: not
    "which games are unusual" but "among games that already look wrong, which are the
    strangest". The bulk it compares against is now other bad games, so an ordinary
    bad night stops scoring as an anomaly.
    """
    import export_candidates as EC
    n0 = len(df)
    df["prod_z"], df["n_prod"] = EC.oriented_mean(df, EC.PROD)
    df = df[df["prod_z"] < EC.PROD_THRESHOLD].copy()
    n1 = len(df)
    df["effort_z"], df["n_effort"] = EC.oriented_mean(df, EC.EFFORT)
    df = df[df["effort_z"] < EC.EFFORT_THRESHOLD].copy()
    n2 = len(df)
    df = df[df["salary"].isna() | (df["salary"] <= EC.MAX_SALARY)].copy()
    print(f"CUTS (thresholds prod<{EC.PROD_THRESHOLD:g} effort<{EC.EFFORT_THRESHOLD:g}, "
          f"salary<=${EC.MAX_SALARY:,})")
    print(f"  {n0:,} -> production {n1:,} -> effort {n2:,} -> salary {len(df):,}")
    return df


def main(cut4=False):
    with db.connect() as conn:
        df = pd.read_sql(SQL, conn)

    if cut4:
        # Fitting on the SURVIVORS asks a different question: among games that already
        # look wrong, which are the most unusual? It also removes the direction problem
        # for free -- cut 4 keeps only rows where the under hit, so there is no
        # overperformance half left for the model to waste capacity on.
        df = apply_cuts(df).reset_index(drop=True)

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df["n_present"] = X.notna().sum(axis=1)

    print(f"\nFEATURES fed to the model: {len(FEATURES)}")
    for label, group in (("PERF  player-relative", PERF),
                         ("BET   league-relative", BET),
                         ("RESID context-adjusted", RESID),
                         ("SPARSE excluded" if not USE_SPARSE else "SPARSE included",
                          SPARSE)):
        used = [c for c in group if c in FEATURES]
        for c in group:
            have = X[c].notna().sum() if c in X.columns else \
                   pd.to_numeric(df[c], errors="coerce").notna().sum()
            mark = " " if c in used else "x"
            print(f"  {mark} {label if c == group[0] else '':<23} {c:<24}"
                  f" {have:>6,} / {len(df):,}  {100*have/len(df):>5.1f}%")
    X = X.fillna(0.0)                      # 0 == "average" for a z-score

    # Which SIDE of normal is each row on? Computed BEFORE fitting, because it decides
    # what gets fitted at all.
    oriented = pd.DataFrame({c: X[c] * (-1 if c in FLIP else 1) for c in FEATURES})
    df["direction"] = np.where(oriented.mean(axis=1) < 0, "under", "over")

    # ---- FIXING THE DIRECTION BLINDNESS ------------------------------------
    # Isolation Forest measures distance from the bulk in ANY direction, so a 50-point
    # explosion is exactly as isolated as a 0-point no-show. Fitted on everything it
    # spent half its budget on the wrong tail: 390 under, 385 over.
    #
    # Restricting the FIT to underperformance makes the question "among games where a
    # player did less than usual, which are the strangest" -- and the bulk it compares
    # against is now other bad games, not all games. A quiet night stops being
    # remarkable simply for being below average.
    #
    # It must gate the FIT, not just filter the output. Ranking all rows and keeping
    # the under half leaves the trees built around a distribution containing both
    # tails, so the scores still describe distance from a mixed centre.
    if UNDER_ONLY:
        keep = df["direction"] == "under"
        print(f"\nDIRECTION: fitting on the {int(keep.sum()):,} underperformance rows "
              f"only ({int((~keep).sum()):,} overperformance rows dropped)")
        df = df[keep].reset_index(drop=True)
        X = X[keep.values].reset_index(drop=True)
        oriented = oriented[keep.values].reset_index(drop=True)
    else:
        print("\nDIRECTION: fitting on ALL rows -- output will be ~half overperformance")

    model = IsolationForest(n_estimators=300, contamination=CONTAMINATION,
                            random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X)

    # decision_function: LOWER = more anomalous. Negated so higher = more anomalous,
    # which reads more naturally as a score.
    df["iso_score"] = -model.decision_function(X)
    df["iso_outlier"] = model.predict(X) == -1
    df["iso_rank"] = df["iso_score"].rank(ascending=False, method="min").astype(int)

    # WHICH FEATURE IS DOING THE WORK. IF has no coefficients, so this compares each
    # feature's mean |z| among the flagged rows against everyone else. A large gap
    # means the model is separating on that axis; a gap near zero means the feature is
    # along for the ride. Worth reading before trusting any ranking.
    flag = df.iso_outlier
    lean = sorted(((c, abs(X.loc[flag, c]).mean() - abs(X.loc[~flag, c]).mean())
                   for c in FEATURES), key=lambda t: -t[1])
    print("\nWHAT THE MODEL IS SEPARATING ON  (mean |z| flagged - mean |z| rest)")
    for c, d in lean:
        bar = "#" * int(max(d, 0) * 20)
        print(f"   {c:<24} {d:+.3f}  {bar}")

    print(f"\nfitted on {len(df):,} rows, {len(FEATURES)} features")
    print(f"  rows with all {len(FEATURES)} features real : "
          f"{(df.n_present == len(FEATURES)).sum():,}")
    print(f"  flagged outliers (contamination={CONTAMINATION}) : {df.iso_outlier.sum():,}")
    print()
    print("DIRECTION of the flagged outliers -- IF cannot distinguish these itself:")
    print(df[df.iso_outlier].direction.value_counts().to_string())
    print()
    print("=== TOP 15 by iso_score ===")
    c = ["iso_rank", "player", "game_date", "direction", "iso_score", "minutes",
         "points", "close_line", "margin_vs_line", "minutes_z", "n_present"]
    print(df.nlargest(15, "iso_score")[c].round(3).to_string(index=False))
    print()
    print("=== TOP 15 restricted to direction == 'under' ===")
    print(df[df.direction == "under"].nlargest(15, "iso_score")[c].round(3)
          .to_string(index=False))

    dest = OUT_CUT4 if cut4 else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("iso_score", ascending=False).to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(df):,} rows)")


if __name__ == "__main__":
    import sys
    main(cut4="cut4" in sys.argv[1:])
