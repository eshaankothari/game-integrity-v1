"""L8 (experimental): supervised weights from the known-flagged games.

FOUR POSITIVES. This is far below what logistic regression needs -- the usual rule is
about ten positive events per feature, so 18 features would want ~180. At n=4 the fit
is in the complete-separation regime: it can reproduce the labels exactly while
learning nothing that transfers, and unpenalised coefficients run to infinity.

Two things make the output worth looking at anyway:
  - Strong L2 (small C) keeps coefficients finite and shrinks them toward zero, so
    what survives is the direction the data leans, not a fitted magnitude.
  - LEAVE-ONE-OUT is the only honest test available. Fit on 3, then ask where the
    held-out 4th ranks among ~15,000. If it lands in the top fraction of a percent
    the signal generalises across games; if it lands mid-pack the model memorised
    the three it saw.

Read the LOO block, not the coefficients.

    python fit_logistic.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
import db
import score_candidates as S

warnings.filterwarnings("ignore")

# (player_id, game_date) of games known or believed to be compromised.
FLAGGED = [
    (1627736, "2023-12-25"),   # Beasley  MIL @ NYK   0 pts on 10.5
    (1627736, "2024-01-06"),   # Beasley  MIL @ HOU   3 pts on 9.5
    (1627736, "2024-01-26"),   # Beasley  CLE @ MIL   3 pts on 9.5
    (1629007, "2024-03-20"),   # Porter   SAC @ TOR   0 pts on 7.5
]

FEATURES = [
    "shortfall",
    "points_resid_z", "fga_resid_z", "rebounds_resid_z", "assists_resid_z",
    "usage_pct_resid_z", "turnover_ratio_resid_z", "distance_resid_z",
    "touches_resid_z", "minutes_resid_z",
    "close_line", "close_under", "under_move_pct",
    "motive", "game_margin", "fouls", "minutes", "experience",
]


def load():
    with db.connect() as conn:
        df = pd.read_sql(S.SQL.format(season=config.SEASON, book=config.BOOK), conn)
    df = df[df["minutes"] > 0].copy()
    df["shortfall"] = (1 - df["points"] / df["line"].replace(0, np.nan)).clip(0, 1)
    df["motive"] = np.where(df["has_listed_salary"] == False, 1.0,
                            1 - pd.to_numeric(df["salary_pct"], errors="coerce"))
    df["key"] = list(zip(df["player_id"], df["game_date"].astype(str)))
    df["y"] = df["key"].isin(FLAGGED).astype(int)
    return df.reset_index(drop=True)


def design(df):
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    # Median, not zero: these are raw units as well as z-scores, so 0 is not a neutral
    # value for close_line or minutes the way it is for a z.
    X = X.fillna(X.median())
    return StandardScaler().fit_transform(X)


def fit(X, y, C=0.05):
    """Strong L2. class_weight balanced or 4 positives are simply ignored."""
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    m.fit(X, y)
    return m


def main():
    df = load()
    X, y = design(df), df["y"].values
    print(f"rows {len(df):,}   positives {int(y.sum())}   "
          f"features {len(FEATURES)}   ratio 1 : {int((1-y.mean())/y.mean()):,}")
    if y.sum() != len(FLAGGED):
        missing = set(FLAGGED) - set(df.loc[df.y == 1, "key"])
        print(f"!! {len(missing)} flagged games not found: {missing}")

    # ---- full fit, for direction only ---------------------------------------
    m = fit(X, y)
    coef = pd.Series(m.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print(f"\nCOEFFICIENTS (standardised, C=0.05 -- direction only, magnitudes are "
          f"not identified at n=4)")
    for k, v in coef.items():
        print(f"   {k:<24} {v:+.3f}  {'#' * int(min(abs(v) * 30, 40))}")

    df["p_fit"] = m.predict_proba(X)[:, 1]
    df["rank_fit"] = df["p_fit"].rank(ascending=False, method="min").astype(int)
    print(f"\nIN-SAMPLE ranks of the 4 (meaningless -- they trained on themselves):")
    for _, r in df[df.y == 1].sort_values("rank_fit").iterrows():
        print(f"   {r.player:<16} {r.game_date}  rank {r.rank_fit:>6,} of {len(df):,}")

    # ---- leave-one-out: the only honest number here --------------------------
    print(f"\nLEAVE-ONE-OUT  (fit on 3, rank the held-out 4th among all {len(df):,})")
    loo = []
    for i in np.flatnonzero(y):
        y_tr = y.copy()
        y_tr[i] = 0                       # hide this one
        p = fit(X, y_tr).predict_proba(X)[:, 1]
        rank = int((p > p[i]).sum()) + 1
        loo.append(rank)
        r = df.iloc[i]
        print(f"   {r.player:<16} {r.game_date}  rank {rank:>6,}"
              f"   top {100*rank/len(df):.2f}%")
    print(f"\n   median held-out rank : {int(np.median(loo)):,} of {len(df):,}")
    print(f"   expected by chance   : {len(df)//2:,}")

    # ---- how the hand-weighted score does on the same 4 ----------------------
    parts = {
        "shortfall":   S.pct_rank(df["shortfall"]),
        "production":  S.pct_rank(-S.oriented_mean(df, S.PROD)[0]),
        "involvement": S.pct_rank(-S.oriented_mean(df, S.EFFORT)[0]),
        "motive":      S.pct_rank(df["motive"]),
        "pulled":      df["line_pulled"].astype(float),
        "under_money": S.pct_rank(-pd.to_numeric(df["under_move_pct"], errors="coerce")),
    }
    tw = sum(S.WEIGHTS.values())
    df["p_hand"] = sum(S.WEIGHTS[k] * v for k, v in parts.items()) / tw
    df["rank_hand"] = df["p_hand"].rank(ascending=False, method="min").astype(int)
    print(f"\nHAND-WEIGHTED score on the same population (no labels used at all):")
    for _, r in df[df.y == 1].sort_values("rank_hand").iterrows():
        print(f"   {r.player:<16} {r.game_date}  rank {r.rank_hand:>6,}"
              f"   top {100*r.rank_hand/len(df):.2f}%")
    return df


if __name__ == "__main__":
    main()
