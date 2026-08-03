"""Logistic fit with the identity features removed, tested three ways.

DROPPED: salary, n_games, minutes.
    The previous fit put its two largest coefficients on salary (-5.96) and n_games
    (+3.64) and reached train AUC 0.997 with test AUC 0.087 -- it had learned "earns
    $2.0M, played 79 games", which is Malik Beasley, and Porter (two-way, 26 games)
    scored below random. Removing them forces the model onto behaviour and market.

KEPT: game_z, effort_z, shortfall_z, the four market components, close_line,
    under_price. close_line is retained but is itself partly identity -- a star draws
    a 28.5 line and a reserve a 5.5 -- so watch its coefficient.

THREE SPLITS, because only the cross-player ones can distinguish "learned the
behaviour" from "learned the player":
    A  train Beasley -> test Porter
    B  train Porter  -> test Beasley
    C  random 80/20 across both      <- will look good even if the model is useless
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

OUT = pathlib.Path(__file__).resolve().parent / "out"
d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str}).reset_index(drop=True)

# Two feature sets, run separately.
#
# AS ASKED is collinear by construction: performance = mean(-game_z, -effort_z,
# +shortfall_z), so it is a linear combination of two of the others. L2 will not break
# on that, but it splits the weight arbitrarily between correlated columns and the
# individual coefficients stop being readable.
#
# COMPONENTS drops the composite and keeps its parts, which is the same information
# without the redundancy. close_line and under_price are gone from both -- close_line
# carried the largest coefficient in every previous split and is largely player
# identity, since a two-way player draws a 5.5 line and a star draws 28.5.
SETS = {
  "as asked  (collinear)": ["effort_z", "market", "performance", "shortfall_z"],
  "components (clean)   ": ["game_z", "effort_z", "shortfall_z", "market"],
}
FEATURES = SETS["components (clean)   "]
SUSPECT = ["Malik Beasley", "Jontay Porter"]
d["under_hit"] = d.points < d.close_line
d["y"] = (d.player.isin(SUSPECT) & d.under_hit).astype(int)

X = d[FEATURES].apply(pd.to_numeric, errors="coerce")
Xs = StandardScaler().fit_transform(X.fillna(X.median()))
y = d.y.values
rng = np.random.default_rng(0)
neg = np.flatnonzero(y == 0)
hold_neg = rng.choice(neg, size=len(neg)//5, replace=False)
keep_neg = np.setdiff1d(neg, hold_neg)

def run(train_player, test_player, label, C=0.05, Xs=None, quiet=False):
    trp = [i for i in np.flatnonzero(y) if d.player[i] == train_player] \
          if train_player else None
    tep = [i for i in np.flatnonzero(y) if d.player[i] == test_player] \
          if test_player else None
    if train_player is None:                      # random split across both
        pos = np.flatnonzero(y); rng2 = np.random.default_rng(1)
        tep = list(rng2.choice(pos, size=len(pos)//5, replace=False))
        trp = [i for i in pos if i not in tep]
    tr = np.concatenate([trp, keep_neg]); te = np.concatenate([tep, hold_neg])
    Xs_ = Xs if Xs is not None else globals()["Xs"]
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=8000)
    m.fit(Xs_[tr], y[tr])
    ptr, pte = m.predict_proba(Xs_[tr])[:,1], m.predict_proba(Xs_[te])[:,1]
    atr, ate = roc_auc_score(y[tr], ptr), roc_auc_score(y[te], pte)
    med = int(np.median([(pte > pte[list(te).index(i)]).sum() + 1 for i in tep]))
    print(f"{label:<32}{atr:>15.4f}{ate:>11.4f}{med:>12,}")
    return m

print(f"positives {int(y.sum())}  (Beasley "
      f"{int((d.player=='Malik Beasley').mul(d.y).sum())}, Porter "
      f"{int((d.player=='Jontay Porter').mul(d.y).sum())})   "
      f"test pool ~3,100, random rank ~1,550\n")

for name, feats in SETS.items():
    Xg = d[feats].apply(pd.to_numeric, errors="coerce")
    Xg = StandardScaler().fit_transform(Xg.fillna(Xg.median()))
    print(f"### {name}   features: {feats}")
    cc = pd.DataFrame(Xg, columns=feats).corr()
    hi = [(a,b,cc.loc[a,b]) for i,a in enumerate(feats) for b in feats[i+1:]
          if abs(cc.loc[a,b]) > .6]
    if hi:
        print("    collinear pairs: " +
              ", ".join(f"{a}~{b} {v:+.2f}" for a,b,v in hi))
    print(f"    {'split':<32}{'train AUC':>11}{'TEST AUC':>11}{'med rank':>12}")
    ms = {}
    for tp, sp, lab in (("Malik Beasley","Jontay Porter","A  Beasley -> Porter"),
                        ("Jontay Porter","Malik Beasley","B  Porter  -> Beasley"),
                        (None, None,                     "C  random 80/20")):
        ms[lab] = run(tp, sp, "    " + lab, Xs=Xg)
    print(f"    coefficients:")
    print(f"      {'feature':<14}" + "".join(f"{k.split()[0]:>10}" for k in ms))
    for i, f_ in enumerate(feats):
        print(f"      {f_:<14}" + "".join(f"{m.coef_[0][i]:>10.3f}" for m in ms.values()))
    print()
