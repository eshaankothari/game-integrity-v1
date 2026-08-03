"""Logistic regression on every feature, with a held-out test set.

SPLIT. 6 positives, so 4 train / 2 test is the only split that leaves anything in
either half. Stratified on the label and seeded, and the two held-out positives are
chosen to span BOTH players -- otherwise a Beasley-only test set measures whether the
model recognises Beasley.

REGULARISATION. Strong L2 (small C). With 12 features and 4 training positives the
unpenalised fit is in the complete-separation regime: it can reproduce the labels
exactly with coefficients running to infinity. Shrinkage keeps them finite, so what
survives is the direction the data leans rather than a fitted magnitude.

WHAT TO READ. The test AUC and where the two held-out positives rank among ~15,000
negatives. The coefficients are reported for inspection but are not identified at this
sample size -- forward selection on the same labels chose eight different feature sets
across six leave-one-out folds.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

OUT = pathlib.Path(__file__).resolve().parent / "out"
d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str}).reset_index(drop=True)

FEATURES = ["game_z", "effort_z", "shortfall_z",
            "mk_p_price", "mk_p_line", "mk_line_mv", "mk_price_mv",
            "minutes", "close_line", "under_price", "salary", "n_games"]

# POSITIVES = every game by either player where the under hit.
#
# This is a much stronger hypothesis than the 6 named games: it asserts both players
# were compromised broadly rather than on specific dates. Worth being explicit about
# what a model can learn from it -- the negatives are every OTHER player's under-hit
# game, so the only thing separating the classes is WHICH PLAYER, and a classifier
# will find that before it finds anything about behaviour.
SUSPECT = ["Malik Beasley", "Jontay Porter"]
d["under_hit"] = d.points < d.close_line
d["y"] = (d.player.isin(SUSPECT) & d.under_hit).astype(int)

X = d[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)

# LEAVE-ONE-PLAYER-OUT. Train on Beasley's positives, test on Porter's.
#
# This is the only split that answers the question that matters: does the model learn
# something about the BEHAVIOUR that transfers to a player it never saw, or does it
# learn to recognise Malik Beasley? A random split cannot tell those apart, because
# both halves would contain the same two players.
pos = np.flatnonzero(d.y.values)
train_pos = [i for i in pos if d.player[i] == "Malik Beasley"]
test_pos  = [i for i in pos if d.player[i] == "Jontay Porter"]

rng = np.random.default_rng(0)
neg = np.flatnonzero(d.y.values == 0)
test_neg = rng.choice(neg, size=len(neg)//5, replace=False)
train_neg = np.setdiff1d(neg, test_neg)
tr = np.concatenate([train_pos, train_neg])
te = np.concatenate([test_pos, test_neg])

print(f"train: {len(tr):,} rows, {len(train_pos)} positives")
print(f"test : {len(te):,} rows, {len(test_pos)} positives")
print(f"  train positives: Malik Beasley, {len(train_pos)} under-hit games")
print(f"  test  positives: Jontay Porter, {len(test_pos)} under-hit games")

for C in (0.01, 0.05, 0.2, 1.0):
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=6000)
    m.fit(Xs[tr], d.y.values[tr])
    ptr, pte = m.predict_proba(Xs[tr])[:,1], m.predict_proba(Xs[te])[:,1]
    atr = roc_auc_score(d.y.values[tr], ptr)
    ate = roc_auc_score(d.y.values[te], pte)
    ranks = sorted(int((pte > pte[list(te).index(i)]).sum()) + 1 for i in test_pos)
    print(f"\nC={C:<5}  train AUC {atr:.4f}   TEST AUC {ate:.4f}   "
          f"Porter ranks {ranks} of {len(te):,}")

C = 0.05
m = LogisticRegression(C=C, class_weight="balanced", max_iter=6000)
m.fit(Xs[tr], d.y.values[tr])
print(f"\nCOEFFICIENTS at C={C} (standardised, direction only):")
for f_, co in sorted(zip(FEATURES, m.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"   {f_:<14} {co:+.4f}  {'#'*int(min(abs(co)*25, 40))}")

d["p_fit"] = m.predict_proba(Xs)[:,1]
d["rank_fit"] = d.p_fit.rank(ascending=False, method="min").astype(int)
d["rank_score"] = d.score.rank(ascending=False, method="min")
print(f"\nPOSITIVES BY PLAYER, fitted model vs the unsupervised score:")
f = d[d.y == 1].copy()
f["split"] = np.where(f.player == "Jontay Porter", "TEST", "train")
print(f.groupby(["player","split"]).agg(
    n=("rank_fit","size"), median_rank_fit=("rank_fit","median"),
    best_rank_fit=("rank_fit","min"),
    median_rank_score=("rank_score","median"),
    best_rank_score=("rank_score","min")).to_string())
print(f"\n   for reference, a random row would sit at rank {len(d)//2:,}")
d.sort_values("p_fit", ascending=False).to_csv(OUT / "fit_all.csv", index=False)
print(f"\n-> {OUT/'fit_all.csv'}")
