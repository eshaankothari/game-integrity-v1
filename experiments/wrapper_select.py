"""Wrapper feature selection: generate subset -> fit -> score -> iterate.

Run TWICE against two different targets, because the choice of target is the whole
question and the two give incompatible answers.

  A. target = the 6 flagged games.  This is what we actually want to detect and it is
     hopeless at this sample size -- forward selection with ~15 candidates and 6
     positive events will find a subset that separates them perfectly, and that subset
     is an artefact of which 6 games were named. The leave-one-out block below shows
     the instability directly rather than asserting it.

  B. target = under_hit, MARKET FEATURES ONLY.  A legitimate question with 15,498
     labels -- which market signals predict the under cashing. Performance features
     are excluded because shortfall = 1 - points/line and under_hit = points < line
     are the same fact; including them would let the wrapper "discover" a tautology.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path(__file__).resolve().parent / "out"
d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str})
d["under_hit"] = (d.points < d.close_line).astype(int)

PERF = ["game_z", "effort_z", "shortfall_z"]
MKT  = ["mk_p_price", "mk_p_line", "mk_line_mv", "mk_price_mv"]
CTX  = ["minutes", "close_line", "under_price", "salary", "n_games"]
ALL  = PERF + MKT + CTX

FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
d["flagged"] = d.apply(lambda r: int(r.gd in FLAG.get(r.player, [])), axis=1)

def prep(cols):
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return StandardScaler().fit_transform(X)

def forward(cols, y, k=6, C=1.0, seed=0):
    """Greedy forward selection, scored by 5-fold CV AUC."""
    chosen, rest, hist = [], list(cols), []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    while rest and len(chosen) < k:
        best = None
        for c in rest:
            X = prep(chosen + [c])
            s = cross_val_score(
                LogisticRegression(C=C, class_weight="balanced", max_iter=4000),
                X, y, cv=cv, scoring="roc_auc").mean()
            if best is None or s > best[0]:
                best = (s, c)
        chosen.append(best[1]); rest.remove(best[1]); hist.append((best[1], best[0]))
    return chosen, hist

print("="*72)
print("A.  TARGET = the 6 flagged games")
print("="*72)
y = d.flagged.values
print(f"positives {y.sum()} of {len(y):,}   candidate features {len(ALL)}\n")
ch, hist = forward(ALL, y, k=5)
for f_, s_ in hist:
    print(f"   + {f_:<14} cv AUC {s_:.4f}")
print(f"\n   selected: {ch}")

print("\n   LEAVE-ONE-OUT: hide one flagged game, re-run selection from scratch.")
print("   If the choice is real, the same features keep getting picked.\n")
sel = []
for i in np.flatnonzero(y):
    y2 = y.copy(); y2[i] = 0
    c2, _ = forward(ALL, y2, k=3)
    sel.append(c2)
    r = d.iloc[i]
    print(f"     hide {r.player.split()[1][:3]} {r.gd[5:]} -> {c2}")
from collections import Counter
cnt = Counter(f_ for s_ in sel for f_ in s_)
print(f"\n   feature chosen in how many of the {len(sel)} folds:")
for f_, n_ in cnt.most_common():
    print(f"      {f_:<14} {n_}/{len(sel)}")

print("\n" + "="*72)
print("B.  TARGET = under_hit,  MARKET FEATURES ONLY")
print("="*72)
y2 = d.under_hit.values
print(f"positives {y2.sum():,} of {len(y2):,}   candidate features {len(MKT+CTX)}\n")
ch2, hist2 = forward(MKT + CTX, y2, k=5)
for f_, s_ in hist2:
    print(f"   + {f_:<14} cv AUC {s_:.4f}")
print(f"\n   selected: {ch2}")
X = prep(ch2)
m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000).fit(X, y2)
print("\n   coefficients (standardised):")
for c_, co in sorted(zip(ch2, m.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"      {c_:<14} {co:+.4f}")
