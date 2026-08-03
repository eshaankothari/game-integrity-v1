"""Isolation Forest on the three weighted blocks, as an alternative to summing them.

WHY THIS MIGHT HELP. The linear score asks "how far along one direction is this row",
so a game must be bad on the weighted AVERAGE to rank. Isolation Forest asks "how
easy is this row to separate from everything else", which can flag a row that is
unremarkable on the average but sits in a sparse CORNER -- extreme on two blocks and
ordinary on the third, say. Those are exactly the rows a weighted sum buries.

WEIGHTS ARE APPLIED BEFORE FITTING, not after. IF splits at a uniformly random point
within each feature's observed range, so a feature with a wider range gets split more
often and therefore counts for more. Multiplying by 0.45 / 0.30 / 0.25 shrinks the
ranges in proportion, which is how a weight expresses itself in this model.

DIRECTION IS HANDLED BY FITTING ON THE BAD HALF ONLY. IF measures distance from the
bulk in ANY direction, so a career game is exactly as isolated as a no-show. Fitted on
everything it spends half its budget on the wrong tail -- measured earlier at 390
under / 385 over. Restricting the FIT (not just the output) also changes what "the
bulk" means: the comparison becomes other bad games rather than all games.
"""
import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sklearn.ensemble import IsolationForest

OUT = pathlib.Path(__file__).resolve().parent / "out"
d = pd.read_csv(OUT / "two_metrics.csv", dtype={"game_id": str}).reset_index(drop=True)

# Block weights read out of two_metrics.py rather than copied, so this cannot silently
# disagree with the linear score it is being compared against. The WITHIN-block weights
# are already baked into the performance/market columns of the CSV -- equal since the
# weight audit -- so only the three block weights are applied here.
from weight_audit import live_weights                                   # noqa: E402
BLOCK_W = live_weights()[2]

# ---- MOTIVE IS EXCLUDED FROM THE FOREST -------------------------------------
# Isolation Forest scores RARITY, and low salary is not rare among games that fail
# badly against a prop line -- it is the norm. So a high motive value sits in the
# crowded middle of the bad-games cloud and adds almost nothing to isolation, while
# a WELL-PAID player having such a night is structurally unusual and gets promoted.
# Measured on the 3-block run: top-200 median salary $3.3M by isolation against $2.6M
# by the linear score, with Nicolas Batum at $11.7M ranked 1st.
#
# That is a coherent measurement, but it is the exact inverse of what motive asserts.
# The linear score says cheap is more suspicious; the forest says cheap is more
# ordinary. Both cannot drive one ranking, so motive is left out of the FEATURES here
# and applied only afterwards, as a filter or a tie-break, where its meaning is a prior
# about who could be approached rather than a claim about what is rare.
USE = ["performance", "market"]
W = {k: v / sum(BLOCK_W[j] for j in USE) for k, v in BLOCK_W.items() if k in USE}
print("forest features: " + "  ".join(f"{k} {v:.2f}" for k, v in W.items())
      + f"   (motive excluded, block weight {BLOCK_W['motive']} reallocated)")

d = d.dropna(subset=list(W)).reset_index(drop=True)

# ---- ONE-SIDED CLIP ---------------------------------------------------------
# Every block is clipped at 0 from below, so anything on the GOOD side collapses to
# exactly 0 before the forest ever sees it.
#
# The previous run is why. Isolation Forest measures distance from the bulk in ANY
# direction, and the sparsest corners of this three-space turned out to be the great
# games by expensive players -- Luka Doncic scoring 73 ranked 20th, Embiid 70 ranked
# 24th, and Stephen Curry occupied four of the top nine. Rare is not the same as bad.
#
# Filtering the FIT to a "suspicious half" did not fix it, because motive is constant
# per player: a $51.9M player carries motive -1.72 on every game he plays, so his rows
# sit in a permanently sparse region whatever happened on the floor. Clipping removes
# the sparse GOOD corner entirely -- a strong game now reads (0, 0, 0), which is the
# single most crowded point in the space and therefore the least isolated possible.
#
# The cost: a large tie mass at 0 on each axis, so the forest cannot rank within the
# good half at all. That is intended -- there is nothing there to rank.
for k in W:
    d[f"{k}_pos"] = d[k].clip(lower=0).round(3)
X = pd.DataFrame({k: d[f"{k}_pos"] * w for k, w in W.items()})
print(f"rows {len(d):,}   clipped to 0 on: " +
      ", ".join(f"{k} {int((d[k] < 0).sum()):,}" for k in W))

m = IsolationForest(n_estimators=400, contamination=0.05, random_state=42, n_jobs=-1)
m.fit(X)
d["iso"] = -m.decision_function(X)          # higher = more isolated
d["iso_rank"] = d.iso.rank(ascending=False, method="min").astype(int)
# Int64, not int: dropna() above now guards only performance and market, so a handful
# of rows have no motive and therefore no production `score`. They are still valid rows
# for the forest -- they just have no linear rank to compare against.
d["lin_rank"] = d.score.rank(ascending=False, method="min").astype("Int64")

# Motive-free linear score, so the forest is compared against a like-for-like baseline
# rather than against a score that uses a feature the forest never saw.
d["score_nm"] = sum(W[k] * d[k] for k in USE).round(3)
d["lin_rank_nm"] = d.score_nm.rank(ascending=False, method="min").astype(int)
print(f"corr(iso, motive-free linear) = {d.iso.corr(d.score_nm):+.3f}   "
      f"spearman {d.iso.corr(d.score_nm, method='spearman'):+.3f}")

print(f"\ncorr(iso, linear score) = {d.iso.corr(d.score):+.3f}   "
      f"spearman {d.iso.corr(d.score, method='spearman'):+.3f}")
top = lambda c, k=200: set(d.nsmallest(k, c).index)
print(f"top-200 overlap: {len(top('iso_rank') & top('lin_rank'))} of 200")

print(f"\nWHAT IF IS SEPARATING ON (mean |value| among the top 200 vs the rest):")
t = d.nsmallest(200, "iso_rank")
for k in W:
    print(f"   {k:<12} top200 {t[k+'_pos'].mean():+.3f}   "
          f"rest {d[~d.index.isin(t.index)][k+'_pos'].mean():+.3f}")

FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
f = d[d.apply(lambda r: r.gd in FLAG.get(r.player, []), axis=1)]
print(f"\nFLAGGED -- isolation vs linear:")
print(f.sort_values("iso_rank")[["player","gd","performance","market","motive",
      "iso_rank","lin_rank_nm","lin_rank"]].to_string(index=False))

print(f"\nTOP 15 BY ISOLATION:")
c = ["iso_rank","lin_rank_nm","lin_rank","player","gd","minutes","points",
     "close_line","performance","market","motive","salary"]
y = d.nsmallest(15, "iso_rank")[c].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))

print(f"\nRANKED HIGH BY ISOLATION, LOW BY THE LINEAR SCORE (what IF adds):")
d["gap"] = d.lin_rank_nm - d.iso_rank
y = d.nlargest(8, "gap")[c].copy()
y["salary"] = y.salary.map(lambda v: "two-way" if pd.isna(v) else f"${v/1e6:.1f}M")
print(y.to_string(index=False))
d.sort_values("iso_rank").to_csv(OUT / "iso_three.csv", index=False)
print(f"\n-> {OUT/'iso_three.csv'}")
