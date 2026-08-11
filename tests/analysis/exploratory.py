"""Exploratory analysis for the write-up: bootstrapped feature tests + the real
logistic regressions. Nothing here is transcribed; every number is recomputed.

    python analysis/exploratory.py
"""
import warnings, pathlib, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, db
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
INK, HOT, DIM, OK = "#051c2c", "#d8383a", "#8c9bab", "#1a7f5a"
rng = np.random.default_rng(0)
N_BOOT, N_PERM = 2000, 2000

with db.connect() as c:
    d = pd.read_sql("""select s.player_id,s.game_id,s.player,s.game_date::text gd,
        s.points,s.close_line,s.under_hit,s.in_shortlist,
        z.game_z,z.effort_z,z.shortfall_z,z.game_z_tier,z.effort_z_tier,
        z.mk_p_price,z.mk_p_line,z.mk_line_mv,z.mk_price_mv,
        z.performance,z.market,z.motive,z.score
        from player_game_scores s join player_game_z z using(player_id,game_id)""", c)
for c_ in d.columns:
    if c_ not in ("player","gd","game_id"): d[c_] = pd.to_numeric(d[c_], errors="coerce")
d["y_under"] = d.under_hit.astype(int)
print(f"{len(d):,} propped games\n")

# ------------------------------------------------------------------ PART 1
# PRE-GAME features can legitimately be tested against the outcome: they are known
# before tip and the outcome is not. POST-GAME features are mechanically entangled
# with under_hit (shortfall = 1 - pts/line is POSITIVE exactly when the under hits),
# so an AUC against it is circular and is reported separately, flagged.
PREGAME = {"mk_p_price":"closing under price", "mk_p_line":"closing line level",
           "mk_line_mv":"line movement", "mk_price_mv":"price-only movement",
           "motive":"motive (1 - salary pct)", "market":"MARKET block"}
POSTGAME = {"game_z":"game_z (negated)", "effort_z":"effort_z (negated)",
            "game_z_tier":"game_z_tier (negated)", "effort_z_tier":"effort_z_tier (negated)",
            "shortfall_z":"shortfall_z", "performance":"PERFORMANCE block"}
NEG = {"game_z","effort_z","game_z_tier","effort_z_tier"}

def boot_auc(x, y, n=N_BOOT):
    m = ~(pd.isna(x) | pd.isna(y)); x, y = np.asarray(x[m]), np.asarray(y[m])
    obs = roc_auc_score(y, x)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    bs = np.array([roc_auc_score(y[i], x[i]) if len(np.unique(y[i]))>1 else np.nan
                   for i in idx])
    perm = np.array([roc_auc_score(rng.permutation(y), x) for _ in range(N_PERM)])
    p = (np.abs(perm-.5) >= abs(obs-.5)).mean()
    return obs, np.nanpercentile(bs,2.5), np.nanpercentile(bs,97.5), p, int(m.sum()), bs

rows, boots = [], {}
for grp, feats in (("pre-game", PREGAME), ("post-game (circular)", POSTGAME)):
    for k, lab in feats.items():
        x = -d[k] if k in NEG else d[k]
        obs, lo, hi, p, n, bs = boot_auc(x, d.y_under)
        boots[lab] = bs
        rows.append(dict(group=grp, feature=lab, n=n, auc=round(obs,4),
                         ci_lo=round(lo,4), ci_hi=round(hi,4),
                         p_perm=("<0.0005" if p==0 else f"{p:.4f}"),
                         sig="yes" if (lo>.5 or hi<.5) else "no"))
T1 = pd.DataFrame(rows)
print("=== [1] BOOTSTRAPPED AUC vs under_hit  (2,000 resamples, 2,000 permutations) ===")
print(T1.to_string(index=False))
T1.to_csv(OUT/"t1_feature_auc.csv", index=False)

fig, ax = plt.subplots(figsize=(7.4,4.4))
sub = T1[T1.group=="pre-game"].iloc[::-1]
yp = np.arange(len(sub))
ax.errorbar(sub.auc, yp, xerr=[sub.auc-sub.ci_lo, sub.ci_hi-sub.auc], fmt="o",
            color=INK, ecolor=DIM, capsize=3, ms=6)
for i,(_,r) in enumerate(sub.iterrows()):
    ax.scatter(r.auc, i, color=OK if r.sig=="yes" else HOT, zorder=5, s=42)
ax.axvline(.5, color=HOT, ls="--", lw=1.2)
ax.set_yticks(yp, sub.feature, fontsize=8.5)
ax.set_xlabel("AUC predicting whether the under hit  (0.5 = no information)")
ax.set_title("Pre-game features vs outcome, bootstrapped 95% CI\n"
             "green = CI excludes 0.5, red = indistinguishable from chance",
             loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"f10_feature_auc.png", bbox_inches="tight"); plt.close(fig)
print("\n  wrote figures/f10_feature_auc.png")

# ------------------------------------------------------------------ PART 2
# THE LOGISTIC REGRESSIONS, RUN FOR REAL.
#
# Two different questions, and only one of them is answerable:
#   (a) fit the three blocks to the LABELS -- 7 positives from 2 players. Leave-one-
#       PLAYER-out is the only honest split; a random split leaks player identity
#       through every player-level feature.
#   (b) fit the pre-game features to the OUTCOME -- 15,498 rows, no label problem.
print("\n\n=== [2a] LOGISTIC ON THE LABELS (what would 'derive' the block weights) ===")
FLAG = {"Malik Beasley": ["2023-11-11","2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
        "Jontay Porter": ["2024-01-20","2024-03-20"]}
d["y_lab"] = [int(r.gd in FLAG.get(r.player, [])) for _, r in d.iterrows()]
BL = ["performance","market","motive"]
L = d.dropna(subset=BL).copy()
print(f"positives {int(L.y_lab.sum())} of {len(L):,}  "
      f"(base rate 1 in {len(L)//max(1,int(L.y_lab.sum())):,})")

def fit_report(train, test, cols, tag):
    sc = StandardScaler().fit(train[cols])
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(sc.transform(train[cols]), train.y_lab)
    co = pd.Series(m.coef_[0], index=cols)
    auc_tr = roc_auc_score(train.y_lab, m.predict_proba(sc.transform(train[cols]))[:,1])
    auc_te = (roc_auc_score(test.y_lab, m.predict_proba(sc.transform(test[cols]))[:,1])
              if test.y_lab.nunique() > 1 else np.nan)
    imp = (co.abs()/co.abs().sum())
    return dict(fold=tag, **{f"coef_{c}": round(co[c],3) for c in cols},
                **{f"w_{c}": f"{imp[c]:.0%}" for c in cols},
                auc_train=round(auc_tr,3),
                auc_test=("n/a" if np.isnan(auc_te) else round(auc_te,3)))

folds = []
for held in ("Malik Beasley","Jontay Porter"):
    tr, te = L[L.player != held], L[L.player == held]
    folds.append(fit_report(tr, te, BL, f"hold out {held}"))
folds.append(fit_report(L, L, BL, "all data (in-sample)"))
F = pd.DataFrame(folds)
print(F.to_string(index=False))
c1 = np.array([F.iloc[0][f"coef_{c}"] for c in BL]); c2 = np.array([F.iloc[1][f"coef_{c}"] for c in BL])
print(f"\ncoefficient correlation between the two folds: r = {np.corrcoef(c1,c2)[0,1]:+.3f}")
print(f"implied weights, fold 1: {[F.iloc[0][f'w_{c}'] for c in BL]}")
print(f"implied weights, fold 2: {[F.iloc[1][f'w_{c}'] for c in BL]}")
print(f"SHIPPED prior          : ['45%', '30%', '25%']")
F.to_csv(OUT/"t2_logistic_labels.csv", index=False)

print("\n=== [2b] LOGISTIC ON THE OUTCOME, PRE-GAME FEATURES ONLY (15,498 rows) ===")
PG = ["mk_p_price","mk_p_line","mk_line_mv","mk_price_mv","motive"]
G = d.dropna(subset=PG).copy()
sc = StandardScaler().fit(G[PG])
m = LogisticRegression(max_iter=2000).fit(sc.transform(G[PG]), G.y_under)
co = pd.Series(m.coef_[0], index=PG)
# bootstrap the coefficients
bs = []
for _ in range(500):
    i = rng.integers(0, len(G), len(G))
    mm = LogisticRegression(max_iter=2000).fit(sc.transform(G.iloc[i][PG]), G.iloc[i].y_under)
    bs.append(mm.coef_[0])
bs = np.array(bs)
T2 = pd.DataFrame({"feature":PG, "coef":co.round(4).values,
    "ci_lo":np.percentile(bs,2.5,axis=0).round(4), "ci_hi":np.percentile(bs,97.5,axis=0).round(4),
    "crosses_0":["yes" if a<0<b or b<0<a else "NO" for a,b in
                 zip(np.percentile(bs,2.5,axis=0), np.percentile(bs,97.5,axis=0))]})
print(T2.to_string(index=False))
print(f"\nin-sample AUC on the outcome: "
      f"{roc_auc_score(G.y_under, m.predict_proba(sc.transform(G[PG]))[:,1]):.4f}   "
      f"(n={len(G):,}, complete rows only)")
T2.to_csv(OUT/"t2b_logistic_outcome.csv", index=False)

# ------------------------------------------------------------------ PART 3
# EVERY VARIANT TRIED, RANKED. Two columns because they answer different questions:
#   gmean_rank  -- where the labels sit in THAT variant's shortlist
#   pct_of_pool -- the same thing scale-free. A variant that cuts harder raises every
#                  rank for free, so raw rank alone rewards shrinking the pool.
print("\n\n=== [3] GEOMETRIC-MEAN RANK OF THE 7 LABELS, EVERY VARIANT ===")
with db.connect() as c:
    F = pd.read_sql("""select s.player_id,s.game_id,s.player,s.game_date::text gd,
        s.salary,s.has_listed_salary,s.tier,s.close_line,s.shortfall,s.minutes,
        z.game_z,z.effort_z,z.shortfall_z,z.game_z_tier,z.effort_z_tier,
        z.performance,z.market,z.motive,
        f.line_move_pct,f.price_only_move, p.experience
        from player_game_scores s join player_game_z z using(player_id,game_id)
        join player_game_features f using(player_id,game_id)
        join players p using(player_id)""", c)
for c_ in F.columns:
    if c_ not in ("player","gd","game_id","tier"): F[c_] = pd.to_numeric(F[c_], errors="coerce")
F["y"] = [int(r.gd in FLAG.get(r.player, [])) for _, r in F.iterrows()]

def funnel(dd, n_cuts=7):
    m = pd.Series(True, index=dd.index)
    if n_cuts >= 1: m &= ~(dd.game_z   >= dd.game_z.quantile(.75))
    if n_cuts >= 2: m &= ~(dd.effort_z >= dd.effort_z.quantile(.75))
    if n_cuts >= 3: m &= ~(dd.market   <= dd.market.quantile(.25))
    if n_cuts >= 4: m &= ~(dd.line_move_pct > 0)
    if n_cuts >= 5: m &= ~(dd.price_only_move > 0)
    if n_cuts >= 6: m &= dd.salary.isna() | (dd.salary <= 20e6)
    if n_cuts >= 7: m &= ~(dd.experience <= 2)
    return m

def perf_from(dd, pw):
    parts = {"game_z":-dd.game_z, "effort_z":-dd.effort_z, "shortfall_z":dd.shortfall_z,
             "game_z_tier":-dd.game_z_tier, "effort_z_tier":-dd.effort_z_tier}
    num = sum(pw.get(k,0)*v.fillna(0) for k,v in parts.items())
    den = sum(pw.get(k,0)*v.notna().astype(float) for k,v in parts.items())
    return num/den.replace(0, np.nan)

TH = {"game_z":1/3,"effort_z":1/3,"shortfall_z":1/3}
TIER5 = {k:1/5 for k in ("game_z","effort_z","shortfall_z","game_z_tier","effort_z_tier")}
TIERONLY = {"game_z_tier":1/2,"effort_z_tier":1/2}

def evaluate(name, bw, mask, perf=None):
    dd = F.copy()
    p = F.performance if perf is None else perf_from(F, perf)
    sc = bw[0]*p + bw[1]*F.market + bw[2]*F.motive
    sub = dd[mask & sc.notna()].copy(); sub["sc"] = sc[mask & sc.notna()]
    sub["rk"] = sub.sc.rank(ascending=False, method="first")
    lab = sub[sub.y == 1]
    if not len(lab): return None
    return dict(variant=name, pool=len(sub), labels_in=len(lab),
                gmean_rank=round(float(np.exp(np.log(lab.rk).mean())),1),
                pct_of_pool=float(np.exp(np.log(lab.rk/len(sub)).mean())),
                best=int(lab.rk.min()), worst=int(lab.rk.max()))

SHIP = (.45,.30,.25); ALLROWS = pd.Series(True, index=F.index)
V = [
 evaluate("FINAL — 7 cuts, 45/30/25",            SHIP, funnel(F,7)),
 evaluate("6 cuts (before experience cut)",       SHIP, funnel(F,6)),
 evaluate("no cuts at all (all 15,498)",          SHIP, ALLROWS),
 evaluate("7 cuts, equal blocks 33/33/33",       (1/3,1/3,1/3), funnel(F,7)),
 evaluate("7 cuts, grid optimum 25/2.5/72.5",    (.25,.025,.725), funnel(F,7)),
 evaluate("7 cuts, market deleted 64/0/36",      (.643,0,.357), funnel(F,7)),
 evaluate("7 cuts, logistic-fitted 7/6/87",      (.07,.06,.87), funnel(F,7)),
 evaluate("7 cuts, performance only",            (1,0,0), funnel(F,7)),
 evaluate("7 cuts, market only",                 (0,1,0), funnel(F,7)),
 evaluate("7 cuts, motive only",                 (0,0,1), funnel(F,7)),
 evaluate("7 cuts, tier blend (5 perf comps)",   SHIP, funnel(F,7), TIER5),
 evaluate("7 cuts, TIER baseline only",          SHIP, funnel(F,7), TIERONLY),
 evaluate("7 cuts + line >= 7.5",                SHIP, funnel(F,7) & (F.close_line>=7.5)),
 evaluate("7 cuts + minutes >= 15",              SHIP, funnel(F,7) & (F.minutes>=15)),
 evaluate("7 cuts + shortfall >= 0.45",          SHIP, funnel(F,7) & (F.shortfall>=.45)),
 evaluate("7 cuts + tier != bench",              SHIP, funnel(F,7) & (F.tier!="bench")),
 evaluate("old half-plane gates (gz<0 & ez<0 & mkt>0)", SHIP,
          (F.game_z<0)&(F.effort_z<0)&(F.market>0)),
]
T3 = pd.DataFrame([v for v in V if v])
T3["pct_of_pool"] = (T3.pct_of_pool*100).round(2)
T3 = T3.sort_values("pct_of_pool")
print(T3.to_string(index=False))
T3.to_csv(OUT/"t3_variant_ranking.csv", index=False)
print(f"\n(7 labels total; `labels_in` is how many survive that variant's cuts.)")
print(f"\nwrote {OUT}/t1_feature_auc.csv, t2_logistic_labels.csv, "
      f"t2b_logistic_outcome.csv, t3_variant_ranking.csv")

# Presented with FINAL pinned as the REFERENCE row and a delta column, rather than
# sorted so it lands on top. On the scale-free measure it places 11th of 17; sorting
# to hide that would be the same error as fitting the weights to the labels.
ref = T3[T3.variant.str.startswith("FINAL")].iloc[0]
T3b = T3.copy()
T3b["vs_final"] = ((T3b.pct_of_pool/ref.pct_of_pool - 1)*100).round(0).astype(int)
T3b["vs_final"] = T3b.vs_final.map(lambda v: "— reference —" if v == 0 else f"{v:+d}%")
T3b = pd.concat([T3b[T3b.variant.str.startswith("FINAL")],
                 T3b[~T3b.variant.str.startswith("FINAL")]])
print("\n\n=== [3] AS PRESENTED: final methodology as the reference row ===")
print(T3b[["variant","pool","labels_in","gmean_rank","pct_of_pool","vs_final",
           "best","worst"]].to_string(index=False))
T3b.to_csv(OUT/"t3_variant_ranking.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2,5.6))
t = T3b.iloc[::-1]
cols = [HOT if v.startswith("FINAL") else (OK if p < ref.pct_of_pool else DIM)
        for v, p in zip(t.variant, t.pct_of_pool)]
ax.barh(np.arange(len(t)), t.pct_of_pool, color=cols)
ax.axvline(ref.pct_of_pool, color=HOT, ls="--", lw=1.2)
for i, (p, n) in enumerate(zip(t.pct_of_pool, t.labels_in)):
    ax.text(p+.15, i, f"{p:.2f}%  ({n}/7)", va="center", fontsize=7.5)
ax.set_yticks(np.arange(len(t)), t.variant, fontsize=7.5)
ax.set_xlabel("geometric-mean position of the labelled games, as % of that variant's pool"
              "   (lower = better)")
ax.set_title("Everything tried, on a scale-free measure\n"
             "red = shipped methodology; green = beat it; grey = did not",
             loc="left", fontsize=10)
ax.set_xlim(0, min(12, t.pct_of_pool.max()*1.15))
fig.tight_layout(); fig.savefig(OUT/"f11_variant_ranking.png", bbox_inches="tight"); plt.close(fig)
print("\n  wrote figures/f11_variant_ranking.png")
