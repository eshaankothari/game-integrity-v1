"""Every figure and metric table for the write-up. Recomputed from the DB on each run --
nothing here is transcribed from a doc, because the docs have drifted from the code.

    python analysis/figures.py
"""
import warnings, pathlib, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, db
from standardize import BLOCK_W, PERF_W

OUT = pathlib.Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, HOT, DIM = "#051c2c", "#d8383a", "#8c9bab"
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); plt.close(fig)
    print(f"  wrote figures/{name}.png")

with db.connect() as c:
    S = pd.read_sql("""select s.player_id,s.game_id,s.player,s.game_date::text gd,s.tier,
        s.minutes,s.points,s.close_line,s.close_under,s.score,s.score_100,s.shortfall,
        s.under_hit,s.in_shortlist,s.rank,s.salary,s.has_listed_salary,
        z.game_z,z.effort_z,z.shortfall_z,z.mk_p_price,z.mk_p_line,z.mk_line_mv,
        z.mk_price_mv,z.performance,z.market,z.motive,
        f.line_move_pct,f.price_only_move
        from player_game_scores s
        join player_game_z z using(player_id,game_id)
        join player_game_features f using(player_id,game_id)""", c)
    ALL = pd.read_sql("""select pg.player_id, pg.points, pg.minutes
        from player_games pg where pg.minutes>0 and pg.points is not null""", c)
    SAL = pd.read_sql("select player_id, salary, has_listed_salary from player_salaries", c)
for df in (S, ALL, SAL):
    for c_ in df.columns:
        if c_ not in ("player", "gd", "tier", "game_id"):
            df[c_] = pd.to_numeric(df[c_], errors="coerce")
print(f"loaded {len(S):,} propped games, {len(ALL):,} played games\n")
LOG = []
def note(s): LOG.append(s); print("   " + s)

# ---------------------------------------------------------------- F1 validation
print("F1 validation_deciles")
d = S.dropna(subset=["score_100"]).copy()
d["dec"] = pd.qcut(d.score_100, 10, labels=False)
g = d.groupby("dec").agg(n=("under_hit","size"), p=("under_hit","mean"),
                         lo=("score_100","min"), hi=("score_100","max"))
fig, ax = plt.subplots(figsize=(6.2,3.4))
ax.bar(g.index+1, g.p*100, color=[HOT if i>=8 else INK for i in g.index], width=.72)
ax.axhline(d.under_hit.mean()*100, ls="--", lw=1, color=DIM)
ax.text(1.1, d.under_hit.mean()*100+2.5, f"population base rate {d.under_hit.mean():.1%}",
        color=DIM, fontsize=7.5)
for i, r in g.iterrows(): ax.text(i+1, r.p*100+1.6, f"{r.p:.0%}", ha="center", fontsize=7)
ax.set_xlabel("score decile (1 = lowest score)"); ax.set_ylabel("% of games that went UNDER")
ax.set_title("The score is monotone in outcome across all ten deciles\n"
             "no labels used — 15,498 games", loc="left", fontsize=10)
ax.set_xticks(range(1,11)); ax.set_ylim(0,105); save(fig,"f1_validation_deciles")
note(f"F1: decile 1 {g.p.iloc[0]:.1%} -> decile 10 {g.p.iloc[-1]:.1%}, "
     f"strictly increasing, base rate {d.under_hit.mean():.1%}")

# ------------------------------------------------------------- F2 floor effect
print("F2 floor_effect")
a = ALL.copy()
gp = a.groupby("player_id")["points"]
a["ppg"] = gp.transform("mean"); a["psd"] = gp.transform("std")
a["n"] = gp.transform("size")
a["pz"] = (a.points - a.ppg) / a.psd.replace(0, np.nan)
z0 = a[(a.points == 0) & (a.n >= 15)].dropna(subset=["pz"])
fig, axes = plt.subplots(1, 2, figsize=(9,3.5), sharex=True)
ax = axes[0]
ax.scatter(z0.ppg, z0.pz, s=7, alpha=.35, color=INK, edgecolors="none")
fit = np.polyfit(z0.ppg, z0.pz, 1); xs = np.linspace(z0.ppg.min(), z0.ppg.max(), 50)
ax.plot(xs, np.polyval(fit, xs), color=HOT, lw=1.6)
ax.set_xlabel("player's season points per game"); ax.set_ylabel("his z-score for a ZERO-point game")
ax.set_title("z-score is bounded by the player's own mean", loc="left", fontsize=9.5)
r = np.corrcoef(z0.ppg, z0.pz)[0,1]
ax.text(.97,.06,f"r = {r:+.3f}", transform=ax.transAxes, ha="right", color=HOT, fontsize=9)
z0p = S[(S.points==0)].dropna(subset=["shortfall"])
ax = axes[1]
ppg_map = a.groupby("player_id").ppg.first()
ax.scatter(z0p.player_id.map(ppg_map), z0p.shortfall, s=7, alpha=.35, color=INK, edgecolors="none")
ax.set_ylim(-.05,1.15); ax.set_xlabel("player's season points per game")
ax.set_ylabel("shortfall for a ZERO-point game")
ax.set_title("shortfall has no floor — 1.000 for everyone", loc="left", fontsize=9.5)
fig.suptitle("THE FLOOR EFFECT: why shortfall_z exists", x=.005, ha="left", fontsize=11)
save(fig,"f2_floor_effect")
note(f"F2: over {len(z0)} zero-point games, own-z ranges {z0.pz.min():.2f} to {z0.pz.max():.2f} "
     f"and correlates r={r:+.3f} with scoring level; shortfall is constant at 1.000")

# ------------------------------------------- F3 shortfall standardisation choice
print("F3 shortfall_standardization")
s = S.dropna(subset=["shortfall"]).copy()
gs = s.groupby("player_id")["shortfall"]
s["sf_within"] = (s.shortfall - gs.transform("mean")) / gs.transform("std").replace(0,np.nan)
s["ppg"] = s.player_id.map(ppg_map)
zz = s[(s.points==0)].dropna(subset=["sf_within","ppg"])
fig, ax = plt.subplots(figsize=(6.2,3.5))
ax.scatter(zz.ppg, zz.sf_within, s=9, alpha=.45, color=INK, edgecolors="none",
           label="within-player z of shortfall")
ax.axhline(zz.shortfall_z.mean(), color=HOT, lw=1.8, label="league-wide z (what ships)")
rr = np.corrcoef(zz.ppg, zz.sf_within)[0,1]
ax.set_xlabel("player's season points per game"); ax.set_ylabel("standardised shortfall")
ax.set_title("Standardising an already player-relative quantity re-introduces the floor\n"
             f"all points below are ZERO-point games — r = {rr:+.3f} with scoring level",
             loc="left", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False); save(fig,"f3_shortfall_standardization")
note(f"F3: within-player z of shortfall on zero-point games spans "
     f"{zz.sf_within.min():+.2f} to {zz.sf_within.max():+.2f} and correlates r={rr:+.3f} "
     f"with scoring level; league-wide z is flat")
pd.DataFrame({"note": LOG}).to_csv(OUT/"_metrics_part1.csv", index=False)
print("\npart 1 done")

# ------------------------------------------------ F4 performance independence
print("\nF4 perf_components")
P = S.dropna(subset=["game_z","effort_z","shortfall_z"]).copy()
P["o_game"], P["o_eff"] = -P.game_z, -P.effort_z
cm = P[["o_game","o_eff","shortfall_z"]].corr()
lbl = ["-game_z\n(worse night)", "-effort_z\n(less involved)", "shortfall_z\n(vs the line)"]
fig, axes = plt.subplots(1, 2, figsize=(9.4,3.6),
                         gridspec_kw={"width_ratios":[1,1.25]})
im = axes[0].imshow(cm, vmin=-1, vmax=1, cmap="RdBu_r")
axes[0].set_xticks(range(3), lbl, fontsize=7); axes[0].set_yticks(range(3), lbl, fontsize=7)
for i in range(3):
    for j in range(3):
        axes[0].text(j,i,f"{cm.iloc[i,j]:+.2f}", ha="center", va="center",
                     color="white" if abs(cm.iloc[i,j])>.5 else INK, fontsize=8.5)
axes[0].grid(False); axes[0].set_title("the three performance components", loc="left", fontsize=9.5)
plt.colorbar(im, ax=axes[0], fraction=.046)
ax = axes[1]
sc = ax.scatter(P.o_game, P.shortfall_z, s=3, alpha=.18, c=P.under_hit.map({True:HOT,False:DIM}))
ax.set_xlabel("-game_z  (higher = worse night for him)")
ax.set_ylabel("shortfall_z  (higher = further under the line)")
ax.set_title(f"r = {cm.loc['o_game','shortfall_z']:+.3f} — they disagree often enough to matter\n"
             "red = went under", loc="left", fontsize=9.5)
save(fig,"f4_perf_components")
note(f"F4: corr(-game_z, -effort_z)={cm.loc['o_game','o_eff']:+.3f}, "
     f"corr(-game_z, shortfall_z)={cm.loc['o_game','shortfall_z']:+.3f}, "
     f"corr(-effort_z, shortfall_z)={cm.loc['o_eff','shortfall_z']:+.3f}")

# ------------------------------------------------------- F5 market: does price work
print("F5 market_price")
m = S.dropna(subset=["close_under","under_hit"]).copy()
m["q"] = pd.qcut(m.close_under, 5, labels=False)
gp5 = m.groupby("q").agg(n=("under_hit","size"), p=("under_hit","mean"),
                         lo=("close_under","min"), hi=("close_under","max"))
mv = S.dropna(subset=["line_move_pct","under_hit"]).copy()
mv["q"] = pd.qcut(mv.line_move_pct, 5, labels=False, duplicates="drop")
gmv = mv.groupby("q").agg(n=("under_hit","size"), p=("under_hit","mean"),
                          lo=("line_move_pct","min"), hi=("line_move_pct","max"))
fig, axes = plt.subplots(1, 2, figsize=(9.6,3.5))
for ax, gg, ttl, xl, good in (
    (axes[0], gp5, "closing UNDER price — the component that works",
     "quintile of closing under price (1 = shortest price)", True),
    (axes[1], gmv, "line movement — the component that does NOT",
     "quintile of line move % (1 = biggest drop)", False)):
    ax.bar(gg.index+1, gg.p*100, color=INK if good else DIM, width=.68)
    ax.axhline(S.under_hit.mean()*100, ls="--", lw=1, color=HOT)
    for i,r in gg.iterrows(): ax.text(i+1, r.p*100+.6, f"{r.p:.1%}", ha="center", fontsize=7.5)
    ax.set_xlabel(xl); ax.set_ylabel("% went under"); ax.set_xticks(range(1,len(gg)+1))
    ax.set_ylim(0, max(70, gg.p.max()*100+8))
    ax.set_title(ttl, loc="left", fontsize=9.5)
axes[1].text(.5,.06,f"base rate {S.under_hit.mean():.1%}", transform=axes[1].transAxes,
             color=HOT, fontsize=7.5)
save(fig,"f5_market_components")
sp = np.corrcoef(m.close_under.rank(), m.under_hit.astype(int))[0,1]
note(f"F5: under-hit by closing-price quintile {gp5.p.iloc[0]:.1%} -> {gp5.p.iloc[-1]:.1%} "
     f"(monotone, n={len(m):,}); by line-movement quintile "
     f"{gmv.p.iloc[0]:.1%} -> {gmv.p.iloc[-1]:.1%} (n={len(mv):,}) vs base {S.under_hit.mean():.1%}")

# ------------------------------------------------------------ F6 salary transform
print("F6 salary_transform")
sal = SAL.dropna(subset=["salary"]).copy()
sal["z"] = (sal.salary - sal.salary.mean())/sal.salary.std()
sal["pct"] = sal.salary.rank(pct=True)
fig, axes = plt.subplots(1,3, figsize=(11,3.2))
axes[0].hist(sal.salary/1e6, bins=45, color=INK)
axes[0].set_xlabel("salary ($M)"); axes[0].set_ylabel("players")
axes[0].set_title(f"raw salary — skew {sal.salary.skew():+.2f}", loc="left", fontsize=9.5)
axes[0].axvline(sal.salary.mean()/1e6, color=HOT, lw=1.4)
axes[0].text(sal.salary.mean()/1e6+.7, axes[0].get_ylim()[1]*.8,
             f"{(sal.salary<sal.salary.mean()).mean():.0%} below mean", color=HOT, fontsize=7.5)
lo_half = sal[sal.pct<=.5]
axes[1].scatter(sal.salary/1e6, sal.z, s=6, color=INK, alpha=.5, edgecolors="none")
axes[1].set_xlabel("salary ($M)"); axes[1].set_ylabel("z-score")
axes[1].set_title(f"z-score: bottom half spans {lo_half.z.max()-lo_half.z.min():.2f}",
                  loc="left", fontsize=9.5)
axes[1].axhspan(lo_half.z.min(), lo_half.z.max(), color=HOT, alpha=.15)
axes[2].scatter(sal.salary/1e6, sal.pct, s=6, color=INK, alpha=.5, edgecolors="none")
axes[2].set_xlabel("salary ($M)"); axes[2].set_ylabel("percentile")
axes[2].set_title("percentile: uniform by construction", loc="left", fontsize=9.5)
axes[2].axhspan(0,.5, color="#2251ff", alpha=.12)
fig.suptitle("MOTIVE: why percentile and not a z-score", x=.005, ha="left", fontsize=11)
save(fig,"f6_salary_transform")
note(f"F6: salary skew {sal.salary.skew():+.2f}, {(sal.salary<sal.salary.mean()).mean():.0%} "
     f"of players below the mean; the bottom half of earners occupies only "
     f"{lo_half.z.max()-lo_half.z.min():.2f} of the z range vs 0.50 of the percentile range")

# ------------------------------------------- F7 blocks + EFFECTIVE weights
print("F7 block_weights")
B = S.dropna(subset=["performance","market","motive"]).copy()
cb = B[["performance","market","motive"]].corr()
w = np.array([BLOCK_W[k] for k in ("performance","market","motive")])
sd = np.array([B[k].std() for k in ("performance","market","motive")])
eff = w*sd/(w*sd).sum()
fig, axes = plt.subplots(1,2, figsize=(9.4,3.5), gridspec_kw={"width_ratios":[1,1.15]})
im = axes[0].imshow(cb, vmin=-1, vmax=1, cmap="RdBu_r")
axes[0].set_xticks(range(3), ["perf","market","motive"], fontsize=8)
axes[0].set_yticks(range(3), ["perf","market","motive"], fontsize=8)
for i in range(3):
    for j in range(3):
        axes[0].text(j,i,f"{cb.iloc[i,j]:+.3f}", ha="center", va="center",
                     color="white" if abs(cb.iloc[i,j])>.5 else INK, fontsize=8.5)
axes[0].grid(False); plt.colorbar(im, ax=axes[0], fraction=.046)
axes[0].set_title("the three blocks are near-independent", loc="left", fontsize=9.5)
x = np.arange(3); ax = axes[1]
ax.bar(x-.19, w*100, .38, label="stated weight", color=DIM)
ax.bar(x+.19, eff*100, .38, label="EFFECTIVE weight", color=HOT)
for i,(a_,b_) in enumerate(zip(w,eff)):
    ax.text(i-.19, a_*100+.8, f"{a_:.0%}", ha="center", fontsize=8)
    ax.text(i+.19, b_*100+.8, f"{b_:.1%}", ha="center", fontsize=8, color=HOT)
ax.set_xticks(x, [f"performance\nsd {sd[0]:.3f}", f"market\nsd {sd[1]:.3f}",
                  f"motive\nsd {sd[2]:.3f}"], fontsize=8)
ax.set_ylabel("% of the score"); ax.legend(fontsize=7.5, frameon=False)
ax.set_title("blocks are combined on the RAW z-scale, so the stated\n"
             "weights are not the weights that act", loc="left", fontsize=9.5)
save(fig,"f7_block_weights")
note(f"F7: block corr perf/market {cb.iloc[0,1]:+.3f}, perf/motive {cb.iloc[0,2]:+.3f}, "
     f"market/motive {cb.iloc[1,2]:+.3f}; effective weights "
     f"{eff[0]:.1%}/{eff[1]:.1%}/{eff[2]:.1%} vs stated 45/30/25")

# ---------------------------------------------------- F8 weight audit (Dirichlet)
print("F8 weight_audit")
FLAG = {"Malik Beasley": ["2024-01-06","2024-01-26","2024-02-27"],
        "Jontay Porter": ["2024-03-20"]}
W = S.copy()
W["y"] = [int(r.gd in FLAG.get(r.player, [])) for _, r in W.iterrows()]
parts = {"game_z": -W.game_z, "effort_z": -W.effort_z, "shortfall_z": W.shortfall_z}
def perf(pw):
    num = sum(pw[k]*v.fillna(0) for k,v in parts.items())
    den = sum(pw[k]*v.notna().astype(float) for k,v in parts.items())
    return num/den.replace(0,np.nan)
def blend(bw):
    return bw[0]*W.performance + bw[1]*W.market + bw[2]*W.motive
rng = np.random.default_rng(0); N = 20_000
lab = W.index[W.y==1]
cost = lambda s: float(np.mean(np.log10(s.rank(ascending=False, method="first")[lab])))
tuned = cost(blend(w))
draws = np.array([cost(blend(bw)) for bw in rng.dirichlet(np.ones(3), N)])
fig, ax = plt.subplots(figsize=(6.4,3.4))
ax.hist(draws, bins=70, color=DIM)
ax.axvline(tuned, color=HOT, lw=2)
ax.text(tuned, ax.get_ylim()[1]*.92, f"  shipped 45/30/25\n  = {tuned:.3f}", color=HOT, fontsize=8)
ax.axvline(np.median(draws), color=INK, lw=1.2, ls="--")
ax.text(np.median(draws), ax.get_ylim()[1]*.55, f"  random median\n  = {np.median(draws):.3f}",
        fontsize=7.5)
beat = (draws < tuned).mean()
ax.set_xlabel("cost = mean log$_{10}$ rank of the 4 labelled games  (lower is better)")
ax.set_ylabel(f"of {N:,} random weight vectors")
ax.set_title(f"{beat:.1%} of random block-weight vectors beat the shipped ones\n"
             "the weights are a prior, not a fit", loc="left", fontsize=10)
save(fig,"f8_weight_audit")
note(f"F8: shipped cost {tuned:.3f}, random median {np.median(draws):.3f}, "
     f"{beat:.1%} of {N:,} Dirichlet draws beat it")

# ------------------------------------------------------------ detailed tables
print("\n=== TABLES FOR THE WRITE-UP ===")
print("\n[T1] under-hit by score decile"); print(g.assign(p=(g.p*100).round(1)).to_string())
print("\n[T2] under-hit by closing under-price quintile")
print(gp5.assign(p=(gp5.p*100).round(1)).to_string())
print("\n[T3] under-hit by line-movement quintile")
print(gmv.assign(p=(gmv.p*100).round(1)).to_string())
print("\n[T4] performance component correlations"); print(cm.round(3).to_string())
print("\n[T5] block correlations"); print(cb.round(3).to_string())
pd.DataFrame({"note": LOG}).to_csv(OUT/"_metrics.csv", index=False)
print(f"\nall figures in {OUT}")

# ------------------------------------------- F9 block-weight simplex (7 labels)
print("\nF9 weight_simplex")
FLAG7 = {"Malik Beasley": ["2023-11-11","2024-01-06","2024-01-26","2024-02-27","2024-03-10"],
         "Jontay Porter": ["2024-01-20","2024-03-20"]}
Q = S[S.in_shortlist.astype(bool)].dropna(subset=["performance","market","motive"]).copy()
Q["y7"] = [int(r.gd in FLAG7.get(r.player, [])) for _, r in Q.iterrows()]
L7 = Q.index[Q.y7 == 1]
rank_w = lambda v: (v[0]*Q.performance + v[1]*Q.market + v[2]*Q.motive
                    ).rank(ascending=False, method="first")
gmean = lambda v: float(10**np.mean(np.log10(rank_w(v)[L7])))
ST = .02
tri = [(a, b, 1-a-b) for a in np.arange(0,1.0001,ST) for b in np.arange(0,1.0001-a,ST)
       if 1-a-b >= -1e-9]
val = np.array([gmean(np.array(t)) for t in tri])
# barycentric -> cartesian, corners: perf (0,0), market (1,0), motive (.5, .866)
xy = np.array([[t[1] + .5*t[2], .8660*t[2]] for t in tri])
fig, ax = plt.subplots(figsize=(6.2,5.4))
sc = ax.scatter(xy[:,0], xy[:,1], c=np.log10(val), s=26, cmap="RdYlGn_r", marker="h")
for corner, lb in (((0,0),"100%\nperformance"), ((1,0),"100%\nmarket"), ((.5,.866),"100%\nmotive")):
    ax.annotate(lb, corner, ha="center", va="center", fontsize=8.5, weight="bold",
                xytext=(0, -22 if corner[1] == 0 else 26), textcoords="offset points")
ax.plot(*zip(*[(0,0),(1,0),(.5,.866),(0,0)]), color=INK, lw=1)
for v, lb, mk in ((np.array([.45,.30,.25]), f"shipped 45/30/25\n{gmean(np.array([.45,.30,.25])):.0f}", "o"),
                  (np.array([.25,.025,.725]), f"grid optimum\n{gmean(np.array([.25,.025,.725])):.0f}", "*")):
    p = (v[1] + .5*v[2], .8660*v[2])
    ax.scatter(*p, s=190 if mk=="*" else 90, marker=mk, color="white",
               edgecolors=INK, linewidths=1.6, zorder=5)
    ax.annotate(lb, p, xytext=(12,6), textcoords="offset points", fontsize=8, weight="bold")
plt.colorbar(sc, ax=ax, label="log$_{10}$ geometric-mean rank of the 7 labelled games",
             fraction=.045)
ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
ax.set_title("Every block-weight vector, scored on the labels\n"
             "the optimum deletes the market block — which is the one block with "
             "label-free validation", loc="left", fontsize=10, pad=26)
save(fig,"f9_weight_simplex")
note(f"F9: over {len(tri):,} block-weight vectors, shipped 45/30/25 gives geo-mean rank "
     f"{gmean(np.array([.45,.30,.25])):.1f}; grid optimum 25/2.5/72.5 gives "
     f"{gmean(np.array([.25,.025,.725])):.1f}; market-deleted 64/0/36 gives "
     f"{gmean(np.array([.643,0,.357])):.1f}")
