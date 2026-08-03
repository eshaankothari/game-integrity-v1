"""Where did the weights come from, and do they matter?  Four tests.

The honest problem up front: there are 6 labelled games from 2 players. Fitting 8
weights to that is not estimation, it is interpolation. So rather than fit once and
believe it, this asks four questions that each fail in a different way.

  A  LEAVE-ONE-PLAYER-OUT LOGISTIC.  Fit on Beasley's games, test on Porter's, and
     vice versa. The ONLY honest split when the label set has 2 players -- a random
     split leaks the player's identity across the fold through every player-level
     feature. If test rank is mid-pack, the fit memorised a person.

  B  RANDOM-WEIGHT ENSEMBLE.  Draw 20,000 weight vectors from a Dirichlet and score
     the league with each. This maps the whole achievable range: if the flagged games
     rank well under almost any weights, the weights are not doing the work and
     arguing about them is wasted effort. If they rank well only in a narrow corner,
     the current weights are a fitted guess wearing a prior's clothes.

  C  ONE-AT-A-TIME SENSITIVITY.  Move each weight to 0 and to 2x, hold the rest,
     renormalise. Shows which single weight the ranking actually hangs on.

  D  EQUAL WEIGHTS.  The null. If equal weights do as well, the tuning bought nothing.

    python analysis/weight_audit.py
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def live_weights():
    """Read PERF_W / MARKET_W / BLOCK_W out of standardize.py (L5) by AST, so this audit
    can never drift from the weights actually in production. Parsed, not imported --
    importing standardize would connect to the database and run L5."""
    import ast
    tree = ast.parse((HERE.parent / "standardize.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm in ("PERF_W", "MARKET_W", "BLOCK_W"):
                # eval, not literal_eval: the weights are written as 1/3 rather than
                # 0.3333 so they sum exactly, and a division is a BinOp that
                # literal_eval rejects. Scope is empty and the source is our own file.
                found[nm] = eval(compile(ast.Expression(node.value), "<w>", "eval"),
                                 {"__builtins__": {}}, {})
    missing = {"PERF_W", "MARKET_W", "BLOCK_W"} - set(found)
    if missing:
        raise SystemExit(f"could not parse {missing} from standardize.py")
    return found["PERF_W"], found["MARKET_W"], found["BLOCK_W"]


PERF_W, MARKET_W, BLOCK_W = live_weights()
N_DRAW = 20_000
SEED = 0

FLAG = {"Malik Beasley": ["2024-01-06", "2024-01-26", "2024-02-27", "2024-03-10"],
        "Jontay Porter": ["2024-01-20", "2024-03-20"]}


# --------------------------------------------------------------------------- data
def load():
    d = pd.read_csv(OUT / "two_metrics.csv")
    d["gd"] = d.gd.astype(str)
    d["y"] = [int(r.gd in FLAG.get(r.player, [])) for _, r in d.iterrows()]
    # orient so HIGHER = more suspicious, matching two_metrics.py
    d["o_game_z"] = -d.game_z
    d["o_effort_z"] = -d.effort_z
    d["o_shortfall_z"] = d.shortfall_z
    return d


def blend(d, pw, mw, bw):
    """Recompute score from components. Present-weight normalisation, as in prod:
    a missing movement term must not drag the block toward zero for a reason that
    has nothing to do with the game."""
    def wmean(w, prefix):
        num = sum(v * d[prefix + k].fillna(0) for k, v in w.items())
        den = sum(v * d[prefix + k].notna().astype(float) for k, v in w.items())
        return num / den.replace(0, np.nan)

    perf = wmean(pw, "o_")
    mkt = wmean(mw, "mk_")
    return (bw["performance"] * perf.fillna(0)
            + bw["market"] * mkt.fillna(0)
            + bw["motive"] * d.motive.fillna(0))


def ranks_of_flagged(score, d):
    r = score.rank(ascending=False, method="min")
    return r[d.y == 1].values


def cost(rk):
    """Mean log10 rank. Log because moving 5000 -> 2500 matters as much as 200 -> 100."""
    return float(np.mean(np.log10(rk)))


# ----------------------------------------------------------------- A  LOPO logistic
def lopo(d):
    feats = ["o_game_z", "o_effort_z", "o_shortfall_z", "market", "motive"]
    X = d[feats].fillna(0).values
    y = d.y.values
    print("A  LEAVE-ONE-PLAYER-OUT LOGISTIC")
    print(f"   features {feats}")
    print(f"   positives {y.sum()}  from {d.loc[d.y == 1, 'player'].nunique()} players "
          f"of {len(d):,} rows\n")

    rows = []
    for held in FLAG:
        tr = d.player != held
        te = d.player == held
        if d.loc[te, "y"].sum() == 0:
            continue
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)
        m.fit(X[tr.values], y[tr.values])
        p = m.predict_proba(X)[:, 1]
        auc_te = roc_auc_score(y[te.values], p[te.values]) if \
            len(set(y[te.values])) > 1 else float("nan")
        # rank of the held-out player's flagged games among the WHOLE league
        rk = pd.Series(p).rank(ascending=False, method="min")[d.y.values.astype(bool)
                                                             & te.values].values
        rows.append((held, m.coef_[0], auc_te, rk))
        print(f"   held out {held:<16} within-player AUC {auc_te:.3f}"
              f"   league ranks {[int(v) for v in rk]}")
        for f_, c_ in sorted(zip(feats, m.coef_[0]), key=lambda t: -abs(t[1])):
            print(f"      {f_:<16}{c_:+.3f}")
        print()
    if len(rows) == 2:
        a, b = rows[0][1], rows[1][1]
        print(f"   coefficient stability across the two fits "
              f"(corr {np.corrcoef(a, b)[0,1]:+.3f}):")
        for f_, x_, y_ in zip(feats, a, b):
            flip = "  SIGN FLIP" if x_ * y_ < 0 else ""
            print(f"      {f_:<16}{x_:+7.3f}  vs {y_:+7.3f}{flip}")
    return rows


# ------------------------------------------------------------- B  random ensemble
def ensemble(d):
    rng = np.random.default_rng(SEED)
    base = cost(ranks_of_flagged(blend(d, PERF_W, MARKET_W, BLOCK_W), d))
    eq_p = {k: 1 / 3 for k in PERF_W}
    eq_m = {k: 1 / 4 for k in MARKET_W}
    eq_b = {k: 1 / 3 for k in BLOCK_W}
    equal = cost(ranks_of_flagged(blend(d, eq_p, eq_m, eq_b), d))

    costs, best = np.empty(N_DRAW), (np.inf, None)
    for i in range(N_DRAW):
        pw = dict(zip(PERF_W, rng.dirichlet(np.ones(3))))
        mw = dict(zip(MARKET_W, rng.dirichlet(np.ones(4))))
        bw = dict(zip(BLOCK_W, rng.dirichlet(np.ones(3))))
        c = cost(ranks_of_flagged(blend(d, pw, mw, bw), d))
        costs[i] = c
        if c < best[0]:
            best = (c, (pw, mw, bw))

    pct = 100 * (costs < base).mean()
    print(f"\nB  RANDOM-WEIGHT ENSEMBLE  ({N_DRAW:,} Dirichlet draws)")
    print(f"   cost = mean log10(rank) of the 6 flagged games.  lower is better.\n")
    print(f"   current weights      {base:.3f}   -> geometric-mean rank "
          f"{10**base:>7,.0f}")
    print(f"   equal weights        {equal:.3f}   -> {10**equal:>7,.0f}")
    print(f"   random  best         {costs.min():.3f}   -> {10**costs.min():>7,.0f}")
    print(f"   random  median       {np.median(costs):.3f}   -> "
          f"{10**np.median(costs):>7,.0f}")
    print(f"   random  worst        {costs.max():.3f}   -> {10**costs.max():>7,.0f}")
    print(f"\n   {pct:.1f}% of random weight vectors BEAT the current ones.")
    print(f"   full achievable spread: {10**costs.min():,.0f} to "
          f"{10**costs.max():,.0f}  ({10**(costs.max()-costs.min()):.1f}x)")
    print(f"\n   best random draw found:")
    for nm, wd in zip(("perf", "market", "block"), best[1]):
        print(f"      {nm:<8}" + "  ".join(f"{k} {v:.2f}" for k, v in wd.items()))
    return costs, base, equal


# --------------------------------------------------------- C  one-at-a-time
def sensitivity(d):
    print(f"\nC  ONE-AT-A-TIME  (set each weight to 0 and 2x, renormalise its block)")
    base_rk = ranks_of_flagged(blend(d, PERF_W, MARKET_W, BLOCK_W), d)
    b = cost(base_rk)
    print(f"   baseline cost {b:.3f}   flagged ranks "
          f"{sorted(int(v) for v in base_rk)}\n")
    print(f"   {'weight':<22}{'x0':>10}{'x2':>10}{'swing':>9}")
    blocks = [("perf", PERF_W), ("market", MARKET_W), ("block", BLOCK_W)]
    out = []
    for bn, wd in blocks:
        for k in wd:
            row = []
            for mult in (0.0, 2.0):
                new = dict(wd)
                new[k] = wd[k] * mult
                tot = sum(new.values())
                if tot == 0:
                    row.append(np.nan)
                    continue
                new = {kk: vv / tot for kk, vv in new.items()}
                args = [PERF_W, MARKET_W, BLOCK_W]
                args[[n for n, _ in blocks].index(bn)] = new
                row.append(cost(ranks_of_flagged(blend(d, *args), d)))
            swing = abs(row[1] - row[0])
            out.append((f"{bn}.{k}", row[0], row[1], swing))
    for nm, z0, z2, sw in sorted(out, key=lambda t: -t[3]):
        bar = "#" * int(min(sw * 60, 40))
        print(f"   {nm:<22}{z0:>10.3f}{z2:>10.3f}{sw:>9.3f}  {bar}")
    return out


# ------------------------------------------------------------------------ main
def main():
    d = load()
    print(f"rows {len(d):,}   flagged found {int(d.y.sum())} of "
          f"{sum(len(v) for v in FLAG.values())}\n")
    if d.y.sum() < sum(len(v) for v in FLAG.values()):
        have = set(zip(d.loc[d.y == 1, "player"], d.loc[d.y == 1, "gd"]))
        want = {(p, g) for p, gs in FLAG.items() for g in gs}
        print(f"   !! missing: {want - have}\n")

    lopo(d)
    costs, base, equal = ensemble(d)
    sensitivity(d)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    a = ax[0]
    a.hist(10**costs, bins=60, color="#4C78A8", edgecolor="white")
    a.axvline(10**base, color="crimson", lw=2,
              label=f"current  {10**base:,.0f}")
    a.axvline(10**equal, color="darkorange", lw=2, ls="--",
              label=f"equal    {10**equal:,.0f}")
    a.set_xlabel("geometric-mean rank of the 6 flagged games")
    a.set_ylabel("weight vectors")
    a.set_title(f"What ANY weighting can achieve  ({N_DRAW:,} draws)")
    a.legend(fontsize=9)

    a = ax[1]
    sc = blend(d, PERF_W, MARKET_W, BLOCK_W)
    a.hist(sc.dropna(), bins=80, color="#B0B0B0", edgecolor="white")
    for _, r in d[d.y == 1].iterrows():
        a.axvline(sc[r.name], color="crimson", lw=1.2)
    a.set_xlabel("score (current weights)")
    a.set_ylabel("player-games")
    a.set_title("Where the 6 flagged games sit in the score distribution")
    fig.tight_layout()
    fig.savefig(OUT / "weight_audit.png", dpi=140)
    print(f"\n-> {OUT/'weight_audit.png'}")


if __name__ == "__main__":
    main()
