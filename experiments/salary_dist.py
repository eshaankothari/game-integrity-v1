"""Salary distribution -- three populations, because they are not the same shape.

  per PLAYER        one row per man on a roster.  This is the league's pay structure.
  per PLAYER-GAME   weighted by games played.  Stars play more, so this shifts right.
                    It is the distribution your cuts actually operate on.
  per PROPPED GAME  only games with a points line.  Books post stars first, so this
                    shifts right again -- the fringe is under-represented before any
                    cut of yours runs.

Two-way / unlisted contracts (has_listed_salary = false) have no dollar figure at all
and are drawn as a separate bar, never as $0.

    python analysis/salary_dist.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/eshaankothari/Desktop/game-integrity-v1")
import config
import db

OUT = pathlib.Path(__file__).resolve().parent / "out"
CUT = 20_000_000          # the salary cut used in combined_cut.py

SQL = """
    SELECT s.player_id, p.full_name AS player, s.salary, s.has_listed_salary,
           s.salary_pct,
           COUNT(g.game_id)                                        AS n_games,
           COUNT(f.close_line)                                     AS n_propped
      FROM player_salaries s
      JOIN players p            ON p.player_id = s.player_id
      LEFT JOIN player_games g  ON g.player_id = s.player_id
                               AND g.minutes > 0
      LEFT JOIN player_game_features f ON f.player_id = g.player_id
                               AND f.game_id = g.game_id
     WHERE s.season = %(season)s
     GROUP BY s.player_id, p.full_name, s.salary, s.has_listed_salary, s.salary_pct
"""


def load():
    with db.connect() as conn:
        d = pd.read_sql(SQL, conn, params={"season": config.SEASON})
    d["salary_m"] = pd.to_numeric(d["salary"], errors="coerce") / 1e6
    return d


def describe(name, vals, w=None):
    """Weighted quantiles -- w is games played, so a 70-game star counts 70x."""
    v = pd.to_numeric(vals, errors="coerce")
    ok = v.notna()
    v = v[ok]
    w = np.ones(len(v)) if w is None else np.asarray(w)[ok.values]
    o = np.argsort(v.values)
    v, w = v.values[o], w[o]
    cw = np.cumsum(w) / w.sum()
    q = lambda p: v[np.searchsorted(cw, p)]
    print(f"   {name:<18} n={int(w.sum()):>7,}   "
          f"p10 ${q(.10):>5.1f}M   median ${q(.50):>5.1f}M   "
          f"p90 ${q(.90):>5.1f}M   mean ${np.average(v, weights=w):>5.1f}M   "
          f"share <= ${CUT/1e6:.0f}M {100*w[v <= CUT/1e6].sum()/w.sum():>5.1f}%")
    return q


def main():
    d = load()
    listed = d[d.salary_m.notna()]
    unlisted = d[d.salary_m.isna()]

    print(f"players {len(d):,}   listed {len(listed):,}   "
          f"two-way/unlisted {len(unlisted):,}")
    print(f"player-games {int(d.n_games.sum()):,}   "
          f"propped {int(d.n_propped.sum()):,}\n")

    print("SALARY ($M)")
    describe("per player", listed.salary_m)
    describe("per player-game", listed.salary_m, listed.n_games)
    describe("per propped game", listed.salary_m, listed.n_propped)

    print(f"\nTOP 10 EARNERS")
    for _, r in listed.nlargest(10, "salary_m").iterrows():
        print(f"   {r.player:<26} ${r.salary_m:>5.1f}M   {int(r.n_games):>3} games"
              f"   {int(r.n_propped):>3} propped")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    bins = np.arange(0, listed.salary_m.max() + 2.5, 2.5)

    # (0,0) per player -------------------------------------------------------
    a = ax[0, 0]
    a.hist(listed.salary_m, bins=bins, color="#4C78A8", edgecolor="white")
    a.bar(-3.0, len(unlisted), width=2.2, color="#B0B0B0", edgecolor="white")
    a.text(-3.0, len(unlisted), f" {len(unlisted)}\n unlisted", ha="center",
           va="bottom", fontsize=8, color="#555")
    a.axvline(listed.salary_m.median(), color="crimson", ls="--", lw=1.4,
              label=f"median ${listed.salary_m.median():.1f}M")
    a.axvline(CUT / 1e6, color="black", ls=":", lw=1.4, label=f"${CUT/1e6:.0f}M cut")
    a.set_title(f"per PLAYER  (n={len(listed)} listed)")
    a.set_xlabel("salary ($M)"); a.set_ylabel("players"); a.legend(fontsize=8)

    # (0,1) per player-game vs per propped game ------------------------------
    a = ax[0, 1]
    a.hist([listed.salary_m, listed.salary_m], bins=bins,
           weights=[listed.n_games, listed.n_propped],
           color=["#4C78A8", "#F58518"], edgecolor="white",
           label=["all player-games", "propped games"])
    a.axvline(CUT / 1e6, color="black", ls=":", lw=1.4)
    a.set_title("per PLAYER-GAME  (weighted by games played)")
    a.set_xlabel("salary ($M)"); a.set_ylabel("player-games"); a.legend(fontsize=8)

    # (1,0) log scale --------------------------------------------------------
    a = ax[1, 0]
    pos = listed[listed.salary_m > 0]
    a.hist(np.log10(pos.salary_m * 1e6), bins=40, color="#54A24B", edgecolor="white")
    a.axvline(np.log10(CUT), color="black", ls=":", lw=1.4, label=f"${CUT/1e6:.0f}M")
    a.set_xticks([5.5, 6, 6.5, 7, 7.5, 8])
    a.set_xticklabels(["$0.3M", "$1M", "$3M", "$10M", "$32M", "$100M"])
    a.set_title("log scale -- the pay structure is roughly bimodal")
    a.set_xlabel("salary"); a.set_ylabel("players"); a.legend(fontsize=8)

    # (1,1) CDF --------------------------------------------------------------
    a = ax[1, 1]
    for vals, wts, lab, col in [
        (listed.salary_m, None, "per player", "#4C78A8"),
        (listed.salary_m, listed.n_games, "per player-game", "#F58518"),
        (listed.salary_m, listed.n_propped, "per propped game", "#54A24B"),
    ]:
        w = np.ones(len(vals)) if wts is None else np.asarray(wts, float)
        o = np.argsort(vals.values)
        a.step(vals.values[o], np.cumsum(w[o]) / w.sum(), where="post",
               lw=1.8, color=col, label=lab)
    a.axvline(CUT / 1e6, color="black", ls=":", lw=1.4)
    a.set_xscale("log")
    a.set_xticks([1, 3, 10, 20, 30, 50])
    a.set_xticklabels(["$1M", "$3M", "$10M", "$20M", "$30M", "$50M"])
    a.set_title(f"CDF -- where the ${CUT/1e6:.0f}M cut falls")
    a.set_xlabel("salary"); a.set_ylabel("cumulative share")
    a.grid(alpha=.25); a.legend(fontsize=8)

    fig.suptitle(f"NBA salary distribution  {config.SEASON}", fontsize=13)
    fig.tight_layout()
    path = f"{OUT}/salary_dist.png"
    fig.savefig(path, dpi=140)
    listed.sort_values("salary_m", ascending=False).to_csv(
        f"{OUT}/salary_dist.csv", index=False)
    print(f"\n-> {path}\n-> {OUT}/salary_dist.csv")


if __name__ == "__main__":
    main()
