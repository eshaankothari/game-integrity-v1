# experiments/

Nothing here is in the data pipeline. The pipeline is `standardize.py` (L5) and
`export_candidates.py` (L6) at the repo root, and they depend on none of these files.

Kept because the **negative results constrain what the pipeline is allowed to claim**.
Every rejection below is a measurement, not an opinion.

## Superseded — these became L5 and L6

| file | became |
|---|---|
| `two_metrics.py` | `standardize.py` — the three blocks |
| `combined_cut.py`, `loose_cut.py` | `export_candidates.py` — the six cuts |
| `export_ranked.py` | L6 writes `out/candidates.csv` directly |
| `score_candidates.py` | old L7, one gate + weighted score |

## Rejected, with the reason

| file | verdict |
|---|---|
| `fit_logistic.py`, `fit_all.py`, `fit_splits.py`, `wrapper_select.py` | Supervised learning on 6 positives from 2 players. Leave-one-**player**-out AUC 0.661 / 0.600 against a 0.50 baseline; with salary included, test AUC **0.087** — it learned "Malik Beasley". |
| `isolation.py`, `iso_three.py`, `iso_peer.py`, `iso_cut.py` | Isolation Forest, four configurations. Lost to a weighted sum every time: it scores **rarity**, and the target class is a **mode** (349 zero-point games form a dense cluster). It also inverts `motive` — being underpaid is not rare among games that fail badly. |
| `peer_z.py` | Peer-matched z (role × minutes band). Fixes the minutes bias (`corr` +0.337 → +0.052) but conditions away the withdrawal itself: Porter 03-20 falls 206 → 541. |
| `nomotive_cut.py` | Salary as a gate instead of a score component. Lost at all seven thresholds — flagged games went from 10.9% of pool ungated to 15.9% gated. |
| `resid_stats.py` | Context residualization. `corr(score, score_residualized) = 0.995`. |
| `effort_search.py` | Exhaustive search over effort-stat subsets. `corr(ICC, |corr with performance|) = +0.829` — reliability and independence are in direct conflict. |

## Adopted — these changed the pipeline

| file | what it settled |
|---|---|
| `weight_audit.py` | Within-block weights are `1/n`. One-at-a-time swings 0.02–0.09 vs 0.58 for the motive block; **48.1% of 20,000 random weight vectors beat the tuned ones**. Reads the live weights out of `standardize.py` by AST so it cannot drift. |
| `tier_blend.py` | Adding `game_z_tier` + `effort_z_tier` as extra components. The only blend of six that improved Beasley (−24.7%) without costing Porter (−2.3%). |

## Unfinished

`expect_model.py` — a market-independent expectation model (gradient boosting on
pregame-only features, quantile regression for a calibrated `P(points ≤ observed)`).
Held-out MAE **5.279** against the closing line's **4.989**, beating a season-average
baseline of 5.502. Two known problems: the quantile grid is **not calibrated** (nominal
5th percentile captures 25.4% of actuals) and it crashes on the final CSV write (`q05` vs
`q5`). Parked, not abandoned.

## Exploration

`queries.py` (registry of read-only queries), `salary_dist.py`, `gs_league.py`,
`three_axis.py`, `simple_z.py`, `minutes_fits.py`, `early_exit.py`.

Outputs land in `experiments/out/`.
