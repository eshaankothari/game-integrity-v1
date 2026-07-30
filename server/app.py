"""L8: read-only API over the pipeline's outputs, for the React dashboard.

SERVES FILES FIRST, DATABASE SECOND. The watchlist, summary, funnel and isolation
endpoints read out/*.csv -- the exact artifacts the scoring run wrote -- so the
dashboard can never disagree with the last `python score_candidates.py`. The
database is consulted only where a CSV cannot answer: the case view's full-season
game log. If Postgres is down, everything except that one panel still works.

WEIGHTS are imported from score_candidates, not copied, so a re-weighting there
re-labels the UI on next restart without a second edit.

    uvicorn server.app:app --reload --port 8000
"""
import math
import pathlib
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db                                    # noqa: E402
from score_candidates import WEIGHTS         # noqa: E402

OUT = ROOT / "out"

WATCHLIST_COLS = [
    "rank", "player", "position", "tier", "game_date", "matchup",
    "minutes", "points", "line", "line_source", "line_pulled",
    "close_under", "salary", "has_listed_salary", "score",
    "s_shortfall", "s_production", "s_involvement", "s_motive",
    "s_pulled", "s_under_money",
    "prod_z", "effort_z", "fga", "touches", "shortfall",
    "game_margin", "plus_minus", "fouls", "started",
    "player_id", "game_id",
]

SORTABLE = {"rank", "score", "game_date", "player", "minutes", "points",
            "prod_z", "effort_z", "salary", "s_shortfall", "s_production",
            "s_involvement", "s_motive", "s_under_money"}

# The funnel is REAL COUNTS from the run's own artifacts, matching the stdout of
# export_candidates.py -- never recomputed here, never approximated.
FUNNEL_FILES = [
    ("propped player-games", "isolation.csv"),
    ("cut 1 · production", "cut1_production.csv"),
    ("cut 2 · effort", "cut2_effort.csv"),
    ("cut 3 · salary", "cut3_salary.csv"),
    ("candidates", "candidates.csv"),
]

app = FastAPI(title="game-integrity-v1 API", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _clean(records):
    """NaN/inf -> None so the JSON is valid and the client sees explicit nulls."""
    return [{k: (None if isinstance(v, float) and not math.isfinite(v) else v)
             for k, v in r.items()} for r in records]


def _load():
    scored = pd.read_csv(OUT / "scored.csv")
    iso = pd.read_csv(OUT / "isolation.csv")
    return scored, iso


SCORED, ISO = _load()


@app.get("/api/summary")
def summary(review_threshold: float = Query(0.75, ge=0, le=1)):
    s = SCORED["score"]
    edges = [i / 20 for i in range(21)]                     # fixed 0.05 bins
    counts = pd.cut(s, bins=edges, include_lowest=True).value_counts().sort_index()
    return {
        "scored": len(SCORED),
        "review_tail": int((s >= review_threshold).sum()),
        "review_threshold": review_threshold,
        "pulled_and_played": int(SCORED["line_pulled"].sum()),
        "unlisted_salary": int((SCORED["has_listed_salary"] == False).sum()),  # noqa: E712
        "top_score": round(float(s.max()), 4),
        "median_score": round(float(s.median()), 4),
        "weights": WEIGHTS,
        "histogram": [{"lo": edges[i], "hi": edges[i + 1], "n": int(c)}
                      for i, c in enumerate(counts)],
    }


@app.get("/api/watchlist")
def watchlist(q: str = "", sort: str = "score", dir: str = "desc",
              limit: int = Query(50, le=500), offset: int = 0):
    if sort not in SORTABLE:
        raise HTTPException(400, f"sort must be one of {sorted(SORTABLE)}")
    df = SCORED
    if q:
        df = df[df["player"].str.contains(q, case=False, na=False)
                | df["matchup"].str.contains(q, case=False, na=False)]
    df = df.sort_values(sort, ascending=(dir == "asc"), na_position="last")
    page = df[WATCHLIST_COLS].iloc[offset:offset + limit]
    return {"total": len(df), "offset": offset,
            "rows": _clean(page.to_dict("records"))}


@app.get("/api/case/{player_id}/{game_id}")
def case(player_id: int, game_id: str):
    row = SCORED[(SCORED["player_id"] == player_id) & (SCORED["game_id"] == game_id)]
    if row.empty:
        raise HTTPException(404, "player-game not in scored.csv")
    r = _clean(row[WATCHLIST_COLS].to_dict("records"))[0]

    # Weighted contributions, computed here from the SAME weights the pipeline
    # used, so the UI's arithmetic is checkable against the score column.
    axes = {k: (row.iloc[0][f"s_{k}"] if pd.notna(row.iloc[0][f"s_{k}"]) else 0.0)
            for k in WEIGHTS}
    total_w = sum(WEIGHTS.values())
    r["axes"] = [{"axis": k, "weight": WEIGHTS[k], "value": round(float(v), 4),
                  "contribution": round(WEIGHTS[k] * float(v) / total_w, 4)}
                 for k, v in axes.items()]

    # Season log from Postgres; the one panel the CSVs cannot fill. Sorted by
    # date so the client can draw points-vs-line across the season.
    r["season_log"], r["season_log_source"] = [], "unavailable"
    try:
        with db.connect() as conn:
            log = pd.read_sql(
                """SELECT g.game_date, g.matchup, pg.minutes, pg.points,
                          f.close_line, f.margin_vs_line
                   FROM player_games pg
                   JOIN games g ON g.game_id = pg.game_id
                   LEFT JOIN player_game_features f
                          ON f.player_id = pg.player_id AND f.game_id = pg.game_id
                   WHERE pg.player_id = %s
                   ORDER BY g.game_date""",
                conn, params=(player_id,))
        r["season_log"] = _clean(log.to_dict("records"))
        r["season_log_source"] = "postgres"
    except Exception:
        pass
    return r


@app.get("/api/funnel")
def funnel():
    stages = []
    for label, fname in FUNNEL_FILES:
        p = OUT / fname
        if p.exists():
            stages.append({"stage": label,
                           "n": int(sum(1 for _ in open(p)) - 1)})
    stages.append({"stage": "scored (gate path)", "n": len(SCORED)})
    return {"stages": stages}


@app.get("/api/isolation")
def isolation(limit: int = Query(15, le=100)):
    cols = ["iso_rank", "iso_score", "player", "game_date", "matchup",
            "minutes", "points", "close_line", "player_id", "game_id"]
    top = ISO.sort_values("iso_rank").head(limit)[cols].copy()
    # Where does the composite put the same player-game? Disagreement between the
    # two rankers is the dashboard's most interesting list.
    ranks = SCORED.set_index(["player_id", "game_id"])["rank"]
    top["composite_rank"] = [
        int(ranks.get((p, g))) if (p, g) in ranks.index else None
        for p, g in zip(top["player_id"], top["game_id"])]
    return {"rows": _clean(top.to_dict("records"))}
