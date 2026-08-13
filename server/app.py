"""L7: read-only API over the pipeline's tables, for the React dashboard.

POSTGRES IS THE ONLY SOURCE. Every endpoint queries the three tables L5 and L6 write:

    player_game_features   RAW -- box score, lines, prices, open->close movement
    player_game_z          STANDARDISED -- the 5 performance and 4 market components,
                           and the three blocks they combine into
    player_game_scores     THE SHORTLIST -- rank, in_shortlist, cut_failed, one row per
                           propped player-game (all 15,498, not just the survivors)

The previous version read out/*.csv and could silently serve a stale run. It also
imported WEIGHTS from a scoring script that has since been retired. Both are gone:
the weights are read from standardize.py, which is the one place they are defined, and
every count comes from a live query.

FILTERING, SORTING AND PAGINATION HAPPEN IN SQL, not in pandas after a full table read.
15,498 rows is small enough that either would work today, but a LIMIT that the database
honours is the difference between a dashboard that stays responsive as seasons are added
and one that does not.

    uvicorn server.app:app --reload --port 8000
"""
import math
import pathlib
import sys

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.core import config, db                              # noqa: E402
from pipeline.llm_review import packet, summarize                    # noqa: E402
from pipeline.score.standardize import BLOCK_W, PERF_W, MARKET_W  # noqa: E402

# The three blocks, in the order the UI shows them. Imported rather than copied, so a
# re-weighting in L5 re-labels the dashboard on next restart with no second edit.
WEIGHTS = dict(BLOCK_W)

# Ground truth: games confirmed suspicious by league/federal action. Mirrors
# CONFIRMED in frontend/src/severity.ts -- update both together.
CONFIRMED_GAMES = [
    "1627736-0022300173",  # Malik Beasley · 2023-11-11
    "1627736-0022300401",  # Malik Beasley · 2023-12-25
    "1627736-0022300496",  # Malik Beasley · 2024-01-06
    "1627736-0022300639",  # Malik Beasley · 2024-01-26
    "1627736-0022300840",  # Malik Beasley · 2024-02-27
    "1627736-0022300924",  # Malik Beasley · 2024-03-10 (not in shortlist)
    "1629007-0022300609",  # Jontay Porter · 2024-01-22
    # "1629007-0022300637",  Jontay Porter · 2024-01-26 (no line -> never matches)
    "1629007-0022300999",  # Jontay Porter · 2024-03-20
]

app = FastAPI(title="game-integrity-v1 API", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Whitelist, not string interpolation. `sort` arrives from the query string and is
# concatenated into ORDER BY, so anything not on this list must be rejected outright.
SORTABLE = {"rank", "score", "score_100", "game_date", "player", "minutes",
            "points", "shortfall", "performance", "market", "motive", "salary",
            "game_z", "effort_z", "close_line"}

# One row per propped player-game, joined across the three tables. Everything the
# watchlist and the case view need except the season log and the residuals.
BASE_SQL = """
SELECT s.rank, s.rank_all, s.in_shortlist, s.cut_failed,
       s.player, s.game_date, s.matchup, s.tier,
       s.minutes, s.points, s.close_line, s.close_under,
       s.shortfall, s.margin_vs_line,
       s.performance, s.market, s.motive, s.score,
       s.score_100, s.performance_100, s.market_100, s.motive_100,
       s.under_hit, s.ejected, s.ejected_alone,
       s.salary, s.has_listed_salary,
       s.player_id, s.game_id,
       f.position, f.fga, f.rebounds, f.assists, f.usage_pct, f.turnover_ratio,
       f.distance, f.touches, f.passes, f.game_score,
       f.open_line, f.open_under, f.line_move_pct, f.under_move_pct,
       f.price_only_move, f.n_player_games,
       z.game_z, z.effort_z, z.game_z_tier, z.effort_z_tier, z.shortfall_z,
       z.p_price, z.p_line, z.n_market,
       z.mk_p_price, z.mk_p_line, z.mk_line_mv, z.mk_price_mv,
       -- Context the dashboard displays but never filters on. game_margin is summed
       -- from the box scores rather than stored: a blowout is the strongest innocent
       -- explanation for a quiet night, so a reviewer needs to see it -- but filtering
       -- on it would delete real candidates alongside the innocent ones.
       pg.plus_minus, pg.fouls, pg.started, pg.steals, pg.blocks,
       sal.salary_pct, pl.experience,
       gm.game_margin,
       -- The independence tail: how rare this combination of a short price AND a small
       -- line is, as a product of two percentiles. Computed here rather than stored,
       -- because it is a presentation of p_price and p_line, not a new measurement.
       (z.p_price * z.p_line) AS tail_pct
  FROM player_game_scores s
  JOIN player_game_features f USING (player_id, game_id)
  JOIN player_game_z        z USING (player_id, game_id)
  JOIN player_games        pg USING (player_id, game_id)
  JOIN players             pl USING (player_id)
  LEFT JOIN player_salaries sal
         ON sal.player_id = s.player_id AND sal.season = %(season)s
  LEFT JOIN (SELECT game_id, max(pts) - min(pts) AS game_margin
               FROM (SELECT game_id, team_id, sum(points) AS pts
                       FROM player_games WHERE points IS NOT NULL
                      GROUP BY 1, 2) t
              GROUP BY game_id HAVING count(*) = 2) gm ON gm.game_id = s.game_id
"""

# BASE_SQL carries a named placeholder, so every caller must pass %(season)s alongside
# its own positional args. Mixing the two styles in one psycopg query is an error, so
# the helper below normalises: callers pass a dict, always.
SEASON = config.SEASON


def q(sql, params=None, one=False):
    """Run a query, return dicts with NaN/inf scrubbed to None.

    db.rows() picks the backend: Postgres by default, the DuckDB export when GI_DB
    points at one. The SQL below is written once, in psycopg dialect, either way.
    """
    out = db.rows(sql, params)
    out = [{k: _fin(v) for k, v in r.items()} for r in out]
    return (out[0] if out else None) if one else out


def _fin(v):
    """NaN/inf -> None so the JSON is valid and the client sees an explicit null.
    Decimal -> float so it serialises at all."""
    import decimal
    if isinstance(v, decimal.Decimal):
        v = float(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _decorate(r):
    """Fields the dashboard expects that are presentation, not storage.

    g_performance / g_market / g_motive were group aggregates in the old six-axis
    scheme. They are now first-class columns, so the mapping is an alias -- kept so the
    React components need no change.

    line_pulled is always False. Pulled lines -- opened, then withdrawn before tip -- are
    excluded from the pipeline entirely (47 player-games), so nothing that reaches this
    API can have one. The field stays so the UI's badge logic keeps working rather than
    reading undefined.
    """
    if r is None:
        return None
    # g_* carry the 0-100 block percentiles, because that is what the dashboard renders.
    # The raw z-scale performance / market / motive stay in the payload under their own
    # names for anything that needs the population-independent value.
    r["g_performance"] = r.get("performance_100")
    r["g_market"] = r.get("market_100")
    r["g_motive"] = r.get("motive_100")
    r["line"] = r.get("close_line")
    r["line_source"] = "close"
    r["line_pulled"] = False

    # prod_z was mean(points_z, assists_z, rebounds_z, fga_z) under the old scheme. Game
    # Score subsumes all four -- and six more -- so game_z is the same claim, better
    # measured. Aliased rather than renamed in the client because CaseView's sentence
    # ("production Xσ ... below his own season") stays true: game_z IS the own-season
    # baseline. effort_z kept its name and its meaning.
    r["prod_z"] = r.get("game_z")

    # The 0-100 scale, aliased to the names the UI reads. score_100 is a LINEAR MIN-MAX
    # RESCALE of `score` over all propped games, NOT a percentile: 99.0 means the game sits
    # 99% of the way from the lowest raw score to the highest, which says nothing about how
    # many games rank above it. Strictly monotone, so it never reorders anything -- but the
    # gaps are the raw gaps, not equal-sized population slices. `scale_note` says the same.
    r["score_pct100"] = r.get("score_100")
    # `score` is the raw z-scale value, kept for provenance. The UI shows score_100, and
    # `rank` is ordered by score_100 -- so sorting by `score` would produce a list whose
    # order disagrees with the rank column printed beside it.
    return r


@app.get("/api/summary")
def summary(review_threshold: float = Query(73.7, ge=0, le=100)):
    """Headline counts plus the score distribution.

    review_threshold is on the 0-100 scale the UI shows, so the API and the dashboard
    speak one language. 73.7 is the default: the top 1 percent, 155 games, which is a
    reviewable number. It is a cutoff on a LINEAR rescale of the raw score, not a
    percentile -- 73.7 does not mean "better than 73.7% of games".

    Both scales are returned. `score` is the raw weighted mean of z-scores and is
    population-independent -- the number to compare across runs. `score_100` is its
    linear 0-100 rescale and is what a human reads.
    """
    agg = q("""
        SELECT count(*)                                        AS scored,
               count(*) FILTER (WHERE in_shortlist)            AS shortlist,
               count(*) FILTER (WHERE has_listed_salary IS FALSE) AS unlisted_salary,
               count(*) FILTER (WHERE score_100 >= %s)         AS review_tail,
               count(*) FILTER (WHERE score_100 >= %s AND in_shortlist)
                                                               AS review_tail_shortlist,
               max(score_100) AS top_score,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY score_100) AS median_score,
               max(score) AS top_score_raw,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY score) AS median_score_raw,
               min(score_100) AS lo, max(score_100) AS hi
          FROM player_game_scores WHERE score_100 IS NOT NULL
    """, (review_threshold, review_threshold), one=True)

    # 20 equal-width bins across the observed range. Data-driven rather than fixed at
    # 0-1: the score is not a probability and pinning the axis would clip both tails.
    lo, hi = agg["lo"], agg["hi"]
    width = (hi - lo) / 20 or 1
    # Arithmetic rather than width_bucket(): DuckDB has no such function, and the
    # expression below is what width_bucket computes anyway. least() pins the top edge,
    # which would otherwise land the single highest score in a 21st bucket of its own.
    hist = q("""
        SELECT least(floor((score_100 - %s) / %s) + 1, 20) AS b, count(*) AS n
          FROM player_game_scores WHERE score_100 IS NOT NULL GROUP BY b ORDER BY b
    """, (lo, width))
    counts = {r["b"]: r["n"] for r in hist}
    return {
        "scored": agg["scored"],
        "shortlist": agg["shortlist"],
        "review_tail": agg["review_tail"],
        "review_tail_shortlist": agg["review_tail_shortlist"],
        "review_threshold": review_threshold,
        "pulled_and_played": 0,          # pulled lines are excluded upstream
        "unlisted_salary": agg["unlisted_salary"],
        "top_score": round(agg["top_score"], 2),
        "median_score": round(agg["median_score"], 2),
        "top_score_raw": round(agg["top_score_raw"], 4),
        "median_score_raw": round(agg["median_score_raw"], 4),
        # NOT a percentile -- standardize.py min-max rescales the raw score, so 73.7
        # does NOT mean "worse than 73.7% of games". The previous wording said
        # percentile and contradicted both the code and the docstring above it.
        "scale_note": ("score_100 is a linear rescale of the raw score onto 0-100 "
                       "over all propped games, so it preserves the raw score's "
                       "shape and spacing. It is strictly monotone: it changes how "
                       "a number reads, never which games surface. The raw score is "
                       "population-independent; the rescale is not."),
        "weights": WEIGHTS,
        "histogram": [{"lo": round(lo + i * width, 4),
                       "hi": round(lo + (i + 1) * width, 4),
                       "n": counts.get(i + 1, 0)} for i in range(20)],
    }


@app.get("/api/watchlist")
def watchlist(q_: str = Query("", alias="q"), sort: str = "rank",
              dir: str = "asc", limit: int = Query(50, le=20000), offset: int = 0,
              shortlist_only: bool = True):
    if sort not in SORTABLE:
        raise HTTPException(400, f"sort must be one of {sorted(SORTABLE)}")
    where = ["s.score_100 IS NOT NULL"]
    params = {"season": SEASON, "limit": limit, "offset": offset}
    if shortlist_only:
        # Confirmed cases (league/federal action) ride along even when the
        # pipeline's cuts excluded them -- Beasley 2024-01-11 scored 42.8 and
        # is exactly why ground truth must not be filtered by our own model.
        where.append("(s.in_shortlist OR s.player_id::text || '-' || s.game_id"
                     " = ANY(%(confirmed)s))")
        params["confirmed"] = CONFIRMED_GAMES
    if q_:
        where.append("(s.player ILIKE %(q)s OR s.matchup ILIKE %(q)s)")
        params["q"] = f"%{q_}%"
    clause = " WHERE " + " AND ".join(where)

    total = q(f"SELECT count(*) AS n FROM player_game_scores s{clause}",
              params, one=True)["n"]
    # NULLS LAST both ways: an unranked row is missing a position, not holding the best
    # or worst one, so it belongs at the end regardless of sort direction.
    rows = q(f"{BASE_SQL}{clause} ORDER BY s.{sort} "
             f"{'ASC' if dir == 'asc' else 'DESC'} NULLS LAST, s.rank_all "
             f"LIMIT %(limit)s OFFSET %(offset)s", params)
    return {"total": total, "offset": offset,
            "rows": [_decorate(r) for r in rows]}


def _case_payload(player_id: int, game_id: str):
    """Everything the case view and the PDF need for one player-game."""
    r = _decorate(q(f"{BASE_SQL} WHERE s.player_id = %(pid)s AND s.game_id = %(gid)s",
                    {"season": SEASON, "pid": player_id, "gid": game_id}, one=True))
    if r is None:
        raise HTTPException(404, "player-game not found")

    # The score, decomposed. Computed here from the SAME weights L5 used, so the UI's
    # arithmetic is checkable against the score column rather than asserted.
    total_w = sum(WEIGHTS.values())
    r["axes"] = [{"axis": k, "weight": WEIGHTS[k],
                  "value": None if r.get(k) is None else round(r[k], 4),
                  "contribution": None if r.get(k) is None
                  else round(WEIGHTS[k] * r[k] / total_w, 4)}
                 for k in ("performance", "market", "motive")]
    r["components"] = {"performance": PERF_W, "market": MARKET_W}

    # Residual z-scores drive the case view's red highlighting: the stat after margin,
    # rest, back-to-back, home, altitude and pace are regressed out PER ROLE TIER. A
    # starter pulled early in a 30-point blowout has a large raw minutes deficit and a
    # near-zero residual one, because the scoreline already explains him.
    resid = q("""SELECT points_resid_z, fga_resid_z, rebounds_resid_z,
                        assists_resid_z, usage_pct_resid_z, turnover_ratio_resid_z,
                        distance_resid_z, touches_resid_z, minutes_resid_z
                   FROM player_game_residuals
                  WHERE player_id = %s AND game_id = %s""",
              (player_id, game_id), one=True) or {}
    r.update(resid)

    # Hustle and exit anatomy. LEFT-joined shape: a game with no play-by-play row
    # carries nulls rather than vanishing from the case view.
    deep = q("""SELECT pg.contested_shots, pg.deflections, pg.loose_balls,
                       pg.box_outs, pg.passes, pg.steals, pg.blocks,
                       pb.ejected, pb.n_stints, pb.last_out_sec,
                       pb.points_competitive, pb.points_garbage
                  FROM player_games pg
                  LEFT JOIN player_game_pbp pb
                         ON pb.player_id = pg.player_id AND pb.game_id = pg.game_id
                 WHERE pg.player_id = %s AND pg.game_id = %s""",
             (player_id, game_id), one=True)
    if deep is not None:
        deep["indep_pct"] = r.get("tail_pct")
    r["deep"] = deep

    # The game's final score, summed from the box scores (the games table
    # stores no team totals). Also says whether HIS team won or lost.
    totals = q("""SELECT t.abbreviation AS abbr, pg.team_id, sum(pg.points)::int AS pts,
                         (pg.team_id = g.home_team_id) AS is_home,
                         bool_or(pg.player_id = %(pid)s) AS is_players
                    FROM player_games pg
                    JOIN games g ON g.game_id = pg.game_id
                    JOIN teams t ON t.team_id = pg.team_id
                   WHERE pg.game_id = %(gid)s AND pg.points IS NOT NULL
                   GROUP BY t.abbreviation, pg.team_id, g.home_team_id""",
               {"gid": game_id, "pid": player_id})
    if totals is not None and len(totals) == 2:
        away = next(s for s in totals if not s["is_home"])
        home = next(s for s in totals if s["is_home"])
        mine = next((s for s in totals if s["is_players"]), None)
        other = next((s for s in totals if mine is not None and s is not mine), None)
        r["final_score"] = {
            "away": {"team": away["abbr"], "team_id": away["team_id"], "pts": away["pts"]},
            "home": {"team": home["abbr"], "team_id": home["team_id"], "pts": home["pts"]},
            "result": (None if mine is None or other is None
                       else "W" if mine["pts"] > other["pts"] else "L"),
        }
    else:
        r["final_score"] = None

    # Who he is: roster attributes from the players table. Age is computed AT
    # THE GAME DATE, not today -- a case file describes the night in question.
    bio = q("""SELECT height_in, weight_lb, birth_date, experience, school
                 FROM players WHERE player_id = %s""", (player_id,), one=True) or {}
    from datetime import date as _date
    bd, gd = bio.pop("birth_date", None), r.get("game_date")
    if isinstance(gd, str):
        gd = _date.fromisoformat(gd[:10])
    if isinstance(bd, str):
        bd = _date.fromisoformat(bd[:10])
    r["age"] = (gd.year - bd.year - ((gd.month, gd.day) < (bd.month, bd.day))
                if bd is not None and gd is not None else None)
    r["height_in"] = bio.get("height_in")
    r["weight_lb"] = bio.get("weight_lb")
    r["experience"] = bio.get("experience")
    r["school"] = bio.get("school")

    # The written case, generated from the same evidence packet the LLM reviewer reads.
    # Deterministic and free, so it is computed per request rather than stored -- there
    # is nothing to go stale and nothing to re-run when the scoring changes.
    #
    # `summary_source` names where the prose came from. When the L9 reviewer exists its
    # output will override this and the field will say so, which is the difference
    # between a reader trusting a sentence and a reader knowing whether to.
    try:
        pkt = packet.build(player_id, game_id=game_id)
        r["summary"] = summarize.summarize(pkt)
        r["summary_flags"] = [{"cause": c, "detail": d}
                              for c, d in summarize.flags(pkt)]
        r["summary_source"] = "rules"
    except Exception as e:                       # never take the case view down for prose
        r["summary"], r["summary_flags"] = None, []
        r["summary_source"] = f"unavailable ({type(e).__name__})"
    r["ai_summary"] = r["summary"]               # legacy key, same value

    r["season_log"] = q("""
        SELECT g.game_date, g.matchup, pg.minutes, pg.points,
               f.close_line, f.margin_vs_line
          FROM player_games pg
          JOIN games g ON g.game_id = pg.game_id
          LEFT JOIN player_game_features f
                 ON f.player_id = pg.player_id AND f.game_id = pg.game_id
         WHERE pg.player_id = %s
         ORDER BY g.game_date""", (player_id,))
    r["season_log_source"] = "postgres"
    return r


@app.get("/api/case/{player_id}/{game_id}")
def case(player_id: int, game_id: str):
    return _case_payload(player_id, game_id)


# --- shot chart + video ------------------------------------------------------
# Both ride the project's cache layer. Shots come ENTIRELY from the play-by-play
# CSVs load_pbp.py already fetched -- zero network. Video resolves a pbp event
# to an mp4 on NBA's CDN via videoeventsasset, cached forever after first hit.

def _clock(v) -> str:
    """'PT09M43.00S' -> '9:43'."""
    import re
    m = re.match(r"PT(\d+)M(\d+)", str(v))
    return f"{int(m.group(1))}:{m.group(2)}" if m else str(v)


@app.get("/api/case/{player_id}/{game_id}/shots")
def case_shots(player_id: int, game_id: str):
    """Field-goal attempts with court coordinates -> the shot chart.

    Reads `player_game_events` (L4c), not the pbp cache. The cache is 117MB of paid
    responses that deliberately does not ship, so a filesystem read here left this
    panel empty for everyone except the machine that ran the ingest.
    """
    rows = db.rows("""
        SELECT action_number, period, clock, x_legacy, y_legacy, shot_distance,
               shot_result, shot_value, description, video_available
          FROM player_game_events
         WHERE player_id = %(p)s AND game_id = %(g)s AND is_field_goal
           AND x_legacy IS NOT NULL AND y_legacy IS NOT NULL
         ORDER BY action_number""", {"p": player_id, "g": game_id})
    return {"shots": [{
        "action_number": int(e["action_number"]),
        "period": int(e["period"]),
        "clock": _clock(e["clock"]),
        "x": _fin(e["x_legacy"]), "y": _fin(e["y_legacy"]),
        "distance": _fin(e["shot_distance"]),
        "made": e["shot_result"] == "Made",
        "value": _fin(e["shot_value"]),
        "description": e["description"] or "",
        "video": bool(e["video_available"]),
    } for e in rows], "source": "play-by-play"}


# Books we actually hold quotes for, plus the two prediction markets the UI offers
# as coming. Listing them here rather than in the client keeps "what can be asked
# for" and "what can be served" in one place.
BOOKS_LIVE = ["fanduel", "draftkings", "williamhill_us"]
BOOKS_SOON = ["polymarket", "kalshi"]


@app.get("/api/case/{player_id}/{game_id}/line-history")
def case_line_history(player_id: int, game_id: str,
                      book: str = Query(default=None)):
    """The pre-tip price/line series for one player-game -- L3b's 'poll' rows plus the
    'open' and 'close' anchors, oldest first.

    ORDERED AND LABELLED BY offset_from_tip_sec, NOT by the requested timestamp. The
    historical endpoint snaps to a ~5-minute grid and returns a snapshot roughly four
    minutes earlier than asked for, so the requested time would put every point a
    constant distance from where it belongs and imply a precision the data lacks.

    One book at a time (default config.BOOK). Mixing books on one series would draw
    steps that are really disagreements between two markets.

    `books` in the response reports which of BOOKS_LIVE actually returned quotes for
    THIS player-game, so the UI can disable a chip that would open an empty chart
    rather than let the reader discover it by clicking.
    """
    book = book if book in BOOKS_LIVE else config.BOOK
    avail = {r["book"]: r["n"] for r in db.rows("""
        SELECT q.book, count(*) n
          FROM prop_quotes q JOIN odds_events e ON e.event_id = q.event_id
         WHERE e.game_id = %(g)s AND q.player_id = %(p)s
           AND q.market = %(m)s AND q.under_price IS NOT NULL
         GROUP BY q.book""", {"g": game_id, "p": player_id, "m": config.MARKET})}

    rows = db.rows("""
        SELECT DISTINCT ON (q.offset_from_tip_sec)
               q.offset_from_tip_sec AS sec,
               q.snapshot_role       AS role,
               q.line, q.over_price, q.under_price
          FROM prop_quotes q
          JOIN odds_events e ON e.event_id = q.event_id
         WHERE e.game_id = %(g)s AND q.player_id = %(p)s
           AND q.book = %(b)s AND q.market = %(m)s
           AND q.under_price IS NOT NULL
         -- DISTINCT ON keeps one row per instant. A book can quote alternate lines
         -- on the same player at the same moment; take the one nearest the close so
         -- the series follows the main market rather than hopping between ladders.
         ORDER BY q.offset_from_tip_sec DESC, q.snapshot_role = 'close' DESC, q.line
    """, {"g": game_id, "p": player_id, "b": book, "m": config.MARKET})

    out = []
    for r in rows:
        line, over, under = _fin(r["line"]), _fin(r["over_price"]), _fin(r["under_price"])
        # De-vigged implied probability of the UNDER: the book's two prices carry its
        # margin, so the raw 1/price overstates both sides. Normalising by their sum
        # removes it and gives a number that moves continuously -- the line itself is
        # quantised to half points and can sit flat through an entire session.
        p_under = None
        if over and under:
            iu, io = 1 / under, 1 / over
            p_under = round(iu / (iu + io) * 100, 2)
        out.append({"minutes_before_tip": round(r["sec"] / 60.0, 1),
                    "role": r["role"], "line": line,
                    "under_price": under, "over_price": over,
                    "p_under": p_under})
    return {"series": out, "book": book, "n": len(out),
            "books": [{"key": b, "n": avail.get(b, 0), "live": True} for b in BOOKS_LIVE]
                     + [{"key": b, "n": 0, "live": False} for b in BOOKS_SOON]}


@app.get("/api/case/{player_id}/{game_id}/plays")
def case_plays(player_id: int, game_id: str):
    """Every pbp event attributed to the player, in game order -- the case
    view's timeline beside the shot chart. `made` is tri-state: True/False for
    shots (FGs and FTs), None for everything else (subs, fouls, rebounds...).

    Reads `player_game_events` (L4c) rather than the pbp cache, for the same reason
    as the shot chart: the cache does not ship, the database does.
    """
    rows = db.rows("""
        SELECT action_number, period, clock, description, action_type,
               shot_result, video_available, score_away, score_home
          FROM player_game_events
         WHERE player_id = %(p)s AND game_id = %(g)s
         ORDER BY action_number""", {"p": player_id, "g": game_id})
    plays = []
    for e in rows:
        away, home = _fin(e["score_away"]), _fin(e["score_home"])
        sr = e["shot_result"]
        plays.append({
            "action_number": int(e["action_number"]),
            "period": int(e["period"]),
            "clock": _clock(e["clock"]),
            "description": e["description"] or "",
            "action_type": e["action_type"] or "",
            "made": (sr == "Made") if sr in ("Made", "Missed") else None,
            "video": bool(e["video_available"]),
            "score": (f"{int(away)}\u2013{int(home)}"
                      if away is not None and home is not None else None),
        })
    return {"plays": plays, "source": "play-by-play"}


@app.get("/api/video/{game_id}/{event_id}")
def video_asset(game_id: str, event_id: int):
    """mp4 + thumbnail for one pbp event, via stats.nba.com videoeventsasset."""
    import cache

    def _fetch():
        from nba_api.stats.endpoints import videoeventsasset
        v = videoeventsasset.VideoEventsAsset(
            game_id=game_id, game_event_id=event_id, timeout=15)
        return v.get_dict(), {"http_status": 200}

    try:
        payload, _src = cache.get(
            "nba", f"video_{game_id}_{event_id}.json", _fetch,
            api="nba", endpoint="videoeventsasset", fmt="json")
    except Exception as e:  # noqa: BLE001 -- surface as a gateway error
        raise HTTPException(502, f"video fetch failed: {e}")

    meta = (payload or {}).get("resultSets", {})
    urls = meta.get("Meta", {}).get("videoUrls", [])
    playlist = meta.get("playlist", [])
    if not urls:
        return {"url": None, "thumb": None, "description": None}
    u = urls[0]
    return {"url": u.get("lurl") or u.get("murl") or u.get("surl"),
            "thumb": u.get("lth") or u.get("mth") or u.get("sth"),
            "description": playlist[0].get("dsc") if playlist else None}


@app.get("/api/case/{player_id}/{game_id}/report.pdf")
def case_report(player_id: int, game_id: str):
    """One-case hand-off report for a reviewer outside the app. Built server-side from
    the same payload the screen renders, so the document and the dashboard cannot
    disagree."""
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    r = _case_payload(player_id, game_id)
    others = q(f"{BASE_SQL} WHERE s.player_id = %(pid)s AND s.score IS NOT NULL "
               f"ORDER BY s.score DESC LIMIT 10", {"season": SEASON, "pid": player_id})
    n_scored = q("SELECT count(*) AS n FROM player_game_scores WHERE score IS NOT NULL",
                 one=True)["n"]

    NAVY = colors.HexColor("#051C2C")
    RED = colors.HexColor("#C8102E")
    GRAY = colors.HexColor("#5A6572")
    LINE = colors.HexColor("#E4E7EB")

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold",
                        fontSize=17, textColor=NAVY, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=8.5,
                         textColor=GRAY, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=ss["Normal"], fontName="Helvetica-Bold",
                        fontSize=10, textColor=NAVY, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=8.5,
                          textColor=GRAY, leading=12)

    def tbl(rows, widths=None, header=False):
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                 ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
                 ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("TOPPADDING", (0, 0), (-1, -1), 3)]
        if header:
            style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                      ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY)]
        t.setStyle(TableStyle(style))
        return t

    def f(v, digits=2):
        if v is None:
            return "—"
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    rank_txt = (f"rank #{r['rank']} of {r['rank_all']:,} by score"
                if r["rank"] else
                f"NOT SHORTLISTED — eliminated by {r['cut_failed']}")
    d = r.get("deep") or {}

    def sig(v):
        """A σ with its sign spelled out, matching the screen's hover text."""
        return "—" if v is None else f"{v:+.2f}σ"

    # the same header facts the casehead line carries
    bio = " · ".join(x for x in [
        r.get("position"), r.get("tier"),
        f"age {r['age']}" if r.get("age") is not None else None,
        ("rookie" if r.get("experience") == 0
         else f"{r['experience']} yr exp" if r.get("experience") is not None else None),
        (f"{r['height_in'] // 12}'{r['height_in'] % 12}\""
         if r.get("height_in") is not None else None),
        f"{r['weight_lb']} lb" if r.get("weight_lb") is not None else None,
    ] if x)
    fs = r.get("final_score")
    final_txt = ("" if not fs else
                 f"final {fs['away']['team']} {fs['away']['pts']} — "
                 f"{fs['home']['team']} {fs['home']['pts']}"
                 + (f" ({'won' if fs['result'] == 'W' else 'lost'} by "
                    f"{abs(fs['away']['pts'] - fs['home']['pts'])})"
                    if fs.get("result") else ""))
    confirmed = f"{player_id}-{game_id}" in CONFIRMED_GAMES

    # NBA brand masthead: internal eyebrow + the split blue/red signature rule.
    NBA_BLUE = colors.HexColor("#1D428A")
    eyebrow = ParagraphStyle("eyebrow", parent=ss["Normal"], fontName="Helvetica-Bold",
                             fontSize=7, textColor=NBA_BLUE, spaceAfter=3)
    brand_rule = Table([["", ""]], colWidths=[3.6 * 72, 3.6 * 72], rowHeights=[2.2])
    brand_rule.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), NBA_BLUE),
        ("BACKGROUND", (1, 0), (1, 0), RED),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    brand_rule.hAlign = "LEFT"

    # Exhibit numbering mirrors the screen: Season, Performance, Shot profile,
    # Market, Motive -- so a reviewer can talk across the two artifacts.
    _ex = {"n": 0}

    def ex(title, sub_txt=None, val=None):
        _ex["n"] += 1
        head = f"Exhibit {_ex['n']} · {title}"
        if val is not None:
            head += f" — {val:.1f} / 100"
        if sub_txt:
            head += f"  ({sub_txt})"
        return Paragraph(head, h2)

    chips = " · ".join(x for x in [
        final_txt or None,
        "UNDER HIT" if r.get("under_hit") else None,
        "CONFIRMED — league/federal action, not a pipeline output"
        if confirmed else None,
    ] if x)

    story = [
        Paragraph("NBA LEAGUE INTEGRITY · INTERNAL", eyebrow),
        brand_rule,
        Spacer(1, 8),
        Paragraph("Game Integrity — Case Report", h1),
        Paragraph(f"{r['player']} · {r['game_date']} · {r['matchup']} · {bio}", sub),
        Paragraph(f"{chips} · {rank_txt} · {n_scored:,} scored player-games · "
                  f"2023–24", sub),

        Paragraph("Case summary", h2),
        tbl([
            ["Score", f"{r['score_100']:.1f} / 100", "Points line",
             f"{f(r['close_line'], 1)} "
             f"({'closing' if r.get('line_source', 'close') == 'close' else 'open — pulled'})"],
            ["Performance", f"{f(r['g_performance'], 1)} / 100", "Result",
             f"{r['points']} pts — {'under hit' if r['under_hit'] else 'over'}"],
            ["Market", f"{f(r['g_market'], 1)} / 100", "Independence tail",
             "—" if r["tail_pct"] is None else f"1 in {round(1 / r['tail_pct']):,}"],
            ["Motive", f"{f(r['g_motive'], 1)} / 100", "Salary",
             "unlisted (two-way/10-day)" if r["salary"] is None
             else f"${r['salary']:,.0f}"],
        ], widths=[80, 110, 110, 160]),
    ]

    # The written case, verbatim from the screen -- same paragraphs, same order.
    if r.get("summary"):
        story += [Spacer(1, 4)]
        story += [Paragraph(p, body) for p in r["summary"].split("\n\n")]

    # ---- Exhibit 1 · Season --------------------------------------------------
    propped = [g for g in r["season_log"]
               if g["close_line"] is not None and g["points"] is not None]
    if propped:
        unders = sum(1 for g in propped if g["points"] < g["close_line"])
        story += [
            ex("Season", "points vs closing line"),
            Paragraph(
                f"{len(propped)} propped games, {unders} unders "
                f"({100 * unders / len(propped):.0f}%), mean margin vs closing line "
                f"{sum(g['points'] - g['close_line'] for g in propped) / len(propped):+.1f} "
                f"pts. This game: {r['points']} pts against "
                f"{f(r['close_line'], 1)}.", body),
        ]

    # ---- Exhibit 2 · Performance --------------------------------------------
    story += [
        ex("Performance", "production + involvement vs his own season",
           r["g_performance"]),
        Paragraph("Box score — residual σ vs his own season (margin, rest, "
                  "back-to-back and pace regressed out per role tier)", body),
        tbl([["Stat", "Value", "Resid σ", "Stat", "Value", "Resid σ"]] + [
            ["Minutes", f(r["minutes"], 1), sig(r.get("minutes_resid_z")),
             "Touches", f(r["touches"], 0), sig(r.get("touches_resid_z"))],
            ["Points", str(r["points"]), sig(r.get("points_resid_z")),
             "Usage", "—" if r["usage_pct"] is None else f"{r['usage_pct'] * 100:.1f}%",
             sig(r.get("usage_pct_resid_z"))],
            ["FG attempts", str(r["fga"]), sig(r.get("fga_resid_z")),
             "Distance", "—" if r["distance"] is None else f"{r['distance']:.1f} mi",
             sig(r.get("distance_resid_z"))],
            ["Rebounds", str(r["rebounds"]), sig(r.get("rebounds_resid_z")),
             "TO ratio", f(r["turnover_ratio"], 1),
             sig(r.get("turnover_ratio_resid_z"))],
            ["Assists", str(r["assists"]), sig(r.get("assists_resid_z")),
             "", "", ""],
        ], widths=[70, 60, 55, 70, 60, 55], header=True),
    ]

    if d:
        story += [
            Paragraph("Hustle (no per-player baseline) & game context", body),
            tbl([
                ["Contested shots", f(d["contested_shots"], 0),
                 "Plus/minus", f(r["plus_minus"], 0)],
                ["Deflections", f(d["deflections"], 0), "Fouls", f(r["fouls"], 0)],
                ["Loose balls", f(d["loose_balls"], 0), "Stints",
                 f(d["n_stints"], 0)],
                ["Box-outs", f(d["box_outs"], 0), "Last off court",
                 "—" if d["last_out_sec"] is None
                 else f"{int(d['last_out_sec'] // 60)}:{int(d['last_out_sec'] % 60):02d}"],
                ["Passes", f(d["passes"], 0), "Ejected",
                 "YES" if d["ejected"] else "no"],
                ["Steals + blocks",
                 f((d["steals"] or 0) + (d["blocks"] or 0), 0), "", ""],
            ], widths=[90, 70, 100, 110]),
        ]

    # ---- Exhibit 3 · Shot profile -------------------------------------------
    shots = case_shots(player_id, game_id)["shots"]
    plays = case_plays(player_id, game_id)["plays"]
    story += [ex("Shot profile", "from the play-by-play")]
    if shots:
        made = sum(1 for s in shots if s["made"])
        threes = [s for s in shots if s["value"] == 3]
        story += [Paragraph(
            f"{made} of {len(shots)} field goals made · "
            f"{sum(1 for s in threes if s['made'])} of {len(threes)} from three · "
            f"mean attempt distance "
            f"{sum(s['distance'] or 0 for s in shots) / len(shots):.1f} ft.", body)]
    else:
        story += [Paragraph(
            "Zero field-goal attempts this game — for a player with a posted "
            "points line, that is itself part of the case." if plays
            else "No play-by-play cached for this game.", body)]
    if plays:
        story += [
            Paragraph("His whole night, in game order:", body),
            tbl([["Q · clock", "Play", "Score"]] + [
                [f"Q{p['period']} {p['clock']}", p["description"],
                 p["score"] or ""] for p in plays[:45]
            ], widths=[60, 300, 50], header=True),
        ]
        if len(plays) > 45:
            story += [Paragraph(f"… {len(plays) - 45} further events omitted.", body)]

    # ---- Exhibit 4 · Market --------------------------------------------------
    story += [
        ex("Market", "what the sportsbook saw; σ vs league, high = under-side "
           "pressure", r["g_market"]),
        tbl([["", "Value", "League σ"]] + [
            ["Points line", f"{f(r['close_line'], 1)}", sig(r.get("mk_p_line"))],
            ["Result", f"{r['points']} pts — "
             f"{'under hit' if r['under_hit'] else 'over'}", "—"],
            ["Under close", f(r["close_under"]), sig(r.get("mk_p_price"))],
            ["Line open → close",
             f"{f(r['open_line'], 1)} → {f(r['close_line'], 1)}",
             sig(r.get("mk_line_mv"))],
            ["Under price open → close",
             f"{f(r['open_under'])} → {f(r['close_under'])}",
             sig(r.get("mk_price_mv"))],
            ["Market components present", f"{r['n_market']} of 4", "—"],
        ], widths=[160, 140, 90], header=True),
        Paragraph("Decimal odds. Movement σ blank where no opening quote existed "
                  "12h out (~48% of rows).", body),
    ]

    # ---- Exhibit 5 · Motive --------------------------------------------------
    story += [
        ex("Motive", "what he stood to lose", r["g_motive"]),
        Paragraph(
            ("unlisted salary — a two-way / 10-day contract: the lowest-paid, "
             "most exposed profile, so it takes maximum motive weight"
             if r["salary"] is None else
             f"${r['salary']:,.0f}"
             + (f" — P{round((r['salary_pct'] or 0) * 100)} of league salaries, "
                f"inside the $20M motive gate"
                if r.get("salary_pct") is not None else "")), body),
        Paragraph(f"score = {BLOCK_W['performance']}·performance + "
                  f"{BLOCK_W['market']}·market + {BLOCK_W['motive']}·motive — the "
                  f"pipeline's own weights, equal within each block.", body),

        Paragraph("This player's other scored games (worst first)", h2),
        tbl([["Date", "Matchup", "Result", "Score /100", "Rank"]] + [
            [str(o["game_date"]), o["matchup"] or "—",
             f"{o['points']} / {f(o['close_line'], 1)}", f(o["score_100"], 1),
             f"#{o['rank']}" if o["rank"] else "cut"]
            for o in others
        ], widths=[70, 90, 70, 60, 50], header=True),
    ]

    caveat = ParagraphStyle("caveat", parent=body, textColor=RED, spaceBefore=12)
    story += [
        Spacer(1, 4),
        Paragraph("SCREENING FLAG — NOT A FINDING. Performance is measured against two "
                  "baselines (his own season and his role) plus the market's own "
                  "forecast; residuals regress out margin, rest, back-to-back and pace "
                  "per role tier. Low minutes is never filtered on: it is both the "
                  "strongest innocent explanation and a plausible signature. Injury is "
                  "not observed at all. Generated by game-integrity-v1 from "
                  "player_game_scores.", caveat),
    ]

    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=50, rightMargin=50,
                      topMargin=44, bottomMargin=44).build(story)
    fname = f"GI_{r['player'].replace(' ', '_')}_{r['game_date']}.pdf"
    return Response(content=buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/funnel")
def funnel():
    """Real counts from `cut_failed`, which records the FIRST cut each row failed.

    Reconstructed as a funnel: every stage shows how many rows were still alive when it
    ran, so the numbers match the stdout of export_candidates.py exactly rather than
    being re-derived with a second set of thresholds.
    """
    total = q("SELECT count(*) AS n FROM player_game_scores", one=True)["n"]
    all_pg = q("SELECT count(*) AS n FROM player_games", one=True)["n"]
    failed = q("""SELECT cut_failed, count(*) AS n FROM player_game_scores
                   WHERE cut_failed IS NOT NULL GROUP BY 1 ORDER BY 1""")
    # The season itself is stage zero: every logged player-game, DNPs included.
    # The gap to stage one is "no points prop was posted", not a cut we made.
    stages = [{"stage": "all player-games", "n": all_pg},
              {"stage": "propped player-games", "n": total,
               "removed": all_pg - total}]
    alive = total
    for f in failed:
        alive -= f["n"]
        stages.append({"stage": f["cut_failed"], "n": alive, "removed": f["n"]})
    return {"stages": stages}


@app.get("/api/player/{player_id}/flags")
def player_flags(player_id: int):
    """Every scored game for one player, worst first -- the roll-up the case view's
    rail shows, so a reviewer sees a pattern rather than one night."""
    rows = q(f"{BASE_SQL} WHERE s.player_id = %(pid)s AND s.score IS NOT NULL "
             f"ORDER BY s.score DESC", {"season": SEASON, "pid": player_id})
    if not rows:
        raise HTTPException(404, "player has no scored games")
    rows = [_decorate(r) for r in rows]
    return {"player": rows[0]["player"], "rows": rows}


@app.get("/api/calendar")
def calendar():
    """One cell per game day: how many propped games, the worst score of the day, and
    which player-game holds it (the click target)."""
    rows = q("""
        SELECT game_date, n, shortlist, max_score, confirmed,
               player, player_id, game_id
          FROM (
            SELECT s.game_date,
                   count(*) OVER (PARTITION BY s.game_date)                  AS n,
                   count(*) FILTER (WHERE s.in_shortlist)
                       OVER (PARTITION BY s.game_date)                       AS shortlist,
                   max(s.score_100) OVER (PARTITION BY s.game_date)          AS max_score,
                   bool_or(s.player_id::text || '-' || s.game_id
                           = ANY(%(confirmed)s))
                       OVER (PARTITION BY s.game_date)                       AS confirmed,
                   s.player, s.player_id, s.game_id,
                   row_number() OVER (PARTITION BY s.game_date
                                      ORDER BY s.score DESC NULLS LAST)      AS rn
              FROM player_game_scores s WHERE s.score_100 IS NOT NULL) t
         WHERE rn = 1 ORDER BY game_date""",
             {"confirmed": CONFIRMED_GAMES})
    for r in rows:
        r["review"] = r.pop("shortlist")
        r["date"] = str(r.pop("game_date"))
        r["max_score"] = round(r["max_score"], 2)
    return {"days": rows}


@app.get("/api/cloud")
def cloud(shortlist_only: bool = False):
    """EVERY propped player-game as a node in (performance, market, motive).

    All 15,498, not just the 4,810 that survive the cuts. The eliminated games are the
    context: without them the cloud shows a shape with no reference, and there is no way
    to see that the shortlist occupies one corner of the space rather than being
    scattered through it. They arrive with in_ledger = false, are drawn dim, and are not
    clickable, so the distinction stays visible without being distracting.

    A hovered dim node can say WHY it is dim: `cut` is the number of the cut that
    removed it, and `cuts` maps those numbers to labels ONCE at the top level rather
    than repeating a 40-character string on 10,684 nodes.

    The payload is deliberately narrow. Everything the case view needs is one click and
    one /api/case away; this endpoint carries only what a node must draw and what a
    tooltip must say, which keeps 15,494 rows around 2 MB instead of 6.
    """
    where = "s.in_shortlist AND " if shortlist_only else ""
    rows = q(f"""
        SELECT s.player, s.game_date, s.player_id, s.game_id,
               s.performance_100 AS performance, s.market_100 AS market,
               s.motive_100 AS motive, s.score_100, s.rank, s.in_shortlist,
               s.cut_failed, s.points, s.close_line,
               (z.p_price * z.p_line) AS tail_pct
          FROM player_game_scores s
          JOIN player_game_z z USING (player_id, game_id)
         WHERE {where}s.performance_100 IS NOT NULL
           AND s.market_100 IS NOT NULL AND s.motive_100 IS NOT NULL
         -- Ordered only so the response is reproducible. With no ORDER BY the rows
         -- arrive in heap order, which differs between a freshly restored table and
         -- one updated in place -- so two databases holding identical data returned
         -- different payloads. The cloud is a scatter; order changes nothing on screen.
         ORDER BY s.player_id, s.game_id""")
    cuts = {}
    out = []
    for r in rows:
        lab = r.pop("cut_failed")
        cut = None
        if lab:
            cut = int(lab.split()[0])
            cuts.setdefault(cut, lab)
        out.append({
            "player": r["player"], "game_date": str(r["game_date"]),
            "player_id": r["player_id"], "game_id": r["game_id"],
            "performance": round(r["performance"], 1),
            "market": round(r["market"], 1),
            "motive": round(r["motive"], 1),
            "score_100": round(r["score_100"], 1) if r["score_100"] is not None else None,
            "rank": r["rank"], "in_ledger": bool(r["in_shortlist"]), "cut": cut,
            "points": r["points"], "line": r["close_line"],
            "tail_pct": round(r["tail_pct"], 5) if r["tail_pct"] is not None else None,
        })
    return {"nodes": out, "cuts": cuts,
            "shortlist": sum(1 for r in out if r["in_ledger"])}


# PINNED TO THE TOP OF THE PLAYER LIST, and shown with their real position.
#
# These are the two publicly-known cases. They are pinned for demonstration -- so the
# list can be pointed at without scrolling -- NOT because the ranking found them there.
# Each pinned row renders its TRUE position beside the name, so the promotion is visible
# rather than disguised. Nothing else about the ranking treats them specially, and
# removing this list changes only the order they appear in.
#
# The alternative was to pick a metric that happened to rank them first. That is the
# same move that drove the supervised fit to AUC 0.087, and it would have made the
# leaderboard a claim about two players rather than a measurement over 378.
PINNED = ["Malik Beasley", "Jontay Porter"]

# How many of a player's worst games the ranking averages. 3 is a compromise: 1 is a
# single night and swings on one freak game (it puts Jaylen Nowell first on a 3-game
# sample), while 5 and above start averaging in ordinary games and bury anyone with a
# short season -- at K=5 Porter falls to #94 on 7 propped games, at K=8 to #118.
TOP_K = 3


@app.get("/api/players")
def players(limit: int = Query(70, le=400), min_games: int = 3):
    """Player-level leaderboard, each row carrying that player's WORST game as the
    click target.

    RANKED BY THE MEAN OF HIS TOP-3 WORST GAMES. A player is interesting if he has
    several bad nights, not one -- a single-game maximum rewards whoever had the most
    extreme evening in a three-game sample, and a season mean rewards whoever played
    fewest games. Averaging the worst three asks for a pattern while still being a peak
    measure.

    WHAT THIS METRIC COSTS, stated because the choice is not neutral:

        metric              Beasley   Porter
        best single game        #15      #58
        MEAN OF TOP 3           #14      #59      <- this one
        mean of top 5           #16      #94
        season mean             #71       #6
        share in top 5%        #106       #5

    Porter played 7 propped games to Beasley's 77. Every rate metric rewards the small
    sample and every peak metric rewards the large one; no definition puts both near the
    top. Rather than choose a metric to fit two known answers, the two are PINNED (see
    PINNED above) with their real positions on display.

    min_games = 3: below that a leaderboard position is one game with no context.
    """
    rows = q("""
        WITH ranked AS (
            SELECT s.*, row_number() OVER (PARTITION BY s.player_id
                                           ORDER BY s.score DESC) AS rn
              FROM player_game_scores s WHERE s.score IS NOT NULL),
        agg AS (
            SELECT player_id, player,
                   count(*)                                          AS n_games,
                   count(*) FILTER (WHERE in_shortlist)              AS n_shortlist,
                   avg(score_100) FILTER (WHERE rn <= %(k)s)         AS topk,
                   max(score_100)                                    AS best,
                   min(rank_all)                                     AS best_rank
              FROM ranked GROUP BY player_id, player
            HAVING count(*) >= %(min_games)s),
        worst AS (
            -- the click target: his single worst game, by the same score the list ranks
            SELECT player_id, game_id, game_date, score_100, rank, rank_all,
                   points, close_line, minutes, in_shortlist
              FROM ranked WHERE rn = 1)
        SELECT a.*, w.game_id, w.game_date, w.score_100 AS worst_score,
               w.rank AS worst_rank, w.points, w.close_line, w.minutes,
               w.in_shortlist AS worst_in_shortlist,
               -- player_id breaks ties. 8 players share an exact topk value, and
               -- without a tiebreaker their order is whatever the engine happens to
               -- produce: Postgres and DuckDB disagreed, and Postgres alone is free
               -- to change its mind after a re-plan. A leaderboard that reorders
               -- between runs looks like the model changed when nothing did.
               row_number() OVER (ORDER BY a.topk DESC, a.player_id) AS position
          FROM agg a JOIN worst w USING (player_id)
         ORDER BY a.topk DESC, a.player_id
    """, {"k": TOP_K, "min_games": min_games})

    for r in rows:
        r["pinned"] = r["player"] in PINNED
        r["topk"] = round(r["topk"], 2) if r["topk"] is not None else None

    # Pinned rows first, in PINNED order, then everyone else by position. `position` is
    # untouched by this, so a pinned row still reports where the metric actually put it.
    pinned = [r for n in PINNED for r in rows if r["player"] == n]
    rest = [r for r in rows if not r["pinned"]][: max(0, limit - len(pinned))]
    return {"rows": pinned + rest, "top_k": TOP_K,
            "pinned": PINNED, "total_players": len(rows)}


@app.get("/api/isolation")
def isolation(limit: int = Query(15, le=100)):
    """RETIRED. Isolation Forest was tested in four configurations -- pooled, pooled
    without motive, per line band, and on the cut survivors -- and lost to the weighted
    sum every time. The reason is structural: it scores RARITY, and the target class
    here is a MODE (349 zero-point games form a dense cluster, not a sparse tail). It
    also inverts the motive axis, because being underpaid is not rare among games that
    fail badly against a line.

    The route stays so the dashboard degrades to an empty panel with an explanation
    instead of a 404. See experiments/README.md.
    """
    return {"rows": [], "retired": True,
            "reason": "Isolation Forest lost to the weighted sum in all four "
                      "configurations tested; it scores rarity and the target class "
                      "is a mode."}
