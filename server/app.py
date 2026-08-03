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
from psycopg.rows import dict_row

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                          # noqa: E402
import db                                              # noqa: E402
from standardize import BLOCK_W, PERF_W, MARKET_W      # noqa: E402

# The three blocks, in the order the UI shows them. Imported rather than copied, so a
# re-weighting in L5 re-labels the dashboard on next restart with no second edit.
WEIGHTS = dict(BLOCK_W)

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
    """Run a query, return dicts with NaN/inf scrubbed to None."""
    with db.connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
    out = [{k: _fin(v) for k, v in r.items()} for r in rows]
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
    r["g_performance"] = r.get("performance")
    r["g_market"] = r.get("market")
    r["g_motive"] = r.get("motive")
    r["line"] = r.get("close_line")
    r["line_source"] = "close"
    r["line_pulled"] = False

    # prod_z was mean(points_z, assists_z, rebounds_z, fga_z) under the old scheme. Game
    # Score subsumes all four -- and six more -- so game_z is the same claim, better
    # measured. Aliased rather than renamed in the client because CaseView's sentence
    # ("production Xσ ... below his own season") stays true: game_z IS the own-season
    # baseline. effort_z kept its name and its meaning.
    r["prod_z"] = r.get("game_z")

    # The 0-100 scale, aliased to the names the UI reads. score_100 is the PERCENTILE of
    # `score` over all propped games -- "worse than X% of them" -- so 99.0 means 155
    # games scored higher, not that the game is 99% of some maximum.
    r["score_pct100"] = r.get("score_100")
    # `score` is the raw z-scale value, kept for provenance. The UI shows score_100, and
    # `rank` is ordered by score_100 -- so sorting by `score` would produce a list whose
    # order disagrees with the rank column printed beside it.
    return r


@app.get("/api/summary")
def summary(review_threshold: float = Query(1.0)):
    """Headline counts plus the score distribution.

    review_threshold is an ABSOLUTE score, not a percentile. A percentile cutoff would
    be tautological -- the top quarter is always a quarter -- and the point of the number
    is that the score distribution is concentrated, so a high bar is genuinely rare:

        score >= 0.8    849 of 15,494       >= 1.1    205
        score >= 1.0    350                 >= 1.3     52

    The scale changed with the scoring rework. The old composite was a weighted mean of
    percentiles bounded to 0-1, where 0.75 was a sensible bar; the current score is a
    weighted mean of z-scores running -1.77 to +1.61, so the default moved to 1.0 to
    keep the tail a similar size.
    """
    agg = q("""
        SELECT count(*)                                        AS scored,
               count(*) FILTER (WHERE in_shortlist)            AS shortlist,
               count(*) FILTER (WHERE has_listed_salary IS FALSE) AS unlisted_salary,
               count(*) FILTER (WHERE score >= %s)             AS review_tail,
               count(*) FILTER (WHERE score >= %s AND in_shortlist)
                                                               AS review_tail_shortlist,
               max(score) AS top_score,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY score) AS median_score,
               min(score) AS lo, max(score) AS hi
          FROM player_game_scores WHERE score IS NOT NULL
    """, (review_threshold, review_threshold), one=True)

    # 20 equal-width bins across the observed range. Data-driven rather than fixed at
    # 0-1: the score is not a probability and pinning the axis would clip both tails.
    lo, hi = agg["lo"], agg["hi"]
    width = (hi - lo) / 20 or 1
    hist = q("""
        SELECT width_bucket(score, %s, %s, 20) AS b, count(*) AS n
          FROM player_game_scores WHERE score IS NOT NULL GROUP BY b ORDER BY b
    """, (lo, hi))
    counts = {r["b"]: r["n"] for r in hist}
    return {
        "scored": agg["scored"],
        "shortlist": agg["shortlist"],
        "review_tail": agg["review_tail"],
        "review_tail_shortlist": agg["review_tail_shortlist"],
        "review_threshold": review_threshold,
        "pulled_and_played": 0,          # pulled lines are excluded upstream
        "unlisted_salary": agg["unlisted_salary"],
        "top_score": round(agg["top_score"], 4),
        "scale_note": ("score_100 is the percentile of score over all propped games: "
                       "99.0 means 1% scored higher. The raw score is a weighted mean "
                       "of z-scores and is population-independent; the percentile is "
                       "not."),
        "median_score": round(agg["median_score"], 4),
        "weights": WEIGHTS,
        "histogram": [{"lo": round(lo + i * width, 4),
                       "hi": round(lo + (i + 1) * width, 4),
                       "n": counts.get(i + 1, 0)} for i in range(20)],
    }


@app.get("/api/watchlist")
def watchlist(q_: str = Query("", alias="q"), sort: str = "rank",
              dir: str = "asc", limit: int = Query(50, le=500), offset: int = 0,
              shortlist_only: bool = True):
    if sort not in SORTABLE:
        raise HTTPException(400, f"sort must be one of {sorted(SORTABLE)}")
    where = ["s.score_100 IS NOT NULL"]
    params = {"season": SEASON, "limit": limit, "offset": offset}
    if shortlist_only:
        where.append("s.in_shortlist")
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

    # Reserved for a later LLM pass. None tells the client to render the empty slot
    # rather than hide the panel, so the layout stays honest about what is next.
    r["ai_summary"] = None

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

    story = [
        Paragraph("Game Integrity — Case Report", h1),
        Paragraph(f"{r['player']} · {r['matchup']} · {r['game_date']} · "
                  f"{r['position'] or '—'} · {r['tier'] or '—'} tier · "
                  f"{rank_txt} · {n_scored:,} scored player-games · 2023–24", sub),

        Paragraph("Summary", h2),
        tbl([
            ["Score", f(r["score"], 3), "Points line", f(r["close_line"], 1)],
            ["Performance", f(r["performance"], 3), "Result",
             f"{r['points']} pts — {'under hit' if r['under_hit'] else 'over'}"],
            ["Market", f(r["market"], 3), "Independence tail",
             "—" if r["tail_pct"] is None else f"1 in {round(1 / r['tail_pct']):,}"],
            ["Motive", f(r["motive"], 3), "Salary",
             "unlisted (two-way/10-day)" if r["salary"] is None
             else f"${r['salary']:,.0f}"],
        ], widths=[80, 110, 110, 160]),
        Paragraph(f"score = {BLOCK_W['performance']}·performance + "
                  f"{BLOCK_W['market']}·market + {BLOCK_W['motive']}·motive — the "
                  f"pipeline's own weights, equal within each block.", body),

        Paragraph("Performance components — σ vs his own season and vs his role", h2),
        tbl([["Component", "Own", "Role", "Component", "Own", "Role"]] + [
            ["Game Score", f(r["game_z"]), f(r["game_z_tier"]),
             "Involvement", f(r["effort_z"]), f(r["effort_z_tier"])],
            ["Shortfall (1 − pts/line)", f(r["shortfall"], 3),
             f"{f(r['shortfall_z'])} σ", "Minutes", f(r["minutes"], 1),
             f(r.get("minutes_resid_z"))],
        ], widths=[110, 55, 55, 90, 55, 55], header=True),

        Paragraph("Box score — residual σ vs his own season (context regressed out)", h2),
        tbl([["Stat", "Value", "Resid σ", "Stat", "Value", "Resid σ"]] + [
            ["Points", str(r["points"]), f(r.get("points_resid_z")),
             "Touches", f(r["touches"], 0), f(r.get("touches_resid_z"))],
            ["FG attempts", str(r["fga"]), f(r.get("fga_resid_z")),
             "Usage", "—" if r["usage_pct"] is None else f"{r['usage_pct'] * 100:.1f}%",
             f(r.get("usage_pct_resid_z"))],
            ["Rebounds", str(r["rebounds"]), f(r.get("rebounds_resid_z")),
             "Distance", "—" if r["distance"] is None else f"{r['distance']:.1f} mi",
             f(r.get("distance_resid_z"))],
            ["Assists", str(r["assists"]), f(r.get("assists_resid_z")),
             "TO ratio", f(r["turnover_ratio"], 1), f(r.get("turnover_ratio_resid_z"))],
        ], widths=[70, 60, 55, 70, 60, 55], header=True),
    ]

    if d:
        story += [
            Paragraph("Hustle & exit anatomy", h2),
            tbl([
                ["Contested shots", f(d["contested_shots"], 0), "Stints",
                 f(d["n_stints"], 0)],
                ["Deflections", f(d["deflections"], 0), "Last off court",
                 "—" if d["last_out_sec"] is None
                 else f"{int(d['last_out_sec'] // 60)}:{int(d['last_out_sec'] % 60):02d}"],
                ["Loose balls", f(d["loose_balls"], 0), "Competitive pts",
                 f(d["points_competitive"], 0)],
                ["Passes", f(d["passes"], 0), "Garbage pts", f(d["points_garbage"], 0)],
                ["Box-outs", f(d["box_outs"], 0), "Ejected",
                 "YES" if d["ejected"] else "no"],
            ], widths=[90, 70, 100, 110]),
        ]

    story += [
        Paragraph("Sportsbook", h2),
        tbl([
            ["Line open → close",
             f"{f(r['open_line'], 1)} → {f(r['close_line'], 1)}", "Line movement",
             "no opening line" if r["line_move_pct"] is None
             else f"{r['line_move_pct'] * 100:.1f}%"],
            ["Under price open → close",
             f"{f(r['open_under'])} → {f(r['close_under'])}", "Under-price movement",
             "no opening line" if r["under_move_pct"] is None
             else f"{r['under_move_pct'] * 100:.1f}%"],
            ["Price moved, line held",
             "—" if r["price_only_move"] is None
             else f"{r['price_only_move'] * 100:.1f}%",
             "Market components present", f"{r['n_market']} of 4"],
        ], widths=[120, 110, 120, 110]),

        Paragraph(f"This player's other scored games (worst first)", h2),
        tbl([["Date", "Matchup", "Result", "Score", "Rank"]] + [
            [str(o["game_date"]), o["matchup"] or "—",
             f"{o['points']} / {f(o['close_line'], 1)}", f(o["score"], 3),
             f"#{o['rank']}" if o["rank"] else "cut"]
            for o in others
        ], widths=[70, 90, 70, 60, 50], header=True),
    ]

    propped = [g for g in r["season_log"]
               if g["close_line"] is not None and g["points"] is not None]
    if propped:
        unders = sum(1 for g in propped if g["points"] < g["close_line"])
        story += [Paragraph(
            f"Season log: {len(propped)} propped games, {unders} unders "
            f"({100 * unders / len(propped):.0f}%), mean margin vs closing line "
            f"{sum(g['points'] - g['close_line'] for g in propped) / len(propped):+.1f} "
            f"pts.", body)]

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
    failed = q("""SELECT cut_failed, count(*) AS n FROM player_game_scores
                   WHERE cut_failed IS NOT NULL GROUP BY 1 ORDER BY 1""")
    stages = [{"stage": "propped player-games", "n": total}]
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
        SELECT game_date, n, shortlist, max_score, player, player_id, game_id
          FROM (
            SELECT s.game_date,
                   count(*) OVER (PARTITION BY s.game_date)                  AS n,
                   count(*) FILTER (WHERE s.in_shortlist)
                       OVER (PARTITION BY s.game_date)                       AS shortlist,
                   max(s.score) OVER (PARTITION BY s.game_date)              AS max_score,
                   s.player, s.player_id, s.game_id,
                   row_number() OVER (PARTITION BY s.game_date
                                      ORDER BY s.score DESC NULLS LAST)      AS rn
              FROM player_game_scores s WHERE s.score IS NOT NULL) t
         WHERE rn = 1 ORDER BY game_date""")
    for r in rows:
        r["review"] = r.pop("shortlist")
        r["date"] = str(r.pop("game_date"))
        r["max_score"] = round(r["max_score"], 4)
    return {"days": rows}


@app.get("/api/cloud")
def cloud():
    """Every shortlisted game as a node in (performance, market, motive)."""
    rows = q("""
        SELECT s.player, s.game_date, s.player_id, s.game_id,
               s.performance, s.market, s.motive, s.rank,
               (z.p_price * z.p_line) AS tail_pct
          FROM player_game_scores s
          JOIN player_game_z z USING (player_id, game_id)
         WHERE s.in_shortlist
           AND s.performance IS NOT NULL AND s.market IS NOT NULL
           AND s.motive IS NOT NULL""")
    for r in rows:
        r["game_date"] = str(r["game_date"])
        r["in_ledger"] = True          # every node here is on the shortlist
        for k in ("performance", "market", "motive"):
            r[k] = round(r[k], 4)
    return {"nodes": rows}


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
