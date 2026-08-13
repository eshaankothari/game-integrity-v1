"""L8: assemble one player-game into a single evidence packet for an LLM.

This is the ingestion problem, and it is mostly a SUBSETTING problem. Everything the
project knows about one night lives across six tables and a 247 KB play-by-play file.
Handing all of it to a model is both unaffordable and counterproductive -- the signal
gets buried in 480 events about other players.

    full play-by-play for one game        480 events   247.2 KB
    the player's own events + his subs     ~50 events     5.7 KB
    the player's own events only            3 events     0.3 KB

WHAT GOES IN, AND WHY EACH SECTION EARNS ITS TOKENS

    identity     who, when, against whom, role, salary. Needed to make sense of
                 everything else -- "3 points" means nothing without "usually scores 11".
    baseline     his season averages. The model cannot judge a night without the norm,
                 and giving it the norm is cheaper than giving it 70 other games.
    performance  the five z-components and the three blocks, IN RAW UNITS as well as
                 z, because a model reasons better about "3 points against a 9.5 line"
                 than about "shortfall_z +1.93".
    market       open/close line and price, and the movement between them.
    context      rest, back-to-back, home, altitude, final margin, blowout flag. These
                 are the innocent explanations, and they must be present or the model
                 will invent their absence.
    pbp          HIS events plus the substitutions that bracket them, with the score at
                 each moment. This is what tells you he was pulled at 2:08 of the first
                 and never came back -- which no aggregate carries.
    health       DNP-injury windows either side. Not yet populated (see L4c).
    news         external context. Not yet populated (see L9).

WHAT IS DELIBERATELY LEFT OUT

    THE SCORE AND THE RANK. `blind=True` (the default) omits score, rank, percentile and
    every block value. A model told that a game ranks #3 of 15,498 will find reasons it
    ranks #3 -- that is not analysis, it is the prompt talking to itself. The packet is
    the evidence; the score is a separate opinion about the evidence, and the whole point
    of asking a model is to get an INDEPENDENT one.

    Set blind=False only for a briefing that is meant to restate a conclusion, never for
    a classification or a devil's-advocate pass.

    python -m pipeline.llm_review.packet 1629007 2024-03-20          # print one packet
    python -m pipeline.llm_review.packet --measure 200               # size distribution over the top N
"""
import argparse
import json
import sys
import warnings

import pandas as pd

from pipeline.core import config
from pipeline.core import db

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

# Events for OTHER players are dropped, but substitutions are kept regardless of who
# they involve: a player's night is defined as much by when he was taken off as by what
# he did, and the sub that replaced him names the coach's choice.
PBP_KEEP_COLS = ["period", "clock", "description", "scoreHome", "scoreAway"]

SQL = """
SELECT s.player, s.game_date, s.matchup, s.tier, s.minutes, s.points,
       s.close_line, s.close_under, s.shortfall, s.margin_vs_line,
       s.salary, s.has_listed_salary, s.under_hit, s.ejected, s.ejected_alone,
       sal.salary_pct,
       s.score, s.score_100, s.rank, s.rank_all, s.in_shortlist, s.cut_failed,
       s.performance, s.market, s.motive,
       f.position, f.fga, f.rebounds, f.assists, f.usage_pct, f.turnover_ratio,
       f.distance, f.touches, f.passes, f.game_score,
       f.open_line, f.open_under, f.line_move_pct, f.under_move_pct,
       f.price_only_move, f.n_player_games,
       z.game_z, z.effort_z, z.game_z_tier, z.effort_z_tier, z.shortfall_z,
       z.p_price, z.p_line, z.n_market,
       pg.steals, pg.blocks, pg.turnovers, pg.fouls, pg.plus_minus, pg.started,
       pg.fgm, pg.fg3m, pg.fg3a, pg.ftm, pg.fta,
       pb.n_stints, pb.first_in_sec, pb.last_out_sec,
       pb.points_competitive, pb.points_garbage,
       gc.is_home, gc.rest_days, gc.is_b2b, gc.altitude_ft, gc.margin,
       gm.game_margin
  FROM player_game_scores s
  JOIN player_game_features f USING (player_id, game_id)
  JOIN player_game_z        z USING (player_id, game_id)
  JOIN player_games        pg USING (player_id, game_id)
  LEFT JOIN player_game_pbp pb USING (player_id, game_id)
  LEFT JOIN player_salaries sal ON sal.player_id = s.player_id
                              AND sal.season = %(season)s
  LEFT JOIN game_context   gc ON gc.game_id = s.game_id AND gc.team_id = pg.team_id
  LEFT JOIN (SELECT game_id, max(pts) - min(pts) AS game_margin
               FROM (SELECT game_id, team_id, sum(points) AS pts
                       FROM player_games WHERE points IS NOT NULL GROUP BY 1, 2) t
              GROUP BY game_id HAVING count(*) = 2) gm ON gm.game_id = s.game_id
 WHERE s.player_id = %(pid)s AND s.game_id = %(gid)s
"""

# His own season, so the model has a norm to judge against without being handed 70 rows.
BASE_SQL = """
SELECT count(*) AS n_propped,
       round(avg(points), 1)      AS avg_points,
       round(avg(minutes), 1)     AS avg_minutes,
       round(avg(close_line), 1)  AS avg_line,
       round(avg(shortfall), 3)   AS avg_shortfall,
       sum((points < close_line)::int) AS unders
  FROM player_game_scores WHERE player_id = %(pid)s AND score IS NOT NULL
"""

# The bar the numbers are being measured against. Constants, not queried, because they
# describe the population and would otherwise cost a query per packet.
LEAGUE_NOTE = ("shortfall = 1 - points/line clipped to [0,1]; 1.000 means he scored "
               "zero. game_z and effort_z are standard deviations against HIS OWN "
               "season; game_z_tier and effort_z_tier are against every player in his "
               "role. Negative = below that baseline.")


def _num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, bool)):
        return v
    f = float(v)
    return round(f, 4)


def build(player_id, game_date=None, game_id=None, blind=True, pbp_mode="own+subs"):
    """One packet. `blind` omits the score, rank and blocks -- see the module docstring."""
    with db.connect() as conn:
        if game_id is None:
            gid = pd.read_sql(
                "SELECT game_id FROM player_game_scores WHERE player_id=%(p)s "
                "AND game_date=%(d)s", conn, params={"p": player_id, "d": game_date})
            if gid.empty:
                raise SystemExit(f"no propped game for player {player_id} on {game_date}")
            game_id = str(gid.game_id.iloc[0])
        r = pd.read_sql(SQL, conn, params={"pid": player_id, "gid": game_id,
                                           "season": config.SEASON})
        if r.empty:
            raise SystemExit(f"no packet for {player_id} / {game_id}")
        r = r.iloc[0]
        base = pd.read_sql(BASE_SQL, conn, params={"pid": player_id}).iloc[0]

    p = {
        "identity": {
            "player": r.player, "game_date": str(r.game_date), "matchup": r.matchup,
            "position": r.position, "role_tier": r.tier,
            "started": bool(r.started) if r.started is not None else None,
            "salary_usd": _num(r.salary),
            "salary_listed": bool(r.has_listed_salary) if r.has_listed_salary is not None else None,
            # Where that sits among players, 0-1. A description of the contract, not the
            # pipeline's motive score -- so it survives the blind packet.
            "salary_percentile": _num(r.salary_pct),
        },
        "his_season": {
            "propped_games": _num(base.n_propped), "avg_points": _num(base.avg_points),
            "avg_minutes": _num(base.avg_minutes), "avg_line": _num(base.avg_line),
            "unders": _num(base.unders), "avg_shortfall": _num(base.avg_shortfall),
        },
        "this_game": {
            "minutes": _num(r.minutes), "points": _num(r.points),
            "fgm_fga": [_num(r.fgm), _num(r.fga)], "fg3m_fg3a": [_num(r.fg3m), _num(r.fg3a)],
            "ftm_fta": [_num(r.ftm), _num(r.fta)],
            "rebounds": _num(r.rebounds), "assists": _num(r.assists),
            "steals": _num(r.steals), "blocks": _num(r.blocks),
            "turnovers": _num(r.turnovers), "fouls": _num(r.fouls),
            "plus_minus": _num(r.plus_minus), "game_score": _num(r.game_score),
            "usage_pct": _num(r.usage_pct), "touches": _num(r.touches),
            "passes": _num(r.passes), "distance_miles": _num(r.distance),
        },
        "vs_baselines": {
            "game_z_own": _num(r.game_z), "effort_z_own": _num(r.effort_z),
            "game_z_role": _num(r.game_z_tier), "effort_z_role": _num(r.effort_z_tier),
            "shortfall": _num(r.shortfall), "shortfall_z_league": _num(r.shortfall_z),
            "_note": LEAGUE_NOTE,
        },
        "market": {
            "close_line": _num(r.close_line), "close_under_price": _num(r.close_under),
            "open_line": _num(r.open_line), "open_under_price": _num(r.open_under),
            "line_move_pct": _num(r.line_move_pct),
            "under_price_move_pct": _num(r.under_move_pct),
            "price_moved_line_held_pct": _num(r.price_only_move),
            "points_vs_line": _num(r.margin_vs_line),
            "under_hit": bool(r.under_hit) if r.under_hit is not None else None,
            "_note": "No opening line was posted for 48% of games; nulls here mean "
                     "unobserved, not unchanged.",
        },
        "game_context": {
            "is_home": bool(r.is_home) if r.is_home is not None else None,
            "rest_days": _num(r.rest_days),
            "back_to_back": bool(r.is_b2b) if r.is_b2b is not None else None,
            "altitude_ft": _num(r.altitude_ft),
            "team_margin": _num(r.margin), "final_game_margin": _num(r.game_margin),
            "blowout": (None if r.game_margin is None else bool(r.game_margin >= 20)),
            "points_in_competitive_time": _num(r.points_competitive),
            "points_in_garbage_time": _num(r.points_garbage),
            "ejected": bool(r.ejected), "ejected_alone": bool(r.ejected_alone),
        },
        "on_off_court": {
            "n_stints": _num(r.n_stints),
            "first_on_court_sec": _num(r.first_in_sec),
            "last_off_court_sec": _num(r.last_out_sec),
            "_note": "Seconds from tip. A regulation game is 2,880. Leaving early and "
                     "not returning is visible here and in no aggregate.",
        },
        # Placeholders so the contract is stable before the layers exist. An LLM that
        # sees an absent key guesses; one that sees an explicit null does not.
        "health": None,   # L4c: DNP-injury windows either side
        "news": None,     # L9: injury reports and beat coverage, with citations
    }

    if pbp_mode != "none":
        p["play_by_play"] = _pbp(game_id, player_id, pbp_mode)

    if not blind:
        p["pipeline_opinion"] = {
            "score_0_100": _num(r.score_100), "rank": _num(r.rank),
            "rank_all": _num(r.rank_all), "in_shortlist": bool(r.in_shortlist),
            "cut_failed": r.cut_failed,
            "performance": _num(r.performance), "market_block": _num(r.market),
            "motive": _num(r.motive),
        }
    return p


def _pbp(game_id, player_id, mode):
    """His events, plus the substitutions that bracket them.

    Subs are kept regardless of who they involve: being taken off is a coach's decision
    about this player, and the sub that replaced him names it. Everything else -- 400-odd
    events about other players -- is dropped, which is the difference between 247 KB and
    6 KB.
    """
    path = config.NBA_CACHE_V1 / f"pbp_{game_id}.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path, dtype={"gameId": str})
    own = d[d.personId == player_id]
    if mode == "own":
        keep = own
    else:
        subs = d[d.description.fillna("").str.contains("SUB", case=False)]
        keep = pd.concat([own, subs]).drop_duplicates("actionNumber")
    keep = keep.sort_values("actionNumber")
    return [{k: (None if pd.isna(v) else v) for k, v in e.items()}
            for e in keep[PBP_KEEP_COLS].to_dict("records")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("player_id", nargs="?", type=int)
    ap.add_argument("game_date", nargs="?")
    ap.add_argument("--show-score", action="store_true",
                    help="include the pipeline's own opinion (default: blind)")
    ap.add_argument("--pbp", default="own+subs", choices=["own", "own+subs", "none"])
    ap.add_argument("--measure", type=int, metavar="N",
                    help="packet-size distribution over the top N shortlisted games")
    a = ap.parse_args()

    if a.measure:
        with db.connect() as conn:
            t = pd.read_sql("""SELECT player_id, game_id, player, minutes
                                 FROM player_game_scores WHERE in_shortlist
                                ORDER BY rank LIMIT %(n)s""",
                            conn, params={"n": a.measure})
        sizes = []
        for _, x in t.iterrows():
            try:
                s = len(json.dumps(build(int(x.player_id), game_id=str(x.game_id),
                                         pbp_mode=a.pbp)))
                sizes.append(s)
            except SystemExit:
                pass
        s = pd.Series(sizes) / 1000
        tok = s * 1000 / 3.6          # ~3.6 chars per token for dense JSON
        print(f"packets {len(s)}   pbp={a.pbp}")
        print(f"   KB      min {s.min():>6.1f}   median {s.median():>6.1f}   "
              f"p90 {s.quantile(.9):>6.1f}   max {s.max():>6.1f}")
        print(f"   tokens  min {tok.min():>6,.0f}   median {tok.median():>6,.0f}   "
              f"p90 {tok.quantile(.9):>6,.0f}   max {tok.max():>6,.0f}")
        tot = tok.sum()
        print(f"\n   {len(s)} packets = {tot/1e6:.2f}M input tokens")
        for name, inp, outp in (("Haiku 4.5", 1.00, 5.00), ("Sonnet 5", 3.00, 15.00)):
            cost = tot / 1e6 * inp + len(s) * 500 / 1e6 * outp
            print(f"      {name:<11} ~${cost:,.2f}   "
                  f"(and ~${cost * 4810 / max(len(s),1):,.0f} for all 4,810)")
        return

    if a.player_id is None:
        raise SystemExit("give a player_id and game_date, or --measure N")
    print(json.dumps(build(a.player_id, a.game_date, blind=not a.show_score,
                           pbp_mode=a.pbp), indent=2, default=str))


if __name__ == "__main__":
    main()
