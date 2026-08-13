"""L5a: per-team-game context -> `game_context`. No API calls; all derived locally.

These are the covariates the residual model needs: the circumstances of a game that
legitimately depress production, so that what remains unexplained is the signal.

SCHEDULE COVARIATES COME FROM `games`, NOT `player_games`. This matters right now.
`games` holds the complete 1,230-game schedule; `player_games` holds only the ~340 we
have box scores for. Sequencing rest days off the partial table produces nonsense --
Boston appeared to have a 9-day gap between Nov 1 and Nov 10, when in truth they
played four games in between that we simply have not fetched yet. Rest, back-to-backs,
home/away and month are schedule facts and must be read from the full schedule.

OUTCOME COVARIATES (final margin) come from `player_games`, so they exist only for
fetched games. Everything else is populated for all 2,460 team-games from day one.

NOT HERE YET: garbage-time possessions. Knowing WHEN a game stopped being competitive
needs play-by-play (1,230 calls). abs_margin is the cheap proxy -- it says the game
ended up decided, not when it became decided. With PBP the stats could instead be
recomputed on competitive minutes only, which is the better version of the same idea.

    python -m pipeline.load_data.load_context            # DRY
    python -m pipeline.load_data.load_context run        # write
"""
from pipeline.core import db

# Arena elevation in feet, keyed by the HOME team. Only Denver (5,280) and Utah
# (4,226) are high enough to plausibly affect a player physically, but the column is
# free and lets the model decide rather than us.
ALTITUDE_FT = {
    1610612743: 5280,   # Denver Nuggets
    1610612762: 4226,   # Utah Jazz
    1610612760: 1200,   # Oklahoma City Thunder
    1610612756: 1086,   # Phoenix Suns
    1610612737: 1050,   # Atlanta Hawks
    1610612750:  830,   # Minnesota Timberwolves
    1610612766:  751,   # Charlotte Hornets
    1610612754:  715,   # Indiana Pacers
    1610612759:  650,   # San Antonio Spurs
    1610612739:  653,   # Cleveland Cavaliers
    1610612749:  617,   # Milwaukee Bucks
    1610612765:  600,   # Detroit Pistons
    1610612741:  594,   # Chicago Bulls
    1610612742:  430,   # Dallas Mavericks
    1610612763:  337,   # Memphis Grizzlies
    1610612761:  251,   # Toronto Raptors
    1610612746:  233,   # LA Clippers
    1610612747:  233,   # Los Angeles Lakers
    1610612753:   82,   # Orlando Magic
    1610612745:   50,   # Houston Rockets
    1610612757:   50,   # Portland Trail Blazers
    1610612744:   43,   # Golden State Warriors
    1610612755:   39,   # Philadelphia 76ers
    1610612751:   33,   # Brooklyn Nets
    1610612752:   33,   # New York Knicks
    1610612758:   30,   # Sacramento Kings
    1610612764:   25,   # Washington Wizards
    1610612738:   20,   # Boston Celtics
    1610612748:    6,   # Miami Heat
    1610612740:    3,   # New Orleans Pelicans
}

DDL = """
CREATE TABLE IF NOT EXISTS game_context (
    game_id      TEXT    NOT NULL REFERENCES games (game_id),
    team_id      INTEGER NOT NULL REFERENCES teams (team_id),
    opp_team_id  INTEGER NOT NULL REFERENCES teams (team_id),
    game_date    DATE    NOT NULL,

    -- schedule facts, from the COMPLETE `games` table
    is_home      BOOLEAN NOT NULL,
    rest_days    INTEGER,          -- NULL on a team's first game of the season
    is_b2b       BOOLEAN,          -- rest_days = 1
    month        INTEGER NOT NULL, -- 10..12, then 1..4
    altitude_ft  INTEGER,          -- of the arena, i.e. the HOME team's city

    -- outcome facts, only for games whose box scores we have
    team_pts     INTEGER,
    opp_pts      INTEGER,
    margin       INTEGER,          -- team_pts - opp_pts, signed
    abs_margin   INTEGER,          -- how decided the game was; garbage-time proxy

    loaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, team_id)
);
CREATE INDEX IF NOT EXISTS game_context_team_idx ON game_context (team_id, game_date);
"""

# One row per team-game. The schedule half is built from `games`, so it is complete;
# the scoring half LEFT JOINs `player_games`, so it is NULL where unfetched.
SQL = """
WITH team_games AS (
    SELECT game_id, game_date, home_team_id AS team_id, away_team_id AS opp_team_id,
           TRUE AS is_home, home_team_id AS arena_team
    FROM games
    UNION ALL
    SELECT game_id, game_date, away_team_id, home_team_id,
           FALSE, home_team_id
    FROM games),
rested AS (
    SELECT *, (game_date - lag(game_date) OVER (PARTITION BY team_id
                                                ORDER BY game_date))::int AS rest_days
    FROM team_games),
pts AS (
    SELECT game_id, team_id, sum(points) AS team_pts
    FROM player_games WHERE points IS NOT NULL GROUP BY 1, 2)
SELECT r.game_id, r.team_id, r.opp_team_id, r.game_date, r.is_home,
       r.rest_days,
       -- coalesce, NOT plain (rest_days = 1): SQL three-valued logic makes that NULL
       -- on a season opener, and a regression would drop those 30 rows as missing.
       -- An opener is definitionally not a back-to-back -- the team had months off.
       coalesce(r.rest_days = 1, FALSE) AS is_b2b,
       extract(month FROM r.game_date)::int AS month,
       r.arena_team,
       p.team_pts, o.team_pts AS opp_pts,
       (p.team_pts - o.team_pts) AS margin,
       abs(p.team_pts - o.team_pts) AS abs_margin
FROM rested r
LEFT JOIN pts p ON p.game_id = r.game_id AND p.team_id = r.team_id
LEFT JOIN pts o ON o.game_id = r.game_id AND o.team_id = r.opp_team_id
ORDER BY r.game_date, r.game_id, r.team_id
"""


def build(conn):
    with conn.cursor() as cur:
        cur.execute(SQL)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["altitude_ft"] = ALTITUDE_FT.get(r.pop("arena_team"))
    return rows


def main(dry=True):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
        rows = build(conn)

        scored = sum(1 for r in rows if r["margin"] is not None)
        print(f"team-games (expect 2 x 1,230)  : {len(rows):,}")
        print(f"  with a final score           : {scored:,}   (rest await L4)")
        print(f"  back-to-backs                : {sum(1 for r in rows if r['is_b2b']):,}")
        print(f"  home games                   : {sum(1 for r in rows if r['is_home']):,}")
        print(f"  missing altitude             : "
              f"{sum(1 for r in rows if r['altitude_ft'] is None):,}")

        if dry:
            db.dry_notice()
            return

        n = db.upsert(conn, "game_context", rows, conflict=["game_id", "team_id"])
        print(f"\nupserted {n:,} rows -> game_context "
              f"({db.count(conn, 'game_context'):,} total)")


if __name__ == "__main__":
    main(dry=db.is_dry())
