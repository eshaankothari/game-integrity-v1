-- game-integrity v1 schema.
--
-- Principle: STORE RAW, DERIVE IN VIEWS. Every table below holds what the API
-- actually returned. Z-scores, line movement and anomaly scores are functions of
-- a baseline choice, and baselines change -- so they are views/materializations
-- built on top, never columns baked in at ingest time.
--
-- Scope: 2023-24 NBA REGULAR SEASON only (game_id prefix '002').
--
-- Apply with:  psql -d game_integrity_v1 -f schema.sql

BEGIN;

-- ---------------------------------------------------------------------------
-- L0. dimensions
-- ---------------------------------------------------------------------------

CREATE TABLE teams (
    team_id      INTEGER PRIMARY KEY,          -- NBA team id, e.g. 1610612749
    abbreviation TEXT NOT NULL,                -- 'MIL'
    nickname     TEXT NOT NULL,                -- 'Bucks'   (OddsAPI match key)
    full_name    TEXT NOT NULL,                -- 'Milwaukee Bucks'
    UNIQUE (abbreviation)
);


CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,       -- NBA personId
    full_name       TEXT NOT NULL,
    canonical_ascii TEXT NOT NULL,             -- accent/case-stripped join key

    -- Player ATTRIBUTES, from CommonTeamRoster (30 calls, one per team). They live
    -- here rather than on player_games because they do not vary game to game.
    --
    -- position is the one L5 needs: a centre's rebound z-score and a guard's are
    -- not comparable quantities, so a purely league-wide baseline makes every
    -- centre look like a rebounding outlier. It CANNOT come from the box score --
    -- that column is populated for starters only.
    position    TEXT,                          -- 'G' | 'F' | 'C' | 'G-F' | ...
    height_in   INTEGER,                       -- '6-8' parsed to inches
    weight_lb   INTEGER,
    birth_date  DATE,
    experience  INTEGER,                       -- seasons played; 'R' rookie -> 0
    school      TEXT
);
CREATE INDEX players_canonical_idx ON players (canonical_ascii);


-- OddsAPI sends free-text player names ("description"); NBA sends personId.
-- Every mapping is recorded here so a name that fails to resolve is VISIBLE
-- rather than silently dropping a player-game from the sample.
CREATE TABLE player_aliases (
    alias_ascii TEXT PRIMARY KEY,              -- ascii-normalized source string
    player_id   INTEGER REFERENCES players (player_id),   -- NULL = unresolved
    source      TEXT NOT NULL,                 -- 'oddsapi'
    method      TEXT NOT NULL,                 -- 'exact' | 'fuzzy' | 'manual'
    n_seen      INTEGER NOT NULL DEFAULT 1,    -- how often this string appeared
    note        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX player_aliases_unresolved_idx ON player_aliases (source)
    WHERE player_id IS NULL;


-- ---------------------------------------------------------------------------
-- L1. games spine
-- ---------------------------------------------------------------------------

CREATE TABLE games (
    game_id      TEXT PRIMARY KEY,             -- '0022300001' (zero-padded, so TEXT)
    season       TEXT NOT NULL,                -- '2023-24'
    season_type  TEXT NOT NULL DEFAULT 'regular'
                 CHECK (season_type IN ('preseason', 'regular', 'allstar', 'playoffs')),
    game_date    DATE NOT NULL,                -- US/Eastern calendar date (NBA's date)
    tipoff_utc   TIMESTAMPTZ,                  -- from NBA schedule; approximate
    home_team_id INTEGER NOT NULL REFERENCES teams (team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams (team_id),
    matchup      TEXT,                         -- 'TOR @ HOU'
    loaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (home_team_id <> away_team_id)
);
CREATE INDEX games_date_idx ON games (game_date);
CREATE INDEX games_season_type_idx ON games (season, season_type);


-- ---------------------------------------------------------------------------
-- L2. OddsAPI event mapping
-- ---------------------------------------------------------------------------

CREATE TABLE odds_events (
    event_id       TEXT PRIMARY KEY,           -- OddsAPI hex id
    game_id        TEXT UNIQUE REFERENCES games (game_id),  -- NULL = unmapped event
    commence_time  TIMESTAMPTZ NOT NULL,       -- AUTHORITATIVE tip: all probe
                                               -- offsets are measured from this
    home_team_name TEXT,                       -- OddsAPI naming ('Houston Rockets')
    away_team_name TEXT,
    matched_by     TEXT,                       -- 'local_season_file' | 'events_endpoint'
    match_note     TEXT,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX odds_events_unmapped_idx ON odds_events (commence_time)
    WHERE game_id IS NULL;


-- ---------------------------------------------------------------------------
-- L3. raw prop quotes  (the expensive layer -- never delete rows here)
-- ---------------------------------------------------------------------------

-- One row per (event, player, market, book, probe). A single API response
-- explodes into ~10 books x ~11 players = ~110 rows, all for one credit.
CREATE TABLE prop_quotes (
    event_id            TEXT NOT NULL REFERENCES odds_events (event_id),
    player_name_raw     TEXT NOT NULL,         -- OddsAPI 'description', verbatim
    player_id           INTEGER REFERENCES players (player_id),  -- resolved later
    market              TEXT NOT NULL,         -- 'player_points'
    book                TEXT NOT NULL,         -- 'fanduel', 'draftkings', ...

    snapshot_requested  TIMESTAMPTZ NOT NULL,  -- the 'date' param we asked for
    snapshot_actual     TIMESTAMPTZ NOT NULL,  -- response 'timestamp'; API snaps
                                               -- BACK to a 5-min grid (~4.5m earlier)
    line_last_update    TIMESTAMPTZ,           -- market.last_update -- when the book
                                               -- last MOVED this line. Staleness
                                               -- signal: tells you whether a probe
                                               -- caught a fresh or long-settled line,
                                               -- at zero extra credit cost.

    -- 'poll' is the honest default: at write time you rarely know a probe is THE
    -- open or THE close. Batch writes open/close because it probes exactly twice;
    -- live polls every ~15min and only learns which was last after tip. Treat
    -- open/close as a DERIVED label (earliest/latest probe per event) so batch and
    -- live agree, rather than a fact asserted at ingest.
    snapshot_role       TEXT NOT NULL CHECK (snapshot_role IN ('open', 'close', 'poll')),
    offset_from_tip_sec INTEGER,               -- commence_time - snapshot_actual

    line                NUMERIC(5,1) NOT NULL, -- 13.5
    over_price          NUMERIC(8,3),          -- decimal odds, 1.90
    under_price         NUMERIC(8,3),

    -- `line` is in the key so a book may quote SEVERAL simultaneous lines for one
    -- player at one instant. BetRivers/Unibet do exactly that -- 10 alternate point
    -- values for Giannis in a single response. Without it those ten rows collide and
    -- ON CONFLICT DO UPDATE silently keeps whichever landed last. The three books we
    -- write (fanduel/draftkings/williamhill_us) each quote one line, so this is
    -- inert today and insurance against a book changing behaviour.
    PRIMARY KEY (event_id, player_name_raw, market, book, snapshot_requested, line)
);
CREATE INDEX prop_quotes_player_idx ON prop_quotes (player_id, market);
CREATE INDEX prop_quotes_role_idx ON prop_quotes (market, book, snapshot_role);
CREATE INDEX prop_quotes_unresolved_idx ON prop_quotes (player_name_raw)
    WHERE player_id IS NULL;


-- ---------------------------------------------------------------------------
-- L4. raw box-score stats  (all players in every game, not just propped ones)
-- ---------------------------------------------------------------------------

CREATE TABLE player_games (
    game_id   TEXT NOT NULL REFERENCES games (game_id),
    player_id INTEGER NOT NULL REFERENCES players (player_id),
    team_id   INTEGER REFERENCES teams (team_id),
    started   BOOLEAN,                       -- inferred: box-score `position` is
                                             -- filled for starters only

    -- Verbatim box-score `comment` when the player did not appear: "DNP - Coach's
    -- Decision", "DND - Injury", "NWT - Not With Team". NULL means they played.
    -- The REASON matters: a healthy scratch on a night with unusual line movement
    -- is a different signal from a known injury.
    dnp_reason TEXT,

    -- BoxScoreTraditionalV3
    minutes             NUMERIC(6,3),          -- decimal minutes ('34:12' -> 34.200)
    points              INTEGER,
    fga                 INTEGER,
    fgm                 INTEGER,
    fg3a                INTEGER,
    fg3m                INTEGER,
    fta                 INTEGER,
    ftm                 INTEGER,
    rebounds            INTEGER,               -- reboundsTotal
    rebounds_off        INTEGER,
    assists             INTEGER,
    steals              INTEGER,
    blocks              INTEGER,
    turnovers           INTEGER,
    fouls               INTEGER,
    plus_minus          NUMERIC(6,1),

    -- BoxScoreAdvancedV3
    usage_pct           NUMERIC(7,4),
    turnover_ratio      NUMERIC(7,3),
    true_shooting_pct   NUMERIC(7,4),
    efg_pct             NUMERIC(7,4),
    assist_pct          NUMERIC(7,4),
    rebound_pct         NUMERIC(7,4),
    offensive_rating    NUMERIC(7,2),
    defensive_rating    NUMERIC(7,2),
    net_rating          NUMERIC(7,2),
    pace                NUMERIC(7,3),

    -- BoxScoreHustleV2
    contested_shots     INTEGER,
    deflections         INTEGER,
    loose_balls         INTEGER,               -- looseBallsRecoveredTotal
    box_outs            INTEGER,
    charges_drawn       INTEGER,
    screen_assists      INTEGER,

    -- BoxScorePlayerTrackV3
    speed               NUMERIC(6,3),          -- mph (a RATE -- never per-36 it)
    distance            NUMERIC(7,3),          -- miles
    touches             INTEGER,
    passes              INTEGER,
    contested_fga       INTEGER,
    uncontested_fga     INTEGER,

    -- provenance: distinguishes "endpoint failed" from "stat is genuinely NULL".
    -- Without these, a hustle-endpoint timeout looks identical to a real zero and
    -- silently biases the z-score population.
    has_traditional BOOLEAN NOT NULL DEFAULT false,
    has_advanced    BOOLEAN NOT NULL DEFAULT false,
    has_hustle      BOOLEAN NOT NULL DEFAULT false,
    has_track       BOOLEAN NOT NULL DEFAULT false,

    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX player_games_player_idx ON player_games (player_id);
CREATE INDEX player_games_incomplete_idx ON player_games (game_id)
    WHERE NOT (has_traditional AND has_advanced AND has_hustle AND has_track);


-- ---------------------------------------------------------------------------
-- L5. standardised features
-- ---------------------------------------------------------------------------

-- One row per (player-game, baseline_mode). The SAME player-game appears once per
-- mode, because a z-score is meaningless without knowing what it was measured
-- against -- so the mode is part of the key, not a footnote.
-- player_game_z AND player_game_scores ARE OWNED BY THEIR LAYERS, NOT BY THIS FILE.
--
-- Both are created and migrated by the code that writes them -- standardize.py (L5)
-- and export_candidates.py (L6) -- because their shape follows the scoring methodology
-- and would otherwise drift out of sync with it every time a component changed.
--
--   player_game_features   RAW. Box score, lines, prices, open->close movement.
--                          Wrong only if the ingest was wrong.
--   player_game_z          STANDARDISED. The five performance components, the four
--                          market components, and the three blocks they combine into.
--                          Recomputed whenever the baseline population changes.
--   player_game_scores     THE FRONTEND TABLE. Every propped player-game with its rank,
--                          `in_shortlist`, and `cut_failed` -- the first cut that
--                          eliminated it, so the UI can explain an absence.
--
-- Run `python standardize.py run && python export_candidates.py` to build all three.
-- L5 drops and rebuilds player_game_z if it still carries `baseline_mode`, the primary
-- key of the retired four-baseline sweep.

CREATE TABLE api_calls (
    id                 BIGSERIAL PRIMARY KEY,
    called_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    api                TEXT NOT NULL CHECK (api IN ('oddsapi', 'nba', 'bbref')),
    endpoint           TEXT NOT NULL,          -- 'historical_events' | 'historical_odds' | ...
    params             JSONB,
    http_status        INTEGER,
    cache_hit          BOOLEAN NOT NULL,       -- true => 0 credits
    cache_source       TEXT CHECK (cache_source IN ('v0', 'v1')),
    cache_path         TEXT,
    requests_used      INTEGER,                -- x-requests-used header
    requests_remaining INTEGER,                -- x-requests-remaining header
    credits_delta      INTEGER,                -- measured cost of THIS call
    error              TEXT
);
CREATE INDEX api_calls_billed_idx ON api_calls (api, called_at) WHERE NOT cache_hit;


-- Season salary, the INCENTIVE covariate. What a player risks by throwing a game is
-- the thing salary measures, and it varies by two orders of magnitude across a roster:
-- a max player risks ~$50M a year, a two-way ~$560K. That asymmetry is the single
-- strongest prior available for who is even plausibly corruptible.
--
-- SEASON-KEYED, and deliberately NOT a column on `players`. `players` holds all 5,026
-- historical NBA players and is season-agnostic; salary changes every year, so hanging
-- it off `players` would break the moment a second season is loaded. (`experience` on
-- `players` has exactly that latent problem today -- fine for a one-season build.)
--
-- salary IS NULL IS MEANINGFUL, NOT MISSING. Basketball-reference lists no figure for
-- two-way and 10-day contracts, so a blank marks the fringe-roster player rather than
-- a gap in the data -- see `has_listed_salary`. Treating those rows as unknown would
-- discard the population the whole exercise is most interested in.
CREATE TABLE player_salaries (
    player_id         INTEGER NOT NULL REFERENCES players (player_id),
    season            TEXT    NOT NULL,        -- '2023-24', matches config.SEASON
    player_name_raw   TEXT,                    -- name as scraped, for auditing the match
    team_abbr         TEXT,                    -- NBA abbreviation, not the bbref one
    salary            BIGINT,                  -- NULL => not listed (two-way / 10-day)
    has_listed_salary BOOLEAN NOT NULL,
    n_teams           INTEGER,                 -- >1 => traded mid-season
    salary_rank       INTEGER,                 -- 1 = highest paid, among listed only
    salary_pct        NUMERIC(6,4),            -- 0-1 percentile, among listed only
    source            TEXT,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season)
);


COMMIT;
