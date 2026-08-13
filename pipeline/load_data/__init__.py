"""L0-L4: everything that pulls data in from outside.

    L0/L1  load_teams, load_games, load_players, load_rosters, load_salaries
    L2/L3  load_events, load_props, load_line_history, load_line_pulls   <- costs money
    L4     load_boxscores, load_pbp, load_pbp_events, load_context

Every loader is DRY BY DEFAULT: it reports what it would fetch and writes nothing
until you pass `run`. Every network call goes through pipeline.core.cache.
"""
