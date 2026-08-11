"""L1b: the 2023-24 regular-season games spine -> `games`.

SUMMARY: Loads all the NBA games from a season into GAMES db using NBA API endpoint. 
Live tracker will call for each new gameday, so that we can see which games are coming up.

ONE API call for the whole season. LeagueGameFinder returns every game twice --
once per team -- so a game's home/away sides are recovered from the MATCHUP
string ('MIL vs. PHI' = MIL home, 'PHI @ MIL' = PHI away) and collapsed to one row.

Regular season only: SEASON_ID starting with '2' (preseason '1', all-star '3',
playoffs '4'), cross-checked against the game_id '002' prefix. Preseason is
excluded on purpose -- those games carry no player-tracking data.

This table is the DENOMINATOR for everything downstream. It is what lets a later
layer ask "which games had NO prop?" -- a question odds-first ingest cannot answer,
because it cannot tell a line that was never offered from one we never looked for.

NOTE: `tipoff_utc` stays NULL here. LeagueGameFinder carries GAME_DATE but no tip
time; L2 fills it from OddsAPI's commence_time, which is authoritative anyway since
every snapshot offset is measured from it.

stats.nba.com stalls intermittently and blocks datacenter/VPN IPs -- run on a
residential connection. The response caches, so this is a one-time cost.

    python load_games.py            # DRY
    python load_games.py run        # fetch (if uncached) + write
"""
import sys
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder

import cache
import config
import db

# Fail fast. A healthy stats.nba.com answers in 1-3s, so a 20s timeout only ever
# waits on a socket that is already dead. Retries exist for INTERMITTENT stalls
# (v0 hit them constantly); they do nothing for a persistent block, so we cap the
# total wait at ~66s instead of ~7min and let the caller decide what to do next.
# The real retry is re-running the loader: upserts make a second pass free.
RETRIES = 3
TIMEOUT = 20
KEY = cache.nba_key("gamefinder", config.SEASON)      # -> gamefinder_2023-24.csv
DTYPE = {"GAME_ID": str, "SEASON_ID": str, "TEAM_ID": "Int64"}


def _call():
    """The only network access in this file. Returns (DataFrame, meta)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            gf = leaguegamefinder.LeagueGameFinder(
                season_nullable=config.SEASON, league_id_nullable="00", timeout=TIMEOUT)
            return gf.get_data_frames()[0], {"http_status": 200, "credits_delta": 0}
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last = e
            if attempt == RETRIES:                    # no point backing off before giving up
                break
            wait = 2 * attempt
            print(f"   stats.nba.com stalled (attempt {attempt}/{RETRIES}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(
        f"stats.nba.com unreachable after {RETRIES} attempts ({type(last).__name__}). "
        "It drops connections from datacenter/VPN IPs -- disconnect any VPN and retry. "
        "Nothing was written; re-run when connectivity is back."
    ) from last


def fetch(conn=None):
    """Season game list -> (df, source). Cached; the wire is a last resort."""
    return cache.get("nba", KEY, _call, api="nba", endpoint="leaguegamefinder",
                     conn=conn, params={"season": config.SEASON, "league_id": "00"},
                     fmt="csv", read_kwargs={"dtype": DTYPE})


def build(df):
    """Two-rows-per-game frame -> one row per game. Returns (rows, problems)."""
    df = df.copy()
    df["game_id"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    # 2 prefix signals regular season game
    df = df[df["SEASON_ID"].str.startswith("2") & df["game_id"].str.startswith("002")]

    rows, problems = [], []
    for gid, grp in df.groupby("game_id", sort=True):
        home = grp[grp["MATCHUP"].str.contains(" vs. ", regex=False)]
        away = grp[grp["MATCHUP"].str.contains(" @ ", regex=False)]
        if len(home) != 1 or len(away) != 1:
            problems.append((gid, len(home), len(away)))
            continue
        h, a = home.iloc[0], away.iloc[0]
        rows.append({
            "game_id": gid,
            "season": config.SEASON,
            "season_type": config.SEASON_TYPE,
            "game_date": pd.to_datetime(h["GAME_DATE"]).date(),
            "tipoff_utc": None,                       # filled at L2 from commence_time
            "home_team_id": int(h["TEAM_ID"]),
            "away_team_id": int(a["TEAM_ID"]),
            "matchup": f"{a['TEAM_ABBREVIATION']} @ {h['TEAM_ABBREVIATION']}",
        })
    return rows, problems


def report(rows, problems, have):
    dates = sorted({r["game_date"] for r in rows})
    teams = {r["home_team_id"] for r in rows} | {r["away_team_id"] for r in rows}
    new = [r for r in rows if r["game_id"] not in have]
    print(f"regular-season games : {len(rows)}")
    print(f"date range           : {dates[0]} -> {dates[-1]}  ({len(dates)} game days)")
    print(f"distinct teams       : {len(teams)}")
    print(f"already in `games`   : {len(have)}")
    print(f"new                  : {len(new)}")
    if problems:
        print(f"\n!! {len(problems)} game_ids without exactly one home + one away row:")
        for gid, nh, na in problems[:10]:
            print(f"   {gid}: home rows={nh} away rows={na}")
    return new


def main(dry=True):
    where = cache.status("nba", KEY)
    if dry and where is None:
        print(f"LeagueGameFinder({config.SEASON}): NOT cached.")
        print("==> PLANNED: 1 NBA API call (free, ~5s). Nothing else to report until cached.")
        db.dry_notice()
        return

    with db.connect() as conn:
        df, source = fetch(conn=conn)
        print(f"source: {source}   raw rows: {len(df)}")

        rows, problems = build(df)
        have = db.existing_ids(conn, "games", "game_id")
        report(rows, problems, have)

        if dry:
            print(cache.summary())
            db.dry_notice()
            return

        missing = ({r["home_team_id"] for r in rows} | {r["away_team_id"] for r in rows}) \
            - db.existing_ids(conn, "teams", "team_id")
        if missing:
            sys.exit(f"!! {len(missing)} team_ids not in `teams` -- run load_teams.py first")

        n = db.upsert(conn, "games", rows, conflict=["game_id"],
                      update=["season", "season_type", "game_date",
                              "home_team_id", "away_team_id", "matchup"])
        print(f"\nupserted {n} rows -> games ({db.count(conn, 'games')} total)")
        print(cache.summary())


if __name__ == "__main__":
    main(dry=db.is_dry())
