"""L1c: player attributes -> `players` (position, height, weight, birth date, experience).

SUMMARY: Loads each player attribute to existing PLAYERS db, enriching and not creating.

WHY THIS EXISTS. L5 standardises "relative to the league", but several of the nine
z-features are strongly position-dependent -- rebounds, assists, usage. Against a
purely league-wide baseline every centre looks like a rebounding outlier and every
guard like an assist outlier, which is noise, not signal. Position-relative
baselines fix that, and position has to come from somewhere.

IT CANNOT COME FROM THE BOX SCORE. That frame does carry a `position` column, but
it is populated for STARTERS ONLY -- roughly 10 of 27 rows a game. Gary Trent Jr.
is a guard and shows blank whenever he comes off the bench. So the box-score column
is really a did-they-start flag, which is how load_boxscores.py uses it.

TWO LIMITATIONS, both accepted deliberately:
  - The roster is an END-OF-SEASON snapshot. A player traded mid-season appears
    under his final team. That is fine here because we take only attributes that do
    not move with a trade; per-game team comes from the box score, which is correct
    by construction.
  - Players on 10-day deals who were waived before the season ended may be on no
    roster at all, so some `position` values stay NULL. L5 must tolerate that
    rather than assume full coverage.

    python -m pipeline.load_data.load_rosters            # DRY
    python -m pipeline.load_data.load_rosters run        # 30 calls, free
"""
import sys
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import commonteamroster

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db

TIMEOUT = 20
RETRIES = 3


def _call(team_id):
    """One CommonTeamRoster call. -> (DataFrame, meta)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = commonteamroster.CommonTeamRoster(
                team_id=team_id, season=config.SEASON, timeout=TIMEOUT)
            return r.common_team_roster.get_data_frame(), {"http_status": 200,
                                                           "credits_delta": 0}
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last = e
            if attempt == RETRIES:
                break
            time.sleep(2 * attempt)
    raise last


def fetch_team(team_id, conn=None, allow_fetch=False):
    key = cache.nba_key("roster", f"{config.SEASON}_{team_id}")
    if not allow_fetch:
        return cache.read_cached("nba", key, fmt="csv")
    try:
        df, _ = cache.get("nba", key, lambda: _call(team_id),
                          api="nba", endpoint="commonteamroster", conn=conn,
                          params={"team_id": team_id, "season": config.SEASON},
                          fmt="csv")
        return df
    except Exception as e:
        print(f"   !! roster {team_id}: {type(e).__name__}")
        return None


def _height_in(v):
    """'6-8' -> 80 inches. Anything unparseable -> None."""
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    if "-" not in s:
        return None
    ft, _, inch = s.partition("-")
    try:
        return int(ft) * 12 + int(inch)
    except ValueError:
        return None


def _exp(v):
    """Years of experience. 'R' means rookie, i.e. zero prior seasons."""
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s == "R":
        return 0
    try:
        return int(s)
    except ValueError:
        return None


def _int(v):
    if v is None or pd.isna(v) or str(v).strip() == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _date(v):
    if v is None or pd.isna(v) or str(v).strip() == "":
        return None
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def build(frames):
    """Roster frames -> attribute rows keyed by player_id (deduped, first wins)."""
    rows, seen = [], set()
    for df in frames:
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            pid = _int(r.get("PLAYER_ID"))
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            rows.append({
                "player_id": pid,
                "position": (str(r.get("POSITION")).strip()
                             if r.get("POSITION") is not None and not pd.isna(r.get("POSITION"))
                             else None),
                "height_in": _height_in(r.get("HEIGHT")),
                "weight_lb": _int(r.get("WEIGHT")),
                "birth_date": _date(r.get("BIRTH_DATE")),
                "experience": _exp(r.get("EXP")),
                "school": (str(r.get("SCHOOL")).strip()
                           if r.get("SCHOOL") is not None and not pd.isna(r.get("SCHOOL"))
                           else None),
            })
    return rows


def main(dry=True):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT team_id FROM teams ORDER BY team_id")
            team_ids = [r[0] for r in cur.fetchall()]
        if not team_ids:
            sys.exit("!! `teams` is empty -- run load_teams.py first")

        cached_n = sum(1 for t in team_ids
                       if cache.status("nba", cache.nba_key("roster", f"{config.SEASON}_{t}")))
        print(f"teams              : {len(team_ids)}")
        print(f"rosters cached     : {cached_n}  <- free")
        print(f"==> calls to make  : {len(team_ids) - cached_n}  (nba_api is FREE)")

        if dry:
            db.dry_notice()
            return

        frames = [fetch_team(t, conn, allow_fetch=True) for t in team_ids]
        rows = build(frames)

        # Enrich ONLY players we already know about. A roster can list someone who
        # never appeared in a 2023-24 regular-season game; inserting them would put
        # rows in `players` that no game references.
        known = db.existing_ids(conn, "players", "player_id")
        rows = [r for r in rows if r["player_id"] in known]

        # db.update, NOT db.upsert. `rows` carries no full_name/canonical_ascii, and
        # ON CONFLICT DO UPDATE would still form an INSERT tuple with those NULL and
        # trip their NOT NULL constraints before ever reaching the UPDATE branch.
        print(f"\nroster entries matched to `players`: {len(rows)}")
        n = db.update(conn, "players", rows, key=["player_id"])
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM players WHERE position IS NOT NULL")
            filled = cur.fetchone()[0]
        print(f"updated {n} rows -> players ({filled} now have a position)")
        print(cache.summary())


if __name__ == "__main__":
    main(dry=db.is_dry())
