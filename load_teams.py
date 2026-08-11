"""L1a: the 30 NBA teams -> `teams`.

SUMMARY: Calls and inserts into TEAMS db all 30 NBA teams from NBA API static_teams endpoint.
This is constant.

ZERO network calls -- nba_api ships a static team table in the package. This is
purely a local lookup, and it is what lets NBA's 'MIL' meet OddsAPI's
'Milwaukee Bucks': the nickname is the join key the event matcher uses at L2.

    python load_teams.py            # DRY
    python load_teams.py run        # write
"""
from nba_api.stats.static import teams as static_teams

import db

def build():
    """Static team table -> rows for `teams`."""
    return [{"team_id": t["id"],
             "abbreviation": t["abbreviation"],
             "nickname": t["nickname"],
             "full_name": t["full_name"]}
            for t in static_teams.get_teams()]


def main(dry=True):
    rows = build()
    with db.connect() as conn:
        have = db.existing_ids(conn, "teams", "team_id")
        new = [r for r in rows if r["team_id"] not in have]

        print(f"static team table : {len(rows)} teams (0 API calls)")
        print(f"already in `teams`: {len(have)}")
        print(f"new               : {len(new)}")

        if dry:
            db.dry_notice()
            return

        n = db.upsert(conn, "teams", rows, conflict=["team_id"])
        print(f"\nupserted {n} rows -> teams ({db.count(conn, 'teams')} total)")


if __name__ == "__main__":
    main(dry=db.is_dry())
