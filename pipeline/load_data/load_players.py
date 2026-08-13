"""L0b: the NBA player dimension -> `players`.

SUMMARY: Loads each player into PLAYERS db and their nicknames from NBA API static_players
endpoint. This is for all players ever in the NBA - only needed to be ran once at the beginning of each
season in live once the NBA_API is updated with pip upgrade nba_api.

ZERO network calls -- nba_api ships a static all-time player table in the package,
and its ids ARE the personIds the box-score endpoints return at L4, so this is the
same key space, just available earlier.

Loaded early on purpose: L3 can then resolve OddsAPI's free-text player names as it
writes, and any name that fails to resolve shows up during the DRY run rather than
after ~800 credits have been spent.

`canonical_ascii` strips accents and case so 'Nikola Jokic' meets 'Nikola Jokic'.
38 all-time names collide once stripped (two Patrick Ewings, two Mike Dunleavys...).
Those are NOT resolved by name -- see resolve_name(); guessing would silently attach
a prop to the wrong person.

    python -m pipeline.load_data.load_players            # DRY
    python -m pipeline.load_data.load_players run        # write
"""
import re
import unicodedata

from nba_api.stats.static import players as static_players

from pipeline.core import db

_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")
_PUNCT = str.maketrans("", "", ".,'")


def ascii_name(s):
    """TIER 1 -- strip accents/case so 'Jokic' and 'Jokic' compare equal."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def depunct_name(s):
    """TIER 2 -- drop . , ' and close up spaced initials.

    Two separate mismatches, both real: 'R.J. Barrett' vs 'RJ Barrett' needs the
    punctuation gone, and OddsAPI's 'T. J. Warren' vs NBA's 'T.J. Warren' leaves a
    SPACE between the initials once the dots are stripped ('t j warren' !=
    'tj warren'). Collapsing single-letter runs fixes the second without creating a
    single new collision -- verified against all 5,024 all-time players.
    """
    s = re.sub(r"\s+", " ", ascii_name(s).translate(_PUNCT)).strip()
    while re.search(r"\b(\w) (?=\w\b)", s):
        s = re.sub(r"\b(\w) (?=\w\b)", r"\1", s)
    return s


def desuffix_name(s):
    """TIER 3 -- also drop trailing Jr/Sr/II/III so 'Kevin Knox' meets 'Kevin Knox II'."""
    s = depunct_name(s)
    while _SUFFIX.search(s):
        s = _SUFFIX.sub("", s)
    return s.strip()


# OddsAPI uses a nickname or formal variant no mechanical rule can reach. Each was
# confirmed against the static table by hand; 'johnny davis' additionally
# disambiguates the 1970s player (76526) from the current one.
MANUAL_ALIASES = {
    "nicolas claxton": 1629651,   # Nic Claxton
    "moe wagner": 1629021,        # Moritz Wagner
    "johnny davis": 1631098,      # Johnny Davis (2022 draft), not 76526
    "herb jones": 1630529,        # Herbert Jones
    "cameron thomas": 1630560,    # Cam Thomas
    "mohamed bamba": 1628964,     # Mo Bamba
    "cam johnson": 1629661,       # Cameron Johnson
    "gregory jackson": 1641713,   # GG Jackson
}


def make_resolver(conn):
    """-> resolve(name) -> (player_id, method).

    Tries progressively looser normalisations and STOPS at the first that maps to
    exactly one player. Order matters: tier 2 resolves 'Gary Trent Jr' to the son
    (1629018) because the suffix is still attached; tier 3 would see 'gary trent'
    and find both father and son, so it must never run first. Any name that stays
    ambiguous resolves to None -- a wrong attribution is worse than a missing one.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT player_id, full_name FROM players")
        people = cur.fetchall()

    tiers = []
    for fn in (ascii_name, depunct_name, desuffix_name):
        buckets = {}
        for pid, full in people:
            buckets.setdefault(fn(full), []).append(pid)
        tiers.append({k: v[0] for k, v in buckets.items() if len(v) == 1})

    def resolve(name):
        for i, (fn, idx) in enumerate(zip((ascii_name, depunct_name, desuffix_name), tiers), 1):
            pid = idx.get(fn(name))
            if pid is not None:
                return pid, f"tier{i}"
        pid = MANUAL_ALIASES.get(depunct_name(name))
        return (pid, "manual") if pid else (None, None)

    return resolve


def build():
    return [{"player_id": p["id"],
             "full_name": p["full_name"],
             "canonical_ascii": ascii_name(p["full_name"])}
            for p in static_players.get_players()]


def ambiguous(rows):
    """canonical names shared by more than one player_id -> {name: [ids]}."""
    by_name = {}
    for r in rows:
        by_name.setdefault(r["canonical_ascii"], []).append(r["player_id"])
    return {n: ids for n, ids in by_name.items() if len(ids) > 1}


def main(dry=True):
    rows = build()
    amb = ambiguous(rows)
    with db.connect() as conn:
        have = db.existing_ids(conn, "players", "player_id")
        new = [r for r in rows if r["player_id"] not in have]

        print(f"static player table : {len(rows)} players (0 API calls)")
        print(f"already in `players`: {len(have)}")
        print(f"new                 : {len(new)}")
        print(f"ambiguous names     : {len(amb)} canonical names map to >1 player_id")

        if dry:
            db.dry_notice()
            return

        n = db.upsert(conn, "players", rows, conflict=["player_id"])
        print(f"\nupserted {n} rows -> players ({db.count(conn, 'players')} total)")


if __name__ == "__main__":
    main(dry=db.is_dry())
