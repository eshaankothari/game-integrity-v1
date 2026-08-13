"""L1d: 2023-24 season salary -> `player_salaries`. 30 free calls, one per team.

SUMMARY: Scrapes the salary table off each team's basketball-reference season page,
resolves the names to player_ids with the same tiered matcher L3 uses, and writes one
row per player with rank and percentile.

WHY SALARY. Every other covariate in this project describes what happened in a game.
Salary describes what a player stood to LOSE, which is the only variable that speaks to
motive. It spans two orders of magnitude on a single roster -- roughly $50M for a max
contract against $560K for a two-way -- so it separates "could not plausibly be bought"
from "could be" far more sharply than minutes or usage ever will.

THERE IS NO API FOR IT. nba_api exposes no salary endpoint, HoopsHype has dropped its
historical season pages, and RealGM 403s. Basketball-reference publishes it per team
per season, which makes the whole season exactly 30 requests, free and cached forever.

A BLANK SALARY IS DATA. bbref lists no figure for two-way and 10-day contracts, so
those cells come back empty -- and that is precisely the fringe-roster population most
exposed to an approach. The blank is recorded as has_listed_salary = FALSE rather than
imputed or dropped, so downstream can treat "not listed" as its own bucket.

TRADED PLAYERS appear on both teams' pages. bbref shows the full-season cap hit on
each, so the rows agree; we keep the MAX and record n_teams so a trade stays visible.

    python -m pipeline.load_data.load_salaries            # DRY
    python -m pipeline.load_data.load_salaries run        # 30 calls, free
"""
import html
import re
import sys
import time

import pandas as pd
import requests

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db
from pipeline.load_data.load_players import ascii_name, depunct_name, desuffix_name, make_resolver

URL = "https://www.basketball-reference.com/teams/{abbr}/{year}.html"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
TIMEOUT = 25
RETRIES = 3
SLEEP = 3.5          # bbref asks for <=20 requests/min; this is ~17

# bbref's abbreviations differ from the NBA's for exactly three teams.
BBREF_ABBR = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}

# '2023-24' -> 2024. bbref keys a season page by the year it ENDS.
YEAR = int(config.SEASON.split("-")[0]) + 1


def _call(bbref_abbr):
    """One team season page, parsed to a DataFrame. -> (DataFrame, meta)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(URL.format(abbr=bbref_abbr, year=YEAR),
                             headers={"User-Agent": UA}, timeout=TIMEOUT)
            r.raise_for_status()
            # bbref serves UTF-8 but declares no charset, so requests falls back to
            # ISO-8859-1 per the HTTP spec and every diacritic arrives as mojibake --
            # 'Nikola JokiÄ‡'. That then fails name resolution for a dozen players,
            # most of them stars. Force the encoding it actually uses.
            r.encoding = "utf-8"
            return parse(r.text), {"http_status": r.status_code, "credits_delta": 0}
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            last = e
            if attempt == RETRIES:
                break
            time.sleep(2 * attempt)
    raise last


def parse(page):
    """Team season page HTML -> DataFrame[player_name_raw, salary].

    bbref buries its secondary tables inside HTML comments to keep them out of the
    initial render, so the comment bodies are concatenated back on before searching.
    """
    blob = page + "".join(re.findall(r"<!--(.*?)-->", page, re.S))
    table = re.search(r'<table[^>]*id="salaries2".*?</table>', blob, re.S)
    if not table:
        return pd.DataFrame(columns=["player_name_raw", "salary"])
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", table.group(0), re.S):
        cells = {}
        for cell in re.findall(r'<t[hd][^>]*data-stat="([a-z_0-9]+)"[^>]*>(.*?)</t[hd]>',
                               tr, re.S):
            cells[cell[0]] = html.unescape(re.sub(r"<[^>]+>", "", cell[1])).strip()
        name, sal = cells.get("player", ""), cells.get("salary", "")
        if not name or name == "Player":
            continue
        rows.append({"player_name_raw": name, "salary": _money(sal)})
    return pd.DataFrame(rows, columns=["player_name_raw", "salary"])


def _money(s):
    """'$36,861,707' -> 36861707. Empty (two-way / 10-day) -> None."""
    digits = re.sub(r"[^0-9]", "", s or "")
    return int(digits) if digits else None


def fetch_team(nba_abbr, conn=None, allow_fetch=False):
    key = cache.bbref_key("salaries", f"{YEAR}_{nba_abbr}")
    if not allow_fetch:
        return cache.read_cached("bbref", key, fmt="csv")
    try:
        df, src = cache.get("bbref", key, lambda: _call(BBREF_ABBR.get(nba_abbr, nba_abbr)),
                            api="bbref", endpoint="bbref_salaries", conn=conn,
                            params={"team": nba_abbr, "year": YEAR}, fmt="csv")
        if src == "network":
            time.sleep(SLEEP)            # only rate-limit real requests, never cache hits
        return df
    except Exception as e:
        print(f"   !! salaries {nba_abbr}: {type(e).__name__}: {e}")
        return None


def _last(s):
    """Surname only, suffix and punctuation stripped. 'Gary Trent Jr.' -> 'trent'."""
    parts = desuffix_name(depunct_name(s)).split()
    return parts[-1] if parts else ""


def team_rosters(conn):
    """{NBA abbr: [{name index}, ...]} from the rosters L1c already cached, free.

    WHY TEAM SCOPE. The bbref salary table writes 'Jaren Jackson', not 'Jaren
    Jackson Jr.', so a league-wide match on that string lands on the FATHER -- who
    retired in 2002 -- and does it silently, producing a confidently wrong salary
    rather than an unresolved name. Same for Gary Trent and Tim Hardaway. But the
    page we scraped it from is Memphis's, and only one Jaren Jackson was on Memphis
    in 2023-24, so the team disambiguates what the name alone cannot.

    Three tiers, each accepted ONLY if unique within the team, so MEM's two Jacksons
    (Jaren and 'Gregory'/GG) fall through to the full-name tiers instead of colliding.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT team_id, abbreviation FROM teams")
        by_id = dict(cur.fetchall())
    out = {}
    for team_id, abbr in by_id.items():
        df = cache.read_cached(
            "nba", cache.nba_key("roster", f"{config.SEASON}_{team_id}"), fmt="csv")
        if df is None or df.empty:
            continue
        people = [(int(r["PLAYER_ID"]), str(r["PLAYER"]))
                  for _, r in df.iterrows() if not pd.isna(r.get("PLAYER_ID"))]
        tiers = []
        for fn in (ascii_name, desuffix_name, _last):
            buckets = {}
            for pid, full in people:
                buckets.setdefault(fn(full), []).append(pid)
            tiers.append({k: v[0] for k, v in buckets.items() if len(set(v)) == 1})
        out[abbr] = tiers
    return out


def build(frames, resolve, rosters, unresolved, methods):
    """Per-team frames -> one row per player_id, MAX salary across teams."""
    best = {}
    for abbr, df in frames:
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            name = str(r["player_name_raw"]).strip()
            pid, method = None, None
            # Team-scoped first: it is strictly better informed than the global
            # matcher, which cannot see which roster the name came off.
            for i, fn in enumerate((ascii_name, desuffix_name, _last), 1):
                hit = (rosters.get(abbr) or [{}, {}, {}])[i - 1].get(fn(name))
                if hit is not None:
                    pid, method = hit, f"team{i}"
                    break
            if pid is None:
                pid, method = resolve(name)   # league-wide fallback

            methods[method or "UNRESOLVED"] = methods.get(method or "UNRESOLVED", 0) + 1
            if pid is None:
                unresolved[name] = unresolved.get(name, 0) + 1
                continue
            sal = None if pd.isna(r["salary"]) else int(r["salary"])
            cur = best.get(pid)
            if cur is None:
                best[pid] = {"player_id": pid, "season": config.SEASON,
                             "player_name_raw": name, "team_abbr": abbr,
                             "salary": sal, "n_teams": 1, "source": "bbref"}
                continue
            cur["n_teams"] += 1
            # Keep the larger figure; a listed salary always beats a blank.
            if sal is not None and (cur["salary"] is None or sal > cur["salary"]):
                cur["salary"], cur["team_abbr"], cur["player_name_raw"] = sal, abbr, name

    rows = list(best.values())
    # Rank and percentile over LISTED salaries only -- mixing the blanks in would
    # rank a two-way player as if he earned zero, which is both wrong and unstable.
    listed = sorted((r for r in rows if r["salary"] is not None),
                    key=lambda r: -r["salary"])
    for i, r in enumerate(listed, 1):
        r["salary_rank"] = i
        r["salary_pct"] = round(1 - (i - 1) / max(len(listed) - 1, 1), 4)
    for r in rows:
        r["has_listed_salary"] = r["salary"] is not None
        r.setdefault("salary_rank", None)
        r.setdefault("salary_pct", None)
    return rows


def main(dry=True):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT abbreviation FROM teams ORDER BY abbreviation")
            abbrs = [r[0] for r in cur.fetchall()]
        if not abbrs:
            sys.exit("!! `teams` is empty -- run load_teams.py first")

        cached_n = sum(1 for a in abbrs
                       if cache.status("bbref", cache.bbref_key("salaries", f"{YEAR}_{a}")))
        missing = len(abbrs) - cached_n
        print(f"season             : {config.SEASON}  (bbref year {YEAR})")
        print(f"teams              : {len(abbrs)}")
        print(f"salary pages cached: {cached_n}  <- free")
        print(f"==> calls to make  : {missing}  (basketball-reference is FREE)")
        if missing:
            print(f"    at {SLEEP}s apart, about {missing * SLEEP / 60:.1f} min")

        if dry:
            db.dry_notice()
            return

        frames = [(a, fetch_team(a, conn, allow_fetch=True)) for a in abbrs]
        resolve, rosters = make_resolver(conn), team_rosters(conn)
        unresolved, methods = {}, {}
        rows = build(frames, resolve, rosters, unresolved, methods)

        # Only players `players` already knows. bbref lists everyone who appeared on a
        # cap sheet, including players who never suited up in a 2023-24 game.
        known = db.existing_ids(conn, "players", "player_id")
        rows = [r for r in rows if r["player_id"] in known]

        listed = [r for r in rows if r["has_listed_salary"]]
        print(f"\nplayers scraped    : {len(rows)}")
        print(f"  with a salary    : {len(listed)}")
        print(f"  not listed       : {len(rows) - len(listed)}  (two-way / 10-day)")
        print(f"  traded mid-season: {sum(1 for r in rows if r['n_teams'] > 1)}")
        print(f"name resolution    : "
              + ", ".join(f"{k}={v:,}" for k, v in sorted(methods.items())))
        print(f"unresolved names   : {len(unresolved)}")
        for name, n in sorted(unresolved.items(), key=lambda x: -x[1])[:15]:
            print(f"   !! {name}  ({n})")

        n = db.upsert(conn, "player_salaries", rows, conflict=["player_id", "season"])
        print(f"\nupserted {n} rows -> player_salaries")
        if listed:
            top = sorted(listed, key=lambda r: -r["salary"])[:3]
            print("top paid           : " + ", ".join(
                f"{r['player_name_raw']} ${r['salary']:,}" for r in top))
        print(cache.summary())


if __name__ == "__main__":
    main(dry=db.is_dry())
