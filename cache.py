"""The ONE place v1 touches the network.

SUMMARY: This file make sure that we are checking all caches before calling any APIs.
It also writes all calls to cache so that we do not rerequest information.

Every API call in this project goes through `get()`. Nothing else may call
requests/nba_api directly -- that rule is what makes "cached, never redundant"
a property of the system instead of a habit each loader has to remember.

Four guarantees:

  1. EVERY RESPONSE IS CACHED. `get()` writes the payload to disk before
     returning it. There is no path through this module that fetches without
     storing.

  2. NEGATIVE RESULTS ARE CACHED TOO. A 404, an empty payload, a game with no
     prop -- all get written like any other response. This is the single biggest
     source of redundant spend if you skip it: the ~15% of player-games that
     never had a line would be re-billed on every run, forever. v0 already does
     this (31 of its 1,605 cached snapshots are empty).

  3. LOOKUP ORDER IS v0 -> v1 -> network. v0's caches were paid for already;
     they are read in place and never written to. Only cache misses reach the
     wire, and they land in v1's own cache dir.

  4. A KEY IS FETCHED AT MOST ONCE PER PROCESS. `_fetched` tracks every key
     pulled from the network; a second attempt raises. That fires only if a key
     builder is wrong -- and it fires on call #2, not after 2,000 billed calls.

KEY FORMATS ARE FROZEN. They reproduce v0's filenames byte-for-byte, which is
what makes its 1,605 paid snapshots and 886 nba responses free to reuse. Change
a key builder and you silently re-pay for all of it.
"""
import json
from collections import OrderedDict

import pandas as pd
import requests

import config
import db

# Free probe: /v4/sports does NOT count against the quota (verified -- the
# x-requests-used counter does not move across repeated calls), so asking for the
# balance never costs a credit.
BALANCE_URL = "https://api.the-odds-api.com/v4/sports"

# Credits per call, MEASURED from x-requests-used deltas in our own ledger --
# not from documentation. OddsAPI bills historical odds at 10x the flat rate
# (10 = 10 x markets x regions; we send 1 market, 1 region). Getting this wrong
# understated every L3 estimate by a factor of ten.
#
# Used for DRY-run planning only. The ledger records the REAL delta per call, so
# if a rate ever changes, `audit_rates()` shows it rather than the plan quietly
# drifting from reality.
CREDITS_PER_CALL = {
    "historical_events": 1,
    "historical_odds": 10,
    "leaguegamefinder": 0,          # nba_api is free
    "bbref_salaries": 0,            # scraped HTML, free
}

_last_used = None                   # running x-requests-used, for measured deltas

MEMO_MAX = 512                      # bounded: payloads are ~50KB, 1,230 events would not fit

# kind -> (v0 dir, read-only and already paid   |   v1 dir, everything we fetch)
DIRS = {
    "snapshot": (config.SNAPSHOT_CACHE, config.SNAPSHOT_CACHE_V1),
    "events":   (config.EVENTS_CACHE,   config.EVENTS_CACHE_V1),
    "nba":      (config.NBA_CACHE,      config.NBA_CACHE_V1),
    # No v0 half: v0 never scraped basketball-reference, so there is nothing to reuse.
    "bbref":    (None,                  config.BBREF_CACHE_V1),
}

_memo = OrderedDict()               # in-process: same key twice = one disk read
_fetched = set()                    # keys pulled from the WIRE this process
_stats = {"mem": 0, "v0": 0, "v1": 0, "network": 0}


# --- key builders: frozen, v0-compatible ------------------------------------

def snapshot_key(event_id, snapshot_iso, market, book):
    """'{event}_{snapYYYY-MM-DDTHHMMSSZ}_{market}_{book}_{regions}.json'

    The ':' strip is what v0 did, so the names line up exactly.
    """
    tag = f"{snapshot_iso}_{market}_{book}_{config.REGIONS}".replace(":", "").replace("/", "")
    return f"{event_id}_{tag}.json"


def events_key(date):
    """'events_2023-10-27.json' -- one per date, covering every game that day."""
    return f"events_{date}.json"


def nba_key(kind, ident):
    """'box_0022300001.csv' | 'adv_' | 'hustle_' | 'track_' | 'sb_2023-12-31.csv'"""
    return f"{kind}_{ident}.csv"


def bbref_key(kind, ident):
    """'salaries_2024_BOS.csv'. Parsed table, not raw HTML -- same shape as nba_key."""
    return f"{kind}_{ident}.csv"


# --- lookup -----------------------------------------------------------------

def _paths(kind, key):
    v0dir, v1dir = DIRS[kind]
    return (v0dir / key if v0dir else None), (v1dir / key)


def status(kind, key):
    """Where this key lives WITHOUT reading it: 'v0' | 'v1' | None.

    The DRY-run primitive -- lets a loader count exactly what it would fetch
    before spending anything.
    """
    p0, p1 = _paths(kind, key)
    if p0 is not None and p0.exists():
        return "v0"
    if p1.exists():
        return "v1"
    return None


def read_cached(kind, key, fmt="json", read_kwargs=None):
    """Payload if this key is cached (v0 or v1), else None. NEVER fetches.

    Lets a loader run in a strictly-offline mode -- process what is already paid
    for and skip the rest -- with no way to accidentally reach the network.
    """
    memo_key = f"{kind}/{key}"
    if memo_key in _memo:
        _stats["mem"] += 1
        return _memo[memo_key]
    p0, p1 = _paths(kind, key)
    for path, src in ((p0, "v0"), (p1, "v1")):
        if path is not None and path.exists():
            payload = _read(path, fmt, read_kwargs)
            _memo_put(memo_key, payload)
            _stats[src] += 1
            return payload
    return None


def _read(path, fmt, read_kwargs=None):
    if fmt == "csv":
        # read_kwargs carries dtype -- without it a CSV round-trip turns
        # GAME_ID '0022300001' into the integer 22300001 and the '002'
        # regular-season prefix is gone.
        return pd.read_csv(path, **(read_kwargs or {}))
    return json.loads(path.read_text())


def _write(path, payload, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        payload.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(payload))


def _memo_put(key, val):
    _memo[key] = val
    while len(_memo) > MEMO_MAX:
        _memo.popitem(last=False)


def get(kind, key, fetcher, *, api, endpoint, conn=None, params=None, fmt="json",
        read_kwargs=None):
    """Cached fetch. Returns (payload, source) with source in mem|v0|v1|network.

    `fetcher()` must return (payload, meta) and must NOT raise on a legitimate
    empty result -- return an empty payload so it gets negative-cached.
    `meta` may carry http_status / requests_used / requests_remaining.

    Only NETWORK calls are written to the api_calls ledger. Logging every cache
    hit would add millions of rows and bury the thing the ledger is for: actual
    spend. Hits are counted in-process instead -- see `summary()`.
    """
    memo_key = f"{kind}/{key}"
    if memo_key in _memo:
        _stats["mem"] += 1
        return _memo[memo_key], "mem"

    p0, p1 = _paths(kind, key)
    for path, src in ((p0, "v0"), (p1, "v1")):
        if path is not None and path.exists():
            payload = _read(path, fmt, read_kwargs)
            _memo_put(memo_key, payload)
            _stats[src] += 1
            return payload, src

    if memo_key in _fetched:
        raise RuntimeError(
            f"redundant network fetch for {memo_key}: this key was already pulled "
            "from the wire in this process, so it should have been served from "
            "cache. A key builder is almost certainly non-deterministic.")

    payload, meta = fetcher()
    _write(p1, payload, fmt)                  # ALWAYS written, including empties
    _fetched.add(memo_key)
    _memo_put(memo_key, payload)
    _stats["network"] += 1

    if conn is not None:
        db.log_call(conn, api=api, endpoint=endpoint, params=params,
                    http_status=meta.get("http_status"), cache_hit=False,
                    cache_source="v1", cache_path=str(p1),
                    requests_used=meta.get("requests_used"),
                    requests_remaining=meta.get("requests_remaining"),
                    credits_delta=measured_delta(meta, endpoint))
    return payload, "network"


def measured_delta(meta, endpoint):
    """What this call ACTUALLY cost, from the header change since the last call.

    Preferred over any declared rate: it is the provider's own accounting. Falls
    back to CREDITS_PER_CALL only for the first call of a process, when there is
    no previous reading to difference against. A 404 costs 0 and shows up as 0
    here, which a hardcoded rate would have gotten wrong too.
    """
    global _last_used
    used = meta.get("requests_used")
    used = int(used) if used is not None else None
    delta = None
    if used is not None:
        if _last_used is not None:
            delta = used - _last_used
        _last_used = used
    if delta is None:
        return CREDITS_PER_CALL.get(endpoint, 0)
    # A delta far above the declared rate means the seed was STALE, not that this
    # single call cost that much. It happens when report_plan() reads the balance,
    # then something else moves the counter before the first billed call -- a DRY run
    # in another shell, say. Observed once: a first call reported 10 when the counter
    # had moved 30, a 20-credit under-report.
    #
    # The delta is still the truthful figure for what was consumed since the last
    # reading, so it is returned as-is rather than clamped -- under-reporting spend is
    # the worse failure. The warning marks it so a lumpy first row is not mistaken for
    # a rate change.
    rate = CREDITS_PER_CALL.get(endpoint)
    if rate and delta > rate:
        print(f"   (note: {endpoint} delta {delta} exceeds the {rate}/call rate -- "
              f"stale baseline, absorbing {delta - rate} credits spent elsewhere)")
    return delta


# --- credit reporting -------------------------------------------------------

def credits():
    """Live OddsAPI balance -> (used, remaining), or None if unreachable/no key.

    Costs nothing: the /v4/sports endpoint is exempt from the quota.
    """
    if not config.ODDS_API_KEY:
        return None
    try:
        r = requests.get(BALANCE_URL, params={"apiKey": config.ODDS_API_KEY}, timeout=15)
        return int(r.headers["x-requests-used"]), int(r.headers["x-requests-remaining"])
    except Exception:
        return None


def ledger_spend(conn):
    """Credits this project has recorded spending, all time."""
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(sum(credits_delta),0) FROM api_calls "
                    "WHERE api='oddsapi' AND NOT cache_hit")
        return cur.fetchone()[0]


def report_plan(n_calls, endpoint, conn=None):
    """Balance + what this run intends to spend + what would remain. Returns `used`.

    Takes a CALL COUNT and the endpoint, and does the multiplication itself. It
    previously took credits directly, and every caller passed a call count -- so a
    835-call run at 10 credits each was reported as "835 credits". Making the rate
    this function's job means no caller can get it wrong again.

    Also seeds `_last_used`, so the first billed call of the run can be measured
    against a real baseline instead of falling back to the declared rate.
    """
    global _last_used
    rate = CREDITS_PER_CALL.get(endpoint, 0)
    planned = n_calls * rate
    bal = credits()
    if bal is None:
        print(f"OddsAPI credits : (balance unavailable)   "
              f"this run plans: {n_calls} calls x {rate} = {planned:,}")
        return None
    used, remaining = bal
    _last_used = used
    print(f"OddsAPI credits : {used:,} used | {remaining:,} remaining")
    print(f"  this run plans: {n_calls:,} calls x {rate} credits = {planned:,}"
          f"  ->  {remaining - planned:,} would remain")
    if conn is not None:
        print(f"  spent by this project so far (ledger): {ledger_spend(conn):,}")
    return used


def report_spend(before):
    """Actual credits burned since `before` (the value report_plan returned)."""
    bal = credits()
    if bal is None or before is None:
        return
    used, remaining = bal
    print(f"OddsAPI credits : {before:,} -> {used:,}  ({used - before} spent this run) "
          f"| {remaining:,} remaining")


# --- reporting --------------------------------------------------------------

def summary():
    """Per-run cache accounting, for the tail of every loader."""
    free = _stats["mem"] + _stats["v0"] + _stats["v1"]
    return (f"cache: {_stats['v0']} from v0 (already paid), {_stats['v1']} from v1, "
            f"{_stats['mem']} in-process  ->  {free} free, "
            f"{_stats['network']} fetched")


def reset():
    _memo.clear()
    _fetched.clear()
    for k in _stats:
        _stats[k] = 0


def audit(conn):
    """Redundant BILLED calls: same (api, endpoint, params) fetched more than once.

    This should always return zero rows. A non-empty result means money was spent
    twice for the same response -- i.e. the cache was bypassed or a key changed.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT api, endpoint, params, count(*) AS times, min(called_at), max(called_at)
            FROM api_calls
            WHERE NOT cache_hit
            GROUP BY api, endpoint, params
            HAVING count(*) > 1
            ORDER BY count(*) DESC""")
        return cur.fetchall()
