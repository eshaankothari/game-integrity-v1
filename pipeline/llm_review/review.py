"""L9: an LLM reviewer over the evidence packets -> `player_game_review`.

    packet.build(blind=True)  ->  Claude  ->  {verdict, explanation, evidence, summary}

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.

It does NOT score or rank. The pipeline ranks; this reads. Two reasons, and the second
is the one that actually settles it:

  1. A language model has no calibration on base rates. Asked "is this suspicious?"
     4,810 times it cannot hold the answer to one in a thousand -- and every game on the
     shortlist genuinely looks bad, which is why it is on the shortlist.

  2. THE SCORING TASK HAS NO LABELS AND THE EXPLANATION TASK HAS HUNDREDS. In the top
     200 there are 4 games believed compromised, against 49 whose innocent explanation
     is already known from the data: 29 with an injury DNP within five games, 20 in a
     blowout, 2 ejections. You can measure a reviewer. You cannot measure a ranker.

So its job is to KILL candidates, not to order them: find the innocent explanation, or
state that the evidence does not contain one.

THE ERRORS ARE ASYMMETRIC, and the prompt is built around that. Wrongly explaining a
game away drops a real case; wrongly failing to explain one costs a reviewer five
minutes. So `none_found` is the default, an explanation must cite specific fields by
name, and the model is told plainly that "no innocent explanation" is a normal and
frequent answer rather than a failure to try.

IT RUNS BLIND. The packet omits the score, the rank and every block value. A model told
a game ranks #3 of 15,498 will find reasons it ranks #3 -- that is the prompt talking to
itself, and it would turn one signal into two that look independent.

EVERY CALL IS CACHED through cache.get() with the prompt hash in the key, so a re-run
costs nothing and editing the prompt is a miss rather than a stale hit.

    LLM_MODEL defaults to gemini-2.5-flash-lite. Set LLM_MODEL to any Claude or Gemini
    name and the provider follows automatically.

    python -m pipeline.llm_review.review --dry 5          # build prompts, price them, call nothing
    python -m pipeline.llm_review.review --top 200        # review the top 200 shortlisted games
    python -m pipeline.llm_review.review --player 1629007 --date 2024-03-20
"""
import argparse
import json
import os
import sys
import warnings

import pandas as pd
import requests

from pipeline.core import cache
from pipeline.core import config
from pipeline.core import db
from pipeline.llm_review import packet

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")
# 2,048, not 900. On Gemini 2.5+ the THINKING tokens are drawn from this same budget,
# and they are invisible until you read usageMetadata: the first real call spent 862
# tokens thinking and 19 answering, hit the cap mid-string, and surfaced as
# "unparseable JSON" rather than as "ran out of room". thinkingBudget=0 below removes
# most of that, and the headroom covers the rest.
MAX_TOKENS = 2048
TIMEOUT = 90

# Realistic ANSWER length for this schema -- 3-4 evidence items and a <=4-sentence
# summary. Thinking tokens are billed as output too, which is the other reason to turn
# them off: they were 4x the answer itself.
TYPICAL_OUT = 300

# Fixed taxonomy. An open-ended "why" invites invention; a closed list makes the answer
# countable, comparable across games, and checkable against the 49 known cases.
CAUSES = [
    "injury_or_illness",       # hurt before, during, or after -- carried into the game
    "garbage_time",            # blowout, pulled once the result was decided
    "foul_trouble",
    "ejection",
    "role_change",             # demoted, new rotation, returning from absence
    "matchup_or_scheme",       # defended out of the game, opponent took away his shot
    "blowout_the_other_way",   # his team so far ahead that starters sat
    "ordinary_bad_shooting",   # nothing unusual beyond missing shots he normally makes
    "none_found",              # THE DEFAULT
]

SYSTEM = f"""You review NBA player-games for a sports-integrity screening tool.

Your ONLY job is to find an INNOCENT explanation for a weak performance, or to state \
that the evidence does not contain one. You are not asked whether the game was \
manipulated, you are not shown any suspicion score, and you must not speculate about \
intent. Another system handles that.

Choose exactly one primary cause from this list:
{chr(10).join('  - ' + c for c in CAUSES)}

RULES, in order of importance:

1. `none_found` is the correct answer whenever the packet does not positively support \
one of the other causes. It is expected to be the most common answer. Failing to find \
an explanation is not failing at your job -- it IS your job, done honestly.

2. Every explanation must cite specific fields from the packet by name, with their \
values. "He was probably hurt" with no supporting field is not an explanation; \
"last_off_court_sec 163 of 2880 with 2 stints and no return" is.

3. Do not infer injury from low minutes alone. Low minutes is the single most common \
shape in this data and has many causes. Say `none_found` unless something in the \
packet points at injury specifically.

4. Absent data is not evidence. `health: null` means nobody looked, not that he was \
healthy. `open_line: null` means no opening line was posted, not that it did not move.

5. Write the summary for a human reviewer who will read 150 of these. State what \
happened and what explains it, in plain language, in at most four sentences. No hedging \
language, no restating the box score line by line, no speculation about betting.

Return ONLY a JSON object, no prose around it:

{{"primary_cause": "<one of the list>",
  "confidence": "low" | "medium" | "high",
  "evidence": [{{"field": "<packet field path>", "value": "<value>", "why": "<one clause>"}}],
  "contradicting": [{{"field": "...", "value": "...", "why": "..."}}],
  "summary": "<= 4 sentences for a human reviewer",
  "unusual_notes": "<anything genuinely odd that the taxonomy does not cover, or null>"}}"""

USER_TMPL = """Review this player-game.

{packet}"""

DDL = """
CREATE TABLE IF NOT EXISTS player_game_review (
    player_id     INTEGER NOT NULL REFERENCES players (player_id),
    game_id       TEXT    NOT NULL REFERENCES games (game_id),
    model         TEXT    NOT NULL,
    prompt_hash   TEXT    NOT NULL,
    primary_cause TEXT,
    confidence    TEXT,
    summary       TEXT,
    evidence      JSONB,
    contradicting JSONB,
    unusual_notes TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    reviewed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, game_id, model, prompt_hash)
);
CREATE INDEX IF NOT EXISTS player_game_review_cause_idx
    ON player_game_review (primary_cause);
"""


def _key():
    return (config.GEMINI_API_KEY if config.LLM_PROVIDER == "gemini"
            else config.ANTHROPIC_API_KEY)


def prompt_for(pkt):
    return USER_TMPL.format(packet=json.dumps(pkt, indent=1, default=str))


# Gemini enforces this SERVER-SIDE, so a malformed answer is impossible rather than
# something to parse defensively around. The enum is the same taxonomy the prompt names,
# in one place, so the two cannot drift.
CITE = {"type": "OBJECT",
        "properties": {"field": {"type": "STRING"}, "value": {"type": "STRING"},
                       "why": {"type": "STRING"}},
        "required": ["field", "value", "why"]}
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "primary_cause": {"type": "STRING", "enum": CAUSES},
        "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "ARRAY", "items": CITE},
        "contradicting": {"type": "ARRAY", "items": CITE},
        "summary": {"type": "STRING"},
        "unusual_notes": {"type": "STRING", "nullable": True},
    },
    "required": ["primary_cause", "confidence", "evidence", "summary"],
}


def _call(prompt_text):
    """One API call, either provider. Returns (payload, meta) for cache.get().

    Never raises on a 4xx. An over-length or refused request is a real, permanent answer
    for this packet and must be negative-cached like any other, or every run re-bills it.
    """
    r = _post(prompt_text)
    body = r.json()

    # A 4xx/5xx from the TRANSPORT is not an answer about this packet -- it is a bad
    # key, a retired model, a rate limit. Caching it would freeze a configuration
    # mistake into the cache forever, and the first symptom would be a run that
    # "completes" instantly with every game unreviewed.
    #
    # This is the opposite of the odds API, where a 404 genuinely means "no line was
    # ever posted for this player" -- a permanent fact worth caching. The distinction is
    # whether the error is ABOUT the request or about the connection to the service.
    if r.status_code != 200:
        raise TransportError(f"{config.LLM_PROVIDER} {r.status_code}: {_err(body)}")
    return body, {"http_status": r.status_code}


def _err(body):
    """Google wraps the useful part in error.details and puts a generic sentence in
    error.message. "Request contains an invalid argument" names nothing; the details
    array names the offending field. Print both."""
    e = body.get("error", body)
    if not isinstance(e, dict):
        return str(e)
    out = [str(e.get("message", e))]
    for d in e.get("details", []) or []:
        for v in d.get("fieldViolations", []) or []:
            out.append(f"    field={v.get('field')}  {v.get('description')}")
        if "fieldViolations" not in d:
            out.append(f"    {json.dumps(d)[:300]}")
    return "\n".join(out)


# Turning thinking OFF matters here -- the first call spent 862 tokens thinking and 19
# answering -- but HOW you turn it off is not portable. Gemini 2.5 takes
# thinkingConfig.thinkingBudget; Gemini 3 takes thinkingConfig.thinkingLevel and rejects
# the older field with a bare "Request contains an invalid argument". Neither is
# discoverable from the model list, so the shape is a setting rather than a guess, and
# `--probe` finds the one your model accepts.
THINKING = os.environ.get("LLM_THINKING", "auto")


def _thinking():
    m = config.LLM_MODEL
    if THINKING == "off":
        return {}
    if THINKING == "budget":
        return {"thinkingConfig": {"thinkingBudget": 0}}
    if THINKING == "level":
        return {"thinkingConfig": {"thinkingLevel": "low"}}
    # auto: newer generations use thinkingLevel, 2.x uses thinkingBudget
    if any(m.startswith(f"gemini-{g}") for g in ("3", "4")):
        return {"thinkingConfig": {"thinkingLevel": "low"}}
    return {"thinkingConfig": {"thinkingBudget": 0}}


class TransportError(RuntimeError):
    """Provider-level failure: bad key, retired model, rate limit. Not cached."""


def _post(prompt_text):
    if config.LLM_PROVIDER == "gemini":
        return requests.post(
            GEMINI_URL.format(model=config.LLM_MODEL), timeout=TIMEOUT,
            headers={"x-goog-api-key": config.GEMINI_API_KEY,
                     "content-type": "application/json"},
            json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
                  "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                  "generationConfig": {
                      "temperature": 0,
                      "maxOutputTokens": MAX_TOKENS,
                      "responseMimeType": "application/json",
                      "responseSchema": SCHEMA,
                      **_thinking()}})
    return requests.post(
            ANTHROPIC_URL, timeout=TIMEOUT,
            headers={"x-api-key": config.ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": config.LLM_MODEL, "max_tokens": MAX_TOKENS,
                  "system": SYSTEM, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt_text}]})


def _text_and_usage(payload):
    """(text, input_tokens, output_tokens) from either provider's response shape."""
    if config.LLM_PROVIDER == "gemini":
        cands = payload.get("candidates") or []
        if not cands:
            return None, None, None
        parts = cands[0].get("content", {}).get("parts") or []
        u = payload.get("usageMetadata", {})
        return ("".join(p.get("text", "") for p in parts),
                u.get("promptTokenCount"), u.get("candidatesTokenCount"))
    if "content" not in payload:
        return None, None, None
    u = payload.get("usage", {})
    return ("".join(b.get("text", "") for b in payload["content"]
                    if b.get("type") == "text"),
            u.get("input_tokens"), u.get("output_tokens"))


def _parse(payload):
    """Pull the JSON object out of the response. Anthropic sometimes fences it; Gemini
    cannot, because the schema is enforced server-side."""
    text, _, _ = _text_and_usage(payload)
    if text is None:
        err = payload.get("error", {})
        return None, err.get("message") or str(err) or "no content"

    # Name truncation for what it is. A cut-off response is valid text and invalid JSON,
    # so the naive report is "unparseable" -- which sends you looking at the schema
    # instead of at the token budget.
    fin = ((payload.get("candidates") or [{}])[0].get("finishReason")
           or payload.get("stop_reason"))
    if fin in ("MAX_TOKENS", "max_tokens"):
        u = payload.get("usageMetadata", {})
        th = u.get("thoughtsTokenCount")
        return None, (f"truncated at MAX_TOKENS ({MAX_TOKENS})"
                      + (f"; {th} of them spent thinking" if th else "")
                      + " -- raise MAX_TOKENS or lower thinkingBudget")
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(t), None
    except json.JSONDecodeError as e:
        return None, f"unparseable: {e}"


def review_one(player_id, game_id, conn, dry=False):
    pkt = packet.build(player_id, game_id=game_id, blind=True)
    text = prompt_for(pkt)
    key = cache.llm_key("review", config.LLM_MODEL, SYSTEM + text,
                        f"{player_id}_{game_id}")
    if dry:
        return {"key": key, "chars": len(SYSTEM) + len(text),
                "cached": cache.status("llm", key) is not None}

    payload, src = cache.get(
        "llm", key, lambda: _call(text),
        # The ledger must record which provider was actually billed. This was hardcoded
        # to "anthropic" and logged Anthropic rows for Gemini calls.
        api=config.LLM_PROVIDER,
        endpoint="messages" if config.LLM_PROVIDER == "anthropic" else "generateContent",
        conn=conn, params={"player_id": player_id, "game_id": game_id})
    obj, err = _parse(payload)
    if obj is None:
        return {"key": key, "source": src, "error": err}
    _, in_tok, out_tok = _text_and_usage(payload)
    return {"key": key, "source": src, "obj": obj,
            "in_tok": in_tok, "out_tok": out_tok,
            "prompt_hash": key.split("_")[-3]}


def list_models():
    """What this key can actually call.

    Model availability moves under you -- gemini-2.5-flash-lite returned
    "no longer available to new users" the first time this layer ran. A retired name is
    a 404, and a 404 is indistinguishable from a bad key unless you can see the list.
    """
    if not _key():
        raise SystemExit("no API key set for provider " + config.LLM_PROVIDER)
    if config.LLM_PROVIDER != "gemini":
        raise SystemExit("--models is Gemini-only; see the Anthropic docs for names")
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers={"x-goog-api-key": config.GEMINI_API_KEY}, timeout=30)
    d = r.json()
    if "models" not in d:
        raise SystemExit(f"{r.status_code}: {d}")
    rows = [(m["name"].replace("models/", ""), m.get("displayName", ""),
             m.get("inputTokenLimit"))
            for m in d["models"]
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and m["name"].startswith("models/gemini")]
    lite = [r_ for r_ in rows if "lite" in r_[0]]
    print(f"{len(rows)} callable gemini models\n")
    print("CHEAPEST TIER (lite):")
    for n, dn, lim in sorted(lite):
        mark = "  <- current" if n == config.LLM_MODEL else ""
        print(f"   {n:<38}{str(lim or ''):>9}  {dn}{mark}")
    print("\nOTHER FLASH:")
    for n, dn, lim in sorted(rows):
        if "flash" in n and "lite" not in n and not any(
                x in n for x in ("thinking", "image", "tts", "audio", "live", "native")):
            mark = "  <- current" if n == config.LLM_MODEL else ""
            print(f"   {n:<38}{str(lim or ''):>9}  {dn}{mark}")
    print(f"\n   set one with:  export LLM_MODEL=<name>")


def probe():
    """Send one tiny request per option shape and report which the model accepts.

    Cheaper than reading release notes and more reliable: thinking control, structured
    output and schema support all vary by model generation, none of it appears in the
    model list, and the failure is a bare 400 that names nothing.
    """
    if not _key():
        raise SystemExit("no API key set for provider " + config.LLM_PROVIDER)
    if config.LLM_PROVIDER != "gemini":
        raise SystemExit("--probe is Gemini-only")

    tiny = {"contents": [{"role": "user", "parts": [{"text": "Reply with {\"ok\":true}"}]}]}
    shapes = {
        "bare":                          {},
        "json mime":                     {"responseMimeType": "application/json"},
        "json mime + schema":            {"responseMimeType": "application/json",
                                          "responseSchema": SCHEMA},
        "thinkingBudget 0":              {"thinkingConfig": {"thinkingBudget": 0}},
        "thinkingLevel low":             {"thinkingConfig": {"thinkingLevel": "low"}},
        "schema + thinkingBudget 0":     {"responseMimeType": "application/json",
                                          "responseSchema": SCHEMA,
                                          "thinkingConfig": {"thinkingBudget": 0}},
        "schema + thinkingLevel low":    {"responseMimeType": "application/json",
                                          "responseSchema": SCHEMA,
                                          "thinkingConfig": {"thinkingLevel": "low"}},
    }
    print(f"probing {config.LLM_MODEL}\n")
    ok = []
    for name, gen in shapes.items():
        body = dict(tiny)
        body["generationConfig"] = {"temperature": 0, "maxOutputTokens": 256, **gen}
        r = requests.post(GEMINI_URL.format(model=config.LLM_MODEL), timeout=TIMEOUT,
                          headers={"x-goog-api-key": config.GEMINI_API_KEY,
                                   "content-type": "application/json"}, json=body)
        d = r.json()
        if r.status_code == 200:
            u = d.get("usageMetadata", {})
            print(f"   OK    {name:<28} thought {u.get('thoughtsTokenCount', 0):>4}  "
                  f"out {u.get('candidatesTokenCount', 0):>4}")
            ok.append(name)
        else:
            first = _err(d).splitlines()
            print(f"   {r.status_code}   {name:<28} {first[0][:80]}")
            for ln in first[1:3]:
                print(f"         {ln.strip()[:90]}")
    print()
    if "schema + thinkingLevel low" in ok:
        print("   -> export LLM_THINKING=level")
    elif "schema + thinkingBudget 0" in ok:
        print("   -> export LLM_THINKING=budget")
    elif "json mime + schema" in ok:
        print("   -> export LLM_THINKING=off   (schema works, thinking not controllable)")
    elif "json mime" in ok:
        print("   -> schema REJECTED: drop responseSchema, keep responseMimeType, and "
              "rely on the prompt for shape")
    else:
        print("   -> nothing worked; check the model name with --models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, help="review the top N shortlisted games")
    ap.add_argument("--player", type=int)
    ap.add_argument("--date")
    ap.add_argument("--dry", type=int, nargs="?", const=5, metavar="N",
                    help="build N prompts, price them, call nothing")
    ap.add_argument("--models", action="store_true",
                    help="list the models this key can actually call, and exit")
    ap.add_argument("--probe", action="store_true",
                    help="find which request options this model accepts, and exit")
    a = ap.parse_args()

    if a.models:
        list_models()
        return
    if a.probe:
        probe()
        return

    with db.connect() as conn:
        if a.player:
            rows = pd.read_sql(
                "SELECT player_id, game_id, player, game_date FROM player_game_scores "
                "WHERE player_id=%(p)s AND game_date=%(d)s",
                conn, params={"p": a.player, "d": a.date})
        else:
            n = a.dry if (a.dry and not a.top) else (a.top or 25)
            rows = pd.read_sql(
                "SELECT player_id, game_id, player, game_date FROM player_game_scores "
                "WHERE in_shortlist ORDER BY rank LIMIT %(n)s", conn, params={"n": n})

        if a.dry:
            info = [review_one(int(r.player_id), str(r.game_id), conn, dry=True)
                    for _, r in rows.iterrows()]
            ch = pd.Series([i["chars"] for i in info])
            tok = ch / 3.6
            print(f"DRY RUN -- {len(info)} packets, nothing called")
            print(f"   prompt chars   median {ch.median():>7,.0f}   max {ch.max():>7,.0f}")
            print(f"   input tokens   median {tok.median():>7,.0f}   "
                  f"total {tok.sum():>9,.0f}")
            print(f"   already cached {sum(i['cached'] for i in info)} of {len(info)}")
            print(f"\n   provider {config.LLM_PROVIDER}   model {config.LLM_MODEL}")
            if config.LLM_MODEL not in config.LLM_PRICES:
                med = tok.median()
                print(f"   no price on file for {config.LLM_MODEL}. Per game it is "
                      f"~{med:,.0f} input + ~{TYPICAL_OUT} output tokens;")
                print(f"   for 4,810 games that is {med*4810/1e6:.1f}M input + "
                      f"{TYPICAL_OUT*4810/1e6:.2f}M output.")
                print(f"   Add it to config.LLM_PRICES to see dollars.\n")
            print(f"   {'model':<28}{'top 200':>10}{'4,810':>10}{'15,494':>10}")
            for m, (i_, o_) in config.LLM_PRICES.items():
                cells = "".join(
                    f"{'$'+format(tok.median()*n/1e6*i_ + n*TYPICAL_OUT/1e6*o_, '.2f'):>10}"
                    for n in (200, 4810, 15494))
                mark = "  <-" if m == config.LLM_MODEL else ""
                print(f"   {m:<28}{cells}{mark}")
            print(f"\n   sample key: {info[0]['key']}")
            if not _key():
                var = "GEMINI_API_KEY" if config.LLM_PROVIDER == "gemini" \
                      else "ANTHROPIC_API_KEY"
                print(f"\n   {var} is not set. Export it to run for real:")
                print(f"      export {var}=...")
            return

        if not _key():
            var = "GEMINI_API_KEY" if config.LLM_PROVIDER == "gemini" \
                  else "ANTHROPIC_API_KEY"
            raise SystemExit(
                f"{var} is not set -- nothing was called and nothing was written. "
                f"Use --dry to price the run first.")

        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()

        out, fails = [], 0
        for i, r in rows.iterrows():
            res = review_one(int(r.player_id), str(r.game_id), conn)
            if "obj" not in res:
                fails += 1
                print(f"   !! {r.player} {r.game_date}: {res.get('error')}")
                continue
            o = res["obj"]
            out.append({
                "player_id": int(r.player_id), "game_id": str(r.game_id),
                "model": config.LLM_MODEL, "prompt_hash": res["prompt_hash"],
                "primary_cause": o.get("primary_cause"),
                "confidence": o.get("confidence"), "summary": o.get("summary"),
                "evidence": json.dumps(o.get("evidence")),
                "contradicting": json.dumps(o.get("contradicting")),
                "unusual_notes": o.get("unusual_notes"),
                "input_tokens": res.get("in_tok"), "output_tokens": res.get("out_tok"),
            })
            print(f"   {r.player:<22}{str(r.game_date)[:10]}  "
                  f"{o.get('primary_cause','?'):<24}{o.get('confidence','?'):<8}[{res['source']}]")

        if out:
            n = db.upsert(conn, "player_game_review", out,
                          conflict=["player_id", "game_id", "model", "prompt_hash"])
            print(f"\nupserted {n:,} -> player_game_review   ({fails} failed)")
            c = pd.DataFrame(out).primary_cause.value_counts()
            print("\ncause breakdown:")
            for k, v in c.items():
                print(f"   {k:<26}{v:>4}  ({100*v/len(out):>4.0f}%)")


if __name__ == "__main__":
    main()
