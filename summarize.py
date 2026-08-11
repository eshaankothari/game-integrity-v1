"""Deterministic prose over an evidence packet. No model, no key, no network.

Same job as the L9 reviewer -- say what happened and what might explain it -- done with
rules. Three reasons it exists even though the LLM layer does:

  1. IT IS THE BASELINE THE LLM HAS TO BEAT. A reviewer's summary is only worth paying
     for if it says more than a template can. Without this you cannot tell whether the
     model is reading the evidence or restating the box score in nicer sentences.

  2. IT CANNOT INVENT. Every sentence is emitted from a condition on a specific field.
     There is no path here that produces a claim the packet does not support, which is
     the failure mode that makes a generated summary next to a suspicion score
     dangerous.

  3. IT RUNS ON ALL 15,494 IN A SECOND, FREE. The LLM is worth scoping to a shortlist;
     this is not worth scoping at all.

WHAT IT DOES NOT DO. It does not judge. `flags` names the innocent explanations the
DATA supports, using the same taxonomy as review.py so the two are directly comparable,
and `none_found` when it supports none. That is a statement about the evidence, not
about the player.

    python summarize.py 1629007 2024-03-20
    python summarize.py --top 20
"""
import argparse
import warnings

import packet

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

# Thresholds. Named and gathered so a reader can disagree with one rather than hunt for
# it inline. Each is the conventional line, not a fitted value.
BLOWOUT = 20            # points; the usual definition of a decided game
FOUL_TROUBLE = 5        # a 6th is a foul-out, a 5th changes how a coach plays you
EARLY_EXIT_SEC = 2400   # left before the 40-minute mark of a 48-minute game
# A blowout only explains a quiet night if he was still out there WHEN it became one.
# Porter left in the first quarter of a game that ended 34 apart -- the final margin
# says nothing about why a player was gone before it existed.
GARBAGE_MIN_EXIT = 1800   # 30:00; earlier than that and the final margin is not a reason
BIG_MISS = 0.60         # shortfall: scored under 40 percent of his line
NOTABLE_Z = 1.0         # standard deviations from his own norm worth a sentence


def _s(v, d=1):
    return "—" if v is None else f"{v:,.{d}f}"


def _plural(n, word):
    """'1 rebound', not '1 rebounds'. Small, but a summary that cannot get agreement
    right reads as machine output and stops being trusted on the parts that matter."""
    if n is None:
        return f"— {word}s"
    n = int(n)
    return f"{n} {word}" + ("" if n == 1 else "s")


# `tier` is a column name. A reader wants "a bench player", not "(bench tier)".
ROLE = {"bench": "a bench player", "rotation": "a rotation player",
        "starter": "a starter"}


def _mmss(sec):
    return "—" if sec is None else f"{int(sec)//60}:{int(sec) % 60:02d}"


def flags(p):
    """Innocent explanations the DATA supports. Same taxonomy as review.py.

    Order matters: the first match is the primary cause, because these are not mutually
    exclusive -- a player can be ejected during a blowout -- and the more specific
    mechanism should win over the more general circumstance.
    """
    g, t, c, o = p["game_context"], p["this_game"], p["on_off_court"], []
    g = g or {}; t = t or {}; c = c or {}

    if g.get("ejected"):
        o.append(("ejection", "he was ejected"
                  + (" alone" if g.get("ejected_alone") else " in a group incident")))
    if (t.get("fouls") or 0) >= FOUL_TROUBLE:
        o.append(("foul_trouble", f"{int(t['fouls'])} fouls"))
    if (g.get("final_game_margin") or 0) >= BLOWOUT:
        exit_sec = c.get("last_off_court_sec")
        garbage_pts = g.get("points_garbage")
        # Require evidence he was ON THE FLOOR late, or that he actually banked points in
        # garbage time. A large final margin alone is a fact about the game, not an
        # explanation for a player who had already left.
        if (exit_sec is not None and exit_sec >= GARBAGE_MIN_EXIT) or garbage_pts:
            share = (f", {int(garbage_pts)} of his {int(t['points'])} in garbage time"
                     if garbage_pts and (t.get("points") or 0) > 0 else "")
            o.append(("garbage_time",
                      f"the game finished {int(g['final_game_margin'])} apart{share}"))
    if (p.get("health") or {}) and (p["health"] or {}).get("injury_dnp_within_5"):
        o.append(("injury_or_illness", "an injury DNP within five games"))
    # NOT listed here even though it is real: the closing context sentence already
    # says it, and repeating it makes a soft circumstance look like a second,
    # independent finding.
    #   if g.get("back_to_back"): ...

    # Deliberately NOT an explanation on its own. A short night is the most common shape
    # in this data and has many causes; it is reported as a fact and left uncoloured.
    if not o:
        o.append(("none_found",
                  "nothing in the packet explains it: no ejection, no foul trouble, "
                  "no blowout, no recorded injury"))
    return o


def summarize(p):
    """FOUR SHORT SENTENCES. Terse on purpose.

    The case view already renders the box score, the residual z-scores, the line
    movement, the salary and the season strip. Prose that restates any of those is
    words a reviewer has to read past to reach the part only this file knows:

        - how the night compares to HIS OWN season, in one clause
        - when he left the floor and whether he came back
        - what he stood to forfeit
        - WHETHER ANYTHING IN THE DATA EXPLAINS IT   <- the reason the summary exists
        - the circumstances of the evening

    Anything already on screen is omitted even when it is interesting.
    """
    i, t, b, v, m, g, o = (p["identity"], p["this_game"], p["his_season"],
                           p["vs_baselines"], p["market"], p["game_context"],
                           p["on_off_court"])
    L = []

    # --- the night against his own season, in one line ----------------------
    role = ROLE.get(i.get("role_tier"), "a player")
    ng = int(b.get("propped_games") or 0)
    head = (f"{_s(t['points'], 0)} points on a {_s(m['close_line'])} line in "
            f"{_s(t['minutes'])} minutes. As {role} he averages "
            f"{_s(b['avg_points'])} on a {_s(b['avg_line'])} line")
    head += (f" and went under in {_s(b['unders'], 0)} of {ng}."
             if ng > 1 else ".")
    if v.get("game_z_own") is not None and abs(v["game_z_own"]) >= NOTABLE_Z:
        head += (f" This game sits {abs(v['game_z_own']):.1f}\u03c3 "
                 f"{'below' if v['game_z_own'] < 0 else 'above'} that norm.")
    L.append(head)

    # --- when he left. The one thing no other panel shows. -------------------
    ex = o.get("last_off_court_sec")
    if ex is None:
        L.append("No substitution record for him, so when he left the floor is "
                 "unknown.")
    elif ex < EARLY_EXIT_SEC and not g.get("ejected"):
        L.append(f"Off the floor at {_mmss(ex)} of 48:00 and did not return.")

    # --- what he stood to forfeit -------------------------------------------
    if i.get("salary_listed") is False:
        L.append("On a two-way or 10-day deal \u2014 the least to forfeit of anyone "
                 "on a roster.")
    elif i.get("salary_usd"):
        pct = i.get("salary_percentile")
        band = ""
        if pct is not None:
            band = (", the bottom quarter of NBA salaries" if pct < .25 else
                    ", the bottom half" if pct < .5 else
                    ", the top quarter" if pct > .75 else "")
        L.append(f"Paid ${i['salary_usd']/1e6:.1f}M for the season{band}.")

    # --- the finding ---------------------------------------------------------
    f = flags(p)
    L.append("Nothing in the data explains it: no ejection, foul trouble, blowout or "
             "recorded injury." if f[0][0] == "none_found"
             else "Explained by " + "; ".join(w for _, w in f) + ".")

    L.append(_context(g))
    return " ".join(x for x in L if x)


def _context(g):
    """One clause. Where, on how much rest, and whether the game stayed live -- the
    three circumstances that change how a quiet night should read."""
    bits = []
    if g.get("is_home") is not None:
        bits.append("At home" if g["is_home"] else "On the road")
    if g.get("back_to_back"):
        bits.append("on the second of a back-to-back")
    elif g.get("rest_days") is not None:
        r = int(g["rest_days"])
        bits.append("on no rest" if r == 0 else
                    "on one day of rest" if r == 1 else f"on {r} days of rest")
    lead = " ".join(bits)

    margin = g.get("final_game_margin")
    if margin is not None:
        tail = (f"the game stayed within {int(margin)}" if margin < BLOWOUT
                else f"the game ended {int(margin)} apart")
        return (f"{lead}; {tail}." if lead else tail.capitalize() + ".")
    return lead + "." if lead else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("player_id", nargs="?", type=int)
    ap.add_argument("game_date", nargs="?")
    ap.add_argument("--top", type=int, help="summarize the top N shortlisted games")
    a = ap.parse_args()

    if a.top:
        import pandas as pd
        import db
        with db.connect() as conn:
            rows = pd.read_sql("SELECT player_id, game_id, rank FROM player_game_scores "
                               "WHERE in_shortlist ORDER BY rank LIMIT %(n)s",
                               conn, params={"n": a.top})
        for _, r in rows.iterrows():
            p = packet.build(int(r.player_id), game_id=str(r.game_id))
            print(f"{'='*78}\n#{int(r['rank'])}  [{flags(p)[0][0]}]\n{'='*78}")
            print(summarize(p), "\n")
        return

    if a.player_id is None:
        raise SystemExit("give a player_id and game_date, or --top N")
    print(summarize(packet.build(a.player_id, a.game_date)))


if __name__ == "__main__":
    main()
