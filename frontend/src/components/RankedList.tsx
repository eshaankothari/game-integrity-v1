import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useDeferredValue, useEffect, useMemo, useState } from "react";
import type { Selection } from "../App";
import { fetchCase, fetchWatchlist, WatchlistRow } from "../api";
import { RED_SCORE, fmtScore, isConfirmed } from "../severity";

// The rail: worst games first. The WHOLE shortlist (~4,800 rows) loads in one
// cached fetch and every filter runs client-side, so search is instant over
// every game, not just the visible top. Only RENDER_CAP rows mount at once --
// the list is for triage, and past ~150 rows the filter is the way down.
//
// One omnibox drives everything. Enter commits the current input as chips;
// each chip is one term, all terms AND together. Two kinds of term:
//   text   -- matches anywhere in the row (player, matchup, pos, tier, date,
//             month name, "pulled")
//   expr   -- field comparisons: min>5, score>=70, pts<3, line>=15,
//             salary>20m, rank<50, team:por, pos:f, month:mar, tier:starter,
//             or the bare keyword `pulled`
// Hovering (or focusing) a row prefetches its case so clicks land instantly.

interface Props {
  selected: Selection | null;
  onSelect: (s: Selection) => void;
  onCollapse: () => void;
}

const LOAD = 5000; // the whole shortlist in one fetch (server cap is higher)
const RENDER_CAP = 150;

function fmtDate(d: string): string {
  return new Date(`${d}T00:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

// Month key ("2023-11") -> short label. One season, so no year suffix.
function monthLabel(key: string): string {
  return new Date(`${key}-01T00:00:00`).toLocaleDateString("en-US", {
    month: "short",
  });
}

// The score as a precise figure over a slim gauge. Deliberately NOT a motion
// component: up to 150 of these re-render on every keystroke of the filter,
// and per-row animation instances are what made the rail laggy.
function ScoreCell({ score, forceHot = false }: { score: number; forceHot?: boolean }) {
  const hot = forceHot || score >= RED_SCORE;
  return (
    <span className="wsc" title={`score ${fmtScore(score)} / 100`}>
      <b className={hot ? "hot" : ""}>{fmtScore(score)}</b>
      <span className="wsc-t">
        <i
          style={{
            background: hot ? "var(--neg)" : "var(--blue)",
            width: `${score}%`,
          }}
        />
      </span>
    </span>
  );
}

// ---- the omnibox grammar --------------------------------------------------

const NUM_FIELDS: Record<string, (r: WatchlistRow) => number | null> = {
  min: (r) => r.minutes,
  minutes: (r) => r.minutes,
  pts: (r) => r.points,
  points: (r) => r.points,
  line: (r) => r.line,
  score: (r) => r.score_100,
  perf: (r) => r.g_performance,
  performance: (r) => r.g_performance,
  market: (r) => r.g_market,
  motive: (r) => r.g_motive,
  salary: (r) => r.salary,
  rank: (r) => r.rank,
  shortfall: (r) => r.shortfall,
};

// Search vocabulary: the matchup only carries "POR @ CHI", so full team
// names are searchable via this map ("blazers", "lakers", "sixers"...).
const TEAM_NAMES: Record<string, string> = {
  ATL: "atlanta hawks", BOS: "boston celtics", BKN: "brooklyn nets",
  CHA: "charlotte hornets", CHI: "chicago bulls", CLE: "cleveland cavaliers",
  DAL: "dallas mavericks", DEN: "denver nuggets", DET: "detroit pistons",
  GSW: "golden state warriors", HOU: "houston rockets", IND: "indiana pacers",
  LAC: "la clippers", LAL: "los angeles lakers", MEM: "memphis grizzlies",
  MIA: "miami heat", MIL: "milwaukee bucks", MIN: "minnesota timberwolves",
  NOP: "new orleans pelicans", NYK: "new york knicks",
  OKC: "oklahoma city thunder", ORL: "orlando magic",
  PHI: "philadelphia 76ers sixers", PHX: "phoenix suns",
  POR: "portland trail blazers", SAC: "sacramento kings",
  SAS: "san antonio spurs", TOR: "toronto raptors", UTA: "utah jazz",
  WAS: "washington wizards",
};
const POS_WORDS: Record<string, string> = {
  G: "guard", F: "forward", C: "center",
};

// "POR @ CHI" -> ["POR", "CHI"]
function matchupCodes(matchup: string): string[] {
  return matchup.split(/[^A-Z]+/).filter((c) => c.length === 3);
}

// "G-F" -> "guard forward"
function posWords(position: string | null): string {
  return (position ?? "")
    .split("-")
    .map((p) => POS_WORDS[p] ?? "")
    .join(" ");
}

// Haystacks are built once per row object and cached: the row objects live
// unchanged in the query cache, and rebuilding 4,810 strings per keystroke
// was measurable lag.
const HAY = new WeakMap<WatchlistRow, string>();
function hay(r: WatchlistRow): string {
  let h = HAY.get(r);
  if (h == null) {
    const d = new Date(`${r.game_date}T00:00:00`);
    h = [
      r.player,
      r.matchup,
      ...matchupCodes(r.matchup).map((c) => TEAM_NAMES[c] ?? ""),
      r.position ?? "",
      posWords(r.position),
      r.tier ?? "",
      r.game_date,
      d.toLocaleDateString("en-US", { month: "long" }), // "march"
      d.toLocaleDateString("en-US", { month: "short", day: "numeric" }), // "mar 24"
      r.line_pulled ? "pulled" : "",
    ]
      .join(" ")
      .toLowerCase();
    HAY.set(r, h);
  }
  return h;
}

interface Term {
  raw: string;
  kind: "expr" | "text";
  pred: (r: WatchlistRow) => boolean;
}

function parseTerm(raw: string): Term {
  const t = raw.trim().toLowerCase();
  if (t === "pulled") {
    return { raw: t, kind: "expr", pred: (r) => r.line_pulled };
  }
  if (t === "confirmed") {
    return {
      raw: t, kind: "expr",
      pred: (r) => isConfirmed(r.player_id, r.game_id),
    };
  }
  // outcome keywords: the verdict chips as filters
  if (t === "under" || t === "under hit") {
    return { raw: "under hit", kind: "expr", pred: (r) => r.under_hit === true };
  }
  if (t === "over" || t === "over hit") {
    return { raw: "over hit", kind: "expr", pred: (r) => r.under_hit === false };
  }
  const m = t.match(/^([a-z_]+)\s*(>=|<=|>|<|=|:)\s*(.+)$/);
  if (m) {
    const [, field, op, rawVal] = m;
    const getter = NUM_FIELDS[field];
    const num = parseFloat(rawVal.replace(/m$/, ""));
    if (getter && op !== ":" && !Number.isNaN(num)) {
      // "salary>20" and "salary>20m" both mean millions
      const v = field === "salary" && num < 10_000 ? num * 1e6 : num;
      return {
        raw: t.replace(/\s+/g, ""),
        kind: "expr",
        pred: (r) => {
          const x = getter(r);
          if (x == null) return false;
          if (op === ">") return x > v;
          if (op === ">=") return x >= v;
          if (op === "<") return x < v;
          if (op === "<=") return x <= v;
          return x === v;
        },
      };
    }
    const val = rawVal.trim();
    // team:por and team:blazers both work; pos:f and pos:forward both work
    const teamPred = (r: WatchlistRow) =>
      r.matchup.toLowerCase().includes(val) ||
      matchupCodes(r.matchup).some((c) =>
        (TEAM_NAMES[c] ?? "").includes(val),
      );
    const posPred = (r: WatchlistRow) =>
      (r.position ?? "").toLowerCase().includes(val) ||
      posWords(r.position).includes(val);
    const cat: Record<string, (r: WatchlistRow) => boolean> = {
      team: teamPred,
      pos: posPred,
      position: posPred,
      month: (r) =>
        monthLabel(r.game_date.slice(0, 7)).toLowerCase() === val.slice(0, 3),
      date: (r) => r.game_date.includes(val),
      tier: (r) => (r.tier ?? "").toLowerCase().includes(val),
    };
    if (cat[field]) {
      return { raw: t.replace(/\s+/g, ""), kind: "expr", pred: cat[field] };
    }
  }
  return { raw: t, kind: "text", pred: (r) => hay(r).includes(t) };
}

// Live input -> terms. The whole input is tried as ONE expression first so
// "min > 5" filters while being typed; otherwise each word is a text token.
function liveTerms(input: string): Term[] {
  const t = input.trim();
  if (t === "") return [];
  const whole = parseTerm(t);
  if (whole.kind === "expr") return [whole];
  return t.split(/\s+/).map(parseTerm);
}

export function RankedList({ selected, onSelect, onCollapse }: Props) {
  const queryClient = useQueryClient();
  const { data, error } = useQuery({
    queryKey: ["watchlist", LOAD],
    queryFn: () => fetchWatchlist(LOAD),
  });

  const [input, setInput] = useState("");
  const [chips, setChips] = useState<Term[]>([]);
  const filtering = chips.length > 0 || input.trim() !== "";

  const rows = data?.rows ?? [];
  // Deferred: the input box updates at typing speed; the 4,810-row filter and
  // 150-row list re-render lag a frame behind instead of blocking keystrokes.
  const deferredInput = useDeferredValue(input);
  const shown = useMemo(() => {
    const active = [...chips, ...liveTerms(deferredInput)];
    if (active.length === 0) return rows;
    return rows.filter((r) => active.every((t) => t.pred(r)));
  }, [rows, chips, deferredInput]);

  const commit = () => {
    const terms = liveTerms(input);
    if (terms.length === 0) return;
    // don't stack duplicate chips
    const known = new Set(chips.map((c) => c.raw));
    setChips([...chips, ...terms.filter((t) => !known.has(t.raw))]);
    setInput("");
  };

  // Auto-open the first VISIBLE row -- with the confirmed filter on by
  // default, that is the worst confirmed game, not the global rank #1.
  useEffect(() => {
    if (shown.length > 0 && selected == null) {
      onSelect({ playerId: shown[0].player_id, gameId: shown[0].game_id });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown]);

  const prefetch = (playerId: number, gameId: string) =>
    queryClient.prefetchQuery({
      queryKey: ["case", playerId, gameId],
      queryFn: () => fetchCase(playerId, gameId),
    });

  return (
    <section className="col" aria-label="Ranked watchlist">
      <div className="coltitle withtool">
        <span>
          {filtering
            ? `Watchlist · ${shown.length.toLocaleString()} of ${rows.length.toLocaleString()} match`
            : `Watchlist · ${data ? data.total.toLocaleString() : "…"} shortlisted games`}
        </span>
        <button className="rail-collapse" onClick={onCollapse}
                title="Collapse the watchlist">
          «
        </button>
      </div>
      <div className="filters">
        <div className="frow">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinejoin="round"
               aria-hidden="true">
            <path d="M3 5h18l-7 8v5l-4 2v-7L3 5z" />
          </svg>
          <input
            type="search"
            placeholder="player, team, min>5, score>70 ⏎ to pin"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Backspace" && input === "" && chips.length > 0) {
                setChips(chips.slice(0, -1));
              }
            }}
            aria-label="Search and filter games"
          />
          {filtering && (
            <button
              className="fclear"
              onClick={() => { setInput(""); setChips([]); }}
            >
              clear ×
            </button>
          )}
        </div>
        {chips.length > 0 && (
          <div className="fchips">
            {chips.map((c) => (
              <button
                key={c.raw}
                className={`fchip ${c.kind === "text" ? "txt" : ""}`}
                title="remove"
                onClick={() => setChips(chips.filter((x) => x.raw !== c.raw))}
              >
                {c.raw}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="scroll">
        {error != null && (
          <div className="status">API unreachable — is uvicorn on :8000? ({String(error)})</div>
        )}
        {!error && !data && <div className="status">Loading…</div>}
        {data && shown.length === 0 && (
          <div className="status">No games match these filters.</div>
        )}
        {/* Plain buttons, no per-row motion: the rail re-renders on every
            keystroke and 150 animation instances made typing lag. */}
        {shown.slice(0, RENDER_CAP).map((r) => {
          const isSel =
            selected?.playerId === r.player_id && selected?.gameId === r.game_id;
          const conf = isConfirmed(r.player_id, r.game_id);
          return (
            <button
              key={`${r.player_id}-${r.game_id}`}
              className={`wrow${conf ? " confirmed" : ""}`}
              aria-current={isSel}
              onClick={() => onSelect({ playerId: r.player_id, gameId: r.game_id })}
              onMouseEnter={() => prefetch(r.player_id, r.game_id)}
              onFocus={() => prefetch(r.player_id, r.game_id)}
            >
              <span className="wrank">
                {r.rank == null ? "—" : String(r.rank).padStart(2, "0")}
              </span>
              <span>
                <span className="wname">
                  {r.player} <span className="wdate">· {fmtDate(r.game_date)}</span>
                  {conf && <span className="conf-tag">CONFIRMED</span>}
                  {r.line_pulled && <span className="pulled-tag">PULLED</span>}
                </span>
                <div className="wsub">
                  {r.points} pts / {r.line} · {r.matchup} · {r.minutes.toFixed(0)} min
                </div>
              </span>
              <ScoreCell score={r.score_100} forceHot={conf} />
            </button>
          );
        })}
        {shown.length > RENDER_CAP && (
          <div className="status">
            Showing the worst {RENDER_CAP} of {shown.length.toLocaleString()}{" "}
            matches — refine the filter to reach the rest.
          </div>
        )}
      </div>
    </section>
  );
}
