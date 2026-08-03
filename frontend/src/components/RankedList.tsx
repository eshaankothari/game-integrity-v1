import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import type { Selection } from "../App";
import { fetchCase, fetchWatchlist } from "../api";
import { RED_SCORE } from "../severity";

// The rail: worst games first, condensed to fit ~100 on a scroll. Each row is
// rank · player + date (bold) · result/matchup · an animated ring showing the
// composite as a percentage. The three group scores still exist in the case
// view; here the ring carries the single number that ranks the list.
// Hovering (or focusing) a row prefetches its case so clicks land instantly.

interface Props {
  selected: Selection | null;
  onSelect: (s: Selection) => void;
}

function fmtDate(d: string): string {
  return new Date(`${d}T00:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

// The score as a precise figure over a slim animated gauge.
function ScoreCell({ score }: { score: number }) {
  const hot = score >= RED_SCORE;
  return (
    <span className="wsc" title={`score ${score.toFixed(3)}`}>
      <b className={hot ? "hot" : ""}>{score.toFixed(3).replace(/^0/, "")}</b>
      <span className="wsc-t">
        <motion.i
          style={{ background: hot ? "var(--neg)" : "var(--blue)" }}
          initial={{ width: 0 }}
          animate={{ width: `${score * 100}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </span>
    </span>
  );
}

// Month key ("2023-11") -> short label for the month select. One season, so
// the month alone is unambiguous -- no year suffix.
function monthLabel(key: string): string {
  return new Date(`${key}-01T00:00:00`).toLocaleDateString("en-US", {
    month: "short",
  });
}

export function RankedList({ selected, onSelect }: Props) {
  const queryClient = useQueryClient();
  const { data, error } = useQuery({
    queryKey: ["watchlist", 100],
    queryFn: () => fetchWatchlist(100),
  });

  // Client-side filters over the loaded rows. All quiet defaults ("" = off);
  // the search box also matches matchups so a game like "DET @ TOR" is
  // findable by either code or by player name.
  const [q, setQ] = useState("");
  const [team, setTeam] = useState("");
  const [pos, setPos] = useState("");
  const [month, setMonth] = useState("");
  const filtering = q !== "" || team !== "" || pos !== "" || month !== "";

  const rows = data?.rows ?? [];

  // Option lists come from the data itself, so they never drift from it.
  const { teams, positions, months } = useMemo(() => {
    const t = new Set<string>();
    const p = new Set<string>();
    const m = new Set<string>();
    for (const r of rows) {
      for (const code of r.matchup.split(/[^A-Z]+/)) if (code.length === 3) t.add(code);
      if (r.position) p.add(r.position);
      m.add(r.game_date.slice(0, 7));
    }
    return {
      teams: [...t].sort(),
      positions: [...p].sort(),
      months: [...m].sort(),
    };
  }, [rows]);

  const shown = rows.filter(
    (r) =>
      (q === "" ||
        `${r.player} ${r.matchup}`.toLowerCase().includes(q.toLowerCase())) &&
      (team === "" || r.matchup.includes(team)) &&
      (pos === "" || r.position === pos) &&
      (month === "" || r.game_date.startsWith(month)),
  );

  useEffect(() => {
    if (data && data.rows.length > 0 && selected == null) {
      onSelect({ playerId: data.rows[0].player_id, gameId: data.rows[0].game_id });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const prefetch = (playerId: number, gameId: string) =>
    queryClient.prefetchQuery({
      queryKey: ["case", playerId, gameId],
      queryFn: () => fetchCase(playerId, gameId),
    });

  return (
    <section className="col" aria-label="Ranked watchlist">
      <div className="coltitle">
        {filtering
          ? `Watchlist · ${shown.length} of ${rows.length} shown`
          : `Watchlist · worst 100 of ${data ? data.total.toLocaleString() : "…"}`}
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
            placeholder="Search player or matchup…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            aria-label="Search player or matchup"
          />
        </div>
        <div className="fsels">
          <select className={team ? "set" : ""} value={team}
                  onChange={(e) => setTeam(e.target.value)} aria-label="Team">
            <option value="">Team</option>
            {teams.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <select className={pos ? "set" : ""} value={pos}
                  onChange={(e) => setPos(e.target.value)} aria-label="Position">
            <option value="">Pos</option>
            {positions.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <select className={month ? "set" : ""} value={month}
                  onChange={(e) => setMonth(e.target.value)} aria-label="Month">
            <option value="">Month</option>
            {months.map((m) => <option key={m} value={m}>{monthLabel(m)}</option>)}
          </select>
          {filtering && (
            <button
              className="fclear"
              onClick={() => { setQ(""); setTeam(""); setPos(""); setMonth(""); }}
            >
              clear ×
            </button>
          )}
        </div>
      </div>
      <div className="scroll">
        {error != null && (
          <div className="status">API unreachable — is uvicorn on :8000? ({String(error)})</div>
        )}
        {!error && !data && <div className="status">Loading…</div>}
        {data && shown.length === 0 && (
          <div className="status">No games match these filters.</div>
        )}
        {shown.map((r, i) => {
          const isSel =
            selected?.playerId === r.player_id && selected?.gameId === r.game_id;
          return (
            <motion.button
              key={`${r.player_id}-${r.game_id}`}
              className="wrow"
              aria-current={isSel}
              onClick={() => onSelect({ playerId: r.player_id, gameId: r.game_id })}
              onMouseEnter={() => prefetch(r.player_id, r.game_id)}
              onFocus={() => prefetch(r.player_id, r.game_id)}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: "easeOut", delay: Math.min(i * 0.015, 0.35) }}
            >
              <span className="wrank">{String(r.rank).padStart(2, "0")}</span>
              <span>
                <span className="wname">
                  {r.player} <span className="wdate">· {fmtDate(r.game_date)}</span>
                  {r.line_pulled && <span className="pulled-tag">PULLED</span>}
                </span>
                <div className="wsub">
                  {r.points} pts / {r.line} · {r.matchup} · {r.minutes.toFixed(0)} min
                </div>
              </span>
              <ScoreCell score={r.score} />
            </motion.button>
          );
        })}
      </div>
    </section>
  );
}
