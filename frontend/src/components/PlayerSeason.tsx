import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import type { Selection } from "../App";
import { fetchCalendar, fetchPlayerFlags } from "../api";
import { RED_SCORE, scoreLevel } from "../severity";

// The selected player's whole season, two ways:
//   1. his scored games ranked worst-first (a pattern reads differently than
//      one bad night -- this is the dossier's player roll-up)
//   2. a mini calendar: the full season's game days as a week-column grid,
//      HIS scored games colored by composite, the open case outlined.
// Clicking either jumps the case view to that game.

interface Props {
  selected: Selection;
  onSelect: (s: Selection) => void;
}

const DAY_MS = 86_400_000;

export function PlayerSeason({ selected, onSelect }: Props) {
  const flags = useQuery({
    queryKey: ["player", selected.playerId],
    queryFn: () => fetchPlayerFlags(selected.playerId),
  });
  // Season span comes from the shared calendar cache -- likely already warm
  // from the landing page; the grid needs it so every player sits on the
  // same October-to-April canvas.
  const season = useQuery({ queryKey: ["calendar"], queryFn: fetchCalendar });

  if (!flags.data || !season.data) return null;
  const rows = flags.data.rows;
  const byDate = new Map(rows.map((r) => [r.game_date, r]));

  const first = new Date(`${season.data.days[0].date}T00:00:00`);
  const lastDay = season.data.days[season.data.days.length - 1].date;
  const last = new Date(`${lastDay}T00:00:00`);
  const start = new Date(first.getTime() - first.getDay() * DAY_MS);
  const weeks: Date[][] = [];
  for (let w = new Date(start); w <= last; w = new Date(w.getTime() + 7 * DAY_MS)) {
    weeks.push(Array.from({ length: 7 }, (_, i) => new Date(w.getTime() + i * DAY_MS)));
  }
  const iso = (d: Date) => d.toISOString().slice(0, 10);

  return (
    <div className="block">
      <h3>{flags.data.player} · {rows.length} scored game{rows.length === 1 ? "" : "s"}</h3>

      <div className="mini-cal">
        {weeks.map((week, wi) => (
          <div className="cal-week" key={wi}>
            {week.map((day) => {
              const r = byDate.get(iso(day));
              const isCur = r?.game_id === selected.gameId;
              return (
                <motion.button
                  key={iso(day)}
                  className={`cal-cell mini ${r ? scoreLevel(r.score) : "lv0"} ${isCur ? "cur" : ""}`}
                  disabled={!r}
                  title={r ? `${r.game_date} · ${r.points} pts / ${r.line} · ${r.score.toFixed(3)}` : undefined}
                  onClick={() => r && onSelect({ playerId: r.player_id, gameId: r.game_id })}
                  whileHover={r ? { scale: 1.5 } : undefined}
                  transition={{ type: "tween", duration: 0.12, ease: "easeOut" }}
                />
              );
            })}
          </div>
        ))}
      </div>

      <table className="retr" style={{ marginTop: 10 }}>
        <tbody>
          {rows.slice(0, 8).map((r) => {
            const isCur = r.game_id === selected.gameId;
            return (
              <motion.tr
                key={r.game_id}
                className={isCur ? "flagrow cur" : "flagrow"}
                onClick={() => onSelect({ playerId: r.player_id, gameId: r.game_id })}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.25 }}
              >
                <td>
                  {r.game_date.slice(5)} · {r.matchup} · {r.points}/{r.line}
                </td>
                <td className={r.score >= RED_SCORE ? "hot" : ""}>
                  {r.score.toFixed(3)}
                </td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
      <div className="blocknote">
        Source: out/scored.csv for this player · worst score first
        {rows.length > 8 && ` · showing 8 of ${rows.length}`}
        {" "}· repeated flags read differently than one bad night
      </div>
    </div>
  );
}
