import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import type { Selection } from "../App";
import { CalendarDay, fetchCalendar } from "../api";
import { scoreLevel } from "../severity";

// GitHub-style season calendar: one cell per game day, October to April,
// shaded by the worst composite scored that day on the UI-wide severity
// scale, so red here means the same extreme tail it means everywhere else.

interface Props {
  onPick: (s: Selection) => void;
}

const DAY_MS = 86_400_000;

// Deliberately stricter red than the rest of the UI: 187 game days shaded by
// the shared 73.7 line drowned the calendar in red. A DAY goes red only when
// it holds at least one 80+ game; days that are red elsewhere step down to
// the solid-blue band here.
function level(d: CalendarDay | undefined): string {
  if (!d) return "lv0";
  if (d.max_score >= 80) return "rev";
  const l = scoreLevel(d.max_score);
  return l === "rev" ? "lv3" : l;
}

export function SeasonCalendar({ onPick }: Props) {
  const { data } = useQuery({ queryKey: ["calendar"], queryFn: fetchCalendar });
  if (!data) return <div className="status">Loading calendar…</div>;

  const byDate = new Map(data.days.map((d) => [d.date, d]));
  const first = new Date(`${data.days[0].date}T00:00:00`);
  const last = new Date(`${data.days[data.days.length - 1].date}T00:00:00`);
  // Start on the Sunday of the first week so columns are whole weeks.
  const start = new Date(first.getTime() - first.getDay() * DAY_MS);

  const weeks: Date[][] = [];
  for (let w = new Date(start); w <= last; w = new Date(w.getTime() + 7 * DAY_MS)) {
    weeks.push(Array.from({ length: 7 }, (_, i) => new Date(w.getTime() + i * DAY_MS)));
  }

  const iso = (d: Date) => d.toISOString().slice(0, 10);
  const monthLabels = weeks.map((week, i) => {
    const m = week[0].toLocaleString("en-US", { month: "short" });
    const prev = i > 0 ? weeks[i - 1][0].toLocaleString("en-US", { month: "short" }) : "";
    return m !== prev ? m : "";
  });

  return (
    <div className="cal">
      <div className="cal-months">
        {monthLabels.map((m, i) => (
          <span key={i}>{m}</span>
        ))}
      </div>
      <div className="cal-grid">
        {weeks.map((week, wi) => (
          <div className="cal-week" key={wi}>
            {week.map((day) => {
              const d = byDate.get(iso(day));
              return (
                <motion.button
                  key={iso(day)}
                  className={`cal-cell ${level(d)}`}
                  disabled={!d}
                  title={
                    d
                      ? `${d.date} · ${d.n} scored · worst ${d.max_score.toFixed(1)} (${d.player})`
                      : iso(day)
                  }
                  onClick={() => d && onPick({ playerId: d.player_id, gameId: d.game_id })}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  whileHover={d ? { scale: 1.35 } : undefined}
                  transition={{
                    opacity: { duration: 0.25, delay: Math.min(wi * 0.015, 0.5) },
                    // tween, not the default spring: springs oscillate around
                    // the target and a 12px square shimmers while they settle
                    scale: { type: "tween", duration: 0.12, ease: "easeOut" },
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>
      <div className="cal-legend">
        <span>quiet</span>
        <i className="cal-cell lv1" /><i className="cal-cell lv2" /><i className="cal-cell lv3" /><i className="cal-cell rev" />
        <span>flag · click a day to open its worst case</span>
      </div>
    </div>
  );
}
