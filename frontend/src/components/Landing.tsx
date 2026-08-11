import { AnimatePresence, motion } from "motion/react";
import { lazy, Suspense, useState } from "react";
import type { Selection } from "../App";
import { Summary } from "../api";
import { FunnelStory } from "./FunnelStory";
import { SeasonCalendar } from "./SeasonCalendar";

// three.js is ~1MB minified, so the 3D view is code-split: it downloads only
// the first time someone opens the Anomaly space tab.
const Cloud3D = lazy(() =>
  import("./Cloud3D").then((m) => ({ default: m.Cloud3D })),
);

// The landing page: one headline, one toggle, one visualization at a time.
// Both views are doors into the same place -- click a day or a node and the
// ledger opens on that case.

interface Props {
  summary: Summary | null;
  dark: boolean;
  onPick: (s: Selection) => void;
  onEnterLedger: () => void;
}

export function Landing({ summary, dark, onPick, onEnterLedger }: Props) {
  const [view, setView] = useState<"calendar" | "cloud" | "method">("calendar");

  const tabs = (
    <div className="home-toggle" role="tablist" aria-label="Landing view">
      <button role="tab" aria-selected={view === "calendar"}
              className={view === "calendar" ? "on" : ""}
              onClick={() => setView("calendar")}>
        Season calendar
      </button>
      <button role="tab" aria-selected={view === "cloud"}
              className={view === "cloud" ? "on" : ""}
              onClick={() => setView("cloud")}>
        3D visualization
      </button>
      <button role="tab" aria-selected={view === "method"}
              className={view === "method" ? "on" : ""}
              onClick={() => setView("method")}>
        The algorithm
      </button>
    </div>
  );

  // The method is a full-page piece: the graphic IS the page, so it escapes
  // the centered column entirely -- only the tabs float above it.
  if (view === "method") {
    return (
      <div className="method-full">
        <div className="method-top">{tabs}</div>
        <FunnelStory onEnter={onEnterLedger} />
      </div>
    );
  }

  return (
    <div className="home">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}>
        <h2 className="home-title">
          One season of NBA betting activity,
          <br />
          screened for suspicious games.
        </h2>
        <p className="home-sub">
          {summary
            ? `${summary.scored.toLocaleString()} player-games ranked, scored, and ready for human review.`
            : "Loading run summary…"}
        </p>
      </motion.div>

      {tabs}

      <AnimatePresence mode="wait">
        <motion.div
          key={view}
          className="home-stage"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
        >
          {view === "calendar" ? (
            <>
              <div className="eyebrow">Exhibit · The season, day by day</div>
              <SeasonCalendar onPick={onPick} />
              <div className="source">
              </div>
            </>
          ) : (
            <>
              <div className="eyebrow">Exhibit · Every game in (performance, market, motive)</div>
              <Suspense fallback={<div className="status">Loading anomaly space…</div>}>
                <Cloud3D dark={dark} onPick={onPick} />
              </Suspense>
            </>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
