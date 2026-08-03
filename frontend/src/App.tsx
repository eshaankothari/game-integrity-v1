import { useQuery } from "@tanstack/react-query";
import { MotionConfig } from "motion/react";
import { useEffect, useState } from "react";
import { fetchSummary } from "./api";
import { CaseView } from "./components/CaseView";
import { CommandPalette } from "./components/CommandPalette";
import { Landing } from "./components/Landing";
import { Methodology } from "./components/Methodology";
import { RankedList } from "./components/RankedList";

// One screen, three columns: ranked rail -> case file -> methodology.
// App-level state is the open player-game, the theme, and the run summary
// (fetched once, shared by the header and the methodology column).

export interface Selection {
  playerId: number;
  gameId: string;
}

export default function App() {
  const [selected, setSelected] = useState<Selection | null>(null);
  const [view, setView] = useState<"home" | "ledger">("home");
  const [dark, setDark] = useState(false);
  const summary = useQuery({ queryKey: ["summary"], queryFn: fetchSummary }).data ?? null;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  // Every door into a case -- rail row, calendar day, 3D node, palette --
  // lands here: select it and put the ledger on screen.
  const openCase = (s: Selection) => {
    setSelected(s);
    setView("ledger");
  };

  // reducedMotion="user" turns every animation into an instant cut when the
  // OS asks for reduced motion -- no per-component checks needed.
  return (
    <MotionConfig reducedMotion="user">
    <div className="app">
      <header>
        <button className="wordmark" onClick={() => setView("home")}
                title="Back to the landing view">
          <h1>
            Game Integrity <em>·</em> 2023–24 Season Ledger
          </h1>
        </button>
        {view === "ledger" ? (
          <button className="toggle" onClick={() => setView("home")}>← season</button>
        ) : (
          <button className="toggle" onClick={() => setView("ledger")}>open ledger →</button>
        )}
        <div className="runmeta">
          {summary && (
            <span>{summary.scored.toLocaleString()} player-games scored</span>
          )}
          <span>screening flags · not findings</span>
          <span className="kbdhint">⌘K to search</span>
          <button className="toggle" onClick={() => setDark(!dark)}>
            {dark ? "white" : "navy"}
          </button>
        </div>
      </header>
      {view === "home" ? (
        <Landing summary={summary} dark={dark} onPick={openCase}
                 onEnterLedger={() => setView("ledger")} />
      ) : (
        <div className="cols">
          <RankedList selected={selected} onSelect={setSelected} />
          <section className="col" aria-label="Case file">
            {selected ? (
              <CaseView playerId={selected.playerId} gameId={selected.gameId} />
            ) : (
              <div className="status">Select a game from the rail.</div>
            )}
          </section>
          <Methodology selected={selected} onSelect={openCase} />
        </div>
      )}
      <CommandPalette
        onSelect={openCase}
        onToggleTheme={() => setDark((d) => !d)}
      />
    </div>
    </MotionConfig>
  );
}
