import { useQuery } from "@tanstack/react-query";
import { MotionConfig } from "motion/react";
import { useEffect, useState } from "react";
import { fetchSummary } from "./api";
import { CaseView } from "./components/CaseView";
import { Landing } from "./components/Landing";
import { Logoman } from "./components/Logoman";
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
  const [railOpen, setRailOpen] = useState(true);
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
            <span className="h1-logo"><Logoman height={26} /></span>
            Game Integrity Product <em>·</em> Detecting Insider Trading
          </h1>
        </button>
        {view === "ledger" ? (
          <button className="toggle" onClick={() => setView("home")}>← season</button>
        ) : (
          <button className="toggle" onClick={() => setView("ledger")}>open ledger →</button>
        )}
        <div className="runmeta">
          <span>2023-24 season ledger</span>
          <button className="toggle" onClick={() => setDark(!dark)}>
            {dark ? "light" : "dark"}
          </button>
        </div>
      </header>
      {view === "home" ? (
        <Landing summary={summary} dark={dark} onPick={openCase}
                 onEnterLedger={() => setView("ledger")} />
      ) : (
        <div className={`cols${railOpen ? "" : " rail-closed"}`}>
          {/* The rail stays MOUNTED through the collapse: the grid column
              tweens shut while the content clips (never squishes) and the
              reopen tab cross-fades in over it. Scroll position, filters and
              pinned chips all survive the round trip. */}
          <div className="railwrap">
            <RankedList selected={selected} onSelect={setSelected}
                        onCollapse={() => setRailOpen(false)} />
            <button className="rail-tab" onClick={() => setRailOpen(true)}
                    title="Show the watchlist"
                    aria-hidden={railOpen} tabIndex={railOpen ? -1 : 0}>
              <span>» watchlist</span>
            </button>
          </div>
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
      {/* corner mark: league + vendor, identity only, never interactive.
          Monochrome so it reads as a watermark in both themes. */}
      <div className="brandmark" aria-hidden="true">
        <Logoman height={30} />
        <i />
        <span className="exl">EXL</span>
      </div>
    </div>
    </MotionConfig>
  );
}
