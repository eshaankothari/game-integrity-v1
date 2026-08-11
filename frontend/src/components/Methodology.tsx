import type { Selection } from "../App";
import { PlayerSeason } from "./PlayerSeason";

// The right column is the open player's season roll-up: his mini calendar
// and his other flagged games, so a night is read against a pattern -- plus
// the PDF hand-off for the case on screen.

interface Props {
  selected: Selection | null;
  onSelect: (s: Selection) => void;
}

export function Methodology({ selected, onSelect }: Props) {
  if (!selected) {
    return (
      <section className="col" aria-label="Player season">
        <div className="rc">
          <div className="status">Open a case to see the player's season.</div>
        </div>
      </section>
    );
  }
  return (
    <section className="col" aria-label="Player season">
      <div className="rc">
        <PlayerSeason key={selected.playerId} selected={selected} onSelect={onSelect} />
        <div className="block">
          <h3>Case report</h3>
          <a
            className="toggle dl"
            href={`/api/case/${selected.playerId}/${selected.gameId}/report.pdf`}
          >
            Download PDF →
          </a>
          <div className="blocknote">
            Everything in the case view plus the season-log summary formatted
            into one document for internal documentation and legal case
            development.
          </div>
        </div>
      </div>
    </section>
  );
}
