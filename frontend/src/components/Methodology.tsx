import type { Selection } from "../App";
import { PlayerSeason } from "./PlayerSeason";

// The right column, now a single job: the selected player's season roll-up --
// his mini calendar and his other flagged games. The funnel, distribution and
// isolation views live on the landing page.

interface Props {
  selected: Selection | null;
  onSelect: (s: Selection) => void;
}

export function Methodology({ selected, onSelect }: Props) {
  return (
    <section className="col" aria-label="Player season">
      <div className="rc">
        {selected && (
          <PlayerSeason key={selected.playerId} selected={selected} onSelect={onSelect} />
        )}
        {selected && (
          <div className="block">
            <h3>Case report</h3>
            <a
              className="toggle dl"
              href={`/api/case/${selected.playerId}/${selected.gameId}/report.pdf`}
            >
              download PDF →
            </a>
            <div className="blocknote">
              Everything on this screen plus the season-log summary and the
              player roll-up, formatted for hand-off. Carries the
              screening-flag caveat and data provenance.
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
