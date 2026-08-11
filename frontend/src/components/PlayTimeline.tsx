import { useQuery } from "@tanstack/react-query";
import { fetchPlays, nbaEventUrl, PlayEvent } from "../api";

// The player's whole night as a timeline, grouped by quarter, beside the
// shot chart. Marker grammar matches the chart: filled blue = made shot,
// hollow red = miss, quiet gray = everything else (fouls, boards, subs).
// Rows with a clip are links to NBA.com, same as the shot dots.

interface Props {
  playerId: number;
  gameId: string;
}

function Row({ e, gameId }: { e: PlayEvent; gameId: string }) {
  const body = (
    <>
      <span className="pbp-t">{e.clock}</span>
      <i className={`pbp-dot${e.made === true ? " in" : e.made === false ? " out" : ""}`} />
      <span className="pbp-d">{e.description}</span>
      {e.score && <span className="pbp-s">{e.score}</span>}
    </>
  );
  return e.video ? (
    <a className="pbp-row link"
       href={nbaEventUrl(gameId, e.action_number, e.description)}
       target="_blank" rel="noreferrer"
       title="watch on nba.com">
      {body}
    </a>
  ) : (
    <div className="pbp-row">{body}</div>
  );
}

export function PlayTimeline({ playerId, gameId }: Props) {
  const { data } = useQuery({
    queryKey: ["plays", playerId, gameId],
    queryFn: () => fetchPlays(playerId, gameId),
  });
  if (!data || data.plays.length === 0) return null;

  const periods = [...new Set(data.plays.map((p) => p.period))];
  return (
    <div className="pbp" aria-label="Player play-by-play timeline">
      <div className="pbp-title">Video play-by-play</div>
      {periods.map((q) => (
        <div key={q}>
          <div className="pbp-q">{q <= 4 ? `Q${q}` : `OT${q - 4}`}</div>
          {data.plays
            .filter((p) => p.period === q)
            .map((e) => <Row key={e.action_number} e={e} gameId={gameId} />)}
        </div>
      ))}
    </div>
  );
}
