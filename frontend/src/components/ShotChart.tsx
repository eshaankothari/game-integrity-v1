import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { fetchShots, nbaEventUrl } from "../api";

// Half-court shot chart from the cached play-by-play, hoop at the top.
// Legacy coordinates: hoop at (0,0), 1 unit = 0.1 ft, x -250..250,
// y -52 (baseline) .. 418 (half-court line). Filled blue = made,
// hollow red = missed -- the page's severity grammar, unchanged.
// A shot with a clip is a plain link straight to NBA.com's event page,
// which plays the video -- no modal, no blocked fetches.

interface Props {
  playerId: number;
  gameId: string;
}

// 3pt arc as a polyline: radius 237.5 between the corner verticals at ±220.
// Computed, not an SVG arc, so there is no sweep-flag ambiguity.
const ARC = Array.from({ length: 89 }, (_, i) => {
  const x = -220 + (i * 440) / 88;
  return `${x.toFixed(1)},${Math.sqrt(237.5 ** 2 - x * x).toFixed(1)}`;
}).join(" ");

export function ShotChart({ playerId, gameId }: Props) {
  const { data } = useQuery({
    queryKey: ["shots", playerId, gameId],
    queryFn: () => fetchShots(playerId, gameId),
  });

  if (!data) return null;
  if (data.shots.length === 0) {
    return (
      <p className="source">
        {data.source === "cached play-by-play"
          ? "Zero field-goal attempts this game — for a player with a posted points line, that is itself part of the case."
          : "No play-by-play cached for this game."}
      </p>
    );
  }

  const made = data.shots.filter((s) => s.made).length;
  return (
    <>
      <div className="shotwrap">
        {/* cropped at y=350: the empty backcourt band above the arc was dead
            space. The rare deep heave clamps to the bottom edge. */}
        <svg viewBox="-250 -52 500 402" className="court" role="img"
             aria-label={`Shot chart: ${made} of ${data.shots.length} field goals made`}>
          {/* court, all hairline */}
          <rect x="-250" y="-52" width="500" height="402" fill="none" />
          <line x1="-30" y1="-12" x2="30" y2="-12" />
          <circle cx="0" cy="0" r="7.5" />
          <rect x="-80" y="-52" width="160" height="190" fill="none" />
          <circle cx="0" cy="138" r="60" fill="none" />
          <line x1="-220" y1="-52" x2="-220" y2="89" />
          <line x1="220" y1="-52" x2="220" y2="89" />
          <polyline points={ARC} fill="none" />
          {/* shots; ones with a clip are plain links to NBA.com's player */}
          {data.shots.map((s) => {
            const dot = (
              <motion.circle
                key={s.action_number}
                className={`shot ${s.made ? "in" : "out"} ${s.video ? "clip" : ""}`}
                cx={s.x} cy={Math.min(s.y, 340)} r="6"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.25, delay: 0.02 * (s.period - 1) }}
              >
                <title>
                  {`Q${s.period} ${s.clock} · ${s.description}${s.video ? " · click to watch on nba.com" : ""}`}
                </title>
              </motion.circle>
            );
            return s.video ? (
              <a key={s.action_number}
                 href={nbaEventUrl(gameId, s.action_number, s.description)}
                 target="_blank" rel="noreferrer"
                 aria-label={`Watch: ${s.description}`}>
                {/* invisible oversized hit target: a hollow miss otherwise
                    only registers clicks on its 2px stroke ring */}
                <circle className="shot-hit" cx={s.x} cy={Math.min(s.y, 340)} r="13">
                  <title>
                    {`Q${s.period} ${s.clock} · ${s.description} · click to watch on nba.com`}
                  </title>
                </circle>
                {dot}
              </a>
            ) : dot;
          })}
        </svg>
        <div className="shotkey">
          <i className="k-in" /> made ({made})
          <i className="k-out" /> missed ({data.shots.length - made})
        </div>
      </div>
    </>
  );
}
