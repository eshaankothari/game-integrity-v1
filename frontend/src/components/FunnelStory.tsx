import { useQuery } from "@tanstack/react-query";
import {
  AnimatePresence,
  motion,
  MotionValue,
  useMotionValueEvent,
  useScroll,
  useTransform,
} from "motion/react";
import { useRef, useState } from "react";
import { fetchFunnel } from "../api";

// The methodology as a horizontally traveling immersive. The chart is laid
// out wider than the viewport; scrolling moves the CAMERA rightward along the
// flow, so the reader physically follows the games through each cut. The
// caption card stays pinned bottom-left and swaps in the direction of travel.
// At the finale the camera pulls back to show the whole flow, the cut-chain
// dims, and the gate ribbon sweeps straight to the ledger.
//
// One narration step per REAL gate: the counts come live from /api/funnel
// (cut_failed in player_game_scores), so the story cannot drift from the
// pipeline the way a hardcoded version once did.

interface Props {
  onEnter: () => void;
}

const NARRATION: Array<{ title: string; body: string }> = [
  {
    title: "It starts with every player-game of 2023–24.",
    body: "32,385 player-games were logged league-wide. For 15,498 of them a sportsbook posted a points line — only those can be judged against a market, so they are the population. Nothing has been eliminated yet.",
  },
  {
    title: "Cut 1 · Game score — the good nights leave.",
    body: "Each game is measured against the player's own season, never the league's — against the league a star's disaster looks ordinary. The top 25% by game score were good nights for him, and a good night is not a case: 3,903 games out.",
  },
  {
    title: "Cut 2 · Effort — the fully involved leave.",
    body: "Missing shots is common and mostly innocent. Running less, touching the ball less, using fewer possessions while still on the floor is the harder thing to explain. The top quarter by involvement — usage, touches, distance, minutes — drops 1,935 more.",
  },
  {
    title: "Cut 3 · Market lean — the over-leaning games leave.",
    body: "If the closing price leaned toward the over, under-side money never showed up — and this is a screen for under-side pressure. The bottom 25% by market lean exits: 2,314 games.",
  },
  {
    title: "Cuts 4–5 · The line must not move up.",
    body: "A line drifting upward, or a price shortening on the over while the line holds, means money arrived on the OVER side — the opposite of the pattern being screened for. Games with upward movement leave (unknowns are kept, not guessed): 410, then 903 more.",
  },
  {
    title: "Cut 6 · Salary — who had something to lose?",
    body: "Everyone above $20M drops — the money is not worth the career. No listed salary is kept deliberately: that marks a two-way or 10-day contract, the lowest-paid, most exposed profile in the league. 4,810 player-games survive every gate. That is the shortlist.",
  },
  {
    title: "A cut hides. A score never deletes.",
    body: "Every one of the 15,498 propped games still carries a score — the gates only choose what the rail shows first, and the confirmed Beasley and Porter games land at ranks 17, 89, 168, 214 and 271 of the shortlist. Ranked, searchable, ready for review.",
  },
];

// geometry: 8 nodes, ~2.3 viewports wide; the viewBox is the camera window.
// XS[0] leaves margin so the first count ("32,385", centered on the node)
// never clips the left edge on load.
const VIEW_W = 880;
const VIEW_H = 430;
const XS = [70, 310, 550, 790, 1030, 1270, 1510, 1750];
const FLOW_W = 1790;
const YC = 200; // flow sits low enough that the tallest band's label clears the top
const HMAX = 260;
const NODE_W = 7;
const ELIM_Y = 400;
const SEGS = XS.length - 1; // ribbon segments between nodes
const LAST = NARRATION.length - 1;

const PAN_END = 0.78;         // scroll fraction spent traveling right
const FINALE = 0.84;          // where the pull-back + gate sweep begins

function bandH(n: number, total: number) {
  return Math.max((n / total) * HMAX, 5);
}

function ribbon(x0: number, h0: number, x1: number, h1: number, yc0 = YC, yc1 = YC) {
  const mx = (x0 + x1) / 2;
  const t0 = yc0 - h0 / 2, b0 = yc0 + h0 / 2;
  const t1 = yc1 - h1 / 2, b1 = yc1 + h1 / 2;
  return `M ${x0} ${t0} C ${mx} ${t0}, ${mx} ${t1}, ${x1} ${t1}
          L ${x1} ${b1} C ${mx} ${b1}, ${mx} ${b0}, ${x0} ${b0} Z`;
}

function ScrollRibbon({ progress, range, d }: {
  progress: MotionValue<number>;
  range: [number, number];
  d: string;
}) {
  const clip = useTransform(progress, range, ["inset(0 100% 0 0)", "inset(0 0% 0 0)"]);
  return <motion.path d={d} fill="var(--blue)" opacity={0.34} style={{ clipPath: clip }} />;
}

function ElimWedge({ progress, range, d, label, x }: {
  progress: MotionValue<number>;
  range: [number, number];
  d: string;
  label: string;
  x: number;
}) {
  const opacity = useTransform(progress, range, [0, 1]);
  return (
    <motion.g style={{ opacity }}>
      <path d={d} fill="url(#elim)" />
      <text x={x} y={ELIM_Y + 16} className="fstory-elim">{label}</text>
    </motion.g>
  );
}

export function FunnelStory({ onEnter }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: containerRef });
  const [step, setStep] = useState(0);

  useMotionValueEvent(scrollYProgress, "change", (v) => {
    setStep(Math.max(0, Math.min(LAST, Math.round(v * (LAST + 0.4)))));
  });

  // THE CAMERA. Travel right along the flow until PAN_END, then pull back:
  // translate returns to 0 while scale shrinks until the whole flow fits.
  // camX HOLDS at the right end until the scale-down starts, then both run
  // over the same window -- returning early made the flow slide visibly
  // rightward before the zoom-out kicked in.
  const camX = useTransform(scrollYProgress,
    [0.04, PAN_END, FINALE, FINALE + 0.12],
    [0, -(FLOW_W - VIEW_W + 15), -(FLOW_W - VIEW_W + 15), 0]);
  const camScale = useTransform(scrollYProgress,
    [FINALE, FINALE + 0.12], [1, (VIEW_W - 8) / FLOW_W]);
  const veilOpacity = useTransform(scrollYProgress, [FINALE + 0.06, FINALE + 0.12], [0, 0.72]);
  const gateClip = useTransform(scrollYProgress, [FINALE + 0.08, 0.98],
    ["inset(0 100% 0 0)", "inset(0 0% 0 0)"]);

  const funnel = useQuery({ queryKey: ["funnel"], queryFn: fetchFunnel });

  // NEVER early-return before .scrolly mounts: useScroll binds to the
  // container on first render, and if the element doesn't exist yet (query
  // still loading), scroll tracking silently never attaches -- which is why
  // the piece used to scroll only on the second visit. The container always
  // renders; only the chart waits for data.
  const ready = funnel.data != null;
  // all games -> propped -> the six gates of the shortlist, in pipeline
  // order. Any stage past the salary gate (experiments) is deliberately not
  // part of this story.
  const counts = ready ? funnel.data!.stages.slice(0, 8).map((s) => s.n) : [];
  const total = ready ? counts[0] : 1;
  const labels = ["all games", "propped", "game score", "effort",
                  "market lean", "line move", "price move", "shortlist"];
  const gateD = ready
    ? ribbon(XS[0] + NODE_W, bandH(total, total),
             XS[SEGS], bandH(counts[SEGS], total))
    : "";

  return (
    <div className="scrolly" ref={containerRef}>
      <div className="scrolly-stage">
        <motion.div className="scrolly-progressbar" style={{ scaleX: scrollYProgress }} />
        {!ready && <div className="status">Loading methodology…</div>}
        {ready && (
        <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="fstory-svg"
             preserveAspectRatio="xMidYMid meet" role="img"
             aria-label="Screening funnel; scrolling travels along the flow">
          <defs>
            <linearGradient id="elim" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" style={{ stopColor: "var(--faint)", stopOpacity: 0.35 }} />
              <stop offset="1" style={{ stopColor: "var(--faint)", stopOpacity: 0 }} />
            </linearGradient>
          </defs>

          {/* everything rides the camera */}
          <motion.g style={{ x: camX, scale: camScale, transformOrigin: "0px 215px" }}>
            {/* cut ribbons + eliminated wedges, revealed by scroll slice */}
            {counts.slice(0, -1).map((n, i) => {
              const kept = counts[i + 1];
              const elimH = bandH(n - kept, total);
              const s = (i + 0.1) * (PAN_END / SEGS);
              const e = (i + 0.9) * (PAN_END / SEGS);
              return (
                <g key={i}>
                  <ScrollRibbon progress={scrollYProgress} range={[s, e]}
                    d={ribbon(XS[i] + NODE_W, bandH(n, total), XS[i + 1], bandH(kept, total))} />
                  <ElimWedge progress={scrollYProgress} range={[e - 0.03, e + 0.03]}
                    d={ribbon(XS[i] + NODE_W, elimH, XS[i + 1], Math.max(elimH * 0.25, 3),
                              YC + bandH(n, total) / 2 - elimH / 2 + 1, ELIM_Y)}
                    label={i === 0
                      ? `${(n - kept).toLocaleString()} never propped`
                      : `−${(n - kept).toLocaleString()} eliminated`}
                    x={(XS[i] + XS[i + 1]) / 2} />
                </g>
              );
            })}

            {/* finale veil dims the cut-chain, then the gate ribbon sweeps over it */}
            <motion.rect x="-40" y="-20" width={FLOW_W + 120} height={VIEW_H + 40}
              fill="var(--bg)" style={{ opacity: veilOpacity }} pointerEvents="none" />
            <motion.path d={gateD} fill="var(--blue)" opacity="0.3"
              style={{ clipPath: gateClip }} />

            {/* node bars spring in as the camera reaches them; the survivor
                node is the red one, exactly like the exhibit. REVEAL maps
                nodes onto narration beats (cuts 4-5 share one beat). */}
            {XS.map((x, i) => {
              // Load shows ONLY the season node; propped arrives on the
              // first scroll beat with cut 1 (cuts 4-5 still share a beat).
              const REVEAL = [0, 1, 1, 2, 3, 4, 4, 5];
              if (step < REVEAL[i]) return null;
              const h = bandH(counts[i], total);
              const last = i === SEGS;
              return (
                <g key={labels[i]}>
                  <motion.rect x={x} y={YC - h / 2} width={NODE_W} height={h}
                    fill={last ? "var(--neg)" : "var(--rule)"}
                    style={{ transformBox: "fill-box", transformOrigin: "center" }}
                    initial={{ scaleY: 0 }} animate={{ scaleY: 1 }}
                    transition={{ type: "spring", stiffness: 240, damping: 22 }} />
                  <motion.g initial={{ opacity: 0, x: 14 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ type: "spring", stiffness: 260, damping: 24, delay: 0.12 }}>
                    <text x={x + NODE_W / 2} y={YC - h / 2 - 20}
                          className={`fstory-count${last ? " hot" : ""}`}>
                      {counts[i].toLocaleString()}
                    </text>
                    <text x={x + NODE_W / 2} y={YC - h / 2 - 7} className="fstory-label">
                      {labels[i]}
                    </text>
                  </motion.g>
                </g>
              );
            })}
          </motion.g>
        </svg>
        )}

        {/* pinned caption; content slides in the direction of travel */}
        <div className="scrolly-caption">
          <AnimatePresence mode="wait">
            <motion.div key={step}
              initial={{ opacity: 0, x: 48 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -32 }}
              transition={{ duration: 0.3, ease: "easeOut" }}>
              <div className="fstory-step">
                {String(step + 1).padStart(2, "0")} / {String(LAST + 1).padStart(2, "0")}
              </div>
              <p className="fstory-title">{NARRATION[step].title}</p>
              <p className="fstory-body">{NARRATION[step].body}</p>
              {step === LAST && (
                <button className="toggle enter" onClick={onEnter}>
                  enter the ledger →
                </button>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="scrolly-hint">scroll — the camera follows the flow →</div>
      </div>

      {/* invisible track: vertical scroll length that powers the journey */}
      <div className="scrolly-spacer" />
    </div>
  );
}
