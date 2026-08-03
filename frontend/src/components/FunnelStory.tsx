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
import { fetchFunnel, fetchSummary } from "../api";

// The methodology as a horizontally traveling immersive. The chart is laid
// out wider than the viewport; scrolling moves the CAMERA rightward along the
// flow, so the reader physically follows the games through each cut. The
// caption card stays pinned bottom-left and swaps in the direction of travel.
// At the finale the camera pulls back to show the whole flow, the cut-chain
// dims, and the gate ribbon sweeps straight to the ledger.

interface Props {
  onEnter: () => void;
}

const NARRATION: Array<{ title: string; body: string }> = [
  {
    title: "It starts with every propped player-game.",
    body: "7,143 games in 2023–24 where a sportsbook posted a points line and the league recorded what happened on the floor. Nothing has been judged yet — this is the whole population.",
  },
  {
    title: "Cut 1 · Production — was this game bad for him?",
    body: "Points, shot attempts, rebounds and assists, each measured against the player's own season, never the league's. Against the league a star's disaster looks ordinary — Lillard scoring 6 was −0.5σ league-wide but −2.2σ against himself. The worst quarter of each player's games survives.",
  },
  {
    title: "Cut 2 · Effort — was he also less involved?",
    body: "Missing shots is common and mostly innocent. Running less, touching the ball less, using fewer possessions while still getting minutes is harder to explain away. Usage, turnovers, distance covered, touches and minutes form the corroboration.",
  },
  {
    title: "Cut 3 · Motive — who had something to gain?",
    body: "Salary is the one variable about what a player stood to lose. Everyone above $20M drops. Players with no listed salary are kept deliberately — that marks a two-way contract, the lowest-paid, most exposed group. Jontay Porter, the one confirmed case in NBA history, is exactly this row.",
  },
  {
    title: "Cuts 4–5 · Ordinary explanations, and the under must hit.",
    body: "Blowouts and foul-outs explain a bad night by themselves, so those games leave. And if the player still cleared his line, nobody betting the under was paid — no payoff, no case. 840 candidates remain.",
  },
  {
    title: "But a cut deletes permanently. So the ledger scores instead.",
    body: "Both known-bad games — Porter on 3/20, Beasley on 4/14 — died to the same defensible-looking blowout filter. A score cannot make that mistake: a weak axis lowers a row, it never erases it. One minimal gate, six weighted axes: 2,733 scored player-games, ranked, ready for review.",
  },
];

// geometry: the flow is ~1.7 viewports wide; the viewBox is the camera window
const VIEW_W = 880;
const VIEW_H = 430;
const XS = [40, 315, 590, 865, 1140, 1415];
const FLOW_W = 1455;
const YC = 180;
const HMAX = 260;
const NODE_W = 7;
const ELIM_Y = 400;
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
  const camX = useTransform(scrollYProgress,
    [0.04, PAN_END, FINALE + 0.1], [0, -(FLOW_W - VIEW_W + 15), 0]);
  const camScale = useTransform(scrollYProgress,
    [FINALE, FINALE + 0.1], [1, (VIEW_W - 20) / FLOW_W]);
  const veilOpacity = useTransform(scrollYProgress, [FINALE + 0.06, FINALE + 0.12], [0, 0.72]);
  const gateClip = useTransform(scrollYProgress, [FINALE + 0.08, 0.98],
    ["inset(0 100% 0 0)", "inset(0 0% 0 0)"]);
  const ledgerIn = useTransform(scrollYProgress, [0.94, 0.99], [0, 1]);

  const funnel = useQuery({ queryKey: ["funnel"], queryFn: fetchFunnel });
  const summary = useQuery({ queryKey: ["summary"], queryFn: fetchSummary });

  // NEVER early-return before .scrolly mounts: useScroll binds to the
  // container on first render, and if the element doesn't exist yet (queries
  // still loading), scroll tracking silently never attaches -- which is why
  // the piece used to scroll only on the second visit. The container always
  // renders; only the chart waits for data.
  const ready = funnel.data != null && summary.data != null;
  const counts = ready ? funnel.data!.stages.slice(0, 5).map((s) => s.n) : [];
  const scored = ready ? summary.data!.scored : 0;
  const total = ready ? counts[0] : 1;
  const labels = ["propped", "production", "effort", "motive", "candidates"];
  const gateD = ready
    ? ribbon(XS[0] + NODE_W, bandH(total, total), XS[5], bandH(scored, total))
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
              const s = (i + 0.1) * (PAN_END / 5);
              const e = (i + 0.9) * (PAN_END / 5);
              return (
                <g key={i}>
                  <ScrollRibbon progress={scrollYProgress} range={[s, e]}
                    d={ribbon(XS[i] + NODE_W, bandH(n, total), XS[i + 1], bandH(kept, total))} />
                  <ElimWedge progress={scrollYProgress} range={[e - 0.03, e + 0.03]}
                    d={ribbon(XS[i] + NODE_W, elimH, XS[i + 1], Math.max(elimH * 0.25, 3),
                              YC + bandH(n, total) / 2 - elimH / 2 + 1, ELIM_Y)}
                    label={`−${(n - kept).toLocaleString()} eliminated`}
                    x={(XS[i] + XS[i + 1]) / 2} />
                </g>
              );
            })}

            {/* finale veil dims the cut-chain, then the gate ribbon sweeps over it */}
            <motion.rect x="-40" y="-20" width={FLOW_W + 120} height={VIEW_H + 40}
              fill="var(--bg)" style={{ opacity: veilOpacity }} pointerEvents="none" />
            <motion.path d={gateD} fill="var(--blue)" opacity="0.3"
              style={{ clipPath: gateClip }} />

            {/* node bars spring in as the camera reaches them */}
            {XS.slice(0, 5).map((x, i) => {
              if (step < i) return null;
              const h = bandH(counts[i], total);
              return (
                <g key={labels[i]}>
                  <motion.rect x={x} y={YC - h / 2} width={NODE_W} height={h}
                    fill="var(--rule)"
                    style={{ transformBox: "fill-box", transformOrigin: "center" }}
                    initial={{ scaleY: 0 }} animate={{ scaleY: 1 }}
                    transition={{ type: "spring", stiffness: 240, damping: 22 }} />
                  <motion.g initial={{ opacity: 0, x: 14 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ type: "spring", stiffness: 260, damping: 24, delay: 0.12 }}>
                    <text x={x + NODE_W / 2} y={YC - h / 2 - 20} className="fstory-count">
                      {counts[i].toLocaleString()}
                    </text>
                    <text x={x + NODE_W / 2} y={YC - h / 2 - 7} className="fstory-label">
                      {labels[i]}
                    </text>
                  </motion.g>
                </g>
              );
            })}

            {/* the ledger node, fading in at the end of the pull-back */}
            <motion.g style={{ opacity: ledgerIn }}>
              <rect x={XS[5]} y={YC - bandH(scored, total) / 2} width={NODE_W}
                    height={bandH(scored, total)} fill="var(--neg)" />
              <text x={XS[5] + NODE_W / 2} y={YC - bandH(scored, total) / 2 - 20}
                    className="fstory-count">{scored.toLocaleString()}</text>
              <text x={XS[5] + NODE_W / 2} y={YC - bandH(scored, total) / 2 - 7}
                    className="fstory-label">ledger</text>
            </motion.g>
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
