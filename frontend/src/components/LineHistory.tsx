import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { LineQuote, fetchLineHistory } from "../api";

// The pre-tip market for one player-game: where the book set the points line, and
// what it asked for the UNDER, from the L3b 'poll' ladder.
//
// NO Y-AXES. Two stacked measures with their own scales would need two axes plus a
// time axis -- three rulers in 170px, which is what made the first version hard to
// read. Instead each series carries its own first and last value INLINE at the ends
// of the mark, which is the only place a reader looks anyway. One axis remains: time.
//
// TWO PANELS, NOT TWO SCALES ON ONE PANEL. Decimal odds and a points line have
// unrelated units, so a shared plot with two y-scales lets the choice of ranges
// decide whether the series look like they move together. On a page arguing that a
// market leaned, that is the one thing the chart must not be free to imply.
//
// DECIMAL ODDS FALL AS THE UNDER GETS MORE LIKELY, backwards from every other chart
// here, so the caption says it in words.

interface Props {
  playerId: number;
  gameId: string;
}

const LABEL: Record<string, string> = {
  fanduel: "FanDuel",
  draftkings: "DraftKings",
  williamhill_us: "Caesars",
  polymarket: "Polymarket",
  kalshi: "Kalshi",
};

// WIDTH IS MEASURED, NOT SCALED. A fixed viewBox stretched with width:100% grows
// the chart in BOTH axes, so on a wide monitor it rendered ~280px tall and dwarfed
// the table above it. Rendering at the container's real pixel width instead keeps
// height fixed and 1:1, so strokes stay 2.5px and 8px type stays 8px at every
// window size -- the same trick the season strip above uses.
const MIN_W = 300;
const PAD_L = 34, PAD_R = 40;
const LINE_TOP = 30, LINE_H = 34;
const PRICE_TOP = 96, PRICE_H = 52;
const H = 176;
const AXIS_Y = PRICE_TOP + PRICE_H + 20;

// A flat series collapses to a zero-height band; pad it so the mark floats mid-band.
// The padded bounds are never printed -- only real quoted values are.
function extent(vals: number[], pad: number): [number, number] {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return hi - lo < 1e-9 ? [lo - pad, hi + pad] : [lo, hi];
}

// Ticks the window can carry. A 14-minute series has no hour boundary in it, so an
// hours-only rule would leave the axis blank except for "tip".
function timeTicks(t0: number, t1: number) {
  // >= 4 rather than >= 2: a three-hour window on two ticks made it hard to place
  // a move in time, which is the whole point of the chart.
  const step = [60, 30, 15, 10, 5, 2].find((s) => (t0 - t1) / s >= 4) ?? 1;
  const out: { m: number; label: string }[] = [];
  for (let m = Math.floor(t0 / step) * step; m > t1 + step / 2; m -= step)
    if (m <= t0) out.push({ m, label: step >= 60 ? `${Math.round(m / 60)}h` : `${Math.round(m)}m` });
  out.push({ m: t1, label: "tip" });
  return out;
}

// STRAIGHT SEGMENTS, NOT A SPLINE. An earlier version eased the price through its
// quotes with a Catmull-Rom curve, which looked smoother and was slightly dishonest:
// the bezier overshoots past a local high or low, drawing a price the book never
// asked. Joining the quotes with straight lines makes every vertex a real quote and
// every kink a real move -- the jaggedness IS the data.
function polyline(p: { x: number; y: number }[]) {
  return p.map((q, i) => `${i ? "L" : "M"} ${q.x.toFixed(1)} ${q.y.toFixed(1)}`).join(" ");
}

// Gridline values on a "nice" step, so the scale reads 1.80 / 1.85 rather than
// 1.7847. Decimal odds move in hundredths, hence the small candidate steps.
function niceTicks(lo: number, hi: number, target = 3) {
  const span = hi - lo;
  const step = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1].find(
    (s) => span / s <= target + 1,
  ) ?? 1;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    out.push(Math.round(v / step) * step);
  return out.length >= 2 ? out : [lo, hi];
}

export function LineHistory({ playerId, gameId }: Props) {
  const wrapRef = useRef<HTMLElement>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // clientWidth includes padding, and this figure is a padded panel -- measure
    // the CONTENT box or the svg draws wider than the room it has and clips.
    const inner = () => {
      const cs = getComputedStyle(el);
      return el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    };
    const ro = new ResizeObserver(() => setW(inner()));
    ro.observe(el);
    setW(inner());
    return () => ro.disconnect();
  }, []);
  const W = Math.max(w, MIN_W);

  const [book, setBook] = useState<string | undefined>(undefined);
  const { data } = useQuery({
    queryKey: ["lineHistory", playerId, gameId, book],
    queryFn: () => fetchLineHistory(playerId, gameId, book),
    placeholderData: keepPreviousData,   // chip switches redraw, never blank out
  });
  const [hover, setHover] = useState<number | null>(null);

  const g = useMemo(() => {
    const s = (data?.series ?? []).filter((q) => q.under_price != null && q.line != null);
    if (s.length < 2) return null;
    const mins = s.map((q) => q.minutes_before_tip);
    const t0 = Math.max(...mins), t1 = Math.min(...mins);
    const span = Math.max(t0 - t1, 1);
    const x = (m: number) => PAD_L + ((t0 - m) / span) * (W - PAD_L - PAD_R);

    const [pLo, pHi] = extent(s.map((q) => q.under_price!), 0.05);
    const [lLo, lHi] = extent(s.map((q) => q.line!), 0.5);
    const yP = (v: number) => PRICE_TOP + PRICE_H - ((v - pLo) / (pHi - pLo || 1)) * PRICE_H;
    const yL = (v: number) => LINE_TOP + LINE_H - ((v - lLo) / (lHi - lLo || 1)) * LINE_H;

    // The points line is a STEP: a book holds a number until it moves it. A slope
    // would invent half-points that were never offered.
    let step = "";
    s.forEach((q, i) => {
      const px = x(q.minutes_before_tip), py = yL(q.line!);
      step += i === 0 ? `M ${px} ${py}` : ` L ${px} ${yL(s[i - 1].line!)} L ${px} ${py}`;
    });
    // A square only where the line actually MOVED, plus the two ends. A marker at
    // every 5-minute quote is 13 identical squares saying nothing changed.
    const moves = s.map((_, i) => i).filter(
      (i) => i === 0 || i === s.length - 1 || s[i].line !== s[i - 1].line,
    );
    const pp = s.map((q) => ({ x: x(q.minutes_before_tip), y: yP(q.under_price!) }));

    // Where each key can sit without the mark running through it. Anchoring to the
    // first point alone is not enough: a series that climbs out of its start crosses
    // the label a third of the way along. Clear the highest point of the stretch the
    // label spans (the leftmost ~40%), which is the only part it can collide with.
    const reach = Math.max(2, Math.ceil(s.length * 0.4));
    const keyY = (ys: number[], floor: number) =>
      Math.max(Math.min(...ys.slice(0, reach)) - 8, floor);

    const priceGrid = niceTicks(pLo, pHi);
    const lineGrid = [...new Set(s.map((q) => q.line!))].sort((a, b) => a - b);

    return { s, x, yP, yL, step, moves, price: polyline(pp), pp, priceGrid, lineGrid,
             keyLine: keyY(s.map((q) => yL(q.line!)), 12),
             keyPrice: keyY(pp.map((p) => p.y), PRICE_TOP - 14),
             ticks: timeTicks(t0, t1), t0 };
  }, [data, W]);

  const chips = data?.books ?? [];
  const active = data?.book ?? "fanduel";

  const bar = (
    <div className="lh-books">
      {chips.map((b) => (
        <button key={b.key} type="button"
                className={`lh-chip${b.key === active ? " on" : ""}${b.live && b.n ? "" : " off"}`}
                disabled={!b.live || !b.n}
                title={b.live ? (b.n ? `${b.n} quotes` : "no quotes captured for this game")
                              : "prediction market — not yet wired up"}
                onClick={() => { setBook(b.key); setHover(null); }}>
          {LABEL[b.key] ?? b.key}
        </button>
      ))}
    </div>
  );

  // THE REF MUST NEVER UNMOUNT. Returning a different element while the query is
  // in flight left wrapRef.current null when the layout effect ran; with [] deps it
  // never ran again, so the ResizeObserver was never attached and the chart drew at
  // the MIN_W fallback forever, whatever the column width. The figure is now the
  // root of every branch.
  const hq: LineQuote | null = hover == null || !g ? null : g.s[hover];
  const drift = g ? (g.s[g.s.length - 1].under_price ?? 0) - (g.s[0].under_price ?? 0) : 0;

  return (
    <figure className="linehist" ref={wrapRef}>
      <div className="lh-title">Line &amp; price movement before tip</div>
      {data && bar}
      {!data ? (
        <div className="status">Loading market…</div>
      ) : !g ? (
        <p className="source">
          Only {data.n} quote{data.n === 1 ? "" : "s"} captured here — not enough to
          draw a curve.
        </p>
      ) : (
      <>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Points line and under price over the ${Math.round(g.t0)} minutes before tip-off`}
           onMouseLeave={() => setHover(null)}>
        {/* ---------- points line ---------- */}
        {/* A rule at every value the book ACTUALLY quoted, not on a synthetic step.
            Lines are half-points and a book sits exactly on one of them, so there
            is nothing between the rules to interpolate. */}
        {g.lineGrid.map((v) => (
          <g key={v}>
            <line className="lh-grid" x1={PAD_L} x2={W - PAD_R} y1={g.yL(v)} y2={g.yL(v)} />
            <text className="lh-tick" x={PAD_L - 6} y={g.yL(v) + 3} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {/* the key sits just above where the mark actually starts, so it names the
            thing it is next to rather than floating at a fixed band top */}
        <text className="lh-key lh-key-line" x={PAD_L} y={g.keyLine}>POINTS LINE</text>
        <motion.path className="lh-step" d={g.step}
                     initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                     transition={{ duration: 0.8, ease: "easeOut" }} />
        {g.moves.map((i) => (
          <motion.rect key={i} className="lh-sq"
                       x={g.x(g.s[i].minutes_before_tip) - 2} y={g.yL(g.s[i].line!) - 2}
                       width={4} height={4}
                       initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                       transition={{ duration: 0.25, delay: 0.45 }} />
        ))}
        <text className="lh-val on" x={W - PAD_R + 6} y={g.yL(g.s[g.s.length - 1].line!) + 3}>
          {g.s[g.s.length - 1].line}
        </text>

        {/* ---------- under price ---------- */}
        {g.priceGrid.map((v) => (
          <g key={v}>
            <line className="lh-grid" x1={PAD_L} x2={W - PAD_R} y1={g.yP(v)} y2={g.yP(v)} />
            <text className="lh-tick" x={PAD_L - 6} y={g.yP(v) + 3} textAnchor="end">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        <text className="lh-key" x={PAD_L} y={g.keyPrice}>UNDER PRICE</text>
        <motion.path className="lh-line" d={g.price}
                     initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                     transition={{ duration: 0.8, ease: "easeOut" }} />
        <text className="lh-val on" x={W - PAD_R + 6} y={g.pp[g.pp.length - 1].y + 3}>
          {g.s[g.s.length - 1].under_price?.toFixed(2)}
        </text>

        {/* ---------- time ---------- */}
        <line className="lh-axis" x1={PAD_L} y1={AXIS_Y - 13} x2={W - PAD_R} y2={AXIS_Y - 13} />
        {g.ticks.map((t, i) => (
          <text key={i} className="lh-tick" x={g.x(t.m)} y={AXIS_Y} textAnchor="middle">
            {t.label}
          </text>
        ))}

        {/* ---------- hover ---------- */}
        {hq && (
          <>
            <motion.line className="lh-cross" initial={false}
                         animate={{ x1: g.x(hq.minutes_before_tip), x2: g.x(hq.minutes_before_tip) }}
                         y1={LINE_TOP - 4} y2={AXIS_Y - 13}
                         transition={{ type: "spring", stiffness: 520, damping: 38 }} />
            <circle className="lh-dot on" cx={g.pp[hover!].x} cy={g.pp[hover!].y} r={3.6} />
          </>
        )}
        {g.s.map((q, i) => (
          <rect key={i} x={g.x(q.minutes_before_tip) - 7} y={LINE_TOP - 8}
                width={14} height={AXIS_Y - LINE_TOP} fill="transparent"
                onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      </>
      )}

      {g && (
      <figcaption className="source">
        {hq ? (
          <>
            <b>{hq.minutes_before_tip.toFixed(0)} min before tip</b> · line{" "}
            <b>{hq.line}</b> · under <b>{hq.under_price?.toFixed(2)}</b>
            {hq.p_under != null && <> · implied <b>{hq.p_under.toFixed(1)}%</b></>}
            {hq.role !== "poll" && <> · {hq.role}</>}
          </>
        ) : (
          <>
            {g.s.length} quotes across the {Math.round(g.t0)} minutes before tip
            {" — "}
            {Math.abs(drift) < 0.005 ? "the under price held"
              : <>the under price <b>{drift < 0 ? "shortened" : "drifted"}</b>
                 {drift < 0 ? ", meaning more money on the under" : ", meaning less"}</>}
          </>
        )}
      </figcaption>
      )}
    </figure>
  );
}
