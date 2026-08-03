import NumberFlow from "@number-flow/react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { ReactNode, useLayoutEffect, useRef, useState } from "react";
import { fetchCase, SeasonGame } from "../api";
import { RED_SCORE } from "../severity";

// The case file for a reviewer who needs everything fast: every evidence
// section is the SAME two-up ledger table -- label · value · viz -- so the
// eye learns one format and reads all of them. Order:
//   casehead -> evidence summary -> takeaway -> season strip ->
//   grouped scores -> box score -> hustle -> game context -> sportsbook.
// Exhibits number themselves so conditional sections never leave gaps.

interface Props {
  playerId: number;
  gameId: string;
}

// ---- shared severity logic ----------------------------------------------
// Residual z = "unlike HIS OWN season, context regressed out". More negative
// is worse for every stat except turnover ratio, where high is the bad side.
function badness(z: number | null, flip = false): number {
  if (z == null) return 0;
  return flip ? z : -z; // positive result = bad
}

// Diverging z-bar: anchored on a center hairline, bad extends left, quiet
// blue-gray until real severity, red past it. Exact σ in the tooltip.
function ZBar({ z, flip = false }: { z: number | null; flip?: boolean }) {
  if (z == null) return <span className="barwrap zb" />;
  const b = badness(z, flip);
  const w = Math.min(Math.abs(b) * 14, 44);
  const color = b > 1.5 ? "var(--neg)" : b > 0.75 ? "var(--neg-mid)" : "var(--pos)";
  return (
    <span className="barwrap zb" title={`${z.toFixed(2)}σ vs his own season`}>
      <motion.span className="bar"
        style={{ ...(b > 0 ? { right: "50%" } : { left: "50%" }), background: color }}
        initial={{ width: 0 }} animate={{ width: w }}
        transition={{ duration: 0.45, ease: "easeOut" }} />
    </span>
  );
}

// Diverging movement bar for sportsbook numbers: leftward = fell toward the
// under side. Same hairline anatomy as ZBar so the two read as one system.
function MoveBar({ v }: { v: number | null }) {
  if (v == null) return <span className="barwrap zb" />;
  const w = Math.min(Math.abs(v) * 350, 44);
  return (
    <span className="barwrap zb" title={`${(v * 100).toFixed(1)}% move`}>
      <motion.span className="bar"
        style={{
          ...(v < 0 ? { right: "50%" } : { left: "50%" }),
          background: v < 0 ? "var(--neg-mid)" : "var(--pos)",
        }}
        initial={{ width: 0 }} animate={{ width: w }}
        transition={{ duration: 0.45, ease: "easeOut" }} />
    </span>
  );
}

// The composite score as a circular gauge beside the evidence summary: a
// quiet track, a sweep that draws to the score, red once it crosses the
// review threshold. Remounts with the case, so the sweep redraws each open.
function ScoreRing({ value }: { value: number }) {
  const r = 30;
  const C = 2 * Math.PI * r;
  const hot = value >= RED_SCORE;
  return (
    <div className="scorering" title={`composite score ${value.toFixed(3)}`}>
      <svg width="76" height="76" viewBox="0 0 76 76" role="img"
           aria-label={`composite score ${value.toFixed(3)}`}>
        <circle cx="38" cy="38" r={r} fill="none"
                stroke="var(--track)" strokeWidth="5" />
        <motion.circle cx="38" cy="38" r={r} fill="none"
          stroke={hot ? "var(--neg)" : "var(--blue)"} strokeWidth="5"
          strokeLinecap="round" transform="rotate(-90 38 38)"
          strokeDasharray={C}
          initial={{ strokeDashoffset: C }}
          animate={{ strokeDashoffset: C * (1 - value) }}
          transition={{ duration: 0.8, ease: "easeOut" }} />
      </svg>
      <div className="scorering-num">
        <b className={hot ? "hot" : ""}>
          <NumberFlow value={value}
            format={{ minimumFractionDigits: 3, maximumFractionDigits: 3 }} />
        </b>
        <span>score</span>
      </div>
    </div>
  );
}

// A section masthead: title, quiet subtitle, and -- for the three score
// groups -- the group's gauge and value inline on the same ruled line. The
// score lives IN the header, so the section reads as one composed unit.
function SectionHead({ title, sub, kind, value }: {
  title: string;
  sub?: string;
  kind?: string;
  value?: number | null;
}) {
  return (
    <div className="gsec-head">
      <span className="gsec-title">{title}</span>
      {sub && <span className="gsec-sub">{sub}</span>}
      {kind && (
        <>
          <span className={`gsec-bar ${kind}`}>
            <motion.i
              initial={{ width: 0 }}
              animate={{ width: `${(value ?? 0) * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </span>
          <span className="gsec-val">{value?.toFixed(3) ?? "—"}</span>
        </>
      )}
    </div>
  );
}

// ---- the one table every section uses -----------------------------------
type Cell = [string, string, ReactNode];

function TwoUp({ cells }: { cells: Cell[] }) {
  const rows: Array<Array<Cell | undefined>> = [];
  for (let i = 0; i < cells.length; i += 2) rows.push([cells[i], cells[i + 1]]);
  return (
    <table className="ledger boxtbl">
      <tbody>
        {rows.map((pair, ri) => (
          <tr key={ri}>
            {pair.flatMap((s, j) =>
              s
                ? [
                    <td key={`${s[0]}-k`} className="stat">{s[0]}</td>,
                    <td key={`${s[0]}-v`} className="num">{s[1]}</td>,
                    <td key={`${s[0]}-z`} className="viz">{s[2]}</td>,
                  ]
                : [<td key={`pad-${j}`} colSpan={3} />],
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SeasonStrip({ log, flaggedDate }: { log: SeasonGame[]; flaggedDate: string }) {
  // Measure the container and draw in true pixels: no preserveAspectRatio
  // stretch, so strokes stay uniform and the flagged dot is a circle, not an
  // ellipse, at every laptop width.
  const wrapRef = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const games = log.filter((g) => g.close_line != null && g.points != null);
  if (games.length < 5) return null;
  const W = Math.max(w, 240), H = 64, zero = 33;
  const step = (W - 14) / (games.length - 1);
  const pts = games.map((g, i) => ({
    x: 7 + i * step,
    y: zero - Math.max(-15, Math.min(15, g.points! - g.close_line!)) * 1.6,
  }));
  // Catmull-Rom -> cubic bezier so the line flows through the points instead
  // of kinking at every game. The data is unchanged; only the path between
  // points is eased.
  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  const area = `${d} L ${pts[pts.length - 1].x.toFixed(1)} ${zero} L ${pts[0].x.toFixed(1)} ${zero} Z`;
  const fi = games.findIndex((g) => g.game_date === flaggedDate);
  const fx = fi >= 0 ? pts[fi].x : null;

  // month markers: one label where each new month first appears. Rendered as
  // HTML below the svg (text inside a preserveAspectRatio="none" svg would
  // stretch), positioned by the same x as their game, as a percentage.
  // A label needs ~8% of the width to breathe; when a month's first game
  // lands too close to the previous label (thin months), skip it rather than
  // let the two collide.
  const months: Array<{ pct: number; label: string }> = [];
  let lastShown = "";
  games.forEach((g, i) => {
    const m = new Date(`${g.game_date}T00:00:00`).toLocaleString("en-US", { month: "short" });
    const pct = (pts[i].x / W) * 100;
    if (m !== lastShown &&
        (months.length === 0 || pct - months[months.length - 1].pct >= 8)) {
      months.push({ pct, label: m });
      lastShown = m;
    }
  });
  return (
    <div className="strip" ref={wrapRef}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Points minus closing line, each propped game this season">
        <path d={area} fill="var(--blue)" opacity="0.08" />
        <line x1="0" y1={zero} x2={W} y2={zero} stroke="var(--line)" />
        <motion.path d={d} fill="none" stroke="var(--blue)" strokeWidth="1.5"
          strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
          transition={{ duration: 0.9, ease: "easeOut" }} />
        {fx != null && fi >= 0 && (
          <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    transition={{ duration: 0.3, delay: 0.8 }}>
            <line x1={fx} y1="0" x2={fx} y2={H} stroke="var(--neg)"
                  strokeWidth="1" strokeDasharray="3 3" />
            <circle cx={fx} cy={pts[fi].y} r="3.5" fill="var(--neg)" />
          </motion.g>
        )}
      </svg>
      <div className="strip-dates">
        {months.map((m) => (
          <span key={m.label} style={{ left: `${m.pct.toFixed(1)}%` }}>{m.label}</span>
        ))}
      </div>
    </div>
  );
}

function money(v: number | null): string {
  return v == null ? "unlisted" : `$${(v / 1e6).toFixed(2)}M`;
}

function pct(v: number | null): string {
  return v == null ? "unknown" : `${(v * 100).toFixed(1)}%`;
}

function clock(sec: number | null): string {
  if (sec == null) return "—";
  const m = Math.floor(sec / 60);
  return `${m}:${String(Math.round(sec % 60)).padStart(2, "0")}`;
}

export function CaseView({ playerId, gameId }: Props) {
  const { data: c, error } = useQuery({
    queryKey: ["case", playerId, gameId],
    queryFn: () => fetchCase(playerId, gameId),
    placeholderData: keepPreviousData,
  });

  if (error != null) return <div className="status">Failed to load case: {String(error)}</div>;
  if (!c) return <div className="status">Loading case…</div>;

  const d = c.deep;

  // Every section as cells for the shared table. Box score carries z-bars;
  // hustle has no baseline so its viz is empty; sportsbook carries move bars.
  const boxCells: Cell[] = ([
    ["Min", c.minutes.toFixed(1), c.minutes_resid_z, false],
    ["Pts", String(c.points), c.points_resid_z, false],
    ["FGA", String(c.fga), c.fga_resid_z, false],
    ["Reb", String(c.rebounds), c.rebounds_resid_z, false],
    ["Ast", String(c.assists), c.assists_resid_z, false],
    ["Touches", c.touches == null ? "—" : String(c.touches), c.touches_resid_z, false],
    ["Usage", c.usage_pct == null ? "—" : `${(c.usage_pct * 100).toFixed(1)}%`, c.usage_pct_resid_z, false],
    ["Dist", c.distance == null ? "—" : `${c.distance.toFixed(1)} mi`, c.distance_resid_z, false],
    ["TO ratio", c.turnover_ratio == null ? "—" : c.turnover_ratio.toFixed(1), c.turnover_ratio_resid_z, true],
  ] as Array<[string, string, number | null, boolean]>).map(
    ([k, v, z, flip]) => [k, v, <ZBar z={z} flip={flip} />] as Cell,
  );

  const hustleCells: Cell[] = d
    ? [
        ["Contested", String(d.contested_shots ?? "—"), null],
        ["Deflections", String(d.deflections ?? "—"), null],
        ["Loose balls", String(d.loose_balls ?? "—"), null],
        ["Box-outs", String(d.box_outs ?? "—"), null],
        ["Passes", String(d.passes ?? "—"), null],
        ["Stl + blk", String((d.steals ?? 0) + (d.blocks ?? 0)), null],
      ]
    : [];

  const contextCells: Cell[] = [
    ["Final margin", String(c.game_margin ?? "—"), null],
    ["Plus/minus", String(c.plus_minus ?? "—"), null],
    ["Fouls", String(c.fouls ?? "—"), null],
    ["Baseline", `${c.n_player_games ?? "—"} games`, null],
    ...(d && d.n_stints != null
      ? ([
          ["Stints", String(d.n_stints), null],
          ["Last off court", clock(d.last_out_sec), null],
          ["Competitive pts", String(d.points_competitive ?? 0), null],
          ["Garbage pts", String(d.points_garbage ?? 0), null],
          ...(d.ejected ? ([["Ejected", "YES", null]] as Cell[]) : []),
        ] as Cell[])
      : []),
  ];

  const bookCells: Cell[] = [
    ["Points line", `${c.line} (${c.line_source === "close" ? "closing" : "open — pulled"})`, null],
    ["Result", `${c.points} pts · under hit`, null],
    ["Line o→c", `${c.open_line ?? "—"} → ${c.close_line ?? "pulled"}`, null],
    ["Under price o→c", `${c.open_under ?? "—"} → ${c.close_under ?? "—"}`, null],
    ["Line move", pct(c.line_move_pct), <MoveBar v={c.line_move_pct} />],
    ["Under-price move", pct(c.under_move_pct), <MoveBar v={c.under_move_pct} />],
  ];

  // sequential exhibit numbering that skips nothing and gaps nothing
  let exhibitNo = 0;
  const ex = (title: string) => {
    exhibitNo += 1;
    return `Exhibit ${exhibitNo} · ${title}`;
  };

  return (
    <div className="case">
      <div className="casehead">
        <h2>{c.player}</h2>
        <span className="cohort">
          {c.position} · {c.tier} · {c.matchup} · {c.game_date} ·{" "}
          {c.started ? "started" : "bench"} · rank #{c.rank}
        </span>
        {c.shortfall != null && (
          <div className="chips">
            <span className="chip"
                  title={`finished ${Math.round(c.shortfall * 100)}% below the line`}>
              under hit
            </span>
          </div>
        )}
      </div>

      <motion.div
        key={`${c.player_id}-${c.game_id}`}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
      <div className="summary tight">
        {/* Official headshot straight off the NBA CDN, keyed by player_id.
            Some two-way / 10-day players have no photo there -- the 404
            handler just removes the frame so nothing broken shows. */}
        <img
          className="headshot"
          src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${c.player_id}.png`}
          alt={c.player}
          onError={(e) => { e.currentTarget.style.display = "none"; }}
        />
        <ScoreRing value={c.score} />
        <div className="sumtext">
          <h3>Evidence summary</h3>
          {c.ai_summary ? (
            <p>{c.ai_summary}</p>
          ) : (
            <p className="reserved">
              Reserved for the AI pass — it will narrate this case from the
              fields below: box-score residuals, line movement, motive context.
            </p>
          )}
        </div>
      </div>

      <p className="takeaway">
        {c.points} points against a {c.line}-point line, with production{" "}
        {Math.abs(c.prod_z ?? 0).toFixed(1)}σ and involvement{" "}
        {Math.abs(c.effort_z ?? 0).toFixed(1)}σ below his own season.
      </p>

      {c.season_log_source === "postgres" && (
        <>
          <SectionHead title={ex("Season")} sub="points vs closing line · flagged game marked" />
          <SeasonStrip log={c.season_log} flaggedDate={c.game_date} />
          <div className="source">
            Source: Postgres player_games × prop closing lines · flagged game marked
          </div>
        </>
      )}

      {/* The score's own structure organizes the page: each group's bar heads
          the section that holds its evidence, so proximity = relevance. */}

      <SectionHead title={ex("Performance")}
                   sub="production + involvement vs his own season"
                   kind="prf" value={c.g_performance} />
      <TwoUp cells={boxCells} />
      {hustleCells.length > 0 && <TwoUp cells={hustleCells} />}
      <TwoUp cells={contextCells} />
      <div className="source">
        Source: residuals per role tier (margin, rest, b2b, pace regressed out),
        hover a bar for the exact σ · hustle counts carry no bars because no
        per-player baseline exists · exit anatomy from play-by-play stints ·
        low minutes is shown, never filtered on
      </div>

      <SectionHead title={ex("Market")} sub="what the sportsbook saw"
                   kind="mkt" value={c.g_market} />
      <TwoUp cells={bookCells} />
      {c.line_pulled ? (
        <p className="source alert">
          The book withdrew this line before tip and he played anyway — 47
          season-wide, the most specific flag this pipeline has.
        </p>
      ) : (
        <p className="source">
          Decimal odds — a leftward bar means the number fell toward the under
          side. Movement unknown for ~48% of rows (no opening quote 12h out).
        </p>
      )}

      <SectionHead title={ex("Motive")} sub="what he stood to lose"
                   kind="mtv" value={c.g_motive} />
      <div className="motive-big">
        <b>{money(c.salary)}</b>
        <span>
          {c.has_listed_salary
            ? `P${Math.round((c.salary_pct ?? 0) * 100)} of league salaries · inside the $20M motive gate`
            : "no listed salary — a two-way / 10-day contract: the lowest-paid, most exposed profile, so it takes maximum motive weight"}
        </span>
      </div>
      <div className="source">
        Source: basketball-reference season salaries · what a player risks by
        throwing a game is roughly what he is paid ·
        score = (4·performance + 5·market + 2·motive) / 11
      </div>
      </motion.div>
    </div>
  );
}
