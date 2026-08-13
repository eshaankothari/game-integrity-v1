import NumberFlow from "@number-flow/react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { ReactNode, useLayoutEffect, useRef, useState } from "react";
import { CaseDetail, fetchCase, fetchPlayerCareer, SeasonGame } from "../api";
import { RED_SCORE, fmtScore, isConfirmed } from "../severity";
import { LineHistory } from "./LineHistory";
import { PlayTimeline } from "./PlayTimeline";
import { ShotChart } from "./ShotChart";

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

// Diverging z-bar. DIRECTION IS THE RAW BACKEND SIGN, always: negative z
// extends left, positive extends right, so the bar can never disagree with
// the printed σ beside it. `flip` affects only the severity COLOR (for
// metrics where high, not low, is the suspicious side -- TO ratio, the
// market components). `vs` names the baseline for the tooltip.
function ZBar({ z, flip = false, vs = "vs his own season" }: {
  z: number | null;
  flip?: boolean;
  vs?: string;
}) {
  if (z == null) return <span className="barwrap zb" />;
  const b = badness(z, flip);
  const w = Math.min(Math.abs(z) * 14, 44);
  const color = b > 1.5 ? "var(--neg)" : b > 0.75 ? "var(--neg-mid)" : "var(--pos)";
  return (
    <span className="barwrap zb" title={`${z.toFixed(2)}σ ${vs}`}>
      <motion.span className="bar"
        style={{ ...(z < 0 ? { right: "50%" } : { left: "50%" }), background: color }}
        initial={{ width: 0 }} animate={{ width: w }}
        transition={{ duration: 0.45, ease: "easeOut" }} />
    </span>
  );
}

// The composite score as a circular gauge beside the evidence summary: a
// quiet track, a sweep that draws to the score, red once it crosses the
// review threshold. Remounts with the case, so the sweep redraws each open.
// Signed, one decimal, explicit +. The sign is the whole point -- an unsigned "1.3σ"
// with the word "below" buried in a sentence is how a reader misreads a good game as a
// bad one.
// The headline sentence under the player image. Written, not templated into a list,
// because it is the first thing read and a reader parses a clause faster than a chip.
// Direction is spelled out ("below his own season") rather than left to a minus sign,
// and each claim carries the column it came from so the number is findable.
// The highlighter: a marker swipe behind each load-bearing figure, drawn
// left-to-right in reading order when the case opens. Neutral evidence gets
// the signal blue at a wash; the market lean keeps the page's severity
// grammar (red = under-side pressure, blue = over).
function Hl({ n, tone, children }: { n: number; tone?: "under"; children: ReactNode }) {
  return (
    <em className={`hl${tone ? ` hl-${tone}` : ""}`}
        style={{ animationDelay: `${0.15 + n * 0.14}s` }}>
      {children}
    </em>
  );
}

function takeaway(c: CaseDetail) {
  const gz = c.game_z, ez = c.effort_z, mk = c.market_100;

  // "an 8.5-point line", not "a 8.5-point line" -- 8, 11 and 18 take "an", and lines
  // land on all three. The article follows the SOUND, so it is keyed off the leading
  // digit rather than a vowel test on the string.
  const art = c.line != null && /^(8|11|18)/.test(String(c.line)) ? "an" : "a";

  // Highlight order is reading order: the swipes cascade through the sentence,
  // so n increments as clauses are (conditionally) added. 0 and 1 are the
  // points and the line, hard-coded in the JSX below.
  let n = 1;
  const clauses: ReactNode[] = [];
  if (gz != null) {
    clauses.push(
      <span key="gz">
        , playing <Hl n={++n}>{Math.abs(gz).toFixed(1)}&sigma; {gz < 0 ? "below" : "above"}</Hl>{" "}
        his own season
      </span>,
    );
  }
  if (ez != null) {
    clauses.push(
      <span key="ez">
        {gz != null ? ", and was " : ", was "}
        <Hl n={++n}>{Math.abs(ez).toFixed(1)}&sigma; {ez < 0 ? "less" : "more"} involved</Hl>
      </span>,
    );
  }

  // A separate sentence: the market is a claim about prices before tip, not about
  // anything he did, and folding it into the same clause read as though it were.
  const lean =
    mk == null
      ? null
      : mk >= 75 ? "heavily under"
      : mk >= 55 ? "under"
      : mk <= 25 ? "over"
      : "neither way";

  return (
    <>
      He scored <Hl n={0}><b>{c.points}</b> points</Hl> against {art}{" "}
      <Hl n={1}><b>{c.line}</b>-point line</Hl>
      {clauses}.
      {lean && (
        <span>
          {" "}The market had leaned{" "}
          <Hl n={n + 1} tone={lean.includes("under") ? "under" : undefined}>{lean}</Hl> before tip.
        </span>
      )}
    </>
  );
}


function ScoreRing({ value, forceHot = false }: { value: number; forceHot?: boolean }) {
  const r = 30;
  const C = 2 * Math.PI * r;
  const hot = forceHot || value >= RED_SCORE;
  return (
    <div className="scorering" title={`composite score ${fmtScore(value)} / 100`}>
      <svg width="76" height="76" viewBox="0 0 76 76" role="img"
           aria-label={`composite score ${fmtScore(value)} out of 100`}>
        <circle cx="38" cy="38" r={r} fill="none"
                stroke="var(--track)" strokeWidth="5" />
        <motion.circle cx="38" cy="38" r={r} fill="none"
          stroke={hot ? "var(--neg)" : "var(--blue)"} strokeWidth="5"
          strokeLinecap="round" transform="rotate(-90 38 38)"
          strokeDasharray={C}
          initial={{ strokeDashoffset: C }}
          animate={{ strokeDashoffset: C * (1 - value / 100) }}
          transition={{ duration: 0.8, ease: "easeOut" }} />
      </svg>
      <div className="scorering-num">
        <b className={hot ? "hot" : ""}>
          <NumberFlow value={value}
            format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
        </b>
        <span>/ 100</span>
      </div>
    </div>
  );
}

// A widget-style scorecard: team codes on the eyebrow (his in blue), the
// score itself as the big hero figure, and the outcome spelled out under it.
// Points roll via NumberFlow when the case changes (the header never
// remounts).
// Official team marks off the NBA CDN, same pattern as the headshots.
function teamLogo(teamId: number): string {
  return `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg`;
}

function ScoreCard({ fs, playerTeam }: {
  fs: { away: { team: string; team_id: number; pts: number };
        home: { team: string; team_id: number; pts: number };
        result: "W" | "L" | null };
  playerTeam: string;
}) {
  const margin = Math.abs(fs.away.pts - fs.home.pts);
  return (
    <div className="scorecard"
         title={fs.result === "W" ? "his team won" :
                fs.result === "L" ? "his team lost" : undefined}>
      <div className="sc-teams">
        <img className="sc-logo" src={teamLogo(fs.away.team_id)} alt=""
             onError={(e) => { e.currentTarget.style.display = "none"; }} />
        <span className={fs.away.team === playerTeam ? "his" : ""}>
          {fs.away.team}
        </span>
        <span className="at">@</span>
        <img className="sc-logo" src={teamLogo(fs.home.team_id)} alt=""
             onError={(e) => { e.currentTarget.style.display = "none"; }} />
        <span className={fs.home.team === playerTeam ? "his" : ""}>
          {fs.home.team}
        </span>
      </div>
      <div className="sc-big">
        <NumberFlow value={fs.away.pts} />
        <span className="dash">–</span>
        <NumberFlow value={fs.home.pts} />
      </div>
      {fs.result && (
        <div className="sc-sub">
          {fs.result === "W" ? "won" : "lost"} by {margin}
        </div>
      )}
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
              animate={{ width: `${value ?? 0}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </span>
          <span className="gsec-val">{fmtScore(value)}</span>
        </>
      )}
    </div>
  );
}

// ---- the performance equation table --------------------------------------
// Two balanced columns: what feeds GAME SCORE on the left, what feeds EFFORT
// on the right, each row label · value · z-bar · exact σ. The component
// scores close it out like the sum line of an equation.
type EqRow = [string, string, number | null, boolean];

function sigTxt(z: number | null): string {
  if (z == null) return "";
  return `${z >= 0 ? "+" : "−"}${Math.abs(z).toFixed(2)}σ`;
}

// EVERY evidence table renders through this ONE component -- performance,
// context and market alike -- so labels, values, the 0σ axis and the σ
// figures sit on identical columns down the whole page. A row without a z
// renders no axis at all: an empty track would imply a measurement that was
// never made. `vs` names the bars' baseline for the tooltip.
function EqTable({ left, right, leftSum, rightSum, showZero = false, vs }: {
  left: EqRow[];
  right: EqRow[];
  leftSum?: EqRow;
  rightSum?: EqRow;
  showZero?: boolean;
  vs?: string;
}) {
  const cells = (r: EqRow | undefined) =>
    r
      ? [
          <td key={`${r[0]}-k`} className="stat">{r[0]}</td>,
          <td key={`${r[0]}-v`} className="num">{r[1]}</td>,
          <td key={`${r[0]}-z`} className="viz">
            {r[2] != null && <ZBar z={r[2]} flip={r[3]} vs={vs} />}
          </td>,
          <td key={`${r[0]}-s`} className="sig">{sigTxt(r[2])}</td>,
        ]
      : [<td key="pad" colSpan={4} />];
  const n = Math.max(left.length, right.length);
  return (
    <table className="ledger boxtbl eqtbl">
      <tbody>
        {/* header cells carry the SAME classes as data cells: table-layout
            fixed reads column widths from the first row, so unclassed cells
            here would un-align this table from every other one */}
        {showZero && (
          <tr className="eq0">
            <td className="stat" /><td className="num" />
            <td className="viz"><span className="z0">0σ</span></td>
            <td className="sig" />
            <td className="stat" /><td className="num" />
            <td className="viz"><span className="z0">0σ</span></td>
            <td className="sig" />
          </tr>
        )}
        {Array.from({ length: n }, (_, i) => (
          <tr key={i}>
            {cells(left[i])}
            {cells(right[i])}
          </tr>
        ))}
        {leftSum && rightSum && (
          <tr className="eqsum">
            {cells(leftSum)}
            {cells(rightSum)}
          </tr>
        )}
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
        {/* the zero line: dotted so "hit the line exactly" is a visible datum */}
        <line x1="0" y1={zero} x2={W} y2={zero} stroke="var(--faint)"
              strokeDasharray="2 4" />
        <text x={W - 4} y={zero - 4} textAnchor="end" fontSize="8"
              fill="var(--faint)">0</text>
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

// ---- career team-movement timeline ---------------------------------------
// The motive exhibit's PICTURE: instability_career is a rate, this is what it
// counted -- one continuous baseline with a slim rounded pill per league
// season sitting ON it, first season to the scored one. The team code sits
// INSIDE its pill (labels beside a bare dot proved harder to read), on the
// quietest fill token rather than a heavy tile; a mid-season trade splits the
// season's pill into half-pills with a surface gap; a change of team from the
// previous stint carries a small cyan inset tick -- the motive color, because
// those boundaries are exactly what the rate counts. A gap season is an
// EMPTY dashed-outline pill ("no NBA team" in the tooltip), so absence can
// never read as data. Fetched per player like the shot chart fetches per
// game; a 404 -- older backend, or no recorded history -- rejects and renders
// NOTHING (invariant 1: missing history is absence, never a calm-looking
// single-team career).
function CareerStrip({ playerId }: { playerId: number }) {
  const { data } = useQuery({
    queryKey: ["career", playerId],
    queryFn: () => fetchPlayerCareer(playerId),
  });
  if (!data || data.stints.length === 0) return null;

  // Stints arrive in career order (seq); group per season, order preserved.
  const bySeason = new Map<string, string[]>();
  data.stints.forEach((s) => {
    const t = bySeason.get(s.season);
    if (t) t.push(s.team_abbr);
    else bySeason.set(s.season, [s.team_abbr]);
  });

  // Every LEAGUE season in the span, so a gap year occupies real width
  // instead of silently collapsing the timeline around it.
  const span: string[] = [];
  for (let y = +data.first_season.slice(0, 4); y <= +data.last_season.slice(0, 4); y++) {
    span.push(`${y}-${String((y + 1) % 100).padStart(2, "0")}`);
  }
  const moves = data.stints.filter(
    (s, i) => i > 0 && s.team_abbr !== data.stints[i - 1].team_abbr,
  ).length;
  const gaps = span.length - bySeason.size;

  // Merge the per-season stints into TENURE SEGMENTS: consecutive seasons on
  // the same team become one bar whose width is proportional to how long he
  // stayed (a trade season splits its weight between the two teams). The
  // shape of a career then reads at a glance -- one long bar is a franchise
  // player, a chain of short ones is the journeyman the instability rate is
  // scoring. Consecutive gap years merge into one dashed segment the same way.
  type Seg = { team: string | null; from: string; to: string; weight: number; moved: boolean };
  const segs: Seg[] = [];
  let prevTeam: string | null = null;
  for (const season of span) {
    const teams: (string | null)[] = bySeason.get(season) ?? [null];
    const w = 1 / teams.length;
    for (const t of teams) {
      const last = segs[segs.length - 1];
      if (last && last.team === t) {
        last.to = season;
        last.weight += w;
      } else {
        segs.push({
          team: t, from: season, to: season, weight: w,
          // the cyan tick = exactly what the rate counts: a change of team
          // from the previous stint (a gap between them does not reset it)
          moved: t != null && prevTeam != null && t !== prevTeam,
        });
      }
      if (t != null) prevTeam = t;
    }
  }

  return (
    <>
      <div className="career-scroll">
        <div className="career-strip" role="img"
             aria-label={`Career team history: ${span
               .map((s) => `${s} ${(bySeason.get(s) ?? ["no NBA team"]).join(" then ")}`)
               .join(", ")}`}>
          {segs.map((seg, i) => {
            const one = seg.from === seg.to;
            const label = one ? seg.from : `${seg.from} → ${seg.to.slice(5)}`;
            return (
              <div key={i}
                   className={`cs-seg${seg.team ? "" : " cs-gap"}${seg.moved ? " cs-move" : ""}`}
                   style={{ "--w": seg.weight } as React.CSSProperties}
                   title={`${label} · ${seg.team ?? "no NBA team"}`}>
                <div className="cs-tenure">
                  {seg.team && <span className="cs-code">{seg.team}</span>}
                </div>
                {/* a trade splits one season into two segments; year the first only */}
                <span className="cs-yr">
                  {i === 0 || segs[i - 1].from !== seg.from ? `’${seg.from.slice(2, 4)}` : ""}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="source">
        {span[0].slice(0, 4)}–{span[span.length - 1]}: the instability rate counts
        each team change (cyan tick) once and each dashed gap season twice
        {moves + gaps > 0
          ? ` — ${moves} change${moves === 1 ? "" : "s"}, ${gaps} gap season${gaps === 1 ? "" : "s"} here`
          : ""}.
      </div>
    </>
  );
}

function money(v: number | null): string {
  return v == null ? "unlisted" : `$${(v / 1e6).toFixed(2)}M`;
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

  // The performance equation: game-score inputs left, effort inputs right,
  // the two component scores closing each column like a sum line. (Grouping
  // follows the pipeline: Pts/Reb/Ast/FGA feed Hollinger game score;
  // touches/distance/usage feed the involvement composite. Minutes moved to
  // game context -- it is a condition of the night, not an input.)
  // Seven rows a side: the box-score inputs of GAME SCORE on the left
  // (steals/blocks/turnovers are Hollinger inputs; passes sits here for
  // balance), the involvement/hustle components of EFFORT on the right.
  // Hustle counts carry no bar because no per-player baseline exists.
  const gsRows: EqRow[] = [
    ["Pts", String(c.points), c.points_resid_z, false],
    ["Reb", String(c.rebounds), c.rebounds_resid_z, false],
    ["Ast", String(c.assists), c.assists_resid_z, false],
    ["FGA", String(c.fga), c.fga_resid_z, false],
    ["Stl + blk", d ? String((d.steals ?? 0) + (d.blocks ?? 0)) : "—", null, false],
    ["TO ratio", c.turnover_ratio == null ? "—" : c.turnover_ratio.toFixed(1), c.turnover_ratio_resid_z, true],
    ["Passes", d?.passes == null ? "—" : String(d.passes), null, false],
  ];
  const efRows: EqRow[] = [
    ["Touches", c.touches == null ? "—" : String(c.touches), c.touches_resid_z, false],
    ["Dist", c.distance == null ? "—" : `${c.distance.toFixed(1)} mi`, c.distance_resid_z, false],
    ["Usage", c.usage_pct == null ? "—" : `${(c.usage_pct * 100).toFixed(1)}%`, c.usage_pct_resid_z, false],
    ["Contested", d?.contested_shots == null ? "—" : String(d.contested_shots), null, false],
    ["Deflections", d?.deflections == null ? "—" : String(d.deflections), null, false],
    ["Loose balls", d?.loose_balls == null ? "—" : String(d.loose_balls), null, false],
    ["Box-outs", d?.box_outs == null ? "—" : String(d.box_outs), null, false],
  ];
  const gsSum: EqRow = ["Game score",
    c.game_score != null ? c.game_score.toFixed(1) : "—", c.game_z, false];
  const efSum: EqRow = ["Effort score",
    c.effort_z != null ? c.effort_z.toFixed(2) : "—", c.effort_z, false];

  // Context rides the SAME grid as the equation, so every value, bar track
  // and σ figure lines up down the whole section.
  const ctxLeft: EqRow[] = [
    ["Min", c.minutes.toFixed(1), c.minutes_resid_z, false],
    ["Fouls", String(c.fouls ?? "—"), null, false],
    ...(d && d.n_stints != null
      ? ([["Last off court", clock(d.last_out_sec), null, false]] as EqRow[])
      : []),
  ];
  const ctxRight: EqRow[] = [
    ["Plus/minus", String(c.plus_minus ?? "—"), null, false],
    ...(d && d.n_stints != null
      ? ([
          ["Stints", String(d.n_stints), null, false],
          ...(d.ejected ? ([["Ejected", "YES", null, false]] as EqRow[]) : []),
        ] as EqRow[])
      : []),
  ];

  // Every market row carries the pipeline's OWN standardized component
  // (standardize.py add_market: mk_* league z-scores, high = under-side
  // pressure) -- the same bars the composite is actually built from. Shown
  // NEGATED so the whole page reads one way: lower = more suspicious. The
  // pipeline stores these high = under-side pressure; a sign flip is a pure
  // relabel and keeps every bar's red side on the left like performance.
  const mneg = (v: number | null) => (v == null ? null : -v);
  const mktLeft: EqRow[] = [
    ["Points line", String(c.line ?? "—"), mneg(c.mk_p_line), false],
    ["Under close", c.close_under == null ? "—" : c.close_under.toFixed(2), mneg(c.mk_p_price), false],
    ["Result", c.under_hit == null ? "—" : c.under_hit ? "under hit" : "over hit", null, false],
  ];
  const mktRight: EqRow[] = [
    ["Line o→c", `${c.open_line ?? "—"} → ${c.close_line ?? "pulled"}`, mneg(c.mk_line_mv), false],
    ["Under o→c", `${c.open_under ?? "—"} → ${c.close_under ?? "—"}`, mneg(c.mk_price_mv), false],
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
        <h2>
          {c.player}
          <span className="hdate">
            {new Date(`${c.game_date}T00:00:00`).toLocaleDateString("en-US", {
              month: "long",
              day: "numeric",
              year: "numeric",
            })}
          </span>
        </h2>
        <span className="cohort">
          {[
            c.matchup,
            c.position,
            c.tier,
            c.age != null && `age ${c.age}`,
            c.experience != null &&
              (c.experience === 0 ? "rookie" : `${c.experience} yr exp`),
            c.height_in != null &&
              `${Math.floor(c.height_in / 12)}'${c.height_in % 12}"`,
            c.weight_lb != null && `${c.weight_lb} lb`,
          ]
            .filter(Boolean)
            .join(" · ")}
        </span>
        <div className="chips">
          {c.final_score != null && (
            <ScoreCard fs={c.final_score}
                       playerTeam={c.matchup.split(/[^A-Z]+/)[0] ?? ""} />
          )}
          {isConfirmed(c.player_id, c.game_id) && (
            <span className="chip conf"
                  title="confirmed suspicious by league / federal action — not a pipeline output">
              confirmed
            </span>
          )}
          {/* the verdict chip answers the actual outcome: red when the under
              hit, blue when he beat the line */}
          {c.under_hit != null && (
            c.under_hit ? (
              <span className="chip under"
                    title={c.shortfall != null
                      ? `finished ${Math.round(c.shortfall * 100)}% below the line`
                      : "finished below the line"}>
                under hit
              </span>
            ) : (
              <span className="chip over"
                    title={`finished above the ${c.line}-point line`}>
                over hit
              </span>
            )
          )}
        </div>
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
        <ScoreRing value={c.score_100}
                   forceHot={isConfirmed(c.player_id, c.game_id)} />
        <div className="sumtext">
          <h3>Case summary</h3>

          {/* No source pill and no cause chips. The summary already SAYS whether
              anything explains the night, in the sentence a reviewer is reading
              anyway -- a chip repeating it was a second, louder copy of the same
              claim. `summary_flags` and `summary_source` are still on the payload
              for the reviewer comparison; they are just not shown twice. */}
          {c.summary ? (
            // Paragraphs are separated by blank lines in the source, so they are split
            // rather than rendered as one block -- the summary is written to be read in
            // beats and collapses badly without them.
            c.summary.split("\n\n").map((para, i) => <p key={i}>{para}</p>)
          ) : (
            <p className="reserved">No summary available for this game.</p>
          )}
        </div>
      </div>

      {/* One sentence, no column names. The metric names are on the axis bars and in
          the methodology; repeating them here interrupted the only line on the page
          meant to be read rather than scanned. */}
      <p className="takeaway" key={`tk-${c.game_id}`}>{takeaway(c)}</p>

      {c.season_log_source === "postgres" && (
        <>
          <SectionHead title={ex("Season")} sub="points vs closing line over season" />
          <SeasonStrip log={c.season_log} flaggedDate={c.game_date} />
        </>
      )}

      {/* The score's own structure organizes the page: each group's bar heads
          the section that holds its evidence, so proximity = relevance. */}

      <SectionHead title={ex("Performance")}
                   sub="production + involvement box score stats compared to his own season"
                   kind="prf" value={c.g_performance} />
      <EqTable left={gsRows} right={efRows} leftSum={gsSum} rightSum={efSum}
               showZero />
      <div className="subhead">Game context</div>
      <EqTable left={ctxLeft} right={ctxRight} />

      <SectionHead title={ex("Game profile")}
                   sub="every shot attempt + play-by-play summary · click for video clip" />
      <div className="shotrow">
        <ShotChart playerId={c.player_id} gameId={c.game_id} />
        <PlayTimeline playerId={c.player_id} gameId={c.game_id} />
      </div>

      <SectionHead title={ex("Market")} sub="from sportsbook prop bets (e.g., FanDuel)"
                   kind="mkt" value={c.g_market} />
      <EqTable left={mktLeft} right={mktRight} showZero
               vs="vs league — lower = more under-side pressure" />
      {/* The caveat belongs with the numbers it qualifies, not after the chart:
          it explains why these components are price-and-movement rather than
          volume, which is a question the table raises and the chart does not. */}
      {c.line_pulled ? (
        <p className="source alert">
          The book withdrew this line before tip and he played anyway.
        </p>
      ) : (
        <p className="source">
          Using price and line movement since public data does not reveal bet volumes
        </p>
      )}
      {/* The table above is the two endpoints the SCORE uses. This is the path
          between them -- pulled separately as 'poll' rows and deliberately not an
          input to any block, so it illustrates the market without changing it. */}
      <LineHistory playerId={c.player_id} gameId={c.game_id} />

      <SectionHead title={ex("Motive")} sub="his financial incentive"
                   kind="mtv" value={c.g_motive} />
      <div className="motive-big">
        <b>{money(c.salary)}</b>
        <span>
          {c.has_listed_salary
            ? `Bottom ${Math.round((c.salary_pct ?? 0) * 100)}% of league salaries`
            : "no listed salary — a two-way / 10-day contract: the lowest-paid, most exposed profile, so it takes maximum motive weight"}
        </span>
      </div>
      {/* The motive block's second term: the career-instability RATE (team
          changes + gap seasons, gaps weighted double, per season of span). The
          payload carries the rate and the gap-season count, not the team-change
          count, so the gloss names only what the numbers actually are. Null --
          older backend, or a player with no career row -- renders nothing:
          missing history is absence, not a rate of zero. */}
      {c.instability_career != null && (
        <div className="motive-big">
          <b>{c.instability_career.toFixed(2)}</b>
          <span>
            {"career instability — team changes + gap seasons per season of his career span"}
            {c.gap_seasons != null && c.gap_seasons > 0 &&
              ` · ${c.gap_seasons} gap season${c.gap_seasons === 1 ? "" : "s"} with no NBA team, weighted double`}
          </span>
        </div>
      )}
      {/* The picture behind the instability line above: which teams, which
          seasons, where the gaps were. Fetches its own per-player payload and
          renders nothing when there is no recorded history. */}
      <CareerStrip playerId={c.player_id} />
      <div className="source">
        Motive is 0.75 salary (inverted percentile — research shows that higher-paid
        players are less likely to engage in insider trading) + 0.25 career instability
      </div>
      </motion.div>
    </div>
  );
}
