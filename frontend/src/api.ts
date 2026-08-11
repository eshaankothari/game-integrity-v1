// Typed access to the FastAPI backend. One fetch helper, two endpoint
// functions, and the row shapes -- nothing clever, so the contract with
// server/app.py is readable in one screen.

export interface WatchlistRow {
  rank: number;
  player: string;
  position: string | null;
  tier: string | null;
  game_date: string;
  matchup: string;
  minutes: number;
  points: number;
  line: number;
  line_source: "close" | "open_pulled";
  line_pulled: boolean;
  close_under: number | null;
  salary: number | null;
  has_listed_salary: boolean;
  // RAW, z-scale: population-independent, the number to compare across runs.
  score: number;
  // 0-100: a LINEAR RESCALE of `score` (not a percentile -- see severity.ts).
  // What the UI renders. A strictly monotone transform, so it never reorders
  // anything.
  score_100: number;
  performance_100: number | null;
  market_100: number | null;
  motive_100: number | null;
  // 1 - points/line, clipped to [0,1]: how decisively the under hit
  shortfall: number | null;
  // did he finish below the line? null when there is no line to compare
  under_hit: boolean | null;
  // The real column names. prod_z is an alias of game_z kept for older components.
  game_z: number | null;
  effort_z: number | null;
  prod_z: number | null;
  // shortfall standardized vs the league (already player-relative through the
  // line, so no per-player second pass), and the raw Hollinger game score
  shortfall_z: number | null;
  game_score: number | null;
  // the market block's four standardized components (league z, oriented so
  // HIGH = under-side pressure): closing-price lean, line level, line
  // movement, price-only movement. The latter two are null without an
  // opening quote.
  mk_p_price: number | null;
  mk_p_line: number | null;
  mk_line_mv: number | null;
  mk_price_mv: number | null;
  // the three blocks on 0-100, aliased server-side to the names the components read
  g_performance: number;
  g_market: number;
  g_motive: number;
  // independence tail from analysis/out/combined_cut.csv, where that run
  // scored the same player-game (null where the two runs disagree on cuts)
  tail_pct: number | null;
  player_id: number;
  game_id: string;
}

// Enrichment from the deeper analysis run: hustle counts with no season
// baseline, and the exit-anatomy fields.
export interface DeepDetail {
  indep_pct: number | null;
  contested_shots: number | null;
  deflections: number | null;
  loose_balls: number | null;
  box_outs: number | null;
  passes: number | null;
  steals: number | null;
  blocks: number | null;
  game_score: number | null;
  ejected: boolean | null;
  n_stints: number | null;
  last_out_sec: number | null;
  points_competitive: number | null;
  points_garbage: number | null;
}

// Everything the case view adds: full box score, the residual z-scores that
// drive the red grading, and both ends of the sportsbook line.
export interface CaseDetail extends WatchlistRow {
  fga: number;
  rebounds: number;
  assists: number;
  usage_pct: number | null;
  turnover_ratio: number | null;
  distance: number | null;
  touches: number | null;
  points_resid_z: number | null;
  fga_resid_z: number | null;
  rebounds_resid_z: number | null;
  assists_resid_z: number | null;
  usage_pct_resid_z: number | null;
  turnover_ratio_resid_z: number | null;
  distance_resid_z: number | null;
  touches_resid_z: number | null;
  minutes_resid_z: number | null;
  close_line: number | null;
  open_line: number | null;
  open_under: number | null;
  line_move_pct: number | null;
  under_move_pct: number | null;
  game_margin: number | null;
  plus_minus: number | null;
  fouls: number | null;
  started: boolean | null;
  // team totals summed from the box scores; result is from HIS team's side
  final_score: {
    away: { team: string; team_id: number; pts: number };
    home: { team: string; team_id: number; pts: number };
    result: "W" | "L" | null;
  } | null;
  // roster bio; age is as of the game date, not today
  age: number | null;
  height_in: number | null;
  weight_lb: number | null;
  experience: number | null;
  school: string | null;
  n_player_games: number | null;
  salary_pct: number | null;
  // Written case, generated from the same evidence packet the LLM reviewer reads.
  summary: string | null;
  // The innocent explanations the DATA supports, same taxonomy as the reviewer.
  // A single entry of `none_found` means the packet contains none.
  summary_flags: Array<{ cause: string; detail: string }>;
  // "rules" today; will name the model when the L9 reviewer overrides it. A reader
  // needs to know whether a sentence was derived or generated.
  summary_source: string;
  ai_summary: string | null;
  deep: DeepDetail | null;
  season_log: SeasonGame[];
  season_log_source: "postgres" | "unavailable";
}

// One game in the player's season, for the strip above the box score.
export interface SeasonGame {
  game_date: string;
  matchup: string;
  minutes: number | null;
  points: number | null;
  close_line: number | null;
  margin_vs_line: number | null;
}

export interface Summary {
  scored: number;
  shortlist: number;
  review_tail: number;
  review_tail_shortlist: number;
  // 0-100, the same scale the UI renders. 99 = the top 1%.
  review_threshold: number;
  pulled_and_played: number;
  unlisted_salary: number;
  top_score: number;
  median_score: number;
  // the raw z-scale equivalents, for anything that needs a stable number
  top_score_raw: number;
  median_score_raw: number;
  scale_note: string;
  histogram: Array<{ lo: number; hi: number; n: number }>;
}

export interface FunnelStage {
  stage: string;
  n: number;
}

export interface IsoRow {
  iso_rank: number;
  iso_score: number;
  player: string;
  game_date: string;
  matchup: string;
  points: number | null;
  close_line: number | null;
  composite_rank: number | null;
  player_id: number;
  game_id: string;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export function fetchWatchlist(limit = 50) {
  return get<{ total: number; rows: WatchlistRow[] }>(
    `/api/watchlist?limit=${limit}`,
  );
}

export function fetchCase(playerId: number, gameId: string) {
  return get<CaseDetail>(`/api/case/${playerId}/${gameId}`);
}

// One field-goal attempt from the cached play-by-play: legacy court
// coordinates (hoop at origin, 1 unit = 0.1 ft), result, and whether NBA has
// a clip for the event.
export interface ShotEvent {
  action_number: number;
  period: number;
  clock: string;
  x: number;
  y: number;
  distance: number | null;
  made: boolean;
  value: number | null;
  description: string;
  video: boolean;
}

// One pre-tip quote. `minutes_before_tip` is measured from the snapshot the API
// actually returned, not the one requested -- the historical endpoint snaps to a
// ~5-minute grid and answers ~4 minutes early, so the requested time would be a
// constant lie. `role` is 'poll' for a curve point, 'open'/'close' for the anchors.
export interface LineQuote {
  minutes_before_tip: number;
  role: "open" | "poll" | "close";
  line: number | null;
  under_price: number | null;
  over_price: number | null;
  p_under: number | null;
}

// `books` reports which sources have quotes for THIS player-game. `live: false`
// marks a venue the UI offers but cannot serve yet (the prediction markets), so a
// chip can be shown as coming rather than opening an empty chart.
export interface BookOption {
  key: string;
  n: number;
  live: boolean;
}

export function fetchLineHistory(playerId: number, gameId: string, book?: string) {
  return get<{ series: LineQuote[]; book: string; n: number; books: BookOption[] }>(
    `/api/case/${playerId}/${gameId}/line-history${book ? `?book=${book}` : ""}`,
  );
}

export function fetchShots(playerId: number, gameId: string) {
  return get<{ shots: ShotEvent[]; source: string }>(
    `/api/case/${playerId}/${gameId}/shots`,
  );
}

// One timeline event: everything the player was involved in, in game order.
// `made` is tri-state -- true/false for shots and free throws, null for
// fouls, rebounds, subs and the rest.
export interface PlayEvent {
  action_number: number;
  period: number;
  clock: string;
  description: string;
  action_type: string;
  made: boolean | null;
  video: boolean;
  score: string | null;
}

export function fetchPlays(playerId: number, gameId: string) {
  return get<{ plays: PlayEvent[]; source: string }>(
    `/api/case/${playerId}/${gameId}/plays`,
  );
}

// The event's clip on NBA.com, same URL shape the site itself uses. Opened
// directly in a new tab: NBA's bot wall admits real browsers only, so any
// in-app fetch (backend OR frontend) gets blocked -- the link never is.
export function nbaEventUrl(gameId: string, eventId: number, title: string) {
  return (
    `https://www.nba.com/stats/events?CFID=&CFPARAMS=&GameEventID=${eventId}` +
    `&GameID=${gameId}&Season=2023-24&flag=1&title=${encodeURIComponent(title)}`
  );
}

export function fetchSummary() {
  return get<Summary>("/api/summary");
}

export function fetchFunnel() {
  return get<{ stages: FunnelStage[] }>("/api/funnel");
}

export function fetchIsolation(limit = 8) {
  return get<{ rows: IsoRow[] }>(`/api/isolation?limit=${limit}`);
}

// One player's scored games, worst first, for the case-side roll-up.
export interface PlayerFlag {
  rank: number | null;
  player: string;
  game_date: string;
  matchup: string;
  minutes: number;
  points: number;
  line: number;
  score: number;
  score_100: number;
  game_id: string;
  player_id: number;
  tail_pct: number | null;
}

export function fetchPlayerFlags(playerId: number) {
  return get<{ player: string; rows: PlayerFlag[] }>(`/api/player/${playerId}/flags`);
}

// One cell per game day; the click target is the day's worst composite.
export interface CalendarDay {
  date: string;
  n: number;
  max_score: number;
  review: number;
  // the day holds a game confirmed suspicious by league / federal action
  confirmed: boolean;
  player: string;
  player_id: number;
  game_id: string;
}

export function fetchCalendar() {
  return get<{ days: CalendarDay[] }>("/api/calendar");
}

// One node per PROPPED player-game (all 15,494), placed on
// (performance, market, motive). Eliminated games arrive too, with
// in_ledger=false and `cut` naming the cut that removed them -- they are the
// context that makes the shortlist's position in the space legible.
export interface CloudNode {
  player: string;
  game_date: string;
  player_id: number;
  game_id: string;
  performance: number;
  market: number;
  motive: number;
  score_100: number | null;
  rank: number | null;
  points: number | null;
  line: number | null;
  tail_pct: number | null;
  in_ledger: boolean;
  // number of the first cut this game failed; null if it passed all six
  cut: number | null;
}

export function fetchCloud() {
  // `cuts` maps cut number -> label once, rather than repeating a 40-char
  // string on 10,684 nodes.
  return get<{ nodes: CloudNode[]; cuts: Record<string, string>; shortlist: number }>(
    "/api/cloud",
  );
}

// One player, with his worst game attached as the click target. Ranked by how
// many of his propped games reach the review tail (top 1% of all scores).
export interface PlayerRow {
  position: number;
  player_id: number;
  player: string;
  n_games: number;
  n_tail: number;
  n_shortlist: number;
  best: number;
  best_rank: number;
  // mean of this player's TOP_K worst games -- what the list is ranked by
  topk: number;
  // pinned to the top for demonstration; `position` still reports where the
  // metric actually placed him
  pinned: boolean;
  // his worst game -- what a click opens
  game_id: string;
  game_date: string;
  worst_score: number;
  worst_rank: number | null;
  worst_in_shortlist: boolean;
  points: number | null;
  close_line: number | null;
  minutes: number | null;
}

export function fetchPlayers(limit = 70) {
  return get<{
    rows: PlayerRow[];
    top_k: number;
    pinned: string[];
    total_players: number;
  }>(`/api/players?limit=${limit}`);
}
