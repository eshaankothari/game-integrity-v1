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
  score: number;
  // 1 - points/line, clipped to [0,1]: how decisively the under hit
  shortfall: number | null;
  prod_z: number | null;
  effort_z: number | null;
  // three grouped sliders, derived server-side from the pipeline's weights
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
  n_player_games: number | null;
  salary_pct: number | null;
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
  review_tail: number;
  review_threshold: number;
  pulled_and_played: number;
  unlisted_salary: number;
  top_score: number;
  median_score: number;
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
  rank: number;
  player: string;
  game_date: string;
  matchup: string;
  minutes: number;
  points: number;
  line: number;
  score: number;
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
  player: string;
  player_id: number;
  game_id: string;
}

export function fetchCalendar() {
  return get<{ days: CalendarDay[] }>("/api/calendar");
}

// One node per combined_cut game, placed on (performance, market, motive).
export interface CloudNode {
  player: string;
  game_date: string;
  player_id: number;
  game_id: string;
  performance: number;
  market: number;
  motive: number;
  tail_pct: number | null;
  in_ledger: boolean;
}

export function fetchCloud() {
  return get<{ nodes: CloudNode[] }>("/api/cloud");
}
