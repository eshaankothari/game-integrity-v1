// One red line for the whole UI. Red is reserved for the extreme tail of the
// watchlist, not every review-threshold crossing -- score text, the case
// gauge, and both calendars all flip at the same number, so red always means
// the same thing wherever it appears.
//
// THESE ARE ON THE 0-100 SCALE, and they are CUTOFFS ON A LINEAR RESCALE of the
// raw score -- not percentiles. Chosen so each one costs a known share of the
// population, which is the only way a colour threshold means anything:
//
//     73.7 -> the top 1%,  155 games    RED
//     67.3 -> the top 3%,  465 games
//     57.9 -> the top 10%, 1,554 games
//
// They looked like percentiles once (99 / 97 / 90) and were wrong: score_100 was
// switched from a percentile to a linear rescale, and on the linear scale
// >= 90 catches TWO games out of 15,494. If the scoring changes again, recompute
// these -- do not carry the numbers across.
export const RED_SCORE = 73.7;

// Shared calendar shading: cool blues below the red line, red above it.
export function scoreLevel(score: number): string {
  if (score >= RED_SCORE) return "rev";
  if (score >= 67.3) return "lv3";
  if (score >= 57.9) return "lv2";
  return "lv1";
}

// Games CONFIRMED suspicious outside the pipeline (league/federal action).
// They render red by designation, whatever their score -- the pipeline's
// cutoffs and ground truth are different things, and the UI shows both
// honestly. Key: `${player_id}-${game_id}`.
export const CONFIRMED = new Set([
  // Malik Beasley · federal investigation
  "1627736-0022300173", // 2023-11-11
  "1627736-0022300401", // 2023-12-25
  "1627736-0022300496", // 2024-01-06
  "1627736-0022300639", // 2024-01-26
  "1627736-0022300840", // 2024-02-27
  "1627736-0022300924", // 2024-03-10 (outside the shortlist; rides along)
  // Jontay Porter · banned by the NBA
  "1629007-0022300609", // 2024-01-22
  // "1629007-0022300637", // 2024-01-26 (no captured prop line -> never renders)
  "1629007-0022300999", // 2024-03-20
  // Porter 2024-01-06: no box-score row in this dataset -- nothing to key on.
]);

export function isConfirmed(playerId: number, gameId: string): boolean {
  return CONFIRMED.has(`${playerId}-${gameId}`);
}

// One place that formats a 0-100 score, so every surface prints it identically.
// One decimal: the values are dense near the top (the top 1% spans 99.0-100.0),
// so integers would collapse the tail the list exists to separate.
export function fmtScore(score: number | null | undefined): string {
  return score == null ? "—" : score.toFixed(1);
}
