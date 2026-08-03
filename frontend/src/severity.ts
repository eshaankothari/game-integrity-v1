// One red line for the whole UI. Red is reserved for the extreme tail of the
// watchlist, not every review-threshold crossing -- score text, the case
// gauge, and both calendars all flip at the same number, so red always means
// the same thing wherever it appears.
export const RED_SCORE = 0.85;

// Shared calendar shading: cool blues below the red line, red above it.
export function scoreLevel(score: number): string {
  if (score >= RED_SCORE) return "rev";
  if (score >= 0.72) return "lv3";
  if (score >= 0.6) return "lv2";
  return "lv1";
}
