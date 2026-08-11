import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./tokens.css";
import "./app.css";

// The data DOES change under the app now, so the cache must expire.
//
// This was `staleTime: Infinity, refetchOnWindowFocus: false`, on the premise that the
// API served one frozen pipeline run from CSV snapshots. That stopped being true when
// the backend moved to querying Postgres directly: re-running
// `standardize.py run && export_candidates.py` rewrites every score, and with an
// infinite staleTime the tab would keep serving the numbers it fetched on first mount
// until a hard reload -- indefinitely, and with no indication anything was stale.
//
// 30s is short enough that a pipeline re-run shows up on the next navigation, and long
// enough that clicking between views does not refetch the same 4,810-row payload.
// refetchOnWindowFocus picks up a re-run while the tab sat in the background, which is
// exactly when a re-run tends to happen.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, refetchOnWindowFocus: true },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
