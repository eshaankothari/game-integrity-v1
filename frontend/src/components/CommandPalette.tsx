import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Command } from "cmdk";
import { useEffect, useState } from "react";
import type { Selection } from "../App";
import { fetchCase, fetchWatchlist } from "../api";
import { RED_SCORE } from "../severity";

// ⌘K / Ctrl+K palette: type a player to jump to their case, or run an
// action. Shares the ["watchlist", 50] cache with the rail -- opening the
// palette costs no request -- and highlighting an item prefetches its case.

interface Props {
  onSelect: (s: Selection) => void;
  onToggleTheme: () => void;
}

export function CommandPalette({ onSelect, onToggleTheme }: Props) {
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: ["watchlist", 50],
    queryFn: () => fetchWatchlist(50),
  });

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const jump = (playerId: number, gameId: string) => {
    onSelect({ playerId, gameId });
    setOpen(false);
  };

  return (
    <Command.Dialog open={open} onOpenChange={setOpen} label="Command palette">
      <Command.Input placeholder="Player, matchup, or action…" />
      <Command.List>
        <Command.Empty>Nothing matches.</Command.Empty>
        <Command.Group heading="Cases">
          {data?.rows.map((r) => (
            <Command.Item
              key={`${r.player_id}-${r.game_id}`}
              value={`${r.player} ${r.matchup} ${r.game_date}`}
              onSelect={() => jump(r.player_id, r.game_id)}
              onMouseEnter={() =>
                queryClient.prefetchQuery({
                  queryKey: ["case", r.player_id, r.game_id],
                  queryFn: () => fetchCase(r.player_id, r.game_id),
                })
              }
            >
              <span className="ci-rank">#{r.rank}</span>
              <span className="ci-name">{r.player}</span>
              <span className="ci-meta">
                {r.points} pts / {r.line} · {r.matchup} · {r.game_date}
              </span>
              <span className={`ci-score ${r.score >= RED_SCORE ? "hot" : ""}`}>
                {r.score.toFixed(3)}
              </span>
            </Command.Item>
          ))}
        </Command.Group>
        <Command.Group heading="Actions">
          <Command.Item value="toggle theme navy white"
            onSelect={() => { onToggleTheme(); setOpen(false); }}>
            <span className="ci-name">Toggle navy / white theme</span>
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}
