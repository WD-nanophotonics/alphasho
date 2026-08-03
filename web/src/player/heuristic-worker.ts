/// <reference lib="webworker" />
import { Position } from "../engine/position";
import { AlphaBetaSearch } from "../engine/search";
import { SearchBudget } from "../engine/types";

interface StartMessage { type: "search/start"; id: number; sfen: string; history: string[]; budget: SearchBudget }
interface CancelMessage { type: "search/cancel"; id: number }

let active: { id: number; search: AlphaBetaSearch } | null = null;
self.onmessage = (event: MessageEvent<StartMessage | CancelMessage>) => {
  if (event.data.type === "search/cancel") {
    if (active?.id === event.data.id) active.search.cancel();
    return;
  }
  const { id, sfen, budget } = event.data;
  try {
    const position = event.data.history.length ? Position.fromHistory(event.data.history, sfen) : new Position(sfen);
    const search = new AlphaBetaSearch(position);
    active = { id, search };
    const result = search.run(budget, (progress) => self.postMessage({ type: "search/progress", id, progress }));
    self.postMessage({ type: "search/result", id, result });
  } catch (error) {
    self.postMessage({ type: "search/error", id, message: error instanceof Error ? error.message : String(error) });
  } finally {
    if (active?.id === id) active = null;
  }
};

export {};
