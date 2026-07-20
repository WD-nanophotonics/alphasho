"""Desktop Player adapter for AlphaZero checkpoints."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from alphasho.ai.mcts import MCTS, SearchCancelled, SearchConfig
from alphasho.ai.model import load_inference_checkpoint
from alphasho.domain import GameSnapshot, MoveOption
from alphasho.engine import SearchPosition


class AlphaZeroPlayer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        _model, evaluator, metadata = load_inference_checkpoint(
            checkpoint_path,
            device=device,
        )
        search_values = metadata.get("search_config", {})
        self.search_config = SearchConfig.from_dict(search_values)
        self._mcts = MCTS(evaluator, self.search_config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def choose_move(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
    ) -> MoveOption:
        if not legal_moves:
            raise ValueError("AlphaZeroPlayer requires at least one legal move")
        self._cancel_event.clear()
        position = SearchPosition.from_snapshot(snapshot)
        self._mcts.reset()
        result = self._mcts.search(
            position,
            temperature=1e-3,
            add_root_noise=False,
            cancel_event=self._cancel_event,
        )
        usi = result.choose(self._rng)
        legal_by_usi = {move.usi: move for move in legal_moves}
        move = legal_by_usi.get(usi)
        if move is None:
            raise RuntimeError(f"Search returned a move outside the UI legal set: {usi}")
        return move


__all__ = ["AlphaZeroPlayer", "SearchCancelled"]
