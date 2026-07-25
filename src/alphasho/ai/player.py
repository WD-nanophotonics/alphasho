"""Desktop Player adapter for AlphaZero checkpoints."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from alphasho.ai.mcts import MCTS, SearchCancelled, SearchConfig
from alphasho.ai.model import load_inference_checkpoint
from alphasho.ai.time_manager import (
    MoveSearchInfo,
    PlayerSearchProgress,
    allocate_thinking_budget,
)
from alphasho.domain import (
    ClockState,
    Color,
    GameSnapshot,
    MoveOption,
    ThinkingConfig,
    TimeControl,
)
from alphasho.engine import SearchPosition


class AlphaZeroPlayer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        seed: int | None = None,
        thinking: ThinkingConfig | None = None,
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
        self.thinking = thinking or ThinkingConfig()
        self.last_search_info: MoveSearchInfo | None = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def choose_move(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
    ) -> MoveOption:
        return self._choose(
            snapshot,
            legal_moves,
            min_playouts=0,
            max_playouts=self.search_config.n_playouts,
            deadline=None,
            max_depth=None,
        )

    def choose_move_with_context(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
        *,
        clock_state: ClockState,
        time_control: TimeControl,
        color: Color,
        progress_callback: Callable[[PlayerSearchProgress], None] | None = None,
    ) -> MoveOption:
        budget = allocate_thinking_budget(
            self.thinking,
            time_control,
            clock_state,
            color,
        )
        started = time.monotonic()
        deadline = (
            None
            if budget.time_limit_seconds is None
            else started + budget.time_limit_seconds
        )

        def report(progress) -> None:
            if progress_callback is None:
                return
            progress_callback(
                PlayerSearchProgress(
                    completed_playouts=progress.completed_playouts,
                    max_playouts=budget.max_playouts,
                    elapsed_seconds=progress.elapsed_seconds,
                    budget_seconds=budget.time_limit_seconds,
                    max_depth=progress.max_depth,
                    root_value=progress.root_value,
                    principal_variation=progress.principal_variation,
                )
            )

        return self._choose(
            snapshot,
            legal_moves,
            min_playouts=budget.min_playouts,
            max_playouts=budget.max_playouts,
            deadline=deadline,
            max_depth=budget.max_depth,
            progress_callback=report,
            budget_seconds=budget.time_limit_seconds,
        )

    def _choose(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
        *,
        min_playouts: int,
        max_playouts: int,
        deadline: float | None,
        max_depth: int | None,
        progress_callback=None,
        budget_seconds: float | None = None,
    ) -> MoveOption:
        if not legal_moves:
            raise ValueError("AlphaZeroPlayer requires at least one legal move")
        self._cancel_event.clear()
        position = SearchPosition.from_snapshot(snapshot)
        terminal = position.terminal_value()
        if terminal is not None:
            raise RuntimeError(
                "UI supplied legal moves for a terminal search position: "
                f"sfen={position.sfen}, repetition_count={position.repetition_count}, "
                f"legal_moves={len(legal_moves)}, terminal_value={terminal}"
            )
        self._mcts.reset()
        result = self._mcts.search(
            position,
            temperature=1e-3,
            add_root_noise=False,
            cancel_event=self._cancel_event,
            progress_callback=progress_callback,
            heartbeat_seconds=0.25,
            min_playouts=min_playouts,
            max_playouts=max_playouts,
            deadline=deadline,
            max_depth=max_depth,
        )
        self.last_search_info = MoveSearchInfo(
            elapsed_seconds=result.stats.elapsed_seconds,
            budget_seconds=budget_seconds,
            playouts=result.stats.playouts,
            max_depth=result.stats.max_depth,
            root_value=result.root_value,
            principal_variation=result.principal_variation,
        )
        usi = result.choose(self._rng)
        legal_by_usi = {move.usi: move for move in legal_moves}
        move = legal_by_usi.get(usi)
        if move is None:
            raise RuntimeError(f"Search returned a move outside the UI legal set: {usi}")
        return move


__all__ = ["AlphaZeroPlayer", "SearchCancelled"]
