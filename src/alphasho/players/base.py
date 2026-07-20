from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from alphasho.domain import GameSnapshot, MoveOption


class Player(Protocol):
    """Common boundary for random, MCTS, neural-network, and human players."""

    def choose_move(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
    ) -> MoveOption:
        ...

