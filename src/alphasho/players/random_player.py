from __future__ import annotations

import random
from collections.abc import Sequence

from alphasho.domain import GameSnapshot, MoveOption


class RandomPlayer:
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def choose_move(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[MoveOption],
    ) -> MoveOption:
        del snapshot
        if not legal_moves:
            raise ValueError("RandomPlayer requires at least one legal move")
        return self._random.choice(legal_moves)

