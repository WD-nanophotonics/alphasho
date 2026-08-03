"""Bounded proof-number style solver for short forcing tsume lines."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

import cshogi

_INFINITY = 1 << 30


class MateStatus(StrEnum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MateResult:
    status: MateStatus
    principal_variation: tuple[str, ...]
    nodes: int
    elapsed_seconds: float
    proof_number: int
    disproof_number: int


@dataclass(frozen=True, slots=True)
class _Proof:
    status: MateStatus
    line: tuple[int, ...] = ()
    proof: int = 1
    disproof: int = 1


class BoundedDfpnMateSolver:
    """Conservative depth/node/time bounded AND/OR mate prover.

    The attacker must check on every attacker ply.  Resource exhaustion and
    path-sensitive repetition are never converted into a mate proof.
    """

    def __init__(
        self,
        board: cshogi.Board,
        *,
        max_ply: int,
        node_limit: int,
        deadline: float | None,
        cancelled: Callable[[], bool] | None = None,
        root_checks: Iterable[int] | None = None,
    ) -> None:
        self.board = board.copy()
        self.attacker = int(board.turn)
        self.max_ply = max(1, int(max_ply) | 1)
        self.node_limit = max(1, int(node_limit))
        self.deadline = deadline
        self.cancelled = cancelled or (lambda: False)
        self.root_checks = tuple(int(move) for move in root_checks or ())
        self.nodes = 0
        self._exhausted = False
        self._tt: dict[tuple[int, bool, int], _Proof] = {}

    def solve(self) -> MateResult:
        started = time.monotonic()
        proof = self._search(True, self.max_ply, set(), self.root_checks or None)
        status = proof.status
        pv: tuple[str, ...] = ()
        if status is MateStatus.PROVEN and self._validate(proof.line):
            pv = tuple(cshogi.move_to_usi(move) for move in proof.line)
        elif status is MateStatus.PROVEN:
            status = MateStatus.UNKNOWN
            proof = _Proof(MateStatus.UNKNOWN)
        return MateResult(
            status=status,
            principal_variation=pv,
            nodes=self.nodes,
            elapsed_seconds=time.monotonic() - started,
            proof_number=proof.proof,
            disproof_number=proof.disproof,
        )

    def _search(
        self,
        attacker_node: bool,
        remaining: int,
        path: set[int],
        supplied_moves: tuple[int, ...] | None = None,
    ) -> _Proof:
        if self._limit_reached():
            return _Proof(MateStatus.UNKNOWN)
        self.nodes += 1
        key_hash = int(self.board.zobrist_hash())
        if key_hash in path:
            # A path repetition cannot establish mate or no-mate safely here.
            return _Proof(MateStatus.UNKNOWN)
        key = (key_hash, attacker_node, remaining)
        cached = self._tt.get(key)
        if cached is not None:
            return cached

        legal = (
            list(supplied_moves)
            if supplied_moves is not None
            else [int(move) for move in self.board.legal_moves]
        )
        if not legal:
            result = (
                _Proof(MateStatus.PROVEN, proof=0, disproof=_INFINITY)
                if not attacker_node
                else _Proof(MateStatus.DISPROVEN, proof=_INFINITY, disproof=0)
            )
            self._tt[key] = result
            return result
        if remaining <= 0:
            # Legal replies still exist after the bounded line is exhausted, so
            # mate is disproven *within this max-ply query* (not globally).
            return _Proof(MateStatus.DISPROVEN, proof=_INFINITY, disproof=0)

        path.add(key_hash)
        try:
            candidates = self._checking_moves(legal) if attacker_node else legal
            if self._exhausted:
                return _Proof(MateStatus.UNKNOWN)
            if attacker_node and not candidates:
                result = _Proof(MateStatus.DISPROVEN, proof=_INFINITY, disproof=0)
                self._tt[key] = result
                return result
            children: list[tuple[int, _Proof]] = []
            for move in candidates:
                if self._limit_reached():
                    return _Proof(MateStatus.UNKNOWN)
                self.board.push(move)
                try:
                    child = self._search(not attacker_node, remaining - 1, path)
                finally:
                    self.board.pop()
                children.append((move, child))
                if attacker_node and child.status is MateStatus.PROVEN:
                    result = _Proof(
                        MateStatus.PROVEN,
                        (move, *child.line),
                        0,
                        _INFINITY,
                    )
                    self._tt[key] = result
                    return result
                if not attacker_node and child.status is MateStatus.DISPROVEN:
                    result = _Proof(
                        MateStatus.DISPROVEN,
                        (move, *child.line),
                        _INFINITY,
                        0,
                    )
                    self._tt[key] = result
                    return result

            statuses = [child.status for _move, child in children]
            if any(status is MateStatus.UNKNOWN for status in statuses):
                return _Proof(MateStatus.UNKNOWN)
            if attacker_node:
                result = _Proof(MateStatus.DISPROVEN, proof=_INFINITY, disproof=0)
            else:
                chosen_move, chosen = max(
                    children, key=lambda item: len(item[1].line)
                )
                result = _Proof(
                    MateStatus.PROVEN,
                    (chosen_move, *chosen.line),
                    0,
                    _INFINITY,
                )
            self._tt[key] = result
            return result
        finally:
            path.remove(key_hash)

    def _checking_moves(self, legal: list[int]) -> list[int]:
        result: list[int] = []
        for move in legal:
            if self._limit_reached():
                break
            self.board.push(move)
            try:
                if self.board.is_check():
                    result.append(move)
            finally:
                self.board.pop()
        return result

    def _limit_reached(self) -> bool:
        if self._exhausted:
            return True
        if self.nodes >= self.node_limit or self.cancelled():
            self._exhausted = True
        elif self.deadline is not None and time.monotonic() >= self.deadline:
            self._exhausted = True
        return self._exhausted

    def _validate(self, line: tuple[int, ...]) -> bool:
        if not line or len(line) % 2 == 0 or len(line) > self.max_ply:
            return False
        board = self.board.copy()
        for ply, move in enumerate(line):
            if move not in {int(candidate) for candidate in board.legal_moves}:
                return False
            board.push(move)
            if ply % 2 == 0 and not board.is_check():
                return False
        return next(iter(board.legal_moves), None) is None and board.is_check()


__all__ = ["BoundedDfpnMateSolver", "MateResult", "MateStatus"]
