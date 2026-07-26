"""Deterministic starting-position pools for training experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from alphasho.engine import SearchPosition
from alphasho.training.adjudication import AdjudicationRules

SEARCH_RULES_VERSION = 2


class TrainingStartMode(StrEnum):
    STANDARD = "standard"
    TACTICAL = "tactical"


@dataclass(frozen=True, slots=True)
class TacticalPosition:
    name: str
    sfen: str
    capture_moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationPosition:
    """A named, fixed opening used only for candidate/best evaluation."""

    name: str
    sfen: str


@dataclass(frozen=True, slots=True)
class StartPositionConfig:
    mode: TrainingStartMode = TrainingStartMode.STANDARD
    path: Path | None = None

    def positions(self) -> tuple[TacticalPosition, ...]:
        if self.mode is TrainingStartMode.STANDARD:
            return ()
        if self.path is None:
            raise ValueError("Tactical start positions require a JSONL path")
        return load_tactical_positions(self.path)

    def initial_sfens(self) -> tuple[str | None, ...]:
        positions = self.positions()
        return (None,) if not positions else tuple(item.sfen for item in positions)

    def choose(self, index: int) -> str | None:
        sfens = self.initial_sfens()
        return sfens[index % len(sfens)]

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.mode.value.encode("utf-8"))
        if self.path is not None:
            digest.update(self.path.resolve().as_posix().encode("utf-8"))
            digest.update(self.path.read_bytes())
        return digest.hexdigest()[:16]


def ruleset_id(
    rules: AdjudicationRules,
    starts: StartPositionConfig,
) -> str:
    payload = (
        f"search-rules-v{SEARCH_RULES_VERSION}:"
        f"{rules.fingerprint}:{starts.fingerprint()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def load_tactical_positions(path: str | Path) -> tuple[TacticalPosition, ...]:
    source = Path(path)
    positions: list[TacticalPosition] = []
    names: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
            position = TacticalPosition(
                name=str(row["name"]),
                sfen=str(row["sfen"]),
                capture_moves=tuple(str(move) for move in row["capture_moves"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid tactical position at {source}:{line_number}"
            ) from exc
        if position.name in names:
            raise ValueError(f"Duplicate tactical position name: {position.name}")
        board = SearchPosition(position.sfen)
        legal = {action.usi: action for action in board.legal_actions()}
        if not position.capture_moves:
            raise ValueError(f"{position.name}: capture_moves cannot be empty")
        for usi in position.capture_moves:
            action = legal.get(usi)
            if action is None:
                raise ValueError(f"{position.name}: illegal capture move {usi}")
            if board.captured_piece(action) is None:
                raise ValueError(f"{position.name}: {usi} is not a capture")
        names.add(position.name)
        positions.append(position)
    if not positions:
        raise ValueError(f"No tactical positions found in {source}")
    return tuple(positions)


def load_evaluation_positions(
    path: str | Path,
) -> tuple[EvaluationPosition, ...]:
    """Load and validate a deterministic evaluation opening suite."""

    source = Path(path)
    positions: list[EvaluationPosition] = []
    names: set[str] = set()
    sfens: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
            position = EvaluationPosition(
                name=str(row["name"]).strip(),
                sfen=str(row["sfen"]).strip(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid evaluation position at {source}:{line_number}"
            ) from exc
        if not position.name:
            raise ValueError(
                f"Evaluation position name cannot be empty at "
                f"{source}:{line_number}"
            )
        if position.name in names:
            raise ValueError(
                f"Duplicate evaluation position name: {position.name}"
            )
        if position.sfen in sfens:
            raise ValueError(
                f"Duplicate evaluation position SFEN: {position.name}"
            )
        board = SearchPosition(position.sfen)
        if not board.legal_actions():
            raise ValueError(
                f"{position.name}: evaluation position has no legal moves"
            )
        names.add(position.name)
        sfens.add(position.sfen)
        positions.append(position)
    if not positions:
        raise ValueError(f"No evaluation positions found in {source}")
    return tuple(positions)


def evaluation_suite_id(
    positions: tuple[EvaluationPosition, ...],
) -> str:
    """Return a stable content hash independent of the suite file path."""

    payload = [
        {"name": position.name, "sfen": position.sfen}
        for position in positions
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]
