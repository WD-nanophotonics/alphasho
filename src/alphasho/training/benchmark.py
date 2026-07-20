"""Cheap fixed tactical diagnostics for policy/value convergence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alphasho.ai import TorchEvaluator, encode_position
from alphasho.engine import SearchPosition
from alphasho.training.positions import load_tactical_positions


@dataclass(frozen=True, slots=True)
class TacticalBenchmarkResult:
    positions: int
    capture_top1_accuracy: float
    capture_probability: float
    positive_value_rate: float
    mean_value: float


def evaluate_tactical_benchmark(
    evaluator: TorchEvaluator,
    path: str | Path,
) -> TacticalBenchmarkResult:
    cases = load_tactical_positions(path)
    positions = [SearchPosition(case.sfen) for case in cases]
    encoded = [encode_position(position) for position in positions]
    logits, values = evaluator.evaluate_batch(
        np.stack([item.features1 for item in encoded]),
        np.stack([item.features2 for item in encoded]),
    )
    top1 = 0
    capture_probability = 0.0
    for row, (case, position) in enumerate(zip(cases, positions, strict=True)):
        legal = position.legal_actions()
        labels = np.asarray([action.label for action in legal], dtype=np.int32)
        selected = np.asarray(logits[row, labels], dtype=np.float64)
        selected -= selected.max()
        probabilities = np.exp(selected)
        probabilities /= probabilities.sum()
        capture_usis = set(case.capture_moves)
        top_action = legal[int(probabilities.argmax())]
        top1 += int(top_action.usi in capture_usis)
        capture_probability += sum(
            float(probability)
            for action, probability in zip(legal, probabilities, strict=True)
            if action.usi in capture_usis
        )
    count = len(cases)
    return TacticalBenchmarkResult(
        positions=count,
        capture_top1_accuracy=top1 / count,
        capture_probability=capture_probability / count,
        positive_value_rate=float(np.mean(values > 0)),
        mean_value=float(np.mean(values)),
    )
