"""cshogi/dlshogi-compatible feature and policy encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cshogi import dlshogi
from numpy.typing import NDArray

from alphasho.engine import POLICY_SIZE, SearchablePosition

FEATURES1_NUM = int(dlshogi.FEATURES1_NUM)
FEATURES2_NUM = int(dlshogi.FEATURES2_NUM)

if (FEATURES1_NUM, FEATURES2_NUM, POLICY_SIZE) != (62, 57, 2187):
    raise RuntimeError(
        "AlphaSho requires the cshogi 1.0.4 dlshogi encoding "
        f"(got {FEATURES1_NUM}, {FEATURES2_NUM}, {POLICY_SIZE})"
    )


@dataclass(frozen=True, slots=True)
class EncodedPosition:
    features1: NDArray[np.uint8]
    features2: NDArray[np.uint8]


def encode_position(position: SearchablePosition) -> EncodedPosition:
    features1 = np.empty((FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((FEATURES2_NUM, 9, 9), dtype=np.float32)
    position.fill_features(features1, features2)
    # cshogi 1.0.4's dlshogi planes are binary. Storing uint8 reduces replay size.
    return EncodedPosition(features1.astype(np.uint8), features2.astype(np.uint8))


def legal_policy_mask(position: SearchablePosition) -> NDArray[np.bool_]:
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for action in position.legal_actions():
        mask[action.label] = True
    return mask


def normalize_legal_logits(
    logits: NDArray[np.floating],
    labels: NDArray[np.integer],
) -> NDArray[np.float32]:
    if labels.size == 0:
        return np.empty(0, dtype=np.float32)
    selected = np.asarray(logits, dtype=np.float64)[labels]
    selected -= np.max(selected)
    probabilities = np.exp(selected)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        probabilities.fill(1.0 / len(probabilities))
    else:
        probabilities /= total
    return probabilities.astype(np.float32)
