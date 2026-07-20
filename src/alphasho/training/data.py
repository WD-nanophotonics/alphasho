"""Compressed, checksummed replay shards."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alphasho.atomic_io import atomic_write


@dataclass(slots=True)
class TrainingSample:
    features1: NDArray[np.uint8]
    features2: NDArray[np.uint8]
    policy_labels: NDArray[np.uint16]
    policy_probabilities: NDArray[np.float16]
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShardEntry:
    path: Path
    samples: int
    source: str
    sha256: str
    model_version: str
    shard_id: str = ""
    ruleset_id: str = ""


class ShardStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"

    def write(
        self,
        samples: list[TrainingSample],
        *,
        source: str,
        model_version: str,
        shard_id: str | None = None,
        ruleset_id: str = "",
    ) -> ShardEntry:
        if not samples:
            raise ValueError("Cannot write an empty replay shard")
        if shard_id:
            existing = next(
                (entry for entry in self.entries() if entry.shard_id == shard_id),
                None,
            )
            if existing is not None:
                if (
                    existing.source != source
                    or existing.model_version != model_version
                    or existing.ruleset_id != ruleset_id
                ):
                    raise ValueError(
                        f"Replay shard id {shard_id!r} was already used "
                        "with different provenance"
                    )
                if not existing.path.exists():
                    raise FileNotFoundError(
                        f"Replay manifest points to a missing shard: {existing.path}"
                    )
                if _sha256(existing.path) != existing.sha256:
                    raise ValueError(
                        f"Replay shard checksum mismatch: {existing.path}"
                    )
                return existing
            safe_id = "".join(
                character
                if character.isalnum() or character in {"-", "_"}
                else "-"
                for character in shard_id
            )
            name = f"{safe_id}.npz"
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            nonce = f"{time.time_ns() & 0xFFFFFF:06x}"
            name = f"{stamp}-{source}-{nonce}.npz"
        destination = self.root / name

        offsets = [0]
        labels: list[NDArray[np.uint16]] = []
        probabilities: list[NDArray[np.float16]] = []
        for sample in samples:
            if len(sample.policy_labels) != len(sample.policy_probabilities):
                raise ValueError("Policy labels and probabilities have different sizes")
            labels.append(np.asarray(sample.policy_labels, dtype=np.uint16))
            probabilities.append(
                np.asarray(sample.policy_probabilities, dtype=np.float16)
            )
            offsets.append(offsets[-1] + len(sample.policy_labels))
        metadata_text = [
            json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True)
            for sample in samples
        ]
        max_metadata = max(1, max(len(item) for item in metadata_text))

        def write_shard(path: Path) -> None:
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray([2], dtype=np.uint16),
                    features1=np.stack(
                        [sample.features1 for sample in samples]
                    ).astype(np.uint8, copy=False),
                    features2=np.stack(
                        [sample.features2 for sample in samples]
                    ).astype(np.uint8, copy=False),
                    policy_offsets=np.asarray(offsets, dtype=np.uint32),
                    policy_labels=np.concatenate(labels),
                    policy_probabilities=np.concatenate(probabilities),
                    values=np.asarray(
                        [sample.value for sample in samples],
                        dtype=np.float16,
                    ),
                    metadata=np.asarray(
                        metadata_text,
                        dtype=f"<U{max_metadata}",
                    ),
                )

        atomic_write(destination, write_shard)
        digest = _sha256(destination)
        entry = ShardEntry(
            destination,
            len(samples),
            source,
            digest,
            model_version,
            shard_id or "",
            ruleset_id,
        )
        manifest_row = {
            "schema_version": 1,
            "path": destination.name,
            "samples": entry.samples,
            "source": source,
            "sha256": digest,
            "model_version": model_version,
            "shard_id": shard_id or "",
            "ruleset_id": ruleset_id,
            "created_unix": time.time(),
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[ShardEntry]:
        if not self.manifest_path.exists():
            return []
        result: list[ShardEntry] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result.append(
                ShardEntry(
                    self.root / str(row["path"]),
                    int(row["samples"]),
                    str(row["source"]),
                    str(row["sha256"]),
                    str(row.get("model_version", "")),
                    str(row.get("shard_id", "")),
                    str(row.get("ruleset_id", "")),
                )
            )
        return result

    def load_entry(self, entry: ShardEntry, *, verify: bool = True) -> list[TrainingSample]:
        if verify and _sha256(entry.path) != entry.sha256:
            raise ValueError(f"Replay shard checksum mismatch: {entry.path}")
        with np.load(entry.path, allow_pickle=False) as arrays:
            features1 = arrays["features1"]
            features2 = arrays["features2"]
            offsets = arrays["policy_offsets"]
            labels = arrays["policy_labels"]
            probabilities = arrays["policy_probabilities"]
            values = arrays["values"]
            metadata = arrays["metadata"]
            samples: list[TrainingSample] = []
            for index in range(len(values)):
                start, end = int(offsets[index]), int(offsets[index + 1])
                samples.append(
                    TrainingSample(
                        features1[index].astype(np.uint8, copy=True),
                        features2[index].astype(np.uint8, copy=True),
                        labels[start:end].astype(np.uint16, copy=True),
                        probabilities[start:end].astype(np.float16, copy=True),
                        float(values[index]),
                        json.loads(str(metadata[index])),
                    )
                )
        if len(samples) != entry.samples:
            raise ValueError(f"Replay shard sample count mismatch: {entry.path}")
        return samples


def load_recent_samples(
    stores: Iterable[ShardStore],
    capacity: int,
    *,
    ruleset_id: str | None = None,
) -> list[TrainingSample]:
    stores = tuple(stores)
    selected: list[TrainingSample] = []
    entries = [
        entry
        for store in stores
        for entry in store.entries()
        if ruleset_id is None or entry.ruleset_id == ruleset_id
    ]
    entries.sort(key=lambda entry: entry.path.stat().st_mtime, reverse=True)
    for entry in entries:
        store = next(store for store in stores if entry.path.parent == store.root)
        selected.extend(store.load_entry(entry))
        if len(selected) >= capacity:
            break
    return selected[:capacity]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
