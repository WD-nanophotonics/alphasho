"""Small, bounded atomic file operations shared by training components."""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_RETRY_DELAYS: tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.70,
    0.90,
)


class FileLockError(RuntimeError):
    """Raised when another process owns an exclusive project lock."""


def replace_with_retry(
    temporary: str | Path,
    destination: str | Path,
    *,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
) -> int:
    """Replace *destination*, retrying bounded Windows sharing violations.

    The fully-written temporary file is deliberately retained if every attempt
    fails, so a later process can recover it.
    """

    source = Path(temporary)
    target = Path(destination)
    attempts = 0
    while True:
        attempts += 1
        try:
            os.replace(source, target)
            return attempts
        except OSError as exc:
            sharing_violation = isinstance(exc, PermissionError) or getattr(
                exc, "winerror", None
            ) in {5, 32}
            if not sharing_violation or attempts > len(retry_delays):
                exc.add_note(f"Complete temporary file retained at: {source}")
                raise
            time.sleep(max(0.0, float(retry_delays[attempts - 1])))


def atomic_write(
    destination: str | Path,
    writer: Callable[[Path], None],
    *,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
) -> int:
    """Write with a stable ``.tmp`` recovery file and atomic replacement."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    writer(temporary)
    _fsync_file(temporary)
    return replace_with_retry(
        temporary,
        target,
        retry_delays=retry_delays,
    )


def atomic_write_text(
    destination: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
) -> int:
    def write(path: Path) -> None:
        with path.open("w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    return atomic_write(destination, write, retry_delays=retry_delays)


def atomic_write_json(
    destination: str | Path,
    values: Any,
    *,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
) -> int:
    return atomic_write_text(
        destination,
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True),
        retry_delays=retry_delays,
    )


def atomic_copy(
    source: str | Path,
    destination: str | Path,
    *,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
) -> int:
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        return 1
    return atomic_write(
        destination_path,
        lambda temporary: temporary.write_bytes(source_path.read_bytes()),
        retry_delays=retry_delays,
    )


def recover_newer_temporary(
    destination: str | Path,
    *,
    validator: Callable[[Path], Any],
    revision: Callable[[Any], int],
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
) -> tuple[Any | None, bool, int]:
    """Load the newest valid committed or temporary representation.

    Revision wins when present. Modification time breaks ties for legacy files
    that predate revision tracking.
    """

    target = Path(destination)
    temporary = target.with_suffix(target.suffix + ".tmp")
    candidates: list[tuple[int, int, Path, Any]] = []
    for path in (target, temporary):
        if not path.exists():
            continue
        try:
            value = validator(path)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        candidates.append(
            (
                int(revision(value)),
                path.stat().st_mtime_ns,
                path,
                value,
            )
        )
    if not candidates:
        return None, False, 0
    _revision, _modified, selected, value = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    if selected == temporary:
        attempts = replace_with_retry(
            temporary,
            target,
            retry_delays=retry_delays,
        )
        return value, True, attempts
    return value, False, 1


class ExclusiveFileLock:
    """A non-blocking cross-platform lock released automatically on process exit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            owner = self._read_owner()
            detail = f" ({owner})" if owner else ""
            raise FileLockError(
                f"Another Trainer is already using {self.path.parent}{detail}"
            ) from exc
        self._handle = handle
        metadata = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_unix": time.time(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        handle.seek(1)
        handle.truncate()
        handle.write(metadata)
        handle.flush()

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> ExclusiveFileLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:  # noqa: ANN001
        self.release()

    def _read_owner(self) -> str:
        try:
            raw = self.path.read_bytes()[1:].decode("utf-8")
            values = json.loads(raw)
            return f"pid={values.get('pid')}, host={values.get('host')}"
        except (OSError, ValueError, UnicodeDecodeError):
            return ""


def _fsync_file(path: Path) -> None:
    # Windows rejects fsync() for a descriptor opened read-only.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
