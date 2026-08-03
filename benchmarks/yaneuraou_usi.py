"""Optional, dependency-free USI adapter for a user-supplied YaneuraOu binary.

The project never downloads or bundles the engine.  This module is benchmark
infrastructure only and deliberately keeps engine output as auditable records.
"""

from __future__ import annotations

import hashlib
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UsiAnalysis:
    multipv: int
    score_cp: int | None
    score_mate: int | None
    lowerbound: bool
    upperbound: bool
    nodes: int | None
    depth: int | None
    principal_variation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UsiSearchResult:
    bestmove: str | None
    ponder: str | None
    analyses: tuple[UsiAnalysis, ...]
    raw_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UsiMateResult:
    status: str
    principal_variation: tuple[str, ...]
    raw_lines: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_info(line: str) -> UsiAnalysis | None:
    tokens = line.split()
    if not tokens or tokens[0] != "info" or "pv" not in tokens:
        return None
    values: dict[str, int] = {}
    for name in ("multipv", "nodes", "depth"):
        if name in tokens:
            try:
                values[name] = int(tokens[tokens.index(name) + 1])
            except (IndexError, ValueError):
                pass
    score_cp: int | None = None
    score_mate: int | None = None
    if "score" in tokens:
        index = tokens.index("score") + 1
        try:
            kind, raw = tokens[index], tokens[index + 1]
            if kind == "cp":
                score_cp = int(raw)
            elif kind == "mate":
                score_mate = int(raw)
        except (IndexError, ValueError):
            pass
    pv_index = tokens.index("pv") + 1
    return UsiAnalysis(
        multipv=values.get("multipv", 1),
        score_cp=score_cp,
        score_mate=score_mate,
        lowerbound="lowerbound" in tokens,
        upperbound="upperbound" in tokens,
        nodes=values.get("nodes"),
        depth=values.get("depth"),
        principal_variation=tuple(tokens[pv_index:]),
    )


class YaneuraOuUsi:
    """Small synchronous USI client with bounded waits and captured output."""

    def __init__(self, executable: Path, *, timeout_seconds: float = 10.0) -> None:
        executable = executable.resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        self.executable = executable
        self.binary_sha256 = sha256_file(executable)
        self.timeout_seconds = timeout_seconds
        self.ids: dict[str, str] = {}
        self.options: dict[str, str] = {}
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        self._send("usi")
        self._read_until("usiok")
        self.ready()

    def _read_output(self) -> None:
        assert self._process.stdout is not None
        try:
            for raw in self._process.stdout:
                self._lines.put(raw.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _send(self, command: str) -> None:
        if self._process.poll() is not None or self._process.stdin is None:
            raise RuntimeError("USI engine is not running")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _read_until(self, terminal: str, *, timeout: float | None = None) -> list[str]:
        deadline = time.monotonic() + (self.timeout_seconds if timeout is None else timeout)
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"USI response did not reach {terminal!r}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"USI response did not reach {terminal!r}") from exc
            if line is None:
                raise RuntimeError("USI engine exited unexpectedly")
            lines.append(line)
            if line.startswith("id "):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    self.ids[parts[1]] = parts[2]
            elif line.startswith("option name "):
                name = line.removeprefix("option name ").split(" type ", 1)[0]
                self.options[name] = line
            if line == terminal or line.startswith(terminal + " "):
                return lines

    def ready(self) -> None:
        self._send("isready")
        self._read_until("readyok")

    def set_option(self, name: str, value: str | int | bool) -> None:
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        self._send(f"setoption name {name} value {rendered}")

    def new_game(self) -> None:
        self._send("usinewgame")
        self.ready()

    def analyse(
        self,
        sfen: str,
        *,
        nodes: int,
        multipv: int = 1,
        options: dict[str, str | int | bool] | None = None,
        moves: tuple[str, ...] = (),
    ) -> UsiSearchResult:
        for name, value in (options or {}).items():
            self.set_option(name, value)
        self.set_option("MultiPV", multipv)
        self.ready()
        suffix = f" moves {' '.join(moves)}" if moves else ""
        self._send(f"position sfen {sfen}{suffix}")
        self._send(f"go nodes {nodes}")
        lines = self._read_until("bestmove", timeout=self.timeout_seconds)
        best_tokens = lines[-1].split()
        bestmove = best_tokens[1] if len(best_tokens) > 1 and best_tokens[1] != "resign" else None
        ponder = best_tokens[3] if len(best_tokens) > 3 and best_tokens[2] == "ponder" else None
        latest: dict[int, UsiAnalysis] = {}
        for line in lines:
            parsed = parse_info(line)
            if parsed is not None:
                latest[parsed.multipv] = parsed
        return UsiSearchResult(
            bestmove=bestmove,
            ponder=ponder,
            analyses=tuple(latest[key] for key in sorted(latest)),
            raw_lines=tuple(lines),
        )

    def solve_mate(self, sfen: str, *, milliseconds: int) -> UsiMateResult:
        self._send(f"position sfen {sfen}")
        self._send(f"go mate {milliseconds}")
        lines = self._read_until("checkmate", timeout=milliseconds / 1_000 + self.timeout_seconds)
        tokens = lines[-1].split()[1:]
        if tokens == ["nomate"]:
            return UsiMateResult("disproven", (), tuple(lines))
        if tokens in (["timeout"], ["notimplemented"]):
            return UsiMateResult("unknown", (), tuple(lines))
        return UsiMateResult("proven", tuple(tokens), tuple(lines))

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._send("quit")
            self._process.wait(timeout=2.0)
        except (RuntimeError, subprocess.TimeoutExpired):
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

    def __enter__(self) -> YaneuraOuUsi:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
