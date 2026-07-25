"""Isolated RTX-class throughput benchmark for production search settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alphasho.ai import encode_position, load_inference_checkpoint
from alphasho.engine import SearchPosition
from alphasho.training.adjudication import TrainingPosition
from alphasho.training.config import (
    ParallelConfig,
    TrainingConfig,
    load_training_config,
)
from alphasho.training.parallel import (
    CompletedEvaluation,
    CompletedSelfPlay,
    InferenceMetrics,
    run_parallel_evaluation,
    run_parallel_selfplay,
)


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    config: Path
    checkpoint: Path
    mode: str
    games: int
    workers: int
    gpu_batch_size: int
    batch_wait_ms: float
    request_queue_size: int
    output: Path
    device: str | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    logical_cpus: int
    affinity_cpus: int | None
    cgroup_quota_cpus: float | None
    cgroup_cpuset_cpus: int | None
    effective_cpus: float
    gpu_name: str | None
    git_commit: str | None
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    samples: int
    cpu_system_mean_percent: float | None
    cpu_system_p95_percent: float | None
    main_process_mean_percent: float | None
    main_process_p95_percent: float | None
    worker_processes_mean_percent: float | None
    worker_processes_p95_percent: float | None
    process_tree_memory_peak_mb: float | None
    gpu_utilization_mean_percent: float | None
    gpu_utilization_p95_percent: float | None
    gpu_power_mean_watts: float | None
    gpu_memory_peak_mb: float | None
    gpu_memory_total_mb: float | None


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("w", encoding="utf-8")

    def write(self, event: str, **fields: object) -> None:
        row = {
            "time": time.time(),
            "event": event,
            **fields,
        }
        self._stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class ResourceSampler:
    """Best-effort Linux /proc and nvidia-smi sampler."""

    def __init__(self, root_pid: int, interval_seconds: float = 1.0) -> None:
        self._root_pid = root_pid
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="alphasho-benchmark-resource-sampler",
            daemon=True,
        )
        self._samples: list[dict[str, float]] = []
        self._previous_cpu: tuple[int, int] | None = None
        self._previous_process: tuple[float, int, int] | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> ResourceSummary:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval * 3))
        return summarize_resources(self._samples)

    @property
    def samples(self) -> tuple[dict[str, float], ...]:
        return tuple(self._samples)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sample: dict[str, float] = {}
            sample.update(self._sample_cpu())
            sample.update(self._sample_gpu())
            if sample:
                sample["monotonic_seconds"] = time.monotonic()
                self._samples.append(sample)

    def _sample_cpu(self) -> dict[str, float]:
        if os.name != "posix" or not Path("/proc/stat").exists():
            return {}
        result: dict[str, float] = {}
        try:
            fields = [
                int(value)
                for value in Path("/proc/stat").read_text(encoding="ascii").splitlines()[
                    0
                ].split()[1:]
            ]
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            total = sum(fields)
            if self._previous_cpu is not None:
                previous_total, previous_idle = self._previous_cpu
                delta_total = total - previous_total
                if delta_total > 0:
                    result["cpu_system_percent"] = 100.0 * (
                        1.0 - (idle - previous_idle) / delta_total
                    )
            self._previous_cpu = (total, idle)

            now = time.monotonic()
            descendants = _descendant_pids(self._root_pid)
            main_ticks = _process_ticks(self._root_pid)
            worker_ticks = sum(_process_ticks(pid) for pid in descendants)
            memory_bytes = _process_rss_bytes(self._root_pid) + sum(
                _process_rss_bytes(pid) for pid in descendants
            )
            result["process_tree_memory_mb"] = memory_bytes / (1024 * 1024)
            if self._previous_process is not None:
                previous_time, previous_main, previous_workers = self._previous_process
                elapsed = now - previous_time
                clock_ticks = os.sysconf("SC_CLK_TCK")
                if elapsed > 0:
                    result["main_process_percent"] = (
                        max(0, main_ticks - previous_main)
                        * 100.0
                        / clock_ticks
                        / elapsed
                    )
                    result["worker_processes_percent"] = (
                        max(0, worker_ticks - previous_workers)
                        * 100.0
                        / clock_ticks
                        / elapsed
                    )
            self._previous_process = (now, main_ticks, worker_ticks)
        except (OSError, ValueError, IndexError):
            return {}
        return result

    @staticmethod
    def _sample_gpu() -> dict[str, float]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return {}
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            values = [
                float(value.strip())
                for value in completed.stdout.splitlines()[0].split(",")
            ]
            return {
                "gpu_utilization_percent": values[0],
                "gpu_power_watts": values[1],
                "gpu_memory_mb": values[2],
                "gpu_memory_total_mb": values[3],
            }
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return {}


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaSho search without touching training artifacts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("selfplay", "evaluation"),
        required=True,
    )
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--gpu-batch-size", type=int, required=True)
    parser.add_argument("--batch-wait-ms", type=float, required=True)
    parser.add_argument("--request-queue-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)
    options = BenchmarkOptions(**vars(arguments))
    _validate_options(options)
    return options


def run_benchmark(options: BenchmarkOptions) -> dict[str, object]:
    config = load_training_config(options.config)
    _validate_output_dir(options.output, config)
    checkpoint = options.checkpoint.resolve(strict=True)
    device = options.device or config.device
    parallel = ParallelConfig(
        selfplay_workers=(
            options.workers if options.mode == "selfplay" else config.parallel.selfplay_workers
        ),
        evaluation_workers=(
            options.workers
            if options.mode == "evaluation"
            else config.parallel.evaluation_workers
        ),
        gpu_batch_size=options.gpu_batch_size,
        batch_wait_ms=options.batch_wait_ms,
        request_queue_size=options.request_queue_size,
    )
    hardware = inspect_hardware(checkpoint)
    effective_cpu_limit = max(1, math.floor(hardware.effective_cpus) - 4)
    if options.workers > effective_cpu_limit:
        raise ValueError(
            f"workers={options.workers} exceeds effective CPU safety limit "
            f"{effective_cpu_limit} ({hardware.effective_cpus:g} CPUs minus 4)"
        )
    _validate_parallel_search(config, parallel)
    options.output.mkdir(parents=True, exist_ok=False)

    writer = JsonlWriter(options.output / "events.jsonl")
    report_path = options.output / "report.json"
    setup_started = time.monotonic()
    measured_started: float | None = None
    sampler: ResourceSampler | None = None
    stop_event = threading.Event()
    completed_games: list[dict[str, object]] = []
    latest_inference = InferenceMetrics()
    success = False
    try:
        writer.write(
            "benchmark-start",
            options=_jsonable(asdict(options)),
            hardware=_jsonable(asdict(hardware)),
            search=config.search.to_dict(),
            ruleset_id=config.ruleset_id,
        )
        _model, evaluator, _metadata = load_inference_checkpoint(
            checkpoint,
            device=device,
        )
        _warm_up(evaluator, config, parallel, options.workers)
        candidate = evaluator
        best = None
        if options.mode == "evaluation":
            _best_model, best, _best_metadata = load_inference_checkpoint(
                checkpoint,
                device=device,
            )
            _warm_up(best, config, parallel, options.workers)

        sampler = ResourceSampler(os.getpid())
        sampler.start()
        measured_started = time.monotonic()
        base_seed = config.seed if options.seed is None else options.seed

        def on_selfplay(
            completed: CompletedSelfPlay,
            inference: InferenceMetrics,
        ) -> None:
            nonlocal latest_inference
            _validate_selfplay(completed, config)
            latest_inference = inference
            row = {
                "game_index": completed.game_index,
                "moves": len(completed.result.moves),
                "playouts": completed.result.playouts,
                "elapsed_seconds": completed.elapsed_seconds,
                "terminal": completed.result.terminal,
                "selection_seconds": completed.result.selection_seconds,
                "encoding_seconds": completed.result.encoding_seconds,
                "evaluator_seconds": completed.result.evaluator_seconds,
                "expansion_seconds": completed.result.expansion_seconds,
            }
            completed_games.append(row)
            writer.write(
                "game-complete",
                **row,
                inference=_jsonable(asdict(inference)),
                gpu_average_batch_size=inference.average_batch_size,
            )

        def on_evaluation(
            completed: CompletedEvaluation,
            inference: InferenceMetrics,
        ) -> None:
            nonlocal latest_inference
            latest_inference = inference
            result = completed.result
            row = {
                "game_index": result.game_index,
                "candidate_color": int(result.candidate_color),
                "moves": result.moves,
                "playouts": result.playouts,
                "elapsed_seconds": completed.elapsed_seconds,
                "terminal": result.adjudication.reason.value,
                "winner": (
                    None
                    if result.adjudication.winner is None
                    else int(result.adjudication.winner)
                ),
            }
            completed_games.append(row)
            writer.write(
                "game-complete",
                **row,
                inference=_jsonable(asdict(inference)),
                gpu_average_batch_size=inference.average_batch_size,
            )

        if options.mode == "selfplay":
            run_parallel_selfplay(
                candidate,
                config.search,
                game_indices=range(options.games),
                configured_workers=options.workers,
                parallel_config=parallel,
                seed_for_game=lambda index: base_seed + index,
                initial_sfen_for_game=config.start_positions.choose,
                model_version=f"benchmark:{hardware.checkpoint_sha256[:12]}",
                rules=config.adjudication,
                ruleset_id=config.ruleset_id,
                temperature_moves=config.temperature_moves,
                stop_event=stop_event,
                on_complete=on_selfplay,
            )
        else:
            if best is None:
                raise RuntimeError("Evaluation benchmark did not load the best model")
            sfens = config.start_positions.initial_sfens()
            run_parallel_evaluation(
                candidate,
                best,
                config.search,
                game_indices=range(options.games),
                configured_workers=options.workers,
                parallel_config=parallel,
                seed_for_game=lambda index: base_seed + index,
                initial_sfen_for_game=lambda index: sfens[
                    (index // 2) % len(sfens)
                ],
                rules=config.adjudication,
                stop_event=stop_event,
                on_complete=on_evaluation,
            )
        success = True
    except BaseException as exc:
        writer.write(
            "benchmark-error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        finished = time.monotonic()
        elapsed = finished - (
            measured_started if measured_started is not None else setup_started
        )
        resources = (
            sampler.stop()
            if sampler is not None
            else summarize_resources(())
        )
        report = build_report(
            options,
            hardware,
            config,
            completed_games,
            latest_inference,
            resources,
            elapsed,
            success,
        )
        _atomic_write_json(report_path, report)
        if sampler is not None:
            for sample in sampler.samples:
                writer.write("resource-sample", **sample)
        writer.write("benchmark-complete", report=report)
        writer.close()
    return report


def build_report(
    options: BenchmarkOptions,
    hardware: HardwareSnapshot,
    config: TrainingConfig,
    games: Sequence[dict[str, object]],
    inference: InferenceMetrics,
    resources: ResourceSummary,
    elapsed_seconds: float,
    success: bool,
) -> dict[str, object]:
    total_playouts = sum(int(game["playouts"]) for game in games)
    total_moves = sum(int(game["moves"]) for game in games)
    return {
        "schema_version": 1,
        "success": success,
        "mode": options.mode,
        "parameters": {
            "games": options.games,
            "workers": options.workers,
            "gpu_batch_size": options.gpu_batch_size,
            "batch_wait_ms": options.batch_wait_ms,
            "request_queue_size": options.request_queue_size,
            "n_playouts": config.search.n_playouts,
            "search_batch_size": config.search.batch_size,
            "seed": config.seed if options.seed is None else options.seed,
        },
        "hardware": _jsonable(asdict(hardware)),
        "checkpoint": str(options.checkpoint.resolve()),
        "config": str(options.config.resolve()),
        "ruleset_id": config.ruleset_id,
        "completed_games": len(games),
        "total_moves": total_moves,
        "total_playouts": total_playouts,
        "wall_seconds": elapsed_seconds,
        "games_per_hour": len(games) / max(elapsed_seconds, 1e-9) * 3600,
        "playouts_per_second": total_playouts / max(elapsed_seconds, 1e-9),
        "inference": {
            **asdict(inference),
            "average_batch_size": inference.average_batch_size,
            "batches_per_second": inference.batches / max(elapsed_seconds, 1e-9),
            "leaves_per_second": inference.leaves / max(elapsed_seconds, 1e-9),
        },
        "resources": _jsonable(asdict(resources)),
        "games": list(games),
    }


def inspect_hardware(checkpoint: Path) -> HardwareSnapshot:
    logical = os.cpu_count() or 1
    affinity = _affinity_cpu_count()
    quota = _cgroup_quota_cpus()
    cpuset = _cgroup_cpuset_cpu_count()
    limits = [float(logical)]
    limits.extend(
        float(value)
        for value in (affinity, quota, cpuset)
        if value is not None and value > 0
    )
    return HardwareSnapshot(
        logical_cpus=logical,
        affinity_cpus=affinity,
        cgroup_quota_cpus=quota,
        cgroup_cpuset_cpus=cpuset,
        effective_cpus=min(limits),
        gpu_name=_gpu_name(),
        git_commit=_git_commit(),
        checkpoint_sha256=_sha256(checkpoint),
    )


def summarize_resources(
    samples: Iterable[dict[str, float]],
) -> ResourceSummary:
    rows = tuple(samples)

    def mean(name: str) -> float | None:
        values = [row[name] for row in rows if name in row]
        return None if not values else statistics.fmean(values)

    def p95(name: str) -> float | None:
        values = sorted(row[name] for row in rows if name in row)
        if not values:
            return None
        return values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)]

    def maximum(name: str) -> float | None:
        values = [row[name] for row in rows if name in row]
        return None if not values else max(values)

    return ResourceSummary(
        samples=len(rows),
        cpu_system_mean_percent=mean("cpu_system_percent"),
        cpu_system_p95_percent=p95("cpu_system_percent"),
        main_process_mean_percent=mean("main_process_percent"),
        main_process_p95_percent=p95("main_process_percent"),
        worker_processes_mean_percent=mean("worker_processes_percent"),
        worker_processes_p95_percent=p95("worker_processes_percent"),
        process_tree_memory_peak_mb=maximum("process_tree_memory_mb"),
        gpu_utilization_mean_percent=mean("gpu_utilization_percent"),
        gpu_utilization_p95_percent=p95("gpu_utilization_percent"),
        gpu_power_mean_watts=mean("gpu_power_watts"),
        gpu_memory_peak_mb=maximum("gpu_memory_mb"),
        gpu_memory_total_mb=maximum("gpu_memory_total_mb"),
    )


def _validate_options(options: BenchmarkOptions) -> None:
    if options.games <= 0:
        raise ValueError("games must be positive")
    if options.workers <= 0:
        raise ValueError("workers must be positive; auto mode is not allowed")
    if options.workers > options.games:
        raise ValueError("workers cannot exceed games")
    if options.gpu_batch_size <= 0:
        raise ValueError("gpu-batch-size must be positive")
    if options.batch_wait_ms < 0:
        raise ValueError("batch-wait-ms cannot be negative")
    if options.request_queue_size <= 0:
        raise ValueError("request-queue-size must be positive")


def _validate_output_dir(output: Path, config: TrainingConfig) -> None:
    target = output.resolve()
    protected = config.artifact_root.resolve()
    if target == protected or protected in target.parents:
        raise ValueError(
            f"Benchmark output must not be inside formal artifact root {protected}"
        )
    if target.exists():
        raise FileExistsError(f"Benchmark output already exists: {target}")


def _validate_parallel_search(
    config: TrainingConfig,
    parallel: ParallelConfig,
) -> None:
    if parallel.gpu_batch_size < config.search.batch_size:
        raise ValueError(
            "gpu-batch-size must be at least the tree search batch size "
            f"({config.search.batch_size})"
        )


def _validate_selfplay(
    completed: CompletedSelfPlay,
    config: TrainingConfig,
) -> None:
    result = completed.result
    if len(result.samples) != len(result.moves):
        raise RuntimeError("Self-play produced a different number of samples and moves")
    initial_sfen = config.start_positions.choose(completed.game_index)
    position = TrainingPosition(config.adjudication, initial_sfen)
    for index, (move, sample) in enumerate(
        zip(result.moves, result.samples, strict=True)
    ):
        legal = {action.usi for action in position.legal_actions()}
        if move not in legal:
            raise RuntimeError(f"Illegal benchmark move at ply {index + 1}: {move}")
        probabilities = np.asarray(
            sample.policy_probabilities,
            dtype=np.float64,
        )
        if probabilities.size == 0 or not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"Invalid policy at ply {index + 1}")
        if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-3):
            raise RuntimeError(f"Unnormalized policy at ply {index + 1}")
        if not math.isfinite(float(sample.value)):
            raise RuntimeError(f"Invalid value target at ply {index + 1}")
        position.push_usi(move)


def _warm_up(
    evaluator: Any,
    config: TrainingConfig,
    parallel: ParallelConfig,
    workers: int,
) -> None:
    encoded = encode_position(SearchPosition(config.start_positions.choose(0)))
    size = min(
        parallel.gpu_batch_size,
        max(config.search.batch_size, workers * config.search.batch_size),
    )
    features1 = np.repeat(encoded.features1[None], size, axis=0)
    features2 = np.repeat(encoded.features2[None], size, axis=0)
    logits, values = evaluator.evaluate_batch(features1, features2)
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(values)):
        raise RuntimeError("Checkpoint warm-up produced NaN or infinite output")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_name() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _affinity_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cgroup_quota_cpus() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota, period = path.read_text(encoding="ascii").split()[:2]
        if quota == "max":
            return None
        return int(quota) / int(period)
    except (OSError, ValueError, IndexError):
        return None


def _cgroup_cpuset_cpu_count() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        Path("/sys/fs/cgroup/cpuset/cpuset.cpus"),
    ):
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if value:
            return _parse_cpu_set(value)
    return None


def _parse_cpu_set(value: str) -> int:
    cpus: set[int] = set()
    for section in value.split(","):
        bounds = section.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        cpus.update(range(start, end + 1))
    return len(cpus)


def _descendant_pids(root_pid: int) -> set[int]:
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = {
                int(value)
                for value in path.read_text(encoding="ascii").split()
            }
        except (OSError, ValueError):
            continue
        new_children = children - descendants
        descendants.update(new_children)
        pending.extend(new_children)
    descendants.discard(root_pid)
    return descendants


def _process_ticks(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
        return int(fields[13]) + int(fields[14])
    except (OSError, ValueError, IndexError):
        return 0


def _process_rss_bytes(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        options = parse_args(argv)
        report = run_benchmark(options)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"throughput benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
