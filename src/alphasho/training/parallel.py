"""Multi-process game search with one dynamically batched inference owner."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alphasho.ai import SearchCancelled, SearchConfig
from alphasho.ai.mcts import BatchEvaluator
from alphasho.domain import Color
from alphasho.training.adjudication import AdjudicationRules
from alphasho.training.config import ParallelConfig
from alphasho.training.evaluation import (
    EvaluationGameResult,
    play_evaluation_game,
)
from alphasho.training.selfplay import SelfPlayResult, play_self_game


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    requests: int = 0
    batches: int = 0
    leaves: int = 0
    queue_seconds: float = 0.0
    inference_seconds: float = 0.0

    @property
    def average_batch_size(self) -> float:
        return self.leaves / max(self.batches, 1)


@dataclass(frozen=True, slots=True)
class CompletedSelfPlay:
    game_index: int
    result: SelfPlayResult
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CompletedEvaluation:
    result: EvaluationGameResult
    elapsed_seconds: float


@dataclass(slots=True)
class _InferenceRequest:
    worker_id: int
    request_id: int
    model_id: str
    features1: NDArray[np.uint8]
    features2: NDArray[np.uint8]
    submitted: float


@dataclass(slots=True)
class _InferenceResponse:
    request_id: int
    logits: NDArray[np.float32] | None = None
    values: NDArray[np.float32] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SelfPlayTask:
    game_index: int
    seed: int
    initial_sfen: str | None


@dataclass(frozen=True, slots=True)
class _EvaluationTask:
    game_index: int
    seed: int
    initial_sfen: str | None


@dataclass(slots=True)
class _WorkerResult:
    worker_id: int
    game_index: int
    payload: object | None
    elapsed_seconds: float
    error: str | None = None


class RemoteBatchEvaluator:
    """Evaluator proxy used inside a worker process."""

    def __init__(
        self,
        worker_id: int,
        model_id: str,
        request_queue: Any,
        response_queue: Any,
        cancel_event: Any,
    ) -> None:
        self._worker_id = worker_id
        self._model_id = model_id
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._cancel_event = cancel_event
        self._request_id = 0

    def evaluate_batch(
        self,
        features1: NDArray[np.uint8],
        features2: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self._cancel_event.is_set():
            raise SearchCancelled("Parallel inference was cancelled")
        request_id = self._request_id
        self._request_id += 1
        request = _InferenceRequest(
            self._worker_id,
            request_id,
            self._model_id,
            features1,
            features2,
            time.monotonic(),
        )
        while True:
            try:
                self._request_queue.put(request, timeout=0.2)
                break
            except queue.Full:
                if self._cancel_event.is_set():
                    raise SearchCancelled(
                        "Parallel inference was cancelled"
                    ) from None
        while True:
            try:
                response: _InferenceResponse = self._response_queue.get(timeout=0.2)
                break
            except queue.Empty:
                if self._cancel_event.is_set():
                    raise SearchCancelled(
                        "Parallel inference was cancelled"
                    ) from None
        if response.request_id != request_id:
            raise RuntimeError(
                "Inference response routing failed: "
                f"expected {request_id}, got {response.request_id}"
            )
        if response.error is not None:
            raise RuntimeError(response.error)
        if response.logits is None or response.values is None:
            raise RuntimeError("Inference server returned an empty response")
        return response.logits, response.values


class BatchedInferenceServer:
    """Owns evaluators in the parent process and dynamically combines requests."""

    def __init__(
        self,
        evaluators: Mapping[str, BatchEvaluator],
        *,
        worker_count: int,
        config: ParallelConfig,
        context: Any | None = None,
    ) -> None:
        self.context = context or mp.get_context("spawn")
        self.request_queue = self.context.Queue(config.request_queue_size)
        self.response_queues = [
            self.context.Queue(max(2, config.request_queue_size // worker_count))
            for _ in range(worker_count)
        ]
        self.cancel_event = self.context.Event()
        self._evaluators = dict(evaluators)
        self._batch_size = config.gpu_batch_size
        self._wait_seconds = config.batch_wait_ms / 1000.0
        self._pending: list[_InferenceRequest] = []
        self._metrics = InferenceMetrics()
        self._metrics_lock = threading.Lock()
        self._closing = False
        self._thread = threading.Thread(
            target=self._run,
            name="alphasho-inference-server",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._closing = True
        try:
            self.request_queue.put(None, timeout=0.2)
        except queue.Full:
            pass
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            self.cancel_event.set()
            raise RuntimeError("Inference server did not stop")

    def metrics(self) -> InferenceMetrics:
        with self._metrics_lock:
            return self._metrics

    def _run(self) -> None:
        while not self._closing or self._pending:
            first = self._next_request(timeout=0.05)
            if first is None:
                continue
            requests = [first]
            leaves = len(first.features1)
            deadline = time.monotonic() + self._wait_seconds
            while leaves < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                candidate = self._next_request(timeout=remaining)
                if candidate is None:
                    break
                if (
                    candidate.model_id != first.model_id
                    or leaves + len(candidate.features1) > self._batch_size
                ):
                    self._pending.append(candidate)
                    continue
                requests.append(candidate)
                leaves += len(candidate.features1)
            self._evaluate(first.model_id, requests)

    def _next_request(self, *, timeout: float) -> _InferenceRequest | None:
        if self._pending:
            return self._pending.pop(0)
        try:
            item = self.request_queue.get(timeout=max(timeout, 1e-6))
        except queue.Empty:
            return None
        if item is None:
            self._closing = True
            return None
        return item

    def _evaluate(
        self,
        model_id: str,
        requests: list[_InferenceRequest],
    ) -> None:
        started = time.monotonic()
        total = sum(len(item.features1) for item in requests)
        try:
            evaluator = self._evaluators[model_id]
            features1 = np.concatenate([item.features1 for item in requests])
            features2 = np.concatenate([item.features2 for item in requests])
            logits, values = evaluator.evaluate_batch(features1, features2)
            offset = 0
            for request in requests:
                size = len(request.features1)
                self.response_queues[request.worker_id].put(
                    _InferenceResponse(
                        request.request_id,
                        logits[offset : offset + size],
                        values[offset : offset + size],
                    )
                )
                offset += size
        except BaseException as exc:
            detail = f"Central inference failed: {type(exc).__name__}: {exc}"
            self.cancel_event.set()
            for request in requests:
                self.response_queues[request.worker_id].put(
                    _InferenceResponse(request.request_id, error=detail)
                )
        elapsed = time.monotonic() - started
        with self._metrics_lock:
            old = self._metrics
            self._metrics = InferenceMetrics(
                requests=old.requests + len(requests),
                batches=old.batches + 1,
                leaves=old.leaves + total,
                queue_seconds=old.queue_seconds
                + sum(max(0.0, started - item.submitted) for item in requests),
                inference_seconds=old.inference_seconds + elapsed,
            )


def resolve_worker_count(configured: int, games: int) -> int:
    if games <= 0:
        return 0
    if configured > 0:
        return min(configured, games)
    cpus = os.cpu_count() or 2
    return min(8, games, max(1, cpus - 2))


def run_parallel_selfplay(
    evaluator: BatchEvaluator,
    search_config: SearchConfig,
    *,
    game_indices: Iterable[int],
    configured_workers: int,
    parallel_config: ParallelConfig,
    seed_for_game: Callable[[int], int],
    initial_sfen_for_game: Callable[[int], str | None],
    model_version: str,
    rules: AdjudicationRules,
    ruleset_id: str,
    temperature_moves: int,
    stop_event: threading.Event,
    on_complete: Callable[[CompletedSelfPlay, InferenceMetrics], None],
) -> None:
    indices = tuple(game_indices)
    workers = resolve_worker_count(configured_workers, len(indices))
    if workers <= 1:
        for game_index in indices:
            if stop_event.is_set():
                raise SearchCancelled("Self-play was stopped")
            started = time.monotonic()
            result = play_self_game(
                evaluator,
                search_config,
                seed=seed_for_game(game_index),
                model_version=model_version,
                rules=rules,
                initial_sfen=initial_sfen_for_game(game_index),
                ruleset_id=ruleset_id,
                temperature_moves=temperature_moves,
                stop_event=stop_event,
            )
            on_complete(
                CompletedSelfPlay(
                    game_index,
                    result,
                    time.monotonic() - started,
                ),
                InferenceMetrics(),
            )
        return

    context = mp.get_context("spawn")
    server = BatchedInferenceServer(
        {"selfplay": evaluator},
        worker_count=workers,
        config=parallel_config,
        context=context,
    )
    tasks = context.Queue(len(indices) + workers)
    results = context.Queue(workers * 2)
    for game_index in indices:
        tasks.put(
            _SelfPlayTask(
                game_index,
                seed_for_game(game_index),
                initial_sfen_for_game(game_index),
            )
        )
    for _ in range(workers):
        tasks.put(None)
    processes = [
        context.Process(
            target=_selfplay_worker,
            args=(
                worker_id,
                tasks,
                results,
                server.request_queue,
                server.response_queues[worker_id],
                server.cancel_event,
                search_config,
                model_version,
                rules,
                ruleset_id,
                temperature_moves,
            ),
            name=f"alphasho-selfplay-{worker_id}",
        )
        for worker_id in range(workers)
    ]
    _run_processes(
        processes,
        results,
        server,
        expected=len(indices),
        stop_event=stop_event,
        on_result=lambda item: on_complete(
            CompletedSelfPlay(
                item.game_index,
                _expect_type(item.payload, SelfPlayResult),
                item.elapsed_seconds,
            ),
            server.metrics(),
        ),
    )


def run_parallel_evaluation(
    candidate: BatchEvaluator,
    best: BatchEvaluator,
    search_config: SearchConfig,
    *,
    game_indices: Iterable[int],
    configured_workers: int,
    parallel_config: ParallelConfig,
    seed_for_game: Callable[[int], int],
    initial_sfen_for_game: Callable[[int], str | None],
    rules: AdjudicationRules,
    stop_event: threading.Event,
    on_complete: Callable[[CompletedEvaluation, InferenceMetrics], None],
) -> None:
    indices = tuple(game_indices)
    workers = resolve_worker_count(configured_workers, len(indices))
    if workers <= 1:
        for game_index in indices:
            if stop_event.is_set():
                raise SearchCancelled("Evaluation was stopped")
            started = time.monotonic()
            candidate_color = (
                Color.BLACK if game_index % 2 == 0 else Color.WHITE
            )
            result = play_evaluation_game(
                candidate,
                best,
                candidate_color,
                search_config,
                rules=rules,
                initial_sfen=initial_sfen_for_game(game_index),
                seed=seed_for_game(game_index),
                progress_callback=None,
                stop_event=stop_event,
                heartbeat_seconds=10.0,
            )
            on_complete(
                CompletedEvaluation(
                    EvaluationGameResult(game_index, candidate_color, result),
                    time.monotonic() - started,
                ),
                InferenceMetrics(),
            )
        return

    context = mp.get_context("spawn")
    server = BatchedInferenceServer(
        {"candidate": candidate, "best": best},
        worker_count=workers,
        config=parallel_config,
        context=context,
    )
    tasks = context.Queue(len(indices) + workers)
    results = context.Queue(workers * 2)
    for game_index in indices:
        tasks.put(
            _EvaluationTask(
                game_index,
                seed_for_game(game_index),
                initial_sfen_for_game(game_index),
            )
        )
    for _ in range(workers):
        tasks.put(None)
    processes = [
        context.Process(
            target=_evaluation_worker,
            args=(
                worker_id,
                tasks,
                results,
                server.request_queue,
                server.response_queues[worker_id],
                server.cancel_event,
                search_config,
                rules,
            ),
            name=f"alphasho-evaluation-{worker_id}",
        )
        for worker_id in range(workers)
    ]
    _run_processes(
        processes,
        results,
        server,
        expected=len(indices),
        stop_event=stop_event,
        on_result=lambda item: on_complete(
            CompletedEvaluation(
                _expect_type(item.payload, EvaluationGameResult),
                item.elapsed_seconds,
            ),
            server.metrics(),
        ),
    )


def _run_processes(
    processes: list[Any],
    results: Any,
    server: BatchedInferenceServer,
    *,
    expected: int,
    stop_event: threading.Event,
    on_result: Callable[[_WorkerResult], None],
) -> None:
    server.start()
    for process in processes:
        process.start()
    completed = 0
    try:
        while completed < expected:
            if stop_event.is_set():
                server.cancel_event.set()
                raise SearchCancelled("Parallel game generation was stopped")
            try:
                item: _WorkerResult = results.get(timeout=0.2)
            except queue.Empty:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in {None, 0}
                ]
                if failed:
                    raise RuntimeError(
                        f"{failed[0].name} exited with code {failed[0].exitcode}"
                    ) from None
                continue
            if item.error is not None:
                server.cancel_event.set()
                raise RuntimeError(item.error)
            on_result(item)
            completed += 1
    finally:
        if completed < expected:
            server.cancel_event.set()
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        server.stop()


def _selfplay_worker(
    worker_id: int,
    tasks: Any,
    results: Any,
    request_queue: Any,
    response_queue: Any,
    cancel_event: Any,
    search_config: SearchConfig,
    model_version: str,
    rules: AdjudicationRules,
    ruleset_id: str,
    temperature_moves: int,
) -> None:
    _limit_worker_threads()
    evaluator = RemoteBatchEvaluator(
        worker_id,
        "selfplay",
        request_queue,
        response_queue,
        cancel_event,
    )
    while not cancel_event.is_set():
        task: _SelfPlayTask | None = tasks.get()
        if task is None:
            return
        started = time.monotonic()
        try:
            result = play_self_game(
                evaluator,
                search_config,
                seed=task.seed,
                model_version=model_version,
                rules=rules,
                initial_sfen=task.initial_sfen,
                ruleset_id=ruleset_id,
                temperature_moves=temperature_moves,
                stop_event=cancel_event,
            )
            results.put(
                _WorkerResult(
                    worker_id,
                    task.game_index,
                    result,
                    time.monotonic() - started,
                )
            )
        except BaseException:
            results.put(
                _WorkerResult(
                    worker_id,
                    task.game_index,
                    None,
                    time.monotonic() - started,
                    traceback.format_exc(),
                )
            )
            cancel_event.set()
            return


def _evaluation_worker(
    worker_id: int,
    tasks: Any,
    results: Any,
    request_queue: Any,
    response_queue: Any,
    cancel_event: Any,
    search_config: SearchConfig,
    rules: AdjudicationRules,
) -> None:
    _limit_worker_threads()
    candidate = RemoteBatchEvaluator(
        worker_id,
        "candidate",
        request_queue,
        response_queue,
        cancel_event,
    )
    best = RemoteBatchEvaluator(
        worker_id,
        "best",
        request_queue,
        response_queue,
        cancel_event,
    )
    while not cancel_event.is_set():
        task: _EvaluationTask | None = tasks.get()
        if task is None:
            return
        started = time.monotonic()
        candidate_color = (
            Color.BLACK if task.game_index % 2 == 0 else Color.WHITE
        )
        try:
            adjudication = play_evaluation_game(
                candidate,
                best,
                candidate_color,
                search_config,
                rules=rules,
                initial_sfen=task.initial_sfen,
                seed=task.seed,
                progress_callback=None,
                stop_event=cancel_event,
                heartbeat_seconds=10.0,
            )
            results.put(
                _WorkerResult(
                    worker_id,
                    task.game_index,
                    EvaluationGameResult(
                        task.game_index,
                        candidate_color,
                        adjudication,
                    ),
                    time.monotonic() - started,
                )
            )
        except BaseException:
            results.put(
                _WorkerResult(
                    worker_id,
                    task.game_index,
                    None,
                    time.monotonic() - started,
                    traceback.format_exc(),
                )
            )
            cancel_event.set()
            return


def _limit_worker_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except (ImportError, RuntimeError):
        pass


def _expect_type(value: object | None, expected: type[Any]) -> Any:
    if not isinstance(value, expected):
        raise RuntimeError(
            f"Worker returned {type(value).__name__}, expected {expected.__name__}"
        )
    return value
