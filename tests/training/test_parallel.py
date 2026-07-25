import threading

import numpy as np
import torch

from alphasho.ai import SearchConfig
from alphasho.training import load_training_config
from alphasho.training.config import ParallelConfig
from alphasho.training.orchestrator import PipelineState
from alphasho.training.parallel import (
    BatchedInferenceServer,
    RemoteBatchEvaluator,
    resolve_worker_count,
    run_parallel_selfplay,
)


class _ConstantEvaluator:
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate_batch(self, features1, features2):  # noqa: ANN001
        size = len(features1)
        return (
            np.full((size, 2187), self.value, dtype=np.float32),
            np.full(size, self.value, dtype=np.float32),
        )


def test_inference_server_combines_requests_and_routes_models() -> None:
    config = ParallelConfig(
        selfplay_workers=2,
        evaluation_workers=2,
        gpu_batch_size=16,
        batch_wait_ms=50,
        request_queue_size=8,
    )
    server = BatchedInferenceServer(
        {
            "candidate": _ConstantEvaluator(1.0),
            "best": _ConstantEvaluator(-1.0),
        },
        worker_count=2,
        config=config,
    )
    previous_threads = torch.get_num_threads()
    server.start()
    assert torch.get_num_threads() == 1
    outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(worker_id: int, model_id: str, size: int) -> None:
        remote = RemoteBatchEvaluator(
            worker_id,
            model_id,
            server.request_queue,
            server.response_queues[worker_id],
            server.cancel_event,
        )
        outputs[worker_id] = remote.evaluate_batch(
            np.zeros((size, 62, 9, 9), dtype=np.uint8),
            np.zeros((size, 57, 9, 9), dtype=np.uint8),
        )

    threads = [
        threading.Thread(target=evaluate, args=(0, "candidate", 3)),
        threading.Thread(target=evaluate, args=(1, "best", 5)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    server.stop()
    assert torch.get_num_threads() == previous_threads

    assert np.all(outputs[0][1] == 1.0)
    assert np.all(outputs[1][1] == -1.0)
    metrics = server.metrics()
    assert metrics.requests == 2
    assert metrics.leaves == 8


def test_pipeline_state_v2_migrates_out_of_order_progress() -> None:
    state = PipelineState.from_dict(
        {
            "schema_version": 2,
            "selfplay_completed": 3,
            "evaluation_completed": 4,
            "candidate_wins": 2,
            "best_wins": 1,
            "draws": 1,
        }
    )

    assert state.schema_version == 4
    assert state.selfplay_completed_games == [0, 1, 2]
    assert [item["game_index"] for item in state.evaluation_results] == [
        0,
        1,
        2,
        3,
    ]


def test_auto_worker_count_reserves_two_cores(monkeypatch) -> None:
    monkeypatch.setattr("alphasho.training.parallel.os.cpu_count", lambda: 12)
    assert resolve_worker_count(0, 32) == 8
    assert resolve_worker_count(3, 2) == 2


def test_two_process_selfplay_uses_central_evaluator() -> None:
    profile = load_training_config("configs/training/smoke.toml")
    completed = []
    config = ParallelConfig(
        selfplay_workers=2,
        evaluation_workers=1,
        gpu_batch_size=4,
        batch_wait_ms=2,
        request_queue_size=8,
    )

    run_parallel_selfplay(
        _ConstantEvaluator(0.0),
        SearchConfig(n_playouts=2, batch_size=1),
        game_indices=(0, 1),
        configured_workers=2,
        parallel_config=config,
        seed_for_game=lambda index: 100 + index,
        initial_sfen_for_game=profile.start_positions.choose,
        model_version="test",
        rules=profile.adjudication,
        ruleset_id=profile.ruleset_id,
        temperature_moves=1,
        stop_event=threading.Event(),
        on_complete=lambda result, metrics: completed.append(
            (result, metrics)
        ),
    )

    assert {item.game_index for item, _metrics in completed} == {0, 1}
    assert all(item.result.samples for item, _metrics in completed)
    assert completed[-1][1].requests > 0
