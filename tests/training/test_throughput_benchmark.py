import json
from pathlib import Path

import pytest

from alphasho.ai import ShogiPolicyValueNet, save_inference_checkpoint
from alphasho.training import load_training_config
from alphasho.training.parallel import InferenceMetrics
from alphasho.training.throughput_benchmark import (
    BenchmarkOptions,
    HardwareSnapshot,
    ResourceSummary,
    _parse_cpu_set,
    _validate_output_dir,
    build_report,
    parse_args,
    run_benchmark,
    summarize_resources,
)


def _options(tmp_path: Path, **overrides) -> BenchmarkOptions:
    values = {
        "config": Path("configs/training/smoke.toml"),
        "checkpoint": tmp_path / "best.pt",
        "mode": "selfplay",
        "games": 1,
        "workers": 1,
        "gpu_batch_size": 4,
        "batch_wait_ms": 0.0,
        "request_queue_size": 8,
        "output": tmp_path / "benchmark",
        "device": "cpu",
        "seed": 123,
    }
    values.update(overrides)
    return BenchmarkOptions(**values)


def _hardware() -> HardwareSnapshot:
    return HardwareSnapshot(
        logical_cpus=32,
        affinity_cpus=32,
        cgroup_quota_cpus=None,
        cgroup_cpuset_cpus=32,
        effective_cpus=32.0,
        gpu_name="test",
        git_commit="abc",
        checkpoint_sha256="1" * 64,
    )


def test_parse_args_requires_explicit_safe_values(tmp_path: Path) -> None:
    options = parse_args(
        [
            "--config",
            "configs/training/baseline.toml",
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--mode",
            "selfplay",
            "--games",
            "32",
            "--workers",
            "12",
            "--gpu-batch-size",
            "128",
            "--batch-wait-ms",
            "2",
            "--output",
            str(tmp_path / "result"),
        ]
    )

    assert options.workers == 12
    assert options.request_queue_size == 64
    with pytest.raises(ValueError, match="workers cannot exceed games"):
        parse_args(
            [
                "--config",
                "configs/training/smoke.toml",
                "--checkpoint",
                str(tmp_path / "best.pt"),
                "--mode",
                "selfplay",
                "--games",
                "1",
                "--workers",
                "2",
                "--gpu-batch-size",
                "4",
                "--batch-wait-ms",
                "0",
                "--output",
                str(tmp_path / "invalid"),
            ]
        )


def test_formal_artifact_root_is_rejected() -> None:
    config = load_training_config("configs/training/baseline.toml")

    with pytest.raises(ValueError, match="formal artifact root"):
        _validate_output_dir(
            config.artifact_root / "benchmarks" / "forbidden",
            config,
        )


def test_report_aggregates_playouts_and_resources(tmp_path: Path) -> None:
    options = _options(tmp_path)
    resources = summarize_resources(
        [
            {
                "cpu_system_percent": 50.0,
                "main_process_percent": 100.0,
                "worker_processes_percent": 400.0,
                "process_tree_memory_mb": 1000.0,
                "gpu_utilization_percent": 30.0,
                "gpu_power_watts": 150.0,
                "gpu_memory_mb": 1024.0,
                "gpu_memory_total_mb": 24576.0,
            },
            {
                "cpu_system_percent": 70.0,
                "main_process_percent": 120.0,
                "worker_processes_percent": 500.0,
                "process_tree_memory_mb": 1100.0,
                "gpu_utilization_percent": 50.0,
                "gpu_power_watts": 180.0,
                "gpu_memory_mb": 1200.0,
                "gpu_memory_total_mb": 24576.0,
            },
        ]
    )
    report = build_report(
        options,
        _hardware(),
        load_training_config(options.config),
        [
            {"moves": 10, "playouts": 4000},
            {"moves": 20, "playouts": 8000},
        ],
        InferenceMetrics(requests=10, batches=5, leaves=100),
        resources,
        10.0,
        True,
    )

    assert report["total_playouts"] == 12000
    assert report["playouts_per_second"] == 1200.0
    assert report["inference"]["average_batch_size"] == 20.0
    assert report["resources"]["gpu_utilization_mean_percent"] == 40.0
    assert report["resources"]["process_tree_memory_peak_mb"] == 1100.0
    assert _parse_cpu_set("0-3,8,10-11") == 7


def test_smoke_benchmark_does_not_write_training_artifacts(tmp_path: Path) -> None:
    config = load_training_config("configs/training/smoke.toml")
    checkpoint = tmp_path / "best.pt"
    save_inference_checkpoint(
        checkpoint,
        ShogiPolicyValueNet(config.model),
        search_config=config.search.to_dict(),
    )
    protected_state = config.artifact_root / "checkpoints" / "pipeline_state.json"
    before = (
        protected_state.read_bytes()
        if protected_state.exists()
        else None
    )

    report = run_benchmark(_options(tmp_path))

    after = (
        protected_state.read_bytes()
        if protected_state.exists()
        else None
    )
    assert report["success"] is True
    assert report["completed_games"] == 1
    assert report["total_playouts"] > 0
    assert before == after
    assert (tmp_path / "benchmark" / "events.jsonl").exists()
    saved = json.loads(
        (tmp_path / "benchmark" / "report.json").read_text(encoding="utf-8")
    )
    assert saved["checkpoint"] == str(checkpoint.resolve())


def test_empty_resource_summary_is_explicit() -> None:
    summary = summarize_resources(())

    assert summary == ResourceSummary(
        samples=0,
        cpu_system_mean_percent=None,
        cpu_system_p95_percent=None,
        main_process_mean_percent=None,
        main_process_p95_percent=None,
        worker_processes_mean_percent=None,
        worker_processes_p95_percent=None,
        process_tree_memory_peak_mb=None,
        gpu_utilization_mean_percent=None,
        gpu_utilization_p95_percent=None,
        gpu_power_mean_watts=None,
        gpu_memory_peak_mb=None,
        gpu_memory_total_mb=None,
    )
