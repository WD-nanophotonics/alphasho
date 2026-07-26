from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

import alphasho.runpod_control as runpod_control
from alphasho.runpod_control import (
    STATUS_MARKER,
    ControlError,
    GitStatus,
    PodStatus,
    RunpodController,
    revisions_match,
    sync_summary,
)
from alphasho.runpod_sync import ControlConfig, SyncConfig, load_sync_config


def _config(tmp_path: Path, *, pause_timeout: int = 120) -> SyncConfig:
    project = tmp_path / "project"
    return SyncConfig(
        project_root=project,
        source_path=project / "configs" / "runpod_sync.toml",
        ssh_target="pod@ssh.runpod.io",
        identity_file=tmp_path / "id_ed25519",
        ssh_executable=tmp_path / "ssh.exe",
        remote_project_root="/workspace/alphasho-v3",
        sync_root=project / "artifacts" / "cloud-sync",
        published_model=project / "artifacts" / "checkpoints" / "best.pt",
        runpodctl=tmp_path / "runpodctl.exe",
        interval_minutes=15,
        log_tail_bytes=1024,
        timeout_seconds=60,
        full_backup_timeout_seconds=600,
        retry_attempts=3,
        task_name="AlphaSho Runpod Sync",
        control=ControlConfig(pause_timeout_seconds=pause_timeout),
    )


def _remote_status(**changes: object) -> PodStatus:
    values = PodStatus(
        online=True,
        project_exists=True,
        venv_exists=True,
        config_exists=True,
        git_revision="abc123",
        git_branch="master",
        gpu="RTX 3090, 20, 1024, 24576",
        log_exists=True,
    ).to_dict()
    values.update(changes)
    return PodStatus(**values)


def _status_output(status: PodStatus) -> str:
    return STATUS_MARKER + json.dumps(status.to_dict()) + "\n"


def test_control_config_is_loaded_and_rejects_unsafe_paths(tmp_path: Path) -> None:
    source = tmp_path / "project" / "configs" / "runpod_sync.toml"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
[connection]
ssh_target = "pod@ssh.runpod.io"
identity_file = "id"
ssh_executable = "ssh.exe"
remote_project_root = "/workspace/alphasho-v3"
[local]
sync_root = "artifacts/cloud-sync"
published_model = "artifacts/checkpoints/best.pt"
runpodctl = "runpodctl.exe"
[control]
tmux_session = "train-session"
status_poll_seconds = 9
training_entry = "../outside.py"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="project-relative"):
        load_sync_config(source)

    content = source.read_text(encoding="utf-8").replace("../outside.py", "train.py")
    source.write_text(content, encoding="utf-8")
    config = load_sync_config(source)
    assert config.control.tmux_session == "train-session"
    assert config.control.status_poll_seconds == 9


def test_status_response_is_parsed_and_git_comparison_is_explicit(tmp_path: Path) -> None:
    expected = _remote_status(
        tmux_running=True,
        trainer_running=True,
        training_state="running",
        active_iteration=21,
        stage="selfplay",
        revision=511,
        selfplay_completed=7,
    )
    controller = RunpodController(
        _config(tmp_path), remote_runner=lambda _command, _timeout: _status_output(expected)
    )

    status = controller.check_status()

    assert status.training_state == "running"
    assert status.active_iteration == 21
    assert status.selfplay_completed == 7
    assert revisions_match(GitStatus(revision="abc123"), status) is True
    assert revisions_match(GitStatus(revision="different"), status) is False


def test_local_git_checks_use_silent_process_options(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class Result:
        returncode = 0
        stdout = "value\n"

    def fake_run(_arguments, **kwargs):
        calls.append(kwargs)
        return Result()

    monkeypatch.setattr(runpod_control.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runpod_control,
        "silent_subprocess_kwargs",
        lambda: {"creationflags": 0x08000000, "startupinfo": "hidden"},
    )

    runpod_control.local_git_status(tmp_path)

    assert len(calls) == 3
    assert all(call["creationflags"] == 0x08000000 for call in calls)
    assert all(call["startupinfo"] == "hidden" for call in calls)


def test_short_ssh_control_uses_silent_process_options(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.ssh_executable.write_bytes(b"ssh")
    config.identity_file.write_bytes(b"key")
    captured: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = "__ALPHASHO_RC__0\n"
        stderr = ""

    def fake_run(_arguments, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(runpod_control.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runpod_control,
        "silent_subprocess_kwargs",
        lambda: {"creationflags": 0x08000000, "startupinfo": "hidden"},
    )

    RunpodController(config)._run_remote("true")

    assert captured["creationflags"] == 0x08000000
    assert captured["startupinfo"] == "hidden"


def test_offline_and_malformed_status_are_safe_results(tmp_path: Path) -> None:
    def fail(_command: str, _timeout: float) -> str:
        raise ControlError("proxy unavailable")

    controller = RunpodController(_config(tmp_path), remote_runner=fail)
    assert controller.check_status() == PodStatus(online=False, error="proxy unavailable")

    malformed = RunpodController(
        _config(tmp_path), remote_runner=lambda _command, _timeout: "login banner"
    )
    status = malformed.check_status()
    assert not status.online
    assert "结构化数据" in status.error


def test_start_is_idempotent_and_refuses_out_of_session_process(tmp_path: Path) -> None:
    running = _remote_status(
        tmux_running=True, trainer_running=True, training_state="running"
    )
    controller = RunpodController(
        _config(tmp_path), remote_runner=lambda _command, _timeout: _status_output(running)
    )
    result = controller.start_training()
    assert result["reason"] == "already_running"

    conflict = _remote_status(trainer_running=True, training_state="conflict")
    conflicting = RunpodController(
        _config(tmp_path),
        remote_runner=lambda _command, _timeout: _status_output(conflict),
    )
    with pytest.raises(ControlError, match="拒绝重复启动"):
        conflicting.start_training()


def test_start_uses_tmux_then_verifies_process(tmp_path: Path) -> None:
    stopped = _remote_status(training_state="stopped")
    running = _remote_status(
        tmux_running=True, trainer_running=True, training_state="running"
    )
    responses = deque([_status_output(stopped), "ok", _status_output(running)])
    commands: list[str] = []

    def run(command: str, _timeout: float) -> str:
        commands.append(command)
        return responses.popleft()

    result = RunpodController(
        _config(tmp_path), remote_runner=run, sleeper=lambda _seconds: None
    ).start_training()

    assert result["started"] is True
    assert "tmux new-session -d" in commands[1]
    assert "ALPHASHO_TRAINING_PROFILE=baseline" in commands[1]


def test_safe_pause_waits_for_stopped_state(tmp_path: Path) -> None:
    running = _remote_status(
        tmux_running=True,
        trainer_running=True,
        training_state="running",
        revision=10,
    )
    stopped = _remote_status(training_state="stopped", revision=11)
    responses = deque([_status_output(running), "sent", _status_output(stopped)])
    commands: list[str] = []

    def run(command: str, _timeout: float) -> str:
        commands.append(command)
        return responses.popleft()

    result = RunpodController(
        _config(tmp_path), remote_runner=run, sleeper=lambda _seconds: None
    ).pause_training()

    assert result["paused"] is True
    assert result["revision_after"] == 11
    assert "tmux send-keys" in commands[1]


def test_sync_summary_reads_last_verified_snapshot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    latest = config.sync_root / "latest"
    latest.mkdir(parents=True)
    (config.sync_root / "sync_status.json").write_text(
        json.dumps({"last_success": "now", "ok": True, "cloud_commit": "abc"}),
        encoding="utf-8",
    )
    (latest / "manifest.json").write_text(
        json.dumps({"models": {"best": {"sha256": "model-hash"}}}),
        encoding="utf-8",
    )

    summary = sync_summary(config)

    assert summary["last_success"] == "now"
    assert summary["best_sha256"] == "model-hash"
