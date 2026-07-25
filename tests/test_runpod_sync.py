from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from alphasho.runpod_sync import (
    MODEL_ARCHIVE_PATHS,
    SyncConfig,
    SyncError,
    _safe_extract,
    _wait_for_connection_code,
    build_remote_command,
    create_remote_snapshot,
    install_snapshot,
    load_sync_config,
    verify_snapshot,
)


def _config(tmp_path: Path) -> SyncConfig:
    project = tmp_path / "project"
    return SyncConfig(
        project_root=project,
        source_path=project / "configs" / "runpod_sync.toml",
        ssh_target="pod-id@ssh.runpod.io",
        identity_file=tmp_path / "id_ed25519",
        ssh_executable=tmp_path / "ssh.exe",
        remote_project_root="/workspace/alphasho-v3",
        sync_root=project / "artifacts" / "cloud-sync",
        published_model=project / "artifacts" / "checkpoints" / "best.pt",
        runpodctl=tmp_path / "runpodctl.exe",
        interval_minutes=15,
        log_tail_bytes=64,
        timeout_seconds=60,
        full_backup_timeout_seconds=600,
        retry_attempts=3,
        task_name="AlphaSho Runpod Sync",
    )


def _cloud_project(tmp_path: Path) -> Path:
    project = tmp_path / "cloud"
    baseline = project / "artifacts" / "baseline"
    checkpoints = baseline / "checkpoints"
    logs = baseline / "logs"
    published = project / "artifacts" / "checkpoints"
    checkpoints.mkdir(parents=True)
    logs.mkdir(parents=True)
    published.mkdir(parents=True)
    (published / "best.pt").write_bytes(b"formal-best-v1")
    (checkpoints / "candidate.pt").write_bytes(b"candidate-v1")
    (checkpoints / "pipeline_state.json").write_text(
        json.dumps(
            {
                "active_iteration": 6,
                "stage": "selfplay",
                "revision": 464,
            }
        ),
        encoding="utf-8",
    )
    (baseline / "effective_config.json").write_text(
        json.dumps({"evaluation_games": 20, "promotion_score": 0.55}),
        encoding="utf-8",
    )
    (logs / "console.log").write_text("old\n" * 100 + "latest\n", encoding="utf-8")
    (logs / "baseline.jsonl").write_text('{"event":"latest"}\n', encoding="utf-8")
    return project


def test_load_config_expands_paths_and_project_relative_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYNC_TEST_HOME", str(tmp_path / "tools"))
    project = tmp_path / "project"
    source = project / "configs" / "runpod_sync.toml"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
[connection]
ssh_target = "pod@ssh.runpod.io"
identity_file = "%SYNC_TEST_HOME%/id"
ssh_executable = "%SYNC_TEST_HOME%/ssh.exe"
remote_project_root = "/workspace/alphasho-v3"
[local]
sync_root = "artifacts/cloud-sync"
published_model = "artifacts/checkpoints/best.pt"
runpodctl = "%SYNC_TEST_HOME%/runpodctl.exe"
[sync]
interval_minutes = 15
log_tail_bytes = 1024
timeout_seconds = 60
full_backup_timeout_seconds = 600
retry_attempts = 3
task_name = "AlphaSho Runpod Sync"
""",
        encoding="utf-8",
    )

    config = load_sync_config(source)

    assert config.project_root == project
    assert config.sync_root == project / "artifacts" / "cloud-sync"
    assert config.runpodctl == tmp_path / "tools" / "runpodctl.exe"


def test_snapshot_install_inherits_unchanged_models_and_publishes_only_best(
    tmp_path: Path,
) -> None:
    cloud = _cloud_project(tmp_path)
    config = _config(tmp_path)
    first_archive = tmp_path / "first.tar.gz"
    first_manifest = create_remote_snapshot(
        archive=first_archive,
        project_root=cloud,
        log_tail_bytes=64,
    )

    installed = install_snapshot(config, first_archive)

    latest = config.sync_root / "latest"
    assert installed["evaluation_games"] == 20
    assert (latest / MODEL_ARCHIVE_PATHS["best"]).read_bytes() == b"formal-best-v1"
    assert (latest / MODEL_ARCHIVE_PATHS["candidate"]).read_bytes() == b"candidate-v1"
    assert config.published_model.read_bytes() == b"formal-best-v1"
    assert b"candidate-v1" not in config.published_model.read_bytes()
    assert (latest / "logs" / "console.log").stat().st_size <= 64

    second_archive = tmp_path / "second.tar.gz"
    second_manifest = create_remote_snapshot(
        archive=second_archive,
        project_root=cloud,
        log_tail_bytes=64,
        known_best_sha=first_manifest["models"]["best"]["sha256"],
        known_candidate_sha=first_manifest["models"]["candidate"]["sha256"],
    )
    assert not second_manifest["models"]["best"]["included"]
    assert not second_manifest["models"]["candidate"]["included"]

    install_snapshot(config, second_archive)

    assert (latest / MODEL_ARCHIVE_PATHS["best"]).read_bytes() == b"formal-best-v1"
    assert (latest / MODEL_ARCHIVE_PATHS["candidate"]).read_bytes() == b"candidate-v1"


def test_changed_candidate_never_replaces_published_best(tmp_path: Path) -> None:
    cloud = _cloud_project(tmp_path)
    config = _config(tmp_path)
    archive = tmp_path / "first.tar.gz"
    create_remote_snapshot(
        archive=archive,
        project_root=cloud,
        log_tail_bytes=64,
    )
    first = install_snapshot(config, archive)
    (cloud / "artifacts" / "baseline" / "checkpoints" / "candidate.pt").write_bytes(
        b"candidate-v2"
    )
    next_archive = tmp_path / "next.tar.gz"
    create_remote_snapshot(
        archive=next_archive,
        project_root=cloud,
        log_tail_bytes=64,
        known_best_sha=first["models"]["best"]["sha256"],
        known_candidate_sha=first["models"]["candidate"]["sha256"],
    )

    install_snapshot(config, next_archive)

    assert config.published_model.read_bytes() == b"formal-best-v1"
    assert (
        config.sync_root / "latest" / MODEL_ARCHIVE_PATHS["candidate"]
    ).read_bytes() == b"candidate-v2"


def test_manifest_checksum_failure_does_not_replace_latest(tmp_path: Path) -> None:
    cloud = _cloud_project(tmp_path)
    config = _config(tmp_path)
    valid = tmp_path / "valid.tar.gz"
    create_remote_snapshot(
        archive=valid,
        project_root=cloud,
        log_tail_bytes=64,
    )
    install_snapshot(config, valid)
    original_manifest = (
        config.sync_root / "latest" / "manifest.json"
    ).read_bytes()

    corrupt = tmp_path / "corrupt.tar.gz"
    with tarfile.open(valid, "r:gz") as source, tarfile.open(corrupt, "w:gz") as target:
        for member in source.getmembers():
            extracted = source.extractfile(member)
            content = extracted.read() if extracted is not None else b""
            if member.name == "state/pipeline_state.json":
                content += b"corrupt"
                member.size = len(content)
            target.addfile(member, io.BytesIO(content))

    with pytest.raises(SyncError, match="size mismatch"):
        install_snapshot(config, corrupt)

    assert (config.sync_root / "latest" / "manifest.json").read_bytes() == original_manifest


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        content = b"bad"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))

    with pytest.raises(SyncError, match="Unsafe archive member"):
        _safe_extract(archive, tmp_path / "extract")

    assert not (tmp_path / "outside.txt").exists()


def test_remote_command_uses_custom_code_and_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command, archive_name = build_remote_command(
        config,
        code="alphasho-012345",
        known_best_sha="best-sha",
        known_candidate_sha="candidate-sha",
        full_backup=False,
    )

    assert "runpodctl send" in command
    assert "--code alphasho-012345" in command
    assert "--known-best-sha best-sha" in command
    assert "--known-candidate-sha candidate-sha" in command
    assert command.endswith("\nexit\n")
    assert archive_name == "alphasho-sync.tar.gz"


def test_runpodctl_checksum_suffix_is_read_from_sender_log(tmp_path: Path) -> None:
    log = tmp_path / "ssh.log"
    log.write_text(
        "requested-code-47\nsending 1 files\ncode is: requested-code-47\n",
        encoding="utf-8",
    )
    process = subprocess.Popen.__new__(subprocess.Popen)
    process.returncode = None

    assert (
        _wait_for_connection_code(log, "requested-code", process, timeout=0.1)
        == "requested-code-47"
    )


def test_verify_snapshot_rejects_unsupported_schema(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"schema_version": 99, "files": {}}',
        encoding="utf-8",
    )
    with pytest.raises(SyncError, match="Unsupported"):
        verify_snapshot(tmp_path)
