"""Pull verified training snapshots from a Runpod through its basic SSH proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from alphasho.atomic_io import (
    ExclusiveFileLock,
    FileLockError,
    atomic_copy,
    atomic_write_json,
)

RUNPODCTL_WINDOWS_URL = (
    "https://github.com/runpod/runpodctl/releases/latest/download/"
    "runpodctl-windows-amd64.exe"
)
SCHEMA_VERSION = 1
MODEL_ARCHIVE_PATHS = {
    "best": "models/best.pt",
    "candidate": "models/candidate.pt",
}


class SyncError(RuntimeError):
    """A safe, user-facing synchronization failure."""


def silent_subprocess_kwargs() -> dict[str, Any]:
    """Hide console children spawned by the pythonw desktop controller."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Remote training controls shared by the desktop controller."""

    tmux_session: str = "alphasho-v3"
    training_profile: str = "baseline"
    training_entry: str = "training_pipeline.py"
    training_config: str = "configs/training/baseline.toml"
    log_path: str = "artifacts/baseline/logs/console.log"
    state_path: str = "artifacts/baseline/checkpoints/pipeline_state.json"
    status_poll_seconds: int = 15
    pause_timeout_seconds: int = 120
    log_tail_lines: int = 300
    log_reconnect_seconds: int = 5


@dataclass(frozen=True, slots=True)
class SyncConfig:
    project_root: Path
    source_path: Path
    ssh_target: str
    identity_file: Path
    ssh_executable: Path
    remote_project_root: str
    sync_root: Path
    published_model: Path
    runpodctl: Path
    interval_minutes: int
    log_tail_bytes: int
    timeout_seconds: int
    full_backup_timeout_seconds: int
    retry_attempts: int
    task_name: str
    control: ControlConfig = field(default_factory=ControlConfig)


def _expand_path(value: str, base: Path | None = None) -> Path:
    expanded = os.path.expandvars(value)
    if os.name == "nt":
        for name, content in os.environ.items():
            expanded = expanded.replace(f"%{name}%", content)
            expanded = expanded.replace(f"%{name.lower()}%", content)
    path = Path(expanded).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def load_sync_config(path: str | Path) -> SyncConfig:
    source = Path(path).resolve()
    values = tomllib.loads(source.read_text(encoding="utf-8"))
    connection = values.get("connection", {})
    local = values.get("local", {})
    sync = values.get("sync", {})
    control = values.get("control", {})
    project_root = source.parent.parent
    config = SyncConfig(
        project_root=project_root,
        source_path=source,
        ssh_target=str(connection["ssh_target"]),
        identity_file=_expand_path(str(connection["identity_file"])),
        ssh_executable=_expand_path(str(connection["ssh_executable"])),
        remote_project_root=str(connection["remote_project_root"]).rstrip("/"),
        sync_root=_expand_path(str(local["sync_root"]), project_root),
        published_model=_expand_path(str(local["published_model"]), project_root),
        runpodctl=_expand_path(str(local["runpodctl"])),
        interval_minutes=int(sync.get("interval_minutes", 15)),
        log_tail_bytes=int(sync.get("log_tail_bytes", 8 * 1024 * 1024)),
        timeout_seconds=int(sync.get("timeout_seconds", 600)),
        full_backup_timeout_seconds=int(
            sync.get("full_backup_timeout_seconds", 7200)
        ),
        retry_attempts=int(sync.get("retry_attempts", 3)),
        task_name=str(sync.get("task_name", "AlphaSho Runpod Sync")),
        control=ControlConfig(
            tmux_session=str(control.get("tmux_session", "alphasho-v3")),
            training_profile=str(control.get("training_profile", "baseline")),
            training_entry=str(control.get("training_entry", "training_pipeline.py")),
            training_config=str(
                control.get("training_config", "configs/training/baseline.toml")
            ),
            log_path=str(
                control.get("log_path", "artifacts/baseline/logs/console.log")
            ),
            state_path=str(
                control.get(
                    "state_path",
                    "artifacts/baseline/checkpoints/pipeline_state.json",
                )
            ),
            status_poll_seconds=int(control.get("status_poll_seconds", 15)),
            pause_timeout_seconds=int(control.get("pause_timeout_seconds", 120)),
            log_tail_lines=int(control.get("log_tail_lines", 300)),
            log_reconnect_seconds=int(control.get("log_reconnect_seconds", 5)),
        ),
    )
    if config.interval_minutes <= 0:
        raise ValueError("sync.interval_minutes must be positive")
    if config.log_tail_bytes <= 0:
        raise ValueError("sync.log_tail_bytes must be positive")
    if config.retry_attempts <= 0:
        raise ValueError("sync.retry_attempts must be positive")
    if not config.ssh_target or not config.remote_project_root.startswith("/"):
        raise ValueError("connection settings are incomplete")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.control.tmux_session):
        raise ValueError("control.tmux_session contains unsafe characters")
    for name in (
        "training_entry",
        "training_config",
        "log_path",
        "state_path",
    ):
        path = PurePosixPath(getattr(config.control, name))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"control.{name} must be a project-relative safe path")
    if min(
        config.control.status_poll_seconds,
        config.control.pause_timeout_seconds,
        config.control.log_tail_lines,
        config.control.log_reconnect_seconds,
    ) <= 0:
        raise ValueError("control timing and log settings must be positive")
    return config


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _git_revision(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            **silent_subprocess_kwargs(),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _tail_bytes(source: Path, destination: Path, limit: int) -> None:
    size = source.stat().st_size
    with source.open("rb") as source_handle:
        if size > limit:
            source_handle.seek(size - limit)
            source_handle.readline()
        with destination.open("wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)


def _copy_snapshot_file(
    source: Path,
    destination: Path,
    *,
    tail_bytes: int | None = None,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tail_bytes is None:
        shutil.copyfile(source, destination)
    else:
        _tail_bytes(source, destination, tail_bytes)
    return {
        "sha256": _sha256(destination),
        "size": destination.stat().st_size,
        "source": source.as_posix(),
        "tail": tail_bytes is not None,
    }


def create_remote_snapshot(
    *,
    archive: Path,
    project_root: Path,
    log_tail_bytes: int,
    known_best_sha: str = "",
    known_candidate_sha: str = "",
    full_backup: bool = False,
) -> dict[str, Any]:
    """Create an immutable archive without modifying the active trainer."""

    project_root = project_root.resolve()
    artifacts = project_root / "artifacts"
    baseline = artifacts / "baseline"
    checkpoints = baseline / "checkpoints"
    state = _read_json(checkpoints / "pipeline_state.json", {})
    effective = _read_json(baseline / "effective_config.json", {})
    sync_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive.with_suffix(archive.suffix + ".tmp")
    with tempfile.TemporaryDirectory(
        prefix="alphasho-export-",
        dir=archive.parent,
    ) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sync_id": sync_id,
            "created_at": _utc_now(),
            "cloud_commit": _git_revision(project_root),
            "full_backup": full_backup,
            "pipeline": state,
            "evaluation_games": effective.get("evaluation_games"),
            "promotion_score": effective.get("promotion_score"),
            "files": {},
            "models": {},
        }
        files: dict[str, Any] = manifest["files"]
        models: dict[str, Any] = manifest["models"]
        model_sources = {
            "best": artifacts / "checkpoints" / "best.pt",
            "candidate": checkpoints / "candidate.pt",
        }
        known_hashes = {
            "best": known_best_sha,
            "candidate": known_candidate_sha,
        }
        for model_id, source in model_sources.items():
            if not source.is_file():
                models[model_id] = {"available": False, "included": False}
                continue
            source_sha = _sha256(source)
            included = full_backup or source_sha != known_hashes[model_id]
            models[model_id] = {
                "available": True,
                "included": included,
                "sha256": source_sha,
                "size": source.stat().st_size,
                "source": source.as_posix(),
            }
            if included:
                relative = MODEL_ARCHIVE_PATHS[model_id]
                files[relative] = _copy_snapshot_file(source, staging / relative)

        state_sources = {
            "state/pipeline_state.json": checkpoints / "pipeline_state.json",
            "state/effective_config.json": baseline / "effective_config.json",
        }
        for relative, source in state_sources.items():
            if source.is_file():
                files[relative] = _copy_snapshot_file(source, staging / relative)

        logs = baseline / "logs"
        log_names = {
            "console.log",
            "baseline.jsonl",
            "status.json",
            "training.en.log",
            "training.zh.log",
            "training.ja.log",
        }
        if logs.is_dir():
            for source in sorted(logs.iterdir()):
                if source.is_file() and source.name in log_names:
                    relative = f"logs/{source.name}"
                    files[relative] = _copy_snapshot_file(
                        source,
                        staging / relative,
                        tail_bytes=None if full_backup else log_tail_bytes,
                    )

        if full_backup:
            full_root = staging / "full" / "baseline"
            shutil.copytree(
                baseline,
                full_root,
                ignore=shutil.ignore_patterns(".trainer.lock"),
            )
            for copied in sorted(full_root.rglob("*")):
                if copied.is_file():
                    relative = copied.relative_to(staging).as_posix()
                    files[relative] = {
                        "sha256": _sha256(copied),
                        "size": copied.stat().st_size,
                        "source": (
                            baseline / copied.relative_to(full_root)
                        ).as_posix(),
                        "tail": False,
                    }
            published = artifacts / "checkpoints" / "best.pt"
            if published.is_file():
                destination = staging / "full" / "published" / "best.pt"
                files["full/published/best.pt"] = _copy_snapshot_file(
                    published,
                    destination,
                )

        atomic_write_json(staging / "manifest.json", manifest)
        with tarfile.open(temporary_archive, "w:gz") as bundle:
            for source in sorted(staging.rglob("*")):
                if source.is_file():
                    bundle.add(
                        source,
                        arcname=source.relative_to(staging).as_posix(),
                        recursive=False,
                    )
        os.replace(temporary_archive, archive)
        return manifest


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise SyncError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination, members=members, filter="data")


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest = _read_json(snapshot / "manifest.json")
    if not isinstance(manifest, dict):
        raise SyncError("Snapshot manifest is missing or invalid")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SyncError("Unsupported snapshot schema")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SyncError("Snapshot file index is invalid")
    actual_files = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    expected_files = set(files) | {"manifest.json"}
    if actual_files != expected_files:
        unexpected = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise SyncError(
            f"Snapshot file index mismatch: unexpected={unexpected}, missing={missing}"
        )
    for relative, metadata in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SyncError(f"Unsafe manifest path: {relative}")
        source = snapshot.joinpath(*path.parts)
        if not source.is_file():
            raise SyncError(f"Snapshot file is missing: {relative}")
        if source.stat().st_size != int(metadata["size"]):
            raise SyncError(f"Snapshot size mismatch: {relative}")
        if _sha256(source) != str(metadata["sha256"]):
            raise SyncError(f"Snapshot checksum mismatch: {relative}")
    return manifest


def _known_model_hashes(latest: Path) -> tuple[str, str]:
    manifest = _read_json(latest / "manifest.json", {})
    models = manifest.get("models", {}) if isinstance(manifest, dict) else {}
    return (
        str(models.get("best", {}).get("sha256", "")),
        str(models.get("candidate", {}).get("sha256", "")),
    )


def _inherit_unchanged_models(staging: Path, latest: Path, manifest: dict[str, Any]) -> None:
    for model_id, relative in MODEL_ARCHIVE_PATHS.items():
        model = manifest.get("models", {}).get(model_id, {})
        if not model.get("available") or model.get("included"):
            continue
        previous = latest / relative
        if not previous.is_file() or _sha256(previous) != model.get("sha256"):
            raise SyncError(f"Cloud omitted {model_id}, but no matching local copy exists")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous, destination)


def install_snapshot(config: SyncConfig, archive: Path) -> dict[str, Any]:
    latest = config.sync_root / "latest"
    staging = config.sync_root / ".latest.new"
    previous = config.sync_root / ".latest.previous"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _safe_extract(archive, staging)
    manifest = verify_snapshot(staging)
    _inherit_unchanged_models(staging, latest, manifest)
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        os.replace(latest, previous)
    try:
        os.replace(staging, latest)
    except OSError:
        if previous.exists() and not latest.exists():
            os.replace(previous, latest)
        raise
    if previous.exists():
        shutil.rmtree(previous)

    best = latest / MODEL_ARCHIVE_PATHS["best"]
    best_meta = manifest.get("models", {}).get("best", {})
    if best_meta.get("available"):
        if not best.is_file() or _sha256(best) != best_meta.get("sha256"):
            raise SyncError("Verified snapshot does not contain the declared best model")
        if (
            not config.published_model.is_file()
            or _sha256(config.published_model) != best_meta.get("sha256")
        ):
            atomic_copy(best, config.published_model)
    return manifest


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_remote_command(
    config: SyncConfig,
    *,
    code: str,
    known_best_sha: str,
    known_candidate_sha: str,
    full_backup: bool,
) -> tuple[str, str]:
    archive_name = (
        f"alphasho-full-{time.time_ns()}.tar.gz"
        if full_backup
        else "alphasho-sync.tar.gz"
    )
    archive = f"/workspace/{archive_name}"
    arguments = [
        ".venv/bin/python",
        "-m",
        "alphasho.runpod_sync",
        "--remote-export",
        archive,
        "--remote-project-root",
        config.remote_project_root,
        "--log-tail-bytes",
        str(config.log_tail_bytes),
        "--known-best-sha",
        known_best_sha,
        "--known-candidate-sha",
        known_candidate_sha,
    ]
    if full_backup:
        arguments.append("--remote-full-backup")
    export = " ".join(_quote(argument) for argument in arguments)
    inner = (
        f"cd {_quote(config.remote_project_root)} && "
        f"PYTHONPATH=src {export} && "
        f"runpodctl send {_quote(archive)} --code {_quote(code)}; "
        "rc=$?; "
        f"rm -f {_quote(archive)}; "
        "exit $rc"
    )
    return f"bash -lc {_quote(inner)}\nexit\n", archive_name


def _ssh_arguments(config: SyncConfig) -> list[str]:
    return [
        str(config.ssh_executable),
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-i",
        str(config.identity_file),
        config.ssh_target,
    ]


def _wait_for_connection_code(
    ssh_log: Path,
    requested_code: str,
    ssh: subprocess.Popen[str],
    *,
    timeout: float = 30.0,
) -> str:
    """Read the checksum suffix that runpodctl appends to a custom code."""

    deadline = time.monotonic() + timeout
    pattern = re.compile(r"code is:\s*([A-Za-z0-9_-]+)")
    fallback = re.compile(
        rf"(?m)^\s*({re.escape(requested_code)}-[A-Za-z0-9]+)\s*$"
    )
    while time.monotonic() < deadline:
        content = ""
        try:
            content = ssh_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        match = pattern.search(content) or fallback.search(content)
        if match:
            return match.group(1)
        if ssh.poll() is not None:
            detail = content[-2000:]
            raise SyncError(
                f"Remote sender exited before publishing a transfer code: {detail}"
            )
        time.sleep(0.2)
    raise SyncError("Timed out waiting for the remote runpodctl transfer code")


def _transfer_once(config: SyncConfig, *, full_backup: bool) -> Path:
    if not config.ssh_executable.is_file():
        raise SyncError(f"SSH executable not found: {config.ssh_executable}")
    if not config.identity_file.is_file():
        raise SyncError(f"SSH identity not found: {config.identity_file}")
    if not config.runpodctl.is_file():
        raise SyncError(
            f"runpodctl not found: {config.runpodctl}; run --install-runpodctl first"
        )
    known_best, known_candidate = _known_model_hashes(
        config.sync_root / "latest"
    )
    code = f"alphasho-{secrets.token_hex(16)}"
    remote_command, archive_name = build_remote_command(
        config,
        code=code,
        known_best_sha=known_best,
        known_candidate_sha=known_candidate,
        full_backup=full_backup,
    )
    timeout = (
        config.full_backup_timeout_seconds if full_backup else config.timeout_seconds
    )
    receive_dir = Path(
        tempfile.mkdtemp(prefix="alphasho-receive-", dir=config.sync_root)
    )
    ssh_log = receive_dir / "ssh.log"
    try:
        with ssh_log.open("w", encoding="utf-8", errors="replace") as output:
            ssh = subprocess.Popen(
                _ssh_arguments(config),
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                **silent_subprocess_kwargs(),
            )
            assert ssh.stdin is not None
            ssh.stdin.write(remote_command)
            ssh.stdin.flush()
            ssh.stdin.close()
            receive_code = _wait_for_connection_code(
                ssh_log,
                code,
                ssh,
            )
            try:
                receive = subprocess.run(
                    [str(config.runpodctl), "receive", receive_code],
                    cwd=receive_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    **silent_subprocess_kwargs(),
                )
            except subprocess.TimeoutExpired as exc:
                ssh.kill()
                raise SyncError(f"runpodctl receive timed out after {timeout}s") from exc
            try:
                ssh_code = ssh.wait(timeout=30)
            except subprocess.TimeoutExpired:
                ssh.kill()
                ssh_code = ssh.wait(timeout=5)
            if receive.returncode:
                raise SyncError(
                    "runpodctl receive failed: "
                    + (receive.stderr or receive.stdout).strip()
                )
            if ssh_code:
                detail = ssh_log.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise SyncError(f"Remote export failed (SSH {ssh_code}): {detail}")
        received = receive_dir / archive_name
        if not received.is_file():
            candidates = list(receive_dir.glob("*.tar.gz"))
            if len(candidates) != 1:
                raise SyncError("Received archive could not be identified")
            received = candidates[0]
        destination = config.sync_root / (
            "full-backups" if full_backup else "incoming"
        )
        destination.mkdir(parents=True, exist_ok=True)
        final = destination / received.name
        os.replace(received, final)
        return final
    finally:
        shutil.rmtree(receive_dir, ignore_errors=True)


def _append_log(config: SyncConfig, message: str) -> None:
    config.sync_root.mkdir(parents=True, exist_ok=True)
    with (config.sync_root / "sync.log").open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _write_status(
    config: SyncConfig,
    *,
    ok: bool,
    error: str = "",
    manifest: dict[str, Any] | None = None,
) -> None:
    path = config.sync_root / "sync_status.json"
    previous = _read_json(path, {})
    status = dict(previous) if isinstance(previous, dict) else {}
    status.update(
        {
            "schema_version": SCHEMA_VERSION,
            "last_attempt": _utc_now(),
            "ok": ok,
            "error": error,
        }
    )
    if ok and manifest is not None:
        status.update(
            {
                "last_success": _utc_now(),
                "sync_id": manifest.get("sync_id"),
                "cloud_commit": manifest.get("cloud_commit"),
                "pipeline": manifest.get("pipeline"),
                "evaluation_games": manifest.get("evaluation_games"),
                "promotion_score": manifest.get("promotion_score"),
                "models": manifest.get("models"),
            }
        )
    atomic_write_json(path, status)


def sync_once(config: SyncConfig, *, full_backup: bool = False) -> dict[str, Any]:
    config.sync_root.mkdir(parents=True, exist_ok=True)
    with ExclusiveFileLock(config.sync_root / ".sync.lock"):
        last_error: Exception | None = None
        for attempt in range(1, config.retry_attempts + 1):
            try:
                archive = _transfer_once(config, full_backup=full_backup)
                if full_backup:
                    with tempfile.TemporaryDirectory(
                        prefix="alphasho-verify-",
                        dir=config.sync_root,
                    ) as temporary:
                        _safe_extract(archive, Path(temporary))
                        manifest = verify_snapshot(Path(temporary))
                else:
                    manifest = install_snapshot(config, archive)
                    archive.unlink(missing_ok=True)
                _write_status(config, ok=True, manifest=manifest)
                _append_log(
                    config,
                    f"success sync_id={manifest.get('sync_id')} "
                    f"commit={manifest.get('cloud_commit')} full={full_backup}",
                )
                return manifest
            except Exception as exc:  # retry boundary
                last_error = exc
                _append_log(
                    config,
                    f"attempt={attempt}/{config.retry_attempts} failed: {exc}",
                )
                if attempt < config.retry_attempts:
                    time.sleep((2, 5, 10)[min(attempt - 1, 2)])
        assert last_error is not None
        _write_status(config, ok=False, error=str(last_error))
        raise SyncError(str(last_error)) from last_error


def install_runpodctl(config: SyncConfig) -> None:
    config.runpodctl.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.runpodctl.with_suffix(".exe.tmp")
    try:
        urllib.request.urlretrieve(RUNPODCTL_WINDOWS_URL, temporary)
        if temporary.stat().st_size < 1024 * 1024:
            raise SyncError("Downloaded runpodctl binary is unexpectedly small")
        os.replace(temporary, config.runpodctl)
        subprocess.run(
            [str(config.runpodctl), "version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            **silent_subprocess_kwargs(),
        )
    finally:
        temporary.unlink(missing_ok=True)


def _task_launcher_path() -> Path:
    base = _expand_path("%LOCALAPPDATA%/AlphaSho")
    return base / "runpod-sync.cmd"


def install_task(config: SyncConfig) -> None:
    if os.name != "nt":
        raise SyncError("Windows Task Scheduler is only available on Windows")
    launcher = _task_launcher_path()
    launcher.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "@echo off",
        f'cd /d "{config.project_root}"',
        (
            f'"{sys.executable}" -m alphasho.runpod_sync '
            f'--config "{config.source_path}" --once'
        ),
    ]
    launcher.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    subprocess.run(
        [
            "schtasks.exe",
            "/Create",
            "/TN",
            config.task_name,
            "/TR",
            str(launcher),
            "/SC",
            "MINUTE",
            "/MO",
            str(config.interval_minutes),
            "/F",
        ],
        check=True,
        **silent_subprocess_kwargs(),
    )
    powershell = _expand_path(
        "%WINDIR%/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    task_name = config.task_name.replace("'", "''")
    settings_script = (
        "$settings = New-ScheduledTaskSettingsSet "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-StartWhenAvailable -MultipleInstances IgnoreNew "
        "-ExecutionTimeLimit (New-TimeSpan -Minutes 12); "
        f"Set-ScheduledTask -TaskName '{task_name}' "
        "-Settings $settings | Out-Null"
    )
    subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            settings_script,
        ],
        check=True,
        **silent_subprocess_kwargs(),
    )


def uninstall_task(config: SyncConfig) -> None:
    if os.name != "nt":
        raise SyncError("Windows Task Scheduler is only available on Windows")
    subprocess.run(
        ["schtasks.exe", "/Delete", "/TN", config.task_name, "/F"],
        check=True,
        **silent_subprocess_kwargs(),
    )
    _task_launcher_path().unlink(missing_ok=True)


def print_status(config: SyncConfig) -> int:
    status = _read_json(config.sync_root / "sync_status.json")
    if not isinstance(status, dict):
        print("No synchronization has completed yet.")
        return 1
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status.get("ok") else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/runpod_sync.toml",
        help="Path to the non-secret synchronization TOML",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--once", action="store_true", help="Run one normal sync")
    actions.add_argument(
        "--full-backup",
        action="store_true",
        help="Download baseline checkpoints, logs, replay, and published best",
    )
    actions.add_argument("--status", action="store_true", help="Show local sync status")
    actions.add_argument(
        "--install-runpodctl",
        action="store_true",
        help="Install the official Windows runpodctl binary",
    )
    actions.add_argument(
        "--install-task",
        action="store_true",
        help="Register the 15-minute Windows task",
    )
    actions.add_argument(
        "--uninstall-task",
        action="store_true",
        help="Remove the Windows task",
    )
    parser.add_argument("--remote-export", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--remote-project-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--log-tail-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--known-best-sha", default="", help=argparse.SUPPRESS)
    parser.add_argument("--known-candidate-sha", default="", help=argparse.SUPPRESS)
    parser.add_argument("--remote-full-backup", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.remote_export:
        if arguments.remote_project_root is None:
            raise SystemExit("--remote-project-root is required")
        manifest = create_remote_snapshot(
            archive=arguments.remote_export,
            project_root=arguments.remote_project_root,
            log_tail_bytes=arguments.log_tail_bytes,
            known_best_sha=arguments.known_best_sha,
            known_candidate_sha=arguments.known_candidate_sha,
            full_backup=arguments.remote_full_backup,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0

    config = load_sync_config(arguments.config)
    try:
        if arguments.install_runpodctl:
            install_runpodctl(config)
            print(f"Installed runpodctl: {config.runpodctl}")
            return 0
        if arguments.install_task:
            install_task(config)
            print(f"Installed task: {config.task_name}")
            return 0
        if arguments.uninstall_task:
            uninstall_task(config)
            print(f"Removed task: {config.task_name}")
            return 0
        if arguments.status:
            return print_status(config)
        manifest = sync_once(config, full_backup=arguments.full_backup)
        print(
            f"Sync complete: {manifest.get('sync_id')} "
            f"commit={manifest.get('cloud_commit')}"
        )
        return 0
    except (SyncError, FileLockError, OSError, subprocess.SubprocessError) as exc:
        print(f"AlphaSho sync failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
