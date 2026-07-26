"""Safe local control primitives for an already-running RunPod."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alphasho.runpod_sync import SyncConfig, _ssh_arguments, silent_subprocess_kwargs

STATUS_MARKER = "__ALPHASHO_STATUS__"
RESULT_MARKER = "__ALPHASHO_RC__"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ControlError(RuntimeError):
    """An actionable remote-control failure."""


class PauseTimeout(ControlError):
    """The trainer did not finish its safe checkpoint before the deadline."""


@dataclass(frozen=True, slots=True)
class GitStatus:
    revision: str = "unknown"
    branch: str = "unknown"
    dirty: bool = False


@dataclass(frozen=True, slots=True)
class PodStatus:
    online: bool
    project_exists: bool = False
    venv_exists: bool = False
    config_exists: bool = False
    tmux_running: bool = False
    trainer_running: bool = False
    training_state: str = "unknown"
    git_revision: str = "unknown"
    git_branch: str = "unknown"
    git_dirty: bool = False
    active_iteration: int = 0
    stage: str = "unknown"
    revision: int = 0
    selfplay_completed: int = 0
    evaluation_completed: int = 0
    gpu: str = "unknown"
    log_exists: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RemoteRunner = Callable[[str, float], str]


def _clean_terminal_output(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).replace("\r", "")


def local_git_status(project_root: Path) -> GitStatus:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            **silent_subprocess_kwargs(),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    revision = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    dirty_output = git("status", "--porcelain", "--untracked-files=normal")
    return GitStatus(
        revision=revision,
        branch=branch or "detached",
        dirty=dirty_output not in {"", "unknown"},
    )


def revisions_match(local: GitStatus, remote: PodStatus) -> bool | None:
    if local.revision == "unknown" or remote.git_revision == "unknown":
        return None
    return local.revision == remote.git_revision


class RunpodController:
    """Runs a small, fixed set of commands through the RunPod SSH proxy."""

    def __init__(
        self,
        config: SyncConfig,
        *,
        remote_runner: RemoteRunner | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._remote_runner = remote_runner
        self._sleep = sleeper

    def _preflight(self) -> None:
        if not self.config.ssh_executable.is_file():
            raise ControlError(f"找不到 SSH：{self.config.ssh_executable}")
        if not self.config.identity_file.is_file():
            raise ControlError(f"找不到 SSH 密钥：{self.config.identity_file}")

    def _run_remote(self, inner: str, timeout: float = 45.0) -> str:
        if self._remote_runner is not None:
            return self._remote_runner(inner, timeout)
        self._preflight()
        wrapped = (
            f"{inner}; rc=$?; "
            f"printf '\\n{RESULT_MARKER}%s\\n' \"$rc\"; exit \"$rc\""
        )
        command = f"bash -lc {shlex.quote(wrapped)}\nexit\n"
        try:
            result = subprocess.run(
                _ssh_arguments(self.config),
                input=command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **silent_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ControlError(f"SSH 操作超过 {timeout:g} 秒") from exc
        except OSError as exc:
            raise ControlError(f"无法启动 SSH：{exc}") from exc
        output = _clean_terminal_output(result.stdout + result.stderr)
        matches = re.findall(rf"{RESULT_MARKER}(\d+)", output)
        remote_code = int(matches[-1]) if matches else result.returncode
        if remote_code:
            detail = output[-2000:].strip()
            raise ControlError(detail or f"远端命令失败（{remote_code}）")
        return output

    def check_status(self) -> PodStatus:
        control = self.config.control
        root = self.config.remote_project_root
        payload = {
            "root": root,
            "entry": control.training_entry,
            "config": control.training_config,
            "state": control.state_path,
            "log": control.log_path,
            "session": control.tmux_session,
        }
        script = f"""
import json, os, pathlib, subprocess
p = json.loads({json.dumps(json.dumps(payload))})
root = pathlib.Path(p['root'])
def run(args):
    try:
        value = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return value.returncode, value.stdout.strip()
    except Exception:
        return 127, ''
def trainer_processes():
    found = []
    for item in pathlib.Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            parts = (item / 'cmdline').read_bytes().split(b'\\0')
            args = [part.decode(errors='replace') for part in parts if part]
        except Exception:
            continue
        if args and 'python' in pathlib.Path(args[0]).name.lower() and any(
            arg.endswith(p['entry']) for arg in args[1:]
        ):
            found.append({{'pid': int(item.name), 'command': ' '.join(args)}})
    return found
state = {{}}
try:
    state = json.loads((root / p['state']).read_text())
except Exception:
    pass
git_rc, git_revision = run(['git', '-C', str(root), 'rev-parse', 'HEAD'])
_, git_branch = run(['git', '-C', str(root), 'branch', '--show-current'])
_, git_dirty = run(['git', '-C', str(root), 'status', '--porcelain'])
tmux_rc, _ = run(['tmux', 'has-session', '-t', p['session']])
processes = trainer_processes()
tmux_running = tmux_rc == 0
trainer_running = bool(processes)
if tmux_running and trainer_running:
    training_state = 'running'
elif trainer_running:
    training_state = 'conflict'
elif tmux_running:
    training_state = 'stale_session'
else:
    training_state = 'stopped'
gpu_rc, gpu = run([
    'nvidia-smi',
    '--query-gpu=name,utilization.gpu,memory.used,memory.total',
    '--format=csv,noheader,nounits',
])
result = {{
    'online': True,
    'project_exists': root.is_dir(),
    'venv_exists': (root / '.venv/bin/python').is_file(),
    'config_exists': (root / p['config']).is_file(),
    'tmux_running': tmux_running,
    'trainer_running': trainer_running,
    'training_state': training_state,
    'git_revision': git_revision if git_rc == 0 else 'unknown',
    'git_branch': git_branch or 'detached',
    'git_dirty': bool(git_dirty),
    'active_iteration': int(state.get('active_iteration', 0)),
    'stage': str(state.get('stage', 'unknown')),
    'revision': int(state.get('revision', 0)),
    'selfplay_completed': int(state.get('selfplay_completed', 0)),
    'evaluation_completed': int(state.get('evaluation_completed', 0)),
    'gpu': gpu if gpu_rc == 0 else 'unavailable',
    'log_exists': (root / p['log']).is_file(),
}}
print('{STATUS_MARKER}' + json.dumps(result, separators=(',', ':')))
""".strip()
        command = f"python3 -c {shlex.quote(script)}"
        try:
            output = self._run_remote(
                command,
                timeout=45,
            )
            matches = re.findall(rf"{STATUS_MARKER}([^\n]+)", output)
            if not matches:
                raise ControlError("远端状态响应缺少结构化数据")
            return PodStatus(**json.loads(matches[-1]))
        except (ControlError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return PodStatus(online=False, error=str(exc))

    def start_training(self) -> dict[str, Any]:
        status = self.check_status()
        if not status.online:
            raise ControlError(status.error or "Pod 当前离线")
        if status.training_state == "running":
            return {"started": False, "reason": "already_running", "status": status}
        if status.trainer_running:
            raise ControlError("发现不属于配置 tmux session 的训练进程，已拒绝重复启动")
        if status.tmux_running:
            raise ControlError("tmux session 存在但训练进程不存在，请先强制清理该 session")
        missing = []
        if not status.project_exists:
            missing.append("远端项目目录")
        if not status.venv_exists:
            missing.append(".venv/bin/python")
        if not status.config_exists:
            missing.append(self.config.control.training_config)
        if missing:
            raise ControlError("启动条件不完整：" + "、".join(missing))

        c = self.config.control
        root = shlex.quote(self.config.remote_project_root)
        entry = shlex.quote(c.training_entry)
        log = shlex.quote(c.log_path)
        profile = shlex.quote(c.training_profile)
        training = (
            f"set -o pipefail; cd {root}; "
            f"ALPHASHO_TRAINING_PROFILE={profile} "
            f".venv/bin/python -u {entry} 2>&1 | tee -a {log}"
        )
        log_parent = shlex.quote(str(Path(c.log_path).parent).replace(os.sep, "/"))
        inner = (
            f"cd {root} && mkdir -p {log_parent} "
            f"&& tmux new-session -d -s {shlex.quote(c.tmux_session)} "
            f"bash -lc {shlex.quote(training)}"
        )
        self._run_remote(inner)
        self._sleep(1.0)
        after = self.check_status()
        if after.training_state != "running":
            raise ControlError("训练命令已发送，但未检测到运行中的训练进程，请查看日志")
        return {"started": True, "reason": "started", "status": after}

    def pause_training(self) -> dict[str, Any]:
        before = self.check_status()
        if not before.online:
            raise ControlError(before.error or "Pod 当前离线")
        if before.training_state == "stopped":
            return {"paused": False, "reason": "already_stopped", "status": before}
        if not before.tmux_running:
            raise ControlError("训练不在配置的 tmux session 中，不能安全发送中断")
        self._run_remote(
            f"tmux send-keys -t {shlex.quote(self.config.control.tmux_session)} C-c"
        )
        deadline = time.monotonic() + self.config.control.pause_timeout_seconds
        last = before
        while time.monotonic() < deadline:
            self._sleep(2.0)
            last = self.check_status()
            if last.online and not last.tmux_running and not last.trainer_running:
                return {
                    "paused": True,
                    "reason": "safe_checkpoint",
                    "revision_before": before.revision,
                    "revision_after": last.revision,
                    "status": last,
                }
        raise PauseTimeout(
            f"等待安全保存超过 {self.config.control.pause_timeout_seconds} 秒；"
            "训练仍可能正在保存，请确认后再强制停止"
        )

    def force_stop(self) -> PodStatus:
        session = shlex.quote(self.config.control.tmux_session)
        self._run_remote(f"tmux kill-session -t {session}")
        self._sleep(1.0)
        return self.check_status()

    def log_connection(self) -> tuple[list[str], str]:
        """Return an SSH invocation for the desktop log-stream thread."""
        self._preflight()
        c = self.config.control
        inner = (
            f"cd {shlex.quote(self.config.remote_project_root)} && "
            f"touch {shlex.quote(c.log_path)} && "
            f"tail -n {c.log_tail_lines} -F -- {shlex.quote(c.log_path)}"
        )
        return _ssh_arguments(self.config), f"bash -lc {shlex.quote(inner)}\n"


def sync_summary(config: SyncConfig) -> dict[str, Any]:
    status_path = config.sync_root / "sync_status.json"
    manifest_path = config.sync_root / "latest" / "manifest.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        status = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        manifest = {}
    best = manifest.get("models", {}).get("best", {})
    return {
        "last_success": status.get("last_success", "尚未同步"),
        "last_attempt": status.get("last_attempt", "尚未同步"),
        "ok": status.get("ok"),
        "error": status.get("error", ""),
        "best_sha256": best.get("sha256", "unknown"),
        "cloud_commit": status.get("cloud_commit", manifest.get("cloud_commit", "unknown")),
    }
