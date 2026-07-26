from __future__ import annotations

import io
from pathlib import Path

import alphasho.ui.runpod_control_window as control_window
from alphasho.runpod_control import PodStatus
from alphasho.runpod_sync import SyncConfig
from alphasho.ui.runpod_control_window import RunpodControlWindow, _LogStreamThread


class _FakeController:
    def __init__(self) -> None:
        self.checks = 0

    def check_status(self) -> PodStatus:
        self.checks += 1
        return PodStatus(
            online=True,
            project_exists=True,
            venv_exists=True,
            config_exists=True,
            training_state="stopped",
            git_revision="remote",
            git_branch="master",
        )


def _config(tmp_path: Path) -> SyncConfig:
    project = tmp_path / "project"
    return SyncConfig(
        project_root=project,
        source_path=project / "configs" / "runpod_sync.toml",
        ssh_target="pod@ssh.runpod.io",
        identity_file=tmp_path / "key",
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
        task_name="test",
    )


def test_window_starts_disconnected_and_status_check_is_background(qtbot, tmp_path: Path) -> None:
    controller = _FakeController()
    window = RunpodControlWindow(_config(tmp_path), controller=controller)  # type: ignore[arg-type]
    qtbot.addWidget(window)

    assert "尚未检查" in window.status_values["connection"].text()
    assert not window._periodic_sync
    window.check_button.click()
    assert window.check_button.isEnabled() is False
    qtbot.waitUntil(lambda: controller.checks == 1 and window.check_button.isEnabled())
    assert "在线" in window.status_values["connection"].text()
    assert window.status_values["training"].text() == "已停止，可恢复"


def test_close_hides_to_tray_instead_of_enabling_sync(qtbot, tmp_path: Path) -> None:
    window = RunpodControlWindow(_config(tmp_path), controller=_FakeController())  # type: ignore[arg-type]
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert not window.isVisible()
    assert not window._periodic_sync


def test_log_stream_thread_is_silent_and_delivers_output(qtbot, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = self
            self.returncode: int | None = None
            self._sent = False

        def readline(self) -> bytes:
            if not self._sent:
                self._sent = True
                self.returncode = 0
                return "训练日志\n".encode()
            return b""

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode or 0

    def fake_popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(control_window.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        control_window,
        "silent_subprocess_kwargs",
        lambda: {"creationflags": 0x08000000, "startupinfo": "hidden"},
    )
    thread = _LogStreamThread(["ssh.exe", "pod"], "tail -F log\n", None)
    output: list[str] = []
    thread.output_received.connect(output.append)

    with qtbot.waitSignal(thread.finished, timeout=2000):
        thread.start()

    assert output == ["训练日志\n"]
    assert captured["creationflags"] == 0x08000000
    assert captured["startupinfo"] == "hidden"
    assert captured["stdin"] == control_window.subprocess.PIPE
    thread.deleteLater()
