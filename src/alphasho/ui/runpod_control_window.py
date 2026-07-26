"""Standalone RunPod training controller window."""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from alphasho.runpod_control import (
    PauseTimeout,
    PodStatus,
    RunpodController,
    local_git_status,
    revisions_match,
    sync_summary,
)
from alphasho.runpod_sync import SyncConfig, silent_subprocess_kwargs, sync_once


class _TaskSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class _Task(QRunnable):
    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            self.signals.succeeded.emit(self.function())
        except Exception as exc:  # worker boundary
            self.signals.failed.emit(exc)
        finally:
            self.signals.finished.emit()


class _LogStreamThread(QThread):
    """Read a long-lived SSH log stream without creating a console window."""

    output_received = Signal(str)
    stream_error = Signal(str)

    def __init__(self, arguments: list[str], command: str, parent: QObject) -> None:
        super().__init__(parent)
        self.arguments = arguments
        self.command = command
        self._stop_requested = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                self.arguments,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                **silent_subprocess_kwargs(),
            )
            with self._process_lock:
                self._process = process
            if self._stop_requested.is_set():
                process.terminate()
                return
            assert process.stdin is not None
            process.stdin.write(self.command.encode("utf-8"))
            process.stdin.flush()
            assert process.stdout is not None
            while not self._stop_requested.is_set():
                content = process.stdout.readline()
                if not content:
                    if process.poll() is not None:
                        break
                    continue
                self.output_received.emit(content.decode("utf-8", errors="replace"))
            return_code = process.poll()
            if (
                return_code not in {None, 0}
                and not self._stop_requested.is_set()
            ):
                self.stream_error.emit(f"日志 SSH 已退出（{return_code}）")
        except Exception as exc:
            if not self._stop_requested.is_set():
                self.stream_error.emit(str(exc))
        finally:
            with self._process_lock:
                self._process = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()

    def stop_stream(self) -> None:
        self._stop_requested.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def kill_stream(self) -> None:
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.kill()


class RunpodControlWindow(QMainWindow):
    def __init__(self, config: SyncConfig, controller: RunpodController | None = None) -> None:
        super().__init__()
        self.config = config
        self.controller = controller or RunpodController(config)
        self.local_git = local_git_status(config.project_root)
        self.remote_status: PodStatus | None = None
        self._allow_exit = False
        self._monitoring_active = False
        self._periodic_sync = False
        self._sync_deadline: float | None = None
        self._busy_names: set[str] = set()
        self._tasks: set[_Task] = set()
        self._log_paused = False
        self._log_thread: _LogStreamThread | None = None

        self.setWindowTitle("AlphaSho · RunPod 训练控制")
        self.resize(1040, 720)
        self._build_ui()
        self._build_tray()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(config.control.status_poll_seconds * 1000)
        self.status_timer.timeout.connect(self.check_connection)

        self.sync_timer = QTimer(self)
        self.sync_timer.setSingleShot(True)
        self.sync_timer.timeout.connect(self._periodic_sync_due)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._update_sync_display)
        self.countdown_timer.start()

        self.log_reconnect_timer = QTimer(self)
        self.log_reconnect_timer.setSingleShot(True)
        self.log_reconnect_timer.timeout.connect(self.connect_log)
        self._refresh_local_display()
        self._refresh_sync_summary()
        self._set_connection_text("尚未检查（程序不会自动连接）")

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        form = QFormLayout()
        self.status_values: dict[str, QLabel] = {}
        rows = [
            ("connection", "Pod 状态"),
            ("training", "训练状态"),
            ("pipeline", "训练进度"),
            ("gpu", "GPU"),
            ("remote", "远端代码"),
            ("local", "本地代码"),
            ("version", "版本比较"),
            ("sync", "成果同步"),
            ("model", "本地正式模型"),
        ]
        for key, title in rows:
            label = QLabel("—")
            label.setTextInteractionFlags(label.textInteractionFlags())
            label.setWordWrap(True)
            self.status_values[key] = label
            form.addRow(title, label)
        root.addLayout(form)

        buttons = QHBoxLayout()
        self.check_button = QPushButton("检查连接")
        self.start_button = QPushButton("启动/恢复训练")
        self.pause_button = QPushButton("安全暂停训练")
        self.sync_button = QPushButton("立即同步")
        self.periodic_button = QPushButton("开启每15分钟同步")
        self.backup_button = QPushButton("下载完整恢复包")
        self.folder_button = QPushButton("打开本地同步目录")
        for button in (
            self.check_button,
            self.start_button,
            self.pause_button,
            self.sync_button,
            self.periodic_button,
            self.backup_button,
            self.folder_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        log_controls = QHBoxLayout()
        self.reconnect_log_button = QPushButton("重新连接日志")
        self.pause_log_checkbox = QCheckBox("暂停自动滚动")
        self.clear_log_button = QPushButton("清空显示")
        self.copy_log_button = QPushButton("复制日志")
        log_controls.addWidget(QLabel("云端 console.log"))
        log_controls.addStretch(1)
        for widget in (
            self.reconnect_log_button,
            self.pause_log_checkbox,
            self.clear_log_button,
            self.copy_log_button,
        ):
            log_controls.addWidget(widget)
        root.addLayout(log_controls)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(20_000)
        root.addWidget(self.log_view, 1)
        self.setCentralWidget(central)

        self.check_button.clicked.connect(self.check_connection)
        self.start_button.clicked.connect(self.request_start)
        self.pause_button.clicked.connect(self.request_pause)
        self.sync_button.clicked.connect(self.sync_now)
        self.periodic_button.clicked.connect(self.toggle_periodic_sync)
        self.backup_button.clicked.connect(self.full_backup)
        self.folder_button.clicked.connect(self.open_sync_folder)
        self.reconnect_log_button.clicked.connect(self.connect_log)
        self.pause_log_checkbox.toggled.connect(self._set_log_paused)
        self.clear_log_button.clicked.connect(self.log_view.clear)
        self.copy_log_button.clicked.connect(self._copy_log)
        self.statusBar().showMessage("就绪；等待人工操作")

    def _build_tray(self) -> None:
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)
        menu = self.tray.contextMenu() or None
        if menu is None:
            from PySide6.QtWidgets import QMenu

            menu = QMenu()
            self.tray.setContextMenu(menu)
        show_action = QAction("显示控制窗口", self)
        sync_action = QAction("立即同步", self)
        periodic_action = QAction("开启/关闭定时同步", self)
        start_action = QAction("启动/恢复训练", self)
        pause_action = QAction("安全暂停训练", self)
        exit_action = QAction("真正退出", self)
        show_action.triggered.connect(self.show_and_raise)
        sync_action.triggered.connect(self.sync_now)
        periodic_action.triggered.connect(self.toggle_periodic_sync)
        start_action.triggered.connect(self.request_start)
        pause_action.triggered.connect(self.request_pause)
        exit_action.triggered.connect(self.request_exit)
        for action in (
            show_action,
            sync_action,
            periodic_action,
            start_action,
            pause_action,
        ):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(exit_action)
        self.tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _run_task(
        self,
        name: str,
        function: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> bool:
        if name in self._busy_names:
            return False
        task = _Task(function)
        self._tasks.add(task)
        self._busy_names.add(name)
        task.signals.succeeded.connect(on_success or (lambda _value: None))
        task.signals.failed.connect(
            on_failure or (lambda error: self._show_error(name, error))
        )

        def finished() -> None:
            self._busy_names.discard(name)
            self._tasks.discard(task)
            self._update_button_states()

        task.signals.finished.connect(finished)
        self._update_button_states()
        QThreadPool.globalInstance().start(task)
        return True

    def _update_button_states(self) -> None:
        control_busy = "control" in self._busy_names or "status" in self._busy_names
        sync_busy = "sync" in self._busy_names
        self.check_button.setEnabled(not control_busy)
        self.start_button.setEnabled(not control_busy)
        self.pause_button.setEnabled(not control_busy)
        self.sync_button.setEnabled(not sync_busy)
        self.backup_button.setEnabled(not sync_busy)

    def check_connection(self) -> None:
        if "status" in self._busy_names or "control" in self._busy_names:
            return
        self._monitoring_active = True
        self.local_git = local_git_status(self.config.project_root)
        self._refresh_local_display()
        if not self.status_timer.isActive():
            self.status_timer.start()
        self._set_connection_text("检查中…")
        self._run_task("status", self.controller.check_status, self._status_received)

    def _status_received(self, status: PodStatus) -> None:
        self.remote_status = status
        self._render_status(status)
        if status.online and status.log_exists and self._log_thread is None:
            self.connect_log()

    def _render_status(self, status: PodStatus) -> None:
        if not status.online:
            self._set_connection_text(f"离线或不可达：{status.error}")
            self.status_values["training"].setText("未知")
            return
        self._set_connection_text(
            f"在线 · {self.config.ssh_target} · {self.config.remote_project_root}"
        )
        names = {
            "running": "运行中",
            "stopped": "已停止，可恢复",
            "conflict": "检测到 tmux 外训练进程（禁止重复启动）",
            "stale_session": "tmux session 异常残留",
        }
        self.status_values["training"].setText(
            names.get(status.training_state, status.training_state)
        )
        self.status_values["pipeline"].setText(
            f"iteration {status.active_iteration} · {status.stage} · revision {status.revision} · "
            f"self-play {status.selfplay_completed} · evaluation {status.evaluation_completed}"
        )
        self.status_values["gpu"].setText(status.gpu)
        dirty = "dirty" if status.git_dirty else "clean"
        self.status_values["remote"].setText(
            f"{status.git_branch} · {status.git_revision[:12]} · {dirty}"
        )
        match = revisions_match(self.local_git, status)
        comparison = "一致" if match else "不一致" if match is False else "无法判断"
        if self.local_git.dirty or status.git_dirty:
            comparison += "；存在未提交改动"
        self.status_values["version"].setText(comparison)
        self.statusBar().showMessage("状态检查完成")

    def request_start(self) -> None:
        if self.remote_status is None or not self.remote_status.online:
            self._run_task(
                "status",
                self.controller.check_status,
                lambda status: self._start_after_status(status),
            )
            return
        self._start_after_status(self.remote_status)

    def _start_after_status(self, status: PodStatus) -> None:
        self.remote_status = status
        self._render_status(status)
        if not status.online:
            self._show_error("启动训练", RuntimeError(status.error or "Pod 离线"))
            return
        mismatch = revisions_match(self.local_git, status) is False
        if mismatch or self.local_git.dirty or status.git_dirty:
            answer = QMessageBox.warning(
                self,
                "代码版本提示",
                "本地与远端代码版本不一致，或存在未提交改动。\n"
                "控制程序不会上传、拉取或修改代码。仍要使用远端现有代码启动训练吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.statusBar().showMessage("正在启动或恢复训练…")
        self._run_task("control", self.controller.start_training, self._control_finished)

    def request_pause(self) -> None:
        self.statusBar().showMessage("正在请求安全暂停并等待 checkpoint…")
        self._run_task(
            "control",
            self.controller.pause_training,
            self._control_finished,
            self._pause_failed,
        )

    def _pause_failed(self, error: Exception) -> None:
        if not isinstance(error, PauseTimeout):
            self._show_error("安全暂停", error)
            return
        answer = QMessageBox.warning(
            self,
            "安全暂停超时",
            f"{error}\n\n强制停止可能丢失当前未完成工作，确定要强制结束 tmux session 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._run_task("control", self.controller.force_stop, self._status_received)

    def _control_finished(self, result: dict[str, Any]) -> None:
        status = result.get("status")
        if isinstance(status, PodStatus):
            self._status_received(status)
        reason = result.get("reason", "完成")
        messages = {
            "started": "训练已经启动并在 tmux 中运行",
            "already_running": "训练原本已在运行，没有重复启动",
            "safe_checkpoint": "训练已安全暂停，checkpoint 已保存",
            "already_stopped": "训练原本已经停止",
        }
        self.statusBar().showMessage(messages.get(reason, str(reason)), 10_000)

    def sync_now(self) -> None:
        self._start_sync(full_backup=False)

    def full_backup(self) -> None:
        self._start_sync(full_backup=True)

    def _start_sync(self, *, full_backup: bool) -> None:
        if "sync" in self._busy_names:
            self.statusBar().showMessage("同步任务正在运行，不会重复启动")
            return
        label = "完整恢复包" if full_backup else "训练成果"
        self.statusBar().showMessage(f"正在下载{label}…")

        def done(_manifest: Any) -> None:
            self._refresh_sync_summary()
            self.statusBar().showMessage(f"{label}同步完成", 10_000)
            if self._periodic_sync and not full_backup:
                self._schedule_next_sync()

        def failed(error: Exception) -> None:
            self._refresh_sync_summary()
            self._show_error(f"{label}同步", error)
            if self._periodic_sync and not full_backup:
                self._schedule_next_sync()

        self._run_task(
            "sync",
            lambda: sync_once(self.config, full_backup=full_backup),
            done,
            failed,
        )

    def toggle_periodic_sync(self) -> None:
        self._periodic_sync = not self._periodic_sync
        if self._periodic_sync:
            self.periodic_button.setText("关闭每15分钟同步")
            self._sync_deadline = None
            self.sync_timer.stop()
            self.sync_now()
        else:
            self.periodic_button.setText("开启每15分钟同步")
            self.sync_timer.stop()
            self._sync_deadline = None
            self._update_sync_display()

    def _periodic_sync_due(self) -> None:
        if not self._periodic_sync:
            return
        if "sync" in self._busy_names:
            self._sync_deadline = time.monotonic() + 60
            self.sync_timer.start(60_000)
            return
        self.sync_now()

    def _schedule_next_sync(self) -> None:
        delay = self.config.interval_minutes * 60
        self._sync_deadline = time.monotonic() + delay
        self.sync_timer.start(delay * 1000)
        self._update_sync_display()

    def _refresh_sync_summary(self) -> None:
        summary = sync_summary(self.config)
        detail = f"上次成功：{summary['last_success']}"
        if summary["error"]:
            detail += f" · 最近错误：{summary['error']}"
        self.status_values["sync"].setText(detail)
        sha = str(summary["best_sha256"])
        exists = self.config.published_model.is_file()
        self.status_values["model"].setText(
            f"{'已就绪' if exists else '不存在'} · {self.config.published_model} · {sha[:12]}"
        )

    def _update_sync_display(self) -> None:
        summary = sync_summary(self.config)
        if not self._periodic_sync:
            suffix = "定时同步关闭"
        elif self._sync_deadline is None:
            suffix = "首次同步进行中或等待执行"
        else:
            remaining = max(0, int(self._sync_deadline - time.monotonic()))
            suffix = f"下次同步 {remaining // 60:02d}:{remaining % 60:02d}"
        self.status_values["sync"].setText(
            f"上次成功：{summary['last_success']} · {suffix}"
        )

    def connect_log(self) -> None:
        self._monitoring_active = True
        self.log_reconnect_timer.stop()
        self._stop_log_thread()
        try:
            arguments, command = self.controller.log_connection()
        except Exception as exc:
            self._append_log(f"[控制程序] 无法连接日志：{exc}\n")
            self._schedule_log_reconnect()
            return
        thread = _LogStreamThread(arguments, command, self)
        thread.output_received.connect(
            lambda content, current=thread: self._read_log_output(current, content)
        )
        thread.stream_error.connect(
            lambda error, current=thread: self._log_error(current, error)
        )
        thread.finished.connect(
            lambda current=thread: self._log_finished(current)
        )
        self._log_thread = thread
        thread.start()
        self._append_log("[控制程序] 正在连接云端训练日志\n")

    def _read_log_output(self, thread: _LogStreamThread, content: str) -> None:
        if self._log_thread is thread:
            self._append_log(content.replace("\r", ""))

    def _append_log(self, content: str) -> None:
        if not content:
            return
        scrollbar = self.log_view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 2
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(content)
        if not self._log_paused and at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _log_finished(self, thread: _LogStreamThread) -> None:
        if self._log_thread is not thread:
            thread.deleteLater()
            return
        self._log_thread = None
        thread.deleteLater()
        self._append_log("\n[控制程序] 日志连接已断开\n")
        self._schedule_log_reconnect()

    def _log_error(self, thread: _LogStreamThread, error: str) -> None:
        if self._log_thread is not thread:
            return
        self._append_log(f"\n[控制程序] 日志连接错误：{error}\n")

    def _schedule_log_reconnect(self) -> None:
        if self._monitoring_active and not self._allow_exit:
            self.log_reconnect_timer.start(
                self.config.control.log_reconnect_seconds * 1000
            )

    def _stop_log_thread(self) -> None:
        thread = self._log_thread
        self._log_thread = None
        if thread is not None and thread.isRunning():
            thread.stop_stream()
            if not thread.wait(1000):
                thread.kill_stream()
                thread.wait(1000)

    def _set_log_paused(self, paused: bool) -> None:
        self._log_paused = paused

    def _copy_log(self) -> None:
        QApplication.clipboard().setText(self.log_view.toPlainText())
        self.statusBar().showMessage("日志已复制", 3000)

    def _refresh_local_display(self) -> None:
        dirty = "dirty" if self.local_git.dirty else "clean"
        self.status_values["local"].setText(
            f"{self.local_git.branch} · {self.local_git.revision[:12]} · {dirty}"
        )

    def _set_connection_text(self, text: str) -> None:
        self.status_values["connection"].setText(text)

    def _show_error(self, operation: str, error: Exception) -> None:
        self.statusBar().showMessage(f"{operation}失败：{error}", 15_000)
        QMessageBox.critical(self, f"{operation}失败", str(error))

    def open_sync_folder(self) -> None:
        self.config.sync_root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config.sync_root)))

    def show_and_raise(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_and_raise()

    def request_exit(self) -> None:
        if self._busy_names:
            answer = QMessageBox.question(
                self,
                "操作仍在进行",
                "同步或远端操作仍在进行。请等待完成后再退出。\n取消本次退出吗？",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            return
        self._allow_exit = True
        self._periodic_sync = False
        self.sync_timer.stop()
        self.status_timer.stop()
        self.log_reconnect_timer.stop()
        self._stop_log_thread()
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_exit or not QSystemTrayIcon.isSystemTrayAvailable():
            event.accept()
            return
        event.ignore()
        self.hide()
        self.tray.showMessage(
            "AlphaSho RunPod 控制",
            "程序仍在托盘运行；云端训练不会因隐藏窗口而停止。",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )
