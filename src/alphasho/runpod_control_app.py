"""Application entry point for the standalone RunPod controller."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from alphasho.runpod_sync import load_sync_config
from alphasho.ui.app import _apply_light_theme
from alphasho.ui.runpod_control_window import RunpodControlWindow

SERVER_NAME = "AlphaSho-RunPod-Control"


def _notify_existing_instance() -> bool:
    socket = QLocalSocket()
    socket.connectToServer(SERVER_NAME)
    if not socket.waitForConnected(500):
        return False
    socket.write(b"show")
    socket.waitForBytesWritten(500)
    socket.disconnectFromServer()
    return True


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AlphaSho RunPod Control")
    app.setOrganizationName("AlphaSho")
    app.setQuitOnLastWindowClosed(False)
    _apply_light_theme(app)
    if _notify_existing_instance():
        return 0

    QLocalServer.removeServer(SERVER_NAME)
    server = QLocalServer(app)
    if not server.listen(SERVER_NAME):
        raise RuntimeError(f"无法建立单实例控制通道：{server.errorString()}")

    project_root = Path(__file__).resolve().parents[2]
    config = load_sync_config(project_root / "configs" / "runpod_sync.toml")
    window = RunpodControlWindow(config)

    def activate() -> None:
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            connection.waitForReadyRead(100)
            connection.deleteLater()
        window.show_and_raise()

    server.newConnection.connect(activate)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
