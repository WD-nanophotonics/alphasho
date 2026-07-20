from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

from alphasho.ui.main_window import MainWindow


def create_application(argv: list[str] | None = None) -> tuple[QApplication, MainWindow]:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("AlphaSho")
    app.setOrganizationName("AlphaSho")
    _apply_light_theme(app)
    window = MainWindow()
    return app, window


def _apply_light_theme(app: QApplication) -> None:
    """Keep every native input readable even when Windows uses a dark theme."""
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f3efdf"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#2e2118"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#fffdf8"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f2e7d1"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#fff8e8"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#2e2118"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#2e2118"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#fffaf0"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#2e2118"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#b7772d"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor("#82796d"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor("#82796d"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor("#82796d"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Base,
        QColor("#e8e3da"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Button,
        QColor("#e8e3da"),
    )
    app.setPalette(palette)


def main() -> int:
    app, window = create_application()
    window.show()
    return app.exec()
