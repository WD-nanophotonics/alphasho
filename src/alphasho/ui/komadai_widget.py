from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from alphasho.domain import Color, GameSnapshot, Piece, PieceType
from alphasho.domain.models import HAND_PIECE_TYPES
from alphasho.ui.piece_assets import PieceAssetStore


class KomadaiWidget(QWidget):
    """A wooden shogi piece stand that shows only pieces actually in hand."""

    piece_clicked = Signal(object)

    DISPLAY_ORDER = (
        PieceType.ROOK,
        PieceType.BISHOP,
        PieceType.GOLD,
        PieceType.SILVER,
        PieceType.KNIGHT,
        PieceType.LANCE,
        PieceType.PAWN,
    )

    def __init__(
        self,
        color: Color,
        assets: PieceAssetStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.color = color
        self._assets = assets
        self._snapshot: GameSnapshot | None = None
        self._selected: PieceType | None = None
        self._interactive = False
        self._flipped = False
        self._label = ""
        self._piece_rects: dict[PieceType, QRectF] = {}
        self.setMinimumSize(110, 220)
        self.setMaximumWidth(165)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(145, 255)

    def update_state(
        self,
        snapshot: GameSnapshot,
        *,
        selected: PieceType | None,
        interactive: bool,
        flipped: bool,
        label: str,
    ) -> None:
        self._snapshot = snapshot
        self._selected = selected
        self._interactive = interactive
        self._flipped = flipped
        self._label = label
        self.update()

    def visible_piece_types(self) -> tuple[PieceType, ...]:
        if self._snapshot is None:
            return ()
        return tuple(
            piece_type
            for piece_type in self.DISPLAY_ORDER
            if self._count(piece_type) > 0
        )

    def piece_rects(self) -> dict[PieceType, QRectF]:
        return dict(self._piece_rects)

    def _count(self, piece_type: PieceType) -> int:
        if self._snapshot is None:
            return 0
        index = HAND_PIECE_TYPES.index(piece_type)
        return self._snapshot.hands[int(self.color)][index]

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() is not Qt.MouseButton.LeftButton
            or not self._interactive
        ):
            return
        for piece_type, rect in self._piece_rects.items():
            if rect.contains(event.position()):
                self.piece_clicked.emit(piece_type)
                return

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        stand = QRectF(4, 4, self.width() - 8, self.height() - 8)
        painter.setPen(QPen(QColor("#7a511d"), 1.5))
        painter.setBrush(QColor("#dcae60"))
        painter.drawRoundedRect(stand, 5, 5)

        painter.setPen(QPen(QColor(133, 82, 24, 32), 1))
        for index in range(1, 10):
            x = stand.left() + stand.width() * index / 10
            painter.drawLine(int(x), int(stand.top()), int(x - 3), int(stand.bottom()))

        font = QFont(self.font())
        font.setPointSizeF(max(8.0, min(11.0, self.width() * 0.075)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#3d2816"))
        painter.drawText(
            QRectF(9, 8, self.width() - 18, 24),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )

        self._piece_rects.clear()
        pieces = self.visible_piece_types()
        if not pieces:
            return
        columns = 2
        gap = 7.0
        top = 38.0
        available_width = self.width() - gap * 3
        piece_width = min(58.0, available_width / columns)
        piece_height = piece_width * 1.06
        for index, piece_type in enumerate(pieces):
            row, column = divmod(index, columns)
            rect = QRectF(
                gap + column * (piece_width + gap),
                top + row * (piece_height + gap),
                piece_width,
                piece_height,
            )
            self._piece_rects[piece_type] = rect
            if self._selected is piece_type:
                painter.setPen(QPen(QColor("#2468a2"), 2.5))
                painter.setBrush(QColor(107, 171, 220, 85))
                painter.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), 5, 5)
            pixmap = self._assets.pixmap(
                Piece(self.color, piece_type),
                rotated=self._flipped,
            )
            if not pixmap.isNull():
                painter.drawPixmap(
                    rect.adjusted(4, 2, -4, -2),
                    pixmap,
                    QRectF(pixmap.rect()),
                )
            count = self._count(piece_type)
            if count > 1:
                badge = QRectF(rect.right() - 21, rect.top() + 2, 19, 19)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#352416"))
                painter.drawEllipse(badge)
                badge_font = QFont(self.font())
                badge_font.setPointSizeF(8.5)
                badge_font.setBold(True)
                painter.setFont(badge_font)
                painter.setPen(QColor("#fff8df"))
                painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(count))
