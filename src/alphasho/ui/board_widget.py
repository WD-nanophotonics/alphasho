from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from alphasho.domain import Color, GameSnapshot, MoveOption, Piece, PieceType
from alphasho.ui.piece_assets import PieceAssetStore

_PROMOTED_TYPES = {
    PieceType.PAWN: PieceType.PROMOTED_PAWN,
    PieceType.LANCE: PieceType.PROMOTED_LANCE,
    PieceType.KNIGHT: PieceType.PROMOTED_KNIGHT,
    PieceType.SILVER: PieceType.PROMOTED_SILVER,
    PieceType.BISHOP: PieceType.HORSE,
    PieceType.ROOK: PieceType.DRAGON,
}


class BoardWidget(QWidget):
    square_clicked = Signal(int)
    promotion_choice_clicked = Signal(object)
    promotion_choice_cancelled = Signal()

    def __init__(self, assets: PieceAssetStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._assets = assets
        self._snapshot: GameSnapshot | None = None
        self._selected: int | None = None
        self._targets: frozenset[int] = frozenset()
        self._flipped = False
        self._rank_labels = tuple("一二三四五六七八九")
        self._promotion_moves: tuple[MoveOption, ...] = ()
        self._promotion_piece: Piece | None = None
        self._promotion_rects: dict[bool, QRectF] = {}
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def sizeHint(self) -> QSize:
        return QSize(680, 680)

    def set_snapshot(self, snapshot: GameSnapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def set_selection(self, selected: int | None, targets: set[int]) -> None:
        self._selected = selected
        self._targets = frozenset(targets)
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        self._flipped = bool(flipped)
        self.clear_promotion_choices()
        self.update()

    def is_flipped(self) -> bool:
        return self._flipped

    def set_rank_labels(self, labels: tuple[str, ...]) -> None:
        if len(labels) != 9:
            raise ValueError("Exactly nine rank labels are required")
        self._rank_labels = labels
        self.update()

    @staticmethod
    def square_to_cell(square: int, flipped: bool = False) -> tuple[int, int]:
        file_index, rank = divmod(square, 9)
        row, column = rank, 8 - file_index
        return (8 - row, 8 - column) if flipped else (row, column)

    @staticmethod
    def cell_to_square(row: int, column: int, flipped: bool = False) -> int:
        if flipped:
            row, column = 8 - row, 8 - column
        return (8 - column) * 9 + row

    def show_promotion_choices(self, moves: tuple[MoveOption, ...]) -> None:
        if self._snapshot is None or len(moves) != 2:
            raise ValueError("Promotion choices require exactly two legal moves")
        source = moves[0].from_square
        if source is None or any(move.from_square != source for move in moves):
            raise ValueError("Promotion choices must share a source square")
        piece = self._snapshot.piece_at(source)
        if piece is None:
            raise ValueError("The promotion source square is empty")
        self._promotion_moves = tuple(sorted(moves, key=lambda move: not move.promotion))
        self._promotion_piece = piece
        self._promotion_rects.clear()
        self.setFocus()
        self.update()

    def clear_promotion_choices(self) -> None:
        self._promotion_moves = ()
        self._promotion_piece = None
        self._promotion_rects.clear()
        self.update()

    def has_promotion_choices(self) -> bool:
        return bool(self._promotion_moves)

    def promotion_choice_rects(self) -> dict[bool, QRectF]:
        return dict(self._promotion_rects)

    def square_rect(self, square: int) -> QRectF:
        board_rect, cell = self._geometry()
        return self._cell_rect(board_rect, cell, square)

    def _geometry(self) -> tuple[QRectF, float]:
        margin = min(58.0, max(42.0, min(self.width(), self.height()) * 0.085))
        side = max(0.0, min(self.width(), self.height()) - margin * 2)
        cell = side / 9.0 if side else 0.0
        left = (self.width() - side) / 2.0
        top = (self.height() - side) / 2.0
        return QRectF(left, top, side, side), cell

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton:
            return
        if self._promotion_moves:
            for move in self._promotion_moves:
                rect = self._promotion_rects.get(move.promotion)
                if rect is not None and rect.contains(event.position()):
                    self.clear_promotion_choices()
                    self.promotion_choice_clicked.emit(move)
                    return
            self.clear_promotion_choices()
            self.promotion_choice_cancelled.emit()
            return

        board_rect, cell = self._geometry()
        if not board_rect.contains(event.position()) or cell <= 0:
            return
        column = min(8, int((event.position().x() - board_rect.left()) / cell))
        row = min(8, int((event.position().y() - board_rect.top()) / cell))
        self.square_clicked.emit(self.cell_to_square(row, column, self._flipped))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._promotion_moves:
            self.clear_promotion_choices()
            self.promotion_choice_cancelled.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#f5f0df"))
        board_rect, cell = self._geometry()
        if cell <= 0:
            return

        self._paint_wood(painter, board_rect, cell)
        self._paint_highlights(painter, board_rect, cell)
        self._paint_grid(painter, board_rect, cell)
        self._paint_coordinates(painter, board_rect, cell)
        self._paint_pieces(painter, board_rect, cell)
        self._paint_promotion_choices(painter, board_rect, cell)

    def _paint_wood(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        painter.fillRect(board_rect, QColor("#e2b565"))
        painter.setPen(QPen(QColor(151, 98, 36, 24), 1))
        for index in range(1, 23):
            x = board_rect.left() + board_rect.width() * index / 23
            bend = (index % 4 - 1.5) * cell * 0.025
            painter.drawLine(
                QPointF(x, board_rect.top()),
                QPointF(x + bend, board_rect.bottom()),
            )

    def _paint_grid(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        painter.setPen(QPen(QColor("#4b2e16"), max(1.0, cell * 0.022)))
        for index in range(10):
            x = board_rect.left() + index * cell
            y = board_rect.top() + index * cell
            painter.drawLine(QPointF(x, board_rect.top()), QPointF(x, board_rect.bottom()))
            painter.drawLine(QPointF(board_rect.left(), y), QPointF(board_rect.right(), y))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#382315"))
        radius = max(2.5, cell * 0.055)
        for row in (3, 6):
            for column in (3, 6):
                center = QPointF(
                    board_rect.left() + column * cell,
                    board_rect.top() + row * cell,
                )
                painter.drawEllipse(center, radius, radius)

    def _cell_rect(self, board_rect: QRectF, cell: float, square: int) -> QRectF:
        row, column = self.square_to_cell(square, self._flipped)
        return QRectF(
            board_rect.left() + column * cell,
            board_rect.top() + row * cell,
            cell,
            cell,
        )

    def _paint_highlights(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        if self._snapshot and self._snapshot.last_move:
            painter.fillRect(
                self._cell_rect(board_rect, cell, self._snapshot.last_move.to_square),
                QColor(244, 183, 45, 95),
            )
        if self._selected is not None:
            painter.fillRect(
                self._cell_rect(board_rect, cell, self._selected),
                QColor(54, 113, 177, 90),
            )
        if self._snapshot and self._snapshot.is_check:
            for square, piece in enumerate(self._snapshot.board):
                if (
                    piece is not None
                    and piece.color is self._snapshot.turn
                    and piece.piece_type is PieceType.KING
                ):
                    painter.fillRect(
                        self._cell_rect(board_rect, cell, square),
                        QColor(201, 49, 45, 115),
                    )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(32, 125, 75, 145))
        radius = max(4.0, cell * 0.09)
        for square in self._targets:
            painter.drawEllipse(
                self._cell_rect(board_rect, cell, square).center(),
                radius,
                radius,
            )

    def _paint_coordinates(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        font = QFont(self.font())
        font.setPointSizeF(max(8.0, cell * 0.15))
        painter.setFont(font)
        painter.setPen(QColor("#4b2e16"))
        file_labels = tuple(str(number) for number in range(9, 0, -1))
        rank_labels = self._rank_labels
        if self._flipped:
            file_labels = tuple(reversed(file_labels))
            rank_labels = tuple(reversed(rank_labels))
        for column, label in enumerate(file_labels):
            painter.drawText(
                QRectF(board_rect.left() + column * cell, board_rect.top() - 25, cell, 20),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )
        for row, label in enumerate(rank_labels):
            painter.drawText(
                QRectF(board_rect.right() + 7, board_rect.top() + row * cell, 26, cell),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

    def _paint_piece(
        self,
        painter: QPainter,
        piece: Piece,
        rect: QRectF,
    ) -> None:
        pixmap = self._assets.pixmap(piece, rotated=self._flipped)
        if not pixmap.isNull():
            painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))

    def _paint_pieces(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        if self._snapshot is None:
            return
        preview_source = (
            self._promotion_moves[0].from_square if self._promotion_moves else None
        )
        preview_target = self._promotion_moves[0].to_square if self._promotion_moves else None
        for square, piece in enumerate(self._snapshot.board):
            if piece is None or square in (preview_source, preview_target):
                continue
            rect = self._cell_rect(board_rect, cell, square).adjusted(
                cell * 0.08,
                cell * 0.05,
                -cell * 0.08,
                -cell * 0.05,
            )
            self._paint_piece(painter, piece, rect)
        if self._promotion_piece is not None and preview_target is not None:
            rect = self._cell_rect(board_rect, cell, preview_target).adjusted(
                cell * 0.08,
                cell * 0.05,
                -cell * 0.08,
                -cell * 0.05,
            )
            self._paint_piece(painter, self._promotion_piece, rect)

    def _paint_promotion_choices(
        self,
        painter: QPainter,
        board_rect: QRectF,
        cell: float,
    ) -> None:
        if not self._promotion_moves or self._promotion_piece is None:
            self._promotion_rects.clear()
            return
        target_rect = self._cell_rect(board_rect, cell, self._promotion_moves[0].to_square)
        option_width = cell * 0.78
        option_height = cell * 0.86
        gap = max(4.0, cell * 0.08)
        panel_width = option_width * 2 + gap * 3
        panel_height = option_height + gap * 2
        mover_at_bottom = (
            self._promotion_piece.color is Color.BLACK
        ) != self._flipped
        panel_y = (
            target_rect.top() - panel_height - gap
            if mover_at_bottom
            else target_rect.bottom() + gap
        )
        panel_x = target_rect.center().x() - panel_width / 2
        panel_x = min(max(4.0, panel_x), self.width() - panel_width - 4.0)
        panel_y = min(max(4.0, panel_y), self.height() - panel_height - 4.0)
        panel = QRectF(panel_x, panel_y, panel_width, panel_height)

        painter.setPen(QPen(QColor("#77501e"), 1.5))
        painter.setBrush(QColor(245, 216, 157, 248))
        painter.drawRoundedRect(panel, 7, 7)

        self._promotion_rects.clear()
        original_type = self._promotion_piece.piece_type
        for index, promotion in enumerate((True, False)):
            rect = QRectF(
                panel.left() + gap + index * (option_width + gap),
                panel.top() + gap,
                option_width,
                option_height,
            )
            self._promotion_rects[promotion] = rect
            painter.setPen(QPen(QColor("#a06a24"), 1))
            painter.setBrush(QColor("#fff4d9"))
            painter.drawRoundedRect(rect, 5, 5)
            piece_type = _PROMOTED_TYPES[original_type] if promotion else original_type
            piece = Piece(self._promotion_piece.color, piece_type)
            self._paint_piece(
                painter,
                piece,
                rect.adjusted(gap * 0.65, gap * 0.35, -gap * 0.65, -gap * 0.35),
            )
