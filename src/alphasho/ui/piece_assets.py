from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap, QTransform

from alphasho.domain import Color, Piece, PieceType


class PieceAssetStore:
    """Loads the user-provided hitomoji piece images once."""

    _NAMES = {
        PieceType.PAWN: "pawn",
        PieceType.LANCE: "lance",
        PieceType.KNIGHT: "knight",
        PieceType.SILVER: "silver",
        PieceType.BISHOP: "bishop",
        PieceType.ROOK: "rook",
        PieceType.GOLD: "gold",
        PieceType.KING: "king",
        PieceType.PROMOTED_PAWN: "prom_pawn",
        PieceType.PROMOTED_LANCE: "prom_lance",
        PieceType.PROMOTED_KNIGHT: "prom_knight",
        PieceType.PROMOTED_SILVER: "prom_silver",
        PieceType.HORSE: "horse",
        PieceType.DRAGON: "dragon",
    }

    def __init__(self, root: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.root = root or project_root / "assets" / "pieces" / "hitomoji"
        self._pixmaps: dict[tuple[Color, PieceType, bool], QPixmap] = {}

    def pixmap(self, piece: Piece, *, rotated: bool = False) -> QPixmap:
        key = (piece.color, piece.piece_type, rotated)
        if key not in self._pixmaps:
            prefix = "black" if piece.color is Color.BLACK else "white"
            suffix = self._NAMES[piece.piece_type]
            filename = f"{prefix}_{suffix}.png"
            if piece.piece_type is PieceType.KING and piece.color is Color.WHITE:
                filename = "white_king2.png"
            pixmap = QPixmap(str(self.root / filename))
            if rotated and not pixmap.isNull():
                pixmap = pixmap.transformed(QTransform().rotate(180))
            self._pixmaps[key] = pixmap
        return self._pixmaps[key]

    def icon(
        self,
        piece: Piece,
        size: QSize | None = None,
        *,
        rotated: bool = False,
    ) -> QIcon:
        size = size or QSize(42, 44)
        pixmap = self.pixmap(piece, rotated=rotated)
        if pixmap.isNull():
            return QIcon()
        return QIcon(pixmap.scaled(size))

    def missing_files(self) -> tuple[Path, ...]:
        missing = []
        for color in Color:
            for piece_type in PieceType:
                piece = Piece(color, piece_type)
                if self.pixmap(piece).isNull():
                    prefix = "black" if color is Color.BLACK else "white"
                    missing.append(self.root / f"{prefix}_{self._NAMES[piece_type]}.png")
        return tuple(missing)
