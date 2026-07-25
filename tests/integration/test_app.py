from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from alphasho.ai import AlphaZeroPlayer
from alphasho.domain import (
    Color,
    GameMode,
    ParticipantKind,
    PieceType,
    ThinkingPreset,
)
from alphasho.heuristicplayer import HeuristicPlayer
from alphasho.ui.main_window import MainWindow
from alphasho.ui.match_dialog import MatchSettingsDialog

OPTIONAL_PROMOTION = "4k4/9/9/4P4/9/9/9/9/4K4 b - 1"


def _window(qtbot, memory_settings) -> MainWindow:
    window = MainWindow(memory_settings)
    qtbot.addWidget(window)
    window.show()
    return window


def test_main_window_smoke(qtbot, memory_settings) -> None:
    window = _window(qtbot, memory_settings)

    assert window.windowTitle().startswith("AlphaSho")
    assert window.controller.game.snapshot().move_number == 1
    assert not window.assets.missing_files()


def test_both_local_sides_and_random_reply_can_move(
    qtbot,
    memory_settings,
) -> None:
    window = _window(qtbot, memory_settings)
    controller = window.controller
    controller.new_game(GameMode.LOCAL)
    controller.choose_move(
        next(move for move in controller.game.legal_moves() if move.usi == "7g7f")
    )
    assert controller.game.snapshot().turn is Color.WHITE
    controller.choose_move(
        next(move for move in controller.game.legal_moves() if move.usi == "3c3d")
    )
    assert controller.game.snapshot().turn is Color.BLACK

    controller.new_game(GameMode.RANDOM_AI)
    controller.choose_move(
        next(move for move in controller.game.legal_moves() if move.usi == "7g7f")
    )
    qtbot.waitUntil(
        lambda: len(controller.game.snapshot().move_history) == 2,
        timeout=2000,
    )


def test_promotion_sprite_commits_promoted_move(
    qtbot,
    memory_settings,
) -> None:
    window = _window(qtbot, memory_settings)
    window.controller.load_sfen(OPTIONAL_PROMOTION)
    window.controller.click_square(39)
    window.controller.click_square(38)
    window.board.repaint()
    QApplication.processEvents()

    promoted = window.board.promotion_choice_rects()[True]
    qtbot.mouseClick(
        window.board,
        Qt.MouseButton.LeftButton,
        pos=promoted.center().toPoint(),
    )

    snapshot = window.controller.game.snapshot()
    assert snapshot.piece_at(39) is None
    assert snapshot.piece_at(38).piece_type is PieceType.PROMOTED_PAWN
    assert snapshot.last_move.usi == "5d5c+"


def test_match_dialog_connects_selected_model_to_either_side(
    qtbot,
    memory_settings,
) -> None:
    window = _window(qtbot, memory_settings)
    dialog = MatchSettingsDialog(
        window.translator,
        window.controller.game.snapshot().sfen,
        window.controller.match_config,
        window,
    )
    qtbot.addWidget(dialog)
    fields = dialog.white_fields
    fields.ai.setChecked(True)
    fields.ai_selector.setCurrentIndex(
        fields.ai_selector.findData(ParticipantKind.ALPHAZERO.value)
    )
    debug_path = str(dialog.debug_model_path)
    fields.model_selector.setCurrentIndex(
        fields.model_selector.findData(debug_path)
    )
    fields.thinking_preset.setCurrentIndex(
        fields.thinking_preset.findData(ThinkingPreset.QUICK.value)
    )

    config = dialog.match_config()
    assert config.white.model_path == debug_path
    assert config.white.thinking.preset is ThinkingPreset.QUICK
    assert config.white.thinking.move_time_seconds == 1.0
    window.controller.new_game(config)
    assert isinstance(window.controller.players[Color.WHITE], AlphaZeroPlayer)


def test_match_dialog_can_select_isolated_heuristic_player(
    qtbot,
    memory_settings,
) -> None:
    window = _window(qtbot, memory_settings)
    dialog = MatchSettingsDialog(
        window.translator,
        window.controller.game.snapshot().sfen,
        window.controller.match_config,
        window,
    )
    qtbot.addWidget(dialog)
    fields = dialog.white_fields
    fields.ai.setChecked(True)
    fields.ai_selector.setCurrentIndex(
        fields.ai_selector.findData(ParticipantKind.HEURISTIC.value)
    )

    config = dialog.match_config()
    assert config.white.kind is ParticipantKind.HEURISTIC
    assert config.white.model_path is None
    window.controller.new_game(config)
    assert isinstance(window.controller.players[Color.WHITE], HeuristicPlayer)
