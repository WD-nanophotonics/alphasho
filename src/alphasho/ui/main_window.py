from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from alphasho.domain import (
    ClockState,
    Color,
    GameRecord,
    GameSnapshot,
    OutcomeReason,
    ParticipantKind,
    TimeControlMode,
    UiMessage,
)
from alphasho.engine import (
    InvalidPositionError,
    RecordError,
    load_record,
    save_record,
)
from alphasho.ui.board_widget import BoardWidget
from alphasho.ui.controller import GameController
from alphasho.ui.i18n import LANGUAGE_NAMES, Language, Translator
from alphasho.ui.komadai_widget import KomadaiWidget
from alphasho.ui.match_dialog import MatchSettingsDialog
from alphasho.ui.piece_assets import PieceAssetStore


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or QSettings("AlphaSho", "AlphaSho")
        self.translator = Translator(self._stored_language())
        self._flipped = self.settings.value("board_flipped", False, type=bool)
        self.resize(1280, 880)
        self.setMinimumSize(980, 700)

        self.assets = PieceAssetStore()
        self.controller = GameController(self)
        self._snapshot = self.controller.game.snapshot()
        self._clock_state = self.controller.clock.state()
        self._selected_drop = None
        self._search_progress = None
        self._announced_outcome = OutcomeReason.ONGOING
        self._loading_comment = False
        self._allow_close = False

        self._build_menus()
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(9, 7, 9, 9)
        root_layout.setSpacing(6)
        root_layout.addLayout(self._build_toolbar())

        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        root_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)
        splitter.addWidget(self._build_play_area())
        splitter.addWidget(self._build_side_panel())
        splitter.setSizes([930, 330])

        self._connect_signals()
        self._apply_style()
        self.board.set_flipped(self._flipped)
        self._retranslate()
        self.controller.publish()

    def _stored_language(self) -> Language:
        value = self.settings.value("language", Language.JAPANESE.value)
        try:
            return Language(str(value))
        except ValueError:
            return Language.JAPANESE

    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.menus = {
            key: bar.addMenu("")
            for key in (
                "menu_file",
                "menu_edit",
                "menu_position",
                "menu_game",
                "menu_view",
                "menu_help",
            )
        }
        self.menu_actions: dict[str, QAction] = {}

        def add(menu_key: str, key: str, handler=None, *, planned=False) -> QAction:
            if key in self.menu_actions:
                action = self.menu_actions[key]
                self.menus[menu_key].addAction(action)
                return action
            action = QAction(self)
            if handler is not None:
                action.triggered.connect(handler)
            action.setEnabled(not planned)
            self.menus[menu_key].addAction(action)
            self.menu_actions[key] = action
            return action

        add("menu_file", "new_game", self._new_game)
        add("menu_file", "open_record", self._open_record)
        add("menu_file", "save_record", self._save_record)
        add("menu_file", "save_record_as", self._save_record_as)
        add("menu_file", "other_formats", planned=True)
        self.menus["menu_file"].addSeparator()
        add("menu_file", "exit", self.close)

        add("menu_edit", "undo", self._undo)
        add("menu_edit", "branch_editor", planned=True)

        add("menu_position", "load_sfen", self._load_sfen)
        add("menu_position", "copy_sfen", self._copy_sfen)
        add("menu_position", "position_editor", planned=True)

        add("menu_game", "new_game", self._new_game)
        add("menu_game", "resign", self._confirm_resign)
        add("menu_game", "declare", self._declare_nyugyoku)
        self.menus["menu_game"].addSeparator()
        add("menu_game", "engine_management", planned=True)
        add("menu_game", "analysis", planned=True)
        add("menu_game", "continuous_analysis", planned=True)
        add("menu_game", "tsume", planned=True)

        flip = add("menu_view", "flip", self._toggle_flip)
        flip.setCheckable(True)
        flip.setChecked(self._flipped)
        add("menu_view", "evaluation_graph", planned=True)

        self.language_menu = self.menus["menu_view"].addMenu("")
        self.language_actions: dict[Language, QAction] = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for language, name in LANGUAGE_NAMES.items():
            action = QAction(name, self)
            action.setCheckable(True)
            action.setData(language.value)
            action.triggered.connect(
                lambda checked=False, value=language: self._set_language(value)
            )
            group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[language] = action
        add("menu_help", "about", self._show_about)

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(5)
        self.toolbar_buttons: dict[str, QPushButton] = {}
        for key, handler in (
            ("new_game", self._new_game),
            ("open_record", self._open_record),
            ("save_record", self._save_record),
            ("undo", self._undo),
            ("resign", self._confirm_resign),
            ("declare", self._declare_nyugyoku),
            ("flip", self._toggle_flip),
            ("load_sfen", self._load_sfen),
            ("copy_sfen", self._copy_sfen),
        ):
            button = QPushButton()
            button.clicked.connect(handler)
            self.toolbar_buttons[key] = button
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        self.language_label = QLabel()
        toolbar.addWidget(self.language_label)
        self.language_combo = QComboBox()
        for language, display_name in LANGUAGE_NAMES.items():
            self.language_combo.addItem(display_name, language.value)
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        toolbar.addWidget(self.language_combo)
        return toolbar

    def _build_play_area(self) -> QWidget:
        play_area = QWidget()
        self.play_layout = QGridLayout(play_area)
        self.play_layout.setContentsMargins(0, 0, 4, 0)
        self.play_layout.setHorizontalSpacing(8)
        self.play_layout.setVerticalSpacing(8)
        self.play_layout.setColumnStretch(1, 1)
        self.play_layout.setRowStretch(0, 1)
        self.play_layout.setRowStretch(1, 1)

        self.white_komadai = KomadaiWidget(Color.WHITE, self.assets)
        self.black_komadai = KomadaiWidget(Color.BLACK, self.assets)
        self.board = BoardWidget(self.assets)
        self.play_layout.addWidget(self.board, 0, 1, 2, 1)
        self._arrange_komadai()
        return play_area

    def _build_side_panel(self) -> QWidget:
        side = QWidget()
        side.setMinimumWidth(270)
        side.setMaximumWidth(390)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(5, 0, 0, 0)
        side_layout.setSpacing(5)

        self.player_box = QGroupBox()
        player_layout = QGridLayout(self.player_box)
        self.player_names: dict[Color, QLabel] = {}
        self.clock_labels: dict[Color, QLabel] = {}
        self.overtime_labels: dict[Color, QLabel] = {}
        for row, color in enumerate((Color.WHITE, Color.BLACK)):
            name = QLabel()
            name.setObjectName(f"player{color.name}")
            clock = QLabel()
            clock.setObjectName("clock")
            overtime = QLabel()
            self.player_names[color] = name
            self.clock_labels[color] = clock
            self.overtime_labels[color] = overtime
            player_layout.addWidget(name, row * 2, 0)
            player_layout.addWidget(clock, row * 2, 1)
            player_layout.addWidget(overtime, row * 2 + 1, 0, 1, 2)
        side_layout.addWidget(self.player_box)

        self.moves_label = QLabel()
        side_layout.addWidget(self.moves_label)
        self.move_list = QListWidget()
        side_layout.addWidget(self.move_list, 2)

        self.comments_label = QLabel()
        side_layout.addWidget(self.comments_label)
        self.comment_edit = QTextEdit()
        self.comment_edit.setMinimumHeight(75)
        self.comment_edit.setMaximumHeight(120)
        side_layout.addWidget(self.comment_edit)

        self.sfen_heading = QLabel()
        side_layout.addWidget(self.sfen_heading)
        self.sfen_label = QLabel()
        self.sfen_label.setObjectName("sfenLabel")
        self.sfen_label.setWordWrap(True)
        self.sfen_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        side_layout.addWidget(self.sfen_label)

        self.evaluation_box = QGroupBox()
        evaluation_layout = QVBoxLayout(self.evaluation_box)
        self.evaluation_placeholder = QLabel()
        self.evaluation_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        evaluation_layout.addWidget(self.evaluation_placeholder)
        side_layout.addWidget(self.evaluation_box)

        self.engine_box = QGroupBox()
        engine_layout = QVBoxLayout(self.engine_box)
        self.engine_placeholder = QLabel()
        self.engine_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        engine_layout.addWidget(self.engine_placeholder)
        side_layout.addWidget(self.engine_box)
        return side

    def _connect_signals(self) -> None:
        self.board.square_clicked.connect(self.controller.click_square)
        self.board.promotion_choice_clicked.connect(self.controller.choose_move)
        self.white_komadai.piece_clicked.connect(
            lambda piece_type: self.controller.click_hand_piece(
                Color.WHITE,
                piece_type,
            )
        )
        self.black_komadai.piece_clicked.connect(
            lambda piece_type: self.controller.click_hand_piece(
                Color.BLACK,
                piece_type,
            )
        )
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.selection_changed.connect(self._on_selection_changed)
        self.controller.promotion_requested.connect(
            self.board.show_promotion_choices
        )
        self.controller.clock_changed.connect(self._on_clock_changed)
        self.controller.record_changed.connect(self._on_record_changed)
        self.controller.error_occurred.connect(self._show_message)
        self.controller.search_progress_changed.connect(
            self._on_search_progress
        )
        self.comment_edit.textChanged.connect(self._comment_changed)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { color: #2e2118; }
            QMainWindow, QDialog { background: #f3efdf; }
            QMenuBar, QMenu { background: #fffaf0; color: #2f2118; }
            QMenuBar::item:selected, QMenu::item:selected { background: #e8d2aa; }
            QMenu::item:disabled { color: #81786c; background: #f1ede5; }
            QLabel#statusLabel {
                background: #3d2a1c; color: #fff8e8; padding: 7px 11px;
                border-radius: 3px; font-size: 14px; font-weight: 600;
            }
            QLabel#clock { font-family: Consolas; font-size: 17px; font-weight: 700; }
            QLabel#sfenLabel {
                background: #fffdf6; border: 1px solid #c1ad86;
                padding: 5px; font-family: Consolas;
            }
            QGroupBox {
                border: 1px solid #b99e71; border-radius: 3px;
                margin-top: 8px; padding-top: 7px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QGroupBox:disabled, QGroupBox::title:disabled { color: #81786c; }
            QPushButton {
                color: #2f2118; background: #fffaf0; padding: 5px 8px;
                border: 1px solid #b99e71; border-radius: 2px;
            }
            QPushButton:hover { background: #efdbb7; }
            QPushButton:disabled { color: #92897d; background: #eae6df; }
            QComboBox {
                color: #2f2118; background: #fffaf0; padding: 4px 7px;
                border: 1px solid #b99e71; border-radius: 2px;
            }
            QComboBox:disabled {
                color: #81786c; background: #e8e3da; border-color: #c8c0b3;
            }
            QLineEdit, QSpinBox {
                color: #241a13; background: #fffdf8; padding: 4px 6px;
                border: 1px solid #9f875f; border-radius: 2px;
                min-height: 23px; selection-background-color: #b7772d;
                selection-color: white;
            }
            QLineEdit:focus, QSpinBox:focus { border: 2px solid #a45d20; }
            QLineEdit:disabled, QSpinBox:disabled {
                color: #81786c; background: #e8e3da; border-color: #c8c0b3;
            }
            QRadioButton, QCheckBox {
                color: #2e2118; background: transparent; spacing: 7px;
                min-height: 23px;
            }
            QRadioButton:disabled, QCheckBox:disabled { color: #81786c; }
            QWidget#timeFields {
                background: #fff9ed; border: 1px solid #c5a978;
                border-radius: 3px;
            }
            QLabel#timeFieldsTitle {
                color: #6f3c16; font-weight: 700; background: transparent;
            }
            QListWidget, QTextEdit {
                color: #2f2118; background: #fffdf8; border: 1px solid #b99e71;
            }
            QSplitter::handle { background: #d5c8ad; width: 2px; }
            """
        )

    def _retranslate(self) -> None:
        tr = self.translator.tr
        self.setWindowTitle(tr("window_title"))
        for key, menu in self.menus.items():
            menu.setTitle(tr(key))
        self.language_menu.setTitle(tr("language"))
        for key, action in self.menu_actions.items():
            action.setText(tr(key))
            if not action.isEnabled():
                action.setStatusTip(tr("planned_status"))
                action.setToolTip(tr("planned_status"))
        for key, button in self.toolbar_buttons.items():
            button.setText(tr(key))
        self.language_label.setText(tr("language"))
        language_index = self.language_combo.findData(self.translator.language.value)
        self.language_combo.blockSignals(True)
        self.language_combo.setCurrentIndex(language_index)
        self.language_combo.blockSignals(False)
        for language, action in self.language_actions.items():
            action.setChecked(language is self.translator.language)
        self.moves_label.setText(tr("moves"))
        self.comments_label.setText(tr("comments"))
        self.sfen_heading.setText(tr("current_sfen"))
        self.player_box.setTitle(tr("time_settings"))
        self.evaluation_box.setTitle(tr("evaluation_graph"))
        self.evaluation_placeholder.setText(tr("planned_status"))
        self.engine_box.setTitle(tr("engine_info"))
        self.engine_placeholder.setText(tr("planned_status"))
        self.board.set_rank_labels(self.translator.rank_labels())
        self._render_state(self._snapshot)
        self._render_clocks(self._clock_state)
        self._refresh_komadai()

    def _language_changed(self, index: int) -> None:
        try:
            language = Language(str(self.language_combo.itemData(index)))
        except ValueError:
            return
        self._set_language(language)

    def _set_language(self, language: Language) -> None:
        self.translator.set_language(language)
        self.settings.setValue("language", language.value)
        self._retranslate()

    def _toggle_flip(self) -> None:
        self.board.clear_promotion_choices()
        self._flipped = not self._flipped
        self.settings.setValue("board_flipped", self._flipped)
        self.board.set_flipped(self._flipped)
        self.menu_actions["flip"].setChecked(self._flipped)
        self._arrange_komadai()
        self._refresh_komadai()

    def _arrange_komadai(self) -> None:
        self.play_layout.removeWidget(self.white_komadai)
        self.play_layout.removeWidget(self.black_komadai)
        upper = self.black_komadai if self._flipped else self.white_komadai
        lower = self.white_komadai if self._flipped else self.black_komadai
        self.play_layout.addWidget(
            upper,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        self.play_layout.addWidget(
            lower,
            1,
            2,
            alignment=Qt.AlignmentFlag.AlignBottom,
        )

    def _new_game(self) -> None:
        self.controller.pause_clock()
        dialog = MatchSettingsDialog(
            self.translator,
            self._snapshot.sfen,
            self.controller.match_config,
            self,
            settings=self.settings,
        )
        if dialog.exec():
            try:
                self._clear_pending_promotion()
                self._announced_outcome = OutcomeReason.ONGOING
                self.controller.new_game(dialog.match_config())
            except InvalidPositionError as exc:
                self._show_message(UiMessage.create(exc.code))
                self.controller.resume_clock()
        else:
            self.controller.resume_clock()

    def _undo(self) -> None:
        self._clear_pending_promotion()
        self.controller.undo()

    def _declare_nyugyoku(self) -> None:
        self._clear_pending_promotion()
        self.controller.declare_nyugyoku()

    def _confirm_resign(self) -> None:
        if self._snapshot.outcome.is_over:
            return
        self.controller.pause_clock()
        result = QMessageBox.question(
            self,
            self.translator.tr("confirm_resign"),
            self.translator.tr(
                "confirm_resign_text",
                player=self.translator.player(self._snapshot.turn),
            ),
        )
        if result == QMessageBox.StandardButton.Yes:
            self._clear_pending_promotion()
            self.controller.resign()
        else:
            self.controller.resume_clock()

    def _load_sfen(self) -> None:
        self.controller.pause_clock()
        text, accepted = QInputDialog.getMultiLineText(
            self,
            self.translator.tr("load_sfen_title"),
            self.translator.tr("load_sfen_prompt"),
            self._snapshot.sfen,
        )
        if not accepted:
            self.controller.resume_clock()
            return
        try:
            self._clear_pending_promotion()
            self._announced_outcome = OutcomeReason.ONGOING
            self.controller.load_sfen(text)
        except InvalidPositionError as exc:
            self._show_message(UiMessage.create(exc.code))
            self.controller.resume_clock()

    def _copy_sfen(self) -> None:
        QApplication.clipboard().setText(self._snapshot.sfen)
        self.statusBar().showMessage(self.translator.tr("sfen_copied"), 2500)

    def _open_record(self) -> None:
        self.controller.pause_clock()
        path, _selected = QFileDialog.getOpenFileName(
            self,
            self.translator.tr("open_record_title"),
            "",
            self.translator.tr("record_filter"),
        )
        if not path:
            self.controller.resume_clock()
            return
        try:
            record = load_record(path)
            self._clear_pending_promotion()
            self._announced_outcome = OutcomeReason.ONGOING
            self.controller.load_game_record(record)
            self.statusBar().showMessage(
                self.translator.tr("record_opened", path=path),
                3500,
            )
        except RecordError as exc:
            self._show_message(UiMessage.create(exc.code, **exc.params))
            self.controller.resume_clock()

    def _save_record(self) -> bool:
        if not self.controller.record.file_path:
            return self._save_record_as()
        return self._write_record(self.controller.record.file_path)

    def _save_record_as(self) -> bool:
        self.controller.pause_clock()
        suggested = self.controller.record.file_path or "alphasho-game.kifu"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.translator.tr("save_record_title"),
            suggested,
            self.translator.tr("record_filter"),
        )
        del selected_filter
        if not path:
            self.controller.resume_clock()
            return False
        if not Path(path).suffix:
            path += ".kifu"
        return self._write_record(path)

    def _write_record(self, path: str) -> bool:
        self.controller.pause_clock()
        try:
            save_record(self.controller.record, path)
            self.controller.mark_record_saved(path)
            self.statusBar().showMessage(
                self.translator.tr("record_saved", path=path),
                3500,
            )
            return True
        except RecordError as exc:
            self._show_message(UiMessage.create(exc.code, **exc.params))
            return False
        finally:
            self.controller.resume_clock()

    def _on_state_changed(self, snapshot: GameSnapshot) -> None:
        self._snapshot = snapshot
        self.board.set_snapshot(snapshot)
        self._render_state(snapshot)
        self._refresh_komadai()
        if (
            snapshot.outcome.is_over
            and snapshot.outcome.reason is not self._announced_outcome
        ):
            self._announced_outcome = snapshot.outcome.reason
            message = self.translator.outcome(snapshot.outcome)
            QTimer.singleShot(
                0,
                lambda: QMessageBox.information(
                    self,
                    self.translator.tr("game_over"),
                    message,
                ),
            )
        elif not snapshot.outcome.is_over:
            self._announced_outcome = OutcomeReason.ONGOING

    def _render_state(self, snapshot: GameSnapshot) -> None:
        if snapshot.outcome.is_over:
            status = self.translator.outcome(snapshot.outcome)
        else:
            status = self.translator.tr(
                "turn",
                move=snapshot.move_number,
                player=self.translator.player(snapshot.turn),
            )
            if snapshot.is_check:
                status += self.translator.tr("check")
            if self.controller.is_ai_thinking():
                if self._search_progress is None:
                    status += self.translator.tr("thinking")
                else:
                    progress = self._search_progress
                    budget = (
                        "∞"
                        if progress.budget_seconds is None
                        else f"{progress.budget_seconds:.1f}"
                    )
                    pv = " ".join(progress.principal_variation[:6]) or "—"
                    speed = progress.completed_playouts / max(
                        progress.elapsed_seconds,
                        1e-9,
                    )
                    status += "・" + self.translator.tr(
                        "thinking_progress",
                        elapsed=f"{progress.elapsed_seconds:.1f}",
                        budget=budget,
                        playouts=f"{progress.completed_playouts:,}",
                        speed=f"{speed:,.0f}",
                        depth=progress.max_depth,
                        value=f"{progress.root_value:+.3f}",
                        pv=pv,
                    )
        self.status_label.setText(status)
        self.sfen_label.setText(snapshot.sfen)
        self.move_list.clear()
        for index, usi in enumerate(snapshot.move_history, start=1):
            seconds = (
                self.controller.record.move_times[index - 1]
                if index <= len(self.controller.record.move_times)
                else 0
            )
            self.move_list.addItem(f"{index:>3}.  {usi}    ({seconds}s)")
        self.move_list.scrollToBottom()
        for color, label in self.player_names.items():
            participant = self.controller.match_config.participant(color)
            name = participant.name or self.translator.player(color)
            kind_key = (
                "participant_random"
                if participant.kind is ParticipantKind.RANDOM_AI
                else (
                    "participant_heuristic"
                    if participant.kind is ParticipantKind.HEURISTIC
                    else (
                        "participant_alphazero"
                        if participant.kind is ParticipantKind.ALPHAZERO
                        else "participant_human"
                    )
                )
            )
            marker = "▶ " if snapshot.turn is color and not snapshot.outcome.is_over else ""
            kind_text = self.translator.tr(kind_key)
            if (
                participant.kind is ParticipantKind.ALPHAZERO
                and participant.model_path
            ):
                kind_text += f" · {Path(participant.model_path).name}"
            label.setText(f"{marker}{name} · {kind_text}")
            label.setStyleSheet(
                "font-weight: 700; color: #8a2f19;"
                if snapshot.turn is color and not snapshot.outcome.is_over
                else ""
            )

    def _on_search_progress(self, progress) -> None:
        self._search_progress = progress
        self._render_state(self._snapshot)

    def _on_clock_changed(self, state: ClockState) -> None:
        self._clock_state = state
        self._render_clocks(state)

    def _render_clocks(self, state: ClockState) -> None:
        control = self.controller.match_config.time_control
        for color in Color:
            if control.mode is TimeControlMode.NONE:
                self.clock_labels[color].setText(self.translator.tr("time_none"))
                text = ""
            elif control.mode is TimeControlMode.BYOYOMI:
                self.clock_labels[color].setText(
                    self._format_time(state.remaining_for(color))
                )
                text = self.translator.tr(
                    "clock_byoyomi",
                    time=self._format_time(state.overtime_for(color)),
                )
            else:
                self.clock_labels[color].setText(
                    self._format_time(state.remaining_for(color))
                )
                text = self.translator.tr(
                    "clock_increment",
                    seconds=control.for_color(color).overtime_seconds,
                )
            self.overtime_labels[color].setText(text)

    @staticmethod
    def _format_time(milliseconds: int) -> str:
        total_seconds = max(0, (milliseconds + 999) // 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return (
            f"{hours:02}:{minutes:02}:{seconds:02}"
            if hours
            else f"{minutes:02}:{seconds:02}"
        )

    def _on_record_changed(self, record: GameRecord) -> None:
        text = record.comments[-1] if record.comments else ""
        if self.comment_edit.toPlainText() != text:
            self._loading_comment = True
            self.comment_edit.setPlainText(text)
            self._loading_comment = False

    def _comment_changed(self) -> None:
        if not self._loading_comment:
            self.controller.set_last_comment(self.comment_edit.toPlainText())

    def _on_selection_changed(self, selected, drop_piece, targets) -> None:  # noqa: ANN001
        self._selected_drop = drop_piece
        self.board.set_selection(selected, set(targets))
        self._refresh_komadai()

    def _refresh_komadai(self) -> None:
        human_turn = self.controller.is_human_turn()
        for color, widget in (
            (Color.BLACK, self.black_komadai),
            (Color.WHITE, self.white_komadai),
        ):
            widget.update_state(
                self._snapshot,
                selected=(
                    self._selected_drop if self._snapshot.turn is color else None
                ),
                interactive=human_turn and self._snapshot.turn is color,
                flipped=self._flipped,
                label=self.translator.tr(
                    "komadai",
                    player=self.translator.player(color),
                ),
            )

    def _clear_pending_promotion(self) -> None:
        self.board.clear_promotion_choices()

    def _show_message(self, message: UiMessage) -> None:
        QMessageBox.warning(
            self,
            self.translator.tr("warning"),
            self.translator.message(message),
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            self.translator.tr("about"),
            self.translator.tr("about_text"),
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if (
            self._allow_close
            or not self.controller.record.dirty
            or QApplication.platformName() == "offscreen"
        ):
            self.controller.shutdown()
            event.accept()
            return
        self.controller.pause_clock()
        result = QMessageBox.question(
            self,
            self.translator.tr("unsaved_title"),
            self.translator.tr("unsaved_prompt"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if result == QMessageBox.StandardButton.Save:
            if self._save_record():
                self._allow_close = True
                self.controller.shutdown()
                event.accept()
            else:
                event.ignore()
                self.controller.resume_clock()
        elif result == QMessageBox.StandardButton.Discard:
            self._allow_close = True
            self.controller.shutdown()
            event.accept()
        else:
            event.ignore()
            self.controller.resume_clock()
