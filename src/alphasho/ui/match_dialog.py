from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alphasho.domain import (
    MatchConfig,
    ParticipantConfig,
    ParticipantKind,
    SideTimeConfig,
    StartPositionKind,
    ThinkingConfig,
    ThinkingPreset,
    ThinkingStrategy,
    TimeControl,
    TimeControlMode,
)
from alphasho.engine import RecordError, load_record
from alphasho.ui.i18n import Translator


@dataclass(slots=True)
class _ParticipantFields:
    human: QRadioButton
    ai: QRadioButton
    ai_selector: QComboBox
    name: QLineEdit
    model_label: QLabel
    model_selector: QComboBox
    model_browse: QPushButton
    model_status: QLabel
    thinking_label: QLabel
    thinking_preset: QComboBox
    thinking_summary: QLabel
    thinking_custom: QWidget
    thinking_strategy: QComboBox
    thinking_seconds: QDoubleSpinBox
    thinking_min_playouts: QSpinBox
    thinking_max_playouts: QSpinBox
    thinking_max_depth: QSpinBox


@dataclass(slots=True)
class _TimeFields:
    widget: QWidget
    title: QLabel
    hours: QSpinBox
    minutes: QSpinBox
    overtime_title: QLabel
    overtime: QSpinBox

    def set_seconds(self, main_seconds: int, overtime_seconds: int) -> None:
        hours, remainder = divmod(max(0, main_seconds), 3600)
        self.hours.setValue(hours)
        self.minutes.setValue(remainder // 60)
        self.overtime.setValue(max(0, overtime_seconds))

    def config(self) -> SideTimeConfig:
        return SideTimeConfig(
            self.hours.value() * 3600 + self.minutes.value() * 60,
            self.overtime.value(),
        )


class MatchSettingsDialog(QDialog):
    """Shogidokoro-inspired, explicit match setup."""

    def __init__(
        self,
        translator: Translator,
        current_sfen: str,
        previous: MatchConfig,
        parent: QWidget | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.translator = translator
        self.current_sfen = current_sfen
        self.previous = previous
        self.settings = settings
        project_root = Path(__file__).resolve().parents[3]
        self.default_model_path = (
            project_root / "artifacts" / "checkpoints" / "best.pt"
        )
        self.model_library_path = project_root / "artifacts" / "models"
        self._file_sfen: str | None = (
            previous.start_sfen
            if previous.start_position is StartPositionKind.FILE
            else None
        )
        self.setObjectName("matchSettingsDialog")
        self.setModal(True)
        self.setMinimumSize(900, 660)
        self.resize(940, 700)
        self.setWindowTitle(translator.tr("match_settings"))

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        columns.addLayout(self._build_left_column(previous), 1)
        columns.addLayout(self._build_right_column(previous), 1)
        root.addLayout(columns, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            translator.tr("start_game")
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            translator.tr("cancel")
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_previous(previous)
        self._connect_controls()
        self._refresh_enabled()

    def _build_left_column(self, previous: MatchConfig) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(10)
        black_box, self.black_fields = self._participant_box(
            "black_player",
            previous.black,
        )
        white_box, self.white_fields = self._participant_box(
            "white_player",
            previous.white,
        )
        column.addWidget(black_box)
        column.addWidget(white_box)
        column.addWidget(self._start_position_box())
        column.addStretch(1)
        return column

    def _participant_box(
        self,
        title_key: str,
        config: ParticipantConfig,
    ) -> tuple[QGroupBox, _ParticipantFields]:
        box = QGroupBox(self.translator.tr(title_key))
        layout = QGridLayout(box)
        layout.setColumnStretch(2, 1)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(9)

        human = QRadioButton(self.translator.tr("participant_human"))
        ai = QRadioButton(self.translator.tr("participant_ai"))
        group = QButtonGroup(box)
        group.addButton(human)
        group.addButton(ai)

        ai_selector = QComboBox()
        ai_selector.addItem(
            self.translator.tr("participant_random"),
            ParticipantKind.RANDOM_AI.value,
        )
        ai_selector.addItem(
            self.translator.tr("participant_heuristic"),
            ParticipantKind.HEURISTIC.value,
        )
        ai_selector.addItem(
            self.translator.tr("participant_alphazero"),
            ParticipantKind.ALPHAZERO.value,
        )
        ai_selector.addItem(
            self.translator.tr("external_usi_planned"),
            ParticipantKind.USI_ENGINE.value,
        )
        model = ai_selector.model()
        if isinstance(model, QStandardItemModel):
            planned = model.item(3)
            planned.setEnabled(False)
            planned.setToolTip(self.translator.tr("planned_status"))

        name = QLineEdit(config.name)
        name.setPlaceholderText(self.translator.tr("name_placeholder"))
        model_label = QLabel(self.translator.tr("model"))
        model_selector = QComboBox()
        model_selector.setMinimumWidth(230)
        self._populate_models(model_selector, config.model_path)
        model_browse = QPushButton(self.translator.tr("browse"))
        model_status = QLabel()
        model_status.setWordWrap(True)
        thinking_label = QLabel(self.translator.tr("thinking_settings"))
        thinking_preset = QComboBox()
        for preset in ThinkingPreset:
            thinking_preset.addItem(
                self.translator.tr(f"thinking_preset_{preset.value}"),
                preset.value,
            )
        thinking_summary = QLabel()
        thinking_summary.setWordWrap(True)
        thinking_custom = QWidget()
        custom_layout = QGridLayout(thinking_custom)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        thinking_strategy = QComboBox()
        for strategy in ThinkingStrategy:
            thinking_strategy.addItem(
                self.translator.tr(f"thinking_strategy_{strategy.value}"),
                strategy.value,
            )
        thinking_seconds = QDoubleSpinBox()
        thinking_seconds.setRange(0.1, 3600.0)
        thinking_seconds.setDecimals(1)
        thinking_seconds.setSuffix(self.translator.tr("thinking_seconds_suffix"))
        thinking_min_playouts = QSpinBox()
        thinking_min_playouts.setRange(1, 1_000_000)
        thinking_max_playouts = QSpinBox()
        thinking_max_playouts.setRange(1, 10_000_000)
        thinking_max_depth = QSpinBox()
        thinking_max_depth.setRange(1, 512)
        custom_layout.addWidget(
            QLabel(self.translator.tr("thinking_strategy")), 0, 0
        )
        custom_layout.addWidget(thinking_strategy, 0, 1)
        custom_layout.addWidget(
            QLabel(self.translator.tr("thinking_time")), 1, 0
        )
        custom_layout.addWidget(thinking_seconds, 1, 1)
        custom_layout.addWidget(
            QLabel(self.translator.tr("thinking_playout_range")), 2, 0
        )
        range_layout = QHBoxLayout()
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(thinking_min_playouts)
        range_layout.addWidget(QLabel("–"))
        range_layout.addWidget(thinking_max_playouts)
        custom_layout.addLayout(range_layout, 2, 1)
        custom_layout.addWidget(
            QLabel(self.translator.tr("thinking_max_depth")), 3, 0
        )
        custom_layout.addWidget(thinking_max_depth, 3, 1)
        layout.addWidget(human, 0, 0)
        layout.addWidget(ai, 0, 1)
        layout.addWidget(ai_selector, 0, 2)
        layout.addWidget(QLabel(self.translator.tr("name")), 1, 0)
        layout.addWidget(name, 1, 1, 1, 2)
        layout.addWidget(model_label, 2, 0)
        layout.addWidget(model_selector, 2, 1)
        layout.addWidget(model_browse, 2, 2)
        layout.addWidget(model_status, 3, 0, 1, 3)
        layout.addWidget(thinking_label, 4, 0)
        layout.addWidget(thinking_preset, 4, 1, 1, 2)
        layout.addWidget(thinking_summary, 5, 0, 1, 3)
        layout.addWidget(thinking_custom, 6, 0, 1, 3)
        fields = _ParticipantFields(
            human=human,
            ai=ai,
            ai_selector=ai_selector,
            name=name,
            model_label=model_label,
            model_selector=model_selector,
            model_browse=model_browse,
            model_status=model_status,
            thinking_label=thinking_label,
            thinking_preset=thinking_preset,
            thinking_summary=thinking_summary,
            thinking_custom=thinking_custom,
            thinking_strategy=thinking_strategy,
            thinking_seconds=thinking_seconds,
            thinking_min_playouts=thinking_min_playouts,
            thinking_max_playouts=thinking_max_playouts,
            thinking_max_depth=thinking_max_depth,
        )
        model_browse.clicked.connect(
            lambda _checked=False, selected=fields: self._browse_model(selected)
        )
        thinking_preset.currentIndexChanged.connect(self._refresh_enabled)
        thinking_strategy.currentIndexChanged.connect(self._refresh_enabled)
        thinking_seconds.valueChanged.connect(self._refresh_enabled)
        thinking_min_playouts.valueChanged.connect(self._refresh_enabled)
        thinking_max_playouts.valueChanged.connect(self._refresh_enabled)
        thinking_max_depth.valueChanged.connect(self._refresh_enabled)
        return box, fields

    def _populate_models(
        self,
        selector: QComboBox,
        selected_path: str | None,
    ) -> None:
        choices = [("model_formal_best", self.default_model_path)]
        if self.model_library_path.is_dir():
            choices.extend(
                ("model_library", path)
                for path in sorted(
                    self.model_library_path.glob("*.pt"),
                    key=lambda candidate: candidate.name.casefold(),
                )
                if path.is_file()
            )
        selected_index = -1
        first_available = -1
        requested = str(Path(selected_path)) if selected_path else ""
        item_model = selector.model()
        for label_key, path in choices:
            index = selector.count()
            path_text = str(path)
            selector.addItem(
                self.translator.tr(label_key, name=path.name),
                path_text,
            )
            item = (
                item_model.item(index)
                if isinstance(item_model, QStandardItemModel)
                else None
            )
            if item is not None:
                item.setToolTip(path_text)
                item.setEnabled(path.is_file())
            if path.is_file() and first_available < 0:
                first_available = index
            if requested and Path(requested) == path:
                selected_index = index
        if requested and selected_index < 0:
            selected_index = selector.count()
            selector.addItem(
                self.translator.tr(
                    "model_custom",
                    name=Path(requested).name,
                ),
                requested,
            )
            selector.setItemData(
                selected_index,
                requested,
                Qt.ItemDataRole.ToolTipRole,
            )
        selector.setCurrentIndex(
            selected_index
            if selected_index >= 0
            else max(0, first_available)
        )

    def _start_position_box(self) -> QGroupBox:
        box = QGroupBox(self.translator.tr("start_position"))
        layout = QGridLayout(box)
        self.start_initial = QRadioButton(self.translator.tr("start_initial"))
        self.start_current = QRadioButton(self.translator.tr("start_current"))
        self.start_file = QRadioButton(self.translator.tr("start_file"))
        self.start_group = QButtonGroup(box)
        for button in (self.start_initial, self.start_current, self.start_file):
            self.start_group.addButton(button)
        self.start_path = QLineEdit()
        self.start_path.setReadOnly(True)
        self.start_path.setPlaceholderText(
            self.translator.tr("position_path_placeholder")
        )
        self.browse_button = QPushButton(self.translator.tr("browse"))
        self.browse_button.clicked.connect(self._browse_position)
        layout.addWidget(self.start_initial, 0, 0, 1, 2)
        layout.addWidget(self.start_current, 1, 0, 1, 2)
        layout.addWidget(self.start_file, 2, 0, 1, 2)
        layout.addWidget(self.start_path, 3, 0)
        layout.addWidget(self.browse_button, 3, 1)
        layout.setColumnStretch(0, 1)
        return box

    def _build_right_column(self, previous: MatchConfig) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(10)
        column.addWidget(self._time_box(previous), 1)
        column.addWidget(self._match_options_box())
        return column

    def _time_box(self, previous: MatchConfig) -> QGroupBox:
        box = QGroupBox(self.translator.tr("time_settings"))
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        self.time_none = QRadioButton(self.translator.tr("time_none"))
        self.time_byoyomi = QRadioButton(self.translator.tr("time_byoyomi"))
        self.time_fischer = QRadioButton(self.translator.tr("time_fischer"))
        self.time_group = QButtonGroup(box)
        for button in (self.time_none, self.time_byoyomi, self.time_fischer):
            self.time_group.addButton(button)
            layout.addWidget(button)

        self.shared_time = QCheckBox(self.translator.tr("shared_time"))
        layout.addWidget(self.shared_time)

        self.common_time = self._time_fields("common_time")
        self.black_time = self._time_fields("black_time")
        self.white_time = self._time_fields("white_time")
        layout.addWidget(self.common_time.widget)
        layout.addWidget(self.black_time.widget)
        layout.addWidget(self.white_time.widget)
        layout.addStretch(1)
        return box

    def _time_fields(self, title_key: str) -> _TimeFields:
        panel = QWidget()
        panel.setObjectName("timeFields")
        layout = QGridLayout(panel)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(6)

        title = QLabel(self.translator.tr(title_key))
        title.setObjectName("timeFieldsTitle")
        hours = QSpinBox()
        hours.setRange(0, 99)
        hours.setAlignment(Qt.AlignmentFlag.AlignRight)
        hours.setMinimumWidth(68)
        minutes = QSpinBox()
        minutes.setRange(0, 59)
        minutes.setAlignment(Qt.AlignmentFlag.AlignRight)
        minutes.setMinimumWidth(68)
        overtime_title = QLabel()
        overtime = QSpinBox()
        overtime.setRange(0, 3600)
        overtime.setAlignment(Qt.AlignmentFlag.AlignRight)
        overtime.setMinimumWidth(78)

        layout.addWidget(title, 0, 0, 1, 6)
        layout.addWidget(QLabel(self.translator.tr("main_time")), 1, 0)
        layout.addWidget(hours, 1, 1)
        layout.addWidget(QLabel(self.translator.tr("hours_unit")), 1, 2)
        layout.addWidget(minutes, 1, 3)
        layout.addWidget(QLabel(self.translator.tr("minutes_unit")), 1, 4)
        layout.addWidget(overtime_title, 2, 0)
        layout.addWidget(overtime, 2, 1)
        layout.addWidget(QLabel(self.translator.tr("seconds_unit")), 2, 2)
        layout.setColumnStretch(5, 1)
        return _TimeFields(
            panel,
            title,
            hours,
            minutes,
            overtime_title,
            overtime,
        )

    def _match_options_box(self) -> QGroupBox:
        box = QGroupBox(self.translator.tr("match_options"))
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        limit_row = QHBoxLayout()
        self.move_limit_enabled = QCheckBox(self.translator.tr("draw_at"))
        self.move_limit = QSpinBox()
        self.move_limit.setRange(1, 10000)
        self.move_limit.setValue(300)
        self.move_limit.setMinimumWidth(85)
        limit_row.addWidget(self.move_limit_enabled)
        limit_row.addWidget(self.move_limit)
        limit_row.addWidget(QLabel(self.translator.tr("draw_at_suffix")))
        limit_row.addStretch(1)
        layout.addLayout(limit_row)

        self.time_forfeit = QCheckBox(self.translator.tr("time_forfeit"))
        layout.addWidget(self.time_forfeit)
        self.continuous_games = QCheckBox(
            self.translator.tr("continuous_games_planned")
        )
        self.continuous_games.setEnabled(False)
        self.continuous_games.setToolTip(self.translator.tr("planned_status"))
        layout.addWidget(self.continuous_games)
        self.auto_save = QCheckBox(self.translator.tr("auto_save_planned"))
        self.auto_save.setEnabled(False)
        self.auto_save.setToolTip(self.translator.tr("planned_status"))
        layout.addWidget(self.auto_save)
        return box

    def _connect_controls(self) -> None:
        for fields in (self.black_fields, self.white_fields):
            fields.human.toggled.connect(self._refresh_enabled)
            fields.ai.toggled.connect(self._refresh_enabled)
            fields.ai_selector.currentIndexChanged.connect(
                self._refresh_enabled
            )
            fields.model_selector.currentIndexChanged.connect(
                self._refresh_enabled
            )
        for button in (self.start_initial, self.start_current, self.start_file):
            button.toggled.connect(self._refresh_enabled)
        for button in (self.time_none, self.time_byoyomi, self.time_fischer):
            button.toggled.connect(self._refresh_enabled)
        self.shared_time.toggled.connect(self._refresh_enabled)
        self.move_limit_enabled.toggled.connect(self._refresh_enabled)

    def _load_previous(self, config: MatchConfig) -> None:
        for side, fields, participant in (
            ("black", self.black_fields, config.black),
            ("white", self.white_fields, config.white),
        ):
            is_human = participant.kind is ParticipantKind.HUMAN
            fields.human.setChecked(is_human)
            fields.ai.setChecked(not is_human)
            index = fields.ai_selector.findData(participant.kind.value)
            fields.ai_selector.setCurrentIndex(max(0, index))
            self._load_thinking(fields, participant.thinking, side)

        start_buttons = {
            StartPositionKind.INITIAL: self.start_initial,
            StartPositionKind.CURRENT: self.start_current,
            StartPositionKind.FILE: self.start_file,
        }
        start_buttons[config.start_position].setChecked(True)

        time_buttons = {
            TimeControlMode.NONE: self.time_none,
            TimeControlMode.BYOYOMI: self.time_byoyomi,
            TimeControlMode.FISCHER: self.time_fischer,
        }
        time_buttons[config.time_control.mode].setChecked(True)
        same_time = config.time_control.black == config.time_control.white
        self.shared_time.setChecked(same_time)
        self.common_time.set_seconds(
            config.time_control.black.main_seconds,
            config.time_control.black.overtime_seconds,
        )
        self.black_time.set_seconds(
            config.time_control.black.main_seconds,
            config.time_control.black.overtime_seconds,
        )
        self.white_time.set_seconds(
            config.time_control.white.main_seconds,
            config.time_control.white.overtime_seconds,
        )
        self.time_forfeit.setChecked(config.time_control.time_forfeit)
        self.move_limit_enabled.setChecked(config.move_limit > 0)
        if config.move_limit > 0:
            self.move_limit.setValue(config.move_limit)

    def _refresh_enabled(self) -> None:
        for fields in (self.black_fields, self.white_fields):
            fields.ai_selector.setEnabled(fields.ai.isChecked())
            is_alphazero = (
                fields.ai.isChecked()
                and fields.ai_selector.currentData()
                == ParticipantKind.ALPHAZERO.value
            )
            is_search_ai = (
                fields.ai.isChecked()
                and fields.ai_selector.currentData()
                in {
                    ParticipantKind.ALPHAZERO.value,
                    ParticipantKind.HEURISTIC.value,
                }
            )
            for widget in (
                fields.model_label,
                fields.model_selector,
                fields.model_browse,
                fields.model_status,
            ):
                widget.setVisible(is_alphazero)
            for widget in (
                fields.thinking_label,
                fields.thinking_preset,
                fields.thinking_summary,
            ):
                widget.setVisible(is_search_ai)
            fields.model_selector.setEnabled(is_alphazero)
            fields.model_browse.setEnabled(is_alphazero)
            is_custom = (
                fields.thinking_preset.currentData()
                == ThinkingPreset.CUSTOM.value
            )
            fields.thinking_custom.setVisible(is_search_ai and is_custom)
            fields.thinking_custom.setEnabled(is_search_ai and is_custom)
            self._update_thinking_summary(fields)
            self._update_model_status(fields)

        file_selected = self.start_file.isChecked()
        self.start_path.setEnabled(file_selected)
        self.browse_button.setEnabled(file_selected)

        timed = not self.time_none.isChecked()
        shared = self.shared_time.isChecked()
        self.shared_time.setEnabled(timed)
        self.common_time.widget.setVisible(timed and shared)
        self.black_time.widget.setVisible(timed and not shared)
        self.white_time.widget.setVisible(timed and not shared)
        self.time_forfeit.setEnabled(timed)

        overtime_key = (
            "byoyomi_seconds"
            if self.time_byoyomi.isChecked()
            else "increment_seconds"
        )
        overtime_text = self.translator.tr(overtime_key)
        for fields in (self.common_time, self.black_time, self.white_time):
            fields.overtime_title.setText(overtime_text)

        self.move_limit.setEnabled(self.move_limit_enabled.isChecked())

    def match_config(self) -> MatchConfig:
        start_kind = (
            StartPositionKind.INITIAL
            if self.start_initial.isChecked()
            else (
                StartPositionKind.CURRENT
                if self.start_current.isChecked()
                else StartPositionKind.FILE
            )
        )
        start_sfen = {
            StartPositionKind.INITIAL: None,
            StartPositionKind.CURRENT: self.current_sfen,
            StartPositionKind.FILE: self._file_sfen,
        }[start_kind]
        mode = (
            TimeControlMode.NONE
            if self.time_none.isChecked()
            else (
                TimeControlMode.BYOYOMI
                if self.time_byoyomi.isChecked()
                else TimeControlMode.FISCHER
            )
        )
        black_time = (
            self.common_time.config()
            if self.shared_time.isChecked()
            else self.black_time.config()
        )
        white_time = (
            self.common_time.config()
            if self.shared_time.isChecked()
            else self.white_time.config()
        )
        return MatchConfig(
            black=self._participant_config(self.black_fields),
            white=self._participant_config(self.white_fields),
            time_control=TimeControl(
                mode,
                black_time,
                white_time,
                self.time_forfeit.isChecked(),
            ),
            start_position=start_kind,
            start_sfen=start_sfen,
            move_limit=(
                self.move_limit.value()
                if self.move_limit_enabled.isChecked()
                else 0
            ),
        )

    def _participant_config(self, fields: _ParticipantFields) -> ParticipantConfig:
        kind = (
            ParticipantKind.HUMAN
            if fields.human.isChecked()
            else ParticipantKind(str(fields.ai_selector.currentData()))
        )
        model_path = (
            str(fields.model_selector.currentData() or "")
            if kind is ParticipantKind.ALPHAZERO
            else None
        )
        return ParticipantConfig(
            kind=kind,
            name=fields.name.text().strip(),
            model_path=model_path,
            thinking=self._thinking_config(fields),
        )

    def _load_thinking(
        self,
        fields: _ParticipantFields,
        config: ThinkingConfig,
        side: str,
    ) -> None:
        if self.settings is not None:
            prefix = f"thinking/{side}/"
            try:
                preset = ThinkingPreset(
                    str(self.settings.value(prefix + "preset", config.preset.value))
                )
                strategy = ThinkingStrategy(
                    str(
                        self.settings.value(
                            prefix + "strategy",
                            config.strategy.value,
                        )
                    )
                )
                config = ThinkingConfig(
                    strategy=strategy,
                    preset=preset,
                    move_time_seconds=float(
                        self.settings.value(
                            prefix + "seconds",
                            config.move_time_seconds,
                        )
                    ),
                    min_playouts=int(
                        self.settings.value(
                            prefix + "min_playouts",
                            config.min_playouts,
                        )
                    ),
                    max_playouts=int(
                        self.settings.value(
                            prefix + "max_playouts",
                            config.max_playouts,
                        )
                    ),
                    max_depth=int(
                        self.settings.value(
                            prefix + "max_depth",
                            config.max_depth,
                        )
                    ),
                    safety_margin_ms=config.safety_margin_ms,
                )
            except (TypeError, ValueError):
                pass
        fields.thinking_preset.setCurrentIndex(
            max(0, fields.thinking_preset.findData(config.preset.value))
        )
        fields.thinking_strategy.setCurrentIndex(
            max(0, fields.thinking_strategy.findData(config.strategy.value))
        )
        fields.thinking_seconds.setValue(config.move_time_seconds)
        fields.thinking_min_playouts.setValue(config.min_playouts)
        fields.thinking_max_playouts.setValue(config.max_playouts)
        fields.thinking_max_depth.setValue(config.max_depth)

    def _thinking_config(self, fields: _ParticipantFields) -> ThinkingConfig:
        preset = ThinkingPreset(str(fields.thinking_preset.currentData()))
        if preset is not ThinkingPreset.CUSTOM:
            return ThinkingConfig.for_preset(preset)
        return ThinkingConfig(
            strategy=ThinkingStrategy(str(fields.thinking_strategy.currentData())),
            preset=preset,
            move_time_seconds=fields.thinking_seconds.value(),
            min_playouts=fields.thinking_min_playouts.value(),
            max_playouts=fields.thinking_max_playouts.value(),
            max_depth=fields.thinking_max_depth.value(),
        )

    def _update_thinking_summary(self, fields: _ParticipantFields) -> None:
        config = self._thinking_config(fields)
        strategy = self.translator.tr(
            f"thinking_strategy_{config.strategy.value}"
        )
        if config.strategy is ThinkingStrategy.AUTO_TIME:
            key = "thinking_summary_auto"
        elif config.strategy is ThinkingStrategy.FIXED_TIME:
            key = "thinking_summary_fixed_time"
        else:
            key = "thinking_summary_fixed_playout"
        fields.thinking_summary.setText(
            self.translator.tr(
                key,
                strategy=strategy,
                seconds=f"{config.move_time_seconds:g}",
                minimum=f"{config.min_playouts:,}",
                maximum=f"{config.max_playouts:,}",
                depth=config.max_depth,
            )
        )

    def _save_thinking_settings(self, config: MatchConfig) -> None:
        if self.settings is None:
            return
        for side, participant in (
            ("black", config.black),
            ("white", config.white),
        ):
            thinking = participant.thinking
            prefix = f"thinking/{side}/"
            self.settings.setValue(prefix + "preset", thinking.preset.value)
            self.settings.setValue(prefix + "strategy", thinking.strategy.value)
            self.settings.setValue(prefix + "seconds", thinking.move_time_seconds)
            self.settings.setValue(prefix + "min_playouts", thinking.min_playouts)
            self.settings.setValue(prefix + "max_playouts", thinking.max_playouts)
            self.settings.setValue(prefix + "max_depth", thinking.max_depth)
        self.settings.sync()

    def _browse_model(self, fields: _ParticipantFields) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            self.translator.tr("select_model_file"),
            "",
            self.translator.tr("model_file_filter"),
        )
        if not path:
            return
        index = fields.model_selector.findData(path)
        if index < 0:
            index = fields.model_selector.count()
            fields.model_selector.addItem(
                self.translator.tr(
                    "model_custom",
                    name=Path(path).name,
                ),
                path,
            )
            fields.model_selector.setItemData(
                index,
                path,
                Qt.ItemDataRole.ToolTipRole,
            )
        fields.model_selector.setCurrentIndex(index)
        self._update_model_status(fields)

    def _update_model_status(self, fields: _ParticipantFields) -> None:
        path = str(fields.model_selector.currentData() or "")
        if path and Path(path).is_file():
            fields.model_status.setText(
                self.translator.tr("model_ready", path=path)
            )
            fields.model_status.setStyleSheet("color: #386641;")
        else:
            fields.model_status.setText(
                self.translator.tr("model_missing", path=path or "—")
            )
            fields.model_status.setStyleSheet("color: #a33a2b;")

    def _browse_position(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            self.translator.tr("select_position_file"),
            "",
            self.translator.tr("position_file_filter"),
        )
        if not path:
            return
        try:
            suffix = Path(path).suffix.lower()
            if suffix in {".kif", ".kifu", ".csa"}:
                sfen = load_record(path).initial_sfen
            else:
                sfen = Path(path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError, RecordError) as exc:
            QMessageBox.warning(
                self,
                self.translator.tr("warning"),
                self.translator.tr("position_file_failed", detail=str(exc)),
            )
            return
        self._file_sfen = sfen
        self.start_path.setText(path)

    def _accept_if_valid(self) -> None:
        if self.start_file.isChecked() and not self._file_sfen:
            QMessageBox.warning(
                self,
                self.translator.tr("warning"),
                self.translator.tr("select_position_required"),
            )
            return
        config = self.match_config()
        for participant in (config.black, config.white):
            if (
                participant.kind in {
                    ParticipantKind.ALPHAZERO,
                    ParticipantKind.HEURISTIC,
                }
                and participant.thinking.min_playouts
                > participant.thinking.max_playouts
            ):
                QMessageBox.warning(
                    self,
                    self.translator.tr("warning"),
                    self.translator.tr("invalid_thinking_playouts"),
                )
                return
        for participant in (config.black, config.white):
            if participant.kind is ParticipantKind.ALPHAZERO:
                checkpoint = Path(participant.model_path or "")
                if not checkpoint.is_file():
                    QMessageBox.warning(
                        self,
                        self.translator.tr("warning"),
                        self.translator.tr(
                            "alphazero_checkpoint_missing",
                            path=str(checkpoint),
                        ),
                    )
                    return
        if config.time_control.mode is not TimeControlMode.NONE:
            for color in (config.time_control.black, config.time_control.white):
                if color.main_seconds <= 0 and color.overtime_seconds <= 0:
                    QMessageBox.warning(
                        self,
                        self.translator.tr("warning"),
                        self.translator.tr("invalid_time_control"),
                    )
                    return
        self._save_thinking_settings(config)
        self.accept()
