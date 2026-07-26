from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from alphasho.domain import (
    Color,
    GameMode,
    GameOutcome,
    GameRecord,
    MatchConfig,
    MoveOption,
    OutcomeReason,
    ParticipantConfig,
    ParticipantKind,
    PieceType,
    StartPositionKind,
    UiMessage,
)
from alphasho.engine import IllegalMoveError, MatchClock, ShogiGame
from alphasho.players import Player, RandomPlayer


class _PlayerTaskSignals(QObject):
    finished = Signal(int, object, str, object, object, object)
    progress = Signal(int, object, str, object)


class _PlayerTask(QRunnable):
    def __init__(
        self,
        player: Player,
        snapshot,
        legal_moves,
        generation: int,
        expected_turn: Color,
        clock_state,
        time_control,
    ) -> None:
        super().__init__()
        self.player = player
        self.snapshot = snapshot
        self.legal_moves = legal_moves
        self.generation = generation
        self.expected_turn = expected_turn
        self.clock_state = clock_state
        self.time_control = time_control
        self.signals = _PlayerTaskSignals()

    @Slot()
    def run(self) -> None:
        move = error = None
        try:
            contextual = getattr(self.player, "choose_move_with_context", None)
            if callable(contextual):
                move = contextual(
                    self.snapshot,
                    self.legal_moves,
                    clock_state=self.clock_state,
                    time_control=self.time_control,
                    color=self.expected_turn,
                    progress_callback=lambda progress: self.signals.progress.emit(
                        self.generation,
                        self.expected_turn,
                        self.snapshot.sfen,
                        progress,
                    ),
                )
            else:
                move = self.player.choose_move(self.snapshot, self.legal_moves)
        except Exception as exc:  # delivered to the GUI thread
            error = exc
        search_info = getattr(self.player, "last_search_info", None)
        self.signals.finished.emit(
            self.generation,
            self.expected_turn,
            self.snapshot.sfen,
            move,
            error,
            search_info,
        )


class GameController(QObject):
    state_changed = Signal(object)
    selection_changed = Signal(object, object, object)
    promotion_requested = Signal(object)
    clock_changed = Signal(object)
    record_changed = Signal(object)
    error_occurred = Signal(object)
    search_progress_changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.game = ShogiGame()
        self.match_config = MatchConfig()
        self.players: dict[Color, Player | None] = {
            Color.BLACK: None,
            Color.WHITE: None,
        }
        self.record = self._new_record()
        self.clock = MatchClock(self.match_config.time_control)
        self._clock_history = [self.clock.state()]
        self._selected_square: int | None = None
        self._selected_drop: PieceType | None = None
        self._selected_moves: tuple[MoveOption, ...] = ()
        self._generation = 0
        self._thread_pool = QThreadPool.globalInstance()
        self._player_tasks: set[_PlayerTask] = set()
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(100)
        self._clock_timer.timeout.connect(self._on_clock_tick)
        self._clock_timer.start()

    @property
    def mode(self) -> GameMode:
        """Compatibility view for integrations written before MatchConfig."""
        kinds = {
            self.match_config.black.kind,
            self.match_config.white.kind,
        }
        return GameMode.LOCAL if kinds == {ParticipantKind.HUMAN} else GameMode.RANDOM_AI

    def publish(self) -> None:
        self.state_changed.emit(self.game.snapshot())
        self.clock_changed.emit(self.clock.state())
        self.record_changed.emit(self.record)
        self._publish_selection()

    def is_human_turn(self) -> bool:
        snapshot = self.game.snapshot()
        return (
            not snapshot.outcome.is_over
            and self.players[snapshot.turn] is None
        )

    def is_ai_thinking(self) -> bool:
        snapshot = self.game.snapshot()
        return (
            not snapshot.outcome.is_over
            and self.players[snapshot.turn] is not None
        )

    def new_game(
        self,
        config: MatchConfig | GameMode | str | None = None,
    ) -> None:
        self._invalidate_player_tasks()
        self.match_config = self._normalize_config(config)
        self._configure_players()
        start_sfen = self.match_config.start_sfen
        self.game.reset(start_sfen)
        self.record = self._new_record()
        self.clock = MatchClock(
            self.match_config.time_control,
            turn=self.game.snapshot().turn,
        )
        self._clock_history = [self.clock.state()]
        self._clear_selection()
        self.publish()
        self.schedule_current_player()

    def load_sfen(self, sfen: str) -> None:
        config = MatchConfig(
            black=self.match_config.black,
            white=self.match_config.white,
            time_control=self.match_config.time_control,
            start_position=StartPositionKind.CURRENT,
            start_sfen=sfen.strip(),
            move_limit=self.match_config.move_limit,
        )
        self.new_game(config)

    def load_game_record(self, record: GameRecord) -> None:
        self._invalidate_player_tasks()
        self.match_config = MatchConfig(
            black=ParticipantConfig(ParticipantKind.HUMAN, record.black_name),
            white=ParticipantConfig(ParticipantKind.HUMAN, record.white_name),
            start_position=StartPositionKind.FILE,
            start_sfen=record.initial_sfen,
        )
        self._configure_players()
        self.game.reset(record.initial_sfen)
        for usi in record.moves:
            self.game.play(usi)
        if record.outcome.is_over and not self.game.snapshot().outcome.is_over:
            self.game.finish(record.outcome)
        self.record = record
        self.clock = MatchClock(
            self.match_config.time_control,
            turn=self.game.snapshot().turn,
        )
        self._clock_history = [self.clock.state()] * (len(record.moves) + 1)
        if self.game.snapshot().outcome.is_over:
            self.clock.pause()
        self._clear_selection()
        self.publish()

    def click_square(self, square: int) -> None:
        if not self.is_human_turn():
            return
        if self._selected_moves:
            candidates = tuple(
                move for move in self._selected_moves if move.to_square == square
            )
            if candidates:
                if len(candidates) == 1:
                    self.choose_move(candidates[0])
                else:
                    self.promotion_requested.emit(candidates)
                return

        snapshot = self.game.snapshot()
        piece = snapshot.piece_at(square)
        if piece is not None and piece.color is snapshot.turn:
            self._selected_square = square
            self._selected_drop = None
            self._selected_moves = self.game.legal_moves(source_square=square)
        else:
            self._clear_selection()
        self._publish_selection()

    def click_hand_piece(self, color: Color, piece_type: PieceType) -> None:
        snapshot = self.game.snapshot()
        if not self.is_human_turn() or color is not snapshot.turn:
            return
        moves = self.game.legal_moves(drop_piece=piece_type)
        if not moves:
            self.error_occurred.emit(UiMessage.create("no_legal_drop"))
            return
        self._selected_square = None
        self._selected_drop = piece_type
        self._selected_moves = moves
        self._publish_selection()

    def choose_move(self, move: MoveOption, *, search_info=None) -> None:
        self._on_clock_tick()
        before = self.game.snapshot()
        if before.outcome.is_over:
            return
        try:
            snapshot = self.game.play(move)
        except IllegalMoveError as exc:
            self.error_occurred.emit(UiMessage.create(exc.code, **exc.params))
            return
        elapsed = self.clock.complete_turn(before.turn)
        self.record.moves.append(move.usi)
        self.record.move_times.append(elapsed)
        self.record.comments.append(
            self._format_search_comment(search_info) if search_info else ""
        )
        self.record.dirty = True
        if (
            self.match_config.move_limit > 0
            and len(self.record.moves) >= self.match_config.move_limit
            and not snapshot.outcome.is_over
        ):
            snapshot = self.game.finish(GameOutcome(OutcomeReason.MOVE_LIMIT))
        self.record.outcome = snapshot.outcome
        self._clock_history.append(self.clock.state())
        if snapshot.outcome.is_over:
            self.clock.pause()
        self._clear_selection()
        self.publish()
        self.schedule_current_player()

    def undo(self) -> None:
        self._invalidate_player_tasks()
        human_count = sum(player is None for player in self.players.values())
        requested_steps = 2 if human_count == 1 else 1
        changed = False
        for _ in range(requested_steps):
            if not self.record.moves:
                break
            self.game.undo()
            self.record.moves.pop()
            if self.record.move_times:
                self.record.move_times.pop()
            if self.record.comments:
                self.record.comments.pop()
            if len(self._clock_history) > 1:
                self._clock_history.pop()
            changed = True
        if changed:
            self.record.outcome = self.game.snapshot().outcome
            self.record.dirty = True
            self.clock.restore(self._clock_history[-1])
        self._clear_selection()
        self.publish()
        self.schedule_current_player()

    def resign(self) -> None:
        self._invalidate_player_tasks()
        snapshot = self.game.resign()
        self.record.outcome = snapshot.outcome
        self.record.dirty = True
        self.clock.pause()
        self._clear_selection()
        self.publish()

    def declare_nyugyoku(self) -> None:
        try:
            snapshot = self.game.declare_nyugyoku()
        except IllegalMoveError as exc:
            self.error_occurred.emit(UiMessage.create(exc.code, **exc.params))
            return
        self._invalidate_player_tasks()
        self.record.outcome = snapshot.outcome
        self.record.dirty = True
        self.clock.pause()
        self._clear_selection()
        self.publish()

    def pause_clock(self) -> None:
        self.clock.pause()
        self.clock_changed.emit(self.clock.state())

    def resume_clock(self) -> None:
        if not self.game.snapshot().outcome.is_over:
            self.clock.resume()
            self.clock_changed.emit(self.clock.state())

    def set_last_comment(self, text: str) -> None:
        if not self.record.moves:
            return
        while len(self.record.comments) < len(self.record.moves):
            self.record.comments.append("")
        if self.record.comments[-1] != text:
            self.record.comments[-1] = text
            self.record.dirty = True
            self.record_changed.emit(self.record)

    def mark_record_saved(self, path: str) -> None:
        self.record.file_path = path
        self.record.dirty = False
        self.record_changed.emit(self.record)

    def shutdown(self) -> None:
        """Cancel searches without blocking the GUI during application exit."""

        self._invalidate_player_tasks()

    def schedule_current_player(self) -> None:
        self.search_progress_changed.emit(None)
        snapshot = self.game.snapshot()
        player = self.players[snapshot.turn]
        if snapshot.outcome.is_over or player is None:
            return
        generation = self._generation
        expected_turn = snapshot.turn
        QTimer.singleShot(
            320,
            lambda: self._run_player(generation, expected_turn),
        )

    def _run_player(self, generation: int, expected_turn: Color) -> None:
        snapshot = self.game.snapshot()
        player = self.players[snapshot.turn]
        if (
            generation != self._generation
            or snapshot.outcome.is_over
            or snapshot.turn is not expected_turn
            or player is None
        ):
            return
        legal_moves = self.game.legal_moves()
        if legal_moves:
            task = _PlayerTask(
                player,
                snapshot,
                legal_moves,
                generation,
                expected_turn,
                self.clock.state(),
                self.match_config.time_control,
            )
            self._player_tasks.add(task)
            def on_finished(
                task_generation,
                task_turn,
                task_sfen,
                move,
                error,
                search_info,
                current=task,
            ) -> None:
                self._player_finished(
                    current,
                    task_generation,
                    task_turn,
                    task_sfen,
                    move,
                    error,
                    search_info,
                )

            task.signals.finished.connect(
                on_finished
            )
            task.signals.progress.connect(self._player_progress)
            self._thread_pool.start(task)

    def _player_finished(
        self,
        task: _PlayerTask,
        generation: int,
        expected_turn: Color,
        expected_sfen: str,
        move: MoveOption | None,
        error: BaseException | None,
        search_info,
    ) -> None:
        self._player_tasks.discard(task)
        snapshot = self.game.snapshot()
        if (
            generation != self._generation
            or snapshot.outcome.is_over
            or snapshot.turn is not expected_turn
            or snapshot.sfen != expected_sfen
        ):
            return
        if error is not None:
            self.search_progress_changed.emit(None)
            if type(error).__name__ != "SearchCancelled":
                self.error_occurred.emit(
                    UiMessage.create("ai_move_failed", detail=str(error))
                )
            return
        if move is not None:
            self.choose_move(move, search_info=search_info)

    def _player_progress(
        self,
        generation: int,
        expected_turn: Color,
        expected_sfen: str,
        progress,
    ) -> None:
        snapshot = self.game.snapshot()
        if (
            generation == self._generation
            and snapshot.turn is expected_turn
            and snapshot.sfen == expected_sfen
            and not snapshot.outcome.is_over
        ):
            self.search_progress_changed.emit(progress)

    def _on_clock_tick(self) -> None:
        state = self.clock.state()
        snapshot = self.game.snapshot()
        if state.expired_color is not None and not snapshot.outcome.is_over:
            self._invalidate_player_tasks()
            outcome = GameOutcome(
                OutcomeReason.TIMEOUT,
                state.expired_color.opponent,
            )
            self.game.finish(outcome)
            self.record.outcome = outcome
            self.record.dirty = True
            self.clock.pause()
            self._clear_selection()
            self.publish()
            return
        self.clock_changed.emit(state)

    def _configure_players(self) -> None:
        self.players = {}
        for color in Color:
            participant = self.match_config.participant(color)
            if participant.kind is ParticipantKind.HUMAN:
                self.players[color] = None
            elif participant.kind is ParticipantKind.RANDOM_AI:
                self.players[color] = RandomPlayer()
            elif participant.kind is ParticipantKind.HEURISTIC:
                from alphasho.heuristicplayer import HeuristicPlayer

                self.players[color] = HeuristicPlayer(thinking=participant.thinking)
            elif participant.kind is ParticipantKind.ALPHAZERO:
                checkpoint = Path(
                    participant.model_path or self.default_model_path()
                )
                if not checkpoint.exists():
                    self.players[color] = None
                    self.error_occurred.emit(
                        UiMessage.create(
                            "alphazero_checkpoint_missing",
                            path=str(checkpoint),
                        )
                    )
                    continue
                try:
                    from alphasho.ai.player import AlphaZeroPlayer

                    self.players[color] = AlphaZeroPlayer(
                        checkpoint,
                        thinking=participant.thinking,
                    )
                except Exception as exc:
                    self.players[color] = None
                    self.error_occurred.emit(
                        UiMessage.create("alphazero_load_failed", detail=str(exc))
                    )
            else:
                self.players[color] = None

    @staticmethod
    def default_model_path() -> str:
        project_root = Path(__file__).resolve().parents[3]
        return str(project_root / "artifacts" / "checkpoints" / "best.pt")

    def _invalidate_player_tasks(self) -> None:
        for player in self.players.values():
            cancel = getattr(player, "cancel", None)
            if callable(cancel):
                cancel()
        self._generation += 1
        self.search_progress_changed.emit(None)

    @staticmethod
    def _format_search_comment(info) -> str:
        pv = " ".join(info.principal_variation[:8]) or "—"
        budget = (
            "∞"
            if info.budget_seconds is None
            else f"{info.budget_seconds:.1f}s"
        )
        return (
            f"AI search: {info.elapsed_seconds:.2f}s/{budget}, "
            f"{info.playouts} playouts, depth {info.max_depth}, "
            f"value {info.root_value:+.3f}, PV {pv}"
        )

    def _new_record(self) -> GameRecord:
        snapshot = self.game.snapshot()
        return GameRecord(
            initial_sfen=snapshot.sfen,
            black_name=self.match_config.black.name,
            white_name=self.match_config.white.name,
        )

    def _normalize_config(
        self,
        config: MatchConfig | GameMode | str | None,
    ) -> MatchConfig:
        if config is None:
            return self.match_config
        if isinstance(config, MatchConfig):
            return config
        try:
            legacy = GameMode(str(config))
        except ValueError as exc:
            raise ValueError(f"Unknown game configuration: {config}") from exc
        if legacy == GameMode.LOCAL:
            return MatchConfig()
        return MatchConfig(
            black=ParticipantConfig(ParticipantKind.HUMAN, ""),
            white=ParticipantConfig(ParticipantKind.RANDOM_AI, ""),
        )

    def _clear_selection(self) -> None:
        self._selected_square = None
        self._selected_drop = None
        self._selected_moves = ()

    def _publish_selection(self) -> None:
        targets = {move.to_square for move in self._selected_moves}
        self.selection_changed.emit(
            self._selected_square,
            self._selected_drop,
            targets,
        )
