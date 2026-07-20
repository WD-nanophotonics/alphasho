"""KIF/CSA game-record import and export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cshogi
from cshogi import CSA, KIF

from alphasho.atomic_io import replace_with_retry
from alphasho.domain import Color, GameOutcome, GameRecord, OutcomeReason
from alphasho.engine.game import IllegalMoveError, InvalidPositionError, ShogiGame


class RecordError(ValueError):
    def __init__(self, code: str, **params: object) -> None:
        self.code = code
        self.params = params
        super().__init__(code)


def load_record(path: str | Path) -> GameRecord:
    source = Path(path)
    suffix = source.suffix.lower()
    try:
        if suffix in {".kif", ".kifu"}:
            parsed = KIF.Parser.parse_file(str(source))
        elif suffix == ".csa":
            games = CSA.Parser.parse_file(str(source))
            if not games:
                raise RecordError("empty_record")
            parsed = games[0]
        else:
            raise RecordError("unsupported_record_format", extension=suffix or "?")
    except RecordError:
        raise
    except (OSError, UnicodeError, RuntimeError, ValueError) as exc:
        raise RecordError("record_open_failed", detail=str(exc)) from exc

    try:
        game = ShogiGame(parsed.sfen)
        moves: list[str] = []
        for native_move in parsed.moves:
            usi = cshogi.move_to_usi(native_move)
            game.play(usi)
            moves.append(usi)
    except (InvalidPositionError, IllegalMoveError, RuntimeError, ValueError) as exc:
        raise RecordError("invalid_record_move", detail=str(exc)) from exc

    names = list(parsed.names or ())
    names.extend(["", ""])
    raw_times = list(parsed.times or ())
    times = [max(0, int(value)) for value in raw_times[: len(moves)]]
    times.extend([0] * (len(moves) - len(times)))
    raw_comments = list(parsed.comments or ())
    comments = [str(value or "") for value in raw_comments[: len(moves)]]
    comments.extend([""] * (len(moves) - len(comments)))
    outcome = game.snapshot().outcome
    if not outcome.is_over:
        outcome = _outcome_from_parser(
            getattr(parsed, "endgame", None),
            getattr(parsed, "win", None),
            game.snapshot().turn,
        )
    return GameRecord(
        initial_sfen=str(parsed.sfen),
        black_name=str(names[0] or ""),
        white_name=str(names[1] or ""),
        moves=moves,
        move_times=times,
        comments=comments,
        outcome=outcome,
        file_path=str(source),
        dirty=False,
    )


def save_record(record: GameRecord, path: str | Path) -> None:
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".kif", ".kifu", ".csa"}:
        raise RecordError("unsupported_record_format", extension=suffix or "?")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        if suffix in {".kif", ".kifu"}:
            _save_kif(record, temporary)
        else:
            _save_csa(record, temporary)
        replace_with_retry(temporary, destination)
    except RecordError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, UnicodeError, RuntimeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise RecordError("record_save_failed", detail=str(exc)) from exc


def _replay_native(record: GameRecord):
    board = cshogi.Board(record.initial_sfen)
    for index, usi in enumerate(record.moves):
        legal = {
            cshogi.move_to_usi(native): native for native in board.legal_moves
        }
        native = legal.get(usi)
        if native is None:
            raise RecordError("invalid_record_move", detail=f"{index + 1}: {usi}")
        yield board, native, index
        board.push(native)


def _save_kif(record: GameRecord, path: Path) -> None:
    exporter = KIF.Exporter(str(path))
    totals = [0, 0]
    try:
        exporter.header(
            [record.black_name, record.white_name],
            handicap=f"sfen {record.initial_sfen}",
        )
        for board, native, index in _replay_native(record):
            color = int(board.turn)
            seconds = _move_time(record, index)
            totals[color] += seconds
            exporter.move(native, seconds, totals[color])
            comment = _comment(record, index)
            if comment:
                for line in comment.splitlines():
                    exporter.kifu.write(f"*{line}\n")
        reason = _kif_end_reason(record.outcome)
        if reason:
            exporter.end(reason)
    finally:
        exporter.close()


def _save_csa(record: GameRecord, path: Path) -> None:
    exporter = CSA.Exporter(str(path), encoding="utf-8")
    try:
        exporter.info(
            cshogi.Board(record.initial_sfen),
            [record.black_name, record.white_name],
            version="V2.2",
        )
        for _board, native, index in _replay_native(record):
            exporter.move(
                native,
                _move_time(record, index),
                _comment(record, index) or None,
            )
        endgame = _csa_endgame(record.outcome)
        if endgame:
            exporter.endgame(endgame)
    finally:
        exporter.close()


def _move_time(record: GameRecord, index: int) -> int:
    return max(0, int(record.move_times[index])) if index < len(record.move_times) else 0


def _comment(record: GameRecord, index: int) -> str:
    return record.comments[index] if index < len(record.comments) else ""


def _outcome_from_parser(
    endgame: str | None,
    win: int | None,
    turn: Color,
) -> GameOutcome:
    if not endgame:
        return GameOutcome()
    winner = None
    if win == cshogi.BLACK_WIN:
        winner = Color.BLACK
    elif win == cshogi.WHITE_WIN:
        winner = Color.WHITE
    if endgame == "%SENNICHITE":
        return GameOutcome(OutcomeReason.REPETITION)
    if endgame == "%KACHI":
        return GameOutcome(OutcomeReason.NYUGYOKU, winner or turn)
    if endgame == "%TIME_UP":
        return GameOutcome(OutcomeReason.TIMEOUT, winner or turn.opponent)
    if endgame in {"%TORYO", "%TSUMI"}:
        return GameOutcome(OutcomeReason.RESIGNATION, winner or turn.opponent)
    return GameOutcome()


def _kif_end_reason(outcome: GameOutcome) -> str | None:
    return {
        OutcomeReason.RESIGNATION: "resign",
        OutcomeReason.CHECKMATE: "resign",
        OutcomeReason.NYUGYOKU: "win",
        OutcomeReason.REPETITION: "sennichite",
        OutcomeReason.PERPETUAL_CHECK: "sennichite",
        OutcomeReason.SUPERIOR_REPETITION: "sennichite",
        OutcomeReason.MOVE_LIMIT: "draw",
        OutcomeReason.TIMEOUT: "illegal_lose",
    }.get(outcome.reason)


def _csa_endgame(outcome: GameOutcome) -> str | None:
    return {
        OutcomeReason.RESIGNATION: "%TORYO",
        OutcomeReason.CHECKMATE: "%TSUMI",
        OutcomeReason.NYUGYOKU: "%KACHI",
        OutcomeReason.REPETITION: "%SENNICHITE",
        OutcomeReason.PERPETUAL_CHECK: "%SENNICHITE",
        OutcomeReason.SUPERIOR_REPETITION: "%SENNICHITE",
        OutcomeReason.MOVE_LIMIT: "%JISHOGI",
        OutcomeReason.TIMEOUT: "%TIME_UP",
    }.get(outcome.reason)
