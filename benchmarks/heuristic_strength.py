"""Paired, auditable ABP benchmark with no synthetic draw adjudication.

The default run is intentionally long: ten fixed openings, both colors, one
second per move, and up to 256 plies. Use ``--positions 1 --max-plies 8`` for a
smoke run. Unfinished games are reported as ``unresolved`` and never scored.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import cshogi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from alphasho.domain import (  # noqa: E402
    ClockState,
    Color,
    GameOutcome,
    ThinkingConfig,
    ThinkingPreset,
    ThinkingStrategy,
    TimeControl,
)
from alphasho.engine import ShogiGame  # noqa: E402
from alphasho.heuristicplayer import HeuristicPlayer  # noqa: E402
from alphasho.heuristicplayer.evaluation import (  # noqa: E402
    EvaluationMode,
    evaluate_breakdown,
)
from alphasho.heuristicplayer.evaluator import HandcraftedEvaluator  # noqa: E402
from alphasho.heuristicplayer.search import SearchTuning  # noqa: E402
from benchmarks.legacy_3262cc8 import (  # noqa: E402
    SOURCE_COMMIT,
    LegacyHeuristicPlayer,
)

DEFAULT_OPENINGS = ROOT / "configs" / "training" / "evaluation_positions.jsonl"
FINGERPRINT_FILES = (
    ROOT / "benchmarks" / "heuristic_strength.py",
    ROOT / "benchmarks" / "legacy_3262cc8.py",
    ROOT / "benchmarks" / "yaneuraou_usi.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "evaluation.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "evaluator.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "backend.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "move_ordering.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "mate_search.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "player.py",
    ROOT / "src" / "alphasho" / "heuristicplayer" / "search.py",
    ROOT / "src" / "alphasho" / "search" / "reporting.py",
)
RESUME_COMPARABLE_FIELDS = (
    "opening_names",
    "profiles",
    "seconds_per_move",
    "max_plies",
    "legacy_source_commit",
    "implementation_fingerprint",
    "process_priority_requested",
    "evaluator_fingerprint",
)


class AblationProfile(StrEnum):
    LEGACY = "legacy"
    TT_PV = "tt_pv"
    Q_LITE = "q_lite"
    POSITIONAL_LITE = "positional_lite"
    KING_SAFETY_LITE = "king_safety_lite"
    FULL_SCAN = "full_scan"
    INCREMENTAL_FAST = "incremental_fast"
    LAZY_FULL = "lazy_full"
    FULL = "full"
    INCREMENTAL_BASELINE = "incremental_baseline"
    ROOT_TACTICAL = "root_tactical"
    PVS_ASPIRATION = "pvs_aspiration"
    SEE_MOVEPICKER = "see_movepicker"
    LMR = "lmr"
    NULL_MOVE = "null_move"
    SHALLOW_PRUNING = "shallow_pruning"
    FULL_CANDIDATE = "full_candidate"
    HOTPATH_BASELINE = "hotpath_baseline"
    PVS_HOTPATH = "pvs_hotpath"
    SEE_HOTPATH = "see_hotpath"
    LMR_HOTPATH = "lmr_hotpath"
    MATE_PROBE = "mate_probe"
    NEXT_FULL_CANDIDATE = "next_full_candidate"


@dataclass(frozen=True, slots=True)
class Opening:
    name: str
    sfen: str


@dataclass(frozen=True, slots=True)
class MoveTrace:
    ply: int
    player: str
    color: str
    sfen: str
    history: tuple[str, ...]
    usi: str
    elapsed_seconds: float
    completed_depth: int
    completed_root_moves: int
    used_fallback: bool
    main_nodes: int
    quiescence_nodes: int
    quiescence_truncations: int
    tt_hits: int
    root_value: float
    principal_variation: tuple[str, ...]
    evaluation: dict[str, int]
    mate_in_one: str | None
    tactical_moves: tuple[str, ...]
    mate_in_one_moves: tuple[str, ...] = ()
    selected_mate_in_one: bool = False
    attempted_depth: int = 0
    fallback_reason: str | None = None
    deadline_overshoot_ms: float = 0.0
    q_depth_truncations: int = 0
    q_leaf_truncations: int = 0
    q_budget_truncations: int = 0
    q_evasion_nodes: int = 0
    move_ordering_seconds: float = 0.0
    legal_generation_seconds: float = 0.0
    checking_probe_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    evaluation_calls: int = 0
    ordering_calls: int = 0
    ordered_moves: int = 0
    search_elapsed_seconds: float = 0.0
    wall_overshoot_ms: float = 0.0
    setup_elapsed_seconds: float = 0.0
    position_history_plies: int = 0
    timeout_stage: str | None = None
    timing_invalid: bool = False
    incremental_evaluation_calls: int = 0
    full_evaluation_calls: int = 0
    lazy_beta_skips: int = 0
    lazy_alpha_skips: int = 0
    cpu_elapsed_seconds: float = 0.0
    cpu_evaluation_seconds: float = 0.0
    cpu_ordering_seconds: float = 0.0
    cpu_see_seconds: float = 0.0
    cpu_tt_seconds: float = 0.0
    cpu_search_overhead_seconds: float = 0.0
    scheduler_wait_seconds: float = 0.0
    timing_cause: str | None = None
    classification_seconds: float = 0.0
    sorting_seconds: float = 0.0
    pvs_null_window_searches: int = 0
    pvs_researches: int = 0
    aspiration_fail_low: int = 0
    aspiration_fail_high: int = 0
    aspiration_researches: int = 0
    see_queries: int = 0
    see_prunes: int = 0
    lmr_reductions: int = 0
    lmr_researches: int = 0
    null_move_attempts: int = 0
    null_move_cutoffs: int = 0
    null_move_verifications: int = 0
    razor_prunes: int = 0
    reverse_futility_prunes: int = 0
    move_picker_stage_counts: dict[str, int] = field(default_factory=dict)
    move_picker_stage_seconds: dict[str, float] = field(default_factory=dict)
    move_picker_generated: int = 0
    move_picker_yielded: int = 0
    move_picker_yielded_by_stage: dict[str, int] = field(default_factory=dict)
    full_evaluation_bound_skips: int = 0
    occupied_piece_iterations: int = 0
    mate_status: str | None = None
    mate_nodes: int = 0
    mate_elapsed_seconds: float = 0.0
    mate_principal_variation: tuple[str, ...] = ()


@dataclass(slots=True)
class ProcessPriority:
    requested: str = "above_normal"
    applied: bool = False
    original: int | None = None
    error: str | None = None
    restored: bool = False

    def apply(self) -> None:
        if sys.platform != "win32":
            self.error = "unsupported_platform"
            return
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            get_process = kernel32.GetCurrentProcess
            get_priority = kernel32.GetPriorityClass
            set_priority = kernel32.SetPriorityClass
            if hasattr(get_process, "restype"):
                get_process.restype = ctypes.c_void_p
                get_priority.argtypes = [ctypes.c_void_p]
                get_priority.restype = ctypes.c_uint32
                set_priority.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                set_priority.restype = ctypes.c_int
            process = get_process()
            self.original = int(get_priority(process))
            if self.original == 0:
                raise OSError(ctypes.get_last_error(), "GetPriorityClass failed")
            if not set_priority(process, 0x00008000):
                raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
            self.applied = True
        except Exception as exc:  # priority is best-effort audit hygiene
            self.error = f"{type(exc).__name__}: {exc}"

    def restore(self) -> None:
        if not self.applied or self.original is None or sys.platform != "win32":
            return
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            get_process = kernel32.GetCurrentProcess
            set_priority = kernel32.SetPriorityClass
            if hasattr(get_process, "restype"):
                get_process.restype = ctypes.c_void_p
                set_priority.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                set_priority.restype = ctypes.c_int
            process = get_process()
            if not set_priority(process, self.original):
                raise OSError(ctypes.get_last_error(), "priority restore failed")
        except Exception as exc:
            self.error = f"restore {type(exc).__name__}: {exc}"
        finally:
            self.restored = True


@dataclass(slots=True)
class GameResult:
    opening: str
    initial_sfen: str
    candidate_profile: str
    candidate_color: str
    outcome_kind: str
    outcome_reason: str
    winner: str | None
    played_plies: int
    final_sfen: str
    final_evaluation: dict[str, int]
    final_candidate_value: int
    provisional_leader: str
    fallback_seen: bool
    eligible_for_strength: bool
    candidate_score: float | None
    timing_invalid_seen: bool = False
    timing_violations: int = 0
    moves: list[MoveTrace] = field(default_factory=list)
    diagnosis: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    candidate_profile: str
    legacy_source_commit: str
    seconds_per_move: float
    max_plies: int
    openings: int
    games: int
    candidate_wins: int
    legacy_wins: int
    rule_draws: int
    unresolved: int
    fallback_games: int
    eligible_games: int
    candidate_score: float | None
    confidence_low: float | None
    confidence_high: float | None
    superiority_claim_supported: bool
    elapsed_seconds: float
    stop_reason: str | None = None
    timing_invalid_games: int = 0
    by_color: dict[str, dict[str, int]] = field(default_factory=dict)
    paired_results: dict[str, int | float | None] = field(default_factory=dict)
    search_metrics: dict[str, int | float | None] = field(default_factory=dict)


class AuditReporter:
    """Durable event stream plus a concise real-time console view."""

    def __init__(self, run_dir: Path, *, heartbeat_plies: int = 1) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_plies = max(1, heartbeat_plies)
        self.events_path = self.run_dir / "events.jsonl"
        self.games_path = self.run_dir / "games.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.console_path = self.run_dir / "console.log"
        self._events = self.events_path.open("a", encoding="utf-8", newline="\n")
        self._console = self.console_path.open("a", encoding="utf-8", newline="\n")
        self.last_failure: dict[str, object] | None = None

    def close(self) -> None:
        self._events.close()
        self._console.close()

    def emit(self, event: str, message: str, **values: object) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        record = {"timestamp": timestamp, "event": event, **values}
        if event == "game_failed":
            self.last_failure = record
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._console.write(line + "\n")
        self._console.flush()
        self._events.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._events.flush()

    def record_move(self, game_label: str, trace: MoveTrace) -> None:
        self._events.write(
            json.dumps(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event": "move",
                    "game": game_label,
                    **asdict(trace),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        self._events.flush()
        if trace.ply % self.heartbeat_plies == 0 or trace.used_fallback or trace.timing_invalid:
            marker = " FALLBACK" if trace.used_fallback else ""
            if trace.timing_invalid:
                marker += " TIMING_INVALID"
            self.emit(
                "move_heartbeat",
                (
                    f"MOVE {game_label} ply={trace.ply + 1} {trace.player}/{trace.color} "
                    f"{trace.usi} depth={trace.completed_depth} "
                    f"roots={trace.completed_root_moves} elapsed={trace.elapsed_seconds:.2f}s "
                    f"qtrunc={trace.quiescence_truncations} "
                    f"cpu={trace.cpu_elapsed_seconds:.2f}s "
                    f"wait={trace.scheduler_wait_seconds:.2f}s "
                    f"over={trace.deadline_overshoot_ms:.0f}ms "
                    f"wallover={trace.wall_overshoot_ms:.0f}ms "
                    f"cause={trace.timing_cause or '-'}{marker}"
                ),
                game=game_label,
                ply=trace.ply,
                used_fallback=trace.used_fallback,
            )

    def append_game(self, game: GameResult) -> None:
        with self.games_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(game), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def write_json(self, path: Path, values: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(FINGERPRINT_FILES):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resume_configuration_mismatches(
    previous: dict[str, object],
    current: dict[str, object],
) -> list[str]:
    return [
        key for key in RESUME_COMPARABLE_FIELDS if previous.get(key) != current.get(key)
    ]


def load_openings(path: Path = DEFAULT_OPENINGS, *, limit: int | None = None) -> list[Opening]:
    openings: list[Opening] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            values = json.loads(line)
            openings.append(Opening(str(values["name"]), str(values["sfen"])))
            if limit is not None and len(openings) >= limit:
                break
    if not openings:
        raise ValueError(f"No benchmark openings found in {path}")
    return openings


def _thinking(seconds: float) -> ThinkingConfig:
    return ThinkingConfig(
        strategy=ThinkingStrategy.FIXED_TIME,
        preset=ThinkingPreset.CUSTOM,
        move_time_seconds=seconds,
        max_playouts=10_000_000,
        max_depth=128,
        safety_margin_ms=0,
    )


def _tuning(profile: AblationProfile) -> SearchTuning:
    return {
        AblationProfile.TT_PV: SearchTuning(False, EvaluationMode.MATERIAL, False),
        AblationProfile.Q_LITE: SearchTuning(
            True,
            EvaluationMode.MATERIAL,
            False,
            q_deep_evaluation_mode=EvaluationMode.MATERIAL,
        ),
        AblationProfile.POSITIONAL_LITE: SearchTuning(
            True,
            EvaluationMode.POSITIONAL,
            False,
            q_deep_evaluation_mode=EvaluationMode.FAST,
        ),
        AblationProfile.KING_SAFETY_LITE: SearchTuning(
            True,
            EvaluationMode.KING_SAFETY,
            False,
            q_deep_evaluation_mode=EvaluationMode.FAST,
            use_lazy_full=False,
        ),
        AblationProfile.FULL_SCAN: SearchTuning(
            True,
            EvaluationMode.KING_SAFETY,
            True,
            q_deep_evaluation_mode=EvaluationMode.FAST,
            use_incremental_evaluation=False,
            use_lazy_full=False,
            reference_evaluation=True,
        ),
        AblationProfile.INCREMENTAL_FAST: SearchTuning(
            True,
            EvaluationMode.KING_SAFETY,
            True,
            q_deep_evaluation_mode=EvaluationMode.FAST,
            use_incremental_evaluation=True,
            use_lazy_full=False,
        ),
        AblationProfile.LAZY_FULL: SearchTuning(
            True,
            EvaluationMode.KING_SAFETY,
            True,
            q_deep_evaluation_mode=EvaluationMode.FAST,
            use_incremental_evaluation=True,
            use_lazy_full=True,
        ),
        AblationProfile.FULL: SearchTuning(
            True,
            EvaluationMode.KING_SAFETY,
            True,
            q_deep_evaluation_mode=EvaluationMode.FAST,
            use_incremental_evaluation=True,
            use_lazy_full=False,
        ),
        AblationProfile.INCREMENTAL_BASELINE: SearchTuning(
            use_root_tactical=False,
        ),
        AblationProfile.ROOT_TACTICAL: SearchTuning(),
        AblationProfile.PVS_ASPIRATION: SearchTuning(
            use_pvs=True,
            use_aspiration=True,
        ),
        AblationProfile.SEE_MOVEPICKER: SearchTuning(
            use_pvs=True,
            use_aspiration=True,
            use_staged_move_picker=True,
            use_see=True,
        ),
        AblationProfile.LMR: SearchTuning(
            use_pvs=True,
            use_aspiration=True,
            use_staged_move_picker=True,
            use_see=True,
            use_lmr=True,
        ),
        AblationProfile.NULL_MOVE: SearchTuning(
            use_pvs=True,
            use_aspiration=True,
            use_staged_move_picker=True,
            use_see=True,
            use_lmr=True,
            use_null_move=True,
        ),
        AblationProfile.SHALLOW_PRUNING: SearchTuning(
            use_pvs=True,
            use_aspiration=True,
            use_staged_move_picker=True,
            use_see=True,
            use_lmr=True,
            use_null_move=True,
            use_shallow_pruning=True,
        ),
        AblationProfile.FULL_CANDIDATE: SearchTuning(
            use_pvs=True,
            use_aspiration=True,
            use_staged_move_picker=True,
            use_see=True,
            use_lmr=True,
            use_null_move=True,
            use_shallow_pruning=True,
        ),
        AblationProfile.HOTPATH_BASELINE: SearchTuning(
            move_picker_mode="packed",
            evaluation_schedule="lazy_full",
        ),
        AblationProfile.PVS_HOTPATH: SearchTuning(
            move_picker_mode="packed",
            evaluation_schedule="lazy_full",
            use_pvs=True,
            use_aspiration=True,
        ),
        AblationProfile.SEE_HOTPATH: SearchTuning(
            move_picker_mode="packed",
            evaluation_schedule="lazy_full",
            use_pvs=True,
            use_aspiration=True,
            use_see=True,
        ),
        AblationProfile.LMR_HOTPATH: SearchTuning(
            move_picker_mode="packed",
            evaluation_schedule="lazy_full",
            use_pvs=True,
            use_aspiration=True,
            use_see=True,
            use_lmr=True,
        ),
        AblationProfile.MATE_PROBE: SearchTuning(
            move_picker_mode="packed",
            evaluation_schedule="lazy_full",
            use_mate_probe=True,
        ),
        AblationProfile.NEXT_FULL_CANDIDATE: SearchTuning(
            move_picker_mode="packed",
            evaluation_schedule="lazy_full",
            use_pvs=True,
            use_aspiration=True,
            use_see=True,
            use_lmr=True,
            use_mate_probe=True,
        ),
    }[profile]


def make_player(profile: AblationProfile, seconds: float):
    if profile is AblationProfile.LEGACY:
        return LegacyHeuristicPlayer(_thinking(seconds))
    return HeuristicPlayer(_thinking(seconds), _search_tuning=_tuning(profile))


def _choose(player, game: ShogiGame):  # noqa: ANN001
    snapshot = game.snapshot()
    started = time.monotonic()
    move = player.choose_move_with_context(
        snapshot,
        game.legal_moves(),
        clock_state=ClockState(
            (3_600_000, 3_600_000),
            (0, 0),
            snapshot.turn,
            True,
        ),
        time_control=TimeControl(),
        color=snapshot.turn,
    )
    return move, time.monotonic() - started


def classify_outcome(outcome: GameOutcome, *, reached_limit: bool) -> tuple[str, str, float | None]:
    if reached_limit and not outcome.is_over:
        return "unresolved", "benchmark_move_limit", None
    if not outcome.is_over:
        raise ValueError("An ongoing game is only classifiable at the benchmark limit")
    if outcome.winner is None:
        return "rule_draw", outcome.reason.value, 0.5
    return "decisive", outcome.reason.value, None


def _mate_in_one_moves(board: cshogi.Board) -> tuple[str, ...]:
    move = board.mate_move_in_1ply()
    if not move:
        return ()
    result: list[str] = []
    for raw_move in board.legal_moves:
        native = int(raw_move)
        board.push(native)
        try:
            if board.is_check() and next(iter(board.legal_moves), None) is None:
                result.append(cshogi.move_to_usi(native))
        finally:
            board.pop()
    return tuple(result)


def _tactical_moves(board: cshogi.Board) -> tuple[str, ...]:
    result: list[str] = []
    for raw_move in board.legal_moves:
        move = int(raw_move)
        if int(board.piece(cshogi.move_to(move))) or cshogi.move_is_promotion(move):
            result.append(cshogi.move_to_usi(move))
    return tuple(result)


def play_game(
    opening: Opening,
    *,
    candidate_profile: AblationProfile,
    candidate_color: Color,
    seconds: float,
    max_plies: int,
    on_move: Callable[[MoveTrace], None] | None = None,
    on_failure: Callable[[dict[str, object]], None] | None = None,
) -> GameResult:
    game = ShogiGame(opening.sfen)
    candidate = make_player(candidate_profile, seconds)
    legacy = make_player(AblationProfile.LEGACY, seconds)
    traces: list[MoveTrace] = []
    fallback_seen = False
    timing_violations = 0
    timing_tolerance_ms = max(50.0, seconds * 1_000.0 * 0.05)

    for ply in range(max_plies):
        snapshot = game.snapshot()
        if snapshot.outcome.is_over:
            break
        is_candidate = snapshot.turn is candidate_color
        player = candidate if is_candidate else legacy
        mode = (
            EvaluationMode.MATERIAL
            if not is_candidate or candidate_profile is AblationProfile.LEGACY
            else _tuning(candidate_profile).evaluation_mode
        )
        board = cshogi.Board(snapshot.sfen)
        evaluation = asdict(evaluate_breakdown(board, mode))
        mate_moves = _mate_in_one_moves(board)
        mate = mate_moves[0] if mate_moves else None
        tactical_moves = _tactical_moves(board)
        try:
            move, elapsed = _choose(player, game)
        except Exception as exc:
            failure = {
                "opening": opening.name,
                "candidate_profile": candidate_profile.value,
                "candidate_color": candidate_color.name,
                "ply": ply,
                "player": "candidate" if is_candidate else "legacy",
                "color": snapshot.turn.name,
                "sfen": snapshot.sfen,
                "history": snapshot.move_history,
                "failure_stage": "choose_move",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            if on_failure is not None:
                on_failure(failure)
            raise
        info = player.last_search_info
        used_fallback = bool(getattr(info, "used_fallback", info.max_depth == 0))
        fallback_seen = fallback_seen or used_fallback
        wall_overshoot_ms = max(0.0, (elapsed - seconds) * 1_000.0)
        deadline_overshoot_ms = float(getattr(info, "deadline_overshoot_ms", 0.0))
        timing_invalid = max(wall_overshoot_ms, deadline_overshoot_ms) > timing_tolerance_ms
        cpu_elapsed_seconds = float(getattr(info, "cpu_elapsed_seconds", elapsed))
        scheduler_wait_seconds = max(0.0, elapsed - cpu_elapsed_seconds)
        cpu_invalid = cpu_elapsed_seconds > seconds
        if timing_invalid and cpu_invalid and scheduler_wait_seconds > timing_tolerance_ms / 1000:
            timing_cause = "mixed"
        elif timing_invalid and cpu_invalid:
            timing_cause = "search_cpu_overrun"
        elif timing_invalid:
            timing_cause = "scheduler_stall"
        else:
            timing_cause = None
        timing_violations += int(timing_invalid)
        trace = MoveTrace(
            ply=ply,
            player="candidate" if is_candidate else "legacy",
            color=snapshot.turn.name,
            sfen=snapshot.sfen,
            history=snapshot.move_history,
            usi=move.usi,
            elapsed_seconds=elapsed,
            completed_depth=info.max_depth,
            completed_root_moves=int(getattr(info, "completed_root_moves", 0)),
            used_fallback=used_fallback,
            main_nodes=int(getattr(info, "main_nodes", info.playouts)),
            quiescence_nodes=int(getattr(info, "quiescence_nodes", 0)),
            quiescence_truncations=int(getattr(info, "quiescence_truncations", 0)),
            tt_hits=int(getattr(info, "tt_hits", 0)),
            root_value=float(info.root_value),
            principal_variation=tuple(info.principal_variation),
            evaluation=evaluation,
            mate_in_one=mate,
            mate_in_one_moves=mate_moves,
            selected_mate_in_one=move.usi in mate_moves,
            tactical_moves=tactical_moves,
            attempted_depth=int(getattr(info, "attempted_depth", info.max_depth)),
            fallback_reason=getattr(info, "fallback_reason", None),
            deadline_overshoot_ms=deadline_overshoot_ms,
            q_depth_truncations=int(getattr(info, "q_depth_truncations", 0)),
            q_leaf_truncations=int(getattr(info, "q_leaf_truncations", 0)),
            q_budget_truncations=int(getattr(info, "q_budget_truncations", 0)),
            q_evasion_nodes=int(getattr(info, "q_evasion_nodes", 0)),
            move_ordering_seconds=float(getattr(info, "move_ordering_seconds", 0.0)),
            legal_generation_seconds=float(
                getattr(info, "legal_generation_seconds", 0.0)
            ),
            checking_probe_seconds=float(getattr(info, "checking_probe_seconds", 0.0)),
            evaluation_seconds=float(getattr(info, "evaluation_seconds", 0.0)),
            evaluation_calls=int(getattr(info, "evaluation_calls", 0)),
            ordering_calls=int(getattr(info, "ordering_calls", 0)),
            ordered_moves=int(getattr(info, "ordered_moves", 0)),
            search_elapsed_seconds=float(getattr(info, "elapsed_seconds", elapsed)),
            wall_overshoot_ms=wall_overshoot_ms,
            setup_elapsed_seconds=float(getattr(info, "setup_elapsed_seconds", 0.0)),
            position_history_plies=int(getattr(info, "position_history_plies", 0)),
            timeout_stage=getattr(info, "timeout_stage", None),
            timing_invalid=timing_invalid,
            incremental_evaluation_calls=int(
                getattr(info, "incremental_evaluation_calls", 0)
            ),
            full_evaluation_calls=int(getattr(info, "full_evaluation_calls", 0)),
            lazy_beta_skips=int(getattr(info, "lazy_beta_skips", 0)),
            lazy_alpha_skips=int(getattr(info, "lazy_alpha_skips", 0)),
            cpu_elapsed_seconds=cpu_elapsed_seconds,
            cpu_evaluation_seconds=float(
                getattr(info, "cpu_evaluation_seconds", 0.0)
            ),
            cpu_ordering_seconds=float(getattr(info, "cpu_ordering_seconds", 0.0)),
            cpu_see_seconds=float(getattr(info, "cpu_see_seconds", 0.0)),
            cpu_tt_seconds=float(getattr(info, "cpu_tt_seconds", 0.0)),
            cpu_search_overhead_seconds=float(
                getattr(info, "cpu_search_overhead_seconds", 0.0)
            ),
            scheduler_wait_seconds=scheduler_wait_seconds,
            timing_cause=timing_cause,
            classification_seconds=float(getattr(info, "classification_seconds", 0.0)),
            sorting_seconds=float(getattr(info, "sorting_seconds", 0.0)),
            pvs_null_window_searches=int(
                getattr(info, "pvs_null_window_searches", 0)
            ),
            pvs_researches=int(getattr(info, "pvs_researches", 0)),
            aspiration_fail_low=int(getattr(info, "aspiration_fail_low", 0)),
            aspiration_fail_high=int(getattr(info, "aspiration_fail_high", 0)),
            aspiration_researches=int(getattr(info, "aspiration_researches", 0)),
            see_queries=int(getattr(info, "see_queries", 0)),
            see_prunes=int(getattr(info, "see_prunes", 0)),
            lmr_reductions=int(getattr(info, "lmr_reductions", 0)),
            lmr_researches=int(getattr(info, "lmr_researches", 0)),
            null_move_attempts=int(getattr(info, "null_move_attempts", 0)),
            null_move_cutoffs=int(getattr(info, "null_move_cutoffs", 0)),
            null_move_verifications=int(
                getattr(info, "null_move_verifications", 0)
            ),
            razor_prunes=int(getattr(info, "razor_prunes", 0)),
            reverse_futility_prunes=int(
                getattr(info, "reverse_futility_prunes", 0)
            ),
            move_picker_stage_counts=dict(
                getattr(info, "move_picker_stage_counts", None) or {}
            ),
            move_picker_stage_seconds=dict(
                getattr(info, "move_picker_stage_seconds", None) or {}
            ),
            move_picker_generated=int(getattr(info, "move_picker_generated", 0)),
            move_picker_yielded=int(getattr(info, "move_picker_yielded", 0)),
            move_picker_yielded_by_stage=dict(
                getattr(info, "move_picker_yielded_by_stage", None) or {}
            ),
            full_evaluation_bound_skips=int(
                getattr(info, "full_evaluation_bound_skips", 0)
            ),
            occupied_piece_iterations=int(
                getattr(info, "occupied_piece_iterations", 0)
            ),
            mate_status=getattr(info, "mate_status", None),
            mate_nodes=int(getattr(info, "mate_nodes", 0)),
            mate_elapsed_seconds=float(getattr(info, "mate_elapsed_seconds", 0.0)),
            mate_principal_variation=tuple(
                getattr(info, "mate_principal_variation", ())
            ),
        )
        traces.append(trace)
        if on_move is not None:
            on_move(trace)
        game.play(move)

    snapshot = game.snapshot()
    kind, reason, draw_score = classify_outcome(
        snapshot.outcome,
        reached_limit=len(traces) >= max_plies,
    )
    score = draw_score
    if kind == "decisive":
        score = 1.0 if snapshot.outcome.winner is candidate_color else 0.0
    timing_invalid_seen = timing_violations > 0
    eligible = kind != "unresolved" and not fallback_seen and not timing_invalid_seen
    final_board = cshogi.Board(snapshot.sfen)
    final_evaluation = asdict(evaluate_breakdown(final_board))
    current_value = int(final_evaluation["total"])
    candidate_value = current_value if snapshot.turn is candidate_color else -current_value
    provisional_leader = (
        "candidate" if candidate_value > 0 else "legacy" if candidate_value < 0 else "even"
    )
    return GameResult(
        opening=opening.name,
        initial_sfen=opening.sfen,
        candidate_profile=candidate_profile.value,
        candidate_color=candidate_color.name,
        outcome_kind=kind,
        outcome_reason=reason,
        winner=snapshot.outcome.winner.name if snapshot.outcome.winner is not None else None,
        played_plies=len(traces),
        final_sfen=snapshot.sfen,
        final_evaluation=final_evaluation,
        final_candidate_value=candidate_value,
        provisional_leader=provisional_leader,
        fallback_seen=fallback_seen,
        eligible_for_strength=eligible,
        candidate_score=score if eligible else None,
        timing_invalid_seen=timing_invalid_seen,
        timing_violations=timing_violations,
        moves=traces,
    )


def summarize(
    games: list[GameResult],
    *,
    profile: AblationProfile,
    seconds: float,
    max_plies: int,
    elapsed: float,
    stop_reason: str | None = None,
) -> BenchmarkSummary:
    eligible_scores = [game.candidate_score for game in games if game.candidate_score is not None]
    scores = [float(score) for score in eligible_scores]
    mean, low, high = _score_interval(scores)
    by_color: dict[str, dict[str, int]] = {}
    for color in Color:
        color_games = [
            game
            for game in games
            if game.candidate_color == color.name and game.candidate_score is not None
        ]
        by_color[color.name] = {
            "eligible": len(color_games),
            "wins": sum(game.candidate_score == 1.0 for game in color_games),
            "losses": sum(game.candidate_score == 0.0 for game in color_games),
            "draws": sum(game.candidate_score == 0.5 for game in color_games),
        }
    paired_results = _paired_results(games)
    search_metrics = _search_metrics(games)
    return BenchmarkSummary(
        candidate_profile=profile.value,
        legacy_source_commit=SOURCE_COMMIT,
        seconds_per_move=seconds,
        max_plies=max_plies,
        openings=len({game.opening for game in games}),
        games=len(games),
        candidate_wins=sum(
            game.outcome_kind == "decisive" and game.winner == game.candidate_color
            for game in games
        ),
        legacy_wins=sum(
            game.outcome_kind == "decisive" and game.winner != game.candidate_color
            for game in games
        ),
        rule_draws=sum(game.outcome_kind == "rule_draw" for game in games),
        unresolved=sum(game.outcome_kind == "unresolved" for game in games),
        fallback_games=sum(game.fallback_seen for game in games),
        eligible_games=len(scores),
        candidate_score=mean,
        confidence_low=low,
        confidence_high=high,
        superiority_claim_supported=len(scores) >= 100 and low is not None and low > 0.5,
        elapsed_seconds=elapsed,
        stop_reason=stop_reason,
        timing_invalid_games=sum(game.timing_invalid_seen for game in games),
        by_color=by_color,
        paired_results=paired_results,
        search_metrics=search_metrics,
    )


def _paired_results(games: list[GameResult]) -> dict[str, int | float | None]:
    grouped: dict[str, list[GameResult]] = {}
    for game in games:
        grouped.setdefault(game.opening, []).append(game)
    candidate_sweeps = legacy_sweeps = split_pairs = drawn_pairs = invalid_pairs = 0
    pair_scores: list[float] = []
    for pair in grouped.values():
        colors = {game.candidate_color for game in pair}
        if colors != {Color.BLACK.name, Color.WHITE.name} or any(
            game.candidate_score is None for game in pair
        ):
            invalid_pairs += 1
            continue
        score = sum(float(game.candidate_score) for game in pair) / 2.0
        pair_scores.append(score)
        if score == 1.0:
            candidate_sweeps += 1
        elif score == 0.0:
            legacy_sweeps += 1
        elif all(game.candidate_score == 0.5 for game in pair):
            drawn_pairs += 1
        else:
            split_pairs += 1
    return {
        "complete_pairs": len(pair_scores),
        "invalid_pairs": invalid_pairs,
        "candidate_sweeps": candidate_sweeps,
        "legacy_sweeps": legacy_sweeps,
        "split_pairs": split_pairs,
        "drawn_pairs": drawn_pairs,
        "candidate_pair_score": (
            sum(pair_scores) / len(pair_scores) if pair_scores else None
        ),
    }


def _search_metrics(games: list[GameResult]) -> dict[str, int | float | None]:
    moves = [move for game in games for move in game.moves if move.player == "candidate"]
    if not moves:
        return {}
    depths = sorted(move.completed_depth for move in moves)
    middle = len(depths) // 2
    median_depth = (
        float(depths[middle])
        if len(depths) % 2
        else (depths[middle - 1] + depths[middle]) / 2.0
    )
    main_nodes = sum(move.main_nodes for move in moves)
    q_nodes = sum(move.quiescence_nodes for move in moves)
    mate_positions = [move for move in moves if move.mate_in_one is not None]
    tactical_positions = [move for move in moves if move.tactical_moves]
    return {
        "candidate_moves": len(moves),
        "mean_depth": sum(move.completed_depth for move in moves) / len(moves),
        "median_depth": median_depth,
        "main_nodes": main_nodes,
        "quiescence_nodes": q_nodes,
        "q_share": q_nodes / max(1, main_nodes + q_nodes),
        "mean_ordering_ms": (
            sum(move.move_ordering_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "mean_evaluation_ms": (
            sum(move.evaluation_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "mean_cpu_evaluation_ms": (
            sum(move.cpu_evaluation_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "mean_cpu_see_ms": (
            sum(move.cpu_see_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "mean_cpu_tt_ms": (
            sum(move.cpu_tt_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "mean_cpu_search_overhead_ms": (
            sum(move.cpu_search_overhead_seconds for move in moves)
            * 1_000.0
            / len(moves)
        ),
        "mean_scheduler_wait_ms": (
            sum(move.scheduler_wait_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "full_evaluation_calls": sum(move.full_evaluation_calls for move in moves),
        "incremental_evaluation_calls": sum(
            move.incremental_evaluation_calls for move in moves
        ),
        "lazy_beta_skips": sum(move.lazy_beta_skips for move in moves),
        "lazy_alpha_skips": sum(move.lazy_alpha_skips for move in moves),
        "pvs_null_window_searches": sum(
            move.pvs_null_window_searches for move in moves
        ),
        "pvs_researches": sum(move.pvs_researches for move in moves),
        "aspiration_researches": sum(move.aspiration_researches for move in moves),
        "see_queries": sum(move.see_queries for move in moves),
        "see_prunes": sum(move.see_prunes for move in moves),
        "lmr_reductions": sum(move.lmr_reductions for move in moves),
        "lmr_researches": sum(move.lmr_researches for move in moves),
        "null_move_attempts": sum(move.null_move_attempts for move in moves),
        "null_move_cutoffs": sum(move.null_move_cutoffs for move in moves),
        "null_move_verifications": sum(
            move.null_move_verifications for move in moves
        ),
        "razor_prunes": sum(move.razor_prunes for move in moves),
        "reverse_futility_prunes": sum(
            move.reverse_futility_prunes for move in moves
        ),
        "move_picker_generated": sum(move.move_picker_generated for move in moves),
        "move_picker_yielded": sum(move.move_picker_yielded for move in moves),
        "full_evaluation_bound_skips": sum(
            move.full_evaluation_bound_skips for move in moves
        ),
        "occupied_piece_iterations": sum(
            move.occupied_piece_iterations for move in moves
        ),
        "mate_probes": sum(
            move.mate_status not in {None, "not_triggered"} for move in moves
        ),
        "mate_proven": sum(move.mate_status == "proven" for move in moves),
        "mate_nodes": sum(move.mate_nodes for move in moves),
        "mean_mate_elapsed_ms": (
            sum(move.mate_elapsed_seconds for move in moves) * 1_000.0 / len(moves)
        ),
        "fallback_moves": sum(move.used_fallback for move in moves),
        "timing_invalid_moves": sum(move.timing_invalid for move in moves),
        "mate_in_one_opportunities": len(mate_positions),
        "mate_in_one_solved": sum(move.selected_mate_in_one for move in mate_positions),
        "tactical_opportunities": len(tactical_positions),
        "tactical_selected": sum(
            move.usi in move.tactical_moves for move in tactical_positions
        ),
    }


def summaries_with_deltas(summaries: list[BenchmarkSummary]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    previous: BenchmarkSummary | None = None
    for summary in summaries:
        values: dict[str, object] = asdict(summary)
        if previous is not None:
            current_pair = summary.paired_results.get("candidate_pair_score")
            previous_pair = previous.paired_results.get("candidate_pair_score")
            values["relative_to_previous"] = {
                "profile": previous.candidate_profile,
                "mean_depth": _numeric_delta(
                    summary.search_metrics.get("mean_depth"),
                    previous.search_metrics.get("mean_depth"),
                ),
                "main_nodes": _numeric_delta(
                    summary.search_metrics.get("main_nodes"),
                    previous.search_metrics.get("main_nodes"),
                ),
                "quiescence_nodes": _numeric_delta(
                    summary.search_metrics.get("quiescence_nodes"),
                    previous.search_metrics.get("quiescence_nodes"),
                ),
                "q_share": _numeric_delta(
                    summary.search_metrics.get("q_share"),
                    previous.search_metrics.get("q_share"),
                ),
                "mean_ordering_ms": _numeric_delta(
                    summary.search_metrics.get("mean_ordering_ms"),
                    previous.search_metrics.get("mean_ordering_ms"),
                ),
                "mean_evaluation_ms": _numeric_delta(
                    summary.search_metrics.get("mean_evaluation_ms"),
                    previous.search_metrics.get("mean_evaluation_ms"),
                ),
                "candidate_pair_score": _numeric_delta(current_pair, previous_pair),
            }
        result.append(values)
        previous = summary
    return result


def _numeric_delta(current: object, previous: object) -> float | None:
    if not isinstance(current, int | float) or not isinstance(previous, int | float):
        return None
    return float(current) - float(previous)


def _score_interval(scores: list[float]) -> tuple[float | None, float | None, float | None]:
    if not scores:
        return None, None, None
    mean = sum(scores) / len(scores)
    count = len(scores)
    z = 1.96
    denominator = 1.0 + z * z / count
    center = (mean + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(mean * (1.0 - mean) / count + z * z / (4 * count * count))
    radius /= denominator
    return mean, max(0.0, center - radius), min(1.0, center + radius)


def early_stop_reason(summary: BenchmarkSummary) -> str | None:
    if summary.fallback_games:
        return "invalid_fallback"
    if summary.timing_invalid_games:
        return "invalid_timing"
    if summary.games >= 4 and summary.unresolved == summary.games:
        return "invalid_all_games_unresolved"
    if summary.eligible_games >= 6 and summary.confidence_low is not None:
        if summary.confidence_low > 0.5:
            return "provisional_clearly_better"
        if summary.confidence_high is not None and summary.confidence_high < 0.5:
            return "provisional_clearly_worse"
    return None


def _critical_trace(game: GameResult) -> tuple[MoveTrace | None, str]:
    candidate_moves = [move for move in game.moves if move.player == "candidate"]
    for move in candidate_moves:
        if move.used_fallback:
            return move, "first_fallback"
    candidate_won = game.winner == game.candidate_color
    previous: MoveTrace | None = None
    for move in candidate_moves:
        if previous is not None:
            if (
                not candidate_won
                and previous.root_value >= 0.0
                and move.root_value <= -0.25
            ):
                return move, "first_negative_evaluation_flip"
            if (
                candidate_won
                and previous.root_value <= 0.0
                and move.root_value >= 0.25
            ):
                return move, "first_positive_evaluation_flip"
        previous = move
    if len(candidate_moves) >= 2:
        pairs = list(zip(candidate_moves, candidate_moves[1:], strict=False))
        if candidate_won:
            _before, critical = max(
                pairs, key=lambda pair: pair[1].root_value - pair[0].root_value
            )
            return critical, "largest_positive_evaluation_change"
        _before, critical = min(
            pairs, key=lambda pair: pair[1].root_value - pair[0].root_value
        )
        return critical, "largest_negative_evaluation_change"
    return (
        (candidate_moves[0], "only_candidate_move")
        if candidate_moves
        else (None, "no_candidate_move")
    )


def _replay_at(game: GameResult, trace: MoveTrace) -> ShogiGame:
    replay = ShogiGame(game.initial_sfen)
    for usi in trace.history:
        replay.play(usi)
    return replay


def diagnose_loss(
    game: GameResult,
    *,
    base_seconds: float,
    seconds: tuple[float, ...] = (1.0, 5.0),
) -> dict[str, object]:
    critical, critical_selection = _critical_trace(game)
    if critical is None:
        return {"classification": "no_candidate_move"}
    candidate_profile = AblationProfile(game.candidate_profile)
    first_divergence: dict[str, object] | None = None
    for trace in (move for move in game.moves if move.player == "candidate"):
        replay = _replay_at(game, trace)
        legacy = make_player(AblationProfile.LEGACY, base_seconds)
        legacy_move, _elapsed = _choose(legacy, replay)
        if legacy_move.usi != trace.usi:
            first_divergence = {
                "sfen": trace.sfen,
                "history": trace.history,
                "candidate_move": trace.usi,
                "legacy_move": legacy_move.usi,
                "candidate_depth": trace.completed_depth,
                "legacy_depth": legacy.last_search_info.max_depth,
            }
            break
    replay = _replay_at(game, critical)
    analyses: list[dict[str, object]] = []
    for budget in seconds:
        row: dict[str, object] = {"seconds": budget}
        for profile in (AblationProfile.LEGACY, candidate_profile):
            player = make_player(profile, budget)
            move, _elapsed = _choose(player, replay)
            info = player.last_search_info
            row[profile.value] = {
                "move": move.usi,
                "depth": info.max_depth,
                "value": info.root_value,
                "pv": info.principal_variation,
                "fallback": bool(getattr(info, "used_fallback", info.max_depth == 0)),
                "q_truncations": int(getattr(info, "quiescence_truncations", 0)),
            }
        analyses.append(row)
    candidate_moves = [str(row[candidate_profile.value]["move"]) for row in analyses]
    legacy_moves = [str(row[AblationProfile.LEGACY.value]["move"]) for row in analyses]
    if critical.used_fallback:
        classification = "root_timeout"
    elif critical.usi not in candidate_moves:
        classification = "original_choice_not_reproduced"
    elif len(set(candidate_moves)) > 1:
        classification = "depth_sensitive"
    elif candidate_moves[-1] == legacy_moves[-1]:
        classification = "no_profile_specific_defect_at_critical_position"
    elif critical.quiescence_truncations:
        classification = "quiescence_truncation_candidate"
    else:
        classification = "evaluation_or_search_preference_unproven"
    return {
        "classification": classification,
        "critical_selection": critical_selection,
        "critical_sfen": critical.sfen,
        "critical_history": critical.history,
        "original_move": critical.usi,
        "original_depth": critical.completed_depth,
        "candidate_profile": candidate_profile.value,
        "mate_in_one": critical.mate_in_one,
        "tactical_moves": critical.tactical_moves,
        "first_divergence": first_divergence,
        "research": analyses,
    }


def run(
    *,
    openings: list[Opening],
    profile: AblationProfile,
    seconds: float,
    max_plies: int,
    diagnose_losses: bool = False,
    reporter: AuditReporter | None = None,
    existing_games: list[GameResult] | None = None,
    allow_early_stop: bool = True,
) -> tuple[BenchmarkSummary, list[GameResult]]:
    started = time.monotonic()
    games = list(existing_games or [])
    completed = {
        (game.opening, game.candidate_color, game.candidate_profile)
        for game in games
    }
    targets = [(opening, color) for opening in openings for color in Color]
    stop_reason: str | None = None
    for index, (opening, color) in enumerate(targets, start=1):
        key = (opening.name, color.name, profile.value)
        if key in completed:
            if reporter is not None:
                reporter.emit(
                    "game_skipped",
                    (
                        f"RESUME skip completed game {index}/{len(targets)} "
                        f"{opening.name}/{color.name}"
                    ),
                    opening=opening.name,
                    candidate_color=color.name,
                    profile=profile.value,
                )
            continue
        label = f"{index}/{len(targets)} {profile.value} {opening.name}/{color.name}"
        if reporter is not None:
            reporter.emit("game_started", f"GAME START {label}", game=label)
        game = play_game(
            opening,
            candidate_profile=profile,
            candidate_color=color,
            seconds=seconds,
            max_plies=max_plies,
            on_move=(
                (lambda trace, game_label=label: reporter.record_move(game_label, trace))
                if reporter is not None
                else None
            ),
            on_failure=(
                (
                    lambda failure, game_label=label: reporter.emit(
                        "game_failed",
                        (
                            f"GAME FAILED {game_label} ply={int(failure['ply']) + 1} "
                            f"{failure['exception_type']}: "
                            f"{failure['exception_message']}"
                        ).rstrip(),
                        game=game_label,
                        **failure,
                    )
                )
                if reporter is not None
                else None
            ),
        )
        if (
            diagnose_losses
            and game.outcome_kind == "decisive"
            and game.winner != game.candidate_color
        ):
            if reporter is not None:
                reporter.emit("diagnosis_started", f"DIAGNOSE loss {label}", game=label)
            game.diagnosis = diagnose_loss(game, base_seconds=seconds)
        games.append(game)
        completed.add(key)
        if reporter is not None:
            reporter.append_game(game)
        interim = summarize(
            games,
            profile=profile,
            seconds=seconds,
            max_plies=max_plies,
            elapsed=time.monotonic() - started,
        )
        if reporter is not None:
            score = "n/a" if interim.candidate_score is None else f"{interim.candidate_score:.3f}"
            interval = (
                "n/a"
                if interim.confidence_low is None
                else f"[{interim.confidence_low:.3f},{interim.confidence_high:.3f}]"
            )
            reporter.emit(
                "game_completed",
                (
                    f"GAME END {label} result={game.outcome_kind}/{game.outcome_reason} "
                    f"score={score} CI95={interval} eligible={interim.eligible_games} "
                    f"unresolved={interim.unresolved} fallback_games={interim.fallback_games} "
                    f"timing_invalid_games={interim.timing_invalid_games}"
                ),
                game=label,
                summary=asdict(interim),
            )
            reporter.write_json(
                reporter.summary_path,
                {"status": "running", "latest_summary": asdict(interim)},
            )
        stop_reason = early_stop_reason(interim) if allow_early_stop else None
        if stop_reason is not None:
            if reporter is not None:
                reporter.emit(
                    "early_stop",
                    f"EARLY STOP {profile.value}: {stop_reason}",
                    reason=stop_reason,
                    summary=asdict(interim),
                )
            break
    elapsed = time.monotonic() - started
    summary = summarize(
        games,
        profile=profile,
        seconds=seconds,
        max_plies=max_plies,
        elapsed=elapsed,
        stop_reason=stop_reason,
    )
    return summary, games


def _write_jsonl(path: Path, games: list[GameResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for game in games:
            handle.write(json.dumps(asdict(game), ensure_ascii=False, sort_keys=True) + "\n")


def _game_from_dict(values: dict[str, object]) -> GameResult:
    moves = []
    for raw in values.get("moves", []):
        move = dict(raw)
        move["history"] = tuple(move["history"])
        move["principal_variation"] = tuple(move["principal_variation"])
        move["mate_principal_variation"] = tuple(
            move.get("mate_principal_variation", ())
        )
        move["tactical_moves"] = tuple(move["tactical_moves"])
        moves.append(MoveTrace(**move))
    game_values = dict(values)
    game_values["moves"] = moves
    return GameResult(**game_values)


def load_completed_games(path: Path) -> list[GameResult]:
    if not path.exists():
        return []
    games: list[GameResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                games.append(_game_from_dict(json.loads(line)))
    return games


def next_game_key(
    openings: list[Opening],
    profiles: list[AblationProfile],
    games: list[GameResult],
) -> dict[str, str] | None:
    completed = {
        (game.opening, game.candidate_color, game.candidate_profile) for game in games
    }
    for profile in profiles:
        for opening in openings:
            for color in Color:
                if (opening.name, color.name, profile.value) not in completed:
                    return {
                        "opening": opening.name,
                        "candidate_color": color.name,
                        "candidate_profile": profile.value,
                    }
    return None


def failed_run_summary(
    reporter: AuditReporter,
    *,
    run_dir: Path,
    failure: str,
    openings: list[Opening],
    profiles: list[AblationProfile],
) -> dict[str, object]:
    completed_games = load_completed_games(reporter.games_path)
    values: dict[str, object] = {
        "status": "failed",
        "run_dir": str(run_dir),
        "failure": failure,
        "completed_games": len(completed_games),
        "partial_game": reporter.last_failure,
        "next_game": next_game_key(openings, profiles, completed_games),
    }
    if reporter.summary_path.exists():
        previous = json.loads(reporter.summary_path.read_text(encoding="utf-8"))
        if "latest_summary" in previous:
            values["latest_summary"] = previous["latest_summary"]
        elif "summaries" in previous:
            values["summaries"] = previous["summaries"]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument(
        "--profile",
        choices=[value.value for value in AblationProfile],
        default="full",
    )
    parser.add_argument("--all-profiles", action="store_true")
    parser.add_argument("--diagnose-losses", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--heartbeat-plies", type=int, default=1)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument(
        "--normal-priority",
        action="store_true",
        help="Do not request Above Normal priority for this benchmark process.",
    )
    args = parser.parse_args()
    if args.positions < 1 or args.seconds <= 0 or args.max_plies < 1:
        parser.error("positions, seconds, and max-plies must be positive")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    openings = load_openings(args.openings, limit=args.positions)
    profiles = list(AblationProfile) if args.all_profiles else [AblationProfile(args.profile)]
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = ROOT / "artifacts" / "reports" / f"abp-{stamp}"
    run_dir = run_dir.resolve()
    if not args.resume and (run_dir / "games.jsonl").exists():
        parser.error(f"run directory already contains games; use --resume: {run_dir}")
    priority = ProcessPriority(
        requested="normal" if args.normal_priority else "above_normal"
    )
    if not args.normal_priority:
        priority.apply()
    reporter = AuditReporter(run_dir, heartbeat_plies=args.heartbeat_plies)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "openings_file": str(args.openings.resolve()),
        "opening_names": [opening.name for opening in openings],
        "profiles": [profile.value for profile in profiles],
        "seconds_per_move": args.seconds,
        "max_plies": args.max_plies,
        "legacy_source_commit": SOURCE_COMMIT,
        "implementation_fingerprint": implementation_fingerprint(),
        "evaluator_fingerprint": HandcraftedEvaluator().fingerprint,
        "process_priority_requested": priority.requested,
        "process_priority": asdict(priority),
        "diagnose_losses": args.diagnose_losses,
        "early_stop_enabled": not args.no_early_stop,
    }
    if args.resume and reporter.manifest_path.exists():
        previous = json.loads(reporter.manifest_path.read_text(encoding="utf-8"))
        mismatches = resume_configuration_mismatches(previous, manifest)
        if mismatches:
            reporter.close()
            parser.error(f"resume configuration mismatch: {', '.join(mismatches)}")
        manifest["created_at"] = previous.get("created_at", manifest["created_at"])
        manifest["resumed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    reporter.write_json(reporter.manifest_path, manifest)
    latest_path = ROOT / "artifacts" / "reports" / "latest.json"
    reporter.write_json(
        latest_path,
        {"run_dir": str(run_dir), "status": "running", "updated_at": manifest["created_at"]},
    )
    existing = load_completed_games(reporter.games_path) if args.resume else []
    all_games: list[GameResult] = []
    summaries: list[BenchmarkSummary] = []
    reporter.emit(
        "run_started",
        (
            f"AUDIT START dir={run_dir} profiles={','.join(profile.value for profile in profiles)} "
            f"openings={len(openings)} seconds={args.seconds} max_plies={args.max_plies}"
        ),
        run_dir=str(run_dir),
        manifest=manifest,
        resumed_games=len(existing),
    )
    try:
        for profile in profiles:
            profile_existing = [
                game for game in existing if game.candidate_profile == profile.value
            ]
            summary, games = run(
                openings=openings,
                profile=profile,
                seconds=args.seconds,
                max_plies=args.max_plies,
                diagnose_losses=args.diagnose_losses,
                reporter=reporter,
                existing_games=profile_existing,
                allow_early_stop=not args.no_early_stop,
            )
            summaries.append(summary)
            all_games.extend(games)
        if args.jsonl is not None:
            _write_jsonl(args.jsonl, all_games)
        final_status = (
            "early_stopped" if any(summary.stop_reason for summary in summaries) else "completed"
        )
        final_values = {
            "status": final_status,
            "run_dir": str(run_dir),
            "summaries": summaries_with_deltas(summaries),
        }
        reporter.write_json(reporter.summary_path, final_values)
        manifest["status"] = final_status
        manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        reporter.write_json(reporter.manifest_path, manifest)
        reporter.write_json(
            latest_path,
            {
                "run_dir": str(run_dir),
                "status": final_status,
                "updated_at": manifest["completed_at"],
            },
        )
        reporter.emit(
            "run_completed",
            f"AUDIT {final_status.upper()} dir={run_dir}",
            **final_values,
        )
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["interrupted_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        reporter.write_json(reporter.manifest_path, manifest)
        reporter.write_json(
            latest_path,
            {
                "run_dir": str(run_dir),
                "status": "interrupted",
                "updated_at": manifest["interrupted_at"],
            },
        )
        reporter.emit(
            "run_interrupted",
            f"AUDIT INTERRUPTED; resume with --run-dir {run_dir} --resume",
            run_dir=str(run_dir),
        )
        return 130
    except Exception as exc:
        failure_text = type(exc).__name__
        if str(exc):
            failure_text += f": {exc}"
        failed_summary = failed_run_summary(
            reporter,
            run_dir=run_dir,
            failure=failure_text,
            openings=openings,
            profiles=profiles,
        )
        reporter.write_json(reporter.summary_path, failed_summary)
        manifest["status"] = "failed"
        manifest["failure"] = failure_text
        manifest["failure_stage"] = (
            reporter.last_failure.get("failure_stage") if reporter.last_failure else "run"
        )
        manifest["next_game"] = failed_summary["next_game"]
        reporter.write_json(reporter.manifest_path, manifest)
        reporter.write_json(
            latest_path,
            {
                "run_dir": str(run_dir),
                "status": "failed",
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        reporter.emit("run_failed", f"AUDIT FAILED: {failure_text}", error=failure_text)
        raise
    finally:
        priority.restore()
        manifest["process_priority"] = asdict(priority)
        try:
            reporter.write_json(reporter.manifest_path, manifest)
        except OSError:
            pass
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
