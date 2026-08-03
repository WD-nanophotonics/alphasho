import json
from pathlib import Path

import cshogi
import pytest

import benchmarks.heuristic_strength as strength_benchmark
from alphasho.domain import Color, GameOutcome, OutcomeReason, ThinkingConfig
from alphasho.engine import ShogiGame
from benchmarks.heuristic_strength import (
    AblationProfile,
    AuditReporter,
    GameResult,
    Opening,
    ProcessPriority,
    _mate_in_one_moves,
    _write_jsonl,
    classify_outcome,
    early_stop_reason,
    failed_run_summary,
    implementation_fingerprint,
    load_completed_games,
    load_openings,
    next_game_key,
    play_game,
    resume_configuration_mismatches,
    run,
    summarize,
)
from benchmarks.legacy_3262cc8 import SOURCE_COMMIT, LegacyHeuristicPlayer


def test_move_limit_is_unresolved_and_never_scored_as_draw() -> None:
    kind, reason, score = classify_outcome(GameOutcome(), reached_limit=True)

    assert kind == "unresolved"
    assert reason == "benchmark_move_limit"
    assert score is None


def test_real_repetition_is_a_rule_draw() -> None:
    game = ShogiGame("8k/9/9/9/9/9/9/9/K8 b - 1")
    cycle = ("9i9h", "1a1b", "9h9i", "1b1a")
    for move in cycle * 3:
        game.play(move)

    kind, reason, score = classify_outcome(game.snapshot().outcome, reached_limit=False)

    assert kind == "rule_draw"
    assert reason == OutcomeReason.REPETITION.value
    assert score == 0.5


def test_checkmate_and_perpetual_check_are_decisive() -> None:
    mate = ShogiGame("8k/7GR/9/9/9/9/9/9/K8 w - 1").snapshot().outcome
    perpetual = GameOutcome(OutcomeReason.PERPETUAL_CHECK, Color.BLACK)

    assert classify_outcome(mate, reached_limit=False)[:2] == (
        "decisive",
        OutcomeReason.CHECKMATE.value,
    )
    assert classify_outcome(perpetual, reached_limit=False)[:2] == (
        "decisive",
        OutcomeReason.PERPETUAL_CHECK.value,
    )


def test_mate_diagnostic_records_all_mating_moves() -> None:
    board = cshogi.Board("8k/9/7G1/9/9/9/9/9/K8 b R 1")

    assert set(_mate_in_one_moves(board)) == {
        f"R*{file_index}a" for file_index in range(3, 10)
    }


def test_benchmark_uses_fixed_evaluation_openings() -> None:
    path = Path("configs/training/evaluation_positions.jsonl")
    openings = load_openings(path)

    assert len(openings) == 10
    assert openings[0].name == "iteration-000086-game-000001-ply12"
    assert all(ShogiGame(opening.sfen).snapshot().move_number == 13 for opening in openings)


def test_legacy_control_matches_commit_3262cc8_golden_search() -> None:
    game = ShogiGame()
    player = LegacyHeuristicPlayer(
        ThinkingConfig(max_playouts=2_000, max_depth=3)
    )

    move = player.choose_move(game.snapshot(), game.legal_moves())

    assert SOURCE_COMMIT == "3262cc8"
    assert move.usi == "7g7f"
    assert player.last_search_info is not None
    assert player.last_search_info.playouts == 1_178
    assert player.last_search_info.max_depth == 3
    # The empty PV is a known behavior of the historical root TT implementation.
    assert player.last_search_info.principal_variation == ()


def test_short_benchmark_game_is_unresolved_not_a_strength_sample() -> None:
    result = play_game(
        Opening("initial", ShogiGame().snapshot().sfen),
        candidate_profile=AblationProfile.FULL,
        candidate_color=Color.BLACK,
        seconds=0.01,
        max_plies=1,
    )

    assert result.outcome_kind == "unresolved"
    assert result.candidate_score is None
    assert not result.eligible_for_strength


def test_jsonl_contains_move_diagnostics(tmp_path: Path) -> None:
    result = play_game(
        Opening("initial", ShogiGame().snapshot().sfen),
        candidate_profile=AblationProfile.FULL,
        candidate_color=Color.BLACK,
        seconds=0.01,
        max_plies=1,
    )
    destination = tmp_path / "reports" / "games.jsonl"

    _write_jsonl(destination, [result])
    values = json.loads(destination.read_text(encoding="utf-8"))

    assert values["outcome_kind"] == "unresolved"
    assert values["candidate_score"] is None
    assert values["moves"][0]["completed_root_moves"] >= 0
    assert "quiescence_truncations" in values["moves"][0]
    assert "deadline_overshoot_ms" in values["moves"][0]
    assert "fallback_reason" in values["moves"][0]
    assert "q_budget_truncations" in values["moves"][0]
    assert "q_evasion_nodes" in values["moves"][0]
    assert "legal_generation_seconds" in values["moves"][0]
    assert "checking_probe_seconds" in values["moves"][0]
    assert "evaluation_seconds" in values["moves"][0]
    assert "evaluation_calls" in values["moves"][0]
    assert "ordering_calls" in values["moves"][0]
    assert "ordered_moves" in values["moves"][0]
    assert "timing_invalid" in values["moves"][0]
    assert "search_elapsed_seconds" in values["moves"][0]
    assert "wall_overshoot_ms" in values["moves"][0]
    assert "setup_elapsed_seconds" in values["moves"][0]
    assert "position_history_plies" in values["moves"][0]
    assert "timeout_stage" in values["moves"][0]
    assert "cpu_elapsed_seconds" in values["moves"][0]
    assert "scheduler_wait_seconds" in values["moves"][0]
    assert "timing_cause" in values["moves"][0]
    assert "full_evaluation_calls" in values["moves"][0]
    assert "lazy_beta_skips" in values["moves"][0]


def test_ablation_profiles_include_reference_incremental_and_lazy() -> None:
    assert {
        AblationProfile.FULL_SCAN,
        AblationProfile.INCREMENTAL_FAST,
        AblationProfile.LAZY_FULL,
    }.issubset(set(AblationProfile))


def test_process_priority_is_restored(monkeypatch) -> None:
    calls: list[int] = []

    class _Kernel32:
        @staticmethod
        def GetCurrentProcess() -> int:
            return 123

        @staticmethod
        def GetPriorityClass(_process: int) -> int:
            return 0x20

        @staticmethod
        def SetPriorityClass(_process: int, value: int) -> int:
            calls.append(value)
            return 1

    class _Windll:
        kernel32 = _Kernel32()

    monkeypatch.setattr(strength_benchmark.sys, "platform", "win32")
    monkeypatch.setattr(strength_benchmark.ctypes, "windll", _Windll(), raising=False)
    priority = ProcessPriority()

    priority.apply()
    priority.restore()

    assert calls == [0x00008000, 0x20]
    assert priority.applied
    assert priority.restored


def test_game_failure_callback_preserves_reproducible_position(monkeypatch) -> None:
    opening = Opening("initial", ShogiGame().snapshot().sfen)
    failures: list[dict[str, object]] = []

    def fail_choose(_player, _game):  # noqa: ANN001
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(strength_benchmark, "_choose", fail_choose)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        play_game(
            opening,
            candidate_profile=AblationProfile.FULL,
            candidate_color=Color.BLACK,
            seconds=0.1,
            max_plies=1,
            on_failure=failures.append,
        )

    assert len(failures) == 1
    failure = failures[0]
    assert failure["opening"] == "initial"
    assert failure["ply"] == 0
    assert failure["sfen"] == opening.sfen
    assert failure["history"] == ()
    assert failure["failure_stage"] == "choose_move"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "synthetic failure"


def test_resume_rejects_changed_implementation_fingerprint() -> None:
    fingerprint = implementation_fingerprint()
    previous = {
        "opening_names": ["initial"],
        "profiles": ["full"],
        "seconds_per_move": 1.0,
        "max_plies": 256,
        "legacy_source_commit": SOURCE_COMMIT,
        "implementation_fingerprint": fingerprint,
    }
    current = dict(previous)

    assert len(fingerprint) == 64
    assert resume_configuration_mismatches(previous, current) == []

    current["implementation_fingerprint"] = "changed"
    assert resume_configuration_mismatches(previous, current) == [
        "implementation_fingerprint"
    ]


def test_next_game_restarts_partial_but_skips_completed() -> None:
    openings = [Opening("initial", ShogiGame().snapshot().sfen)]
    completed = _synthetic_result(score=1.0)
    completed.opening = "initial"
    completed.candidate_color = "BLACK"
    completed.candidate_profile = "full"

    assert next_game_key(openings, [AblationProfile.FULL], [completed]) == {
        "opening": "initial",
        "candidate_color": "WHITE",
        "candidate_profile": "full",
    }


def test_failed_summary_keeps_completed_and_partial_game(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed-audit"
    reporter = AuditReporter(run_dir)
    opening = Opening("initial", ShogiGame().snapshot().sfen)
    completed = _synthetic_result(score=1.0)
    completed.opening = opening.name
    completed.candidate_color = "BLACK"
    completed.candidate_profile = "full"
    try:
        reporter.append_game(completed)
        reporter.write_json(
            reporter.summary_path,
            {"status": "running", "latest_summary": {"games": 1}},
        )
        reporter.emit(
            "game_failed",
            "GAME FAILED synthetic",
            opening=opening.name,
            candidate_color="WHITE",
            candidate_profile="full",
            ply=4,
            sfen=opening.sfen,
            history=(),
            player="candidate",
            failure_stage="choose_move",
            exception_type="RuntimeError",
            exception_message="boom",
        )

        values = failed_run_summary(
            reporter,
            run_dir=run_dir,
            failure="RuntimeError: boom",
            openings=[opening],
            profiles=[AblationProfile.FULL],
        )
    finally:
        reporter.close()

    assert values["status"] == "failed"
    assert values["completed_games"] == 1
    assert values["partial_game"]["ply"] == 4
    assert values["next_game"] == {
        "opening": "initial",
        "candidate_color": "WHITE",
        "candidate_profile": "full",
    }
    assert values["latest_summary"] == {"games": 1}


def _synthetic_result(*, score: float | None, fallback: bool = False) -> GameResult:
    return GameResult(
        opening="synthetic",
        initial_sfen=ShogiGame().snapshot().sfen,
        candidate_profile="full",
        candidate_color="BLACK",
        outcome_kind="decisive",
        outcome_reason=OutcomeReason.CHECKMATE.value,
        winner="WHITE" if score == 0.0 else "BLACK",
        played_plies=1,
        final_sfen=ShogiGame().snapshot().sfen,
        final_evaluation={},
        final_candidate_value=0,
        provisional_leader="even",
        fallback_seen=fallback,
        eligible_for_strength=score is not None,
        candidate_score=score,
    )


def test_fallback_game_reports_real_result_but_is_excluded_from_score() -> None:
    summary = summarize(
        [_synthetic_result(score=None, fallback=True)],
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )

    assert summary.candidate_wins == 1
    assert summary.fallback_games == 1
    assert summary.eligible_games == 0
    assert summary.candidate_score is None
    assert early_stop_reason(summary) == "invalid_fallback"


def test_timing_invalid_game_is_excluded_from_strength() -> None:
    game = _synthetic_result(score=1.0)
    game.timing_invalid_seen = True
    game.timing_violations = 1
    game.eligible_for_strength = False
    game.candidate_score = None

    summary = summarize(
        [game],
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )

    assert summary.timing_invalid_games == 1
    assert summary.eligible_games == 0
    assert early_stop_reason(summary) == "invalid_timing"


def test_summary_reports_complete_paired_results_and_color_split() -> None:
    black = _synthetic_result(score=1.0)
    white = _synthetic_result(score=0.0)
    white.candidate_color = "WHITE"

    summary = summarize(
        [black, white],
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )

    assert summary.paired_results["complete_pairs"] == 1
    assert summary.paired_results["split_pairs"] == 1
    assert summary.paired_results["candidate_pair_score"] == 0.5
    assert summary.by_color["BLACK"]["wins"] == 1
    assert summary.by_color["WHITE"]["losses"] == 1


def test_superiority_claim_requires_100_eligible_games_and_confidence() -> None:
    ninety_nine = [_synthetic_result(score=1.0) for _ in range(99)]
    one_hundred = ninety_nine + [_synthetic_result(score=1.0)]

    before = summarize(
        ninety_nine,
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )
    after = summarize(
        one_hundred,
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )

    assert not before.superiority_claim_supported
    assert after.superiority_claim_supported


def test_provisional_early_stop_uses_confidence_not_raw_score() -> None:
    five_wins = [_synthetic_result(score=1.0) for _ in range(5)]
    six_wins = five_wins + [_synthetic_result(score=1.0)]

    before = summarize(
        five_wins,
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )
    after = summarize(
        six_wins,
        profile=AblationProfile.FULL,
        seconds=1.0,
        max_plies=256,
        elapsed=0.0,
    )

    assert early_stop_reason(before) is None
    assert early_stop_reason(after) == "provisional_clearly_better"
    assert not after.superiority_claim_supported


def test_streaming_run_persists_events_games_summary_and_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "audit"
    reporter = AuditReporter(run_dir)
    opening = Opening("initial", ShogiGame().snapshot().sfen)
    try:
        summary, games = run(
            openings=[opening],
            profile=AblationProfile.FULL,
            seconds=0.01,
            max_plies=1,
            reporter=reporter,
            allow_early_stop=False,
        )
    finally:
        reporter.close()

    assert summary.games == 2
    assert len(games) == 2
    assert len(load_completed_games(run_dir / "games.jsonl")) == 2
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "move"' in events
    assert '"event": "game_completed"' in events
    assert (run_dir / "summary.json").exists()

    resumed_reporter = AuditReporter(run_dir)
    try:
        resumed_summary, resumed_games = run(
            openings=[opening],
            profile=AblationProfile.FULL,
            seconds=0.01,
            max_plies=1,
            reporter=resumed_reporter,
            existing_games=load_completed_games(run_dir / "games.jsonl"),
            allow_early_stop=False,
        )
    finally:
        resumed_reporter.close()

    assert resumed_summary.games == 2
    assert len(resumed_games) == 2
    assert len(load_completed_games(run_dir / "games.jsonl")) == 2


def test_reporter_json_write_creates_parent_directory(tmp_path: Path) -> None:
    reporter = AuditReporter(tmp_path / "audit")
    target = tmp_path / "reports" / "latest.json"
    try:
        reporter.write_json(target, {"status": "running"})
    finally:
        reporter.close()

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running"}
