import time

from alphasho.domain import (
    ClockState,
    Color,
    ThinkingConfig,
    ThinkingPreset,
    ThinkingStrategy,
    TimeControl,
    TimeControlMode,
)
from alphasho.engine import ShogiGame
from alphasho.heuristicplayer import HeuristicPlayer


def _config(*, nodes: int, depth: int) -> ThinkingConfig:
    return ThinkingConfig(
        strategy=ThinkingStrategy.FIXED_PLAYOUT,
        preset=ThinkingPreset.CUSTOM,
        max_playouts=nodes,
        max_depth=depth,
    )


def test_returns_legal_move_without_mutating_game() -> None:
    game = ShogiGame()
    snapshot = game.snapshot()
    legal = game.legal_moves()
    player = HeuristicPlayer(_config(nodes=500, depth=3))

    move = player.choose_move(snapshot, legal)

    assert move in legal
    assert game.snapshot() == snapshot
    assert player.last_search_info is not None
    assert player.last_search_info.max_depth >= 1


def test_prefers_free_high_value_capture() -> None:
    game = ShogiGame("4k4/9/9/4r4/4P4/9/9/9/4K4 b - 1")
    player = HeuristicPlayer(_config(nodes=2_000, depth=2))

    move = player.choose_move(game.snapshot(), game.legal_moves())

    assert move.usi == "5e5d"


def test_context_search_obeys_small_time_budget() -> None:
    game = ShogiGame()
    player = HeuristicPlayer(
        ThinkingConfig(
            strategy=ThinkingStrategy.FIXED_TIME,
            preset=ThinkingPreset.CUSTOM,
            move_time_seconds=0.02,
            max_playouts=1_000_000,
            max_depth=20,
            safety_margin_ms=0,
        )
    )
    started = time.monotonic()
    move = player.choose_move_with_context(
        game.snapshot(),
        game.legal_moves(),
        clock_state=ClockState((10_000, 10_000), (0, 0), Color.BLACK, True),
        time_control=TimeControl(mode=TimeControlMode.FISCHER),
        color=Color.BLACK,
    )

    assert move in game.legal_moves()
    assert time.monotonic() - started < 0.5


def test_context_auto_uses_allocated_clock_time(monkeypatch) -> None:
    game = ShogiGame()
    player = HeuristicPlayer(ThinkingConfig.for_preset(ThinkingPreset.BALANCED))
    captured = {}

    def capture_budget(
        snapshot,
        legal_moves,
        seconds,
        node_limit,
        max_depth,
        progress_callback,
    ):
        captured.update(
            seconds=seconds,
            node_limit=node_limit,
            max_depth=max_depth,
        )
        return legal_moves[0]

    monkeypatch.setattr(player, "_search", capture_budget)
    move = player.choose_move_with_context(
        game.snapshot(),
        game.legal_moves(),
        clock_state=ClockState(
            (3_600_000, 3_600_000),
            (0, 0),
            Color.BLACK,
            True,
        ),
        time_control=TimeControl(mode=TimeControlMode.FISCHER),
        color=Color.BLACK,
    )

    assert move in game.legal_moves()
    assert captured == {
        "seconds": 54.0,
        "node_limit": 5_400_000,
        "max_depth": 128,
    }
