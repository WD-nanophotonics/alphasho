from alphasho.domain import Color, MatchConfig, ParticipantConfig, ParticipantKind
from alphasho.ui.controller import _timeout_winner


def test_human_player_never_loses_on_time() -> None:
    config = MatchConfig(
        black=ParticipantConfig(ParticipantKind.HUMAN),
        white=ParticipantConfig(ParticipantKind.RANDOM_AI),
    )

    assert _timeout_winner(config, Color.BLACK) is None


def test_non_human_player_still_loses_on_time() -> None:
    config = MatchConfig(
        black=ParticipantConfig(ParticipantKind.HUMAN),
        white=ParticipantConfig(ParticipantKind.HEURISTIC),
    )

    assert _timeout_winner(config, Color.WHITE) is Color.BLACK
    assert _timeout_winner(config, None) is None
