import pytest


class MemorySettings:
    """Small QSettings-compatible store that keeps tests out of the registry."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values if values is not None else {}

    def value(self, key: str, default=None, type=None):  # noqa: A002, ANN001
        result = self.values.get(key, default)
        return type(result) if type is not None else result

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value

    def sync(self) -> None:
        pass


@pytest.fixture
def memory_settings() -> MemorySettings:
    return MemorySettings()

