from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFileDialog

from alphasho.domain import MatchConfig
from alphasho.ui.i18n import Language, Translator
from alphasho.ui.match_dialog import MatchSettingsDialog


def _dialog(qtbot, tmp_path: Path) -> MatchSettingsDialog:
    dialog = MatchSettingsDialog(
        Translator(Language.ENGLISH),
        "",
        MatchConfig(),
    )
    qtbot.addWidget(dialog)
    dialog.default_model_path = tmp_path / "checkpoints" / "best.pt"
    dialog.model_library_path = tmp_path / "models"
    return dialog


def test_model_selector_lists_official_best_and_sorted_library(
    qtbot,
    tmp_path: Path,
) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.default_model_path.parent.mkdir()
    dialog.default_model_path.write_bytes(b"best")
    dialog.model_library_path.mkdir()
    second = dialog.model_library_path / "zeta.pt"
    first = dialog.model_library_path / "Alpha.pt"
    ignored = dialog.model_library_path / "notes.txt"
    second.write_bytes(b"zeta")
    first.write_bytes(b"alpha")
    ignored.write_text("not a checkpoint", encoding="utf-8")

    selector = QComboBox()
    dialog._populate_models(selector, str(second))

    assert [selector.itemData(index) for index in range(selector.count())] == [
        str(dialog.default_model_path),
        str(first),
        str(second),
    ]
    assert selector.currentData() == str(second)


def test_model_selector_keeps_missing_official_best_but_disables_it(
    qtbot,
    tmp_path: Path,
) -> None:
    dialog = _dialog(qtbot, tmp_path)
    selector = QComboBox()

    dialog._populate_models(selector, None)

    assert selector.count() == 1
    assert selector.itemData(0) == str(dialog.default_model_path)
    item = selector.model().item(0)
    assert item is not None
    assert not item.isEnabled()


def test_custom_model_is_preserved_and_missing_model_is_reported(
    qtbot,
    tmp_path: Path,
) -> None:
    dialog = _dialog(qtbot, tmp_path)
    missing = tmp_path / "missing.pt"
    selector = QComboBox()

    dialog._populate_models(selector, str(missing))

    assert selector.currentData() == str(missing)
    fields = dialog.white_fields
    fields.model_selector.clear()
    fields.model_selector.addItem("missing", str(missing))
    dialog._update_model_status(fields)
    assert "not found" in fields.model_status.text().lower()


def test_browse_adds_an_arbitrary_model_file(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dialog = _dialog(qtbot, tmp_path)
    selected = tmp_path / "manual.pt"
    selected.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(selected), "PyTorch models (*.pt)"),
    )

    dialog._browse_model(dialog.white_fields)

    assert dialog.white_fields.model_selector.currentData() == str(selected)
    assert (
        dialog.white_fields.model_selector.itemData(
            dialog.white_fields.model_selector.currentIndex(),
            Qt.ItemDataRole.ToolTipRole,
        )
        == str(selected)
    )
