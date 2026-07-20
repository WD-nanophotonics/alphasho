"""Single implementation behind every PyCharm training entry."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from alphasho.training.config import QuickTrainingSettings, load_training_config
from alphasho.training.orchestrator import AlphaZeroTrainer


def run_training_entry(quick_values: dict[str, Any]) -> None:
    """Run quick settings, or a fixed profile selected by a run configuration."""

    fixed_profile = os.environ.get("ALPHASHO_TRAINING_PROFILE")
    if fixed_profile:
        project_root = Path(__file__).resolve().parents[3]
        settings = load_training_config(
            project_root / "configs" / "training" / f"{fixed_profile}.toml"
        )
    else:
        settings = QuickTrainingSettings.from_mapping(quick_values)
    trainer = AlphaZeroTrainer(settings)
    timed_out = threading.Event()

    def stop_smoke() -> None:
        timed_out.set()
        trainer.request_stop()

    profile = settings.name if fixed_profile else settings.profile
    watchdog = (
        threading.Timer(20.0, stop_smoke)
        if profile == "smoke"
        else None
    )
    if watchdog is not None:
        watchdog.daemon = True
        watchdog.start()
    try:
        trainer.run()
    finally:
        if watchdog is not None:
            watchdog.cancel()
    if timed_out.is_set():
        raise TimeoutError("AlphaSho Training Smoke exceeded 20 seconds")
