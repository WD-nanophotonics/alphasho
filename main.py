"""PyCharm-friendly AlphaSho desktop entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def run() -> int:
    from alphasho.ui.app import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
