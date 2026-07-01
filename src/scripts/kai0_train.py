"""Compatibility shim for the old Kai0 training module path."""

from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from robo_train.scripts.kai0_train import main


if __name__ == "__main__":
    main()
