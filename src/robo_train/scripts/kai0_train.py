"""Compatibility shim for the old Kai0-specific CLI module."""

from __future__ import annotations

import sys

from robo_train.cli.train import main as train_main


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    args = ["--source-root" if arg == "--kai0-source-root" else arg for arg in args]
    train_main(["--framework", "kai0", *args])


if __name__ == "__main__":
    main()
