#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

EXTRA_ARGS=()
if [[ $# -gt 0 && "$1" != --* ]]; then
  EXTRA_ARGS+=(--encode-variant "$1")
  shift
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m src.scripts.kai0_train pi05_arrange_flowers_encode "${EXTRA_ARGS[@]}" "$@"
