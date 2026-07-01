#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${PROJECT_ROOT}/scripts/train/kai0/_bootstrap.sh"
PYTHON_BIN="$(resolve_project_python "${PROJECT_ROOT}" "${KAI0_BOOTSTRAP_CONFIG:-${PROJECT_ROOT}/configs/base/train/kai0_jax.yaml}")"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m src.scripts.kai0_train pi05_put_the_books_back_table30v2_joint "$@"
