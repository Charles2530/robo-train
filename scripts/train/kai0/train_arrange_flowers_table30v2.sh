#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${PROJECT_ROOT}/scripts/train/kai0/_bootstrap.sh"
PYTHON_BIN="$(resolve_project_python "${PROJECT_ROOT}" "${KAI0_BOOTSTRAP_CONFIG:-${PROJECT_ROOT}/configs/base/train/kai0_jax.yaml}")"

append_table30v2_card_defaults \
  "run1-arrange-flowers-table30v2-50000step-1card-bs8" \
  "run1-arrange-flowers-table30v2-50000step-8card-bs256" \
  "$@"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m src.scripts.kai0_train pi05_arrange_flowers_table30v2 "${TABLE30V2_DEFAULT_ARGS[@]}" "$@"
