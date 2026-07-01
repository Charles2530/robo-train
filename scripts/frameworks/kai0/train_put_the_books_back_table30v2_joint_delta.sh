#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${PROJECT_ROOT}/scripts/frameworks/kai0/_bootstrap.sh"
PYTHON_BIN="$(resolve_project_python "${PROJECT_ROOT}" "${KAI0_BOOTSTRAP_CONFIG:-${PROJECT_ROOT}/configs/frameworks/kai0/base/kai0_jax.yaml}")"
export_robo_train_pythonpath "$PROJECT_ROOT"

append_table30v2_card_defaults \
  "run1-put-books-back-table30v2-joint-delta-50000step-1card-bs8" \
  "run1-put-books-back-table30v2-joint-delta-50000step-8card-bs256" \
  "$@"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m robo_train.cli.train --framework kai0 pi05_put_the_books_back_table30v2_joint_delta "${TABLE30V2_DEFAULT_ARGS[@]}" "$@"
