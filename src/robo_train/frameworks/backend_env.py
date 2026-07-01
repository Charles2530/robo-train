"""Helpers for passing environment variables to framework backend subprocesses.

Dry-run payloads expose a curated ``env`` summary so secrets are not printed to
JSON logs.  Real launches attach the full ``runtime_env`` for subprocess
execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# Keys surfaced in dry-run / JSON payloads. Framework launchers may extend this.
BACKEND_ENV_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "FFMPEG_HOME",
        "JAX_PLATFORMS",
        "KAI0_ROOT",
        "LD_LIBRARY_PATH",
        "OPENPI_VIDEO_BACKEND",
        "PYTHONPATH",
        "TASK_A_ROOT",
        "WANDB_ENABLED",
        "WANDB_MODE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
    }
)


def summarize_backend_env(
    env: Mapping[str, str],
    *,
    extra_keys: Iterable[str] = (),
) -> dict[str, str]:
    """Return a JSON-safe subset of backend environment variables."""
    keys = BACKEND_ENV_SUMMARY_KEYS | frozenset(extra_keys)
    return {key: env[key] for key in sorted(keys) if key in env}


def attach_runtime_env(
    backend_payload: dict[str, Any],
    runtime_env: Mapping[str, str],
) -> dict[str, Any]:
    """Attach the full subprocess environment used for real launches."""
    backend_payload["runtime_env"] = dict(runtime_env)
    return backend_payload


def resolve_backend_subprocess_env(backend: Mapping[str, Any]) -> dict[str, str] | None:
    """Resolve the environment dict passed to ``subprocess.Popen(..., env=...)``."""
    runtime_env = backend.get("runtime_env")
    if runtime_env is not None:
        return dict(runtime_env)
    env = backend.get("env")
    if env is None:
        return None
    return dict(env)
