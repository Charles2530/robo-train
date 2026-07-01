"""Protocol implemented by train framework adapters."""

from __future__ import annotations

from typing import Any, Protocol

from robo_train.schema.experiment_config import ExperimentConfig


class FrameworkPlugin(Protocol):
    """Adapter boundary for framework-specific training backends."""

    name: str

    def list_profiles(self) -> list[str]:
        """Return available profile names."""

    def load_profile(self, name_or_script: str) -> Any:
        """Load a framework-specific profile object."""

    def to_experiment_config(
        self,
        profile: Any,
        *,
        exp_name: str | None = None,
        batch_size: int | None = None,
        num_train_steps: int | None = None,
        num_workers: int | None = None,
    ) -> ExperimentConfig:
        """Convert a profile into the shared experiment schema."""

    def build_launch_payload(
        self,
        profile: Any,
        *,
        exp_name: str | None = None,
        batch_size: int | None = None,
        num_train_steps: int | None = None,
        num_workers: int | None = None,
        dry_run: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a JSON-serializable launch payload."""
