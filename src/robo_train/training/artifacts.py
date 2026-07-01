"""Serializable training artifacts and checkpoint metadata."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CheckpointMetadata(BaseModel):
    """Metadata that identifies what a checkpoint was trained from."""

    checkpoint_id: str
    experiment_id: str
    step: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    policy_type: str
    dataset_ids: list[str] = Field(default_factory=list)
    created_by: str = "src"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingArtifact(BaseModel):
    """Bundle checkpoint metadata with reproducibility snapshots."""

    checkpoint: CheckpointMetadata
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    processor_stats: dict[str, Any] = Field(default_factory=dict)
    adapter_stats: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        """Save the artifact as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingArtifact":
        """Load the artifact from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
