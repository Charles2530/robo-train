"""Dataset manifest schema for multi-source embodied training data."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class DatasetManifest(BaseModel):
    """Portable metadata describing one training/evaluation dataset."""

    dataset_id: str
    name: str
    source_framework: str
    source_uri: str | None = None
    task_families: list[str] = Field(default_factory=list)
    robot_families: list[str] = Field(default_factory=list)
    splits: dict[str, float] = Field(default_factory=lambda: {"train": 1.0})
    sampling_weight: float = Field(default=1.0, gt=0)
    stats_path: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    lineage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "DatasetManifest":
        """Check that split weights and identity fields are usable."""
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if not self.source_framework.strip():
            raise ValueError("source_framework must be non-empty")
        if not self.splits:
            raise ValueError("splits must contain at least one split")
        total = 0.0
        for split, fraction in self.splits.items():
            if not split.strip():
                raise ValueError("split names must be non-empty")
            if fraction <= 0.0:
                raise ValueError("split fractions must be positive")
            total += fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
        return self

    @property
    def split_names(self) -> list[str]:
        """Return split names in manifest order."""
        return list(self.splits)

    def to_json(self, path: str | Path) -> None:
        """Save the manifest as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetManifest":
        """Load a manifest from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
