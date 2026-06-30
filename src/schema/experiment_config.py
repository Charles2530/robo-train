"""Config schemas for reproducible embodied policy experiments."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.schema.dataset_manifest import DatasetManifest
from src.schema.training_profile import TrainingProfile


class DatasetConfig(BaseModel):
    """Dataset usage inside an experiment."""

    manifest: DatasetManifest
    split: str = "train"
    batch_weight: float = Field(default=1.0, gt=0)
    max_episodes: int | None = Field(default=None, gt=0)
    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_split(self) -> "DatasetConfig":
        """Require the selected split to exist in the manifest."""
        if self.split not in self.manifest.splits:
            raise ValueError(f"split {self.split!r} is not declared by dataset {self.manifest.dataset_id!r}")
        return self


class PolicyConfig(BaseModel):
    """Policy construction settings independent from trainer code."""

    policy_type: str = "unified_mock"
    latent_dim: int = Field(default=32, gt=0)
    action_horizon: int = Field(default=8, gt=0)
    action_dim: int | None = Field(default=None, gt=0)
    pretrained_checkpoint: str | None = None
    freeze_backbone: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    """Algorithm and optimization settings."""

    algorithm: str = "behavior_cloning"
    max_epochs: int = Field(default=1, gt=0)
    batch_size: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    loss_weights: dict[str, float] = Field(default_factory=dict)
    gradient_clip_norm: float | None = Field(default=None, gt=0)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level experiment config spanning dataset, model, and train params."""

    experiment_id: str
    seed: int = 42
    datasets: list[DatasetConfig]
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    training_profile: TrainingProfile = Field(default_factory=TrainingProfile.default_vla)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_experiment(self) -> "ExperimentConfig":
        """Ensure the experiment has at least one dataset and a useful id."""
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.datasets:
            raise ValueError("datasets must contain at least one dataset config")
        return self

    @property
    def dataset_ids(self) -> list[str]:
        """Return dataset ids used by this experiment."""
        return [dataset.manifest.dataset_id for dataset in self.datasets]

    def to_json(self, path: str | Path) -> None:
        """Save the experiment config as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load an experiment config from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
