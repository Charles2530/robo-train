"""Training infrastructure for the data-first MVP."""

from src.training.algorithm_registry import AlgorithmRegistry, build_policy_from_config
from src.training.artifacts import CheckpointMetadata, TrainingArtifact
from src.training.loss_registry import compute_loss_profile
from src.training.trainer import MockTrainer

__all__ = [
    "AlgorithmRegistry",
    "CheckpointMetadata",
    "MockTrainer",
    "TrainingArtifact",
    "build_policy_from_config",
    "compute_loss_profile",
]
