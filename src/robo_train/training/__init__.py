"""Training infrastructure for the data-first MVP."""

from robo_train.training.algorithm_registry import AlgorithmRegistry, build_policy_from_config
from robo_train.training.artifacts import CheckpointMetadata, TrainingArtifact
from robo_train.training.loss_registry import compute_loss_profile
from robo_train.training.trainer import MockTrainer

__all__ = [
    "AlgorithmRegistry",
    "CheckpointMetadata",
    "MockTrainer",
    "TrainingArtifact",
    "build_policy_from_config",
    "compute_loss_profile",
]
