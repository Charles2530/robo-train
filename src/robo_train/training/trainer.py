"""Mock trainer that exercises processor and policy training paths."""

from typing import Any

from robo_train.data.processors.pipeline import ProcessorPipeline
from robo_train.schema.episode import Episode
from robo_train.schema.experiment_config import ExperimentConfig
from robo_train.training.models.policy import UnifiedEmbodiedPolicy


class MockTrainer:
    """Run one deterministic mock training step."""

    def __init__(
        self,
        policy: UnifiedEmbodiedPolicy,
        processor: ProcessorPipeline,
        episodes: list[Episode],
        experiment: ExperimentConfig | None = None,
    ) -> None:
        self.policy = policy
        self.processor = processor
        self.episodes = episodes
        self.experiment = experiment

    def train_one_epoch(self) -> dict[str, Any]:
        """Fit processors, call policy.forward, and return scalar losses."""
        self.processor.fit(self.episodes)
        batch = self.processor.transform(self.episodes)
        output = self.policy.forward(batch)
        summary = {
            "episodes": len(self.episodes),
            "loss": float(output["loss"]),
            "action_loss": float(output["action_loss"]),
            "world_action_loss": float(output["world_action_loss"]),
            "stage_advantage_loss": float(output["stage_advantage_loss"]),
            "safety_loss": float(output["safety_loss"]),
        }
        if self.experiment is not None:
            summary.update(
                {
                    "experiment_id": self.experiment.experiment_id,
                    "algorithm": self.experiment.trainer.algorithm,
                    "datasets": self.experiment.dataset_ids,
                    "policy_type": self.experiment.policy.policy_type,
                }
            )
        return summary
