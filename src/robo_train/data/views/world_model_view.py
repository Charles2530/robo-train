"""World-model data view."""

from typing import Any

import numpy as np

from robo_train.data.processors.batch_builder import EmbodiedBatch
from robo_train.data.views.base import BaseDataView


class WorldModelDataView(BaseDataView):
    """Action-conditioned future-prediction view for world-action models."""

    name = "world_model"

    def __init__(self, future_horizon: int | None = None, target_dim: int = 16) -> None:
        if future_horizon is not None and future_horizon <= 0:
            raise ValueError("future_horizon must be positive")
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")
        self.future_horizon = future_horizon
        self.target_dim = target_dim

    def build(self, batch: EmbodiedBatch) -> dict[str, Any]:
        """Return context observations, action history, and future targets."""
        horizon = self.future_horizon or int(batch.metadata.get("action_horizon", 1))
        return {
            "context_images": batch.images,
            "context_state": batch.state,
            "action_history": batch.actions,
            "future_state_target": self._future_target(batch, horizon),
            "future_horizon": horizon,
            "robot_cards": batch.robot_cards,
            "task_cards": batch.task_cards,
            "action_semantics": batch.action_semantics,
        }

    def _future_target(self, batch: EmbodiedBatch, horizon: int) -> np.ndarray:
        batch_size = len(batch.robot_cards)
        if batch.state is None:
            return np.zeros((batch_size, horizon, self.target_dim), dtype=np.float32)

        state = np.asarray(batch.state, dtype=np.float32)
        target = state[:, :horizon, :]
        if target.shape[1] < horizon:
            pad_steps = horizon - target.shape[1]
            target = np.concatenate([target, np.repeat(target[:, -1:, :], pad_steps, axis=1)], axis=1)
        if target.shape[-1] > self.target_dim:
            target = target[..., : self.target_dim]
        elif target.shape[-1] < self.target_dim:
            pad_dim = self.target_dim - target.shape[-1]
            pad = np.zeros((*target.shape[:-1], pad_dim), dtype=np.float32)
            target = np.concatenate([target, pad], axis=-1)
        return target.astype(np.float32)
