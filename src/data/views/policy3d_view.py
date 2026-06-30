"""3D policy data view."""

from typing import Any

import numpy as np

from src.data.processors.batch_builder import EmbodiedBatch
from src.data.views.base import BaseDataView


class Policy3DDataView(BaseDataView):
    """State/point-cloud view for 3D robot policy training."""

    name = "policy_3d"

    def build(self, batch: EmbodiedBatch) -> dict[str, Any]:
        """Return geometry-first inputs and action targets."""
        point_cloud = self._point_cloud_from_batch(batch)
        return {
            "point_cloud": point_cloud,
            "state": batch.state,
            "actions": batch.actions,
            "robot_cards": batch.robot_cards,
            "task_cards": batch.task_cards,
            "action_semantics": batch.action_semantics,
            "modality_mask": {**batch.modality_mask, "point_cloud": True},
        }

    def _point_cloud_from_batch(self, batch: EmbodiedBatch) -> np.ndarray:
        if batch.state is not None:
            state = np.asarray(batch.state, dtype=np.float32)
            xyz = state[..., :3]
            if xyz.shape[-1] < 3:
                pad = np.zeros((*xyz.shape[:-1], 3 - xyz.shape[-1]), dtype=np.float32)
                xyz = np.concatenate([xyz, pad], axis=-1)
            return xyz.astype(np.float32)

        batch_size = len(batch.robot_cards)
        horizon = int(batch.metadata.get("history_window", 1))
        return np.zeros((batch_size, horizon, 3), dtype=np.float32)
