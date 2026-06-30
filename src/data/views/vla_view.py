"""VLA data view."""

from typing import Any

from src.data.processors.batch_builder import EmbodiedBatch
from src.data.views.base import BaseDataView


class VLADataView(BaseDataView):
    """Image-language-action view for VLA and OpenVLA/Octo-style policies."""

    name = "vla"

    def build(self, batch: EmbodiedBatch) -> dict[str, Any]:
        """Return multimodal VLA inputs and action targets."""
        return {
            "images": batch.images,
            "language": batch.language,
            "state": batch.state,
            "actions": batch.actions,
            "robot_cards": batch.robot_cards,
            "task_cards": batch.task_cards,
            "action_semantics": batch.action_semantics,
            "modality_mask": batch.modality_mask,
        }
