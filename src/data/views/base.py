"""Base data view contracts."""

from abc import ABC, abstractmethod
from typing import Any

from src.data.processors.batch_builder import EmbodiedBatch
from src.schema.training_profile import DataProfile, TrainingProfile


class BaseDataView(ABC):
    """Converts one `EmbodiedBatch` into model-family-specific inputs."""

    name: str

    @abstractmethod
    def build(self, batch: EmbodiedBatch) -> dict[str, Any]:
        """Build a view dictionary for a model family."""


def build_data_view(profile_or_name: TrainingProfile | str) -> BaseDataView:
    """Create a data view from a profile object or profile name."""
    from src.data.views.policy3d_view import Policy3DDataView
    from src.data.views.vla_view import VLADataView
    from src.data.views.world_model_view import WorldModelDataView

    if isinstance(profile_or_name, TrainingProfile):
        name = str(profile_or_name.data_profile)
    else:
        name = str(profile_or_name)

    if name == DataProfile.VLA.value:
        return VLADataView()
    if name == DataProfile.POLICY_3D.value:
        return Policy3DDataView()
    if name == DataProfile.WORLD_MODEL.value:
        return WorldModelDataView()
    raise KeyError(f"unknown data view {name!r}")
