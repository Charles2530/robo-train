"""Pydantic schemas for the Universal Embodied IR."""

from src.schema.action_semantics import ActionSemantics
from src.schema.dataset_manifest import DatasetManifest
from src.schema.episode import Episode
from src.schema.experiment_config import (
    DatasetConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainerConfig,
)
from src.schema.frame_graph import FrameGraph, FrameTransform
from src.schema.robot_card import RobotCard
from src.schema.scene_card import ObjectCard, SceneCard, WorkspaceBounds
from src.schema.sensor_card import CameraIntrinsics, SensorCard, SensorType
from src.schema.stream import StreamData, StreamSpec, StreamType
from src.schema.task_card import TaskCard
from src.schema.training_profile import (
    DataProfile,
    EmbodimentProfile,
    LossProfile,
    ModelFamily,
    TrainingProfile,
)

__all__ = [
    "ActionSemantics",
    "DataProfile",
    "DatasetConfig",
    "DatasetManifest",
    "EmbodimentProfile",
    "Episode",
    "ExperimentConfig",
    "FrameGraph",
    "FrameTransform",
    "LossProfile",
    "ModelFamily",
    "ObjectCard",
    "PolicyConfig",
    "RobotCard",
    "CameraIntrinsics",
    "SceneCard",
    "SensorCard",
    "SensorType",
    "StreamData",
    "StreamSpec",
    "StreamType",
    "TaskCard",
    "TrainerConfig",
    "TrainingProfile",
    "WorkspaceBounds",
]
