"""Pydantic schemas for the Universal Embodied IR."""

from robo_train.schema.action_semantics import ActionSemantics
from robo_train.schema.dataset_manifest import DatasetManifest
from robo_train.schema.episode import Episode
from robo_train.schema.experiment_config import (
    DatasetConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainerConfig,
)
from robo_train.schema.frame_graph import FrameGraph, FrameTransform
from robo_train.schema.robot_card import RobotCard
from robo_train.schema.scene_card import ObjectCard, SceneCard, WorkspaceBounds
from robo_train.schema.sensor_card import CameraIntrinsics, SensorCard, SensorType
from robo_train.schema.stream import StreamData, StreamSpec, StreamType
from robo_train.schema.task_card import TaskCard
from robo_train.schema.training_profile import (
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
