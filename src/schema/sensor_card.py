"""Sensor metadata and calibration references for embodied robot data."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SensorType(str, Enum):
    """Supported sensor families."""

    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"
    POINT_CLOUD = "point_cloud"
    TACTILE = "tactile"
    PROPRIOCEPTION = "proprioception"
    AUDIO = "audio"


class CameraIntrinsics(BaseModel):
    """Pinhole-style camera intrinsics."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: list[float] = Field(default_factory=list)


class SensorCard(BaseModel):
    """Portable sensor description linked to a frame graph frame."""

    sensor_id: str
    sensor_type: SensorType | str
    frame: str
    fps: float = Field(gt=0)
    intrinsics: CameraIntrinsics | None = None
    extrinsics_ref: str | None = None
    shape: list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> "SensorCard":
        """Require positive dimensions when a shape is declared."""
        if self.shape is not None and any(dim <= 0 for dim in self.shape):
            raise ValueError("sensor shape dimensions must be positive")
        return self
