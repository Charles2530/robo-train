"""Multimodal stream schema for Universal Embodied IR episodes."""

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class StreamType(str, Enum):
    """Supported Universal Embodied stream types."""

    RGB_VIDEO = "rgb_video"
    DEPTH_VIDEO = "depth_video"
    POINTCLOUD_SEQUENCE = "pointcloud_sequence"
    TACTILE_ARRAY = "tactile_array"
    AUDIO = "audio"
    PROPRIOCEPTION = "proprioception"
    LANGUAGE = "language"
    ACTION = "action"
    LABEL = "label"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class StreamSpec(BaseModel):
    """Metadata for a stream, optionally without inline data."""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    name: str
    stream_type: StreamType | str
    fps: float
    shape: list[int]
    dtype: str
    frame: str | None = None
    path: str | None = None
    timestamps: list[float] | None = None


class StreamData(StreamSpec):
    """A stream with optional inline demo data."""

    data: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("data")
    def serialize_data(self, value: Any) -> Any:
        """Convert NumPy values to plain Python containers."""
        return _to_jsonable(value)

    def as_array(self, dtype: str | None = None) -> np.ndarray:
        """Return inline data as a NumPy array."""
        if self.data is None:
            return np.empty((0,), dtype=dtype or "float32")
        return np.asarray(self.data, dtype=dtype)

    def first_dim(self) -> int | None:
        """Return the stream's time dimension when known."""
        if self.shape:
            return int(self.shape[0])
        if self.data is None:
            return None
        array = np.asarray(self.data)
        return int(array.shape[0]) if array.ndim else None
