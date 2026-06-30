"""Coordinate-frame graph schema for 3D-aware embodied episodes."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class FrameTransform(BaseModel):
    """A directed transform edge between two named coordinate frames."""

    parent: str
    child: str
    transform_type: str
    matrix: list[list[float]] | None = None
    source: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transform(self) -> "FrameTransform":
        """Keep transform edges usable and matrix-shaped."""
        if self.parent == self.child:
            raise ValueError("transform parent and child frames must differ")
        if self.matrix is not None:
            if len(self.matrix) != 4 or any(len(row) != 4 for row in self.matrix):
                raise ValueError("transform matrix must be 4x4")
        return self


class FrameGraph(BaseModel):
    """Named frames plus directed transforms used to align 3D streams."""

    frames: list[str]
    transforms: list[FrameTransform] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> "FrameGraph":
        """Require unique frames and transform endpoints declared in frames."""
        if len(set(self.frames)) != len(self.frames):
            raise ValueError("frame names must be unique")
        known = set(self.frames)
        for transform in self.transforms:
            if transform.parent not in known:
                raise ValueError(f"transform references unknown frame {transform.parent!r}")
            if transform.child not in known:
                raise ValueError(f"transform references unknown frame {transform.child!r}")
        return self

    def has_frame(self, frame: str) -> bool:
        """Return whether a frame name is declared."""
        return frame in set(self.frames)

    def get_transform(self, parent: str, child: str) -> FrameTransform:
        """Return a directed transform or raise a clear KeyError."""
        for transform in self.transforms:
            if transform.parent == parent and transform.child == child:
                return transform
        raise KeyError(f"no transform from {parent!r} to {child!r}")
