"""Scene and object schemas for object-centric embodied tasks."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class WorkspaceBounds(BaseModel):
    """Axis-aligned workspace limits in a scene frame."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    frame: str = "robot_base"

    @model_validator(mode="after")
    def validate_bounds(self) -> "WorkspaceBounds":
        """Require each lower bound to be less than its upper bound."""
        for axis_name in ("x", "y", "z"):
            lower, upper = getattr(self, axis_name)
            if lower >= upper:
                raise ValueError(f"{axis_name} lower bound must be less than upper bound")
        return self


class ObjectCard(BaseModel):
    """Object-level metadata for 3D grounding and future prediction."""

    object_id: str
    category: str
    affordances: list[str] = Field(default_factory=list)
    frame: str | None = None
    estimated_pose: list[list[float]] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pose(self) -> "ObjectCard":
        """Validate homogeneous pose matrices when present."""
        if self.estimated_pose is not None:
            if len(self.estimated_pose) != 4 or any(len(row) != 4 for row in self.estimated_pose):
                raise ValueError("estimated_pose must be 4x4")
        return self


class SceneCard(BaseModel):
    """Portable scene description for 3D world-model views."""

    scene_id: str
    objects: dict[str, ObjectCard] = Field(default_factory=dict)
    workspace_bounds: WorkspaceBounds | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_object_keys(self) -> "SceneCard":
        """Keep object dictionary keys aligned with object ids."""
        for key, value in self.objects.items():
            if key != value.object_id:
                raise ValueError(f"object key {key!r} must match object_id {value.object_id!r}")
        return self
