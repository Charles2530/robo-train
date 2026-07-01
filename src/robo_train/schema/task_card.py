"""Task metadata for instructions, objects, stages, and success criteria."""

from typing import Any

from pydantic import BaseModel, Field


class TaskCard(BaseModel):
    """Portable task description consumed by data processors and runtime."""

    task_id: str
    instruction: str
    skill_tags: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    success_condition: str | None = None
    stages: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
