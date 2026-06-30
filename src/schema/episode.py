"""Episode schema for the Universal Embodied IR."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.schema.action_semantics import ActionSemantics
from src.schema.frame_graph import FrameGraph
from src.schema.robot_card import RobotCard
from src.schema.scene_card import SceneCard
from src.schema.stream import StreamData
from src.schema.task_card import TaskCard


class Episode(BaseModel):
    """A self-contained multimodal robot episode in Universal Embodied IR."""

    model_config = ConfigDict(use_enum_values=True)

    episode_id: str
    dataset_id: str
    source_type: str
    robot: RobotCard
    task: TaskCard
    scene: SceneCard | None = None
    frame_graph: FrameGraph | None = None
    action_semantics: ActionSemantics
    streams: dict[str, StreamData]
    annotations: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Return the best-known episode horizon."""
        if "action" in self.streams:
            steps = self.streams["action"].first_dim()
            if steps is not None:
                return steps
        for stream in self.streams.values():
            steps = stream.first_dim()
            if steps is not None and stream.stream_type != "language":
                return steps
        return 0

    def get_stream(self, name: str) -> StreamData:
        """Return a named stream or raise KeyError with a clear message."""
        if name not in self.streams:
            raise KeyError(f"episode {self.episode_id!r} has no stream {name!r}")
        return self.streams[name]

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable episode metadata for logs and CLIs."""
        return {
            "episode_id": self.episode_id,
            "dataset_id": self.dataset_id,
            "source_type": self.source_type,
            "robot_id": self.robot.robot_id,
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "scene_id": self.scene.scene_id if self.scene is not None else self.task.scene_id,
            "num_steps": self.num_steps,
            "streams": sorted(self.streams),
            "frames": self.frame_graph.frames if self.frame_graph is not None else [],
            "action_representation": self.action_semantics.representation,
            "action_dim": self.action_semantics.action_dim,
        }

    def to_json(self, path: str | Path) -> None:
        """Save this episode as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Episode":
        """Load an episode from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
