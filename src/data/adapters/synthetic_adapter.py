"""Deterministic synthetic dataset adapter for the runnable demo."""

from pathlib import Path
from typing import Any

import numpy as np

from src.data.adapters.base_adapter import BaseAdapter
from src.data.adapters.registry import AdapterRegistry
from src.schema.action_semantics import ActionSemantics
from src.schema.episode import Episode
from src.schema.frame_graph import FrameGraph, FrameTransform
from src.schema.robot_card import default_franka_card
from src.schema.scene_card import ObjectCard, SceneCard, WorkspaceBounds
from src.schema.stream import StreamData, StreamType
from src.schema.task_card import TaskCard


class SyntheticAdapter(BaseAdapter):
    """Generate StarVLA-style raw IR episodes with canonical 7D actions."""

    def __init__(
        self,
        seed: int = 42,
        steps: int = 32,
        image_height: int = 64,
        image_width: int = 64,
        state_dim: int = 14,
        tactile_dim: int = 8,
        fps: float = 20.0,
    ) -> None:
        self.seed = seed
        self.steps = steps
        self.image_height = image_height
        self.image_width = image_width
        self.state_dim = state_dim
        self.tactile_dim = tactile_dim
        self.fps = fps

    def to_ir(self, source: Any) -> list[Episode]:
        """Generate synthetic episodes.

        The source may be a dictionary containing `episodes`; otherwise one
        episode is generated.
        """
        count = int(source.get("episodes", 1)) if isinstance(source, dict) else 1
        rng = np.random.default_rng(self.seed)
        episodes: list[Episode] = []
        for index in range(count):
            episodes.append(self._make_episode(index=index, rng=rng))
        return episodes

    def from_ir(self, episodes: list[Episode], output_path: Path) -> None:
        """Export episodes as one JSON file per episode."""
        output_path.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            episode.to_json(output_path / f"{episode.episode_id}.json")

    def _make_episode(self, index: int, rng: np.random.Generator) -> Episode:
        semantics = ActionSemantics.default_eef_delta(fps=self.fps, horizon=8)
        instruction = "pick up the red block and place it into the bowl"
        timestamps = (np.arange(self.steps, dtype=np.float64) / self.fps).round(6).tolist()

        front = rng.integers(
            0,
            255,
            size=(self.steps, self.image_height, self.image_width, 3),
            dtype=np.uint8,
        )
        wrist = rng.integers(
            0,
            255,
            size=(self.steps, self.image_height, self.image_width, 3),
            dtype=np.uint8,
        )
        state = rng.normal(0.0, 0.35, size=(self.steps, self.state_dim)).astype(np.float32)
        tactile = rng.normal(0.0, 0.1, size=(self.steps, self.tactile_dim)).astype(np.float32)

        base = np.linspace(-0.2, 0.2, self.steps, dtype=np.float32)
        action = rng.normal(0.0, 0.03, size=(self.steps, semantics.action_dim)).astype(np.float32)
        action[:, 0] += base
        action[:, -1] = np.linspace(0.0, 1.0, self.steps, dtype=np.float32)

        stage_names = ["approach", "grasp", "lift", "place"]
        stage_labels = [stage_names[min(i * len(stage_names) // self.steps, len(stage_names) - 1)] for i in range(self.steps)]
        scene = SceneCard(
            scene_id="synthetic_tabletop",
            objects={
                "red_block": ObjectCard(
                    object_id="red_block",
                    category="cube",
                    affordances=["graspable", "movable"],
                    frame="object_red_block",
                    confidence=0.9,
                ),
                "bowl": ObjectCard(
                    object_id="bowl",
                    category="container",
                    affordances=["place_inside"],
                    frame="object_bowl",
                    confidence=0.9,
                ),
            },
            workspace_bounds=WorkspaceBounds(x=(-0.5, 0.5), y=(-0.4, 0.4), z=(0.0, 0.8)),
            metadata={"source": "synthetic"},
        )
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frame_graph = FrameGraph(
            frames=["world", "robot_base", "camera_front", "camera_wrist", "panda_hand", "object_red_block", "object_bowl"],
            transforms=[
                FrameTransform(parent="world", child="robot_base", transform_type="fixed", matrix=identity),
                FrameTransform(parent="camera_front", child="robot_base", transform_type="calibrated", matrix=identity),
                FrameTransform(parent="camera_wrist", child="panda_hand", transform_type="calibrated", matrix=identity),
                FrameTransform(parent="object_red_block", child="camera_front", transform_type="perception_estimate", confidence=0.9),
                FrameTransform(parent="object_bowl", child="camera_front", transform_type="perception_estimate", confidence=0.9),
            ],
        )

        return Episode(
            episode_id=f"synthetic-{index:04d}",
            dataset_id="synthetic_demo",
            source_type="human_demo",
            robot=default_franka_card(),
            task=TaskCard(
                task_id="pick-red-block-place-bowl",
                instruction=instruction,
                skill_tags=["pick", "place"],
                objects=["red block", "bowl"],
                scene_id="synthetic_tabletop",
                success_condition="red block is inside the bowl",
                stages=stage_names,
                metadata={"synthetic": True},
            ),
            scene=scene,
            frame_graph=frame_graph,
            action_semantics=semantics,
            streams={
                "front_rgb": StreamData(
                    name="front_rgb",
                    stream_type=StreamType.RGB_VIDEO,
                    fps=self.fps,
                    shape=list(front.shape),
                    dtype="uint8",
                    frame="camera_front",
                    timestamps=timestamps,
                    data=front.tolist(),
                ),
                "wrist_rgb": StreamData(
                    name="wrist_rgb",
                    stream_type=StreamType.RGB_VIDEO,
                    fps=self.fps,
                    shape=list(wrist.shape),
                    dtype="uint8",
                    frame="camera_wrist",
                    timestamps=timestamps,
                    data=wrist.tolist(),
                ),
                "state": StreamData(
                    name="state",
                    stream_type=StreamType.PROPRIOCEPTION,
                    fps=self.fps,
                    shape=list(state.shape),
                    dtype="float32",
                    frame="robot_base",
                    timestamps=timestamps,
                    data=state.tolist(),
                ),
                "action": StreamData(
                    name="action",
                    stream_type=StreamType.ACTION,
                    fps=self.fps,
                    shape=list(action.shape),
                    dtype="float32",
                    frame="robot_base",
                    timestamps=timestamps,
                    data=action.tolist(),
                    metadata={"role": "canonical_action"},
                ),
                "raw_action": StreamData(
                    name="raw_action",
                    stream_type=StreamType.ACTION,
                    fps=self.fps,
                    shape=[self.steps, 0],
                    dtype="float32",
                    frame="vendor",
                    timestamps=timestamps,
                    data=[[] for _ in range(self.steps)],
                    metadata={"role": "raw_action", "available": False},
                ),
                "tactile": StreamData(
                    name="tactile",
                    stream_type=StreamType.TACTILE_ARRAY,
                    fps=self.fps,
                    shape=list(tactile.shape),
                    dtype="float32",
                    frame="gripper",
                    timestamps=timestamps,
                    data=tactile.tolist(),
                ),
                "language": StreamData(
                    name="language",
                    stream_type=StreamType.LANGUAGE,
                    fps=0.0,
                    shape=[1],
                    dtype="str",
                    frame=None,
                    data=[instruction],
                ),
                "stage_labels": StreamData(
                    name="stage_labels",
                    stream_type=StreamType.LABEL,
                    fps=self.fps,
                    shape=[self.steps],
                    dtype="str",
                    frame=None,
                    timestamps=timestamps,
                    data=stage_labels,
                ),
            },
            labels={"success": True, "stages": stage_labels},
            metadata={"seed": self.seed, "canonical_action_stream": "action", "raw_action_stream": "raw_action"},
        )


AdapterRegistry.register("synthetic", SyntheticAdapter)
