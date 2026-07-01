from pathlib import Path

import pytest

from robo_train.schema.action_semantics import ActionSemantics
from robo_train.schema.episode import Episode
from robo_train.schema.frame_graph import FrameGraph, FrameTransform
from robo_train.schema.robot_card import default_franka_card
from robo_train.schema.scene_card import ObjectCard, SceneCard, WorkspaceBounds
from robo_train.schema.sensor_card import CameraIntrinsics, SensorCard
from robo_train.schema.stream import StreamData, StreamType
from robo_train.schema.task_card import TaskCard


def test_schema_serializes_episode_round_trip(tmp_path: Path):
    semantics = ActionSemantics.default_eef_delta()
    episode = Episode(
        episode_id="ep-1",
        dataset_id="demo",
        source_type="human_demo",
        robot=default_franka_card(),
        task=TaskCard(
            task_id="pick-place",
            instruction="pick up the red block",
            skill_tags=["pick", "place"],
            objects=["red block"],
            stages=["approach", "grasp"],
        ),
        action_semantics=semantics,
        streams={
            "action": StreamData(
                name="action",
                stream_type=StreamType.ACTION,
                fps=20.0,
                shape=[2, semantics.action_dim],
                dtype="float32",
                frame="robot_base",
                data=[[0.0] * semantics.action_dim, [0.1] * semantics.action_dim],
            )
        },
    )

    path = tmp_path / "episode.json"
    episode.to_json(path)

    restored = Episode.from_json(path)

    assert restored.episode_id == "ep-1"
    assert restored.action_semantics.action_dim == semantics.action_dim
    assert restored.num_steps == 2
    assert restored.get_stream("action").shape == [2, semantics.action_dim]


def test_action_semantics_rejects_mismatched_dim():
    with pytest.raises(ValueError):
        ActionSemantics(
            representation="eef_delta_pose_gripper",
            control_mode="delta",
            frame="robot_base",
            fields=["dx", "dy"],
            units=["m", "m"],
            fps=20.0,
            action_dim=7,
        )


def test_protocol_first_episode_serializes_scene_frame_graph_and_sensor_cards(tmp_path: Path):
    semantics = ActionSemantics.default_eef_delta()
    sensor = SensorCard(
        sensor_id="front",
        sensor_type="rgb",
        frame="camera_front",
        fps=30.0,
        intrinsics=CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0),
    )
    scene = SceneCard(
        scene_id="kitchen_table_001",
        objects={
            "red_block": ObjectCard(
                object_id="red_block",
                category="cube",
                affordances=["graspable", "movable"],
                frame="object_red_block",
                confidence=0.82,
            )
        },
        workspace_bounds=WorkspaceBounds(x=(-0.5, 0.5), y=(-0.4, 0.4), z=(0.0, 0.8)),
    )
    frame_graph = FrameGraph(
        frames=["world", "robot_base", "camera_front", "object_red_block"],
        transforms=[
            FrameTransform(
                parent="camera_front",
                child="robot_base",
                transform_type="calibrated",
                matrix=[
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
            FrameTransform(
                parent="object_red_block",
                child="camera_front",
                transform_type="perception_estimate",
                confidence=0.82,
            ),
        ],
    )

    episode = Episode(
        episode_id="ep-protocol",
        dataset_id="demo",
        source_type="robot_demo",
        robot=default_franka_card(),
        task=TaskCard(task_id="pick-place", instruction="put the red block in the bowl"),
        scene=scene,
        frame_graph=frame_graph,
        action_semantics=semantics,
        streams={
            "action": StreamData(
                name="action",
                stream_type=StreamType.ACTION,
                fps=20.0,
                shape=[1, semantics.action_dim],
                dtype="float32",
                frame="robot_base",
                data=[[0.0] * semantics.action_dim],
            )
        },
        annotations={"sensors": {"front": sensor.model_dump(mode="json")}},
    )

    path = tmp_path / "protocol_episode.json"
    episode.to_json(path)
    restored = Episode.from_json(path)

    assert restored.scene.scene_id == "kitchen_table_001"
    assert restored.scene.objects["red_block"].affordances == ["graspable", "movable"]
    assert restored.frame_graph.has_frame("camera_front")
    assert restored.frame_graph.get_transform("camera_front", "robot_base").transform_type == "calibrated"
    assert restored.annotations["sensors"]["front"]["intrinsics"]["width"] == 640


def test_protocol_schema_rejects_invalid_frame_graph_and_workspace_bounds():
    with pytest.raises(ValueError, match="unknown frame"):
        FrameGraph(
            frames=["robot_base"],
            transforms=[FrameTransform(parent="camera_front", child="robot_base", transform_type="calibrated")],
        )

    with pytest.raises(ValueError, match="lower bound"):
        WorkspaceBounds(x=(0.5, -0.5), y=(-0.4, 0.4), z=(0.0, 0.8))
