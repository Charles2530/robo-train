"""Robot metadata used to condition policies and validate action compatibility."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbodimentType(str, Enum):
    """High-level robot embodiment families."""

    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"


class RobotCard(BaseModel):
    """Portable robot description for data, model conditioning, and runtime."""

    model_config = ConfigDict(use_enum_values=True)

    robot_id: str
    embodiment_type: EmbodimentType | str
    joint_names: list[str]
    joint_limits: dict[str, tuple[float, float]]
    eef_names: list[str]
    sensors: dict[str, Any] = Field(default_factory=dict)
    controller: dict[str, Any] = Field(default_factory=dict)
    supported_action_representations: list[str]
    control_frequency: float = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_joint_limits(self) -> "RobotCard":
        """Ensure every limited joint exists and each limit is ordered."""
        known = set(self.joint_names)
        for joint, limits in self.joint_limits.items():
            if joint not in known:
                raise ValueError(f"joint_limits contains unknown joint {joint!r}")
            lower, upper = limits
            if lower >= upper:
                raise ValueError(f"joint limit for {joint!r} must be ordered")
        return self


def default_aloha_dual_arm_card() -> RobotCard:
    """Return a compact dual-arm ALOHA-like robot card."""
    joints = [f"left_j{i}" for i in range(7)] + [f"right_j{i}" for i in range(7)]
    return RobotCard(
        robot_id="aloha_dual_arm_mock",
        embodiment_type=EmbodimentType.DUAL_ARM,
        joint_names=joints,
        joint_limits={name: (-3.14, 3.14) for name in joints},
        eef_names=["left_eef", "right_eef"],
        sensors={"front_rgb": "camera", "wrist_rgb": "camera", "proprioception": "joint_state"},
        controller={"type": "eef_delta", "rate_hz": 50.0},
        supported_action_representations=[
            "eef_delta_pose_gripper",
            "eef_absolute_pose_gripper",
            "joint_position",
        ],
        control_frequency=50.0,
        metadata={"vendor": "mock", "family": "aloha"},
    )


def default_franka_card() -> RobotCard:
    """Return a compact Franka-like single-arm robot card."""
    joints = [f"panda_joint{i}" for i in range(1, 8)]
    return RobotCard(
        robot_id="franka_panda_mock",
        embodiment_type=EmbodimentType.SINGLE_ARM,
        joint_names=joints,
        joint_limits={name: (-2.9, 2.9) for name in joints},
        eef_names=["panda_hand"],
        sensors={"front_rgb": "camera", "wrist_rgb": "camera", "proprioception": "joint_state"},
        controller={"type": "eef_delta", "rate_hz": 20.0},
        supported_action_representations=[
            "eef_delta_pose_gripper",
            "eef_absolute_pose_gripper",
            "joint_position",
            "joint_velocity",
        ],
        control_frequency=20.0,
        metadata={"vendor": "mock", "family": "franka"},
    )
