"""Action semantics for canonical and raw robot action streams."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionRepresentation(str, Enum):
    """Supported action representation names."""

    EEF_DELTA_POSE_GRIPPER = "eef_delta_pose_gripper"
    EEF_ABSOLUTE_POSE_GRIPPER = "eef_absolute_pose_gripper"
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    BASE_VELOCITY = "base_velocity"
    RAW_VENDOR_SPECIFIC = "raw_vendor_specific"


class ControlMode(str, Enum):
    """Supported controller modes."""

    DELTA = "delta"
    ABSOLUTE = "absolute"
    VELOCITY = "velocity"
    TORQUE = "torque"
    HYBRID = "hybrid"


class ActionSemantics(BaseModel):
    """Describes the meaning, frame, units, rate, and dimensionality of actions."""

    model_config = ConfigDict(use_enum_values=True)

    representation: ActionRepresentation | str
    control_mode: ControlMode | str
    frame: str
    fields: list[str]
    units: list[str]
    fps: float = Field(gt=0)
    horizon: int | None = None
    action_dim: int = Field(gt=0)
    gripper_convention: str | None = None

    @model_validator(mode="after")
    def validate_lengths(self) -> "ActionSemantics":
        """Ensure field and unit metadata matches the action dimensionality."""
        if len(self.fields) != self.action_dim:
            raise ValueError("len(fields) must equal action_dim")
        if len(self.units) != self.action_dim:
            raise ValueError("len(units) must equal action_dim")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be positive when provided")
        return self

    @classmethod
    def default_eef_delta(cls, fps: float = 20.0, horizon: int | None = 8) -> "ActionSemantics":
        """Return the default 7D end-effector delta pose plus gripper semantics."""
        return cls(
            representation=ActionRepresentation.EEF_DELTA_POSE_GRIPPER,
            control_mode=ControlMode.DELTA,
            frame="robot_base",
            fields=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            units=["m", "m", "m", "rad", "rad", "rad", "normalized"],
            fps=fps,
            horizon=horizon,
            action_dim=7,
            gripper_convention="0=open,1=closed",
        )
