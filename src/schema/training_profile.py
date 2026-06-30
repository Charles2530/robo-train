"""Training-family profiles for VLA, 3D policy, and world-model paths."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataProfile(str, Enum):
    """Model-family-specific data view names."""

    VLA = "vla"
    POLICY_3D = "policy_3d"
    WORLD_MODEL = "world_model"


class ModelFamily(str, Enum):
    """High-level model families supported by the training architecture."""

    VLA_POLICY = "vla_policy"
    POLICY_3D = "policy_3d"
    WORLD_ACTION_MODEL = "world_action_model"


class LossProfile(str, Enum):
    """Loss families matched to model families."""

    VLA_BC = "vla_bc"
    POLICY_3D_BC = "policy_3d_bc"
    WORLD_MODEL = "world_model"


class EmbodimentProfile(str, Enum):
    """Robot/embodiment families that affect action and runtime profiles."""

    FRANKA_SINGLE_ARM = "franka_single_arm"
    ALOHA_DUAL_ARM = "aloha_dual_arm"
    UR_SINGLE_ARM = "ur_single_arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    GENERIC = "generic"


class TrainingProfile(BaseModel):
    """Declares data/model/loss/embodiment compatibility for an experiment."""

    model_config = ConfigDict(use_enum_values=True)

    data_profile: DataProfile | str
    model_family: ModelFamily | str
    loss_profile: LossProfile | str
    embodiment_profile: EmbodimentProfile | str = EmbodimentProfile.GENERIC
    modalities: list[str] = Field(default_factory=list)
    action_space: str = "eef_delta_pose_gripper"
    predicts_future: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def validate_compatibility(self) -> "TrainingProfile":
        """Keep data views, model families, and losses aligned."""
        expected = {
            DataProfile.VLA.value: (ModelFamily.VLA_POLICY.value, LossProfile.VLA_BC.value),
            DataProfile.POLICY_3D.value: (ModelFamily.POLICY_3D.value, LossProfile.POLICY_3D_BC.value),
            DataProfile.WORLD_MODEL.value: (ModelFamily.WORLD_ACTION_MODEL.value, LossProfile.WORLD_MODEL.value),
        }
        model_family, loss_profile = expected[str(self.data_profile)]
        if self.model_family != model_family:
            raise ValueError(f"{self.data_profile} data_profile requires model_family={model_family!r}")
        if self.loss_profile != loss_profile:
            raise ValueError(f"{self.data_profile} data_profile requires loss_profile={loss_profile!r}")

        modalities = set(self.modalities)
        if self.data_profile == DataProfile.VLA.value and "language" not in modalities:
            raise ValueError("vla training requires language modality")
        if self.data_profile == DataProfile.POLICY_3D.value and "point_cloud" not in modalities:
            raise ValueError("policy_3d training requires point_cloud modality")
        if self.data_profile == DataProfile.WORLD_MODEL.value:
            if not self.predicts_future:
                raise ValueError("world_model training requires predicts_future=True")
            if "future" not in modalities:
                raise ValueError("world_model training requires future modality")
        return self

    @classmethod
    def default_vla(
        cls, embodiment_profile: EmbodimentProfile | str = EmbodimentProfile.GENERIC
    ) -> "TrainingProfile":
        """Return the default VLA behavior-cloning training profile."""
        return cls(
            data_profile=DataProfile.VLA,
            model_family=ModelFamily.VLA_POLICY,
            loss_profile=LossProfile.VLA_BC,
            embodiment_profile=embodiment_profile,
            modalities=["front_rgb", "wrist_rgb", "language", "state"],
        )

    @classmethod
    def default_policy3d(
        cls, embodiment_profile: EmbodimentProfile | str = EmbodimentProfile.GENERIC
    ) -> "TrainingProfile":
        """Return the default 3D policy training profile."""
        return cls(
            data_profile=DataProfile.POLICY_3D,
            model_family=ModelFamily.POLICY_3D,
            loss_profile=LossProfile.POLICY_3D_BC,
            embodiment_profile=embodiment_profile,
            modalities=["point_cloud", "state"],
        )

    @classmethod
    def default_world_model(
        cls, embodiment_profile: EmbodimentProfile | str = EmbodimentProfile.GENERIC
    ) -> "TrainingProfile":
        """Return the default world-action-model training profile."""
        return cls(
            data_profile=DataProfile.WORLD_MODEL,
            model_family=ModelFamily.WORLD_ACTION_MODEL,
            loss_profile=LossProfile.WORLD_MODEL,
            embodiment_profile=embodiment_profile,
            modalities=["front_rgb", "state", "action", "future"],
            predicts_future=True,
        )
