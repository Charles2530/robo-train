import pytest

from robo_train.schema.training_profile import (
    DataProfile,
    EmbodimentProfile,
    LossProfile,
    ModelFamily,
    TrainingProfile,
)


def test_training_profile_defaults_are_compatible():
    vla = TrainingProfile.default_vla(embodiment_profile=EmbodimentProfile.FRANKA_SINGLE_ARM)
    policy3d = TrainingProfile.default_policy3d(embodiment_profile=EmbodimentProfile.FRANKA_SINGLE_ARM)
    world_model = TrainingProfile.default_world_model(embodiment_profile=EmbodimentProfile.ALOHA_DUAL_ARM)

    assert vla.data_profile == DataProfile.VLA
    assert vla.model_family == ModelFamily.VLA_POLICY
    assert vla.loss_profile == LossProfile.VLA_BC
    assert "language" in vla.modalities

    assert policy3d.data_profile == DataProfile.POLICY_3D
    assert policy3d.model_family == ModelFamily.POLICY_3D
    assert "point_cloud" in policy3d.modalities

    assert world_model.data_profile == DataProfile.WORLD_MODEL
    assert world_model.model_family == ModelFamily.WORLD_ACTION_MODEL
    assert world_model.loss_profile == LossProfile.WORLD_MODEL
    assert world_model.predicts_future is True


def test_training_profile_rejects_mismatched_data_model_loss():
    with pytest.raises(ValueError, match="vla"):
        TrainingProfile(
            data_profile=DataProfile.VLA,
            model_family=ModelFamily.POLICY_3D,
            loss_profile=LossProfile.VLA_BC,
            embodiment_profile=EmbodimentProfile.FRANKA_SINGLE_ARM,
            modalities=["front_rgb", "language", "state"],
        )

    with pytest.raises(ValueError, match="world_model"):
        TrainingProfile(
            data_profile=DataProfile.WORLD_MODEL,
            model_family=ModelFamily.WORLD_ACTION_MODEL,
            loss_profile=LossProfile.POLICY_3D_BC,
            embodiment_profile=EmbodimentProfile.FRANKA_SINGLE_ARM,
            modalities=["front_rgb", "state", "action"],
        )


def test_training_profile_rejects_required_modality_gaps():
    with pytest.raises(ValueError, match="language"):
        TrainingProfile(
            data_profile=DataProfile.VLA,
            model_family=ModelFamily.VLA_POLICY,
            loss_profile=LossProfile.VLA_BC,
            embodiment_profile=EmbodimentProfile.FRANKA_SINGLE_ARM,
            modalities=["front_rgb", "state"],
        )

    with pytest.raises(ValueError, match="future"):
        TrainingProfile(
            data_profile=DataProfile.WORLD_MODEL,
            model_family=ModelFamily.WORLD_ACTION_MODEL,
            loss_profile=LossProfile.WORLD_MODEL,
            embodiment_profile=EmbodimentProfile.FRANKA_SINGLE_ARM,
            modalities=["front_rgb", "state", "action"],
        )


def test_family_config_samples_validate():
    samples = [
        TrainingProfile.default_vla(),
        TrainingProfile.default_policy3d(),
        TrainingProfile.default_world_model(),
    ]
    expected = [
        (DataProfile.VLA, LossProfile.VLA_BC),
        (DataProfile.POLICY_3D, LossProfile.POLICY_3D_BC),
        (DataProfile.WORLD_MODEL, LossProfile.WORLD_MODEL),
    ]

    for profile, (data_profile, loss_profile) in zip(samples, expected, strict=True):
        assert profile.data_profile == data_profile
        assert profile.loss_profile == loss_profile
        assert profile.embodiment_profile
