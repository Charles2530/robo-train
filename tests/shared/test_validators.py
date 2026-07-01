from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter
from robo_train.data.validators.action_semantics_validator import ActionSemanticsValidator
from robo_train.data.validators.base_validator import ValidatorPipeline
from robo_train.data.validators.modality_validator import MissingModalityValidator
from robo_train.data.validators.robot_compatibility_validator import RobotCompatibilityValidator
from robo_train.data.validators.timestamp_validator import TimestampValidator


def _pipeline():
    return ValidatorPipeline(
        [
            TimestampValidator(),
            ActionSemanticsValidator(),
            RobotCompatibilityValidator(),
            MissingModalityValidator(),
        ]
    )


def test_valid_episode_passes_validator_pipeline():
    episode = SyntheticAdapter(seed=1).to_ir({"episodes": 1})[0]

    report = _pipeline().validate(episode)

    assert report.passed
    assert report.errors == []


def test_wrong_action_dim_fails_validator():
    episode = SyntheticAdapter(seed=1).to_ir({"episodes": 1})[0]
    episode.streams["action"].data = [[0.0] * 6 for _ in range(episode.num_steps)]
    episode.streams["action"].shape = [episode.num_steps, 6]

    report = _pipeline().validate(episode)

    assert not report.passed
    assert any("action_dim" in error for error in report.errors)


def test_incompatible_robot_action_semantics_fails_validator():
    episode = SyntheticAdapter(seed=1).to_ir({"episodes": 1})[0]
    episode.robot.supported_action_representations = ["joint_position"]

    report = _pipeline().validate(episode)

    assert not report.passed
    assert any("not supported" in error for error in report.errors)
