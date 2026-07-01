"""Robot/action compatibility validation."""

from robo_train.schema.episode import Episode
from robo_train.data.validators.base_validator import ValidationReport, Validator


class RobotCompatibilityValidator(Validator):
    """Check that the robot supports the episode action representation and rate."""

    def validate(self, episode: Episode) -> ValidationReport:
        report = ValidationReport()
        representation = str(episode.action_semantics.representation)
        if representation not in episode.robot.supported_action_representations:
            report.add_error(
                f"{episode.episode_id}: action representation {representation!r} is not supported "
                f"by robot {episode.robot.robot_id!r}"
            )
        if episode.robot.control_frequency < episode.action_semantics.fps:
            report.add_error(
                f"{episode.episode_id}: robot control frequency {episode.robot.control_frequency} "
                f"is below action fps {episode.action_semantics.fps}"
            )
        return report
