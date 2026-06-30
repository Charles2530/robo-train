"""Validation for required basic modalities."""

from src.schema.episode import Episode
from src.data.validators.base_validator import ValidationReport, Validator


class MissingModalityValidator(Validator):
    """Ensure an episode has at least perception or state plus instruction."""

    def validate(self, episode: Episode) -> ValidationReport:
        report = ValidationReport()
        has_vision = any(stream.stream_type in {"rgb_video", "depth_video", "pointcloud_sequence"} for stream in episode.streams.values())
        has_state = any(stream.stream_type == "proprioception" for stream in episode.streams.values())
        if not has_vision and not has_state:
            report.add_error(f"{episode.episode_id}: missing both visual and proprioception streams")
        if not episode.task.instruction.strip():
            report.add_error(f"{episode.episode_id}: task instruction is empty")
        return report
