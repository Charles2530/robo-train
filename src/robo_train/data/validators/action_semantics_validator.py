"""Action stream validation against declared action semantics."""

import numpy as np

from robo_train.schema.episode import Episode
from robo_train.data.validators.base_validator import ValidationReport, Validator


class ActionSemanticsValidator(Validator):
    """Check canonical action shape and fps match `ActionSemantics`."""

    def __init__(self, fps_tolerance: float = 1e-3) -> None:
        self.fps_tolerance = fps_tolerance

    def validate(self, episode: Episode) -> ValidationReport:
        report = ValidationReport()
        if "action" not in episode.streams:
            report.add_error(f"{episode.episode_id}: missing canonical action stream")
            return report

        stream = episode.streams["action"]
        data = stream.as_array()
        if data.ndim != 2:
            report.add_error(f"{episode.episode_id}: action stream must be 2D [T, action_dim]")
        elif int(data.shape[-1]) != episode.action_semantics.action_dim:
            report.add_error(
                f"{episode.episode_id}: action_dim mismatch, stream has {data.shape[-1]} "
                f"but semantics declares {episode.action_semantics.action_dim}"
            )
        if np.isfinite(stream.fps) and abs(stream.fps - episode.action_semantics.fps) > self.fps_tolerance:
            report.add_error(
                f"{episode.episode_id}: action fps {stream.fps} does not match semantics fps "
                f"{episode.action_semantics.fps}"
            )
        return report
