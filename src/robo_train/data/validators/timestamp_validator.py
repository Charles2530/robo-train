"""Timestamp validation for stream monotonicity and length."""

from robo_train.schema.episode import Episode
from robo_train.data.validators.base_validator import ValidationReport, Validator


class TimestampValidator(Validator):
    """Check stream timestamps are monotonic and match stream length."""

    def validate(self, episode: Episode) -> ValidationReport:
        report = ValidationReport()
        for name, stream in episode.streams.items():
            if stream.timestamps is None:
                continue
            expected = stream.first_dim()
            if expected is not None and len(stream.timestamps) != expected:
                report.add_error(
                    f"{episode.episode_id}:{name} timestamps length {len(stream.timestamps)} "
                    f"does not match data length {expected}"
                )
            if any(curr <= prev for prev, curr in zip(stream.timestamps, stream.timestamps[1:])):
                report.add_error(f"{episode.episode_id}:{name} timestamps must be strictly increasing")
        return report
