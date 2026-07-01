"""Simple time synchronization processor."""

from copy import deepcopy

from robo_train.data.processors.base import EpisodeProcessor
from robo_train.schema.episode import Episode


class TimeSyncProcessor(EpisodeProcessor):
    """Truncate time-indexed streams to a common policy horizon."""

    def __init__(self, policy_fps: float = 20.0) -> None:
        self.policy_fps = policy_fps

    def transform(self, episode: Episode) -> Episode:
        """Return a copy with time-indexed streams truncated to the shortest length."""
        synced = episode.model_copy(deep=True)
        lengths = [
            stream.first_dim()
            for stream in synced.streams.values()
            if stream.stream_type != "language" and stream.first_dim() is not None and stream.first_dim() > 0
        ]
        if not lengths:
            return synced
        horizon = min(lengths)
        for stream in synced.streams.values():
            if stream.stream_type == "language" or stream.first_dim() is None or stream.first_dim() <= horizon:
                continue
            stream.data = deepcopy(stream.data[:horizon]) if stream.data is not None else None
            stream.shape[0] = horizon
            if stream.timestamps is not None:
                stream.timestamps = stream.timestamps[:horizon]
        return synced
