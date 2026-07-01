"""State and action normalization processor."""

import numpy as np

from robo_train.data.processors.base import EpisodeProcessor
from robo_train.schema.episode import Episode


class NormalizationProcessor(EpisodeProcessor):
    """Fit mean/std statistics and normalize state and canonical action streams."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.stats: dict[str, dict[str, list[float]]] = {}

    def fit(self, episodes: list[Episode]) -> None:
        """Fit normalization statistics from available state and action streams."""
        for name in ("state", "action"):
            arrays = [episode.streams[name].as_array("float32") for episode in episodes if name in episode.streams]
            if not arrays:
                continue
            values = np.concatenate(arrays, axis=0)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std = np.where(std < self.epsilon, 1.0, std)
            self.stats[name] = {"mean": mean.tolist(), "std": std.tolist()}

    def transform(self, episode: Episode) -> Episode:
        """Normalize state and action streams when stats are fitted."""
        normalized = episode.model_copy(deep=True)
        for name, stats in self.stats.items():
            if name not in normalized.streams:
                continue
            array = normalized.streams[name].as_array("float32")
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            normalized.streams[name].data = ((array - mean) / std).astype(np.float32).tolist()
        return normalized
