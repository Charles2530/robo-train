"""Deterministic mixed episode dataloader."""

from collections.abc import Mapping

import numpy as np

from robo_train.schema.episode import Episode


class MixedEpisodeDataLoader:
    """Sample episodes from multiple in-memory datasets with source weights."""

    def __init__(
        self,
        datasets: Mapping[str, list[Episode]],
        batch_size: int,
        weights: Mapping[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not datasets:
            raise ValueError("datasets must be non-empty")
        self.datasets = {name: list(episodes) for name, episodes in datasets.items()}
        for name, episodes in self.datasets.items():
            if not episodes:
                raise ValueError(f"dataset {name!r} must contain at least one episode")
        self.batch_size = batch_size
        self.source_names = list(self.datasets)
        raw_weights = np.asarray([float((weights or {}).get(name, 1.0)) for name in self.source_names], dtype=np.float64)
        if np.any(raw_weights <= 0):
            raise ValueError("all dataset weights must be positive")
        self.probabilities = raw_weights / raw_weights.sum()
        self.rng = np.random.default_rng(seed)
        self.cursors = {name: 0 for name in self.source_names}
        self.sampled_sources: list[str] = []

    def next_batch(self) -> list[Episode]:
        """Return the next mixed batch."""
        batch: list[Episode] = []
        for _ in range(self.batch_size):
            source = str(self.rng.choice(self.source_names, p=self.probabilities))
            episodes = self.datasets[source]
            cursor = self.cursors[source]
            batch.append(episodes[cursor % len(episodes)])
            self.cursors[source] = cursor + 1
            self.sampled_sources.append(source)
        return batch
