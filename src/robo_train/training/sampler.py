"""Simple deterministic episode sampler."""

import numpy as np

from robo_train.schema.episode import Episode


class EpisodeSampler:
    """Sample episodes reproducibly."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(self, episodes: list[Episode], count: int) -> list[Episode]:
        """Sample up to `count` episodes without replacement."""
        if count >= len(episodes):
            return list(episodes)
        indices = self.rng.choice(len(episodes), size=count, replace=False)
        return [episodes[int(index)] for index in indices]
