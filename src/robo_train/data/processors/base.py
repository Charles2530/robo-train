"""Minimal processor interface for episode-level transforms."""

from abc import ABC

from robo_train.schema.episode import Episode


class EpisodeProcessor(ABC):
    """Optional fit/transform interface for UEFS processors."""

    def fit(self, episodes: list[Episode]) -> None:
        """Fit processor state from episodes."""

    def transform(self, episode: Episode) -> Episode:
        """Transform one episode."""
        return episode
