"""Normalization statistics for episode streams."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.schema.episode import Episode


class StreamNormStats(BaseModel):
    """Mean and std for one numeric stream."""

    mean: list[float]
    std: list[float]
    count: int = Field(ge=0)


class NormStats(BaseModel):
    """Normalization statistics keyed by stream name."""

    streams: dict[str, StreamNormStats]
    epsilon: float = 1e-6

    @classmethod
    def fit(cls, episodes: list[Episode], stream_names: list[str]) -> "NormStats":
        """Fit normalization statistics from inline episode stream data."""
        stats: dict[str, StreamNormStats] = {}
        for stream_name in stream_names:
            arrays = [
                episode.streams[stream_name].as_array("float32")
                for episode in episodes
                if stream_name in episode.streams and episode.streams[stream_name].data is not None
            ]
            if not arrays:
                continue
            matrix = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays], axis=0)
            std = matrix.std(axis=0)
            std = np.where(std < cls.model_fields["epsilon"].default, 1.0, std)
            stats[stream_name] = StreamNormStats(
                mean=matrix.mean(axis=0).astype(float).tolist(),
                std=std.astype(float).tolist(),
                count=int(matrix.shape[0]),
            )
        return cls(streams=stats)

    def normalize_episode(self, episode: Episode) -> Episode:
        """Return a deep-copied episode with known numeric streams normalized."""
        normalized = episode.model_copy(deep=True)
        for stream_name, stats in self.streams.items():
            if stream_name not in normalized.streams:
                continue
            stream = normalized.streams[stream_name]
            if stream.data is None:
                continue
            array = stream.as_array("float32")
            mean = np.asarray(stats.mean, dtype=np.float32)
            std = np.asarray(stats.std, dtype=np.float32)
            stream.data = ((array - mean) / (std + self.epsilon)).astype(np.float32).tolist()
        return normalized

    def to_json(self, path: str | Path) -> None:
        """Save stats as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "NormStats":
        """Load stats from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
