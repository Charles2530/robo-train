"""Dataset shard metadata for file-backed data infra."""

from pathlib import Path

from pydantic import BaseModel, Field


class DatasetShardSpec(BaseModel):
    """A small file shard descriptor used by data preparation scripts."""

    path: str
    format: str = "json"
    num_episodes: int = Field(default=0, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def suffix(self) -> str:
        """Return the file suffix without a leading dot."""
        return Path(self.path).suffix.lstrip(".")
