"""Base adapter interface.

This mirrors StarVLA's dataloader boundary: adapters emit model-agnostic
Universal Embodied IR episodes and do not tokenize, encode, or bind to a model.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from robo_train.schema.episode import Episode


class BaseAdapter(ABC):
    """Convert an external data source to and from Universal Embodied IR."""

    @abstractmethod
    def to_ir(self, source: Any) -> list[Episode]:
        """Convert an external source into IR episodes."""

    @abstractmethod
    def from_ir(self, episodes: list[Episode], output_path: Path) -> None:
        """Export IR episodes into the adapter's external format."""


DataAdapter = BaseAdapter
