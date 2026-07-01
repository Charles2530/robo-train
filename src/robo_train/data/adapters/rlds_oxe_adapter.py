"""RLDS/OXE adapter stub for future dataset integration."""

from pathlib import Path
from typing import Any

from robo_train.data.adapters.base_adapter import BaseAdapter
from robo_train.data.adapters.registry import AdapterRegistry
from robo_train.schema.episode import Episode


class RLDSOXEAdapter(BaseAdapter):
    """Placeholder for mapping Open X-Embodiment / RLDS examples into UEFS IR."""

    def to_ir(self, source: Any) -> list[Episode]:
        raise NotImplementedError("RLDSOXEAdapter should map RLDS steps into Episode streams and labels.")

    def from_ir(self, episodes: list[Episode], output_path: Path) -> None:
        raise NotImplementedError("RLDSOXEAdapter should write IR into RLDS/OXE-style examples.")


AdapterRegistry.register("rlds_oxe", RLDSOXEAdapter)
