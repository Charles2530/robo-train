"""HDF5 adapter stub for future dataset integration."""

from pathlib import Path
from typing import Any

from src.data.adapters.base_adapter import BaseAdapter
from src.data.adapters.registry import AdapterRegistry
from src.schema.episode import Episode


class HDF5Adapter(BaseAdapter):
    """Placeholder for mapping HDF5 robot datasets into Universal Embodied IR."""

    def to_ir(self, source: Any) -> list[Episode]:
        raise NotImplementedError("HDF5Adapter should map HDF5 groups into Episode streams and cards.")

    def from_ir(self, episodes: list[Episode], output_path: Path) -> None:
        raise NotImplementedError("HDF5Adapter should write Episode streams and metadata to HDF5 groups.")


AdapterRegistry.register("hdf5", HDF5Adapter)
