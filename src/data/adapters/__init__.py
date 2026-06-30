"""Dataset adapters that map external formats to Universal Embodied IR."""

from src.data.adapters.base_adapter import BaseAdapter, DataAdapter
from src.data.adapters.registry import AdapterRegistry
from src.data.adapters.synthetic_adapter import SyntheticAdapter

__all__ = ["AdapterRegistry", "BaseAdapter", "DataAdapter", "SyntheticAdapter"]
