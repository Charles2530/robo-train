"""Dataset adapters that map external formats to Universal Embodied IR."""

from robo_train.data.adapters.base_adapter import BaseAdapter, DataAdapter
from robo_train.data.adapters.registry import AdapterRegistry
from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter

__all__ = ["AdapterRegistry", "BaseAdapter", "DataAdapter", "SyntheticAdapter"]
