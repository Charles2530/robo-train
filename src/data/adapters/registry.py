"""Adapter registry for dataset plugins."""

from typing import Any, TypeVar

from src.data.adapters.base_adapter import BaseAdapter


AdapterT = TypeVar("AdapterT", bound=BaseAdapter)


class AdapterRegistry:
    """Small name-to-adapter registry for dataset plugins."""

    _registry: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[AdapterT]) -> None:
        """Register an adapter class."""
        cls._registry[name] = adapter_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseAdapter:
        """Create a registered adapter by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise KeyError(f"unknown adapter {name!r}; available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def names(cls) -> list[str]:
        """Return registered adapter names."""
        return sorted(cls._registry)
