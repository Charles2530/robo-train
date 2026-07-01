"""Registry for framework plugins."""

from __future__ import annotations

from robo_train.frameworks.base import FrameworkPlugin


class FrameworkRegistry:
    """Small name-based registry for train framework plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, FrameworkPlugin] = {}

    def register(self, plugin: FrameworkPlugin) -> FrameworkPlugin:
        if plugin.name in self._plugins:
            raise ValueError(f"framework plugin already registered: {plugin.name}")
        self._plugins[plugin.name] = plugin
        return plugin

    def get(self, name: str) -> FrameworkPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._plugins)) or "<none>"
            raise KeyError(f"unknown framework {name!r}; known frameworks: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._plugins)


FRAMEWORK_REGISTRY = FrameworkRegistry()
