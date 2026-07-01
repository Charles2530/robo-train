"""Framework plugin interfaces and registry."""

from robo_train.frameworks.base import FrameworkPlugin
from robo_train.frameworks.registry import FRAMEWORK_REGISTRY, FrameworkRegistry
from robo_train.frameworks import kai0 as kai0

__all__ = ["FRAMEWORK_REGISTRY", "FrameworkPlugin", "FrameworkRegistry"]
