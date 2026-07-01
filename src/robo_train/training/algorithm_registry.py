"""Registries for training algorithms and policy factories."""

from collections.abc import Callable

from robo_train.schema.experiment_config import PolicyConfig
from robo_train.training.models.policy import UnifiedEmbodiedPolicy

PolicyFactory = Callable[[PolicyConfig, int], UnifiedEmbodiedPolicy]


class AlgorithmRegistry:
    """Small registry for policy and algorithm families."""

    def __init__(self) -> None:
        self._policy_factories: dict[str, PolicyFactory] = {}

    def register_policy(self, name: str, factory: PolicyFactory) -> None:
        """Register a named policy factory."""
        normalized = name.strip()
        if not normalized:
            raise ValueError("policy name must be non-empty")
        self._policy_factories[normalized] = factory

    def create_policy(self, config: PolicyConfig, seed: int = 42) -> UnifiedEmbodiedPolicy:
        """Create a policy from config."""
        if config.policy_type not in self._policy_factories:
            known = ", ".join(sorted(self._policy_factories)) or "<none>"
            raise KeyError(f"unknown policy type {config.policy_type!r}; known policies: {known}")
        return self._policy_factories[config.policy_type](config, seed)

    @property
    def policy_types(self) -> list[str]:
        """Return registered policy type names."""
        return sorted(self._policy_factories)


def _build_unified_mock(config: PolicyConfig, seed: int) -> UnifiedEmbodiedPolicy:
    return UnifiedEmbodiedPolicy(
        seed=seed,
        action_horizon=config.action_horizon,
        latent_dim=config.latent_dim,
    )


DEFAULT_ALGORITHM_REGISTRY = AlgorithmRegistry()
DEFAULT_ALGORITHM_REGISTRY.register_policy("unified_mock", _build_unified_mock)


def build_policy_from_config(config: PolicyConfig, seed: int = 42) -> UnifiedEmbodiedPolicy:
    """Build a policy using the default registry."""
    return DEFAULT_ALGORITHM_REGISTRY.create_policy(config=config, seed=seed)
