"""Registry for model-family-specific loss profiles."""

from collections.abc import Callable
from typing import Any

import numpy as np

from robo_train.schema.training_profile import LossProfile
from robo_train.training.losses import mse

LossFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, float]]


def compute_vla_bc_loss(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    """Compute a simple VLA behavior-cloning loss."""
    action_loss = mse(np.asarray(prediction["pred_actions"]), np.asarray(target["actions"]))
    language_conditioning_loss = float(target.get("language_conditioning_loss", 0.0))
    return {
        "loss": float(action_loss + 0.01 * language_conditioning_loss),
        "action_loss": float(action_loss),
        "language_conditioning_loss": language_conditioning_loss,
    }


def compute_policy3d_bc_loss(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    """Compute action and geometry consistency losses for 3D policies."""
    action_loss = mse(np.asarray(prediction["pred_actions"]), np.asarray(target["actions"]))
    geometry_loss = mse(np.asarray(prediction["point_features"]), np.asarray(target["point_cloud"]))
    return {
        "loss": float(action_loss + 0.1 * geometry_loss),
        "action_loss": float(action_loss),
        "geometry_consistency_loss": float(geometry_loss),
    }


def compute_world_model_loss(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    """Compute action and future-prediction losses for world-model training."""
    action_loss = mse(np.asarray(prediction["pred_actions"]), np.asarray(target["actions"]))
    future_loss = mse(np.asarray(prediction["future_latent"]), np.asarray(target["future_state_target"]))
    return {
        "loss": float(action_loss + 0.5 * future_loss),
        "action_loss": float(action_loss),
        "future_prediction_loss": float(future_loss),
    }


class LossRegistry:
    """Small name-to-loss registry."""

    def __init__(self) -> None:
        self._registry: dict[str, LossFn] = {}

    def register(self, name: str, loss_fn: LossFn) -> None:
        """Register a loss function by profile name."""
        normalized = name.strip()
        if not normalized:
            raise ValueError("loss profile name must be non-empty")
        self._registry[normalized] = loss_fn

    def compute(self, name: LossProfile | str, prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
        """Compute a registered loss profile."""
        normalized = str(name.value if isinstance(name, LossProfile) else name)
        if normalized not in self._registry:
            known = ", ".join(sorted(self._registry)) or "<none>"
            raise KeyError(f"unknown loss profile {normalized!r}; known profiles: {known}")
        return self._registry[normalized](prediction, target)

    @property
    def names(self) -> list[str]:
        """Return registered loss profile names."""
        return sorted(self._registry)


DEFAULT_LOSS_REGISTRY = LossRegistry()
DEFAULT_LOSS_REGISTRY.register(LossProfile.VLA_BC.value, compute_vla_bc_loss)
DEFAULT_LOSS_REGISTRY.register(LossProfile.POLICY_3D_BC.value, compute_policy3d_bc_loss)
DEFAULT_LOSS_REGISTRY.register(LossProfile.WORLD_MODEL.value, compute_world_model_loss)


def compute_loss_profile(
    loss_profile: LossProfile | str, prediction: dict[str, Any], target: dict[str, Any]
) -> dict[str, float]:
    """Compute a loss using the default registry."""
    return DEFAULT_LOSS_REGISTRY.compute(loss_profile, prediction, target)
