"""Minimal deterministic NumPy policy for training-infra smoke tests."""

from typing import Any

import numpy as np

from robo_train.data.processors.batch_builder import EmbodiedBatch


class UnifiedEmbodiedPolicy:
    """A small action policy that keeps the training path runnable."""

    def __init__(self, seed: int = 42, action_horizon: int = 8, latent_dim: int = 32) -> None:
        self.seed = seed
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        rng = np.random.default_rng(seed)
        self.action_projection = rng.normal(0.0, 0.05, size=(latent_dim, latent_dim)).astype(np.float32)
        self.future_projection = rng.normal(0.0, 0.03, size=(latent_dim, latent_dim)).astype(np.float32)

    def forward(self, batch: EmbodiedBatch) -> dict[str, Any]:
        """Return mock training predictions and scalar losses."""
        latent = self._encode_batch(batch)
        action_latent = np.tanh(latent @ self.action_projection)
        future_step = np.tanh(latent @ self.future_projection)
        pred_actions = self._predict_actions(action_latent, batch.action_semantics.action_dim)
        future_latent = np.repeat(future_step[:, None, :], self.action_horizon, axis=1).astype(np.float32)

        if batch.actions is not None:
            target = np.asarray(batch.actions, dtype=np.float32)
            action_loss = float(np.mean((pred_actions - target) ** 2))
        else:
            action_loss = float(np.mean(pred_actions**2))
        world_action_loss = float(np.mean(future_latent**2))
        loss = action_loss + 0.1 * world_action_loss

        return {
            "loss": loss,
            "action_loss": action_loss,
            "world_action_loss": world_action_loss,
            "stage_advantage_loss": 0.0,
            "safety_loss": 0.0,
            "pred_actions": pred_actions,
            "future_latent": future_latent,
            "point_features": self._point_features(batch),
        }

    def predict_action(self, batch: EmbodiedBatch) -> dict[str, Any]:
        """Return a JSON-friendly action chunk from an already built batch."""
        output = self.forward(batch)
        return {
            "actions": {"canonical": output["pred_actions"][0].tolist()},
            "action_semantics": batch.action_semantics.model_dump(mode="json"),
            "debug": {"used_modalities": [key for key, value in batch.modality_mask.items() if value]},
        }

    def _encode_batch(self, batch: EmbodiedBatch) -> np.ndarray:
        batch_size = len(batch.robot_cards)
        parts: list[np.ndarray] = []
        if batch.state is not None:
            state = np.asarray(batch.state, dtype=np.float32)
            parts.append(self._pad_or_trim(state.mean(axis=1), self.latent_dim))
        for images in batch.images.values():
            image_array = np.asarray(images, dtype=np.float32)
            image_features = image_array.reshape(batch_size, image_array.shape[1], -1).mean(axis=(1, 2), keepdims=False)
            parts.append(np.repeat(image_features[:, None] / 255.0, self.latent_dim, axis=1).astype(np.float32))
        if not parts:
            return np.zeros((batch_size, self.latent_dim), dtype=np.float32)
        return np.tanh(np.mean(parts, axis=0)).astype(np.float32)

    def _predict_actions(self, action_latent: np.ndarray, action_dim: int) -> np.ndarray:
        action_base = self._pad_or_trim(action_latent, action_dim)
        steps = np.linspace(1.0, 0.5, self.action_horizon, dtype=np.float32)
        return np.stack([action_base * step for step in steps], axis=1).astype(np.float32)

    def _point_features(self, batch: EmbodiedBatch) -> np.ndarray:
        if batch.state is None:
            batch_size = len(batch.robot_cards)
            return np.zeros((batch_size, self.action_horizon, 3), dtype=np.float32)
        state = np.asarray(batch.state, dtype=np.float32)
        xyz = state[:, : self.action_horizon, :3]
        if xyz.shape[1] < self.action_horizon:
            pad = np.repeat(xyz[:, -1:, :], self.action_horizon - xyz.shape[1], axis=1)
            xyz = np.concatenate([xyz, pad], axis=1)
        return xyz.astype(np.float32)

    @staticmethod
    def _pad_or_trim(array: np.ndarray, dim: int) -> np.ndarray:
        if array.shape[-1] == dim:
            return array.astype(np.float32)
        if array.shape[-1] > dim:
            return array[..., :dim].astype(np.float32)
        pad = np.zeros((*array.shape[:-1], dim - array.shape[-1]), dtype=np.float32)
        return np.concatenate([array, pad], axis=-1).astype(np.float32)
