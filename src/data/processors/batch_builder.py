"""Batch construction from Universal Embodied IR episodes."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.processors.modality_select import ModalitySelectProcessor
from src.schema.action_semantics import ActionSemantics
from src.schema.episode import Episode
from src.schema.robot_card import RobotCard
from src.schema.task_card import TaskCard


@dataclass
class EmbodiedBatch:
    """Model-ready multimodal batch with action and metadata contracts."""

    images: dict[str, np.ndarray]
    state: np.ndarray | None
    tactile: np.ndarray | None
    language: list[str]
    actions: np.ndarray | None
    action_semantics: ActionSemantics
    robot_cards: list[RobotCard]
    task_cards: list[TaskCard]
    modality_mask: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchBuilder:
    """Build fixed-shape `EmbodiedBatch` objects from episodes."""

    def __init__(
        self,
        history_window: int = 8,
        action_horizon: int = 8,
        modality_selector: ModalitySelectProcessor | None = None,
    ) -> None:
        self.history_window = history_window
        self.action_horizon = action_horizon
        self.modality_selector = modality_selector or ModalitySelectProcessor()

    def build(self, episodes: list[Episode]) -> EmbodiedBatch:
        """Build one batch from the first window of each episode."""
        if not episodes:
            raise ValueError("BatchBuilder requires at least one episode")

        images: dict[str, list[np.ndarray]] = {name: [] for name in self.modality_selector.image_streams}
        states: list[np.ndarray] = []
        tactiles: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        languages: list[str] = []
        scene_cards = []
        frame_graphs = []
        aggregate_mask = {name: True for name in self.modality_selector.image_streams}
        aggregate_mask.update({"state": True, "tactile": True})

        for episode in episodes:
            mask = self.modality_selector.modality_mask(episode)
            for key, value in mask.items():
                aggregate_mask[key] = aggregate_mask.get(key, True) and value

            for name in self.modality_selector.image_streams:
                if name not in episode.streams:
                    continue
                images[name].append(self._first_steps(episode.streams[name].as_array(), self.history_window))

            if self.modality_selector.use_state and "state" in episode.streams:
                states.append(self._first_steps(episode.streams["state"].as_array("float32"), self.history_window))
            if self.modality_selector.use_tactile and "tactile" in episode.streams:
                tactiles.append(self._first_steps(episode.streams["tactile"].as_array("float32"), self.history_window))
            if "action" in episode.streams:
                actions.append(self._first_steps(episode.streams["action"].as_array("float32"), self.action_horizon))

            languages.append(episode.task.instruction)
            scene_cards.append(episode.scene)
            frame_graphs.append(episode.frame_graph)

        image_arrays = {name: np.stack(items, axis=0) for name, items in images.items() if items}
        state_array = np.stack(states, axis=0) if states else None
        tactile_array = np.stack(tactiles, axis=0) if tactiles else None
        action_array = np.stack(actions, axis=0) if actions else None

        return EmbodiedBatch(
            images=image_arrays,
            state=state_array,
            tactile=tactile_array,
            language=languages,
            actions=action_array,
            action_semantics=episodes[0].action_semantics,
            robot_cards=[episode.robot for episode in episodes],
            task_cards=[episode.task for episode in episodes],
            modality_mask=aggregate_mask,
            metadata={
                "history_window": self.history_window,
                "action_horizon": self.action_horizon,
                "scene_cards": scene_cards,
                "frame_graphs": frame_graphs,
            },
        )

    @staticmethod
    def _first_steps(array: np.ndarray, length: int) -> np.ndarray:
        """Return a fixed-length prefix, padding with the last step when needed."""
        if array.shape[0] >= length:
            return array[:length]
        if array.shape[0] == 0:
            raise ValueError("cannot build a fixed window from an empty stream")
        pad_count = length - array.shape[0]
        pad = np.repeat(array[-1:], pad_count, axis=0)
        return np.concatenate([array, pad], axis=0)
