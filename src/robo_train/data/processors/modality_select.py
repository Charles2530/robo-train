"""Modality selection for model inputs."""

from dataclasses import dataclass

from robo_train.schema.episode import Episode


@dataclass
class ModalitySelectProcessor:
    """Select optional modalities while retaining explicit masks."""

    image_streams: tuple[str, ...] = ("front_rgb", "wrist_rgb")
    use_state: bool = True
    use_tactile: bool = True

    def modality_mask(self, episode: Episode) -> dict[str, bool]:
        """Return which requested modalities are present."""
        mask = {name: name in episode.streams for name in self.image_streams}
        mask["state"] = self.use_state and "state" in episode.streams
        mask["tactile"] = self.use_tactile and "tactile" in episode.streams
        return mask
