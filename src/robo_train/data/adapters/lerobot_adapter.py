"""LeRobot adapter stub for future dataset integration."""

from pathlib import Path
from typing import Any

from robo_train.data.adapters.base_adapter import BaseAdapter
from robo_train.data.adapters.registry import AdapterRegistry
from robo_train.schema.episode import Episode


class LeRobotAdapter(BaseAdapter):
    """Placeholder for converting LeRobot v2/v3 datasets into UEFS IR.

    A real implementation would map LeRobot observations to `StreamData`,
    action arrays to canonical/raw action streams, and dataset task metadata to
    `TaskCard` while preserving robot metadata in `RobotCard`.
    """

    def to_ir(self, source: Any) -> list[Episode]:
        raise NotImplementedError("LeRobotAdapter should map LeRobot observations/actions into Episode IR.")

    def from_ir(self, episodes: list[Episode], output_path: Path) -> None:
        raise NotImplementedError("LeRobotAdapter should export IR into a LeRobot-compatible layout.")


AdapterRegistry.register("lerobot", LeRobotAdapter)
