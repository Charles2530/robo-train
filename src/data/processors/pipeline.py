"""Processor pipeline from Episode IR to model-ready batches."""

from src.data.processors.batch_builder import BatchBuilder, EmbodiedBatch
from src.data.processors.modality_select import ModalitySelectProcessor
from src.data.processors.normalize import NormalizationProcessor
from src.data.processors.time_sync import TimeSyncProcessor
from src.schema.episode import Episode


class ProcessorPipeline:
    """Composable sync, normalization, modality selection, and batch building."""

    def __init__(
        self,
        sync: TimeSyncProcessor,
        normalize: NormalizationProcessor,
        batch_builder: BatchBuilder,
    ) -> None:
        self.sync = sync
        self.normalize = normalize
        self.batch_builder = batch_builder

    @classmethod
    def default(cls, history_window: int = 8, action_horizon: int = 8, policy_fps: float = 20.0) -> "ProcessorPipeline":
        """Create the default lightweight processor pipeline."""
        selector = ModalitySelectProcessor()
        return cls(
            sync=TimeSyncProcessor(policy_fps=policy_fps),
            normalize=NormalizationProcessor(),
            batch_builder=BatchBuilder(
                history_window=history_window,
                action_horizon=action_horizon,
                modality_selector=selector,
            ),
        )

    def fit(self, episodes: list[Episode]) -> None:
        """Fit processor state from episodes."""
        synced = [self.sync.transform(episode) for episode in episodes]
        self.normalize.fit(synced)

    def transform(self, episodes: list[Episode]) -> EmbodiedBatch:
        """Transform episodes into one model-ready batch."""
        processed = [self.normalize.transform(self.sync.transform(episode)) for episode in episodes]
        return self.batch_builder.build(processed)
