"""Processors that transform IR episodes into model batches."""

from robo_train.data.processors.batch_builder import BatchBuilder, EmbodiedBatch
from robo_train.data.processors.pipeline import ProcessorPipeline

__all__ = ["BatchBuilder", "EmbodiedBatch", "ProcessorPipeline"]
