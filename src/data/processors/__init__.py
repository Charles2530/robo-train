"""Processors that transform IR episodes into model batches."""

from src.data.processors.batch_builder import BatchBuilder, EmbodiedBatch
from src.data.processors.pipeline import ProcessorPipeline

__all__ = ["BatchBuilder", "EmbodiedBatch", "ProcessorPipeline"]
