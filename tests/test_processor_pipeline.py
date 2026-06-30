from src.data.adapters.synthetic_adapter import SyntheticAdapter
from src.data.processors.pipeline import ProcessorPipeline


def test_processor_pipeline_builds_predictable_batch():
    episodes = SyntheticAdapter(seed=3, steps=24).to_ir({"episodes": 2})
    pipeline = ProcessorPipeline.default(history_window=8, action_horizon=8)

    pipeline.fit(episodes)
    batch = pipeline.transform(episodes)

    assert set(batch.images) == {"front_rgb", "wrist_rgb"}
    assert batch.images["front_rgb"].shape == (2, 8, 64, 64, 3)
    assert batch.images["wrist_rgb"].shape == (2, 8, 64, 64, 3)
    assert batch.state.shape == (2, 8, 14)
    assert batch.actions.shape == (2, 8, 7)
    assert batch.modality_mask["front_rgb"] is True
    assert batch.modality_mask["wrist_rgb"] is True
    assert batch.modality_mask["state"] is True
    assert batch.modality_mask["tactile"] is True
