import numpy as np

from src.data.adapters.synthetic_adapter import SyntheticAdapter
from src.data.dataloader import MixedEpisodeDataLoader
from src.data.storage import NormStats


def test_norm_stats_fit_transform_and_serializes_round_trip(tmp_path):
    episodes = SyntheticAdapter(seed=11, steps=6).to_ir({"episodes": 2})

    stats = NormStats.fit(episodes, stream_names=["state", "action"])
    transformed = stats.normalize_episode(episodes[0])

    path = tmp_path / "norm_stats.json"
    stats.to_json(path)
    restored = NormStats.from_json(path)

    state = np.asarray(transformed.streams["state"].data, dtype=np.float32)
    assert set(restored.streams) == {"state", "action"}
    assert restored.streams["action"].count == 12
    assert abs(float(state.mean())) < 1.0


def test_mixed_episode_dataloader_batches_multiple_sources_deterministically():
    left = SyntheticAdapter(seed=1, steps=8).to_ir({"episodes": 2})
    right = SyntheticAdapter(seed=2, steps=8).to_ir({"episodes": 2})
    loader = MixedEpisodeDataLoader(
        datasets={"left": left, "right": right},
        batch_size=3,
        weights={"left": 1.0, "right": 1.0},
        seed=7,
    )

    first = loader.next_batch()
    second = loader.next_batch()

    assert len(first) == 3
    assert len(second) == 3
    assert {episode.dataset_id for episode in first + second} == {"synthetic_demo"}
    assert loader.sampled_sources[:6] == ["right", "right", "right", "left", "left", "right"]
