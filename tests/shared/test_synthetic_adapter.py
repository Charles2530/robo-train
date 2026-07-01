from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter


def test_synthetic_adapter_generates_complete_episode():
    episodes = SyntheticAdapter(seed=7, steps=16).to_ir({"episodes": 1})

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.num_steps == 16
    assert "front_rgb" in episode.streams
    assert "wrist_rgb" in episode.streams
    assert "state" in episode.streams
    assert "action" in episode.streams
    assert "language" in episode.streams
    assert episode.get_stream("front_rgb").shape == [16, 64, 64, 3]
    assert episode.get_stream("wrist_rgb").shape == [16, 64, 64, 3]
    assert episode.get_stream("state").shape[0] == 16
    assert episode.get_stream("action").shape == [16, 7]
    assert episode.task.instruction == "pick up the red block and place it into the bowl"
