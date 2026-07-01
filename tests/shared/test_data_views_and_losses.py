import numpy as np

from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter
from robo_train.data.processors.pipeline import ProcessorPipeline
from robo_train.data.views import VLADataView, Policy3DDataView, WorldModelDataView, build_data_view
from robo_train.schema.training_profile import TrainingProfile
from robo_train.training.loss_registry import compute_loss_profile
from robo_train.training.models.policy import UnifiedEmbodiedPolicy


def _demo_batch():
    episodes = SyntheticAdapter(seed=3).to_ir({"episodes": 2, "steps": 12})
    processor = ProcessorPipeline.default(history_window=4, action_horizon=4)
    processor.fit(episodes)
    return processor.transform(episodes)


def test_data_views_expose_family_specific_inputs():
    batch = _demo_batch()

    vla = VLADataView().build(batch)
    policy3d = Policy3DDataView().build(batch)
    world_model = WorldModelDataView(future_horizon=3).build(batch)

    assert {"images", "language", "state", "actions", "robot_cards", "action_semantics"} <= set(vla)
    assert "point_cloud" not in vla

    assert {"point_cloud", "state", "actions", "robot_cards", "action_semantics"} <= set(policy3d)
    assert policy3d["point_cloud"].shape[:2] == (2, 4)

    assert {"context_images", "context_state", "action_history", "future_state_target", "future_horizon"} <= set(world_model)
    assert world_model["future_horizon"] == 3
    assert world_model["action_history"].shape == (2, 4, batch.action_semantics.action_dim)


def test_build_data_view_from_training_profile_and_name():
    assert isinstance(build_data_view("vla"), VLADataView)
    assert isinstance(build_data_view(TrainingProfile.default_policy3d()), Policy3DDataView)
    assert isinstance(build_data_view(TrainingProfile.default_world_model()), WorldModelDataView)


def test_loss_registry_computes_family_specific_losses():
    prediction = {
        "pred_actions": np.zeros((2, 4, 7), dtype=np.float32),
        "future_latent": np.ones((2, 4, 8), dtype=np.float32),
        "point_features": np.ones((2, 4, 3), dtype=np.float32),
    }
    target = {
        "actions": np.ones((2, 4, 7), dtype=np.float32),
        "future_state_target": np.zeros((2, 4, 8), dtype=np.float32),
        "point_cloud": np.zeros((2, 4, 3), dtype=np.float32),
    }

    vla_loss = compute_loss_profile("vla_bc", prediction, target)
    policy3d_loss = compute_loss_profile("policy_3d_bc", prediction, target)
    world_loss = compute_loss_profile("world_model", prediction, target)

    assert set(vla_loss) == {"loss", "action_loss", "language_conditioning_loss"}
    assert vla_loss["loss"] > 0.0

    assert set(policy3d_loss) == {"loss", "action_loss", "geometry_consistency_loss"}
    assert policy3d_loss["geometry_consistency_loss"] > 0.0

    assert set(world_loss) == {"loss", "action_loss", "future_prediction_loss"}
    assert world_loss["future_prediction_loss"] > 0.0


def test_policy_forward_can_feed_family_specific_loss_profiles():
    batch = _demo_batch()
    output = UnifiedEmbodiedPolicy(seed=7, action_horizon=4, latent_dim=16).forward(batch)
    world_view = WorldModelDataView(future_horizon=4).build(batch)

    losses = compute_loss_profile(
        "world_model",
        {"pred_actions": output["pred_actions"], "future_latent": output["future_latent"]},
        {"actions": batch.actions, "future_state_target": world_view["future_state_target"]},
    )

    assert losses["loss"] >= losses["action_loss"]
