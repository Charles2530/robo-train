from pathlib import Path

import numpy as np
import pytest

from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter
from robo_train.data.processors.pipeline import ProcessorPipeline
from robo_train.training.models.policy import UnifiedEmbodiedPolicy
from robo_train.schema.dataset_manifest import DatasetManifest
from robo_train.schema.experiment_config import (
    DatasetConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainerConfig,
)
from robo_train.training.algorithm_registry import AlgorithmRegistry, build_policy_from_config
from robo_train.training.trainer import MockTrainer
from robo_train.training.artifacts import CheckpointMetadata, TrainingArtifact


def test_dataset_manifest_serializes_dataset_lineage(tmp_path: Path):
    manifest = DatasetManifest(
        dataset_id="demo-synthetic",
        name="Synthetic Pick Place",
        source_framework="robo_train.synthetic",
        source_uri="memory://synthetic",
        task_families=["pick_place"],
        robot_families=["franka"],
        splits={"train": 0.8, "val": 0.2},
        sampling_weight=1.5,
        stats_path="stats/demo-synthetic.json",
        quality_flags=["validated_timestamps"],
        lineage={"generator": "SyntheticAdapter", "seed": 42},
    )

    path = tmp_path / "dataset_manifest.json"
    manifest.to_json(path)
    restored = DatasetManifest.from_json(path)

    assert restored.dataset_id == "demo-synthetic"
    assert restored.split_names == ["train", "val"]
    assert restored.lineage["generator"] == "SyntheticAdapter"


def test_dataset_manifest_rejects_invalid_split_fraction():
    with pytest.raises(ValueError, match="split fractions"):
        DatasetManifest(
            dataset_id="bad",
            name="Bad",
            source_framework="test",
            source_uri=None,
            task_families=["pick"],
            robot_families=["franka"],
            splits={"train": 0.9, "val": 0.9},
        )


def test_experiment_config_builds_policy_and_trainer_summary():
    experiment = ExperimentConfig(
        experiment_id="demo-bc",
        seed=7,
        datasets=[
            DatasetConfig(
                manifest=DatasetManifest(
                    dataset_id="demo-synthetic",
                    name="Synthetic",
                    source_framework="robo_train.synthetic",
                    source_uri=None,
                    task_families=["pick_place"],
                    robot_families=["franka"],
                    splits={"train": 1.0},
                ),
                split="train",
                batch_weight=2.0,
            )
        ],
        policy=PolicyConfig(policy_type="unified_mock", latent_dim=16, action_horizon=4),
        trainer=TrainerConfig(algorithm="behavior_cloning", max_epochs=1, batch_size=2),
    )
    policy = build_policy_from_config(experiment.policy, seed=experiment.seed)
    episodes = SyntheticAdapter(seed=experiment.seed).to_ir({"episodes": 2, "steps": 12})
    processor = ProcessorPipeline.default(history_window=4, action_horizon=4)

    summary = MockTrainer(
        policy=policy,
        processor=processor,
        episodes=episodes,
        experiment=experiment,
    ).train_one_epoch()

    assert summary["experiment_id"] == "demo-bc"
    assert summary["algorithm"] == "behavior_cloning"
    assert summary["datasets"] == ["demo-synthetic"]
    assert summary["episodes"] == 2
    assert summary["loss"] >= 0.0


def test_algorithm_registry_registers_named_policy_factories():
    registry = AlgorithmRegistry()
    registry.register_policy(
        "tiny_policy",
        lambda config, seed: UnifiedEmbodiedPolicy(
            seed=seed,
            latent_dim=config.latent_dim,
            action_horizon=config.action_horizon,
        ),
    )
    policy = registry.create_policy(
        PolicyConfig(policy_type="tiny_policy", latent_dim=8, action_horizon=3),
        seed=123,
    )

    assert isinstance(policy, UnifiedEmbodiedPolicy)
    assert policy.latent_dim == 8
    assert policy.action_horizon == 3
    with pytest.raises(KeyError, match="unknown policy type"):
        registry.create_policy(PolicyConfig(policy_type="missing"), seed=0)


def test_training_artifact_records_checkpoint_context(tmp_path: Path):
    metadata = CheckpointMetadata(
        checkpoint_id="ckpt-0001",
        experiment_id="demo-bc",
        step=100,
        metrics={"loss": 0.12},
        policy_type="unified_mock",
        dataset_ids=["demo-synthetic"],
    )
    artifact = TrainingArtifact(
        checkpoint=metadata,
        config_snapshot={"seed": 7},
        processor_stats={"action_mean": np.zeros(2).tolist()},
    )

    path = tmp_path / "artifact.json"
    artifact.to_json(path)
    restored = TrainingArtifact.from_json(path)

    assert restored.checkpoint.checkpoint_id == "ckpt-0001"
    assert restored.checkpoint.dataset_ids == ["demo-synthetic"]
    assert restored.processor_stats["action_mean"] == [0.0, 0.0]
