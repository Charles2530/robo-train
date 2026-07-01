from pathlib import Path

import tomllib

import robo_train
from robo_train.schema.dataset_manifest import DatasetManifest
from robo_train.schema.experiment_config import DatasetConfig, ExperimentConfig, PolicyConfig, TrainerConfig
from robo_train.schema.training_profile import TrainingProfile


def test_package_metadata_uses_robo_train_name():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "robo-train"
    assert Path("src/robo_train").is_dir()
    assert not Path("uefs").exists()
    assert robo_train.__version__


def test_mvp_package_focuses_on_data_and_training_infra():
    expected_dirs = {"data", "schema", "scripts", "training"}
    package_dirs = {
        path.name
        for path in Path("src/robo_train").iterdir()
        if path.is_dir() and not path.name.startswith("__") and not path.name.startswith(".")
    }

    assert expected_dirs <= package_dirs
    assert {"runtime", "envs", "eval", "train_deploy_alignment", "model", "trainer"}.isdisjoint(package_dirs)


def test_mvp_dependencies_do_not_include_runtime_server_stack():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = "\n".join(pyproject["project"]["dependencies"])

    assert "fastapi" not in deps
    assert "uvicorn" not in deps
    assert "httpx" not in deps


def test_framework_mapping_doc_covers_reference_projects():
    text = Path("docs/framework_mapping.md").read_text(encoding="utf-8")

    for expected in [
        "StarVLA",
        "training infra",
        "data infra",
        "LightX2V",
        "LeRobot",
        "robomimic",
        "DatasetManifest",
        "ExperimentConfig",
        "DataView",
        "LossProfile",
        "EmbodimentProfile",
        "Protocol-first",
        "FrameGraph",
        "SceneCard",
        "VLADataView",
        "Policy3DDataView",
        "WorldModelDataView",
    ]:
        assert expected in text


def test_experiment_config_schema_validates_minimal_synthetic_config():
    config = ExperimentConfig(
        experiment_id="robo-train-synthetic-bc-test",
        datasets=[
            DatasetConfig(
                manifest=DatasetManifest(
                    dataset_id="synthetic_test",
                    name="Synthetic Pick Place",
                    source_framework="robo_train.synthetic",
                    source_uri="memory://synthetic",
                    task_families=["pick_place"],
                    robot_families=["franka"],
                    splits={"train": 1.0},
                )
            )
        ],
        training_profile=TrainingProfile.default_vla(embodiment_profile="franka_single_arm"),
        policy=PolicyConfig(policy_type="unified_mock"),
        trainer=TrainerConfig(algorithm="behavior_cloning"),
    )

    assert config.experiment_id == "robo-train-synthetic-bc-test"
    assert config.datasets[0].manifest.dataset_id == "synthetic_test"
    assert config.policy.policy_type == "unified_mock"
    assert config.trainer.algorithm == "behavior_cloning"


def test_family_training_profiles_validate_in_experiment_configs():
    configs = [
        ExperimentConfig(experiment_id="test-vla", datasets=[_synthetic_dataset()], training_profile=TrainingProfile.default_vla()),
        ExperimentConfig(
            experiment_id="test-policy3d",
            datasets=[_synthetic_dataset()],
            training_profile=TrainingProfile.default_policy3d(),
        ),
        ExperimentConfig(
            experiment_id="test-world-model",
            datasets=[_synthetic_dataset()],
            training_profile=TrainingProfile.default_world_model(),
        ),
    ]

    for config, data_profile in zip(configs, ["vla", "policy_3d", "world_model"], strict=True):
        assert config.training_profile.data_profile == data_profile


def _synthetic_dataset() -> DatasetConfig:
    return DatasetConfig(
        manifest=DatasetManifest(
            dataset_id="synthetic_test",
            name="Synthetic",
            source_framework="robo_train.synthetic",
            source_uri="memory://synthetic",
            task_families=["pick_place"],
            robot_families=["franka"],
            splits={"train": 1.0},
        )
    )
