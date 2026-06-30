from pathlib import Path

import tomllib

import src
from src.schema.experiment_config import ExperimentConfig


def test_package_metadata_uses_src_name():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "src"
    assert Path("src").is_dir()
    assert not Path("uefs").exists()
    assert src.__version__


def test_mvp_package_focuses_on_data_and_training_infra():
    expected_dirs = {"data", "schema", "scripts", "training"}
    package_dirs = {
        path.name
        for path in Path("src").iterdir()
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


def test_demo_experiment_config_validates():
    config = ExperimentConfig.from_json("configs/experiment_demo.json")

    assert config.experiment_id == "src-synthetic-bc-demo"
    assert config.datasets[0].manifest.dataset_id == "synthetic_demo"
    assert config.policy.policy_type == "unified_mock"
    assert config.trainer.algorithm == "behavior_cloning"


def test_family_demo_configs_exist_and_validate():
    expected_profiles = {
        "configs/vla_demo.json": "vla",
        "configs/policy3d_demo.json": "policy_3d",
        "configs/world_model_demo.json": "world_model",
    }

    for path, data_profile in expected_profiles.items():
        config = ExperimentConfig.from_json(path)
        assert config.training_profile.data_profile == data_profile
