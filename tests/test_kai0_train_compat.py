import json
import subprocess
import sys
from pathlib import Path

from src.training.kai0_profiles import build_experiment_config, load_kai0_train_profile


def test_kai0_layered_profile_preserves_dataset_and_checkpoint_paths():
    profile = load_kai0_train_profile("pi05_put_the_books_back_table30v2_joint_delta")

    assert profile.config_name == "pi05_put_the_books_back_table30v2_joint_delta"
    assert profile.data.dataset_path == "/data/buaa/code/djy/kai0/processed/RoboChallenge_Table30v2_v2.1_pyav_g2/put_the_books_back"
    assert profile.checkpoint.params_path == "/data/buaa/code/djy/kai0/checkpoints/pi05_base/params"
    assert profile.data.image_map == {
        "top_head": "global_image",
        "hand_left": "left_wrist_image",
        "hand_right": "right_wrist_image",
    }
    assert profile.action.action_dim == 14
    assert profile.action.representation == "joint_delta"
    assert profile.action.use_next_state_as_action is True
    assert profile.action.use_delta_joint_actions is True
    assert profile.train.batch_size == 256
    assert profile.train.num_train_steps == 50_000


def test_kai0_profile_builds_experiment_config_with_overrides():
    profile = load_kai0_train_profile("pi05_arrange_flowers_table30v2")
    experiment = build_experiment_config(
        profile,
        exp_name="smoke-table30v2",
        batch_size=8,
        num_train_steps=12,
        num_workers=2,
    )

    assert experiment.experiment_id == "smoke-table30v2"
    assert experiment.datasets[0].manifest.source_uri == profile.data.dataset_path
    assert experiment.datasets[0].manifest.dataset_id == "kai0_pi05_arrange_flowers_table30v2"
    assert experiment.policy.action_dim == 8
    assert experiment.trainer.batch_size == 8
    assert experiment.trainer.kwargs["num_train_steps"] == 12
    assert experiment.trainer.kwargs["num_workers"] == 2


def test_kai0_train_cli_dry_run_does_not_need_kai0_folder():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.kai0_train",
            "pi05_arrange_flowers",
            "--dry-run",
            "--json",
            "--exp_name",
            "dry-run-arrange",
            "--num_train_steps",
            "3",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["profile"]["config_name"] == "pi05_arrange_flowers"
    assert payload["profile"]["data"]["dataset_path"] == "/data/buaa/code/djy/kai0/processed/RoboChallenge_Table30_v2.1_pyav_g2/arrange_flowers"
    assert payload["experiment"]["experiment_id"] == "dry-run-arrange"
    assert payload["experiment"]["trainer"]["kwargs"]["num_train_steps"] == 3


def test_top_level_kai0_compatible_train_scripts_call_local_launcher():
    expected_scripts = {
        "train_arrange_flowers.sh",
        "train_arrange_flowers_encode.sh",
        "train_arrange_flowers_sparse.sh",
        "train_arrange_flowers_table30v2.sh",
        "train_put_the_books_back_table30v2.sh",
        "train_put_the_books_back_table30v2_joint.sh",
        "train_put_the_books_back_table30v2_joint_delta.sh",
        "train_pytorch.sh",
    }

    scripts_dir = Path("scripts/train")
    assert {path.name for path in scripts_dir.glob("train_*.sh")} == expected_scripts
    for script in scripts_dir.glob("train_*.sh"):
        text = script.read_text(encoding="utf-8")
        assert "src.scripts.kai0_train" in text
        assert "kai0/tools_charles" not in text
        assert "cd /data/" not in text
