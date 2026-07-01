import json
import os
import subprocess
import sys
from pathlib import Path

from robo_train.frameworks.backend_env import resolve_backend_subprocess_env
from robo_train.frameworks.kai0.plugin import Kai0FrameworkPlugin
from robo_train.training.kai0_launcher import build_kai0_launch_spec
from robo_train.training.kai0_profiles import build_experiment_config, load_kai0_train_profile


def test_kai0_layered_profile_preserves_dataset_and_checkpoint_paths():
    profile = load_kai0_train_profile("pi05_put_the_books_back_table30v2_joint_delta")

    assert profile.config_name == "pi05_put_the_books_back_table30v2_joint_delta"
    assert profile.data.dataset_path == "datasets/RoboChallenge_Table30v2_v2.1_pyav_g2/put_the_books_back"
    assert profile.checkpoint.params_path == "checkpoints/kai0/kai0-base/pi05_base/params"
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
            "robo_train.cli.train",
            "--framework",
            "kai0",
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
        env={**os.environ, "PYTHONPATH": f"{Path.cwd() / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"},
    )
    payload = json.loads(result.stdout)

    assert payload["profile"]["config_name"] == "pi05_arrange_flowers"
    assert payload["profile"]["data"]["dataset_path"] == "datasets/RoboChallenge_Table30_v2.1_pyav_g2/arrange_flowers"
    assert payload["experiment"]["experiment_id"] == "dry-run-arrange"
    assert payload["experiment"]["trainer"]["kwargs"]["num_train_steps"] == 3
    assert payload["kai0_backend"]["command"][1] == "scripts/train.py"
    assert "--data.repo-id=" in " ".join(payload["kai0_backend"]["command"])
    assert "robo-train/datasets/RoboChallenge_Table30_v2.1_pyav_g2/arrange_flowers" in " ".join(
        payload["kai0_backend"]["command"]
    )


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

    scripts_dir = Path("scripts/frameworks/kai0")
    assert {path.name for path in scripts_dir.glob("train_*.sh")} == expected_scripts
    for script in scripts_dir.glob("train_*.sh"):
        text = script.read_text(encoding="utf-8")
        assert "robo_train.cli.train --framework kai0" in text
        assert "kai0/tools_charles" not in text
        assert "cd /data/" not in text
        assert "/../../.." in text


def test_kai0_backend_env_defaults_do_not_override_user_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    profile = load_kai0_train_profile("pi05_put_the_books_back_table30v2_joint_delta")

    spec = build_kai0_launch_spec(profile, num_train_steps=2)

    assert spec.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "--batch_size=8" in spec.command


def test_kai0_backend_runtime_env_preserves_user_secrets(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    profile = load_kai0_train_profile("pi05_put_the_books_back_table30v2_joint")
    plugin = Kai0FrameworkPlugin()

    dry_payload = plugin.build_launch_payload(profile, dry_run=True, num_train_steps=2)
    runtime_payload = plugin.build_launch_payload(profile, dry_run=False, num_train_steps=2)

    assert "WANDB_API_KEY" not in dry_payload["backend"]["env"]
    assert dry_payload["backend"].get("runtime_env") is None
    assert runtime_payload["backend"]["runtime_env"]["WANDB_API_KEY"] == "test-wandb-key"
    assert resolve_backend_subprocess_env(runtime_payload["backend"])["WANDB_API_KEY"] == "test-wandb-key"


def test_kai0_backend_library_order_matches_reference_shell(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/conda/lib:/existing/lib:/usr/local/ffmpeg/lib")
    profile = load_kai0_train_profile("pi05_put_the_books_back_table30v2_joint")

    spec = build_kai0_launch_spec(profile, num_train_steps=2)
    libs = spec.env["LD_LIBRARY_PATH"].split(os.pathsep)

    assert libs[:3] == ["/usr/local/ffmpeg/lib", "/opt/conda/lib", "/existing/lib"]


def test_table30v2_shell_selects_single_card_defaults():
    result = subprocess.run(
        [
            "bash",
            "scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh",
            "--dry-run",
            "--json",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    payload = json.loads(result.stdout)
    command = payload["kai0_backend"]["command"]

    assert payload["kai0_backend"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "--exp_name=run1-put-books-back-table30v2-joint-50000step-1card-bs8" in command
    assert "--batch_size=8" in command
    assert "--num_workers=2" in command


def test_table30v2_shell_accepts_cuda_visible_typo():
    result = subprocess.run(
        [
            "bash",
            "scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh",
            "--dry-run",
            "--json",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "CUDA_VISIBEL_DEVICES": "0"},
    )
    payload = json.loads(result.stdout)
    command = payload["kai0_backend"]["command"]

    assert payload["kai0_backend"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "--exp_name=run1-put-books-back-table30v2-joint-50000step-1card-bs8" in command
    assert "--batch_size=8" in command


def test_table30v2_shell_scales_two_card_defaults():
    result = subprocess.run(
        [
            "bash",
            "scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh",
            "--dry-run",
            "--json",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
    )
    payload = json.loads(result.stdout)
    command = payload["kai0_backend"]["command"]

    assert payload["kai0_backend"]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "--exp_name=run1-put-books-back-table30v2-joint-50000step-2card-bs64" in command
    assert "--batch_size=64" in command
    assert "--num_workers=8" in command


def test_table30v2_shell_preserves_eight_card_defaults():
    result = subprocess.run(
        [
            "bash",
            "scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh",
            "--dry-run",
            "--json",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    )
    payload = json.loads(result.stdout)
    command = payload["kai0_backend"]["command"]

    assert payload["kai0_backend"]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "--exp_name=run1-put-books-back-table30v2-joint-50000step-8card-bs256" in command
    assert "--batch_size=256" in command
    assert "--num_workers=32" in command


def test_table30v2_shell_accepts_num_workers_env_override():
    result = subprocess.run(
        [
            "bash",
            "scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh",
            "--dry-run",
            "--json",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "NUM_WORKERS": "32"},
    )
    payload = json.loads(result.stdout)
    command = payload["kai0_backend"]["command"]

    assert "--batch_size=256" in command
    assert "--num_workers=32" in command


def test_table30v2_shell_preserves_explicit_train_args():
    result = subprocess.run(
        [
            "bash",
            "scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh",
            "--dry-run",
            "--json",
            "--exp_name",
            "manual-run",
            "--batch_size",
            "4",
            "--num_workers",
            "1",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    payload = json.loads(result.stdout)
    command = payload["kai0_backend"]["command"]

    assert "--exp_name=manual-run" in command
    assert "--batch_size=4" in command
    assert "--num_workers=1" in command
    assert "--exp_name=run1-put-books-back-table30v2-joint-50000step-1card-bs8" not in command
