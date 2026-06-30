"""Kai0-compatible train profiles backed by layered YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import load_layered_yaml
from src.schema.dataset_manifest import DatasetManifest
from src.schema.experiment_config import DatasetConfig, ExperimentConfig, PolicyConfig, TrainerConfig
from src.schema.training_profile import TrainingProfile


DEFAULT_KAI0_CONFIG_DIR = Path("configs/experiments/kai0")


class Kai0ProfileMetadata(BaseModel):
    """Identity and script compatibility metadata."""

    config_name: str
    script_name: str
    task: str
    run_name: str | None = None
    description: str | None = None


class Kai0LauncherConfig(BaseModel):
    """Original launcher family captured without depending on Kai0 code."""

    type: str = "jax"
    command: str = "scripts/train.py"
    nproc_per_node: int | None = None


class Kai0ModelConfig(BaseModel):
    """Model-family knobs shared by Kai0-derived profiles."""

    family: str = "pi05"
    policy_type: str = "unified_mock"
    latent_dim: int = Field(default=32, gt=0)
    action_horizon: int = Field(default=8, gt=0)
    head: str | None = None


class Kai0DataConfig(BaseModel):
    """Dataset and modality mapping from Kai0 train configs."""

    dataset_path: str
    required_subdir: str | None = None
    default_prompt: str
    image_map: dict[str, str] = Field(default_factory=dict)
    state_dim: int = Field(gt=0)
    robot_family: str
    task_family: str
    fps: float = Field(default=20.0, gt=0)
    norm_stats_path: str | None = None
    format: str = "lerobot"


class Kai0ActionConfig(BaseModel):
    """Action label semantics for Kai0-derived training profiles."""

    representation: str
    action_dim: int = Field(gt=0)
    control_mode: str = "absolute"
    use_next_state_as_action: bool = False
    use_delta_joint_actions: bool = False
    label_source: str = "action"


class Kai0CheckpointConfig(BaseModel):
    """Checkpoint paths preserved from the Kai0 scripts/configs."""

    params_path: str


class Kai0TrainParams(BaseModel):
    """Train parameters copied from Kai0 shell defaults."""

    algorithm: str = "kai0_openpi_compat"
    batch_size: int = Field(default=256, gt=0)
    num_workers: int = Field(default=8, ge=0)
    num_train_steps: int = Field(default=50_000, gt=0)
    log_interval: int = Field(default=100, gt=0)
    save_interval: int = Field(default=5_000, gt=0)
    keep_period: int = Field(default=5_000, gt=0)
    fsdp_devices: int = Field(default=1, gt=0)
    overwrite: bool = True
    resume: bool = False
    wandb_enabled: bool = False


class Kai0TrainProfile(BaseModel):
    """Layered Kai0 profile resolved into this framework's train config."""

    model_config = ConfigDict(extra="allow")

    profile: Kai0ProfileMetadata
    launcher: Kai0LauncherConfig = Field(default_factory=Kai0LauncherConfig)
    model: Kai0ModelConfig = Field(default_factory=Kai0ModelConfig)
    data: Kai0DataConfig
    action: Kai0ActionConfig
    checkpoint: Kai0CheckpointConfig
    train: Kai0TrainParams = Field(default_factory=Kai0TrainParams)
    env: dict[str, str] = Field(default_factory=dict)

    @property
    def config_name(self) -> str:
        """Return the Kai0 TrainConfig name."""
        return self.profile.config_name

    @property
    def script_name(self) -> str:
        """Return the compatible train script name."""
        return self.profile.script_name

    @property
    def run_name(self) -> str:
        """Return the default experiment run name."""
        return self.profile.run_name or f"run1-{self.config_name}"


def list_kai0_train_profiles(config_dir: str | Path = DEFAULT_KAI0_CONFIG_DIR) -> list[str]:
    """List available Kai0-compatible profile names."""
    root = Path(config_dir)
    return sorted(path.stem for path in root.glob("*.yaml"))


def load_kai0_train_profile(name_or_script: str, config_dir: str | Path = DEFAULT_KAI0_CONFIG_DIR) -> Kai0TrainProfile:
    """Load a Kai0-compatible train profile by file stem, config name, or script name."""
    key = name_or_script.strip()
    if not key:
        raise ValueError("profile name must be non-empty")

    root = Path(config_dir)
    candidate = root / f"{Path(key).stem}.yaml"
    if candidate.exists():
        return Kai0TrainProfile.model_validate(load_layered_yaml(candidate))

    matches: list[Kai0TrainProfile] = []
    for path in root.glob("*.yaml"):
        profile = Kai0TrainProfile.model_validate(load_layered_yaml(path))
        names = {path.stem, profile.config_name, profile.script_name, Path(profile.script_name).stem}
        if key in names or Path(key).name in names:
            matches.append(profile)

    if not matches:
        known = ", ".join(list_kai0_train_profiles(root))
        raise KeyError(f"unknown Kai0 train profile {name_or_script!r}; known profiles: {known}")
    if len(matches) > 1:
        names = ", ".join(profile.config_name for profile in matches)
        raise ValueError(f"ambiguous Kai0 train profile {name_or_script!r}: {names}")
    return matches[0]


def build_experiment_config(
    profile: Kai0TrainProfile,
    *,
    exp_name: str | None = None,
    batch_size: int | None = None,
    num_train_steps: int | None = None,
    num_workers: int | None = None,
) -> ExperimentConfig:
    """Convert a Kai0 layered profile into the framework ExperimentConfig."""
    manifest = DatasetManifest(
        dataset_id=f"kai0_{profile.config_name}",
        name=profile.profile.task,
        source_framework="kai0",
        source_uri=profile.data.dataset_path,
        task_families=[profile.data.task_family],
        robot_families=[profile.data.robot_family],
        splits={"train": 1.0},
        stats_path=profile.data.norm_stats_path,
        metadata={
            "prompt": profile.data.default_prompt,
            "image_map": profile.data.image_map,
            "required_subdir": profile.data.required_subdir,
            "format": profile.data.format,
            "state_dim": profile.data.state_dim,
            "fps": profile.data.fps,
        },
    )
    train_kwargs = _train_kwargs(profile, num_train_steps=num_train_steps, num_workers=num_workers)
    return ExperimentConfig(
        experiment_id=exp_name or profile.run_name,
        datasets=[DatasetConfig(manifest=manifest, split="train")],
        policy=PolicyConfig(
            policy_type=profile.model.policy_type,
            latent_dim=profile.model.latent_dim,
            action_horizon=profile.model.action_horizon,
            action_dim=profile.action.action_dim,
            pretrained_checkpoint=profile.checkpoint.params_path,
            kwargs={
                "family": profile.model.family,
                "head": profile.model.head,
                "prompt": profile.data.default_prompt,
                "image_map": profile.data.image_map,
            },
        ),
        training_profile=TrainingProfile.default_vla(embodiment_profile=profile.data.robot_family),
        trainer=TrainerConfig(
            algorithm=profile.train.algorithm,
            max_epochs=1,
            batch_size=batch_size or profile.train.batch_size,
            kwargs=train_kwargs,
        ),
        tags=["kai0-compatible", profile.model.family, profile.launcher.type],
        notes=profile.profile.description,
    )


def build_launch_payload(
    profile: Kai0TrainProfile,
    *,
    exp_name: str | None = None,
    batch_size: int | None = None,
    num_train_steps: int | None = None,
    num_workers: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Build a JSON-friendly launch payload for dry-runs and wrappers."""
    experiment = build_experiment_config(
        profile,
        exp_name=exp_name,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
    )
    profile_payload = profile.model_dump(mode="json")
    profile_payload.update(
        {
            "config_name": profile.config_name,
            "script_name": profile.script_name,
            "run_name": profile.run_name,
        }
    )
    return {
        "dry_run": dry_run,
        "profile": profile_payload,
        "launcher": profile.launcher.model_dump(mode="json"),
        "env": profile.env,
        "experiment": experiment.model_dump(mode="json"),
        "checks": {
            "dataset_path": profile.data.dataset_path,
            "required_subdir": profile.data.required_subdir,
            "checkpoint": profile.checkpoint.params_path,
            "depends_on_local_kai0_folder": False,
        },
    }


def _train_kwargs(
    profile: Kai0TrainProfile,
    *,
    num_train_steps: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    return {
        "num_train_steps": num_train_steps or profile.train.num_train_steps,
        "num_workers": profile.train.num_workers if num_workers is None else num_workers,
        "log_interval": profile.train.log_interval,
        "save_interval": profile.train.save_interval,
        "keep_period": profile.train.keep_period,
        "fsdp_devices": profile.train.fsdp_devices,
        "overwrite": profile.train.overwrite,
        "resume": profile.train.resume,
        "wandb_enabled": profile.train.wandb_enabled,
        "action": profile.action.model_dump(mode="json"),
    }
