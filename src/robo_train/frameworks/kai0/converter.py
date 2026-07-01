"""Kai0 profile conversion to shared train payloads."""

from __future__ import annotations

from typing import Any

from robo_train.frameworks.kai0.profile_schema import Kai0TrainProfile
from robo_train.schema.dataset_manifest import DatasetManifest
from robo_train.schema.experiment_config import DatasetConfig, ExperimentConfig, PolicyConfig, TrainerConfig
from robo_train.schema.training_profile import TrainingProfile


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
        dataset_id=profile.data.dataset_id or f"kai0_{profile.config_name}",
        name=profile.profile.task,
        source_framework="kai0",
        source_uri=profile.data.dataset_path,
        task_families=[profile.data.task_family],
        robot_families=[profile.data.robot_family],
        splits={"train": 1.0},
        stats_path=profile.data.norm_stats_path,
        metadata={
            "dataset_type": profile.data.dataset_type,
            "prompt": profile.data.default_prompt,
            "image_map": profile.data.image_map,
            "required_subdir": profile.data.required_subdir,
            "format": profile.data.format,
            "state_dim": profile.data.state_dim,
            "fps": profile.data.fps,
            "supported_frameworks": profile.data.supported_frameworks,
            "preprocessors": profile.data.preprocessors,
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
