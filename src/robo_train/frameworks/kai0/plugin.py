"""Kai0 framework plugin."""

from __future__ import annotations

from typing import Any

from robo_train.frameworks.backend_env import attach_runtime_env
from robo_train.frameworks.kai0.converter import build_experiment_config, build_launch_payload
from robo_train.frameworks.kai0.launcher import build_kai0_launch_spec
from robo_train.frameworks.kai0.loader import list_kai0_train_profiles, load_kai0_train_profile
from robo_train.frameworks.kai0.profile_schema import Kai0TrainProfile
from robo_train.frameworks.registry import FRAMEWORK_REGISTRY
from robo_train.schema.experiment_config import ExperimentConfig


class Kai0FrameworkPlugin:
    """Framework plugin for Kai0/OpenPI-compatible training."""

    name = "kai0"

    def list_profiles(self) -> list[str]:
        return list_kai0_train_profiles()

    def load_profile(self, name_or_script: str) -> Kai0TrainProfile:
        return load_kai0_train_profile(name_or_script)

    def to_experiment_config(
        self,
        profile: Any,
        *,
        exp_name: str | None = None,
        batch_size: int | None = None,
        num_train_steps: int | None = None,
        num_workers: int | None = None,
    ) -> ExperimentConfig:
        return build_experiment_config(
            profile,
            exp_name=exp_name,
            batch_size=batch_size,
            num_train_steps=num_train_steps,
            num_workers=num_workers,
        )

    def build_launch_payload(
        self,
        profile: Any,
        *,
        exp_name: str | None = None,
        batch_size: int | None = None,
        num_train_steps: int | None = None,
        num_workers: int | None = None,
        dry_run: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = build_launch_payload(
            profile,
            exp_name=exp_name,
            batch_size=batch_size,
            num_train_steps=num_train_steps,
            num_workers=num_workers,
            dry_run=dry_run,
        )
        encode_variant = kwargs.get("encode_variant")
        if encode_variant is None and profile.script_name == "train_arrange_flowers_encode.sh":
            encode_variant = "ffmpeg_gop30"
        spec = build_kai0_launch_spec(
            profile,
            exp_name=exp_name,
            batch_size=batch_size,
            num_train_steps=num_train_steps,
            num_workers=num_workers,
            encode_variant=encode_variant,
            source_root=kwargs.get("source_root"),
        )
        backend_payload = spec.as_payload()
        if not dry_run:
            attach_runtime_env(backend_payload, spec.env)
        payload["backend"] = backend_payload
        payload["kai0_backend"] = backend_payload
        return payload


KAI0_PLUGIN = FRAMEWORK_REGISTRY.register(Kai0FrameworkPlugin())
