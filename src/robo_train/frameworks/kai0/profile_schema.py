"""Kai0 profile schema."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_KAI0_CONFIG_DIR = Path("configs/frameworks/kai0/tasks")


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
    source_root: str | None = None
    python_bin: str | None = None
    torchrun_bin: str | None = None
    fallback_python_bins: list[str] = Field(default_factory=list)


class Kai0ModelConfig(BaseModel):
    """Model-family knobs shared by Kai0-derived profiles."""

    family: str = "pi05"
    policy_type: str = "unified_mock"
    latent_dim: int = Field(default=32, gt=0)
    action_horizon: int = Field(default=8, gt=0)
    head: str | None = None


class Kai0DataConfig(BaseModel):
    """Dataset identity, path, and modality mapping consumed by Kai0."""

    dataset_id: str | None = None
    dataset_type: str = "unknown"
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
    supported_frameworks: list[str] = Field(default_factory=lambda: ["kai0"])
    preprocessors: list[str] = Field(default_factory=list)


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

    @model_validator(mode="after")
    def validate_dataset_framework_support(self) -> "Kai0TrainProfile":
        """Reject dataset/framework combinations that are not declared supported."""
        if "kai0" not in self.data.supported_frameworks:
            dataset = self.data.dataset_id or self.data.dataset_path
            supported = ", ".join(self.data.supported_frameworks) or "<none>"
            raise ValueError(f"dataset {dataset!r} does not support framework 'kai0'; supported: {supported}")
        return self
