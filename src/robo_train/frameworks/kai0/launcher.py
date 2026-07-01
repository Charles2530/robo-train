"""Launch real Kai0/OpenPI training from this repository's profiles.

The Kai0 source checkout is treated as a read-only backend.  Dataset,
checkpoint, run, and debug choices stay in this repository so future model
families can add their own launchers without copying shell scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
from typing import Mapping

from robo_train.frameworks.backend_env import summarize_backend_env
from robo_train.frameworks.kai0.profile_schema import Kai0TrainProfile


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ENCODE_VARIANTS = {
    "ffmpeg_gop2",
    "ffmpeg_gop8",
    "ffmpeg_gop16",
    "ffmpeg_gop30",
    "pyav_g2",
    "pyav_g8",
    "pyav_g16",
    "pyav_g30",
}


@dataclass(frozen=True)
class Kai0LaunchSpec:
    """Resolved backend launch command."""

    cwd: Path
    command: list[str]
    env: dict[str, str]
    log_path: Path
    dataset_path: Path
    checkpoint_path: Path
    prepared_dataset_path: Path | None = None
    preflight_warnings: list[str] = field(default_factory=list)

    def as_payload(self) -> dict[str, object]:
        """Return a JSON-friendly view of the launch command."""
        return {
            "cwd": str(self.cwd),
            "command": self.command,
            "env": summarize_backend_env(self.env),
            "log_path": str(self.log_path),
            "dataset_path": str(self.dataset_path),
            "checkpoint_path": str(self.checkpoint_path),
            "prepared_dataset_path": str(self.prepared_dataset_path) if self.prepared_dataset_path else None,
            "preflight_warnings": self.preflight_warnings,
        }


def build_kai0_launch_spec(
    profile: Kai0TrainProfile,
    *,
    exp_name: str | None = None,
    batch_size: int | None = None,
    num_train_steps: int | None = None,
    num_workers: int | None = None,
    encode_variant: str | None = None,
    source_root: str | Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> Kai0LaunchSpec:
    """Resolve a Kai0 backend command without executing it."""
    configured_source = source_root or os.environ.get("KAI0_SOURCE_ROOT") or profile.launcher.source_root
    if not configured_source:
        raise ValueError(
            "Kai0 source root is not configured; set launcher.source_root in the profile or KAI0_SOURCE_ROOT"
        )
    source = Path(configured_source).expanduser().resolve()
    if not (source / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"Kai0 source root does not look valid: {source}")

    env = _build_env(profile, source, env_overrides=env_overrides)
    dataset_path = _resolve_project_path(profile.data.dataset_path)
    prepared_dataset = None
    if encode_variant is not None:
        prepared_dataset = _prepare_encode_dataset(
            profile,
            source,
            env,
            encode_variant=encode_variant,
            env_overrides=env_overrides,
        )
        dataset_path = prepared_dataset

    checkpoint_path = _resolve_project_path(profile.checkpoint.params_path)
    run_name = exp_name or profile.run_name
    warnings = _preflight(profile, dataset_path=dataset_path, checkpoint_path=checkpoint_path)
    command = _build_command(
        profile,
        source=source,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        env=env,
    )
    return Kai0LaunchSpec(
        cwd=source,
        command=command,
        env=env,
        log_path=PROJECT_ROOT / "logs" / f"{run_name}.log",
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        prepared_dataset_path=prepared_dataset,
        preflight_warnings=warnings,
    )


def run_kai0_launch(spec: Kai0LaunchSpec) -> int:
    """Execute a resolved launch command and tee stdout/stderr to its log file."""
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            spec.command,
            cwd=spec.cwd,
            env=spec.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def _build_command(
    profile: Kai0TrainProfile,
    *,
    source: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    run_name: str,
    batch_size: int | None,
    num_train_steps: int | None,
    num_workers: int | None,
    env: Mapping[str, str],
) -> list[str]:
    args = [
        profile.config_name,
        f"--exp_name={run_name}",
        f"--data.repo-id={dataset_path}",
        f"--batch_size={batch_size or _default_batch_size(profile, env)}",
        f"--num_workers={profile.train.num_workers if num_workers is None else num_workers}",
        f"--num_train_steps={num_train_steps or profile.train.num_train_steps}",
        f"--log_interval={profile.train.log_interval}",
        f"--save_interval={profile.train.save_interval}",
        f"--keep_period={profile.train.keep_period}",
        f"--fsdp_devices={profile.train.fsdp_devices}",
        f"--checkpoint-base-dir={PROJECT_ROOT / 'checkpoints' / 'kai0' / 'runs'}",
        f"--assets-base-dir={PROJECT_ROOT / 'assets' / 'kai0'}",
    ]
    if profile.train.overwrite:
        args.append("--overwrite")
    if profile.train.resume:
        args.append("--resume")
    args.append("--wandb-enabled" if profile.train.wandb_enabled else "--no-wandb-enabled")

    if profile.launcher.type == "torchrun":
        args.append(f"--pytorch-weight-path={checkpoint_path}")
        nproc = profile.launcher.nproc_per_node or 1
        executable = _resolve_executable(profile.launcher.torchrun_bin, source=source, fallback="torchrun")
        return [
            executable,
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            "scripts/train_pytorch.py",
            *args,
        ]

    args.append(f"--weight-loader.params-path={checkpoint_path}")
    executable = _resolve_python(profile, source=source)
    return [executable, profile.launcher.command, *args]


def _build_env(
    profile: Kai0TrainProfile,
    source: Path,
    *,
    env_overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in profile.env.items():
        env.setdefault(key, value)
    if env_overrides:
        env.update(env_overrides)

    ffmpeg_home = env.get("FFMPEG_HOME")
    if ffmpeg_home:
        env["PATH"] = _prepend_path(f"{ffmpeg_home}/bin", env.get("PATH", ""))
    conda_lib_path = env.get("CONDA_LIB_PATH")
    ordered_libs = [path for path in (f"{ffmpeg_home}/lib" if ffmpeg_home else None, conda_lib_path) if path]
    env["LD_LIBRARY_PATH"] = _prepend_paths(ordered_libs, env.get("LD_LIBRARY_PATH", ""))
    env["KAI0_ROOT"] = str(PROJECT_ROOT)
    env["TASK_A_ROOT"] = str(PROJECT_ROOT / "datasets" / "xiaoyu_Task_A")
    env["PYTHONPATH"] = _prepend_path(str(source / "src"), env.get("PYTHONPATH", ""))

    if env.get("DEBUG") == "1":
        env.setdefault("OPENPI_DEBUG_TRAIN", "1")
        env.setdefault("OPENPI_DEBUG_INTERVAL", "1")
        env.setdefault("OPENPI_DEBUG_SAMPLE_INTERVAL", "64")
        env.setdefault("OPENPI_DEBUG_SLOW_MS", "500")
    return env


def _prepare_encode_dataset(
    profile: Kai0TrainProfile,
    source: Path,
    env: Mapping[str, str],
    *,
    encode_variant: str,
    env_overrides: Mapping[str, str] | None,
) -> Path:
    variant = _resolve_encode_variant(encode_variant)
    src_dataset = _resolve_project_path(
        (env_overrides or {}).get("SRC_DATASET", "datasets/temp/arrange_flowers_v2.1")
    )
    if not (src_dataset / "videos").is_dir():
        sparse = _resolve_project_path("datasets/temp/arrange_flowers_v2.1_sparse")
        if (sparse / "videos").is_dir():
            src_dataset = sparse
    bench_dir = _resolve_project_path((env_overrides or {}).get("BENCH_DIR", "video_encode_bench_arrange_flowers"))
    encode_mp4 = bench_dir / f"{variant}.mp4"
    dst = _resolve_project_path(f"datasets/temp/arrange_flowers_encode_{variant}")
    executable = _resolve_python(profile, source=source)
    cmd = [
        executable,
        "scripts/prepare_arrange_flowers_encode_dataset.py",
        "--src",
        str(src_dataset),
        "--encode-mp4",
        str(encode_mp4),
        "--dst",
        str(dst),
    ]
    if (env_overrides or {}).get("PREP_FORCE") == "1":
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=source, env=dict(env), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else str(dst)
    return Path(last_line).resolve()


def _resolve_encode_variant(raw: str) -> str:
    variant = raw.removesuffix(".mp4")
    if variant in {"2", "8", "16", "30"}:
        variant = f"ffmpeg_gop{variant}"
    if variant not in ENCODE_VARIANTS:
        known = ", ".join(sorted(ENCODE_VARIANTS))
        raise ValueError(f"unknown encode variant {raw!r}; expected one of: {known}")
    return variant


def _preflight(profile: Kai0TrainProfile, *, dataset_path: Path, checkpoint_path: Path) -> list[str]:
    warnings: list[str] = []
    required = dataset_path / profile.data.required_subdir if profile.data.required_subdir else dataset_path
    if not required.exists():
        warnings.append(f"dataset requirement missing: {required}")
    if not checkpoint_path.exists():
        warnings.append(f"checkpoint missing: {checkpoint_path}")
    return warnings


def _default_batch_size(profile: Kai0TrainProfile, env: Mapping[str, str]) -> int:
    if profile.config_name.endswith("table30v2") or "table30v2" in profile.config_name:
        return profile.train.batch_size if "," in env.get("CUDA_VISIBLE_DEVICES", "") else 8
    return profile.train.batch_size


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.absolute()
    return (PROJECT_ROOT / candidate).absolute()


def _prepend_path(path: str, current: str) -> str:
    if not current:
        return path
    parts = current.split(os.pathsep)
    if path in parts:
        return current
    return os.pathsep.join([path, *parts])


def _prepend_paths(paths: list[str], current: str) -> str:
    current_parts = [part for part in current.split(os.pathsep) if part] if current else []
    unique_paths = []
    for path in paths:
        if path not in unique_paths:
            unique_paths.append(path)
    rest = [part for part in current_parts if part not in unique_paths]
    return os.pathsep.join([*unique_paths, *rest])


def _resolve_python(profile: Kai0TrainProfile, *, source: Path) -> str:
    configured = os.environ.get("KAI0_PYTHON") or profile.launcher.python_bin
    executable = _resolve_executable(configured, source=source, fallback=None)
    if executable:
        return executable
    for candidate in profile.launcher.fallback_python_bins:
        executable = _resolve_executable(candidate, source=source, fallback=None)
        if executable:
            return executable
    return "python"


def _resolve_executable(configured: str | None, *, source: Path, fallback: str | None) -> str | None:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = source / path
        if path.exists():
            return str(path)
    return fallback
