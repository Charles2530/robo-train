"""Kai0 profile loading."""

from __future__ import annotations

from pathlib import Path

from robo_train.config import load_layered_yaml
from robo_train.frameworks.kai0.profile_schema import DEFAULT_KAI0_CONFIG_DIR, Kai0TrainProfile


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
