"""Unified framework training CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from robo_train.frameworks import FRAMEWORK_REGISTRY
from robo_train.frameworks.backend_env import resolve_backend_subprocess_env


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    plugin = FRAMEWORK_REGISTRY.get(args.framework)
    profile = _apply_train_overrides(plugin.load_profile(args.profile), args)
    experiment = plugin.to_experiment_config(
        profile,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
    )
    payload = plugin.build_launch_payload(
        profile,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
        source_root=args.source_root,
        encode_variant=args.encode_variant,
    )
    payload.setdefault("experiment", experiment.model_dump(mode="json"))

    if args.dry_run:
        _print_payload(payload, as_json=args.as_json)
        return
    if args.check_inputs:
        _require_backend_inputs(payload)
    raise SystemExit(_run_backend_payload(payload["backend"]))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training framework launcher")
    parser.add_argument("profile", help="Framework profile name, config name, or script name")
    parser.add_argument("--framework", required=True, choices=FRAMEWORK_REGISTRY.names())
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the train payload")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--source-root", dest="source_root")
    parser.add_argument("--exp_name", "--exp-name", dest="exp_name")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("--num_train_steps", "--num-train-steps", dest="num_train_steps", type=int)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int)
    parser.add_argument("--log_interval", "--log-interval", dest="log_interval", type=int)
    parser.add_argument("--save_interval", "--save-interval", dest="save_interval", type=int)
    parser.add_argument("--keep_period", "--keep-period", dest="keep_period", type=int)
    parser.add_argument("--fsdp_devices", "--fsdp-devices", dest="fsdp_devices", type=int)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--wandb-enabled", "--wandb_enabled", dest="wandb_enabled", action="store_true", default=None)
    parser.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false")
    parser.add_argument("--check-inputs", action="store_true", help="Require dataset/checkpoint paths to exist")
    parser.add_argument("--encode-variant", help="Optional prepared video encode variant")
    return parser.parse_args(argv)


def _apply_train_overrides(profile: Any, args: argparse.Namespace) -> Any:
    train = getattr(profile, "train", None)
    if train is None or not hasattr(profile, "model_copy") or not hasattr(train, "model_copy"):
        return profile
    override_keys = [
        "log_interval",
        "save_interval",
        "keep_period",
        "fsdp_devices",
        "overwrite",
        "resume",
        "wandb_enabled",
    ]
    updates = {key: getattr(args, key) for key in override_keys if getattr(args, key) is not None}
    if not updates:
        return profile
    return profile.model_copy(update={"train": train.model_copy(update=updates)})


def _require_backend_inputs(payload: dict[str, Any]) -> None:
    missing = []
    checks = payload.get("checks", {})
    dataset_path = checks.get("dataset_path")
    required_subdir = checks.get("required_subdir")
    if dataset_path:
        dataset_root = Path(dataset_path)
        if required_subdir:
            dataset_root = dataset_root / required_subdir
        if not dataset_root.exists():
            missing.append(f"dataset path: {dataset_root}")
    checkpoint = checks.get("checkpoint")
    if checkpoint and not Path(checkpoint).exists():
        missing.append(f"checkpoint: {checkpoint}")
    if missing:
        raise SystemExit("missing required input paths:\n- " + "\n- ".join(missing))


def _run_backend_payload(backend: dict[str, Any]) -> int:
    warnings = backend.get("preflight_warnings", [])
    if warnings:
        raise SystemExit("backend preflight failed:\n- " + "\n- ".join(warnings))
    log_path = Path(backend["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            backend["command"],
            cwd=backend["cwd"],
            env=resolve_backend_subprocess_env(backend),
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


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    profile = payload["profile"]["profile"]
    experiment = payload["experiment"]
    print("Training payload")
    print(f"- config: {profile['config_name']}")
    print(f"- experiment: {experiment['experiment_id']}")
    print(f"- dataset: {payload['checks']['dataset_path']}")
    print(f"- checkpoint: {payload['checks']['checkpoint']}")
    backend = payload.get("backend")
    if backend:
        print(f"- backend cwd: {backend['cwd']}")
        print(f"- backend command: {' '.join(backend['command'])}")
        print(f"- backend log: {backend['log_path']}")
        for warning in backend.get("preflight_warnings", []):
            print(f"- warning: {warning}")


if __name__ == "__main__":
    main()
