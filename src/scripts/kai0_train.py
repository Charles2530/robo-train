"""Kai0-compatible training entrypoint backed by local layered profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.adapters.synthetic_adapter import SyntheticAdapter
from src.data.processors.pipeline import ProcessorPipeline
from src.training.algorithm_registry import build_policy_from_config
from src.training.kai0_launcher import build_kai0_launch_spec, run_kai0_launch
from src.training.kai0_profiles import Kai0TrainProfile, build_experiment_config, build_launch_payload, load_kai0_train_profile
from src.training.trainer import MockTrainer


def main(argv: list[str] | None = None) -> None:
    """Run a Kai0-compatible dry-run or local training smoke test."""
    args = _parse_args(argv)
    profile = _apply_train_overrides(load_kai0_train_profile(args.profile), args)
    _check_inputs(profile, strict=args.check_inputs)

    payload = build_launch_payload(
        profile,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    if args.encode_variant:
        payload["encode_variant"] = args.encode_variant

    if args.backend == "kai0":
        encode_variant = args.encode_variant
        if encode_variant is None and profile.script_name == "train_arrange_flowers_encode.sh":
            encode_variant = "ffmpeg_gop30"
        spec = build_kai0_launch_spec(
            profile,
            exp_name=args.exp_name,
            batch_size=args.batch_size,
            num_train_steps=args.num_train_steps,
            num_workers=args.num_workers,
            encode_variant=encode_variant,
            source_root=args.kai0_source_root,
        )
        payload["kai0_backend"] = spec.as_payload()
        if args.dry_run:
            _print_payload(payload, as_json=args.as_json)
            return
        if spec.preflight_warnings:
            raise SystemExit("Kai0 preflight failed:\n- " + "\n- ".join(spec.preflight_warnings))
        raise SystemExit(run_kai0_launch(spec))

    if args.dry_run:
        _print_payload(payload, as_json=args.as_json)
        return

    experiment = build_experiment_config(
        profile,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
    )
    payload["smoke_summary"] = _run_smoke_training(profile, experiment)
    _print_payload(payload, as_json=args.as_json)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kai0-compatible training profile launcher")
    parser.add_argument("profile", help="Kai0 config name, profile file stem, or train_*.sh script name")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the local train payload")
    parser.add_argument("--backend", choices=["kai0", "smoke"], default="kai0", help="Training backend to execute")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--kai0-source-root", default=None, help="Read-only Kai0 source checkout used as backend")
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
    parser.add_argument("--encode-variant", help="Optional encode variant used by train_arrange_flowers_encode.sh")
    return parser.parse_args(argv)


def _apply_train_overrides(profile: Kai0TrainProfile, args: argparse.Namespace) -> Kai0TrainProfile:
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
    return profile.model_copy(update={"train": profile.train.model_copy(update=updates)})


def _check_inputs(profile: Kai0TrainProfile, *, strict: bool) -> None:
    if not strict:
        return
    missing = []
    dataset_root = Path(profile.data.dataset_path)
    if profile.data.required_subdir:
        dataset_root = dataset_root / profile.data.required_subdir
    if not dataset_root.exists():
        missing.append(f"dataset path: {dataset_root}")
    if not Path(profile.checkpoint.params_path).exists():
        missing.append(f"checkpoint: {profile.checkpoint.params_path}")
    if missing:
        raise SystemExit("missing required Kai0 input paths:\n- " + "\n- ".join(missing))


def _run_smoke_training(profile: Kai0TrainProfile, experiment: Any) -> dict[str, Any]:
    episodes = SyntheticAdapter(seed=experiment.seed, state_dim=profile.data.state_dim).to_ir({"episodes": 2})
    processor = ProcessorPipeline.default(action_horizon=profile.model.action_horizon, policy_fps=profile.data.fps)
    policy = build_policy_from_config(experiment.policy, seed=experiment.seed)
    summary = MockTrainer(policy=policy, processor=processor, episodes=episodes, experiment=experiment).train_one_epoch()
    summary["mode"] = "local_synthetic_smoke"
    summary["real_dataset_path"] = profile.data.dataset_path
    return summary


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    profile = payload["profile"]["profile"]
    experiment = payload["experiment"]
    print("Kai0-compatible training payload")
    print(f"- config: {profile['config_name']}")
    print(f"- experiment: {experiment['experiment_id']}")
    print(f"- dataset: {payload['checks']['dataset_path']}")
    print(f"- checkpoint: {payload['checks']['checkpoint']}")
    print(f"- launcher: {payload['launcher']['type']} {payload['launcher']['command']}")
    if "kai0_backend" in payload:
        backend = payload["kai0_backend"]
        print(f"- backend cwd: {backend['cwd']}")
        print(f"- backend command: {' '.join(backend['command'])}")
        print(f"- backend log: {backend['log_path']}")
        for warning in backend.get("preflight_warnings", []):
            print(f"- warning: {warning}")
    if "smoke_summary" in payload:
        print(f"- smoke loss: {payload['smoke_summary']['loss']:.6f}")


if __name__ == "__main__":
    main()
