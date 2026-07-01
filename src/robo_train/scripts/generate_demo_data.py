"""Generate and validate synthetic demo episodes."""

import argparse
from pathlib import Path

from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter
from robo_train.data.validators.action_semantics_validator import ActionSemanticsValidator
from robo_train.data.validators.base_validator import ValidatorPipeline
from robo_train.data.validators.modality_validator import MissingModalityValidator
from robo_train.data.validators.robot_compatibility_validator import RobotCompatibilityValidator
from robo_train.data.validators.timestamp_validator import TimestampValidator


def default_validator_pipeline() -> ValidatorPipeline:
    """Return the default IR validator pipeline."""
    return ValidatorPipeline(
        [
            TimestampValidator(),
            ActionSemanticsValidator(),
            RobotCompatibilityValidator(),
            MissingModalityValidator(),
        ]
    )


def main() -> None:
    """Generate synthetic episodes and save them as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./demo_data")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapter = SyntheticAdapter(seed=args.seed)
    episodes = adapter.to_ir({"episodes": args.episodes})
    report = default_validator_pipeline().validate_many(episodes)
    if not report.passed:
        raise SystemExit(f"validation failed: {report.errors}")
    adapter.from_ir(episodes, Path(args.output))

    print("Generated Demo Data")
    print(f"- output: {args.output}")
    print(f"- episodes: {len(episodes)}")
    print(f"- validation: {'passed' if report.passed else 'failed'}")
    for episode in episodes:
        print(f"- {episode.episode_id}: {episode.num_steps} steps")


if __name__ == "__main__":
    main()
