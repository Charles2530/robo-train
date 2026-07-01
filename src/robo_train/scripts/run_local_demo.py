"""Local data-to-training demo for the MVP infra stack."""

from robo_train.data.adapters.synthetic_adapter import SyntheticAdapter
from robo_train.data.processors.pipeline import ProcessorPipeline
from robo_train.scripts.generate_demo_data import default_validator_pipeline
from robo_train.training.models.policy import UnifiedEmbodiedPolicy
from robo_train.training.trainer import MockTrainer


def main() -> None:
    """Run synthetic data, validation, processing, and one mock training epoch."""
    episodes = SyntheticAdapter(seed=42).to_ir({"episodes": 3})
    validation = default_validator_pipeline().validate_many(episodes)
    if not validation.passed:
        raise SystemExit(f"validation failed: {validation.errors}")

    processor = ProcessorPipeline.default(history_window=8, action_horizon=8)
    policy = UnifiedEmbodiedPolicy(seed=42, action_horizon=8, latent_dim=32)
    trainer = MockTrainer(policy=policy, processor=processor, episodes=episodes)
    training_summary = trainer.train_one_epoch()
    batch = processor.transform(episodes)

    print("UEFS Data + Training Infra Demo Summary")
    print(f"- episodes: {len(episodes)}")
    print(f"- validation: {'passed' if validation.passed else 'failed'}")
    print(f"- batch images: {', '.join(name.replace('_rgb', '') for name in sorted(batch.images))}")
    print(f"- action representation: {batch.action_semantics.representation}")
    print(f"- mock training loss: {training_summary['loss']:.6f}")
    print(f"- action target shape: {list(batch.actions.shape) if batch.actions is not None else None}")


if __name__ == "__main__":
    main()
