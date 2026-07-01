# Training Family Clarity Design

> **Archived/Stale:** This design predates the `robo_train` package rename and plugin migration. It is kept only for historical context; use the current `src/robo_train/`, `configs/frameworks/`, and `scripts/frameworks/` layout as the source of truth.

## Goal

Clarify UEFS training architecture so VLA, 3D policy, and world-model/video-style training paths share the same Universal Episode IR while using distinct data views, loss profiles, and embodiment/runtime profiles.

## User Understanding Check

The user is mostly right:

- Dataloader behavior and loss functions must differ for VLA, 3D policies, and world models.
- World-model training is close to video-generation training, but robotics world models also condition on actions, robot state, and embodiment.
- Training and inference differ by robot arm type, but the trainer should not fork completely by robot. Robot-specific logic belongs in embodiment profiles, action projection, control frequency, frames, gripper conventions, and runtime profiles.

The architecture should therefore split by four axes:

1. `DataProfile`: `vla`, `policy_3d`, `world_model`
2. `ModelFamily`: `vla_policy`, `policy_3d`, `world_action_model`
3. `LossProfile`: `vla_bc`, `policy_3d_bc`, `world_model`
4. `EmbodimentProfile`: `franka_single_arm`, `aloha_dual_arm`, `mobile_manipulator`, etc.

## Architecture

External data should still flow through adapters into `Episode`. Dataset-specific loading stays in `data/adapters/`; model-family-specific collation moves to `data/views/`.

```text
Raw Dataset
  -> Adapter
  -> Episode IR
  -> DataView
      - VLADataView
      - Policy3DDataView
      - WorldModelDataView
  -> Policy / ModelFamily
  -> LossProfile
  -> EmbodimentAdapter + ActionProjector
  -> RuntimeProfile
```

## Components

- `src/schema/training_profile.py`
  - Defines `DataProfile`, `ModelFamily`, `LossProfile`, `EmbodimentProfile`, and `TrainingProfile`.
  - Validates that data/model/loss selections are compatible.
- `src/data/views/`
  - Converts existing `EmbodiedBatch` objects into model-family-specific dictionaries.
  - `VLADataView` keeps image, language, proprioception, robot metadata, and actions.
  - `Policy3DDataView` prioritizes point/state geometry and can synthesize a mock point cloud from state for the demo.
  - `WorldModelDataView` emits observation/action history plus future target summaries.
- `src/trainer/loss_registry.py`
  - Registers loss profile functions by name.
  - Lets `ExperimentConfig.training_profile.loss_profile` select the loss family.
- `src/trainer/losses_vla.py`, `losses_3d.py`, `losses_wm.py`
  - Keep loss-family logic separate and testable.
- Historical root-level family demo JSON samples
  - These were later removed when configs were reorganized around
    `configs/base/`, `configs/datasets/`, and `configs/frameworks/`.
- Docs
  - README, README_zh, architecture, and framework mapping should explain that adapters unify data, views split training inputs, and embodiment profiles split robot-specific runtime behavior.

## Testing

Add tests before implementation:

- Training profile validation accepts compatible VLA/3D/WM combinations and rejects mismatches.
- Data views produce distinct keys for VLA, 3D, and world-model batches.
- Loss registry computes the expected loss names for all three families.
- Config samples load through `ExperimentConfig` and include matching `TrainingProfile`.
- Docs mention DataView, LossProfile, and EmbodimentProfile.

## Non-Goals

- No real PyTorch training.
- No real video diffusion model.
- No real point-cloud backbone.
- No real hardware-specific controller implementation.
- No forked trainer per robot arm.
