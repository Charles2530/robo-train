# Universal Embodied Foundation Stack

`src` is a lightweight, runnable Python demo for a Protocol-first embodied data
and training infra stack. It first normalizes data, robot embodiment, action
semantics, coordinate frames, scenes, and training views, then uses a minimal
NumPy policy to exercise the training path. Runtime serving, benchmark envs, and
real-robot deployment are intentionally outside this MVP.

## What It Demonstrates

- Universal Embodied IR with `ActionSemantics`, `RobotCard`, `TaskCard`, `FrameGraph`, `SceneCard`, streams, and episodes.
- StarVLA-style adapters that emit raw, model-agnostic IR rather than tokenized model inputs.
- Validator and processor pipelines that prepare multimodal robot data for a policy.
- `DatasetManifest` and `ExperimentConfig` contracts for multi-dataset, config-driven training.
- `TrainingProfile` with `DataProfile`, `ModelFamily`, `LossProfile`, and `EmbodimentProfile` compatibility checks.
- Family-specific `DataView` objects: `VLADataView`, `Policy3DDataView`, and `WorldModelDataView`.
- Data infra utilities for `NormStats`, shard specs, and deterministic mixed episode loading.
- `AlgorithmRegistry` and `TrainingArtifact` scaffolding for swappable algorithms and reproducible checkpoints.
- `LossRegistry` with separate VLA, 3D policy, and world-model loss profiles.
- A small `UnifiedEmbodiedPolicy.forward()` path for training-infra smoke tests.

## Reference Frameworks

UEFS is not a copy of any one project. It borrows the engineering contracts that
make several embodied-AI frameworks easier to extend, then reduces them to a
small NumPy demo.

| Reference | What UEFS borrows | Where it appears here | What UEFS intentionally does not copy |
| --- | --- | --- | --- |
| [StarVLA](https://github.com/starVLA/starVLA) | Lego-like module boundaries, raw model-agnostic dataloader output, and a model API centered on `forward()` plus `predict_action()`. | `schema/`, `data/adapters/`, `data/processors/`, `training/models/policy.py`, `training/trainer.py`. | Real VLM/VLA backbones, tokenizer plumbing, distributed training, benchmark configs, and full model zoo logic. |
| [DreamZero / World Action Model](https://dreamzero0.github.io/) | Future-prediction objectives can share the same data and training infra as action policies. | `WorldModelDataView` and `world_model` loss profile keep this training path visible. | Video diffusion, real future-frame generation, GPU inference, WebSocket/distributed inference, and zero-shot claims. |
| [kai0 / chi0](https://github.com/OpenDriveLab/kai0) | Train-script compatibility, task prompts, image maps, action dimensions, joint/delta action choices, and original dataset/checkpoint path conventions. | `configs/experiments/kai0/`, `src/training/kai0_profiles.py`, `src/training/kai0_launcher.py`, `src/scripts/kai0_train.py`, `scripts/train/kai0/`. | Robot deployment, DAgger collection, policy server, and secrets from shell scripts. |
| [LightX2V](https://github.com/ModelTC/LightX2V) | Practical repo ergonomics: top-level layered `configs/`, runnable scripts, quickstart-first README, and task-oriented command lines. | `configs/base/`, `configs/experiments/kai0/`, `src/config/layered.py`, `src/scripts/`, `scripts/train/kai0/`. | Image/video generation backends, acceleration kernels, model-format conversion, and GPU serving stack. |
| [LeRobot](https://github.com/huggingface/lerobot) | Dataset/policy/training/deployment separation, reusable configs, and robotics dataset metadata. | `DatasetManifest`, `ExperimentConfig`, `data/adapters/`, `configs/`. | Hugging Face Hub integration, real robot drivers, real pretrained policies, and distributed training. |
| [robomimic](https://robomimic.github.io/) | Config-driven imitation learning, algorithm abstraction, dataset splits, and reproducible experiment artifacts. | `ExperimentConfig`, `AlgorithmRegistry`, `TrainingArtifact`, `training/trainer.py`. | Real PyTorch algorithms, robosuite integration, and full HDF5 training pipelines. |
| [ManiSkill](https://maniskill.readthedocs.io/) | Future benchmark adapters should stay outside data/training internals. | Deferred until eval infra is added. | GPU simulation, physics assets, task suites, and reinforcement-learning environments. |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | Future simulator integration should be an adapter layer, not training-core logic. | Deferred until eval/runtime infra is added. | Isaac Sim dependency, manager-based RL stack, physics simulation, and asset pipelines. |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Sequence-action policies should register as training policies without changing data infra. | Future policy registry entry. | Real diffusion/flow model training and image backbone implementation. |
| [OpenVLA](https://github.com/openvla/openvla) and [Octo](https://github.com/octo-models/octo) | Pretrained generalist VLA policy fine-tuning and checkpoint-oriented workflows. | `PolicyConfig.pretrained_checkpoint`, `freeze_backbone`, `EmbodimentAdapter`, `TrainingArtifact`. | Real VLM/VLA weights, tokenizers, model serving, and hardware-specific inference. |
| [MimicGen](https://github.com/NVlabs/mimicgen) and [RoboCasa](https://github.com/robocasa/robocasa) | Generated demonstration provenance and task-family metadata. | `DatasetManifest.lineage`, `quality_flags`, adapter registry. | Demonstration generation, scene assets, kitchen benchmarks, and simulator rollouts. |

The detailed file-by-file mapping lives in
[`docs/framework_mapping.md`](docs/framework_mapping.md).

## What These Frameworks Have In Common

All four references separate **data format**, **model internals**, **training
loop**, **runtime**, and **evaluation/deployment evidence**. That separation is
the main architectural lesson UEFS keeps.

They also converge on a few practical principles:

- **Stable protocol before model choice.** StarVLA's raw dict, kai0's deployment payloads, and UEFS's `Episode`/`EmbodiedBatch` all make data contracts explicit before choosing a backbone.
- **Config and metadata are part of training.** LeRobot and robomimic show that datasets, splits, normalization stats, algorithms, and runtime horizons should be saved as structured experiment state.
- **Layered configs beat script sprawl.** LightX2V-style `defaults` let a profile compose model, training, and task layers while keeping dataset and checkpoint paths explicit.
- **Adapters unify data; DataView objects split training inputs.** VLA, 3D, and WM should not require three incompatible dataset stacks. UEFS converts sources to `Episode` first, then uses `VLADataView`, `Policy3DDataView`, or `WorldModelDataView`.
- **LossProfile is the main training-family switch.** VLA action BC, 3D geometry-aware policy loss, and WM future-prediction loss live in separate loss modules.
- **EmbodimentProfile handles robot-arm differences.** Franka, ALOHA, UR, and mobile manipulators should differ through robot/action/runtime profiles, not by duplicating the whole trainer.
- **One policy surface, many model families.** Whether the internal policy is VLA, world-action, or state-only, the outside world wants a training call and an action prediction call.
- **Action chunks are a deployment unit.** DreamZero, Diffusion Policy, and kai0 all care about chunked actions and smoothing; UEFS treats `[horizon, action_dim]` chunks as first-class outputs.
- **Tasks and environments are adapters.** ManiSkill and Isaac Lab keep simulator details out of policy code; UEFS does the same with `envs/`.
- **Deployment produces training evidence.** Rollouts, failures, recoveries, safety events, and stage progress are not logs for humans only; they are future alignment data.
- **Configs and scripts matter.** LightX2V's layout is a reminder that a research framework is only useful when common runs are obvious and repeatable.

## Training Differences By Model Family

UEFS exposes one `UnifiedEmbodiedPolicy`, but real systems would train different
model families differently. The demo keeps these differences visible through
separate heads, metadata, and alignment hooks.

| Model family | Typical inputs | Main training target | Extra losses or signals | Deployment behavior | UEFS representation |
| --- | --- | --- | --- | --- | --- |
| **VLA / OpenPi-style action policy** | RGB images, language instruction, proprioception, robot metadata. | Predict canonical robot actions, often with behavior cloning or flow/diffusion action losses. | Optional language-retention loss, action smoothness, safety/risk heads, stage progress. | Direct `predict_action()` returns an action chunk. | `MockVisionEncoder`, `MockLanguageEncoder`, `MockProprioEncoder`, `ActionHead`, `forward()` action loss. |
| **DreamZero-style World Action Model** | Observation history plus task conditioning, often with video/world-model latents. | Jointly predict future world state/video latent and actions. The world target is not just an auxiliary decoration; it shapes the representation used for action. | Future latent/video reconstruction, action prediction, chunk consistency, latency/smoothing constraints. | Returns both action chunk and `predicted_future` summary. | `WorldActionHead` produces `future_latent` and action-conditioned latent from the same shared rollout. |
| **kai0 / chi0-style robust manipulation policy** | Demonstrations plus future deployment rollouts, recovery data, and stage labels. | Improve action policy under distribution shift between training and deployment. | Deferred to runtime/alignment phase. | Not part of MVP. | Future alignment package. |
| **Pure 3D or state-based policy** | Proprioception, low-dimensional state, point clouds, robot morphology; language may be optional. | Predict actions from geometric/state features rather than VLM-scale image-language features. | State/action normalization, morphology conditioning, smoothness and safety losses. | Same action API, usually cheaper inference and fewer visual dependencies. | `MockProprioEncoder`, `MorphologyEncoder`, optional/absent image streams via `modality_mask`. |
| **Diffusion Policy-style sequence policy** | Image/state history plus language or task conditioning. | Learn action sequences with a denoising/flow objective rather than direct one-step regression. | Noise schedule, sequence consistency, action normalization. | Deferred to a future runtime phase. | `PolicyConfig` future registry entry. |
| **Sim benchmark policy** | Simulator observations, task ids, reset/step rewards, success signals. | Train or evaluate against standardized task suites. | Success rate, rollout length, resets, environment-specific rewards. | Deferred to an eval phase. | Future benchmark adapters. |

The common API is deliberate: swapping a real VLA, a real WAM, or a state-only
controller should not require changing adapters, validators, runtime clients, or
rollout logging. Only the internals of encoders/heads/trainers should change.

## Architecture

```text
DatasetManifest + ExperimentConfig
  -> Raw Data
  -> Adapter (data/adapters/)
  -> Universal Embodied IR (schema/)
  -> Validator (data/validators/)
  -> ProcessorPipeline (data/processors/)
  -> TrainingProfile
  -> DataView (VLADataView / Policy3DDataView / WorldModelDataView)
  -> NormStats + MixedEpisodeDataLoader
  -> AlgorithmRegistry (training/algorithm_registry.py)
  -> LossRegistry + LossProfile
  -> UnifiedEmbodiedPolicy (training/models/policy.py)
  -> TrainingArtifact
```

## Quick Start

```bash
pip install -e .
pytest -q
python -m src.scripts.run_local_demo
python -m src.scripts.generate_demo_data --output ./demo_data --episodes 3
python -m src.scripts.kai0_train pi05_arrange_flowers --dry-run --json
bash scripts/train/kai0/train_arrange_flowers_table30v2.sh --dry-run
```

## Directory Structure

```text
configs/
  base/                    # reusable model/train layers
  experiments/kai0/         # Kai0-compatible task profiles
docs/
  architecture.md
  framework_mapping.md
src/
  config/                  # layered YAML loader
  schema/                  # cross-layer Universal Embodied IR
  data/                    # adapters, validators, processors, storage, dataloader
    views/                 # VLA, 3D, and WM model-family views
  training/                # registry, artifacts, mock trainer, losses, minimal policy
  scripts/                 # python -m demo entrypoints
scripts/train/kai0/         # Kai0-compatible shell wrappers
tests/
```

## Core Design

The IR is Protocol-first: it separates raw actions from canonical actions,
keeps scene and frame metadata beside streams, and avoids collapsing every
modality into one universal tensor. External datasets
often use vendor-specific control commands while policies need a stable target
space. `ActionSemantics` names that target space: representation, control mode,
frame, fields, units, fps, horizon, and dimensionality.

`DatasetManifest` and `ExperimentConfig` are the compatibility layer for real
training. A future HDF5, LeRobot, RLDS/OXE, MimicGen, or RoboCasa adapter can
describe its source and splits in one place, while a trainer can select policy
type and loss family without editing scripts.

Kai0-compatible training profiles use a LightX2V-style layered config layout:
`configs/base/model/*.yaml` defines shared model shape,
`configs/base/train/*.yaml` defines launcher and training defaults, and
`configs/experiments/kai0/*.yaml` only overrides task data, image mapping,
prompt, action semantics, and run names. The local profiles point data and
checkpoints at this repository (`data/` and `checkpoints/kai0/...`) while the
launcher calls a configured, read-only Kai0/OpenPI source checkout for the
heavy trainer.

`TrainingProfile` is the clearer split between model families. It validates
that `DataProfile`, `ModelFamily`, `LossProfile`, and `EmbodimentProfile` match:
VLA uses image-language-state batches and `vla_bc`; 3D policies use
point/state geometry and `policy_3d_bc`; WM uses context/action/future targets
and `world_model`.

`RobotCard` and `ActionSemantics` are first-class objects so the same policy can
reason about embodiment differences instead of silently assuming one robot and
one fixed 7D action shape. The demo uses `eef_delta_pose_gripper`, but the
schema supports joint, base, and vendor-specific representations.

`AlgorithmRegistry` keeps model families behind names. Today it constructs the
NumPy `UnifiedEmbodiedPolicy`; tomorrow it can register `diffusion_policy`,
`openvla_finetune`, `octo_finetune`, `robomimic_bc`, or a state-only controller
without changing the data or runtime contracts.

The minimal policy lives under `training/models/` because the MVP only needs a
model surface to validate training infra. Heavier VLA, WAM, diffusion, runtime,
and benchmark code can be added later without changing the data protocol.

## Current Non-Goals

- No fully managed long-running large model training campaign.
- No runtime server or real robot connection.
- No real HDF5, LeRobot, or RLDS/OXE parsing.
- No real video generation or diffusion.
- No DAgger loop, policy server, or simulator backend. Kai0/OpenPI training
  can be launched through the configured compatibility backend.

The interfaces are intentionally shaped so those pieces can be added later
without changing the data or training contracts.
