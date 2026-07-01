# Framework Mapping

UEFS is now a small Protocol-first data infra and training infra demo. Data,
robot embodiment, action semantics, coordinate frames, scenes, and training
views are normalized before any specific model family is chosen. Following the
LightX2V style, the MVP keeps top-level configs and scripts visible while
concentrating real code in `data/`, `schema/`, and `training/`. Runtime,
benchmark, and real-robot deployment surfaces are intentionally out of scope for
this phase.

## File-Level Mapping

| UEFS module | Reference framework | Design idea | Reference source |
| --- | --- | --- | --- |
| `schema/` | StarVLA, LeRobot | Raw, model-agnostic protocol is shared across data infra and training infra. | StarVLA dataloader/framework separation, LeRobot metadata |
| `data/adapters/` | StarVLA, LeRobot, RLDS/OXE | Dataset-specific code stops at raw IR and does not perform model encoding. | StarVLA `dataloader/`, LeRobot datasets, Open X-Embodiment/RLDS style episode records |
| `data/storage/` | LightX2V, LeRobot | `NormStats` and shard specs make prepared data reusable and keep normalization out of ad hoc scripts. | prepared dataset artifacts, metadata, and training statistics |
| `data/dataloader/` | LightX2V, robomimic | `MixedEpisodeDataLoader` provides deterministic multi-dataset mixing for training. | dataset mixing and sampler utilities |
| `data/views/` | OpenVLA, Octo, DP3, DreamZero | Model-family-specific DataView objects split one shared `EmbodiedBatch` into VLA, 3D, or world-model inputs. | VLA multimodal batches, 3D point/state batches, video/world-model context-target batches |
| `data/views/vla_view.py` | OpenVLA, Octo, LeRobot | `VLADataView` keeps image, language, proprioception, robot metadata, and action targets together. | generalist VLA fine-tuning and language-conditioned robot policies |
| `data/views/policy3d_view.py` | DP3, ManiSkill, Isaac Lab | `Policy3DDataView` emphasizes point/state geometry and action targets. | 3D Diffusion Policy, point-cloud/state manipulation policies |
| `data/views/world_model_view.py` | DreamZero, video-generation training stacks | `WorldModelDataView` emits context observations, action history, and future-state targets. | world-action modeling and action-conditioned future prediction |
| `schema/dataset_manifest.py` | LeRobot, robomimic, MimicGen, RoboCasa | Dataset identity, split ratios, source framework, quality flags, stats path, and lineage are explicit training inputs. | LeRobot dataset metadata, robomimic HDF5 datasets, MimicGen/RoboCasa generated data |
| `schema/frame_graph.py` | ManiSkill, Isaac Lab, 3D-VLA, PointWorld | `FrameGraph` makes calibrated frames and perception-estimated object frames first-class IR, so RGB-D and point clouds are not anonymous tensors. | simulator/world frame contracts, camera calibration, 3D scene alignment |
| `schema/scene_card.py` | RoboCasa, 3D-VLA, GAM | `SceneCard` and `ObjectCard` carry object categories, affordances, workspace bounds, and object frames. | object-centric manipulation scenes and goal relations |
| `schema/sensor_card.py` | LeRobot, ROS-style robot stacks | `SensorCard` stores sensor type, frame, fps, shape, and camera intrinsics or extrinsic references. | robot metadata and calibration records |
| `schema/experiment_config.py` | robomimic, LeRobot, LightX2V | Dataset, model/policy, and train params are configured together, not hidden in scripts. | robomimic config system, LeRobot training configs, LightX2V visible configs |
| `schema/training_profile.py` | robomimic, OpenVLA, DP3, DreamZero | `TrainingProfile` validates `DataProfile`, `ModelFamily`, `LossProfile`, and `EmbodimentProfile` compatibility. | config-driven selection of observation modality, algorithm family, loss family, and robot profile |
| `config/layered.py` | LightX2V | `defaults`-based YAML layering keeps base model/train config separate from task-specific overrides. | visible composable configs |
| `frameworks/dreamzero/` | DreamZero, LightX2V, robomimic | Reserved TODO folder for a future DreamZero/WAM plugin. It is intentionally not registered until the real backend contract is defined. | world-action-model config compatibility planning |
| `frameworks/kai0/{loader,converter,launcher,plugin}.py` | kai0, LightX2V, robomimic | Kai0 train profiles resolve into local `ExperimentConfig` objects and real Kai0/OpenPI backend commands while keeping data/checkpoint paths in this repo. | train config compatibility without copying old repo internals |
| `training/models/policy.py` | StarVLA, OpenVLA, Octo | The MVP keeps a tiny `forward()` policy only to exercise the training path. | StarVLA model entrypoint shape without heavy backbones |
| `training/algorithm_registry.py` | robomimic, LeRobot | Algorithms and policies are registered by name so BC, VLA fine-tuning, or state-only policies can share launcher code. | robomimic algorithm abstraction, LeRobot policy factory pattern |
| `training/loss_registry.py` | robomimic, Diffusion Policy, DreamZero | Loss functions are selected by `LossProfile`, not hardcoded inside the trainer. | separate imitation, 3D geometry, and future-prediction losses |
| `training/artifacts.py` | robomimic, OpenVLA, Octo | Checkpoints carry experiment id, dataset ids, policy type, metrics, and processor stats. | reproducible imitation-learning and VLA fine-tuning artifacts |
| `configs/` and `scripts/` | LightX2V, LeRobot, kai0, DreamZero | Keep config and runnable task entrypoints visible at the top level, including `scripts/frameworks/kai0` train wrappers and DreamZero TODO placeholders. | LightX2V `configs/`, LeRobot examples, kai0 train scripts |

## Shared Architecture Pattern

The strongest frameworks separate seven concerns:

1. **Data identity and provenance.** LeRobot, robomimic, MimicGen, and RoboCasa
   make dataset source, split, task family, and quality assumptions visible.
   UEFS represents that with `DatasetManifest` and dataset adapters.
2. **Experiment configuration.** robomimic-style configs make the same launcher
   work across algorithms. UEFS now uses `ExperimentConfig` for dataset,
   policy/model, and train params only.
3. **Protocol-first views, not one universal tensor.** The adapter layer should
   not fork into unrelated loaders for every model, and it also should not
   squeeze every modality into one fixed tensor. UEFS keeps one
   `Episode`/`EmbodiedBatch` protocol, then uses `VLADataView`,
   `Policy3DDataView`, and `WorldModelDataView` to present the same data in the
   shape expected by VLA, 3D, and WM trainers.
4. **Algorithm registration.** BC, diffusion, VLA fine-tuning, and state-only
   policies should be selected by config, not by editing scripts. UEFS uses
   `AlgorithmRegistry` for this seam.
5. **LossProfile selection.** VLA action loss, 3D geometry-aware loss, and
   world-model future-prediction loss are separate trainer concerns. UEFS uses
   `LossProfile` plus `LossRegistry` instead of hiding the loss inside one
   monolithic trainer.
6. **EmbodimentProfile separation.** Robot-specific differences belong in
   `EmbodimentProfile`, `RobotCard`, `ActionSemantics`, action projection, and
   runtime profiles. The trainer should not fork completely for Franka, ALOHA,
   or UR arms unless the algorithm itself changes.
7. **Artifacts and evidence.** Checkpoints, processor stats, and dataset stats
   must travel together. UEFS keeps this in `TrainingArtifact` and `NormStats`.

## Clarified VLA / 3D / WM Split

The clean split is not "three unrelated dataloaders." The adapter layer stays
unified, and model-family differences begin at DataView and LossProfile:

| Axis | VLA | 3D Policy | World Model / WM |
| --- | --- | --- | --- |
| DataView | `VLADataView` | `Policy3DDataView` | `WorldModelDataView` |
| Main input | RGB/video history, language, state | point cloud or state geometry | context observations, action history, future target |
| Main target | action chunk | action chunk | future state/latent |
| Similar references | OpenVLA, Octo, LeRobot | DP3, ManiSkill, Isaac Lab | DreamZero, video generation |
| Robot split | via `EmbodimentProfile` and action semantics | via `EmbodimentProfile` and geometry/state schema | via `EmbodimentProfile` and action conditioning |

## Training Differences

| Model family | What changes in training | UEFS extension point |
| --- | --- | --- |
| Behavior cloning / robomimic-style imitation | Supervised action loss over HDF5 or episode datasets; observation/action normalization and dataset splits matter most. | `DatasetManifest`, `ExperimentConfig.trainer.algorithm="behavior_cloning"`, `ProcessorPipeline`, `ActionHead`. |
| Diffusion Policy | Predicts action sequences; training loss is sequence denoising or flow/diffusion rather than plain MSE. | Future `policy_type="diffusion_policy"` in `AlgorithmRegistry`. |
| VLA fine-tuning / OpenVLA / Octo | Loads a pretrained vision-language-action model, may freeze backbones, and uses robot/action adapters to map canonical actions. | `PolicyConfig.pretrained_checkpoint`, `freeze_backbone`, `ActionSemantics`. |
| World-action / DreamZero-style models | Trains future latent and action prediction jointly so dynamics shape action representation. | `WorldModelDataView`, `world_model` loss profile, `future_latent`. |
| Generated demonstration data / MimicGen / RoboCasa | Uses generated or scripted demonstrations with lineage, quality filters, and task-family metadata. | `DatasetManifest.lineage`, `quality_flags`, adapter registry. |

The names are intentionally flat and snake_case so future real backends can
replace mock NumPy internals without changing external imports or CLI commands.
