# UEFS Architecture

`src` is organized as a Protocol-first data infra and training infra MVP. Data,
robots, action semantics, coordinate frames, scenes, and training views are
normalized before a specific VLA, 3D policy, or world-model training family is
selected. Runtime, benchmark, and real-robot deployment layers are intentionally
left out of this first cut.

## Main Boundaries

1. **Dataset identity** uses `DatasetManifest` to record source framework,
   splits, sampling weight, stats, quality flags, and lineage.
2. **Adapter** converts external data into Universal Embodied IR.
3. **Embodiment protocol** keeps `RobotCard`, `ActionSemantics`, `FrameGraph`,
   `SceneCard`, and optional `SensorCard` metadata attached to episodes.
4. **Validator** checks the IR and returns explicit errors and warnings.
5. **Processor** turns IR episodes into `EmbodiedBatch` objects.
6. **Training profile** validates the intended `DataProfile`, `ModelFamily`,
   `LossProfile`, and `EmbodimentProfile`.
7. **Data view** converts the shared batch into VLA, 3D policy, or world-model
   inputs.
8. **Experiment config** keeps dataset, policy/model, and train params together.
9. **Algorithm registry** creates policies by config name.
10. **Loss registry** computes model-family-specific objectives.
11. **Policy** consumes batches and returns training predictions.
12. **Artifacts and stats** record checkpoints, processor stats, and norm stats.

```text
DatasetManifest + ExperimentConfig
  -> Adapter (data/adapters/)
  -> Universal Embodied IR (schema/)
  -> Validator (data/validators/)
  -> ProcessorPipeline (data/processors/)
  -> TrainingProfile (schema/training_profile.py)
  -> DataView (data/views/)
      - VLADataView
      - Policy3DDataView
      - WorldModelDataView
  -> MixedEpisodeDataLoader + NormStats (data/)
  -> AlgorithmRegistry (training/algorithm_registry.py)
  -> LossRegistry + LossProfile (training/loss_registry.py)
  -> UnifiedEmbodiedPolicy (training/models/policy.py)
  -> TrainingArtifact
```

## Compatibility Strategy

- LeRobot-style datasets should become `DatasetManifest` plus an adapter that
  emits `Episode` objects.
- VLA, 3D, and WM training should share adapters and split at `DataView`, not
  fork into unrelated dataloaders.
- 3D-aware training should keep `FrameGraph` and `SceneCard` with the episode so
  point clouds, camera streams, object frames, and action effects can be aligned
  later without changing the data protocol.
- robomimic-style HDF5 experiments should map config fields into
  `ExperimentConfig` and register an algorithm in `AlgorithmRegistry`.
- VLA/3D/WM losses should be selected through `LossProfile` so the same trainer
  shell can call language-conditioned BC, geometry-aware policy losses, or
  future-prediction losses.
- Diffusion Policy-style models should register a future policy type without
  changing the data infra.
- OpenVLA/Octo-style fine-tuning should load checkpoints through
  `PolicyConfig.pretrained_checkpoint` and record the result in
  `TrainingArtifact`.
- MimicGen/RoboCasa data should keep generated-data provenance in
  `DatasetManifest.lineage` and `quality_flags`.

`EmbodimentProfile` is the place for robot-arm differences such as Franka
single-arm, ALOHA dual-arm, UR single-arm, or mobile manipulation. The profile
selects action-space assumptions and runtime behavior; it should not duplicate
the whole dataloader or trainer unless the model family truly changes.

All core paths are deterministic when constructed with a seed. The mock model
uses NumPy projections, deterministic language hashing, shared world-action
latents, action clipping, and JSON-serializable runtime outputs.
