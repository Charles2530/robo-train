# Kai0 Framework Plugin

This directory adapts Robo-Train configs to the real Kai0/OpenPI training
backend. Kai0 source code remains read-only in `/data/buaa/code/djy/kai0`.

## File Responsibilities

- `profile_schema.py`
  - Defines the Pydantic schema for a resolved Kai0 profile.
  - Owns Kai0-specific config sections: `profile`, `launcher`, `model`, `data`,
    `action`, `checkpoint`, `train`, and `env`.
  - Validates dataset/framework compatibility, for example rejecting a dataset
    whose `supported_frameworks` does not include `kai0`.

- `loader.py`
  - Lists and loads YAML profiles from `configs/frameworks/kai0/tasks/`.
  - Resolves layered `defaults`, including dataset/robot configs from
    `configs/datasets/<dataset>/<robot>.yaml`.
  - Does not build commands, touch files, or apply training overrides.

- `converter.py`
  - Converts a resolved Kai0 profile into shared Robo-Train objects:
    `ExperimentConfig`, `DatasetManifest`, `PolicyConfig`, and `TrainerConfig`.
  - Produces JSON-friendly dry-run payload metadata.
  - Does not know how to execute Kai0.

- `launcher.py`
  - Resolves the real backend command, environment, dataset path, checkpoint
    path, log path, and optional preprocessing such as encode dataset prep.
  - Performs preflight checks for required dataset/checkpoint paths.
  - Is the only Kai0 module that should know about `scripts/train.py`,
    `LD_LIBRARY_PATH`, `KAI0_ROOT`, or the Kai0 source checkout.

- `plugin.py`
  - Implements the generic `FrameworkPlugin` protocol.
  - Wires loader, converter, and launcher together.
  - Registers `kai0` in `FRAMEWORK_REGISTRY`.

## Config Shape

Kai0 tasks are built from three config layers:

1. Shared model config: `configs/base/model/*.yaml`
2. Dataset/robot config: `configs/datasets/<dataset>/<robot>.yaml`
3. Framework/backend config: `configs/frameworks/kai0/base/*.yaml`
4. Task/action override: `configs/frameworks/kai0/tasks/*.yaml`

Unsupported dataset/framework combinations should fail during profile loading,
before any backend process is launched.
