# LightX2V-Style Infra Slimdown Implementation Plan

> **Archived/Stale:** This plan predates the `robo_train` package rename and plugin migration. It is kept only for historical context; use the current `src/robo_train/`, `configs/frameworks/`, and `scripts/frameworks/` layout as the source of truth.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the repo around data infra and training infra, following the LightX2V-style split of configs, scripts, data utilities, and training utilities.

**Architecture:** Keep protocol schema and data infra as the first stable layer. Move trainer/model/loss/artifact code into `src/training/`, keep only a minimal NumPy action policy for training smoke tests, and remove runtime/env/eval/alignment surfaces from the MVP package.

**Tech Stack:** Python 3.12, NumPy, Pydantic, PyYAML, pytest.

## Global Constraints

- Keep the package name `src` for this demo.
- Do not add heavy ML dependencies.
- Preserve a runnable synthetic data to training path.
- Remove FastAPI/httpx/uvicorn runtime dependencies from MVP.
- Keep tests as the source of truth for the new package boundary.

---

### Task 1: Structure Contract

**Files:**
- Modify: `tests/test_project_structure.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: tests that assert the MVP owns only `schema`, `data`, `training`, `scripts`, and docs/configs.

- [ ] Write failing tests for the slim package structure.
- [ ] Run `PYTHONPATH=. .venv/bin/pytest tests/test_project_structure.py -q` and verify it fails because old runtime/model/env/eval packages still exist.
- [ ] Remove runtime dependencies from `pyproject.toml`.
- [ ] Run the project-structure tests again after package deletion.

### Task 2: Training Package Migration

**Files:**
- Create: `src/training/`
- Move/adapt: `src/trainer/*` into `src/training/*`
- Create: `src/training/models/policy.py`
- Modify: training tests to import from `src.training`.

**Interfaces:**
- Produces: `src.training.build_policy_from_config`, `src.training.MockTrainer`, `src.training.compute_loss_profile`, and `src.training.TrainingArtifact`.

- [ ] Write tests that import the new training package names.
- [ ] Run focused training tests and verify import failures.
- [ ] Add the minimal `src/training` implementation.
- [ ] Delete old `src/trainer` and `src/model` after consumers are migrated.

### Task 3: Data Infra Focus

**Files:**
- Create: `src/data/storage/`
- Create: `src/data/dataloader/`
- Modify: `README.md`, `README_zh.md`, `docs/architecture.md`, `docs/framework_mapping.md`

**Interfaces:**
- Produces: `DatasetShardSpec`, `NormStats`, and `MixedEpisodeDataLoader` as lightweight infra placeholders with tests.

- [ ] Write tests for norm stats and mixed dataloader behavior.
- [ ] Run focused data tests and verify failures.
- [ ] Add minimal data infra code.
- [ ] Update docs to describe the data/training-first MVP.

### Task 4: Cleanup And Verification

**Files:**
- Delete: `src/runtime/`, `src/envs/`, `src/eval/`, `src/train_deploy_alignment/`
- Delete or rewrite: runtime/env/eval/alignment tests.
- Modify: `.gitignore`

**Interfaces:**
- Produces: a smaller MVP with no runtime server/client test surface.

- [ ] Remove obsolete tests and generated files.
- [ ] Add ignore rules for `__pycache__/` and macOS `._*`.
- [ ] Run `PYTHONPATH=. .venv/bin/pytest -q`.
