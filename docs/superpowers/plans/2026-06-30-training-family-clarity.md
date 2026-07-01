# Training Family Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UEFS clearly distinguish VLA, 3D policy, and world-model training paths through data views, loss profiles, and embodiment profiles.

**Architecture:** Keep adapters unified around `Episode` IR, then split model-family behavior in `data/views/` and `trainer/loss_*`. Extend `ExperimentConfig` with a validated `TrainingProfile` so configs declare data/model/loss/robot-family compatibility.

**Tech Stack:** Python 3.10+, Pydantic v2, NumPy, pytest, existing UEFS package layout.

## Global Constraints

- Keep the project lightweight and NumPy-only.
- Do not add heavyweight simulator, VLA, diffusion, or PyTorch dependencies.
- Preserve existing public APIs where possible.
- Use TDD: write failing tests before implementation.
- Use ASCII for new code and docs unless existing content requires otherwise.

---

### Task 1: Training Profile Schema

**Files:**
- Create: `src/schema/training_profile.py`
- Modify: `src/schema/__init__.py`
- Test: `tests/test_training_family_profiles.py`

**Interfaces:**
- Produces: `TrainingProfile(data_profile, model_family, loss_profile, embodiment_profile, modalities, action_space)`
- Produces: `TrainingProfile.default_vla()`, `.default_policy3d()`, `.default_world_model()`

- [ ] Write tests for compatible and incompatible training profiles.
- [ ] Run `pytest tests/test_training_family_profiles.py::test_training_profile_defaults_are_compatible -q` and observe import failure.
- [ ] Implement schema and exports.
- [ ] Run the focused test and confirm pass.

### Task 2: Model-Family Data Views

**Files:**
- Create: `src/data/views/__init__.py`
- Create: `src/data/views/base.py`
- Create: `src/data/views/vla_view.py`
- Create: `src/data/views/policy3d_view.py`
- Create: `src/data/views/world_model_view.py`
- Test: `tests/test_data_views_and_losses.py`

**Interfaces:**
- Consumes: `EmbodiedBatch`
- Produces: `BaseDataView.build(batch) -> dict`
- Produces: `build_data_view(profile_or_name)`

- [ ] Write tests showing VLA, 3D, and WM views emit different keys.
- [ ] Run focused data-view tests and observe import failure.
- [ ] Implement views with deterministic NumPy behavior.
- [ ] Run focused data-view tests and confirm pass.

### Task 3: Loss Registry and Family Losses

**Files:**
- Create: `src/trainer/loss_registry.py`
- Create: `src/trainer/losses_vla.py`
- Create: `src/trainer/losses_3d.py`
- Create: `src/trainer/losses_wm.py`
- Modify: `src/trainer/__init__.py`
- Test: `tests/test_data_views_and_losses.py`

**Interfaces:**
- Produces: `LossRegistry.register(name, fn)`
- Produces: `compute_loss_profile(loss_profile, prediction, target) -> dict[str, float]`

- [ ] Write tests that each loss profile returns expected named losses.
- [ ] Run focused tests and observe import failure.
- [ ] Implement registry and three loss modules.
- [ ] Run focused tests and confirm pass.

### Task 4: Config Samples and Experiment Wiring

**Files:**
- Modify: `src/schema/experiment_config.py`
- Add: `configs/vla_demo.json`
- Add: `configs/policy3d_demo.json`
- Add: `configs/world_model_demo.json`
- Modify: `tests/test_project_structure.py`

**Interfaces:**
- Consumes: `TrainingProfile`
- Produces: `ExperimentConfig.training_profile`

- [ ] Write tests that all three config samples validate and use matching profiles.
- [ ] Run focused config tests and observe failure.
- [ ] Implement `training_profile` field and add config samples.
- [ ] Run focused config tests and confirm pass.

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `README_zh.md`
- Modify: `docs/architecture.md`
- Modify: `docs/framework_mapping.md`
- Modify: `tests/test_project_structure.py`

**Interfaces:**
- Produces: docs that explicitly describe `DataView`, `LossProfile`, and `EmbodimentProfile`.

- [ ] Write structure test requiring the new concepts in docs.
- [ ] Run focused doc test and observe failure.
- [ ] Update docs.
- [ ] Run focused doc test and confirm pass.

### Task 6: Verification

**Files:**
- All touched files.

**Interfaces:**
- Produces: passing full test suite and runnable local demo.

- [ ] Run `.venv/bin/python -m pytest -q`.
- [ ] Run `.venv/bin/python -m robo_train.scripts.run_local_demo`.
- [ ] Remove generated caches.
- [ ] Summarize verification evidence.
