# Adaptation Guide

本文说明如何给 Robo-Train 添加新的数据集或新的训练框架。原则是：数据适配留在 `data/`，框架适配留在 `frameworks/`，任务和路径留在 `configs/`，常用运行命令留在 `scripts/`，不要在统一 CLI 里写具体框架分支。

## 新增数据集

新增数据集时，先判断它是“已有框架后端直接可读的数据路径”，还是“需要 Robo-Train 自己解析成通用 IR 的新格式”。

### 情况 A：已有训练后端直接读取

例如 Kai0/OpenPI 已经能读取 LeRobot 风格目录，本仓库只需要新增 profile：

1. 在对应框架任务目录新增 YAML：
   - Kai0: `configs/frameworks/kai0/tasks/<profile>.yaml`
   - 未来其它框架: `configs/frameworks/<framework>/tasks/<profile>.yaml`
2. 在 YAML 中声明：
   - `profile.config_name`
   - `profile.script_name`
   - `profile.task`
   - `profile.run_name`
   - `data.dataset_path`
   - `data.required_subdir`
   - `data.default_prompt`
   - `data.image_map`
   - `data.state_dim`
   - `data.robot_family`
   - `data.task_family`
   - `action.representation`
   - `action.action_dim`
   - `action.control_mode`
   - `action.label_source`
3. 如需常用 shell 命令，新增：
   - `scripts/frameworks/<framework>/train_<task>.sh`
4. 为该 profile 增加框架专属测试：
   - `tests/frameworks/<framework>/test_<task>.py`
5. 至少验证：
   - `python -m robo_train.cli.train --framework <framework> <profile> --dry-run --json`
   - 对应 shell wrapper 的 `--dry-run --json`
   - 若有 GPU 和数据，跑极短 smoke train。

### 情况 B：Robo-Train 需要自己解析新格式

如果新数据格式不能直接交给后端读取，需要实现 adapter，把原始数据转为 `Episode` IR。

需要修改：

1. 新增或实现 adapter：
   - `src/robo_train/data/adapters/<format>_adapter.py`
   - 继承 `BaseAdapter`
   - 实现 `to_ir(source) -> list[Episode]`
   - 如需要导出，实现 `from_ir(episodes, output_path)`
2. 注册 adapter：
   - 在 adapter 文件中调用 `AdapterRegistry.register("<format>", AdapterClass)`
   - 如果需要自动导入，检查 `src/robo_train/data/adapters/__init__.py`
3. 根据数据质量补 validator：
   - `src/robo_train/data/validators/`
   - 常见检查包括 modality 缺失、timestamp 对齐、action 语义、robot/action 维度兼容。
4. 如需要预处理，补 processor：
   - `src/robo_train/data/processors/`
   - 常见处理包括 modality select、time sync、normalization、batch build。
5. 如需要模型家族专属输入，补或复用 DataView：
   - VLA: `VLADataView`
   - 3D: `Policy3DDataView`
   - World Model: `WorldModelDataView`
6. 新增共享测试：
   - `tests/shared/test_<format>_adapter.py`
   - 覆盖 adapter 输出的 `Episode`、action semantics、stream shape、timestamp、robot card。
7. 如数据集要被某个训练框架直接跑，再补对应框架 profile：
   - `configs/frameworks/<framework>/tasks/<profile>.yaml`

不要做：

- 不要在 `src/robo_train/cli/train.py` 中写数据集名称判断。
- 不要在 adapter 中做 tokenizer、模型编码或后端专属 batch 拼装。
- 不要把绝对路径写在 Python 代码中；路径放在 YAML config 或环境变量中。

## 新增训练框架

新增框架时，按 Kai0 的 plugin 结构复制，不要修改 Kai0 逻辑，不要在统一 CLI 中加框架分支。

### 需要新增的代码

在 `src/robo_train/frameworks/<name>/` 下新增：

```text
profile_schema.py     # Pydantic schema: profile/data/model/train/launcher/checkpoint/env
loader.py             # list_profiles/load_profile，读取 configs/frameworks/<name>/tasks/*.yaml
converter.py          # 转成 ExperimentConfig 和 JSON payload
launcher.py           # 解析真实后端 command/env/log/checkpoint/dataset
plugin.py             # 实现 FrameworkPlugin 并注册到 FRAMEWORK_REGISTRY
__init__.py           # 导出公开 API
```

框架 plugin 必须实现 `src/robo_train/frameworks/base.py` 中的协议：

```text
list_profiles()
load_profile(name_or_script)
to_experiment_config(profile, ...)
build_launch_payload(profile, ...)
```

注册方式参考：

```python
FRAMEWORK_REGISTRY.register(MyFrameworkPlugin())
```

并在 `src/robo_train/frameworks/__init__.py` 中导入该框架包，让注册生效。

### 需要新增的配置

```text
configs/frameworks/<name>/base/
  <backend>.yaml       # launcher、默认 train 参数、checkpoint/env 默认值
configs/frameworks/<name>/tasks/
  <task>.yaml          # 数据路径、prompt、image_map、action 语义、run_name
```

配置中应保留可扩展字段：

- `launcher`: 后端类型、source root、python/torchrun 命令。
- `data`: dataset path、required subdir、format、image map、state/action dim、robot/task family。
- `model`: 模型族、policy type、latent/action horizon、其它模型参数。
- `train`: batch size、num workers、steps、log/save interval、resume/overwrite/wandb。
- `checkpoint`: 预训练权重或恢复路径。
- `env`: 后端需要的环境变量默认值。

### 需要新增的脚本

常用命令放在：

```text
scripts/frameworks/<name>/train_<task>.sh
```

脚本只负责：

- 设置 `PROJECT_ROOT`
- 设置 `PYTHONPATH`
- 根据 `CUDA_VISIBLE_DEVICES` 做必要的 batch/worker/exp_name 默认值
- 调用统一 CLI：

```bash
python -m robo_train.cli.train --framework <name> <profile> "$@"
```

不要把大量 Python 逻辑写进 shell。复杂逻辑应进入该框架的 `launcher.py` 或配置。

### 需要新增的测试

框架专属测试放在：

```text
tests/frameworks/<name>/
```

至少覆盖：

- profile 能被 `loader.py` 找到并加载。
- profile 转出的 `ExperimentConfig` 正确。
- `--dry-run --json` 不需要真实后端即可解析 payload。
- shell wrapper 调用了 `python -m robo_train.cli.train --framework <name>`。
- 后端 command/env/log/checkpoint/dataset 路径符合预期。
- secret 环境变量不会出现在 dry-run JSON 中；真实运行 payload 可以通过 runtime env 传给子进程。

共享协议测试放在：

```text
tests/shared/
```

例如新增 DataView、LossProfile、Adapter、Validator 时应放到 shared 测试，而不是框架目录。

## 新增模型家族或训练目标

如果新框架需要新的训练家族，而不是复用 VLA / 3D / World Model：

1. 扩展 `src/robo_train/schema/training_profile.py`
   - `DataProfile`
   - `ModelFamily`
   - `LossProfile`
   - compatibility validation
2. 新增 DataView：
   - `src/robo_train/data/views/<family>_view.py`
   - 在 `src/robo_train/data/views/base.py` 或 `__init__.py` 中接入。
3. 新增 loss：
   - `src/robo_train/training/losses.py` 或拆分出的 loss module
   - 在 `src/robo_train/training/loss_registry.py` 注册。
4. 如需要新 policy/algorithm：
   - `src/robo_train/training/models/`
   - `src/robo_train/training/algorithm_registry.py`
5. 增加共享测试：
   - `tests/shared/test_data_views_and_losses.py`
   - `tests/shared/test_training_family_profiles.py`

## 当前 TODO 占位

DreamZero/WAM 目前只保留目录：

```text
src/robo_train/frameworks/dreamzero/
configs/frameworks/dreamzero/
scripts/frameworks/dreamzero/
```

这些目录用于后续接入真实 DreamZero/WAM 后端。当前不要注册 partial plugin，也不要添加可运行 wrapper，除非真实 backend command、profile schema、dataset contract 和 checkpoint 格式已经确定。

## 最低验证清单

每次新增数据集或框架后至少运行：

```bash
pytest -q
python -m robo_train.cli.train --framework <framework> <profile> --dry-run --json
```

如果新增了 shell wrapper：

```bash
bash scripts/frameworks/<framework>/train_<task>.sh --dry-run --json
```

如果目标是可训练路径，还需要在单卡上跑短 smoke train，并检查：

- 能读到数据和 norm stats。
- 能恢复 checkpoint。
- 首个 step 有 loss/grad_norm/param_norm。
- 稳定后的 s/iter 与参考后端同量级。
