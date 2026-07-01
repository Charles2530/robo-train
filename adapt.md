# Adaptation Guide

本文说明如何给 Robo-Train 添加新的数据集或新的训练框架。原则是：数据适配留在 `data/`，框架适配留在 `frameworks/`，任务和路径留在 `configs/`，常用运行命令留在 `scripts/`，不要在统一 CLI 里写具体框架分支。

## 新增数据集

新增数据集时，先把数据集本身写成独立 config，再决定它是“已有框架后端直接可读的数据路径”，还是“需要 Robo-Train 自己解析成通用 IR 的新格式”。

### 数据集配置层

每个具身数据集应按“数据集家族/机器人构型”组织配置，例如：

```text
configs/datasets/robochallenge_table30v2/aloha_dual_arm.yaml
configs/datasets/robochallenge_table30v2/arx5_single_arm.yaml
configs/datasets/xiaoyu_task_a/agilex_dual_arm.yaml
configs/datasets/libero/generic.yaml
```

数据集/机器人 config 只描述数据集家族、机器人构型和通用数据形态，不描述某个具体任务：

```yaml
data:
  dataset_id: robochallenge_table30v2_aloha_dual_arm
  dataset_type: RoboChallenge_Table30v2
  dataset_path: datasets/RoboChallenge_Table30v2_v2.1_pyav_g2
  robot_family: aloha_dual_arm
  task_family: generic
  format: lerobot_v3
  supported_frameworks:
    - kai0
  preprocessors:
    - pyav_g2_reencoded
```

如果某个组合不支持，例如 LIBERO 暂时不能直接用 Kai0 训练，就不要在
`supported_frameworks` 里写 `kai0`。Kai0 profile 加载时会直接报错，而不是启动后端才失败。

### 情况 A：已有训练后端直接读取

例如 Kai0/OpenPI 已经能读取 LeRobot 风格目录，本仓库只需要新增或复用 dataset/robot config，再新增 framework task profile：

1. 在 `configs/datasets/<dataset>/<robot>.yaml` 中声明数据集类型、机器人构型、通用路径和支持的框架。
2. 在对应框架任务目录新增 YAML：
   - Kai0: `configs/frameworks/kai0/tasks/<profile>.yaml`
   - 未来其它框架: `configs/frameworks/<framework>/tasks/<profile>.yaml`
3. 在 task YAML 的 `defaults` 中组合 dataset、model 和 framework base：
   ```yaml
   defaults:
     - ../../../base/model/pi05.yaml
     - ../../../datasets/robochallenge_table30v2/aloha_dual_arm.yaml
     - ../base/kai0_jax.yaml
   ```
4. 在 task YAML 中只声明训练任务/action 覆盖：
   - `profile.config_name`
   - `profile.script_name`
   - `profile.task`
   - `profile.run_name`
   - `data.dataset_path`
   - `data.default_prompt`
   - `data.task_family`
   - `action.representation`
   - `action.action_dim`
   - `action.control_mode`
   - `action.label_source`
5. 如需常用 shell 命令，新增：
   - `scripts/frameworks/<framework>/train_<task>.sh`
6. 为该 profile 增加框架专属测试：
   - `tests/frameworks/<framework>/test_<task>.py`
7. 至少验证：
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

新增框架时，按 Kai0 的 plugin 结构复制，不要修改 Kai0 逻辑，不要在统一 CLI 中加框架分支。每个框架目录都应该有 README 解释文件职责，Kai0 的示例在 `src/robo_train/frameworks/kai0/README.md`。

### 需要新增的代码

在 `src/robo_train/frameworks/<name>/` 下新增：

```text
profile_schema.py     # Pydantic schema: profile/data/model/train/launcher/checkpoint/env
loader.py             # 只负责 list/load YAML profile，并解析 defaults
converter.py          # 只负责转成 ExperimentConfig 和 dry-run JSON payload
launcher.py           # 只负责真实后端 command/env/log/checkpoint/dataset/preprocess
plugin.py             # 只负责实现 FrameworkPlugin 并注册到 FRAMEWORK_REGISTRY
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
