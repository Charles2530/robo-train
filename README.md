# Robo-Train

Robo-Train 是一个面向具身智能模型训练的可扩展框架。它把数据协议、训练任务配置、模型/框架后端启动逻辑分开，使同一套数据基础设施可以服务 VLA、3D policy、world model / WAM 等不同训练家族。

当前重点是稳定 Kai0/OpenPI 兼容训练：本仓库负责配置、路径、脚本、dry-run、环境变量和训练 payload；真实 Kai0 训练后端仍由只读的 `/data/buaa/code/djy/kai0` 执行。这样可以保证迁移后的脚本行为和原 Kai0 `tools_charles` 训练逻辑保持一致，同时为后续扩展其它 VLA、3D 和 WM 框架保留统一入口。

## 当前能力

- 统一训练入口：`python -m robo_train.cli.train --framework <name> <profile>`
- 框架插件协议：`src/robo_train/frameworks/base.py`
- Kai0 插件化适配：`src/robo_train/frameworks/kai0/`
- Kai0 兼容 shell wrapper：`scripts/frameworks/kai0/`
- 多层 YAML 配置：`configs/base/model/` + `configs/datasets/` + `configs/frameworks/<framework>/base/` + `configs/frameworks/<framework>/tasks/`
- 通用数据 IR：`Episode`、`ActionSemantics`、`RobotCard`、`TaskCard`、`FrameGraph`、`SceneCard`
- 通用数据处理：adapter、validator、processor、data view、mixed dataloader、norm stats
- 训练家族抽象：`TrainingProfile`、`DataProfile`、`ModelFamily`、`LossProfile`、`EmbodimentProfile`
- VLA / 3D / World Model 数据视图：`VLADataView`、`Policy3DDataView`、`WorldModelDataView`

## 已支持的训练框架

| 框架 | 状态 | 入口 | 说明 |
| --- | --- | --- | --- |
| Kai0 / OpenPI JAX | 已支持 | `configs/frameworks/kai0/base/kai0_jax.yaml`、`scripts/frameworks/kai0/*.sh` | 当前主力训练路径。支持 Table30/Table30v2 任务、pi05 base checkpoint、JAX 后端、单卡/多卡脚本默认值。 |
| Kai0 / OpenPI PyTorch | 已支持 | `configs/frameworks/kai0/base/kai0_torch.yaml`、`scripts/frameworks/kai0/train_pytorch.sh` | 保留 PyTorch 训练入口和 flatten/fold AWBC profile。 |
| DreamZero / WAM | 仅预留 TODO | `src/robo_train/frameworks/dreamzero/`、`configs/frameworks/dreamzero/`、`scripts/frameworks/dreamzero/` | 只保留目录和 TODO，不注册 plugin，不提供可运行实现。 |

统一 CLI 当前实际注册的 framework 只有 `kai0`：

```bash
PYTHONPATH=src python - <<'PY'
from robo_train.frameworks import FRAMEWORK_REGISTRY
print(FRAMEWORK_REGISTRY.names())
PY
```

## 数据集与框架组合

`datasets` 当前是指向 `/data/buaa/vla_dataset/processed/` 的软链接。数据集类型、机器人构型、格式、image map 和支持矩阵放在 `configs/datasets/<dataset>/<robot>.yaml`；框架 task profile 通过 `defaults` 组合数据集构型 config、模型 config 和框架 base config，并在 task 内覆盖具体任务路径、prompt、保存名和训练参数。运行逻辑是：

```text
dataset family + robot embodiment + framework type + task/path/action overrides
```

不支持的组合必须在 profile 加载阶段报错。例如 `LIBERO` 当前保留数据集配置，但没有声明支持 `kai0`，因此不能直接用 Kai0 profile 启动。

| 数据集 / 任务 | Dataset/Robot Config | 任务路径 | Kai0 Profile | 动作/标签 |
| --- | --- | --- | --- | --- |
| RoboChallenge Table30 arrange flowers | `configs/datasets/robochallenge_table30/arx5_single_arm.yaml` | `datasets/RoboChallenge_Table30_v2.1_pyav_g2/arrange_flowers` | `pi05_arrange_flowers`、`pi05_arrange_flowers_sparse`、`pi05_arrange_flowers_encode` | ARX5 单臂，EEF action，7D。 |
| RoboChallenge Table30v2 arrange flowers | `configs/datasets/robochallenge_table30v2/arx5_single_arm.yaml` | `datasets/RoboChallenge_Table30v2_v2.1_pyav_g2/arrange_flowers` | `pi05_arrange_flowers_table30v2` | ARX5 单臂，EEF quaternion action，8D。 |
| RoboChallenge Table30v2 put the books back | `configs/datasets/robochallenge_table30v2/aloha_dual_arm.yaml` | `datasets/RoboChallenge_Table30v2_v2.1_pyav_g2/put_the_books_back` | `pi05_put_the_books_back_table30v2` | ALOHA 双臂，EEF quaternion action，16D。 |
| RoboChallenge Table30v2 put the books back joint | 同上 | 同上 | `pi05_put_the_books_back_table30v2_joint` | ALOHA 双臂，next-state joint action，14D。 |
| RoboChallenge Table30v2 put the books back joint delta | 同上 | 同上 | `pi05_put_the_books_back_table30v2_joint_delta` | ALOHA 双臂，delta joint action，14D。 |
| xiaoyu Task A flatten/fold AWBC | `configs/datasets/xiaoyu_task_a/agilex_dual_arm.yaml` | `datasets/xiaoyu_Task_A/advantage` | `pi05_flatten_fold_awbc_torch` | AgileX 双臂，joint action，14D，PyTorch profile。 |
| LIBERO | `configs/datasets/libero/generic.yaml` | `datasets/LIBERO` | 暂不支持 | 需要后续预处理/框架适配。 |

## 数据格式支持

| 数据格式 / Adapter | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| Synthetic | 已支持 | `src/robo_train/data/adapters/synthetic_adapter.py` | 用于本仓库的单元测试和本地 demo。 |
| Kai0 / LeRobot v3 profile path | 已用于真实训练 | `configs/frameworks/kai0/tasks/*.yaml` | 真实训练直接交给 Kai0/OpenPI 后端读取，不在本仓库重复实现完整 LeRobot loader。 |
| HDF5 | TODO stub | `src/robo_train/data/adapters/hdf5_adapter.py` | 已有 adapter 类和注册名，尚未实现真实解析。 |
| LeRobot | TODO stub | `src/robo_train/data/adapters/lerobot_adapter.py` | 已有 adapter 类和注册名，尚未实现通用 IR 转换。 |
| RLDS / OXE | TODO stub | `src/robo_train/data/adapters/rlds_oxe_adapter.py` | 已有 adapter 类和注册名，尚未实现真实解析。 |

## 常用命令

安装和测试：

```bash
pip install -e .
pytest -q
```

Kai0 dry-run：

```bash
python -m robo_train.cli.train --framework kai0 pi05_put_the_books_back_table30v2_joint --dry-run --json
```

Table30v2 put-books-back joint 单卡训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh
```

兼容错误拼写的单卡环境变量：

```bash
CUDA_VISIBEL_DEVICES=0 bash scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh
```

两卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh
```

8 卡训练时，Table30v2 脚本默认使用 `batch_size=256`、`num_workers=32`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh
```

也可以显式传环境变量或 CLI 参数覆盖 dataloader workers：

```bash
NUM_WORKERS=32 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/frameworks/kai0/train_put_the_books_back_table30v2_joint.sh
```

## 目录结构

```text
configs/
  base/model/                    # 可复用模型层配置
  datasets/                      # 数据集家族/机器人构型、格式和框架支持矩阵
  frameworks/kai0/base/          # Kai0 launcher 和训练默认值
  frameworks/kai0/tasks/         # Kai0 任务 profiles
  frameworks/dreamzero/          # DreamZero TODO 占位
scripts/
  frameworks/kai0/               # Kai0 兼容训练脚本
  frameworks/dreamzero/          # DreamZero TODO 占位
src/robo_train/
  cli/                           # 统一训练 CLI
  config/                        # layered YAML loader
  data/                          # adapters, validators, processors, views, dataloader
  frameworks/                    # framework plugin 协议和实现
    kai0/                        # Kai0 plugin
    dreamzero/                   # TODO 占位
  schema/                        # 通用具身数据和实验配置 schema
  training/                      # 训练抽象、loss、artifact、mock policy
tests/
  shared/                        # 共享 data/schema/training 测试
  frameworks/kai0/               # Kai0 专属兼容测试
```

## 扩展入口

新增数据集或新增训练框架时，不要把逻辑写死在 CLI 里。请参考 [adapt.md](adapt.md)：

- 新增数据集/机器人构型：优先新增 `configs/datasets/<dataset>/<robot>.yaml`；如需解析新格式，再改 `src/robo_train/data/` 和 `tests/shared/`。
- 新增具体训练任务：在 `configs/frameworks/<framework>/tasks/` 中通过 `defaults` 引用 dataset/robot config，并在 task 内填写具体任务路径、prompt、保存名和训练参数；不支持的组合不要写入 `supported_frameworks`。
- 新增训练框架：新增 `src/robo_train/frameworks/<name>/`、`configs/frameworks/<name>/`、`scripts/frameworks/<name>/` 和 `tests/frameworks/<name>/`。
- 统一 CLI `src/robo_train/cli/train.py` 应只通过 `FRAMEWORK_REGISTRY` 调用 plugin，不应该出现具体框架的分支判断。
