# Universal Embodied Foundation Stack

`src` 是一个轻量、可运行的 Python demo，用来展示一套 Protocol-first 的具身数据基础设施和训练基础设施：先统一数据、机器人本体、动作语义、坐标系、场景和训练视图协议，再用一个最小 NumPy policy 跑通训练路径。Runtime serving、benchmark env 和真机部署不放在当前 MVP 主干里。

## 这个 Demo 展示什么

- Universal Embodied IR：包含 `ActionSemantics`、`RobotCard`、`TaskCard`、`FrameGraph`、`SceneCard`、多模态 streams 和 episodes。
- StarVLA 风格的 adapter：输出 raw、model-agnostic 的 IR，而不是在数据层做 tokenizer 或模型编码。
- Validator 和 processor pipeline：把多模态机器人数据检查并转换成 policy 可用的 batch。
- `DatasetManifest` 和 `ExperimentConfig`：支持多数据集、可复现实验配置和训练元数据。
- `TrainingProfile`：显式校验 `DataProfile`、`ModelFamily`、`LossProfile` 和 `EmbodimentProfile` 是否匹配。
- 面向训练家族的 `DataView`：`VLADataView`、`Policy3DDataView`、`WorldModelDataView`。
- Data infra：`NormStats`、shard spec、确定性的 mixed episode dataloader。
- `AlgorithmRegistry` 和 `TrainingArtifact`：支持可替换算法、策略工厂和 checkpoint 上下文。
- `LossRegistry`：把 VLA、3D policy、World Model 的 loss profile 分开。
- 最小 `UnifiedEmbodiedPolicy.forward()`：用于训练基础设施 smoke test。

## 参考了哪些框架

UEFS 不是复刻任何一个项目。它提取了多个具身智能框架里值得保留的工程契约，再压缩成一个小型 NumPy demo。

| 参考框架 | UEFS 借鉴的部分 | 在本项目中的位置 | UEFS 没有照搬的部分 |
| --- | --- | --- | --- |
| [StarVLA](https://github.com/starVLA/starVLA) | Lego-like 模块边界、raw model-agnostic dataloader 输出、以 `forward()` 和 `predict_action()` 为核心的模型 API。 | `schema/`、`data/adapters/`、`data/processors/`、`training/models/policy.py`、`training/trainer.py`。 | 真实 VLM/VLA backbone、tokenizer 细节、分布式训练、benchmark 配置和完整 model zoo。 |
| [DreamZero / World Action Model](https://dreamzero0.github.io/) | future-prediction objective 可以和 action policy 共用同一套 data/training infra。 | `WorldModelDataView`、`world_model` loss profile，以及 `configs/frameworks/dreamzero/` 和 `src/robo_train/frameworks/dreamzero/` 下的 TODO 占位保留这条未来路径。 | 视频扩散、真实未来帧生成、GPU 推理、WebSocket/分布式推理和 zero-shot 能力声明。 |
| [kai0 / chi0](https://github.com/OpenDriveLab/kai0) | 训练脚本兼容、任务 prompt、图像字段映射、action 维度、joint/delta action 选择，以及原始数据和 checkpoint 路径约定。 | `configs/frameworks/kai0/tasks/`、`src/robo_train/frameworks/kai0/{profile_schema,loader,converter,launcher,plugin}.py`、`src/robo_train/cli/train.py`、`scripts/frameworks/kai0/`。 | 机器人部署、DAgger 采集、policy server，以及 shell 脚本里的 secret。 |
| [LightX2V](https://github.com/ModelTC/LightX2V) | 实用 repo 组织方式：顶层多层 `configs/`、可运行 scripts、README 优先 quickstart、任务式命令行入口。 | `configs/base/`、`configs/frameworks/kai0/tasks/`、`src/robo_train/config/layered.py`、`src/robo_train/scripts/`、`scripts/frameworks/kai0/`。 | 图像/视频生成后端、加速 kernel、模型格式转换和 GPU serving stack。 |
| [LeRobot](https://github.com/huggingface/lerobot) | dataset / policy / training / deployment 分层、可复用配置，以及机器人数据集 metadata。 | `DatasetManifest`、`ExperimentConfig`、`data/adapters/`、`configs/`。 | Hugging Face Hub 集成、真实机器人驱动、真实预训练 policy 和分布式训练。 |
| [robomimic](https://robomimic.github.io/) | config-driven imitation learning、算法抽象、dataset split 和可复现实验 artifact。 | `ExperimentConfig`、`AlgorithmRegistry`、`TrainingArtifact`、`training/trainer.py`。 | 真实 PyTorch 算法、robosuite 集成和完整 HDF5 训练 pipeline。 |
| [ManiSkill](https://maniskill.readthedocs.io/) | 未来 benchmark adapter 应该在 eval 层，不侵入 data/training 主干。 | 暂缓到 eval infra 阶段。 | GPU 仿真、物理资产、完整任务套件和 RL 环境。 |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | 未来 simulator 集成应是 adapter 层，而不是 training core 逻辑。 | 暂缓到 eval/runtime infra 阶段。 | Isaac Sim 依赖、manager-based RL stack、物理仿真和资产 pipeline。 |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | sequence-action policy 应作为训练 policy 注册，不改变 data infra。 | 未来 policy registry entry。 | 真实 diffusion/flow 训练和图像 backbone。 |
| [OpenVLA](https://github.com/openvla/openvla) / [Octo](https://github.com/octo-models/octo) | 预训练 generalist VLA policy fine-tuning 和 checkpoint 工作流。 | `PolicyConfig.pretrained_checkpoint`、`freeze_backbone`、`EmbodimentAdapter`、`TrainingArtifact`。 | 真实 VLM/VLA 权重、tokenizer、model serving 和硬件专用推理。 |
| [MimicGen](https://github.com/NVlabs/mimicgen) / [RoboCasa](https://github.com/robocasa/robocasa) | 生成式 demonstration 的来源追踪和 task-family metadata。 | `DatasetManifest.lineage`、`quality_flags`、adapter registry。 | demonstration 生成、场景资产、厨房 benchmark 和仿真 rollout。 |

更细的文件级映射见 [`docs/framework_mapping.md`](docs/framework_mapping.md)。

## 这些框架的共同点

这些参考框架虽然目标不同，但在工程结构上有一些共识：都把 **数据格式**、**模型内部实现**、**训练循环**、**运行时部署**、以及 **评估/部署证据** 分开。UEFS 保留的主要就是这条边界。

它们还共享几个实践原则：

- **先稳定协议，再选择模型。** StarVLA 的 raw dict、kai0 的部署 payload、UEFS 的 `Episode`/`EmbodiedBatch`，本质上都是先把数据契约定义清楚，再替换模型 backbone。
- **配置和 metadata 是训练系统的一部分。** LeRobot 和 robomimic 都说明 dataset、split、normalization stats、algorithm、runtime horizon 不应该散落在脚本里。
- **多层配置比脚本堆叠更稳。** LightX2V 风格的 `defaults` 可以组合 model、train 和 task 层，同时把 dataset/checkpoint 路径保留为显式配置。
- **Adapter 统一数据，DataView 再分训练输入。** VLA、3D、WM 不应该从最底层分裂成三套互不兼容 dataloader；UEFS 先统一成 `Episode`，再通过 `VLADataView`、`Policy3DDataView`、`WorldModelDataView` 分流。
- **LossProfile 是训练家族的主要开关。** VLA 的 action BC、3D policy 的 geometry-aware loss、WM 的 future-prediction loss 分别放在独立 loss 模块中。
- **EmbodimentProfile 处理机械臂差异。** Franka、ALOHA、UR、mobile manipulator 的差异应该落在 robot/action/runtime profile，而不是复制整套 trainer。
- **一个 policy surface，多个模型家族。** 内部可以是 VLA、World-Action Model 或 state-only policy，但外部通常只需要训练接口和动作预测接口。
- **Action chunk 是部署基本单元。** DreamZero、Diffusion Policy 和 kai0 都重视 chunked actions 和 smoothing；UEFS 也把 `[horizon, action_dim]` action chunk 作为一等输出。
- **Task 和 environment 应该是 adapter。** ManiSkill 和 Isaac Lab 都把 simulator 细节挡在 policy 之外；UEFS 的 `envs/` 也遵循这个边界。
- **部署日志会回流训练。** Rollout、failure、recovery、safety event、stage progress 不是只给人看的 log，而是后续 alignment 和数据闭环的来源。
- **Configs 和 scripts 很重要。** LightX2V 的组织方式提醒我们：研究框架只有在常用命令清晰、可复现时才真正好用。

## 不同模型训练的差异点

UEFS 对外只暴露一个 `UnifiedEmbodiedPolicy`，但真实系统里不同模型家族的训练目标和信号并不一样。这个 demo 通过不同 heads、metadata 和 alignment hooks，把这些差异保留下来。

| 模型家族 | 典型输入 | 主要训练目标 | 额外 loss 或训练信号 | 部署行为 | UEFS 中的对应实现 |
| --- | --- | --- | --- | --- | --- |
| **VLA / OpenPi-style action policy** | RGB 图像、语言指令、proprioception、机器人元数据。 | 预测 canonical robot actions，常见方式是 behavior cloning、flow 或 diffusion action loss。 | 可选语言保持 loss、action smoothness、安全/risk head、stage progress。 | `predict_action()` 直接返回 action chunk。 | `MockVisionEncoder`、`MockLanguageEncoder`、`MockProprioEncoder`、`ActionHead`、`forward()` 中的 action loss。 |
| **DreamZero-style World Action Model** | Observation history、任务条件，以及视频/world-model latent。 | 联合预测未来 world state/video latent 和 actions。world target 不是装饰性辅助头，而是会塑造 action 使用的共享表征。 | Future latent/video reconstruction、action prediction、chunk consistency、latency/smoothing constraints。 | 同时返回 action chunk 和 `predicted_future` summary。 | `WorldActionHead` 从同一个 shared rollout 中产生 `future_latent` 和 action-conditioned latent。 |
| **kai0 / chi0-style robust manipulation policy** | Demonstrations、未来部署 rollouts、recovery data、stage labels。 | 改善训练分布和部署分布之间的偏移。 | 暂缓到 runtime/alignment 阶段。 | 不属于当前 MVP。 | 未来 alignment package。 |
| **纯 3D 或 state-based policy** | Proprioception、低维 state、point cloud、robot morphology；语言可以是可选项。 | 从几何或状态特征预测 action，而不是依赖 VLM 级别的图像-语言特征。 | State/action normalization、morphology conditioning、smoothness 和 safety losses。 | 复用同一个 action API，通常推理更轻、更少依赖视觉模态。 | `MockProprioEncoder`、`MorphologyEncoder`、通过 `modality_mask` 支持可缺失图像流。 |
| **Diffusion Policy-style sequence policy** | 图像/state history，加语言或 task 条件。 | 学习 action sequence，训练目标通常是 denoising/flow，而不是直接 one-step MSE。 | Noise schedule、sequence consistency、action normalization。 | 暂缓到 runtime 阶段。 | `PolicyConfig` 未来 registry entry。 |
| **Sim benchmark policy** | 仿真 observation、task id、reset/step reward、success signal。 | 在标准化任务套件上训练或评估。 | Success rate、rollout length、reset、环境特定 reward。 | 暂缓到 eval 阶段。 | 未来 benchmark adapters。 |

统一 API 是有意为之：以后把 mock VLA、mock WAM 或 state-only controller 换成真实模型时，不应该改 adapters、validators、runtime clients 或 rollout logging，只需要替换 encoders、heads、trainers 的内部实现。

## 架构

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

## 快速开始

```bash
pip install -e .
pytest -q
python -m robo_train.scripts.run_local_demo
python -m robo_train.scripts.generate_demo_data --output ./demo_data --episodes 3
python -m robo_train.cli.train --framework kai0 pi05_arrange_flowers --dry-run --json
bash scripts/frameworks/kai0/train_arrange_flowers_table30v2.sh --dry-run
```

## 目录结构

```text
configs/
  base/model/              # 可复用 model 层
  frameworks/dreamzero/    # DreamZero/WAM TODO 占位
  frameworks/kai0/base/    # Kai0 launcher 和训练默认值
  frameworks/kai0/tasks/   # Kai0 兼容任务 profiles
docs/
  architecture.md
  framework_mapping.md
src/
  robo_train/
    cli/                   # 统一 framework CLI
    config/                # layered YAML loader
    frameworks/            # framework plugin 协议和适配器
      dreamzero/           # DreamZero TODO 占位
      kai0/                # Kai0 plugin schema、loader、converter、launcher
    schema/                # 跨 data/model/runtime 共享的 Universal Embodied IR
    data/                  # adapters, validators, processors, storage, dataloader
      views/               # VLA, 3D, WM 三类训练视图
    training/              # registry, artifacts, mock trainer, losses, minimal policy
    scripts/               # python -m demo 入口
scripts/frameworks/dreamzero/    # DreamZero TODO 占位
scripts/frameworks/kai0/         # Kai0 兼容 shell wrappers
tests/
```

## 核心设计

IR 是 Protocol-first 的：它明确区分 raw actions 和 canonical actions，也把场景、物体和坐标系 metadata 和多模态 streams 放在同一个 episode 里，而不是把所有模态强行压成一个万能 tensor。真实外部数据集通常会用厂商或框架自己的动作格式，而 policy 训练和部署需要稳定的目标动作空间。`ActionSemantics` 就是对这个目标空间的显式描述：representation、control mode、frame、fields、units、fps、horizon 和 action_dim。

`DatasetManifest` 和 `ExperimentConfig` 是面向真实训练的兼容层。未来接入 HDF5、LeRobot、RLDS/OXE、MimicGen 或 RoboCasa 时，数据来源、split、采样权重、质量标记和 lineage 可以集中描述；trainer 也可以通过配置选择 policy 类型和 loss 家族，而不是修改脚本。

Kai0 兼容训练 profile 采用 LightX2V 风格的多层配置：
`configs/base/model/*.yaml` 定义共享模型形态，
`configs/frameworks/kai0/base/*.yaml` 定义 launcher 和训练默认值，
`configs/frameworks/kai0/tasks/*.yaml` 只覆盖任务数据、图像映射、prompt、动作语义和 run name。当前本地 profile 指向本仓库的 `datasets/` 和 `checkpoints/kai0/...`，真实重训练由配置中的只读 Kai0/OpenPI 源码后端执行。

DreamZero/WAM 目前只预留目录：`configs/frameworks/dreamzero/`、
`src/robo_train/frameworks/dreamzero/` 和 `scripts/frameworks/dreamzero/`
里保留 TODO 占位。之后接真实 backend 时，再按同一个 plugin 协议实现，并解析到
`TrainingProfile.default_world_model()` 和 `WorldModelDataView`。

`TrainingProfile` 是更清楚的模型家族分流点。它要求 `DataProfile`、`ModelFamily`、`LossProfile` 和 `EmbodimentProfile` 对齐：VLA 使用图像-语言-state batch 和 `vla_bc`；3D policy 使用 point/state geometry 和 `policy_3d_bc`；WM 使用 context/action/future target 和 `world_model`。

`RobotCard` 和 `ActionSemantics` 是一等公民，因为同一个 policy 不应该静默假设只有一种机器人和固定 7D action。当前 demo 默认使用 `eef_delta_pose_gripper`，但 schema 支持 joint、base 和 vendor-specific representations。

`AlgorithmRegistry` 把模型家族放到名字后面。现在它构造 NumPy 版 `UnifiedEmbodiedPolicy`；以后可以注册 `diffusion_policy`、`openvla_finetune`、`octo_finetune`、`robomimic_bc` 或 state-only controller，而不改变 data/runtime 契约。

最小 policy 放在 `training/models/` 下，因为当前 MVP 只需要一个模型表面来验证 training infra。更重的 VLA、WAM、diffusion、runtime 和 benchmark 代码后续可以接回，但不会改变 data protocol。

## 当前 Demo 不做什么

- 不负责完整、长期的大模型训练 campaign。
- 不做 runtime server 或真实机器人连接。
- 不实现真实 HDF5、LeRobot 或 RLDS/OXE 读取。
- 不做真实视频生成或 diffusion。
- 不做 DAgger 优化循环、policy server 或仿真后端；Kai0/OpenPI 训练可以通过兼容后端启动。

这些接口保留下来，是为了未来替换真实后端时不需要改动 data 和 training 的外部契约。
